# diacritic_restoration/vocab.py
import json
import time
from collections import Counter
from typing import Dict, List
import pandas as pd
import torch

from diacritic_restoration.utils import remove_accents_char
from common.logger import get_logger

logger = get_logger(__name__)


class CharVocab:
    def __init__(self):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.special_tokens = [self.pad_token, self.unk_token]
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

    def build(self, texts: List[str]):
        logger.debug("Building CharVocab from scratch...")
        chars = set()
        for text in texts:
            for ch in str(text):
                chars.add(ch)

        vocab = self.special_tokens + sorted(chars)
        self.stoi = {token: idx for idx, token in enumerate(vocab)}
        self.itos = {idx: token for token, idx in self.stoi.items()}
        logger.info(f"CharVocab successfully built. Total tokens: {len(self.stoi):,}")

    @property
    def pad_id(self):
        return self.stoi[self.pad_token]

    @property
    def unk_id(self):
        return self.stoi[self.unk_token]

    def __len__(self):
        return len(self.stoi)

    def encode(self, text: str, max_len: int) -> List[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in str(text)]
        ids = ids[:max_len]
        ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int], original_length: int = None) -> str:
        if original_length is not None:
            ids = ids[:original_length]

        chars = []
        for idx in ids:
            idx = int(idx)
            if idx == self.pad_id:
                continue
            token = self.itos.get(idx, self.unk_token)
            if token not in self.special_tokens:
                chars.append(token)
        return "".join(chars)


class WordVocab:
    def __init__(self):
        self.pad_token = "<pad_word>"
        self.unk_token = "<unk_word>"
        self.space_token = "<space>"
        self.special_tokens = [self.pad_token, self.unk_token, self.space_token]
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

    @property
    def pad_id(self):
        return self.stoi[self.pad_token]

    @property
    def unk_id(self):
        return self.stoi[self.unk_token]

    @property
    def space_id(self):
        return self.stoi[self.space_token]

    def __len__(self):
        return len(self.stoi)

    def build(self, texts: List[str], max_size: int, min_freq: int):
        logger.debug(f"Building WordVocab (max_size={max_size}, min_freq={min_freq})...")
        counter = Counter()
        for text in texts:
            for word in str(text).split():
                counter[word] += 1
                
        words = [w for w, c in counter.most_common() if c >= min_freq]
        logger.debug(f"Words meeting min_freq={min_freq} criterion: {len(words):,}")
        
        words = words[:max_size]
        vocab = self.special_tokens + words
        self.stoi = {token: idx for idx, token in enumerate(vocab)}
        self.itos = {idx: token for token, idx in self.stoi.items()}
        logger.info(f"WordVocab successfully built. Total tokens: {len(self.stoi):,}")


def encode_word_ids_per_char(text: str, word_vocab: WordVocab, max_len: int) -> List[int]:
    ids = []
    text = str(text)
    i = 0
    n = len(text)

    while i < n:
        if text[i].isspace():
            ids.append(word_vocab.space_id)
            i += 1
            continue

        j = i
        while j < n and not text[j].isspace():
            j += 1

        word = text[i:j]
        word_id = word_vocab.stoi.get(word, word_vocab.unk_id)
        ids.extend([word_id] * len(word))
        i = j

    ids = ids[:max_len]
    ids += [word_vocab.pad_id] * (max_len - len(ids))
    return ids


def build_word_vocab(train_df: pd.DataFrame, cfg) -> WordVocab:
    word_vocab = WordVocab()
    word_vocab.build(
        texts=train_df[cfg.data.input_col].tolist(),
        max_size=cfg.vocab.max_word_vocab_size,
        min_freq=cfg.vocab.min_word_freq,
    )
    logger.info(f"Final initialized WordVocab size: {len(word_vocab):,}")
    return word_vocab


def build_vocab_from_words_file_and_dataframe(df: pd.DataFrame, cfg) -> CharVocab:
    vocab = CharVocab()
    words_texts = []
    
    logger.info(f"Attempting to read custom words file from: {cfg.data.words_path}")
    try:
        with open(cfg.data.words_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    word = str(item.get("text", "")).strip()
                    if word:
                        words_texts.append(word)
                except json.JSONDecodeError:
                    logger.debug(f"JSONDecodeError skipped at {cfg.data.words_path} line {line_idx}")
                    continue
        logger.info(f"Successfully loaded words from external file: {len(words_texts):,} unique entries.")
    except FileNotFoundError:
        logger.warning(f"Words file NOT found at '{cfg.data.words_path}'. Building vocabulary solely using Dataframe.")

    df_texts = (
        df[cfg.data.input_col].astype(str).tolist()
        + df[cfg.data.target_col].astype(str).tolist()
    )

    logger.info("Extracting distinct characters from aggregated text components...")
    vocab.build(words_texts + df_texts)
    logger.info(f"Final initialized CharVocab size: {len(vocab):,}")
    return vocab


def build_allowed_token_mask(vocab: CharVocab, cfg) -> torch.Tensor:
    """
    allowed_mask[input_id, output_id] = True if output token is a valid
    accented form of input token.

    Examples:
        o -> o, ò, ó, ỏ, õ, ọ, ô, ồ, ố, ..., ơ, ờ, ớ, ...
        d -> d, đ
        t -> t only
    """
    logger.info("Generating allowed diacritic token token mask tensor...")
    start_time = time.time()
    
    vocab_size = len(vocab)
    allowed = torch.zeros(vocab_size, vocab_size, dtype=torch.bool)

    allowed[vocab.pad_id, vocab.pad_id] = True
    allowed[vocab.unk_id, vocab.unk_id] = True

    for input_token, input_id in vocab.stoi.items():
        if input_token in vocab.special_tokens or len(input_token) != 1:
            continue

        input_base = remove_accents_char(input_token)

        for output_token, output_id in vocab.stoi.items():
            if output_token in vocab.special_tokens or len(output_token) != 1:
                continue

            output_base = remove_accents_char(output_token)

            if output_base == input_base:
                allowed[input_id, output_id] = True

    # Đo thời gian sinh ma trận ràng buộc
    elapsed = time.time() - start_time
    logger.info(f"Allowed token mask matrix built successfully in {elapsed:.4f}s.")
    logger.debug(f"Mask matrix properties - Shape: {allowed.shape} | Device: {cfg.training.device}")

    return allowed.to(cfg.training.device)