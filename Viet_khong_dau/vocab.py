import json
from typing import List, Dict
from collections import Counter
import pandas as pd
from config import Config
import torch
from utils import remove_accents_char, remove_accents_text, normalize_text


class CharVocab:
    def __init__(self):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.special_tokens = [self.pad_token, self.unk_token]
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

    def build(self, texts: List[str]):
        chars = set()
        for text in texts:
            for ch in str(text):
                chars.add(ch)

        vocab = self.special_tokens + sorted(chars)
        self.stoi = {token: idx for idx, token in enumerate(vocab)}
        self.itos = {idx: token for token, idx in self.stoi.items()}

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
        counter = Counter()
        for text in texts:
            for word in str(text).split():
                counter[word] += 1
        words = [w for w, c in counter.most_common() if c >= min_freq]
        words = words[:max_size]
        vocab = self.special_tokens + words
        self.stoi = {token: idx for idx, token in enumerate(vocab)}
        self.itos = {idx: token for token, idx in self.stoi.items()}


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


def build_word_vocab(train_df: pd.DataFrame, cfg: Config) -> WordVocab:
    word_vocab = WordVocab()
    word_vocab.build(
        texts=train_df[cfg.input_col].tolist(),
        max_size=cfg.max_word_vocab_size,
        min_freq=cfg.min_word_freq,
    )
    print(f"Word vocab size: {len(word_vocab):,}")
    return word_vocab


def load_words_from_txt(words_path: str) -> List[str]:
    """Read JSONL words.txt and use the `text` field."""
    words = []
    with open(words_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                word = str(item.get("text", "")).strip()
                if word:
                    words.append(word)
            except json.JSONDecodeError:
                continue
    return words




def make_pair_from_target(target: str, cfg: Config):
    """Create a valid no-accent -> accented pair from accented target text."""
    tgt = normalize_text(target, lowercase=cfg.lowercase)
    if not tgt:
        return None
    if len(tgt) > cfg.max_len:
        tgt = tgt[: cfg.max_len]
    src = remove_accents_text(tgt)
    if len(src) != len(tgt):
        return None
    return {cfg.input_col: src, cfg.target_col: tgt}


def clean_original_pairs(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Clean original CSV pairs and keep only valid sequence-labeling samples."""
    pairs = []
    skipped = 0
    for _, row in df.iterrows():
        src = normalize_text(row[cfg.input_col], lowercase=cfg.lowercase)
        tgt = normalize_text(row[cfg.target_col], lowercase=cfg.lowercase)
        if len(src) > cfg.max_len:
            src = src[: cfg.max_len]
            tgt = tgt[: cfg.max_len]
        if len(src) == len(tgt) and remove_accents_text(tgt) == src:
            pairs.append({cfg.input_col: src, cfg.target_col: tgt})
        else:
            # Fallback: if target is valid Vietnamese text, derive input from target.
            pair = make_pair_from_target(tgt, cfg)
            if pair is not None:
                pairs.append(pair)
            else:
                skipped += 1

    out = pd.DataFrame(pairs).drop_duplicates().reset_index(drop=True)
    print(f"Original clean pairs: {len(out):,} | skipped: {skipped:,}")
    return out


def build_word_pairs_from_words_file(cfg: Config) -> pd.DataFrame:
    """
    Build extra pairs from words.txt:
        target = accented word/phrase
        input  = remove_accents(target)

    These are not strong context examples, but they teach valid accent forms.
    """
    if not cfg.use_word_pairs:
        return pd.DataFrame(columns=[cfg.input_col, cfg.target_col])

    words = load_words_from_txt(cfg.words_path)
    pairs = []

    for word in words:
        pair = make_pair_from_target(word, cfg)
        if pair is None:
            continue
        # Avoid very short one-character pairs dominating the data.
        if len(pair[cfg.target_col]) < 2:
            continue
        pairs.append(pair)
        if len(pairs) >= cfg.max_word_pairs:
            break

    out = pd.DataFrame(pairs).drop_duplicates().reset_index(drop=True)
    print(f"Synthetic word/phrase pairs: {len(out):,}")
    return out


def build_sentence_chunks_from_targets(train_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Create more sentence-level context by taking word windows from real target sentences.
    This is much better than random word sequences because chunks preserve real local context.
    """
    if not cfg.use_sentence_chunks:
        return pd.DataFrame(columns=[cfg.input_col, cfg.target_col])

    pairs = []
    for tgt in train_df[cfg.target_col].astype(str).tolist():
        tgt = normalize_text(tgt, lowercase=cfg.lowercase)
        words = tgt.split()
        if len(words) < cfg.chunk_min_words:
            continue

        max_w = min(cfg.chunk_max_words, len(words))
        for window in range(cfg.chunk_min_words, max_w + 1):
            for start in range(0, len(words) - window + 1, cfg.chunk_stride):
                chunk = " ".join(words[start : start + window])
                pair = make_pair_from_target(chunk, cfg)
                if pair is not None:
                    pairs.append(pair)

    out = pd.DataFrame(pairs).drop_duplicates().reset_index(drop=True)
    print(f"Synthetic sentence chunk pairs: {len(out):,}")
    return out


def split_dataframe(df: pd.DataFrame, cfg: Config):
    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    n_total = len(df)
    n_train = int(n_total * cfg.train_ratio)
    n_valid = int(n_total * cfg.valid_ratio)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    valid_df = df.iloc[n_train:n_train + n_valid].reset_index(drop=True)
    test_df = df.iloc[n_train + n_valid:].reset_index(drop=True)
    return train_df, valid_df, test_df

def build_vocab_from_words_file_and_dataframe(df: pd.DataFrame, cfg: Config) -> CharVocab:
    vocab = CharVocab()
    words_texts = load_words_from_txt(cfg.words_path)

    csv_texts = (
        df[cfg.input_col].astype(str).tolist()
        + df[cfg.target_col].astype(str).tolist()
    )

    vocab.build(words_texts + csv_texts)

    print(f"Loaded words from words.txt: {len(words_texts):,}")
    print(f"Vocab size: {len(vocab)}")
    return vocab


def build_allowed_token_mask(vocab: CharVocab, cfg: Config):
    """
    allowed_mask[input_id, output_id] = True if output token is a valid
    accented form of input token.

    Examples:
        o -> o, ò, ó, ỏ, õ, ọ, ô, ồ, ố, ..., ơ, ờ, ớ, ...
        d -> d, đ
        t -> t only
    """
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

    return allowed.to(cfg.device)