import json
import time
from types import SimpleNamespace
from typing import List
import pandas as pd
import torch

from diacritic_restoration.model import ContextAwareAccentTagger
from diacritic_restoration.utils import apply_constraint_to_logits, normalize_text, remove_accents_text
from diacritic_restoration.vocab import CharVocab, WordVocab, build_allowed_token_mask, encode_word_ids_per_char
from common.logger import get_logger

logger = get_logger(__name__)


class DiacriticDataProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        logger.info("DiacriticDataProcessor initialized.")

    def load_words_from_txt(self) -> List[str]:
        words = []
        logger.info(f"Loading words vocab from: {self.cfg.data.words_path}")
        try:
            with open(self.cfg.data.words_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        word = str(item.get("text", "")).strip()
                        if word:
                            words.append(word)
                    except json.JSONDecodeError:
                        logger.debug(f"JSONDecodeError at line {line_idx} in {self.cfg.data.words_path}")
                        continue
            logger.info(f"Loaded {len(words):,} words.")
        except FileNotFoundError as e:
            logger.error(f"Dictionary file not found: {self.cfg.data.words_path}", exc_info=True)
            raise e
        return words

    def make_pair_from_target(self, target: str):
        tgt = normalize_text(target, lowercase=self.cfg.augmentation.lowercase)
        if not tgt:
            return None
        if len(tgt) > self.cfg.model.max_len:
            tgt = tgt[:self.cfg.model.max_len]
        src = remove_accents_text(tgt)
        if len(src) != len(tgt):
            logger.debug(f"Length mismatch: tgt='{tgt}' ({len(tgt)}) | src='{src}' ({len(src)})")
            return None
        return {self.cfg.data.input_col: src, self.cfg.data.target_col: tgt}

    def clean_original_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Cleaning text pairs. Raw rows: {len(df):,}")
        pairs = []
        skipped_count = 0
        
        for _, row in df.iterrows():
            src = normalize_text(row[self.cfg.data.input_col], lowercase=self.cfg.augmentation.lowercase)
            tgt = normalize_text(row[self.cfg.data.target_col], lowercase=self.cfg.augmentation.lowercase)
            if len(src) > self.cfg.model.max_len:
                src = src[:self.cfg.model.max_len]
                tgt = tgt[:self.cfg.model.max_len]
            if len(src) == len(tgt) and remove_accents_text(tgt) == src:
                pairs.append({self.cfg.data.input_col: src, self.cfg.data.target_col: tgt})
            else:
                pair = self.make_pair_from_target(tgt)
                if pair is not None:
                    pairs.append(pair)
                else:
                    skipped_count += 1
                    
        cleaned_df = pd.DataFrame(pairs).drop_duplicates().reset_index(drop=True)
        logger.info(f"Cleaning completed. Retained: {len(cleaned_df):,} | Dropped: {skipped_count:,}")
        return cleaned_df

    def build_word_pairs_from_words_file(self) -> pd.DataFrame:
        if not self.cfg.augmentation.use_word_pairs:
            logger.debug("Augmentation 'use_word_pairs' is disabled.")
            return pd.DataFrame(columns=[self.cfg.data.input_col, self.cfg.data.target_col])
            
        logger.info("Building word pairs augmentation...")
        words = self.load_words_from_txt()
        pairs = []
        for word in words:
            pair = self.make_pair_from_target(word)
            if pair is None or len(pair[self.cfg.data.target_col]) < 2:
                continue
            pairs.append(pair)
            if len(pairs) >= self.cfg.augmentation.max_word_pairs:
                logger.info(f"Reached max_word_pairs limit: {self.cfg.augmentation.max_word_pairs:,}")
                break
                
        word_pairs_df = pd.DataFrame(pairs).drop_duplicates().reset_index(drop=True)
        logger.info(f"Generated word pairs: {len(word_pairs_df):,}")
        return word_pairs_df

    def build_sentence_chunks_from_targets(self, train_df: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.augmentation.use_sentence_chunks:
            logger.debug("Augmentation 'use_sentence_chunks' is disabled.")
            return pd.DataFrame(columns=[self.cfg.data.input_col, self.cfg.data.target_col])
            
        logger.info("Building sliding window sentence chunks...")
        pairs = []
        for tgt in train_df[self.cfg.data.target_col].astype(str).tolist():
            tgt = normalize_text(tgt, lowercase=self.cfg.augmentation.lowercase)
            words = tgt.split()
            if len(words) < self.cfg.augmentation.chunk_min_words:
                continue
            max_w = min(self.cfg.augmentation.chunk_max_words, len(words))
            for window in range(self.cfg.augmentation.chunk_min_words, max_w + 1):
                for start in range(0, len(words) - window + 1, self.cfg.augmentation.chunk_stride):
                    chunk = " ".join(words[start : start + window])
                    pair = self.make_pair_from_target(chunk)
                    if pair is not None:
                        pairs.append(pair)
                        
        chunks_df = pd.DataFrame(pairs).drop_duplicates().reset_index(drop=True)
        logger.info(f"Generated sentence chunks: {len(chunks_df):,}")
        return chunks_df

    def split_dataframe(self, df: pd.DataFrame):
        logger.info("Splitting dataset...")
        df = df.sample(frac=1.0, random_state=self.cfg.training.seed).reset_index(drop=True)
        n_total = len(df)
        n_train = int(n_total * self.cfg.data.train_ratio)
        n_valid = int(n_total * self.cfg.data.valid_ratio)
        
        logger.info(f"Split ratios -> Train: {self.cfg.data.train_ratio} | Valid: {self.cfg.data.valid_ratio}")
        return (
            df.iloc[:n_train].reset_index(drop=True),
            df.iloc[n_train:n_train + n_valid].reset_index(drop=True),
            df.iloc[n_train + n_valid:].reset_index(drop=True)
        )


class DiacriticRestorer:
    def __init__(self, checkpoint_path: str, cfg):
        self.cfg = cfg
        logger.info(f"Initializing DiacriticRestorer from checkpoint: {checkpoint_path}")
        self.model, self.char_vocab, self.word_vocab, self.allowed_mask = self._load_checkpoint(checkpoint_path)
        logger.info("DiacriticRestorer pipeline successfully built.")

    @torch.no_grad()
    def _predict_batch(self, src_char: torch.Tensor, src_word: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        logits = self.model(src_char, src_word)
        logits = apply_constraint_to_logits(logits, src_char, self.allowed_mask)
        return logits.argmax(dim=-1)

    def process(self, text: str, stopwords) -> str:
        self.model.eval()
        start_time = time.time()
        
        raw_length = len(text)
        text = normalize_text(str(text), lowercase=self.cfg.augmentation.lowercase)
        logger.debug(f"Inference input string length: {raw_length} -> Normalized: {len(text)}")

        src_char_ids = self.char_vocab.encode(text, self.cfg.model.max_len)
        src_word_ids = encode_word_ids_per_char(
            text=text,
            word_vocab=self.word_vocab,
            max_len=self.cfg.model.max_len,
        )

        src_char = torch.tensor([src_char_ids], dtype=torch.long, device=self.cfg.training.device)
        src_word = torch.tensor([src_word_ids], dtype=torch.long, device=self.cfg.training.device)

        with torch.no_grad():
            pred_ids = self._predict_batch(src_char=src_char, src_word=src_word)
    
        pred_ids = pred_ids[0].detach().cpu().tolist()
        output_chars = []
        limit = min(len(text), self.cfg.model.max_len)

        for i in range(limit):
            input_ch = text[i]
            if input_ch not in self.char_vocab.stoi:
                output_chars.append(input_ch)
                continue

            pred_id = int(pred_ids[i])
            pred_ch = self.char_vocab.itos.get(pred_id)

            if pred_ch is None or pred_ch in self.char_vocab.special_tokens:
                output_chars.append(input_ch)
                continue

            output_chars.append(pred_ch)

        if len(text) > self.cfg.model.max_len:
            logger.warning(
                f"Input length exceeds max_len ({self.cfg.model.max_len}). "
                f"Remaining {len(text) - self.cfg.model.max_len} chars appended without restoration."
            )
            output_chars.append(text[self.cfg.model.max_len:])

        restored_text = "".join(output_chars)
        elapsed_inference = time.time() - start_time
        logger.debug(f"Inference completed in {elapsed_inference:.4f}s.")
        
        return restored_text

    def _load_checkpoint(self, checkpoint_path: str):
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        torch.serialization.add_safe_globals([SimpleNamespace])
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.cfg.training.device)
        except Exception as e:
            logger.error(f"Failed to load checkpoint file at {checkpoint_path}.", exc_info=True)
            raise e

        char_vocab = CharVocab()
        char_vocab.stoi = checkpoint["char_vocab_stoi"]
        char_vocab.itos = {int(k): v for k, v in checkpoint["char_vocab_itos"].items()}
        logger.info(f"Loaded CharVocab size: {len(char_vocab):,}")

        word_vocab = WordVocab()
        word_vocab.stoi = checkpoint["word_vocab_stoi"]
        word_vocab.itos = {int(k): v for k, v in checkpoint["word_vocab_itos"].items()}
        logger.info(f"Loaded WordVocab size: {len(word_vocab):,}")

        model = ContextAwareAccentTagger(
            char_vocab_size=len(char_vocab),
            word_vocab_size=len(word_vocab),
            char_pad_id=char_vocab.pad_id,
            word_pad_id=word_vocab.pad_id,
            cfg=self.cfg,
        ).to(self.cfg.training.device)

        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[7:]] = value
            elif key.startswith("_orig_mod."):
                new_state_dict[key[10:]] = value
            elif key.startswith("model."):
                new_state_dict[key[6:]] = value
            else:
                new_state_dict[key] = value

        try:
            model.load_state_dict(new_state_dict)
            logger.info("Model weights loaded with strict matching.")
        except Exception:
            logger.warning("Strict match failed. Retrying with strict=False.")
            model.load_state_dict(new_state_dict, strict=False)
            logger.info("Model weights loaded with strict=False.")

        model.eval()
        allowed_mask = build_allowed_token_mask(char_vocab, self.cfg)
        return model, char_vocab, word_vocab, allowed_mask