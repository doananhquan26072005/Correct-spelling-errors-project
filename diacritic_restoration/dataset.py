import time
from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from diacritic_restoration.processor import DiacriticDataProcessor
from diacritic_restoration.utils import remove_accents_text, normalize_text
from diacritic_restoration.vocab import CharVocab, WordVocab, encode_word_ids_per_char, build_vocab_from_words_file_and_dataframe, build_word_vocab

from common.logger import get_logger

logger = get_logger(__name__)

class AccentContextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, char_vocab: CharVocab, word_vocab: WordVocab, cfg):
        self.inputs = []
        self.targets = []
        self.char_vocab = char_vocab
        self.word_vocab = word_vocab
        self.max_len = cfg.model.max_len

        skipped_len = 0
        skipped_base = 0

        logger.debug(f"Initializing AccentContextDataset with {len(df):,} raw rows.")

        for _, row in df.iterrows():
            src = normalize_text(row[cfg.data.input_col], lowercase=cfg.augmentation.lowercase)
            tgt = normalize_text(row[cfg.data.target_col], lowercase=cfg.augmentation.lowercase)

            if len(src) > self.max_len:
                src = src[: self.max_len]
                tgt = tgt[: self.max_len]

            if len(src) != len(tgt):
                skipped_len += 1
                continue

            if remove_accents_text(tgt) != src:
                skipped_base += 1
                continue

            self.inputs.append(src)
            self.targets.append(tgt)

        logger.info(f"Dataset loaded: {len(self.inputs):,} samples.")
        if skipped_len or skipped_base:
            logger.warning(
                f"Filtered samples | Skipped len mismatch: {skipped_len:,} | Skipped base mismatch: {skipped_base:,}"
            )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> dict:
        src_text = self.inputs[idx]
        tgt_text = self.targets[idx]

        src_char_ids = self.char_vocab.encode(src_text, self.max_len)
        tgt_ids = self.char_vocab.encode(tgt_text, self.max_len)
        src_word_ids = encode_word_ids_per_char(src_text, self.word_vocab, self.max_len)
        length = min(len(src_text), self.max_len)

        item = {
            "src_char": torch.tensor(src_char_ids, dtype=torch.long),
            "src_word": torch.tensor(src_word_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

        logger.debug(
            f"Get item [{idx}] - len: {length} | src_char: {item['src_char'].shape} | src_word: {item['src_word'].shape}"
        )
        return item


class DiacriticDataLoaderFactory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.processor = DiacriticDataProcessor(cfg)
        logger.info("DiacriticDataLoaderFactory initialized.")

    def build_loaders_and_vocabs(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader, any, any]:
        logger.info("Building loaders and vocabs from DataFrames.")
        start_time = time.time()

        def prepare_df(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
            df_filtered = df[[self.cfg.data.input_col, self.cfg.data.target_col]].dropna().reset_index(drop=True)
            cleaned_df = self.processor.clean_original_pairs(df_filtered)
            logger.debug(f"[{split_name}] Size after clean: {len(cleaned_df):,}")
            return cleaned_df

        train_orig_df = prepare_df(df_train, "Train")
        valid_df = prepare_df(df_valid, "Valid")
        test_df = prepare_df(df_test, "Test")

        if len(train_orig_df) == 0:
            logger.error("Train DataFrame is empty after cleaning!")
            raise ValueError("No valid samples in train_df after cleaning.")

        parts = [train_orig_df]
        if self.cfg.augmentation.use_sentence_chunks:
            chunks_df = self.processor.build_sentence_chunks_from_targets(train_orig_df)
            parts.append(chunks_df)
            logger.debug(f"Augmentation: Added {len(chunks_df):,} sentence chunks.")
            
        if self.cfg.augmentation.use_word_pairs:
            word_pairs_df = self.processor.build_word_pairs_from_words_file()
            parts.append(word_pairs_df)
            logger.debug(f"Augmentation: Added {len(word_pairs_df):,} word pairs.")

        train_df = pd.concat(parts, ignore_index=True)
        train_df = train_df.drop_duplicates().sample(frac=1.0, random_state=self.cfg.training.seed).reset_index(drop=True)
        logger.debug(f"Train size after dedup & shuffle: {len(train_df):,}")

        if len(train_df) > self.cfg.augmentation.max_train_samples:
            logger.info(f"Downsampling train dataset to max_train_samples: {self.cfg.augmentation.max_train_samples:,}")
            train_df = train_df.sample(n=self.cfg.augmentation.max_train_samples, random_state=self.cfg.training.seed).reset_index(drop=True)

        logger.info("Dataset split sizes:")
        logger.info(f"  > Train (orig) : {len(train_orig_df):,}")
        logger.info(f"  > Train (aug)  : {len(train_df):,}")
        logger.info(f"  > Valid (orig) : {len(valid_df):,}")
        logger.info(f"  > Test (orig)  : {len(test_df):,}")

        logger.info("Building CharVocab and WordVocab...")
        vocab_source_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        char_vocab = build_vocab_from_words_file_and_dataframe(vocab_source_df, self.cfg)
        word_vocab = build_word_vocab(train_df, self.cfg)
        logger.info(f"Vocab built | CharVocab: {len(char_vocab):,} | WordVocab: {len(word_vocab):,}")

        logger.info("Creating AccentContextDataset instances...")
        train_ds = AccentContextDataset(train_df, char_vocab, word_vocab, self.cfg)
        valid_ds = AccentContextDataset(valid_df, char_vocab, word_vocab, self.cfg)
        test_ds = AccentContextDataset(test_df, char_vocab, word_vocab, self.cfg)

        logger.info("Constructing PyTorch DataLoaders...")
        train_loader = DataLoader(train_ds, batch_size=self.cfg.training.batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_ds, batch_size=self.cfg.training.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=self.cfg.training.batch_size, shuffle=False, num_workers=0)

        elapsed_time = time.time() - start_time
        logger.info(f"Loaders and vocabs built successfully in {elapsed_time:.2f}s.")

        return train_loader, valid_loader, test_loader, char_vocab, word_vocab