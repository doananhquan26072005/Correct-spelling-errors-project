# diacritic_restoration/dataset.py
import json
from typing import List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Đồng bộ hóa import nội bộ cùng cấp (Flat Architecture)
from diacritic_restoration.processor import DiacriticDataProcessor
from diacritic_restoration.utils import remove_accents_text, normalize_text, remove_accents_char
from diacritic_restoration.vocab import CharVocab, WordVocab, encode_word_ids_per_char, build_vocab_from_words_file_and_dataframe, build_word_vocab


class AccentContextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, char_vocab: CharVocab, word_vocab: WordVocab, cfg):
        self.inputs = []
        self.targets = []
        self.char_vocab = char_vocab
        self.word_vocab = word_vocab
        self.max_len = cfg.model.max_len

        skipped_len = 0
        skipped_base = 0

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

        print(f"Dataset samples: {len(self.inputs):,}")
        if skipped_len or skipped_base:
            print(f"  skipped length mismatch: {skipped_len:,}")
            print(f"  skipped base mismatch  : {skipped_base:,}")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> dict:
        src_text = self.inputs[idx]
        tgt_text = self.targets[idx]

        src_char_ids = self.char_vocab.encode(src_text, self.max_len)
        tgt_ids = self.char_vocab.encode(tgt_text, self.max_len)
        src_word_ids = encode_word_ids_per_char(src_text, self.word_vocab, self.max_len)
        length = min(len(src_text), self.max_len)

        return {
            "src_char": torch.tensor(src_char_ids, dtype=torch.long),
            "src_word": torch.tensor(src_word_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


class DiacriticDataLoaderFactory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.processor = DiacriticDataProcessor(cfg)

    def build_loaders_and_vocabs(self) -> Tuple[DataLoader, DataLoader, DataLoader, CharVocab, WordVocab]:
        raw_df = pd.read_csv(self.cfg.data.csv_path)
        raw_df = raw_df[[self.cfg.data.input_col, self.cfg.data.target_col]].dropna().reset_index(drop=True)

        original_df = self.processor.clean_original_pairs(raw_df)
        if len(original_df) == 0:
            raise ValueError("No valid original samples after cleaning.")

        train_orig_df, valid_df, test_df = self.processor.split_dataframe(original_df)

        parts = [train_orig_df]
        if self.cfg.augmentation.use_sentence_chunks:
            parts.append(self.processor.build_sentence_chunks_from_targets(train_orig_df))
        if self.cfg.augmentation.use_word_pairs:
            parts.append(self.processor.build_word_pairs_from_words_file())

        train_df = pd.concat(parts, ignore_index=True)
        train_df = train_df.drop_duplicates().sample(frac=1.0, random_state=self.cfg.training.seed).reset_index(drop=True)

        if len(train_df) > self.cfg.augmentation.max_train_samples:
            train_df = train_df.sample(n=self.cfg.augmentation.max_train_samples, random_state=self.cfg.training.seed).reset_index(drop=True)

        print("Final split sizes:")
        print(f"  Train original : {len(train_orig_df):,}")
        print(f"  Train augmented: {len(train_df):,}")
        print(f"  Valid original : {len(valid_df):,}")
        print(f"  Test original  : {len(test_df):,}")

        vocab_source_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        char_vocab = build_vocab_from_words_file_and_dataframe(vocab_source_df, self.cfg)
        word_vocab = build_word_vocab(train_df, self.cfg)

        train_ds = AccentContextDataset(train_df, char_vocab, word_vocab, self.cfg)
        valid_ds = AccentContextDataset(valid_df, char_vocab, word_vocab, self.cfg)
        test_ds = AccentContextDataset(test_df, char_vocab, word_vocab, self.cfg)

        train_loader = DataLoader(train_ds, batch_size=self.cfg.training.batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_ds, batch_size=self.cfg.training.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=self.cfg.training.batch_size, shuffle=False, num_workers=0)

        return train_loader, valid_loader, test_loader, char_vocab, word_vocab