from torch.utils.data import Dataset, DataLoader
from config import Config
from vocab import CharVocab, WordVocab, encode_word_ids_per_char, build_vocab_from_words_file_and_dataframe, build_word_vocab, clean_original_pairs, split_dataframe, build_sentence_chunks_from_targets, build_word_pairs_from_words_file
import torch
import pandas as pd
from utils import remove_accents_text, normalize_text

class AccentContextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, char_vocab: CharVocab, word_vocab: WordVocab, cfg: Config):
        self.inputs = []
        self.targets = []
        skipped_len = 0
        skipped_base = 0

        for _, row in df.iterrows():
            src = normalize_text(row[cfg.input_col], lowercase=cfg.lowercase)
            tgt = normalize_text(row[cfg.target_col], lowercase=cfg.lowercase)

            if len(src) > cfg.max_len:
                src = src[: cfg.max_len]
                tgt = tgt[: cfg.max_len]

            if len(src) != len(tgt):
                skipped_len += 1
                continue

            if remove_accents_text(tgt) != src:
                skipped_base += 1
                continue

            self.inputs.append(src)
            self.targets.append(tgt)

        self.char_vocab = char_vocab
        self.word_vocab = word_vocab
        self.max_len = cfg.max_len

        print(f"Dataset samples: {len(self.inputs):,}")
        if skipped_len or skipped_base:
            print(f"  skipped length mismatch: {skipped_len:,}")
            print(f"  skipped base mismatch  : {skipped_base:,}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
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


def build_dataloaders(cfg: Config):
    raw_df = pd.read_csv(cfg.csv_path)
    raw_df = raw_df[[cfg.input_col, cfg.target_col]].dropna().reset_index(drop=True)

    original_df = clean_original_pairs(raw_df, cfg)
    if len(original_df) == 0:
        raise ValueError("No valid original samples after cleaning.")

    train_orig_df, valid_df, test_df = split_dataframe(original_df, cfg)

    parts = [train_orig_df]
    if cfg.use_sentence_chunks:
        parts.append(build_sentence_chunks_from_targets(train_orig_df, cfg))
    if cfg.use_word_pairs:
        parts.append(build_word_pairs_from_words_file(cfg))

    train_df = pd.concat(parts, ignore_index=True)
    train_df = train_df.drop_duplicates().sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    if len(train_df) > cfg.max_train_samples:
        train_df = train_df.sample(n=cfg.max_train_samples, random_state=cfg.seed).reset_index(drop=True)

    print("Final split sizes:")
    print(f"  Train original : {len(train_orig_df):,}")
    print(f"  Train augmented: {len(train_df):,}")
    print(f"  Valid original : {len(valid_df):,}")
    print(f"  Test original  : {len(test_df):,}")

    vocab_source_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    char_vocab = build_vocab_from_words_file_and_dataframe(vocab_source_df, cfg)
    word_vocab = build_word_vocab(train_df, cfg)

    train_ds = AccentContextDataset(train_df, char_vocab, word_vocab, cfg)
    valid_ds = AccentContextDataset(valid_df, char_vocab, word_vocab, cfg)
    test_ds = AccentContextDataset(test_df, char_vocab, word_vocab, cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader, char_vocab, word_vocab