import torch
from dataclasses import dataclass

@dataclass
class Config:
    csv_path: str = "viet_khong_dau.csv"
    words_path: str = "words.txt"

    input_col: str = "input"
    target_col: str = "target"

    max_len: int = 256
    batch_size: int = 64

    # Smaller model is usually better for limited data from scratch.
    d_model: int = 192
    char_emb_dim: int = 128
    word_emb_dim: int = 64
    nhead: int = 4
    num_encoder_layers: int = 4
    dim_feedforward: int = 768
    dropout: float = 0.2

    epochs: int = 80
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1

    seed: int = 42
    checkpoint_path: str = "accent_tagger_context_v3.pt"

    # Data generation / augmentation
    lowercase: bool = True
    use_word_pairs: bool = False
    max_word_pairs: int = 30_000
    use_sentence_chunks: bool = True
    chunk_min_words: int = 5
    chunk_max_words: int = 24
    chunk_stride: int = 2
    max_train_samples: int = 300_000

    # Accent positions are rare, so give them larger loss weight.
    copy_loss_weight: float = 1.0
    accent_loss_weight: float = 10.0

    # Recommended: use constraint for eval/inference, not for training loss.
    use_constraint_in_training: bool = True

    max_word_vocab_size: int = 50_000
    min_word_freq: int = 1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
