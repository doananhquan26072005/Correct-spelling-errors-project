import torch.nn as nn
import torch
import math
from config import Config
from vocab import CharVocab, WordVocab


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class ContextAwareAccentTagger(nn.Module):
    """
    Char-level sequence labeling with word-level context.

    Each character receives:
        char embedding + word embedding of the word it belongs to.
    Then Transformer self-attention learns context over the whole sentence.
    """
    def __init__(self, char_vocab_size: int, word_vocab_size: int, char_pad_id: int, word_pad_id: int, cfg: Config):
        super().__init__()

        assert cfg.char_emb_dim + cfg.word_emb_dim == cfg.d_model, (
            "char_emb_dim + word_emb_dim must equal d_model"
        )

        self.char_pad_id = char_pad_id
        self.word_pad_id = word_pad_id
        self.d_model = cfg.d_model

        self.char_embedding = nn.Embedding(char_vocab_size, cfg.char_emb_dim, padding_idx=char_pad_id)
        self.word_embedding = nn.Embedding(word_vocab_size, cfg.word_emb_dim, padding_idx=word_pad_id)
        self.input_dropout = nn.Dropout(cfg.dropout)
        self.positional_encoding = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_encoder_layers)
        self.output_layer = nn.Linear(cfg.d_model, char_vocab_size)

    def forward(self, src_char, src_word):
        padding_mask = src_char.eq(self.char_pad_id)

        char_emb = self.char_embedding(src_char)
        word_emb = self.word_embedding(src_word)
        x = torch.cat([char_emb, word_emb], dim=-1)
        x = x * math.sqrt(self.d_model)
        x = self.input_dropout(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.output_layer(x)


def build_model(char_vocab: CharVocab, word_vocab: WordVocab, cfg: Config):
    model = ContextAwareAccentTagger(
        char_vocab_size=len(char_vocab),
        word_vocab_size=len(word_vocab),
        char_pad_id=char_vocab.pad_id,
        word_pad_id=word_vocab.pad_id,
        cfg=cfg,
    ).to(cfg.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model