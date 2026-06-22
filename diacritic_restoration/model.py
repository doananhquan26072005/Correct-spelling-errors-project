import math
import torch
import torch.nn as nn

from diacritic_restoration.vocab import CharVocab, WordVocab
from common.logger import get_logger

logger = get_logger(__name__)


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
        logger.debug(f"PositionalEncoding initialized | max_len: {max_len} | d_model: {d_model}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class ContextAwareAccentTagger(nn.Module):
    """Char-level sequence labeling with word-level context via embedding concatenation."""
    
    def __init__(self, char_vocab_size: int, word_vocab_size: int, char_pad_id: int, word_pad_id: int, cfg):
        super().__init__()

        assert cfg.model.char_emb_dim + cfg.model.word_emb_dim == cfg.model.d_model, (
            "char_emb_dim + word_emb_dim must equal d_model"
        )

        self.char_pad_id = char_pad_id
        self.word_pad_id = word_pad_id
        self.d_model = cfg.model.d_model

        self.char_embedding = nn.Embedding(char_vocab_size, cfg.model.char_emb_dim, padding_idx=char_pad_id)
        self.word_embedding = nn.Embedding(word_vocab_size, cfg.model.word_emb_dim, padding_idx=word_pad_id)
        
        self.input_dropout = nn.Dropout(cfg.model.dropout)
        self.positional_encoding = PositionalEncoding(cfg.model.d_model, cfg.model.max_len, cfg.model.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model.d_model,
            nhead=cfg.model.nhead,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.model.num_encoder_layers)
        self.output_layer = nn.Linear(cfg.model.d_model, char_vocab_size)
        
        logger.info(
            f"Model initialized | Char Emb: {cfg.model.char_emb_dim} | Word Emb: {cfg.model.word_emb_dim} | "
            f"Heads: {cfg.model.nhead} | Layers: {cfg.model.num_encoder_layers}"
        )

    def forward(self, src_char: torch.Tensor, src_word: torch.Tensor) -> torch.Tensor:
        padding_mask = src_char.eq(self.char_pad_id)

        char_emb = self.char_embedding(src_char)
        word_emb = self.word_embedding(src_word)

        x = torch.cat([char_emb, word_emb], dim=-1)
        x = x * math.sqrt(self.d_model)
        x = self.input_dropout(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        logits = self.output_layer(x)

        logger.debug(f"Forward pass completed | output shape: {logits.shape}")
        return logits


def build_model(char_vocab: CharVocab, word_vocab: WordVocab, cfg) -> nn.Module:
    logger.info(f"Building model on device: {cfg.training.device}")
    
    try:
        model = ContextAwareAccentTagger(
            char_vocab_size=len(char_vocab),
            word_vocab_size=len(word_vocab),
            char_pad_id=char_vocab.pad_id,
            word_pad_id=word_vocab.pad_id,
            cfg=cfg,
        ).to(cfg.training.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("Model built successfully.")
        logger.info(f"  > Total parameters    : {total_params:,}")
        logger.info(f"  > Trainable parameters: {trainable_params:,}")
        
        return model
    except Exception as e:
        logger.error("Failed to build or allocate model.", exc_info=True)
        raise e