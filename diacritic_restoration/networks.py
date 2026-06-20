# diacritic_restoration/networks.py
import math
import torch
import torch.nn as nn

# Import trực tiếp từ vocab.py để tránh vòng lặp import qua dataset.py
from diacritic_restoration.vocab import CharVocab, WordVocab


class PositionalEncoding(nn.Module):
    """Lớp mã hóa vị trí (Positional Encoding) giúp Transformer nhận biết 
    thứ tự của các ký tự trong chuỗi văn bản."""
    
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class ContextAwareAccentTagger(nn.Module):
    """Mô hình Sequence Labeling cấp độ Ký tự (Char-level) kết hợp ngữ cảnh cấp độ Từ (Word-level).
    
    Ý tưởng: Mỗi ký tự nhận vào sẽ được ghép (concatenate) giữa:
        Char Embedding + Word Embedding của từ chứa ký tự đó.
    Sau đó, khối Transformer Encoder sẽ học ngữ cảnh trên toàn bộ câu để dự đoán dấu.
    """
    
    def __init__(self, char_vocab_size: int, word_vocab_size: int, char_pad_id: int, word_pad_id: int, cfg):
        super().__init__()

        # Kiểm tra tính hợp lệ của kiến trúc qua config lồng nhóm từ yaml
        assert cfg.model.char_emb_dim + cfg.model.word_emb_dim == cfg.model.d_model, (
            "char_emb_dim + word_emb_dim phải bằng d_model"
        )

        self.char_pad_id = char_pad_id
        self.word_pad_id = word_pad_id
        self.d_model = cfg.model.d_model

        # Các tầng nhúng (Embeddings)
        self.char_embedding = nn.Embedding(char_vocab_size, cfg.model.char_emb_dim, padding_idx=char_pad_id)
        self.word_embedding = nn.Embedding(word_vocab_size, cfg.model.word_emb_dim, padding_idx=word_pad_id)
        
        self.input_dropout = nn.Dropout(cfg.model.dropout)
        self.positional_encoding = PositionalEncoding(cfg.model.d_model, cfg.model.max_len, cfg.model.dropout)

        # Khối Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model.d_model,
            nhead=cfg.model.nhead,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.model.num_encoder_layers)
        
        # Tầng Fully Connected đầu ra dự đoán ID ký tự chuẩn
        self.output_layer = nn.Linear(cfg.model.d_model, char_vocab_size)

    def forward(self, src_char: torch.Tensor, src_word: torch.Tensor) -> torch.Tensor:
        # Tạo mặt nạ padding để Transformer bỏ qua các ký tự <pad>
        padding_mask = src_char.eq(self.char_pad_id)

        # Nhúng ký tự và từ tương ứng
        char_emb = self.char_embedding(src_char)
        word_emb = self.word_embedding(src_word)
        
        # Kết hợp tính năng cấp char và cấp word: Kích thước: [Batch, Length, d_model]
        x = torch.cat([char_emb, word_emb], dim=-1)
        x = x * math.sqrt(self.d_model)
        
        x = self.input_dropout(x)
        x = self.positional_encoding(x)
        
        # Đưa qua mạng self-attention của Transformer
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        
        return self.output_layer(x)


def build_model(char_vocab: CharVocab, word_vocab: WordVocab, cfg) -> nn.Module:
    """Hàm Factory hỗ trợ khởi tạo nhanh mô hình và tự động đẩy lên thiết bị phần cứng."""
    model = ContextAwareAccentTagger(
        char_vocab_size=len(char_vocab),
        word_vocab_size=len(word_vocab),
        char_pad_id=char_vocab.pad_id,
        word_pad_id=word_vocab.pad_id,
        cfg=cfg,
    ).to(cfg.training.device)
    
    print(f"Khởi tạo mô hình thành công với tổng số tham số: {sum(p.numel() for p in model.parameters()):,}")
    return model