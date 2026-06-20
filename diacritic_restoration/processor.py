import json
from types import SimpleNamespace
from typing import List, Tuple, Dict, Any
import pandas as pd
import torch

# --- IMPORT NỘI BỘ PHÂN HỆ DIACRITIC_RESTORATION ---
# Các class từ điển và hàm mã hóa (gộp trong dataset.py)
from diacritic_restoration.vocab import (
    CharVocab,
    WordVocab,
    build_allowed_token_mask,
    encode_word_ids_per_char,
)

# Các hàm helper và đo đạc (gộp trong utils.py)
from diacritic_restoration.utils import (
    normalize_text, 
    remove_accents_text, 
    apply_constraint_to_logits
)

# Kiến trúc mô hình mạng nơ-ron Transformer (nằm trong networks.py)
from diacritic_restoration.networks import ContextAwareAccentTagger

class DiacriticDataProcessor:
    def __init__(self, cfg):
        self.cfg = cfg

    def load_words_from_txt(self) -> List[str]:
        words = []
        with open(self.cfg.data.words_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    item = json.loads(line)
                    word = str(item.get("text", "")).strip()
                    if word: words.append(word)
                except json.JSONDecodeError: continue
        return words

    def make_pair_from_target(self, target: str):
        tgt = normalize_text(target, lowercase=self.cfg.augmentation.lowercase)
        if not tgt or len(tgt) > self.cfg.model.max_len: return None
        src = remove_accents_text(tgt)
        if len(src) != len(tgt): return None
        return {self.cfg.data.input_col: src, self.cfg.data.target_col: tgt}

    def clean_original_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        pairs = []
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
                if pair is not None: pairs.append(pair)
        return pd.DataFrame(pairs).drop_duplicates().reset_index(drop=True)

    def build_word_pairs_from_words_file(self) -> pd.DataFrame:
        if not self.cfg.augmentation.use_word_pairs:
            return pd.DataFrame(columns=[self.cfg.data.input_col, self.cfg.data.target_col])
        words = self.load_words_from_txt()
        pairs = []
        for word in words:
            pair = self.make_pair_from_target(word)
            if pair is None or len(pair[self.cfg.data.target_col]) < 2: continue
            pairs.append(pair)
            if len(pairs) >= self.cfg.augmentation.max_word_pairs: break
        return pd.DataFrame(pairs).drop_duplicates().reset_index(drop=True)

    def build_sentence_chunks_from_targets(self, train_df: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.augmentation.use_sentence_chunks:
            return pd.DataFrame(columns=[self.cfg.data.input_col, self.cfg.data.target_col])
        pairs = []
        for tgt in train_df[self.cfg.data.target_col].astype(str).tolist():
            tgt = normalize_text(tgt, lowercase=self.cfg.augmentation.lowercase)
            words = tgt.split()
            if len(words) < self.cfg.augmentation.chunk_min_words: continue
            max_w = min(self.cfg.augmentation.chunk_max_words, len(words))
            for window in range(self.cfg.augmentation.chunk_min_words, max_w + 1):
                for start in range(0, len(words) - window + 1, self.cfg.augmentation.chunk_stride):
                    chunk = " ".join(words[start : start + window])
                    pair = self.make_pair_from_target(chunk)
                    if pair is not None: pairs.append(pair)
        return pd.DataFrame(pairs).drop_duplicates().reset_index(drop=True)

    def split_dataframe(self, df: pd.DataFrame):
        df = df.sample(frac=1.0, random_state=self.cfg.training.seed).reset_index(drop=True)
        n_total = len(df)
        n_train = int(n_total * self.cfg.data.train_ratio)
        n_valid = int(n_total * self.cfg.data.valid_ratio)
        return (
            df.iloc[:n_train].reset_index(drop=True),
            df.iloc[n_train:n_train + n_valid].reset_index(drop=True),
            df.iloc[n_train + n_valid:].reset_index(drop=True)
        )
    
class DiacriticRestorer:
    """Class chịu trách nhiệm load mô hình khôi phục dấu và thực hiện inference 
    (suy luận) cho chuỗi văn bản không dấu đầu vào."""
    
    def __init__(self, checkpoint_path: str, cfg):
        self.cfg = cfg
        self.model, self.char_vocab, self.word_vocab, self.allowed_mask = self._load_checkpoint(checkpoint_path)

    def _predict_batch(self, src_char: torch.Tensor, src_word: torch.Tensor) -> torch.Tensor:
        """Hàm dự đoán nhãn tối ưu kết hợp mặt nạ ràng buộc (Allowed Mask)."""
        logits = self.model(src_char, src_word)
        logits = apply_constraint_to_logits(logits, src_char, self.allowed_mask)
        return logits.argmax(dim=-1)

    def process(self, text: str) -> str:
        """Phương thức Public: Nhận vào văn bản không dấu, trả về văn bản có dấu chuẩn."""
        self.model.eval()
        text = normalize_text(str(text), lowercase=self.cfg.augmentation.lowercase)

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
            output_chars.append(text[self.cfg.model.max_len:])

        return "".join(output_chars)

    def _load_checkpoint(self, checkpoint_path: str):
        """Hàm nội bộ: Tự động phân tách weight wrapper và load mô hình an toàn."""
        torch.serialization.add_safe_globals([SimpleNamespace])
        checkpoint = torch.load(checkpoint_path, map_location=self.cfg.training.device)

        # Cập nhật ngược lại các thông số config tĩnh từ checkpoint nếu có
        if "config" in checkpoint:
            # Lưu ý xử lý an toàn tùy thuộc vào cấu trúc Box hay Namespace
            pass

        char_vocab = CharVocab()
        char_vocab.stoi = checkpoint["char_vocab_stoi"]
        char_vocab.itos = {int(k): v for k, v in checkpoint["char_vocab_itos"].items()}

        word_vocab = WordVocab()
        word_vocab.stoi = checkpoint["word_vocab_stoi"]
        word_vocab.itos = {int(k): v for k, v in checkpoint["word_vocab_itos"].items()}

        model = ContextAwareAccentTagger(
            char_vocab_size=len(char_vocab),
            word_vocab_size=len(word_vocab),
            char_pad_id=char_vocab.pad_id,
            word_pad_id=word_vocab.pad_id,
            cfg=self.cfg,
        ).to(self.cfg.training.device)

        # Loại bỏ các prefix wrapper sinh ra do DataParallel hoặc torch.compile
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."): new_state_dict[key[7:]] = value
            elif key.startswith("_orig_mod."): new_state_dict[key[10:]] = value
            elif key.startswith("model."): new_state_dict[key[6:]] = value
            else: new_state_dict[key] = value

        try:
            model.load_state_dict(new_state_dict)
        except Exception:
            model.load_state_dict(new_state_dict, strict=False)

        model.eval()
        allowed_mask = build_allowed_token_mask(char_vocab, self.cfg)
        return model, char_vocab, word_vocab, allowed_mask