# diacritic_restoration/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Import trực tiếp từ vocab.py để tránh phụ thuộc vòng qua dataset.py
from diacritic_restoration.vocab import CharVocab, WordVocab
from diacritic_restoration.utils import apply_constraint_to_logits, compute_text_metrics

class DiacriticTrainer:
    """Class chịu trách nhiệm quản lý toàn bộ vòng đời huấn luyện và đánh giá 
    của mô hình Khôi phục dấu tiếng Việt (Bước 1)."""
    
    def __init__(self, model: nn.Module, char_vocab: CharVocab, word_vocab: WordVocab, allowed_mask, cfg):
        self.model = model
        self.char_vocab = char_vocab
        self.word_vocab = word_vocab
        self.allowed_mask = allowed_mask
        self.cfg = cfg
        
        # Khởi tạo Optimizer và Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=4,
        )
        self.best_valid_loss = float("inf")

    def _weighted_accent_loss(self, logits: torch.Tensor, tgt: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """Phương thức nội bộ: Tính toán Cross Entropy tùy biến tăng trọng số cho các từ thêm dấu."""
        vocab_size = logits.size(-1)
        pad_id = self.char_vocab.pad_id

        loss_per_token = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            tgt.reshape(-1),
            ignore_index=pad_id,
            reduction="none",
        ).reshape(tgt.shape)

        valid_mask = tgt.ne(pad_id)
        accent_mask = src.ne(tgt) & valid_mask
        copy_mask = src.eq(tgt) & valid_mask

        weights = torch.zeros_like(loss_per_token)
        weights = weights + copy_mask.float() * self.cfg.loss.copy_loss_weight
        weights = weights + accent_mask.float() * self.cfg.loss.accent_loss_weight

        loss = (loss_per_token * weights).sum() / weights.sum().clamp_min(1.0)
        return loss

    def _train_one_epoch(self, loader) -> float:
        """Huấn luyện mô hình qua 1 epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            src_char = batch["src_char"].to(self.cfg.training.device)
            src_word = batch["src_word"].to(self.cfg.training.device)
            tgt = batch["tgt"].to(self.cfg.training.device)

            logits = self.model(src_char, src_word)

            if self.cfg.loss.use_constraint_in_training:
                logits = apply_constraint_to_logits(logits, src_char, self.allowed_mask)

            loss = self._weighted_accent_loss(logits=logits, tgt=tgt, src=src_char)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader) -> float:
        """Đánh giá mô hình trên tập Validation."""
        self.model.eval()
        total_loss = 0.0

        for batch in loader:
            src_char = batch["src_char"].to(self.cfg.training.device)
            src_word = batch["src_word"].to(self.cfg.training.device)
            tgt = batch["tgt"].to(self.cfg.training.device)

            logits = self.model(src_char, src_word)

            if self.cfg.loss.use_constraint_in_training:
                logits = apply_constraint_to_logits(logits, src_char, self.allowed_mask)

            loss = self._weighted_accent_loss(logits=logits, tgt=tgt, src=src_char)
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate_dataset(self, test_loader) -> Dict[str, float]:
        """Phương thức quét qua tập test để tính toán và báo cáo chỉ số End-to-End."""
        self.model.eval()
        all_inputs = []
        all_preds = []
        all_targets = []

        for batch in test_loader:
            src_char = batch["src_char"].to(self.cfg.training.device)
            src_word = batch["src_word"].to(self.cfg.training.device)
            lengths = batch["length"].tolist()

            logits = self.model(src_char, src_word)
            logits = apply_constraint_to_logits(logits, src_char, self.allowed_mask)
            pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()

            preds = [
                self.char_vocab.decode(ids, original_length=length)
                for ids, length in zip(pred_ids, lengths)
            ]

            all_inputs.extend(batch["src_text"])
            all_preds.extend(preds)
            all_targets.extend(batch["tgt_text"])

        metrics = compute_text_metrics(all_inputs, all_preds, all_targets)

        print("\n" + "="*20 + " KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ " + "="*20)
        print(f"  • Char accuracy       : {metrics['char_accuracy']:.4f}")
        print(f"  • Word accuracy       : {metrics['word_accuracy']:.4f}")
        print(f"  • Accent-only accuracy: {metrics['accent_only_accuracy']:.4f}")
        print(f"  • Exact match (SER)   : {metrics['exact_match']:.4f}")
        print(f"  • CER                 : {metrics['cer']:.4f}\n")

        print("Xem thử một vài kết quả dự đoán thực tế:")
        for i in range(min(5, len(all_preds))):
            print("-" * 50)
            print("Đầu vào không dấu:", all_inputs[i])
            print("Mô hình khôi phục:", all_preds[i])
            print("Văn bản đích chuẩn:", all_targets[i])
            
        return metrics

    def fit(self, train_loader, valid_loader, test_loader=None) -> nn.Module:
        """Kích hoạt vòng lặp huấn luyện và tự động kích hoạt đánh giá tập test nếu có."""
        for epoch in range(1, self.cfg.training.epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            valid_loss = self.validate(valid_loader)

            self.scheduler.step(valid_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d}/{self.cfg.training.epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"valid loss: {valid_loss:.4f} | "
                f"lr: {current_lr:.2e}"
            )

            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                checkpoint_data = {
                    "model_state_dict": self.model.state_dict(),
                    "char_vocab_stoi": self.char_vocab.stoi,
                    "char_vocab_itos": self.char_vocab.itos,
                    "word_vocab_stoi": self.word_vocab.stoi if self.word_vocab else None,
                    "word_vocab_itos": self.word_vocab.itos if self.word_vocab else None,
                    "config": self.cfg,
                    "valid_loss": valid_loss,
                }
                torch.save(checkpoint_data, self.cfg.training.checkpoint_path)
                print(f"Saved best checkpoint to {self.cfg.training.checkpoint_path}")

        if test_loader is not None:
            print("\n[*] Đã kết thúc huấn luyện. Tự động chuyển sang đánh giá tập Test...")
            best_checkpoint = torch.load(self.cfg.training.checkpoint_path, map_location=self.cfg.training.device, weights_only=False)
            self.model.load_state_dict(best_checkpoint["model_state_dict"])
            self.evaluate_dataset(test_loader)

        return self.model