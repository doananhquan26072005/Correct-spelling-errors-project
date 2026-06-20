# diacritic_restoration/trainer.py
import time
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from diacritic_restoration.utils import apply_constraint_to_logits, compute_text_metrics
from diacritic_restoration.vocab import CharVocab, WordVocab
from common.logger import get_logger

logger = get_logger(__name__)


class DiacriticTrainer:
    def __init__(self, model: nn.Module, char_vocab: CharVocab, word_vocab: WordVocab, allowed_mask, cfg):
        self.model = model
        self.char_vocab = char_vocab
        self.word_vocab = word_vocab
        self.allowed_mask = allowed_mask
        self.cfg = cfg
        
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
        logger.info("DiacriticTrainer successfully initialized with tqdm integration.")

    def _weighted_accent_loss(self, logits: torch.Tensor, tgt: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """
        Cross entropy follow each char, but increase weight for accent characters.

        reason:
            most chars are copy, if we don't increase accent weight, 
            can get high char acc by copying a lot, but accent-only acc is still low.
        """
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

    def _train_one_epoch(self, loader, epoch_idx: int) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(
            enumerate(loader, 1),
            total=len(loader),
            desc=f"Epoch {epoch_idx:03d} [Train]",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            leave=False 
        )

        for batch_idx, batch in pbar:
            src_char = batch["src_char"].to(self.cfg.training.device)
            src_word = batch["src_word"].to(self.cfg.training.device)
            tgt = batch["tgt"].to(self.cfg.training.device)

            logits = self.model(src_char, src_word)

            if self.cfg.loss.use_constraint_in_training:
                logits = apply_constraint_to_logits(logits, src_char, self.allowed_mask)

            loss = self._weighted_accent_loss(logits=logits, tgt=tgt, src=src_char)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Loss exploded to {loss.item()} at batch {batch_idx}! Terminating gradient step.")
                raise ValueError("Loss explosion detected.")

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
            self.optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss

            pbar.set_postfix({"batch_loss": f"{current_loss:.4f}"})

            if batch_idx % 10 == 0:
                logger.debug(f"Epoch {epoch_idx} | Batch {batch_idx}/{len(loader)} | Loss: {current_loss:.4f}")

        avg_loss = total_loss / len(loader)
        return avg_loss

    @torch.no_grad()
    def validate(self, loader) -> float:
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
        self.model.eval()
        logger.info("Executing End-to-End evaluation process on test dataset...")
        
        all_inputs = []
        all_preds = []
        all_targets = []

        pbar = tqdm(test_loader, desc="Evaluating [Test]", leave=False)
        for batch in pbar:
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

        logger.info("==================== TEST SET PERFORMANCE METRICS ====================")
        logger.info(f"  • Char accuracy        : {metrics['char_accuracy']:.4f}")
        logger.info(f"  • Word accuracy        : {metrics['word_accuracy']:.4f}")
        logger.info(f"  • Accent-only accuracy : {metrics['accent_only_accuracy']:.4f}")
        logger.info(f"  • Exact match (SER)    : {metrics['exact_match']:.4f}")
        logger.info(f"  • CER                  : {metrics['cer']:.4f}")
        logger.info("======================================================================")
            
        return metrics

    def fit(self, train_loader, valid_loader, test_loader=None) -> nn.Module:
        logger.info(f"Starting model optimization pipeline. Total epochs: {self.cfg.training.epochs}")
        
        start_total_time = time.time()

        for epoch in range(1, self.cfg.training.epochs + 1):
            train_loss = self._train_one_epoch(train_loader, epoch_idx=epoch)
            valid_loss = self.validate(valid_loader)

            self.scheduler.step(valid_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch:03d}/{self.cfg.training.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Valid Loss: {valid_loss:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            if valid_loss < self.best_valid_loss:
                logger.info(f"Valid loss decreased from {self.best_valid_loss:.4f} to {valid_loss:.4f}. Saving checkpoint...")
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

        total_train_time = time.time() - start_total_time

        hours = int(total_train_time // 3600)
        minutes = int((total_train_time % 3600) // 60)
        seconds = int(total_train_time % 60)
        
        logger.info("======================================================================")
        logger.info(f"[*] TRAINING CYCLE FINISHED.")
        logger.info(f"    > Total Training Time: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_train_time:.2f} seconds)")
        logger.info("======================================================================")

        if test_loader is not None:
            logger.info("Activating automatic test evaluation on best secured checkpoint model...")
            try:
                best_checkpoint = torch.load(self.cfg.training.checkpoint_path, map_location=self.cfg.training.device, weights_only=False)
                self.model.load_state_dict(best_checkpoint["model_state_dict"])
                self.evaluate_dataset(test_loader)
            except Exception:
                logger.error("Failed to load the best saved checkpoint for test evaluation.", exc_info=True)

        return self.model