import torch
import torch.nn as nn
from config import Config
from vocab import CharVocab
from utils import apply_constraint_to_logits
import torch.nn.functional as F
from vocab import CharVocab, WordVocab
from model import ContextAwareAccentTagger

def weighted_accent_loss(logits, tgt, src, pad_id: int, cfg: Config):
    """
    Cross entropy follow each char, but increase weight for accent characters.

    reason:
        most chars are copy, if we don't increase accent weight, 
        can get high char acc by copying a lot, but accent-only acc is still low.
    """
    vocab_size = logits.size(-1)

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
    weights = weights + copy_mask.float() * cfg.copy_loss_weight
    weights = weights + accent_mask.float() * cfg.accent_loss_weight

    loss = (loss_per_token * weights).sum() / weights.sum().clamp_min(1.0)

    return loss


def train_one_epoch(model, loader, optimizer, allowed_mask, cfg: Config, char_vocab: CharVocab):
    model.train()
    total_loss = 0.0

    for batch in loader:
        src_char = batch["src_char"].to(cfg.device)
        src_word = batch["src_word"].to(cfg.device)
        tgt = batch["tgt"].to(cfg.device)

        logits = model(src_char, src_word)

        if cfg.use_constraint_in_training:
            logits = apply_constraint_to_logits(logits, src_char, allowed_mask)

        loss = weighted_accent_loss(
            logits=logits,
            tgt=tgt,
            src=src_char,
            pad_id=char_vocab.pad_id,
            cfg=cfg,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, allowed_mask, cfg: Config, char_vocab: CharVocab):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        src_char = batch["src_char"].to(cfg.device)
        src_word = batch["src_word"].to(cfg.device)
        tgt = batch["tgt"].to(cfg.device)

        logits = model(src_char, src_word)

        if cfg.use_constraint_in_training:
            logits = apply_constraint_to_logits(logits, src_char, allowed_mask)

        loss = weighted_accent_loss(
            logits=logits,
            tgt=tgt,
            src=src_char,
            pad_id=char_vocab.pad_id,
            cfg=cfg,
        )

        total_loss += loss.item()

    return total_loss / len(loader)


def train_model(model, train_loader, valid_loader, char_vocab: CharVocab, word_vocab: WordVocab, allowed_mask, cfg: Config):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    best_valid_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            allowed_mask=allowed_mask,
            cfg=cfg,
            char_vocab=char_vocab,
        )

        valid_loss = validate(
            model=model,
            loader=valid_loader,
            allowed_mask=allowed_mask,
            cfg=cfg,
            char_vocab=char_vocab,
        )

        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"valid loss: {valid_loss:.4f} | "
            f"lr: {current_lr:.2e}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "char_vocab_stoi": char_vocab.stoi,
                    "char_vocab_itos": char_vocab.itos,
                    "word_vocab_stoi": word_vocab.stoi,
                    "word_vocab_itos": word_vocab.itos,
                    "config": cfg.__dict__,
                    "valid_loss": valid_loss,
                },
                cfg.checkpoint_path,
            )
            print(f"Saved best checkpoint to {cfg.checkpoint_path}")

    return model

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