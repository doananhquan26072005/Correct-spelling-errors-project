# diacritic_restoration/utils.py
import unicodedata
from typing import List
import Levenshtein

def remove_accents_char(ch: str) -> str:
    if ch == "đ":
        return "d"
    if ch == "Đ":
        return "D"

    normalized = unicodedata.normalize("NFD", ch)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def remove_accents_text(text: str) -> str:
    return "".join(remove_accents_char(ch) for ch in str(text))


def normalize_text(text: str, lowercase: bool = True) -> str:
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.strip().split())
    if lowercase:
        text = text.lower()
    return text


def apply_constraint_to_logits(logits, src, allowed_mask):
    """
    Force each position to predict only valid accented versions of src char.
    """
    position_allowed = allowed_mask[src]  # [B, L, V]
    return logits.masked_fill(~position_allowed, -1e9)


def word_accuracy(preds: List[str], targets: List[str]):
    correct_words = 0
    total_words = 0

    for pred, target in zip(preds, targets):
        pred_words = pred.split()
        target_words = target.split()
        min_len = min(len(pred_words), len(target_words))

        for i in range(min_len):
            if pred_words[i] == target_words[i]:
                correct_words += 1

        total_words += max(len(pred_words), len(target_words))

    return correct_words / max(1, total_words)


def accent_only_accuracy(inputs: List[str], preds: List[str], targets: List[str]):
    correct = 0
    total = 0

    for src, pred, target in zip(inputs, preds, targets):
        min_len = min(len(src), len(pred), len(target))

        for i in range(min_len):
            if src[i] != target[i]:
                total += 1
                if pred[i] == target[i]:
                    correct += 1

    return correct / max(1, total)


def compute_text_metrics(inputs: List[str], preds: List[str], targets: List[str]):
    correct_chars = 0
    total_chars = 0
    exact_match = 0
    total_edit_distance = 0
    total_target_chars = 0

    for pred, target in zip(preds, targets):
        if pred == target:
            exact_match += 1

        min_len = min(len(pred), len(target))
        for i in range(min_len):
            if pred[i] == target[i]:
                correct_chars += 1

        total_chars += max(len(pred), len(target))
        total_edit_distance += Levenshtein.distance(pred, target)
        total_target_chars += max(1, len(target))

    metrics = {
        "char_accuracy": correct_chars / max(1, total_chars),
        "word_accuracy": word_accuracy(preds, targets),
        "accent_only_accuracy": accent_only_accuracy(inputs, preds, targets),
        "exact_match": exact_match / max(1, len(targets)),
        "cer": total_edit_distance / max(1, total_target_chars),
    }
    
    return metrics