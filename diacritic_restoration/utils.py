import unicodedata


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
    """Force each position to predict only valid accented versions of src char."""
    position_allowed = allowed_mask[src]  # [B, L, V]
    return logits.masked_fill(~position_allowed, -1e9)

