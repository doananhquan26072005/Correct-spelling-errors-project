from model import ContextAwareAccentTagger
from utils import apply_constraint_to_logits, normalize_text
import torch
from config import Config
from vocab import CharVocab, WordVocab, encode_word_ids_per_char
from vocab import build_allowed_token_mask
from metric import compute_text_metrics

# Test evaluation
@torch.no_grad()
def predict_batch(model, src_char, src_word, allowed_mask):
    model.eval()
    logits = model(src_char, src_word)
    logits = apply_constraint_to_logits(logits, src_char, allowed_mask)
    return logits.argmax(dim=-1)


@torch.no_grad()
def evaluate_test(model, test_loader, char_vocab: CharVocab, allowed_mask, cfg: Config):
    model.eval()

    all_inputs = []
    all_preds = []
    all_targets = []

    for batch in test_loader:
        src_char = batch["src_char"].to(cfg.device)
        src_word = batch["src_word"].to(cfg.device)
        lengths = batch["length"].tolist()

        pred_ids = predict_batch(model, src_char, src_word, allowed_mask).detach().cpu().tolist()

        preds = [
            char_vocab.decode(ids, original_length=length)
            for ids, length in zip(pred_ids, lengths)
        ]

        all_inputs.extend(batch["src_text"])
        all_preds.extend(preds)
        all_targets.extend(batch["tgt_text"])

    metrics = compute_text_metrics(all_inputs, all_preds, all_targets)

    print("Test metrics:")
    print(f"  Char accuracy       : {metrics['char_accuracy']:.4f}")
    print(f"  Word accuracy       : {metrics['word_accuracy']:.4f}")
    print(f"  Accent-only accuracy: {metrics['accent_only_accuracy']:.4f}")
    print(f"  Exact match         : {metrics['exact_match']:.4f}")
    print(f"  CER                 : {metrics['cer']:.4f}")

    print("\nSome predictions:")
    for i in range(min(10, len(all_preds))):
        print("-" * 60)
        print("Input :", all_inputs[i])
        print("Pred  :", all_preds[i])
        print("Target:", all_targets[i])

    return metrics


# Inference

def restore_accents(text: str, model, char_vocab: CharVocab, word_vocab: WordVocab, allowed_mask, cfg: Config):
    """
    Inference safe:
    - Char in vocab: model predict.
    - char not vocab: not change.
    - if model predict <unk> / <pad>: copy input.
    """

    model.eval()

    text = normalize_text(str(text), lowercase=cfg.lowercase)

    src_char_ids = char_vocab.encode(text, cfg.max_len)
    src_word_ids = encode_word_ids_per_char(
        text=text,
        word_vocab=word_vocab,
        max_len=cfg.max_len,
    )

    src_char = torch.tensor([src_char_ids], dtype=torch.long, device=cfg.device)

    src_word = torch.tensor([src_word_ids], dtype=torch.long,device=cfg.device)

    with torch.no_grad():
        pred_ids = predict_batch(
            model=model,
            src_char=src_char,
            src_word=src_word,
            allowed_mask=allowed_mask,
        )

    pred_ids = pred_ids[0].detach().cpu().tolist()

    output_chars = []

    limit = min(len(text), cfg.max_len)

    for i in range(limit):
        input_ch = text[i]

        if input_ch not in char_vocab.stoi:
            output_chars.append(input_ch)
            continue

        pred_id = int(pred_ids[i])
        pred_ch = char_vocab.itos.get(pred_id)

        if pred_ch is None:
            output_chars.append(input_ch)
            continue

        if pred_ch in char_vocab.special_tokens:
            output_chars.append(input_ch)
            continue

        output_chars.append(pred_ch)

    if len(text) > cfg.max_len:
        output_chars.append(text[cfg.max_len:])

    return "".join(output_chars)


def load_checkpoint(checkpoint_path: str, cfg: Config):
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)

    if "config" in checkpoint:
        for key, value in checkpoint["config"].items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

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
        cfg=cfg,
    ).to(cfg.device)

    # Strip wrapper prefixes
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value
        elif key.startswith("_orig_mod."):
            new_state_dict[key[10:]] = value
        elif key.startswith("model."):
            new_state_dict[key[6:]] = value
        else:
            new_state_dict[key] = value

    try:
        model.load_state_dict(new_state_dict)
    except Exception as e:
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()

    allowed_mask = build_allowed_token_mask(char_vocab, cfg)

    return model, char_vocab, word_vocab, allowed_mask