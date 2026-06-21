# Viet_khong_dau — Comprehensive Pipeline Report: Vietnamese Diacritic Restoration

> A summary of the full training pipeline for a **diacritic restoration** model for Vietnamese, prepared for a final-semester Natural Language Processing (NLP) project report.

---

## 1. Background & Problem Motivation

On messaging platforms and social media, Vietnamese users frequently type **unaccented text** (e.g., `"toi ten la"` instead of `"tôi tên là"` — "my name is"). Automatic diacritic restoration helps to:

- Improve typing experience / autocorrect.
- Provide preprocessing for downstream NLP tasks (tokenization, classification, machine translation).
- Normalize inconsistent Vietnamese text.

**Problem solved:** given an **unaccented** sentence as input, predict the **accented** equivalent sentence with the same meaning.

---

## 2. Problem Formulation

Instead of treating this as a **seq2seq machine translation** problem (autoregressive, slow), the project reformulates it as a **Char-Level Sequence Labeling** task:

> Each input character is assigned a label: the correctly accented character that corresponds to it.

| Input (src, no accents) | `t` | `o` | `i` |   | `t` | `e` | `n` |
|--------------------------|-----|-----|-----|---|-----|-----|-----|
| Target (accents)         | `t` | `ô` | `i` |   | `t` | `ê` | `n` |

**Advantages of this approach:**
- Input and output have the **same length** → uses a Transformer Encoder + Linear head (no Decoder needed, no teacher-forcing).
- Fast training; inference runs the whole sentence in parallel in a single forward pass.
- Combined with **word-level context** (each character also knows which word it belongs to), so the model can use local context.

---

## 3. Architecture Overview

```
                        ┌─────────────── src_char (B, L) ───────────────┐
   Char Embedding       │  Char Embedding (char_emb_dim=128)             │
                        └───────────────────────────────────────────────┘
                        ┌─────────────── src_word (B, L) ───────────────┐
   Word Embedding       │  Word Embedding (word_emb_dim=64)             │  ← same word_id for every char in a word
                        └───────────────────────────────────────────────┘
                          │ concat along feature dim → d_model=192
                          ▼
                 Positional Encoding (sin/cos)
                          ▼
            Transformer Encoder (4 layers, 4 heads, FFN=768, Pre-Norm)
                          ▼
            Linear(d_model → vocab_size)  → logits (B, L, V)
                          ▼
            Constraint Mask (allowed_mask)  → block non-linguistic predictions
                          ▼
            argmax → predicted accented character
```

**Core idea:** each character receives a **char embedding + the word embedding of the word containing it**; Transformer self-attention learns sentence-wide context; a linear layer projects to the character vocabulary size to predict the accent.

---

## 4. Pipeline / Data Flow

```
viet_khong_dau.csv  ──► build_dataloaders()
words.txt            ─►  │
                         ├── clean_original_pairs()          (filter valid pairs)
                         ├── split_dataframe()               (80/10/10)
                         ├── build_sentence_chunks_from_targets()  (augment)
                         ├── build_word_pairs_from_words_file()     (augment, optional)
                         ├── build_vocab_from_words_file_and_dataframe()  (CharVocab)
                         ├── build_word_vocab()                       (WordVocab)
                         └── AccentContextDataset → DataLoader (train/valid/test)

CharVocab ──► build_allowed_token_mask() ──► allowed_mask (V×V boolean)

DataLoader + vocab + allowed_mask ──► build_model() ──► ContextAwareAccentTagger
                                      │
                                      ├── train_model()  (AdamW + ReduceLROnPlateau)
                                      │      ├── train_one_epoch()  + weighted_accent_loss()
                                      │      └── validate()
                                      │      └── save best checkpoint by valid loss
                                      ▼
                        load_checkpoint() ──► evaluate_test() + restore_accents() (inference)
                                              └── metric.compute_text_metrics()
```

---

## 5. Folder Structure

```
Viet_khong_dau/
├── config.py          # Global configuration (dataclass Config)
├── vocab.py           # Char/word vocabularies + augmentation + constraint mask
├── dataset.py         # PyTorch Dataset & DataLoader
├── model.py           # Neural network architecture (Transformer Encoder)
├── training.py        # Training loop + custom loss
├── inference.py       # Test evaluation + real-world inference + checkpoint loading
├── metric.py          # Evaluation metrics (CER, char/word/exact-match acc)
├── utils.py           # Text utilities (strip accents, normalize, constraint)
├── main.py            # Entry point running the full pipeline
├── accent_tagger_context_v3.pt   # Trained model checkpoint
├── README.md
└── PROJECT_SUMMARY_EN.md   # ← this document
```

---

## 6. Detailed Module / Function Reference

### 6.1 `config.py` — Configuration Hub

All configuration lives in a single `@dataclass` for easy tuning.

| Group | Field | Value | Purpose |
|-------|-------|-------|---------|
| Paths | `csv_path`, `words_path` | – | Sentence-pair data + word dictionary |
| Columns | `input_col="input"`, `target_col="target"` | – | CSV column names |
| Sequence | `max_len=256`, `batch_size=64` | – | Max length / batch size |
| **Model** | `d_model=192`, `char_emb_dim=128`, `word_emb_dim=64` | – | `char_emb + word_emb = d_model` (required) |
| | `nhead=4`, `num_encoder_layers=4`, `dim_feedforward=768`, `dropout=0.2` | – | Transformer hyperparams |
| **Training** | `epochs=80`, `learning_rate=1e-4`, `weight_decay=1e-4`, `grad_clip=1.0` | – | Optimization & gradient stability |
| Splits | `train/valid/test_ratio = 0.8/0.1/0.1`, `seed=42` | – | Reproducibility |
| **Augmentation** | `use_sentence_chunks=True`, `chunk_min_words=5`, `chunk_max_words=24`, `chunk_stride=2` | – | Generate sub-sentences preserving local context |
| | `use_word_pairs=False`, `max_word_pairs=30000` | – | Generate single-word pairs (optional) |
| | `max_train_samples=300000` | – | Cap on training data |
| **Loss balancing** | `copy_loss_weight=1.0`, `accent_loss_weight=10.0` | – | Heavily penalize accent positions (rare) |
| Constraint | `use_constraint_in_training=True` | – | Use mask during training loss |
| Word vocab | `max_word_vocab_size=50000`, `min_word_freq=1` | – | Word vocabulary cap |
| | `device` | cuda/cpu | Auto-detect GPU |

> **Note for report:** `d_model = char_emb_dim + word_emb_dim` is an invariant asserted in `model.py` — this is required for the two embeddings to concat to the correct dimension.

---

### 6.2 `utils.py` — Text Utilities

Foundational module used everywhere in the project.

| Function | Role |
|----------|------|
| `remove_accents_char(ch)` | Strip accents from a single char using `unicodedata.normalize("NFD")` and filtering category `Mn` (mark, nonspacing). Special-cases `đ→d`, `Đ→D` because they don't decompose via NFD. |
| `remove_accents_text(text)` | Apply `remove_accents_char` across a whole string → produce an unaccented input from an accented sentence. |
| `normalize_text(text, lowercase=True)` | (1) NFC Unicode normalization, (2) collapse extra whitespace, (3) lowercase if requested. Ensures consistent text across the pipeline. |
| `apply_constraint_to_logits(logits, src, allowed_mask)` | **Output constraint:** uses `allowed_mask[src]` to get a valid-position matrix `[B,L,V]`, sets `-1e9` on disallowed logit positions → after `argmax` the model can only pick a valid accented variant of the source character. |

> **Why it matters:** `apply_constraint_to_logits` is the "key trick" that turns the problem from "predict across the entire vocab" into "choose only among the valid accent variants of that character" (e.g., `a` can only become `a/á/à/ả/ã/ạ/â/ơ/...`, never `đ`). It both shrinks the search space and eliminates non-linguistic errors.

---

### 6.3 `vocab.py` — Vocabularies, Augmentation & Constraint Mask

The largest module, containing three groups of logic: (1) building vocabularies, (2) generating augmented data, (3) the constraint matrix.

#### a) Vocabulary classes

| Class | Role |
|-------|------|
| `CharVocab` | Maps characters ↔ IDs. Special tokens: `<pad>`, `<unk>`. `build(texts)` collects all unique chars; `encode(text, max_len)` pads/truncates to `max_len`; `decode(ids, original_length)` drops pad and rejoins the string. |
| `WordVocab` | Maps words ↔ IDs. Special tokens: `<pad_word>`, `<unk_word>`, `<space>`. `build(texts, max_size, min_freq)` counts frequencies and keeps the top words. |

#### b) Encoding / helper functions

| Function | Role |
|----------|------|
| `encode_word_ids_per_char(text, word_vocab, max_len)` | For each character, assigns the **word_id of the word containing it** (whitespace → `space_id`). Result is an array with length = number of characters, telling the model "which word this char belongs to". |
| `build_word_vocab(train_df, cfg)` | Initializes `WordVocab` from the `input` column of the train set. |
| `load_words_from_txt(words_path)` | Reads `words.txt` as JSONL, takes the `text` field (Vietnamese dictionary). |
| `build_vocab_from_words_file_and_dataframe(df, cfg)` | Builds `CharVocab` from the union of `words.txt` + input/target columns → ensures every valid character is present. |

#### c) Cleaning & pair generation

| Function | Role |
|----------|------|
| `make_pair_from_target(target, cfg)` | From an **accented** sentence: normalize → truncate to `max_len` → `src = remove_accents(tgt)`. Only keeps the pair when `len(src)==len(tgt)` (guarantees 1-to-1 labeling). |
| `clean_original_pairs(df, cfg)` | Filter original CSV pairs: keep only when `len(src)==len(tgt)` **and** `remove_accents(tgt)==src`. Failed pairs are attempted to be recovered via `make_pair_from_target`. Drops duplicates. |
| `split_dataframe(df, cfg)` | Shuffle (seeded) → split 80/10/10 train/valid/test. |

#### d) Augmentation

| Function | Role |
|----------|------|
| `build_sentence_chunks_from_targets(train_df, cfg)` | Slides a window over **words** (stride=2, size 5→24 words) on real target sentences → generates **sub-chunks**. Better than random words because it preserves **real local context**, exposing the model to varied sentence lengths. |
| `build_word_pairs_from_words_file(cfg)` | Generates `unaccented → accented` word pairs from `words.txt`. Off by default (`use_word_pairs=False`). Teaches valid accent forms but with weak context. |

#### e) Constraint matrix (core)

| Function | Role |
|----------|------|
| `build_allowed_token_mask(vocab, cfg)` | Builds a boolean matrix `V×V`: `allowed[input_id, output_id]=True` when `output` is a valid accented form of `input` (same base after accent removal). `pad→pad`, `unk→unk`. Examples: `d→{d, đ}`, `o→{o, ò, ó, ỏ, õ, ọ, ô, ồ, ố,..., ơ, ờ, ớ,...}`, `t→{t}`. Moved to `cfg.device`. |

> **Emphasize in report:** sentence-chunk augmentation + constraint mask + weighted loss are the three main techniques addressing two core issues: (1) limited data, (2) imbalance between "copy characters" (majority) and "accent-bearing characters" (minority).

---

### 6.4 `dataset.py` — Loading & Preparing PyTorch Data

| Component | Role |
|-----------|------|
| `AccentContextDataset` (Dataset) | `__init__`: normalize input/target, drop bad samples (length mismatch / base mismatch), keep raw text. `__getitem__`: encode `src_char_ids`, `tgt_ids`, `src_word_ids`, compute `length`; returns a dict of tensors + original strings (used for printing results). |
| `build_dataloaders(cfg)` | **Orchestrates all preprocessing:** read CSV → `clean_original_pairs` → `split_dataframe` → concatenate augmentation (chunks/word_pairs) → cap at `max_train_samples` → build `CharVocab` & `WordVocab` → create 3 `AccentContextDataset` → wrap in 3 `DataLoader`. |

> Note: augmentation is **only applied to the train split**; valid/test keep original data for honest evaluation. `DataLoader` uses `num_workers=0`.

---

### 6.5 `model.py` — Neural Network Architecture

| Component | Role |
|-----------|------|
| `PositionalEncoding` | Classic sin/cos positional encoding (max_len=5000). Since Transformers don't inherently know order, this injects position information. Supports odd `d_model`. |
| `ContextAwareAccentTagger` | The main model. **forward:** padding_mask from `src_char==pad_id`; `char_emb ⊕ word_emb` (concat) → multiply by `√d_model` → dropout → positional encoding → `TransformerEncoder` (`norm_first=True`, Pre-Norm, more stable) → `Linear(d_model→V)`. Returns logits. |
| `build_model(...)` | Initialize the model from vocab sizes, print parameter count. |

**Architectural points worth noting (for the report):**
- **Pre-LN** (`norm_first=True`): more stable training than Post-LN, especially for a small model trained from scratch.
- **Concat char+word embeddings** rather than chars only: injects word-level context at the input.
- **Padding mask**: ignored pad positions during attention.
- Small model (d_model=192, 4 layers) → suits limited data, avoids overfitting.

---

### 6.6 `training.py` — Training Loop & Custom Loss

| Function | Role |
|----------|------|
| `weighted_accent_loss(logits, tgt, src, pad_id, cfg)` | Cross-Entropy per-token (`reduction="none"`, ignore pad). Classify positions: `accent_mask = src≠tgt` (needs accent change) & `copy_mask = src==tgt` (keep as-is). Apply weights: accent ×10, copy ×1. Weighted mean → **forces the model to focus on learning accents instead of "getting lucky" by copying.** |
| `train_one_epoch(...)` | One epoch: forward → (optional) constraint mask → `weighted_accent_loss` → `zero_grad` → `backward` → `clip_grad_norm_` (prevent explosions) → `optimizer.step()`. Returns average loss. |
| `validate(...)` | Same as above but `@torch.no_grad`, `model.eval()`, no weight updates. |
| `train_model(...)` | Main loop: optimizer **AdamW** + scheduler `ReduceLROnPlateau` (factor=0.5, patience=4). Each epoch: train + validate + scheduler.step(valid_loss). Saves the best checkpoint (by valid loss) including state_dict, vocab, config. |
| `build_model(...)` | (duplicate of `model.py`'s, present here for convenient import in `main.py`) |

> **Report:** the custom loss addresses **class imbalance** at the character level — since most characters are "copies," plain CE loss makes the model lazy at copying. Multiplying accent positions by 10× is a key design decision, equivalent to focal/class-weighting.

---

### 6.7 `inference.py` — Evaluation & Inference

| Function | Role |
|----------|------|
| `predict_batch(model, src_char, src_word, allowed_mask)` | Forward + constraint mask → `argmax` returns predicted character IDs. |
| `evaluate_test(model, test_loader, char_vocab, allowed_mask, cfg)` | Run the full test set, decode predictions, compute `compute_text_metrics`, print metrics + 10 sample predictions (Input/Pred/Target). |
| `restore_accents(text, model, char_vocab, word_vocab, allowed_mask, cfg)` | **Safe real-world inference:** normalize → encode → predict → decode; handle `<unk>`/`<pad>` by **keeping the original character**; out-of-vocab characters (punctuation, emoji) are copied verbatim; the part beyond `max_len` is re-appended. |
| `load_checkpoint(checkpoint_path, cfg)` | Load checkpoint, restore `Config`, rebuild `CharVocab`/`WordVocab` from saved dicts, create model, **strip wrapper prefixes** (`module.`/`_orig_mod.`/`model.` from DataParallel/torch.compile), load state_dict. Rebuild `allowed_mask`. |

> **Robustness of `restore_accents`:** since real user text may contain odd characters (emoji, extended Latin punctuation), this function doesn't "break" them but keeps them as-is, making the model usable beyond the test set.

---

### 6.8 `metric.py` — Evaluation Metrics

| Function / Metric | Definition |
|--------------------|-----------|
| `word_accuracy(preds, targets)` | Ratio of fully-correct predicted words (compared word-by-word). Denominator = max(word count). |
| `accent_only_accuracy(inputs, preds, targets)` | Considers only positions where `src[i] != tgt[i]` (where an accent is needed). Ratio of correct predictions at **exactly the accent-bearing positions** → reflects accent-learning quality, excluding "lucky copies". |
| `compute_text_metrics(...)` | Aggregates: `char_accuracy`, `word_accuracy`, `accent_only_accuracy`, `exact_match`, `cer`. Uses `Levenshtein.distance` (library) for CER. |

**Meaning of metrics (suggested framing for the report):**
- **Char accuracy**: easy to achieve high (most chars are copies) → not convincing on its own.
- **Word accuracy**: stricter — a word is correct only when every character is correct.
- **Accent-only accuracy**: the most important metric, measuring the core task.
- **Exact match**: ratio of sentences predicted 100% correctly.
- **CER**: Character Error Rate (normalized Levenshtein distance) — lower is better.

---

### 6.9 `main.py` — Entry Point

Execution order:
1. `cfg = Config()`.
2. `build_dataloaders(cfg)` → 3 loaders + 2 vocabs.
3. `build_allowed_token_mask` → `allowed_mask`.
4. `build_model(...)` → with `DataParallel` if multiple GPUs.
5. *(the `train_model` block is commented out — uses a pre-trained checkpoint).*
6. `load_checkpoint(...)` → load model + vocab from `accent_tagger_context_v3.pt`.
7. `evaluate_test(...)` → print metrics + samples.
8. Run inference on a few unaccented example sentences via `restore_accents`.

> By default `main.py` runs in **evaluate-and-demo** mode (training is commented). To retrain, uncomment the `train_model` block.

---

## 7. Key Techniques (for the report)

1. **Problem reformulation** from seq2seq to char-level sequence labeling → same length, no decoder, parallel inference.
2. **Word-level context**: char embedding + word embedding concat → the model knows which word each char belongs to.
3. **Constraint mask** (`allowed_mask`): blocks non-linguistic predictions, reducing the label space to valid accent variants.
4. **Weighted loss**: 10× weight on accent positions → handles copy/accent imbalance.
5. **Data augmentation** via sentence-chunks (preserves real local context) + word pairs (optional).
6. **Pre-Norm Transformer** + AdamW + ReduceLROnPlateau + grad clip → stable training.
7. **Safe inference**: preserves out-of-vocab / `<unk>` / `<pad>` characters.
8. **Self-contained checkpoint**: stores vocab + config → reloads without needing original files.

---

## 8. Hyperparameter Summary (quick reference)

| Component | Value |
|-----------|-------|
| d_model / char_emb / word_emb | 192 / 128 / 64 |
| Heads / Layers / FFN / Dropout | 4 / 4 / 768 / 0.2 |
| Max len / Batch size | 256 / 64 |
| Epochs / LR / WD / Grad clip | 80 / 1e-4 / 1e-4 / 1.0 |
| Scheduler | ReduceLROnPlateau(factor=0.5, patience=4) |
| Train/valid/test ratio | 0.8 / 0.1 / 0.1 |
| Accent / Copy loss weight | 10.0 / 1.0 |
| Augment | sentence_chunks (5–24 words, stride 2), max 300k samples |

---

## 9. How to Run

```bash
# Requirements: torch, pandas, Levenshtein, (cuda optional)
# Ensure paths in config.py point correctly to viet_khong_dau.csv and words.txt

# Evaluate + demo (default):
python main.py

# To retrain: uncomment the train_model block in main.py and run again
```

---

## 10. Limitations & Future Work (suggested for the report)

- **Long context**: `max_len=256` truncates long sentences; could chunk during inference or use a recurrent model.
- **Rare-word OOV**: word vocab capped at 50k → rare words fall to `<unk_word>`, reducing context.
- **Segmented inference**: `restore_accents` handles one sentence/chunk; extend to auto-chunk long documents.
- **Data dependency**: relies on the quality of `viet_khong_dau.csv`; could supplement with real news text.
- **Extensions**: combine with spelling correction, or jointly predict accents + word segmentation.
