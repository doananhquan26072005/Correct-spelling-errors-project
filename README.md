# Vietnamese Text Correction System

An integrated NLP pipeline for **Vietnamese** text that combines two complementary tasks
into a single end-to-end system:

1. **Diacritic Restoration** — re-adding tone/accent marks to unaccented Vietnamese
   text (e.g. `toi ten la` → `tôi tên là`), using a context-aware Transformer.
2. **Spell Correction** — detecting and correcting typos, telex/key-adjacency errors
   and teen-code abbreviations (e.g. `khum bjet` → `không biết`), using a candidate
   generation + learning-to-rank approach.

The two subsystems are wired together in `main.py` as a **joint pipeline**: unaccented
input is first restored with diacritics, then cleaned by the spell-correction ranker.


---

## Table of Contents

- [Vietnamese Text Correction System](#vietnamese-text-correction-system)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Architecture](#architecture)
    - [Diacritic Restoration](#diacritic-restoration)
    - [Spell Correction](#spell-correction)
    - [Joint Pipeline](#joint-pipeline)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Run the integrated joint pipeline](#run-the-integrated-joint-pipeline)
    - [Train / evaluate each subsystem independently](#train--evaluate-each-subsystem-independently)
  - [Data \& Pretrained Resources](#data--pretrained-resources)
  - [Logging](#logging)

---

## Overview

Real-world Vietnamese user text is frequently written **without diacritics** and riddled
with **typos and slang ("teen code")**. This project tackles both problems:

- A **Transformer encoder** operates at the character level with word-level context to
  predict the correct accented character at each position (sequence labeling).
- A **candidate-generation + LightGBM ranker** pipeline detects suspect tokens,
  generates dictionary candidates (edit-distance + telex-aware), extracts rich features
  (KenLM n-gram scores, SkipGram embedding similarity, n-gram counts, edit distance,
  length ratio), and ranks the best correction.

The two are combined so that the restorer and the corrector reinforce each other on
noisy input.

---

## Project Structure

```
colab/
├── main.py                      # Integrated joint pipeline (diacritic + spell)
├── requirements.txt
├── configs/
│   ├── correction_config.yaml   # Spell-correction hyperparameters & paths
│   └── diacritic_config.yaml    # Diacritic-restoration model/training config
├── common/
│   ├── config.py                # YAML → SimpleNamespace loader
│   └── logger.py                # File + console logger
├── diacritic_restoration/       # Transformer-based accent tagger
│   ├── model.py                 # ContextAwareAccentTagger (Transformer encoder)
│   ├── dataset.py               # AccentContextDataset + DataLoader factory
│   ├── vocab.py                 # CharVocab / WordVocab + allowed-token mask
│   ├── processor.py             # DiacriticDataProcessor / DiacriticRestorer
│   ├── trainer.py               # DiacriticTrainer
│   ├── metrics.py               # word_accuracy / accent_only_accuracy / text metrics
│   └── utils.py                 # accent removal, normalization, logit constraints
├── spell_correction/            # Candidate-generation + learning-to-rank
│   ├── abbr_processor.py        # Teen-code / abbreviation expansion
│   ├── candidate_generator.py   # Dictionary lookup + telex-aware edit distance
│   ├── feature_extractor.py     # KenLM, n-gram, embedding-similarity features
│   ├── skipgram_trainer.py       # SkipGram word embeddings (PyTorch)
│   ├── lightgbm_trainer.py      # LightGBM LambdaMART ranker training
│   ├── pipeline.py              # SpellCorrectionPipeline (detect → rank → correct)
│   ├── evaluator.py             # Error detection & ranking/word/end-to-end metrics
│   ├── dataset.py               # HuggingFace dataset mapping + train/val/test split
│   └── visualizer.py            # Prediction analysis & sampling
├── scripts/
│   ├── diacritic_train.py       # Standalone diacritic training/eval entry point
│   └── correction_train.py      # Standalone spell-correction training/eval entry point
├── data/
│   ├── external/                # vocab, stopwords, telex & teen-code dictionaries
│   ├── raw/                     # viet_khong_dau.csv (accent-removed corpus)
│   └── processed/               # cached LightGBM training matrix (.npz)
├── models/                      # Trained checkpoints (trigram LM, accent tagger)
└── logs/                        # Daily run logs
```

---

## Architecture

### Diacritic Restoration

A **context-aware character-level sequence labeler** (`ContextAwareAccentTagger`):

- **Inputs:** character IDs and word IDs per position. Char embeddings (128-d) and word
  embeddings (64-d) are concatenated to form a 192-d (`d_model`) representation, giving
  the model both fine-grained character signal and word-level context.
- **Encoder:** standard Transformer encoder with sinusoidal positional encoding,
  pre-norm layers, 4 heads, 4 layers, feed-forward dim 768, dropout 0.2.
- **Output:** per-character logits over the character vocabulary.
- **Constraints:** an allowed-token mask restricts predictions to legal
  character transformations, and a weighted multi-component loss
  (`copy_loss_weight`, `accent_loss_weight`) emphasizes accent prediction.

Training data is augmented from word-pairs and sentence chunks of an
accent-removed corpus (`data/raw/viet_khong_dau.csv`).

### Spell Correction

A **detect → generate → rank** pipeline:

1. **Abbreviation expansion** — teen-code / slang tokens are normalized first
   (`abbr_processor.py`, driven by `data/external/teen_code.txt`).
2. **Error detection** — KenLM trigram perplexity heuristics flag suspect tokens
   (configurable `alpha`, `hard_ceiling`, `hard_floor` thresholds).
3. **Candidate generation** — dictionary lookup via telex-aware edit distance
   (substitution/adjacency/transposition costs), with n-gram frequency fitting.
4. **Feature extraction** — for each candidate:
   KenLM score, SkipGram embedding similarity (distance-weighted context),
   unigram/bigram/trigram counts, edit distance, and length ratio.
5. **Ranking** — a **LightGBM LambdaMART ranker** picks the best candidate per
   error token; the highest-scoring candidate replaces the typo.

SkipGram embeddings (300-d, window 3) are trained from scratch in PyTorch and used
both for similarity features and as a downloadable pretrained checkpoint.

### Joint Pipeline

`main.py` chains the two subsystems on the global test split:

- If the input **already contains diacritics**, spell correction runs directly.
- Otherwise, the **diacritic restorer** runs first, then the **spell corrector**
  cleans the restored text.

End-to-end evaluation (word accuracy + exact-match) and prediction visualization are
run on the joint output.

---

## Installation

> **Note on `kenlm`:** building `kenlm` on Windows requires a C++ toolchain. It is
> easiest to run this project on Linux/WSL or in Google Colab (the project's original
> environment, hence the folder name).

```bash
git clone <repo-url>
cd colab
python -m venv .venv
source .venv/bin/activate        # Windows (WSL/Linux recommended): .venv\Scripts\activate

sudo apt-get update && sudo apt-get install -y build-essential cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev

pip install -r requirements.txt
pip install [https://github.com/kpu/kenlm/archive/master.zip](https://github.com/kpu/kenlm/archive/master.zip)

git clone [https://github.com/kpu/kenlm.git](https://github.com/kpu/kenlm.git)
mkdir -p kenlm/build && cd kenlm/build
cmake ..
make -j 4
cd ../..
```

`requirements.txt` covers PyTorch, LightGBM, HuggingFace `datasets`, `gdown`,
`python-Levenshtein`, `kenlm`, and supporting libraries.

---

## Configuration

All runtime settings live in YAML files under `configs/`, loaded into a nested
`SimpleNamespace` by `common/config.py`.

- **`diacritic_config.yaml`** — data paths/splits, vocabulary limits, augmentation
  strategy, Transformer architecture (`d_model`, `nhead`, layers, dropout), and
  training hyperparameters (batch size, epochs, LR, weight decay, grad clip,
  checkpoint path). `device: auto` resolves to CUDA when available.
- **`correction_config.yaml`** — resource paths (vocab, stopwords, telex/teen-code,
  trigram LM, SkipGram checkpoint), model dims, SkipGram training, LightGBM
  ranker params, error-detection heuristics, candidate-generation costs, feature
  normalization ceilings, and the linear feature-combination weights.

Edit these files rather than touching source code to tune behavior.

---

## Usage

### Run the integrated joint pipeline

```bash
python main.py
```

Loads both configs, builds resources, trains/loads the LightGBM ranker, loads the
diacritic checkpoint, and evaluates the joint diacritic + spell-correction pipeline
on the global test split.

### Train / evaluate each subsystem independently

```bash
# Diacritic restoration: train the Transformer and run inference demos
python scripts/diacritic_train.py

# Spell correction: train SkipGram + LightGBM ranker and evaluate
python scripts/correction_train.py
```

Each script is self-contained: it loads its config, prepares data, trains (or loads)
the model, and reports metrics through the shared `Evaluator` / `Visualizer`.

---

## Data & Pretrained Resources

- **Corpus:** `yammdd/vietnamese-error-correction-corpus` (loaded via HuggingFace
  `datasets`), split into `df1` (spell correction) and `df2` (diacritic restoration).
- **Accent-removed corpus:** `data/raw/viet_khong_dau.csv` for diacritic training.
- **Dictionaries (`data/external/`):** `vocabulary.txt`, `words.txt`,
  `vietnamese-stopwords.txt`, `telex.txt` (telex rules), `teen_code.txt` (slang).
- **Pretrained artifacts (`models/`):**
  - `trigram.bin` — KenLM trigram language model.
  - `accent_tagger_context_v3.pt` — trained diacritic Transformer checkpoint.
  - SkipGram weights are downloaded automatically via `gdown` from the Google Drive
    ID in `correction_config.yaml` (`paths.skipgram_model_url`).
- **Cached features:** `data/processed/dataset_spell_correction.npz` — precomputed
  LightGBM training matrix.

---

## Logging

`common/logger.py` writes to `logs/spelling_checker_YYYYMMDD.log` (overwritten per
run) and mirrors to stdout. Use `get_logger("ModuleName")` in any module.
