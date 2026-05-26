# Vietnamese Accent Restoration (Viet_khong_dau)

This project implements a **Context-Aware Accent Tagger** for Vietnamese text. It restores missing accents in unaccented Vietnamese sentences using a deep learning approach based on Transformers. 

Instead of treating the problem purely as character or sequence translation, it formulates it as a **Char-Level Sequence Labeling** task with **Word-Level Context**, leveraging a Transformer Encoder. To enhance accuracy, it also utilizes a dictionary-based output constraint that forces the model to only predict linguistically valid accented characters (e.g., the letter `d` can only become `d` or `đ`).

---

## Architecture Overview

1. **Embeddings:** Each character receives both a character embedding and an embedding of the word it belongs to.
2. **Contextualization:** These combined embeddings pass through a **Transformer Encoderlayer**.
3. **Prediction:** A final linear layer predicts the correct accented version corresponding to each input character.
4. **Constraint:** The output logits are masked so the model is heavily penalized/prevented from predicting mathematically impossible character transformations.

---

## Codebase Modules Summary

### 1. `config.py` - Configuration Management
Serves as the central configuration hub for the project using Python's `@dataclass`. 
* **Model Hyperparameters:** Sets the Transformer logic (embedding dimensions, heads, layers, dropout).
* **Training Hyperparameters:** Batch size, learning rate, learning rate schedule, epochs, loss weights, gradient clipping.
* **Data Settings:** Defines dataset paths, lengths, and handles configurations for data augmentation strategies (e.g., using sentence chunks or synthetic word pairs).

### 2. `dataset.py` - Data Loading & Preparation
Handles preparing the raw CSV files for PyTorch training.
* **`AccentContextDataset`:** A custom PyTorch `Dataset` that normalizes text, validates sequence lengths, and converts characters and words into tensor ID shapes.
* **`build_dataloaders`:** Cleans the data, applies robust data generation techniques (sentence chunks, duplicate dropping), builds the vocabulary, and splits the data into `train`, `valid`, and `test` PyTorch `DataLoader` objects.

### 3. `vocab.py` - Vocabulary & Augmentation Logic
Manages how text is tokenized into numeric IDs and handles complex data processing.
* **`CharVocab` & `WordVocab`:** Classes to build, encode, and decode character and word-level IDs, managing special tokens like `<pad>` and `<unk>`.
* **Augmentation:** Contains utilities like `build_sentence_chunks_from_targets` and `build_word_pairs_from_words_file` to artificially expand the dataset logically to make the model more robust.
* **`build_allowed_token_mask`:** A crucial function that creates a Boolean mask defining the allowed transitions for every character (e.g., `a` is allowed to become `á`, `à`, `ã`, `ạ`, `a`, but not `đ`).

### 4. `model.py` - Core Neural Network
Defines the PyTorch architecture.
* **`PositionalEncoding`:** Injects positional embeddings so the Transformer understands sequence order.
* **`ContextAwareAccentTagger`:** The main neural network. It concatenates the character embedded vectors with their parent word's embedded vectors, adds positional encoding, passes the results through a `TransformerEncoder`, and projects them to the character vocabulary size to predict the correct accent.

### 5. `training.py` - Training Loop & Custom Loss
Manages the end-to-end model training logic.
* **`weighted_accent_loss`:** Calculates Cross-Entropy loss but applies custom multipliers. Because most characters in Vietnamese words don't change form when accents are applied, this function heavily penalizes the model for missing the actual *accented* characters.
* **`train_one_epoch` & `validate`:** Handle forward passes, the logical constraint mask application, backpropagation, and clipping.
* **`train_model`:** The primary loop handling epochs, best-checkpoint saving via Validation Loss, and the `ReduceLROnPlateau` scheduler.

### 6. `inference.py` - Evaluation & Prediction
Provides tools for predicting text using a trained checkpoint.
* **`evaluate_test`:** Runs the test dataset through the model and computes overall metrics to display.
* **`restore_accents`:** A robust inference function designed for real-world usage. It handles `<unk>` tokens safely, ensures non-vocabulary characters (like punctuation or emojis) are ignored/copied exactly as they were, and scales character bounds effectively.
* **`load_checkpoint`:** Securely loads model weights and dynamically fixes wrapper prefixes (like those from `DataParallel`).

### 7. `metric.py` - Evaluation Metrics
Tracks the accuracy and error rates of the generated strings. 
* Implements the **Levenshtein Distance** algorithm from scratch.
* Computes comprehensive metrics: **Character Accuracy**, **Word Accuracy**, **Accent-Only Accuracy** (checking how accurate the model is purely at places naturally requiring accents), **Exact Match**, and **Character Error Rate (CER)**.

### 8. `utils.py` - Text Utilities
Provides universal text normalization handlers used everywhere in the project.
* **`remove_accents_char` / `remove_accents_text`:** Uses `unicodedata` NFD normalization to strip diacritics and return base alphabet shapes.
* **`normalize_text`:** Downcases and sanitizes extra whitespace.
* **`apply_constraint_to_logits`:** Uses the vocabulary mask to enforce `-1e9` penalty to illegal character predictions.

### 9. `main.py` - Execution Entry Point
The runner script that ties all components together. 
1. Initializes `Config`.
2. Generates dataloaders and vocabularies.
3. Builds the `ContextAwareAccentTagger` and allows multi-GPU `DataParallel` dispatch.
4. Triggers `train_model()`.
5. Loads the best checkpoint to run `evaluate_test()`.
6. Prints sample inferences on hardcoded missing-accent sentences to visually demonstrate the results.