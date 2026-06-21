# Vietnamese Correct Spelling Errors

A Vietnamese NLP project for automatic text correction, including:

* **Diacritic Restoration** (khôi phục dấu tiếng Việt)
* **Spelling Correction** (sửa lỗi chính tả)
* End-to-end correction pipeline

The project combines statistical language models and deep learning models to improve Vietnamese text quality.

---

## Features

* Vietnamese diacritic restoration
* Vietnamese spelling correction
* End-to-end correction pipeline
* Configurable through YAML files
* Training and evaluation scripts
* Logging support
* Modular architecture for future extensions

---

## Project Structure

```text
project/
│
├── common/
│   ├── config.py
│   ├── logger.py
│   └── utils.py
│
├── configs/
│   ├── correction_config.yaml
│   └── diacritic_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── models/
│
├── notebooks/
│
├── spell_correction/
│   ├── processor.py
│   ├── process_dataset.py
│   ├── trainer.py
│   └── evaluator.py
│
├── correction_train.py
├── diacritic_train.py
├── main.py
└── requirements.txt
```

---

## System Overview

```text
                Raw Vietnamese Text
                        │
                        ▼
             Text Preprocessing
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
 Diacritic Restoration          Spelling Correction
        │                               │
        └───────────────┬───────────────┘
                        ▼
             Corrected Vietnamese Text
```

---

## Technologies

* Python
* PyTorch
* KenLM
* HuggingFace Datasets
* NumPy
* Pandas
* PyYAML
* gdown

---

## Installation

Clone the repository

```bash
git clone https://github.com/doananhquan26072005/Correct-spelling-errors-project.git
cd Correct-spelling-errors-project
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

The project uses Vietnamese datasets for:

* Diacritic restoration
* Spelling correction
* Vocabulary
* Teen code dictionary
* Telex typing rules
* Vietnamese stopwords

External resources are stored in

```text
data/external/
```

Processed datasets are stored in

```text
data/processed/
```

---

## Configuration

All experiment settings are managed by YAML configuration files.

### Diacritic Restoration

```text
configs/diacritic_config.yaml
```

### Spelling Correction

```text
configs/correction_config.yaml
```

Configurations include:

* dataset paths
* model parameters
* optimizer settings
* training hyperparameters
* checkpoint paths
* logging options

---

## Training

### Train Diacritic Restoration

```bash
python diacritic_train.py
```

### Train Spelling Correction

```bash
python correction_train.py
```

---

## Inference

Run the complete correction pipeline

```bash
python main.py
```

The pipeline performs:

1. Load configuration
2. Load language model
3. Restore Vietnamese diacritics
4. Correct spelling errors
5. Evaluate results

---

## Models

Current project contains pretrained models inside

```text
models/
```

Additional models can be downloaded automatically when required.

---

## Logging

Logging utilities are implemented under

```text
common/logger.py
```

Logs include

* training progress
* evaluation metrics
* pipeline execution
* error tracking

---

## Evaluation

The evaluation module supports:

* model evaluation
* pipeline evaluation
* end-to-end testing

Implemented in

```text
spell_correction/evaluator.py
```

---

## Future Improvements

* Beam search decoding
* Transformer-based language model
* Better candidate generation
* Real-word error correction
* Web API deployment
* Streamlit interface
* Docker support

---

## Requirements

Main dependencies include

* Python 3.10+
* PyTorch
* KenLM
* datasets
* PyYAML
* tqdm
* NumPy
* Pandas
* python-Levenshtein

Install all dependencies using

```bash
pip install -r requirements.txt
```

---

## Authors

Developed as a Vietnamese Natural Language Processing project focusing on automatic spelling correction and diacritic restoration.
