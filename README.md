# Vietnamese Spelling Correction

A modular Vietnamese spelling correction system developed using Natural Language Processing (NLP) techniques. The project focuses on detecting and correcting spelling errors by combining candidate generation, feature engineering, statistical language models, and machine learning.

## Features

* Vietnamese spelling error detection and correction
* Candidate generation based on edit operations
* Abbreviation normalization
* Feature extraction for candidate ranking
* Skip-gram word embedding training
* LightGBM-based candidate ranking
* Configurable training pipeline
* Logging support
* Visualization and evaluation tools

---

## Project Structure

```text
.
├── common/
│   ├── config.py                 # Configuration loader
│   └── logger.py                 # Logging utilities
│
├── configs/
│   └── correction_config.yaml    # Training configuration
│
├── data/
│   ├── external/                 # External resources
│   ├── processed/                # Processed datasets
│   └── raw/                      # Raw datasets
│
├── models/                       # Saved models
│
├── notebooks/                    # Experiments
│
├── scripts/
│   ├── correction_train.py       # Main training script
│   └── diacritic_train.py
│
├── spell_correction/
│   ├── __init__.py
│   ├── abbr_processor.py         # Abbreviation processing
│   ├── candidate_generator.py    # Candidate generation
│   ├── dataset.py                # Dataset preparation
│   ├── evaluator.py              # Evaluation utilities
│   ├── feature_extractor.py      # Candidate features
│   ├── lightgbm_trainer.py       # LightGBM training
│   ├── pipeline.py               # End-to-end correction pipeline
│   ├── skipgram_trainer.py       # Skip-gram embedding training
│   └── visualizer.py             # Visualization
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Pipeline

The correction pipeline consists of the following stages:

```text
Input sentence
        │
        ▼
Text preprocessing
        │
        ▼
Abbreviation normalization
        │
        ▼
Candidate generation
        │
        ▼
Feature extraction
        │
        ▼
LightGBM ranking model
        │
        ▼
Corrected sentence
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/doananhquan26072005/Correct-spelling-errors-project.git
cd Correct-spelling-errors-project
```

Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Installing KenLM

This project uses **KenLM** as the statistical language model for candidate scoring.

### Windows

#### Option 1 (Recommended)

Install the Python package directly:

```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
```

If the installation fails, install a C++ compiler (Visual Studio Build Tools) and CMake first.

#### Option 2 (Build from Source)

Clone the repository:

```bash
git clone https://github.com/kpu/kenlm.git
cd kenlm
```

Create a build directory:

```bash
mkdir build
cd build
```

Configure and build:

```bash
cmake ..
cmake --build . --config Release
```

Then install the Python bindings:

```bash
cd ../python
pip install .
```

---

### Linux

Install required packages:

```bash
sudo apt update
sudo apt install build-essential cmake libboost-all-dev zlib1g-dev
```

Clone and build KenLM:

```bash
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir build
cd build
cmake ..
make -j$(nproc)
```

Install the Python bindings:

```bash
cd ../python
pip install .
```

---

### Verify Installation

Run the following command:

```python
import kenlm
print(kenlm.__version__)
```

or simply

```python
import kenlm
```

If no error is raised, KenLM has been installed successfully.

---

### Using a Language Model

Download or train a KenLM language model (e.g., `model_lm.bin`) and specify its path in `configs/correction_config.yaml`:

```yaml
language_model:
  path: models/model_lm.bin
```

The project will automatically load this model during spelling correction.


## Configuration

Training parameters are stored in

```text
configs/correction_config.yaml
```

Typical configuration options include

* dataset paths
* vocabulary paths
* training parameters
* model parameters
* logging configuration
* output directories

---

## Training

Train the correction model

```bash
python scripts/correction_train.py
```

---

## Running the Pipeline

Run the complete spelling correction pipeline

```bash
python main.py
```

---

## Models

The trained models are stored in

```text
models/
```

Depending on the configuration, the project can generate

* Skip-gram embeddings
* LightGBM ranking model
* Intermediate artifacts

---

## Logging

Logging utilities are implemented in

```text
common/logger.py
```

Training logs include

* training progress
* evaluation metrics
* warnings
* runtime information

---

## Evaluation

Evaluation utilities are provided in

```text
spell_correction/evaluator.py
```

Possible evaluation metrics include

* Accuracy
* Precision
* Recall
* F1-score
* Word-level correction accuracy

---

## Dependencies

Major libraries used in this project include

* Python 3.10+
* PyTorch
* LightGBM
* NumPy
* Pandas
* PyYAML
* tqdm
* scikit-learn

Install all required packages with

```bash
pip install -r requirements.txt
```

---

## Future Work

* Deep learning ranking models
* Context-aware candidate generation
* Transformer-based correction
* REST API deployment
* Web interface
* Docker support

---

## Authors

This project was developed as part of a Vietnamese Natural Language Processing study on automatic spelling correction.
