import os
import sys
import subprocess
import kenlm
from common.logger import get_logger

logger = get_logger(__name__)

ROOT_DIR = os.getcwd()

KENLM_DIR = os.path.join(ROOT_DIR, "kenlm")
BUILD_DIR = os.path.join(KENLM_DIR, "build")
LMPLZ_EXEC = os.path.join(BUILD_DIR, "bin", "lmplz")
BUILD_BINARY_EXEC = os.path.join(BUILD_DIR, "bin", "build_binary")

CORPUS_PATH = os.path.join(ROOT_DIR, "data", "processed", "corpus.txt")

MODELS_DIR = os.path.join(ROOT_DIR, "models")
ARPA_OUTPUT_PATH = os.path.join(MODELS_DIR, "model.arpa")
BINARY_OUTPUT_PATH = os.path.join(MODELS_DIR, "trigram.bin")

NGRAM_ORDER = 3

def run_command(command, description=""):
    if description:
        logger.info(f"[PROCESS] {description}")

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        cleaned_line = line.strip()
        if cleaned_line:
            logger.info(cleaned_line)

    process.wait()

    if process.returncode != 0:
        error_msg = f"Command failed (exit code {process.returncode}): {command}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def prepare_environment():
    logger.info("Checking for KenLM binaries at root...")
    
    if os.path.exists(LMPLZ_EXEC) and os.path.exists(BUILD_BINARY_EXEC):
        logger.info("KenLM binaries found. Skipping C++ compilation.")
        return

    logger.info("KenLM binaries missing. Starting environment setup and compilation...")
    
    run_command(
        "sudo apt-get update && sudo apt-get install build-essential cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev -y",
        description="Installing build-essential, cmake, boost, and compression libraries"
    )

    if not os.path.exists(KENLM_DIR):
        run_command(
            f"git clone https://github.com/kpu/kenlm.git {KENLM_DIR}",
            description="Cloning KenLM repository to root directory"
        )
    else:
        logger.info(f"KenLM directory already exists at {KENLM_DIR}. Skipping clone.")

    os.makedirs(BUILD_DIR, exist_ok=True)
    run_command(
        f"cd {BUILD_DIR} && cmake .. && make -j4",
        description="Configuring CMake and compiling KenLM source with 4 threads"
    )


def main():
    logger.info("--- STARTING N-GRAM TRAINING PIPELINE ---")

    prepare_environment()

    if not os.path.exists(LMPLZ_EXEC) or not os.path.exists(BUILD_BINARY_EXEC):
        logger.error("Failed to verify KenLM binaries. Execution aborted.")
        sys.exit(1)

    if not os.path.exists(CORPUS_PATH):
        logger.error(f"Input corpus file not found at: {CORPUS_PATH}")
        logger.warning("Please ensure 'corpus.txt' is placed inside 'data/processed/' before running.")
        sys.exit(1)

    os.makedirs(MODELS_DIR, exist_ok=True)

    lmplz_cmd = f"{LMPLZ_EXEC} -o {NGRAM_ORDER} < {CORPUS_PATH} > {ARPA_OUTPUT_PATH}"
    run_command(
        lmplz_cmd, 
        description=f"Running lmplz to train {NGRAM_ORDER}-gram model"
    )

    build_binary_cmd = f"{BUILD_BINARY_EXEC} {ARPA_OUTPUT_PATH} {BINARY_OUTPUT_PATH}"
    run_command(
        build_binary_cmd, 
        description="Converting ARPA format to optimized Binary format (.bin)"
    )
    
    if os.path.exists(ARPA_OUTPUT_PATH):
        os.remove(ARPA_OUTPUT_PATH)
        logger.info("Removed intermediate ARPA file to keep 'models/' directory clean.")

    logger.info("--- TRAINING PIPELINE COMPLETED SUCCESSFULLY ---")


if __name__ == "__main__":
    main()