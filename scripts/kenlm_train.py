import os
import sys
import subprocess
import kenlm
from common.logger import get_logger

logger = get_logger(__name__)

CURRENT_DIR = os.getcwd()
KENLM_DIR = os.path.join(CURRENT_DIR, "kenlm")
BUILD_DIR = os.path.join(KENLM_DIR, "build")

LMPLZ_EXEC = os.path.join(BUILD_DIR, "bin", "lmplz")
BUILD_BINARY_EXEC = os.path.join(BUILD_DIR, "bin", "build_binary")

CORPUS_PATH = os.path.join(CURRENT_DIR, "corpus.txt")
ARPA_OUTPUT_PATH = os.path.join(CURRENT_DIR, "model.arpa")
BINARY_OUTPUT_PATH = os.path.join(CURRENT_DIR, "model_ken_lm.bin")

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


def main():
    logger.info("--- STARTING N-GRAM TRAINING PIPELINE ---")

    if not os.path.exists(LMPLZ_EXEC) or not os.path.exists(BUILD_BINARY_EXEC):
        logger.error(
            f"KenLM binaries not found at {BUILD_DIR}/bin/. "
            "Please ensure KenLM source is built before running."
        )
        sys.exit(1)

    if not os.path.exists(CORPUS_PATH):
        logger.error(f"Input corpus file not found at: {CORPUS_PATH}")
        logger.warning("Please prepare 'corpus.txt' before starting the training.")
        sys.exit(1)

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
    
    logger.info("--- TRAINING PIPELINE COMPLETED SUCCESSFULLY ---")


if __name__ == "__main__":
    main()