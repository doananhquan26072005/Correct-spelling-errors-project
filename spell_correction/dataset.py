import json
import os
import re
import numpy as np
import torch

from common.logger import get_logger
from spell_correction.skipgram_trainer import SkipGram

logger = get_logger(__name__)


class ResourceLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        logger.info("ResourceLoader initialized.")

    def load_vocab_and_dicts(self):
        logger.info("ResourceLoader: Loading dictionaries and telex configurations...")

        try:
            # 1. Nạp Vocabulary
            with open(self.cfg.paths.vocab_file, 'r', encoding='utf-8') as f:
                vocab = list(dict.fromkeys(f.read().splitlines()))
            word_to_idx = {word: i for i, word in enumerate(vocab)}
            logger.debug(f"Loaded vocab file. Unique words: {len(vocab):,}")

            # 2. Nạp Stopwords
            with open(self.cfg.paths.stopwords_file, 'r', encoding='utf-8') as f:
                stopwords = set(f.read().splitlines())
            logger.debug(f"Loaded stopwords file. Total stopwords: {len(stopwords):,}")

            # 3. Nạp và sửa lỗi file Telex JSON
            with open(self.cfg.paths.telex_file, "r", encoding="utf-8") as f:
                telex_raw = f.read()
                telex_raw = re.sub(r',\s*}', '\n}', telex_raw)
                telex_dict = json.loads(telex_raw)
            logger.debug(f"Loaded and parsed telex dictionary rules: {len(telex_dict):,}")

            logger.info(f"All dictionary assets loaded successfully.")

            return {
                "vocab": vocab,
                "word_to_idx": word_to_idx,
                "stopwords": stopwords,
                "telex_dict": telex_dict
            }
        except FileNotFoundError as e:
            logger.error(f"Critical configuration file missing during dict initialization.", exc_info=True)
            raise e
        except json.JSONDecodeError as e:
            logger.error(f"Telex mapping file contains malformed JSON structure.", exc_info=True)
            raise e


def process_dataset(examples, word_to_idx):
    inputs = []
    targets = []
    skipped_count = 0
    raw_samples = len(examples['input'])

    for inp, tgt in zip(examples['input'], examples['target']):
        inp_str = str(inp).lower()
        tgt_str = str(tgt).lower()

        # Xóa dấu câu và xóa chữ số
        inp_clean = re.sub(r'[^\w\s_]|\d+', '', inp_str)
        tgt_clean = re.sub(r'[^\w\s_]|\d+', '', tgt_str)

        inp_tokens = inp_clean.split()
        tgt_tokens = tgt_clean.split()

        # Kiểm tra điều kiện độ dài: Chỉ giữ lại nếu bằng nhau và không rỗng
        if len(inp_tokens) != len(tgt_tokens) or len(inp_tokens) == 0:
            skipped_count += 1
            continue

        # Thay thế các từ không phải tiếng Việt hoặc OOV nằm bên tập target
        new_inp_tokens = []
        for inp_word, tgt_word in zip(inp_tokens, tgt_tokens):
            if tgt_word not in word_to_idx:
                new_inp_tokens.append(tgt_word)
            else:
                new_inp_tokens.append(inp_word)

        inputs.append(" ".join(new_inp_tokens))
        targets.append(" ".join(tgt_tokens))

    # Log DEBUG để theo dõi hiệu quả làm sạch data mà không làm rác terminal khi chạy nhiều batch map()
    logger.debug(
        f"Processed batch size: {raw_samples} | Retained: {len(inputs)} | Skipped (Length mismatch/Empty): {skipped_count}"
    )

    return {"input": inputs, "target": targets}


def split_data(dataset):
    """
    Phân tách DatasetDict thành 2 luồng lỗi chuyên biệt:
    - df_error1: Lỗi chính tả / Teencode / Viết tắt (Câu Input vẫn chứa ký tự dấu tiếng Việt).
    - df_error2: Lỗi gõ tiếng Việt không dấu (Câu Input mất hoàn toàn ký tự dấu).
    """
    logger.info("Splitting dataset into two distinct error pipelines (Typo vs. Unmarked)...")
    
    VIETNAMESE_DIACRITICS = re.compile(
        r'[àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]'
    )

    def error_1(example):
        inp = str(example['input'])
        return bool(VIETNAMESE_DIACRITICS.search(inp))

    def error_2(example):
        inp = str(example['input'])
        return not bool(VIETNAMESE_DIACRITICS.search(inp))

    # Lọc song song trực tiếp trên toàn bộ các split (train, validation, test)
    df_error1 = dataset.filter(error_1)
    df_error2 = dataset.filter(error_2)
    
    # Đo và ghi nhận tỷ lệ phân chia để làm báo cáo khoa học đồ án
    for split_name in dataset.keys():
        total_len = len(dataset[split_name])
        len_e1 = len(df_error1[split_name])
        len_e2 = len(df_error2[split_name])
        
        logger.info(f"Split results for [{split_name}]:")
        logger.info(f"  > Total samples: {total_len:,}")
        logger.info(f"  > Error 1 (Typo/Teencode) : {len_e1:,} ({len_e1/max(1, total_len)*100:.2f}%)")
        logger.info(f"  > Error 2 (Unmarked Text) : {len_e2:,} ({len_e2/max(1, total_len)*100:.2f}%)")
        
    return df_error1, df_error2