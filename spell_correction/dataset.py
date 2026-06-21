import json
import os
import re
import time
import gdown
import numpy as np
import torch
import torch.nn as nn
from datasets import DatasetDict

from common.logger import get_logger
from spell_correction.models import SkipGram

logger = get_logger(__name__)


class ResourceLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        logger.info("ResourceLoader initialized.")

    def _download_from_gdrive(self, file_id, output_path):
        logger.info(f"File NOT found locally at {output_path}. Initiating download from Google Drive...")
        try:
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_path, quiet=True)
            logger.info(f"Successfully downloaded and saved asset to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to download asset from Google Drive ID: {file_id}", exc_info=True)
            raise e

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

    def create_norm_embedding_matrix(self, vocab_size):
        model_path = self.cfg.paths.skipgram_model_file
        device = self.cfg.DEVICE

        if not os.path.exists(model_path):
            gdrive_id = getattr(self.cfg.paths, "skipgram_gdrive_id", None)
            if not gdrive_id:
                logger.error(f"Local file {model_path} missing and 'skipgram_gdrive_id' undefined in config.")
                raise ValueError(f"Không thấy file {model_path} cục bộ và thiếu 'skipgram_gdrive_id' trong config!")
            self._download_from_gdrive(gdrive_id, model_path)

        logger.info(f"ResourceLoader: Initializing SkipGram model architecture onto device: {device}")
        
        try:
            embed_dim = self.cfg.model.embed_dim if hasattr(self.cfg, 'model') else 128
            
            model_skipgram = SkipGram(vocab_size, embed_dim).to(device)
            model_skipgram.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            logger.info(f"Successfully loaded SkipGram weights checkpoint from: {model_path}")
            
            embeddings = model_skipgram.embedding.weight.data
            embedding_matrix = embeddings.cpu().numpy()

            logger.info("ResourceLoader: Computing L2-normalization for Embedding Matrix...")
            norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            norm_embedding_matrix = embedding_matrix / norms
            
            logger.debug(f"Normalized Embedding Matrix shape: {norm_embedding_matrix.shape}")
            return norm_embedding_matrix

        except Exception as e:
            logger.error("Failed to extract or normalize the Embedding matrix from SkipGram model.", exc_info=True)
            raise e

    def load_all(self):
        """Hàm điều phối: Load dicts trước -> lấy Vocab Size -> Khởi tạo Matrix sau."""
        logger.info("=== STARTING FULL RESOURCE DEPLOYMENT LOOP ===")
        resources = self.load_vocab_and_dicts()
        vocab_size = len(resources["vocab"])
        
        resources["norm_embedding_matrix"] = self.create_norm_embedding_matrix(vocab_size)
        logger.info("=== RESOURCE DEPLOYMENT LOOP COMPLETED ===")
        return resources


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