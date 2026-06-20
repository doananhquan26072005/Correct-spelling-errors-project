# process_dataset.py
import re
from datasets import DatasetDict

import os
import json
import numpy as np

# -*- coding: utf-8 -*-
import os
import re
import json
import numpy as np
import gdown

class ResourceLoader:
    def __init__(self, cfg):
        """
        Khởi tạo bộ nạp tài nguyên từ cấu hình hệ thống.
        :param cfg: Cấu hình động (SimpleNamespace hoặc Dict)
        """
        self.cfg = cfg

    def _download_from_gdrive(self, file_id, output_path):
        """Hàm hỗ trợ tải file lớn từ Google Drive bằng gdown."""
        print(f"[!] Không tìm thấy file tại {output_path}. Đang tiến hành tải từ Google Drive...")
        url = f'https://drive.google.com/uc?id={file_id}'
        
        # Tạo thư mục cha nếu chưa tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Tải file
        gdown.download(url, output_path, quiet=False)
        print(f"[+] Tải thành công và lưu tại: {output_path}")

    def load_vocab_and_dicts(self):
        """Nạp từ điển vocab, stopwords và xử lý sửa lỗi cú pháp file telex."""
        print("[*] Lớp ResourceLoader: Đang nạp từ điển và telex...")
        
        with open(self.cfg.paths.vocab_file, 'r', encoding='utf-8') as f:
            vocab = list(dict.fromkeys(f.read().splitlines()))
        word_to_idx = {word: i for i, word in enumerate(vocab)}

        with open(self.cfg.paths.stopwords_file, 'r', encoding='utf-8') as f:
            stopwords = set(f.read().splitlines())

        with open(self.cfg.paths.telex_file, "r", encoding="utf-8") as f:
            telex_raw = f.read()
            telex_raw = re.sub(r',\s*}', '\n}', telex_raw)
            telex_dict = json.loads(telex_raw)

        return {
            "vocab": vocab,
            "word_to_idx": word_to_idx,
            "stopwords": stopwords,
            "telex_dict": telex_dict
        }

    def load_embeddings(self):
        """Kiểm tra, tải (nếu thiếu) và nạp ma trận nhúng Skip-gram từ file .npz."""
        # Xác định đường dẫn file .npz thực tế
        npz_path = self.cfg.paths.skipgram_model_file.replace(".pth", "_matrix.npz")
        
        # Kiểm tra xem file đã tồn tại cục bộ chưa
        if not os.path.exists(npz_path):
            # Lấy GDrive File ID từ config. Nếu không có trong config, bạn có thể truyền cứng ở đây
            gdrive_id = getattr(self.cfg.paths, "skipgram_gdrive_id", None)
            
            if not gdrive_id:
                raise ValueError(
                    f"Không tìm thấy file {npz_path} cục bộ VÀ không tìm thấy cấu hình "
                    f"'skipgram_gdrive_id' trong file cấu hình YAML để tải về!"
                )
            
            # Tiến hành tải file tự động
            self._download_from_gdrive(gdrive_id, npz_path)
            
        print("[*] Lớp ResourceLoader: Đang nạp ma trận nhúng Skip-gram vào bộ nhớ...")
        matrix_data = np.load(npz_path, allow_pickle=True)
        return matrix_data['norm_embedding_matrix']

    def load_all(self):
        """Hàm tổng hợp để nạp toàn bộ tài nguyên."""
        resources = self.load_vocab_and_dicts()
        resources["norm_embedding_matrix"] = self.load_embeddings()
        return resources

def process_dataset(examples, word_to_idx):
    """
    Tiền xử lý dữ liệu: chuyển chữ thường, xóa số, xóa dấu câu, 
    bảo vệ cấu trúc index câu và đồng bộ hóa các từ OOV của nhãn Target.
    """
    inputs = []
    targets = []

    for inp, tgt in zip(examples['input'], examples['target']):
        # Chuyển về chữ thường
        inp_str = str(inp).lower()
        tgt_str = str(tgt).lower()

        # Xóa dấu câu và xóa chữ số
        inp_clean = re.sub(r'[^\w\s_]|\d+', '', inp_str)
        tgt_clean = re.sub(r'[^\w\s_]|\d+', '', tgt_str)

        inp_tokens = inp_clean.split()
        tgt_tokens = tgt_clean.split()

        # Kiểm tra điều kiện độ dài: Chỉ giữ lại nếu bằng nhau và không rỗng
        if len(inp_tokens) != len(tgt_tokens) or len(inp_tokens) == 0:
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

    return {"input": inputs, "target": targets}


def split_data(dataset):
    """
    Phân tách DatasetDict thành 2 luồng lỗi chuyên biệt:
    - df_error1: Lỗi chính tả / Teencode / Viết tắt (Câu Input vẫn chứa ký tự dấu tiếng Việt).
    - df_error2: Lỗi gõ tiếng Việt không dấu (Câu Input mất hoàn toàn ký tự dấu).
    """
    VIETNAMESE_DIACRITICS = re.compile(
        r'[àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]'
    )

    # Bộ lọc lỗi chính tả / viết tắt
    def error_1(example):
        inp = str(example['input'])
        return bool(VIETNAMESE_DIACRITICS.search(inp))

    # Bộ lọc lỗi mất dấu hoàn toàn
    def error_2(example):
        inp = str(example['input'])
        return not bool(VIETNAMESE_DIACRITICS.search(inp))

    # Lọc song song trực tiếp trên toàn bộ các split (train, validation, test)
    df_error1 = dataset.filter(error_1)
    df_error2 = dataset.filter(error_2)
    
    return df_error1, df_error2