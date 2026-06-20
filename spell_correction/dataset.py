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
import torch
import torch.nn as nn

class SkipGram(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):

        embed = self.embedding(x)
        out = self.linear(embed)

        return out

class ResourceLoader:
    def __init__(self, cfg):
        """
        Khởi tạo bộ nạp tài nguyên từ cấu hình hệ thống.
        :param cfg: Cấu hình động (SimpleNamespace hoặc Dict)
        """
        self.cfg = cfg

    def _download_from_gdrive(self, file_id, output_path):
        """Hàm hỗ trợ tải file weights lớn (.pth) từ Google Drive bằng gdown."""
        print(f"[!] Không tìm thấy file tại {output_path}. Đang tiến hành tải từ Google Drive...")
        url = f'https://drive.google.com/uc?id={file_id}'
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

    def create_norm_embedding_matrix(self, vocab_size):
        """
        Tải mô hình PyTorch, trích xuất ma trận nhúng và tính toán chuẩn hóa động.
        """
        model_path = self.cfg.paths.skipgram_model_file
        device = self.cfg.DEVICE
        
        # 1. Tự động tải từ GDrive nếu file .pth chưa có ở local
        if not os.path.exists(model_path):
            gdrive_id = getattr(self.cfg.paths, "skipgram_gdrive_id", None)
            if not gdrive_id:
                raise ValueError(f"Không thấy file {model_path} cục bộ và thiếu 'skipgram_gdrive_id' trong config!")
            self._download_from_gdrive(gdrive_id, model_path)

        print("[*] Lớp ResourceLoader: Đang khởi tạo và nạp trọng số mô hình SkipGram...")
        
        # 2. Lấy cấu hình chiều Vector nhúng (Mặc định là 128 nếu config không có)
        embed_dim = self.cfg.model.embed_dim if hasattr(self.cfg, 'model') else 128
        
        # 3. Khởi tạo mô hình và nạp State Dict ( weights_only=True tuân thủ bảo mật Pytorch )
        # Lưu ý: Hãy định nghĩa hoặc import class SkipGram của bạn ở đầu file
        model_skipgram = SkipGram(vocab_size, embed_dim).to(device)
        model_skipgram.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        
        # 4. Trích xuất ma trận nhúng sang Numpy
        embeddings = model_skipgram.embedding.weight.data
        embedding_matrix = embeddings.cpu().numpy()

        # 5. Tiến hành chuẩn hóa L2-norm động để tạo ra norm_embedding_matrix
        print("[*] Lớp ResourceLoader: Đang tính toán chuẩn hóa L2 cho Embedding Matrix...")
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        norm_embedding_matrix = embedding_matrix / norms
        
        return norm_embedding_matrix

    def load_all(self):
        """Hàm điều phối: Load dicts trước -> lấy Vocab Size -> Khởi tạo Matrix sau."""
        resources = self.load_vocab_and_dicts()
        
        # Lấy kích thước vocab động từ file vừa nạp để truyền vào PyTorch model
        vocab_size = len(resources["vocab"])
        
        # Khởi tạo ma trận nhúng dựa trên vocab_size thực tế
        resources["norm_embedding_matrix"] = self.create_norm_embedding_matrix(vocab_size)
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