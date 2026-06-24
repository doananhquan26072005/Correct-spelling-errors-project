from datasets import load_dataset
import re
import pandas as pd

# Hàm để xử lý dataset
def process_dataset(examples, word_to_idx):
    inputs = []
    targets = []

    for inp, tgt in zip(examples['input'], examples['target']):
        # Viết thường
        inp_str = str(inp).lower()
        tgt_str = str(tgt).lower()

        # Xóa dấu câu và xóa chữ số
        inp_clean = re.sub(r'[^\w\s_]|\d+', '', inp_str)
        tgt_clean = re.sub(r'[^\w\s_]|\d+', '', tgt_str)

        inp_tokens = inp_clean.split()
        tgt_tokens = tgt_clean.split()

        # Kiểm tra điều kiện độ dài: Chỉ giữ lại nếu bằng nhau
        if len(inp_tokens) != len(tgt_tokens) or len(inp_tokens) == 0:
          continue

        # Thay thế các từ không phải tiếng Việt mà bị lỗi
        new_inp_tokens = []
        for inp_word, tgt_word in zip(inp_tokens, tgt_tokens):
            if tgt_word not in word_to_idx:
                new_inp_tokens.append(tgt_word)
            else:
                new_inp_tokens.append(inp_word)

        inputs.append(" ".join(new_inp_tokens))
        targets.append(" ".join(tgt_tokens))

    return {"input": inputs, "target": targets}

# Hàm để chia dataset ra 2 phần: lỗi chính tả/viết tắt và không dấu
def split_data(dataset):
    VIETNAMESE_DIACRITICS = re.compile(r'[àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]')
    # Lỗi chính tả
    def error_1(example):
        inp = str(example['input'])

        return bool(VIETNAMESE_DIACRITICS.search(inp))

    # Lỗi không dấu
    def error_2(example):
        inp = str(example['input'])

        return not bool(VIETNAMESE_DIACRITICS.search(inp))

    # Lọc song song trên cả 3 tập (train, validation, test) của DatasetDict lỗi
    df_error1 = dataset.filter(error_1)
    df_error2 = dataset.filter(error_2)
    
    return df_error1, df_error2