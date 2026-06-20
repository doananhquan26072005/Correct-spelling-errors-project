# process_dataset.py
import re
from datasets import DatasetDict

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