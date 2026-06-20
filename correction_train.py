# -*- coding: utf-8 -*-
import os
import json
import re
import math
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
# import kenlm
from datasets import load_dataset

# Import các hàm xử lý dữ liệu ngoại vi của bạn
from spell_correction.process_dataset import process_dataset, split_data

# Import các module đã cấu trúc lại theo SimpleNamespace ở trên
from utils import load_config  # Hàm load_config của bạn
from spell_correction.processor import TeencodeProcessor, CandidateGenerator, FeatureExtractor
from spell_correction.evaluator import Evaluator
from spell_correction.trainer import LightGBMRankerTrainer

# ==========================================
# 1. KHỞI TẠO CẤU HÌNH ĐỘNG (CONFIG)
# ==========================================
cfg = load_config("configs/correction_config.yaml")

# Tạo ánh xạ paths/tham số từ config sang môi trường hiện tại
# Phục vụ trường hợp bạn muốn định nghĩa thêm DEVICE động
if not hasattr(cfg, "DEVICE"):
    cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. LOAD TÀI NGUYÊN VÀ DATASET
# ==========================================
print("[*] Đang nạp từ điển và tài nguyên hệ thống...")
with open(cfg.paths.vocab_file, 'r', encoding='utf-8') as f:
    vocab = list(dict.fromkeys(f.read().splitlines()))

word_to_idx = {word: i for i, word in enumerate(vocab)}

with open(cfg.paths.stopwords_file, 'r', encoding='utf-8') as f:
    stopwords = set(f.read().splitlines())

with open(cfg.paths.telex_file, "r", encoding="utf-8") as f:
    telex_raw = f.read()
    telex_raw = re.sub(r',\s*}', '\n}', telex_raw)
    telex_dict = json.loads(telex_raw)

# Khởi tạo Teencode Processor từ class đã gửi
teencode_engine = TeencodeProcessor(cfg.paths.teen_code_file)

# Nạp Dataset
dataset = load_dataset("yammdd/vietnamese-error-correction-corpus")

# ==========================================
# 3. TIỀN XỬ LÝ VÀ CHIA DATASET
# ==========================================
print("[*] Đang tiền xử lý Dataset...")
df = dataset.map(
    process_dataset,
    batched=True,
    remove_columns=dataset['train'].column_names,
    fn_kwargs={"word_to_idx": word_to_idx}
)

df1, df2 = split_data(df)

df_train = pd.DataFrame(df['train'])
df1_valid = pd.DataFrame(df1['validation'])

# ==========================================
# 4. KHỞI TẠO CÁC ENGINES SUY LUẬN CHÍNH
# ==========================================
print("[*] Đang nạp mô hình KenLM và khởi tạo Candidate Generator...")
model_lm = kenlm.Model(cfg.paths.trigram_lm_file)

# Khởi tạo bộ sinh ứng viên (Candidate Generator)
generator = CandidateGenerator(vocab=vocab, telex_dict=telex_dict, cfg=cfg)
print("[*] Đang thống kê tần suất N-gram từ tập huấn luyện (fit_ngram_counts)...")
generator.fit_ngram_counts(df_train['target'])

# Nạp ma trận nhúng Word Embedding Skip-gram từ file nén .npz
print("[*] Đang nạp ma trận nhúng Skip-gram đã tối ưu...")
matrix_data = np.load(cfg.paths.skipgram_model_file.replace(".pth", "_matrix.npz"), allow_pickle=True)
norm_embedding_matrix = matrix_data['norm_embedding_matrix']

# Khởi tạo bộ trích xuất đặc trưng (Feature Extractor)
extractor_engine = FeatureExtractor(
    word_to_idx=word_to_idx,
    norm_embedding_matrix=norm_embedding_matrix,
    stopwords=stopwords,
    cfg=cfg
)

# Khởi tạo cấu trúc đánh giá trung tâm (Evaluator)
evaluator = Evaluator(model_lm=model_lm, config=cfg)

# ==========================================
# 5. KHỞI TẠO VÀ TẢI MÔ HÌNH LIGHTGBM RANKER
# ==========================================
print("[*] Đang nạp mô hình phân hạng LightGBM Ranker...")
ranker_trainer = LightGBMRankerTrainer(cfg)

# Nếu bạn đã chạy file create_data_LightGBM.ipynb và có file lưu sẵn:
X_train, y_train, group_train = ranker_trainer.load_training_data(cfg.paths.lightgbm_data_path)
ranker = ranker_trainer.train(X_train, y_train, group_train)

# ==========================================
# 6. PIPELINE SỬA LỖI CHÍNH TẢ TOÀN CỤC (CORRECTION)
# ==========================================
def pipeline_correct_sentence(sentence: str) -> str:
    """Hàm Pipeline khép kín nhận diện, trích xuất đặc trưng và sửa lỗi câu hoàn chỉnh."""
    error_indices = evaluator.detect_error(sentence)
    sentence_tokens = sentence.split()

    for idx in error_indices:
        if idx >= len(sentence_tokens):
            continue
        error_word = sentence_tokens[idx]

        # Sinh ứng viên thô qua lookup (k_candidates lấy từ cấu hình động)
        k_candidates = cfg.candidate_generation.top_k_raw if hasattr(cfg, 'candidate_generation') else 2
        raw_candidates = generator.lookup(error_word, word_to_idx, k=k_candidates)

        if not raw_candidates:
            continue

        # Trích xuất vector đặc trưng động
        candidates_with_features = extractor_engine.extract_candidates_and_features(
            error_word=error_word,
            sentence_words=sentence_tokens,
            error_idx=idx,
            error_indices=error_indices,
            candidates=raw_candidates,
            model_lm=model_lm,
            generator=generator
        )

        if not candidates_with_features:
            continue

        cand_words = [item[0] for item in candidates_with_features]
        X_infer = np.array([item[1] for item in candidates_with_features])

        # Dự đoán phân hạng điểm số bằng LightGBM Ranker
        scores = ranker.predict(X_infer)
        best_candidate = cand_words[np.argmax(scores)]

        # Thay thế từ đúng ngữ cảnh vào chuỗi token phục vụ bước tiếp theo
        sentence_tokens[idx] = best_candidate

    return " ".join(sentence_tokens)

# ==========================================
# 7. CHẠY TOÀN BỘ CÁC MODULE ĐÁNH GIÁ (EVALUATION)
# ==========================================
print("\n" + "="*30 + " BẮT ĐẦU CHU TRÌNH ĐÁNH GIÁ " + "="*30)

# 7.1 Đánh giá cấu phần Detect Lỗi
evaluator.evaluate_error_detection(
    validation_df=df1_valid, 
    teencode_engine=teencode_engine
)

# 7.2 Đánh giá cấu phần Ranker ứng viên (MRR, Hit@K)
evaluator.evaluate_ranking_performance(
    validation_df=df1_valid,
    teencode_engine=teencode_engine,
    extractor_engine=extractor_engine,
    generator=generator,
    ranker=ranker
)

# 7.3 Đánh giá chất lượng sửa từ (Word Accuracy)
evaluator.evaluate_word_accuracy(
    validation_df=df1_valid,
    teencode_engine=teencode_engine,
    pipeline_correct_fn=pipeline_correct_sentence,
    word_to_idx=word_to_idx
)

# 7.4 Đánh giá hệ thống toàn cục (End-to-End WER/SER)
evaluator.evaluate_end_to_end(
    validation_df=df1_valid,
    teencode_engine=teencode_engine,
    pipeline_correct_fn=pipeline_correct_sentence
)

# ==========================================
# 8. TRỰC QUAN HÓA KẾT QUẢ SUY LUẬN THỰC TẾ
# ==========================================
exact_sentences = pd.DataFrame(columns=['Input', 'Fixed', 'Target'])
error_sentence = pd.DataFrame(columns=['Input', 'Fixed', 'Target'])
error_words = pd.DataFrame(columns=['Error', 'Correct'])

print("[*] Đang phân loại câu mẫu trực quan...")
for _, row in tqdm(df1_valid.head(200).iterrows(), total=200, desc="Phân tích trực quan mẫu"): 
    input_sent = str(row['input'])
    target_sent = str(row['target'])

    # Tiền xử lý teencode
    cleaned_input = teencode_engine.replace_abbreviations(input_sent)

    # Quét nhãn gốc
    _, error_indices = evaluator.find_misspelled_words_and_targets(cleaned_input, target_sent, word_to_idx)

    # Chạy qua Pipeline chính
    fixed_sentence_str = pipeline_correct_sentence(cleaned_input)

    target_tokens = target_sent.split()
    fixed_tokens = fixed_sentence_str.split()

    if len(target_tokens) != len(fixed_tokens):
        continue

    errors_in_sentence = 0
    for idx in error_indices:
        if fixed_tokens[idx] != target_tokens[idx]:
            errors_in_sentence += 1
            error_words.loc[error_words.shape[0]] = [fixed_tokens[idx], target_tokens[idx]]

    if errors_in_sentence == 0 and not error_indices:
        exact_sentences.loc[exact_sentences.shape[0]] = [input_sent, fixed_sentence_str, target_sent]

    if errors_in_sentence != 0:
        error_sentence.loc[error_sentence.shape[0]] = [input_sent, fixed_sentence_str, target_sent]

print("\n[+] HOÀN TẤT PIPELINE KIỂM THỬ!")