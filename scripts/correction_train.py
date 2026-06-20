# -*- coding: utf-8 -*-
import os
import re
import math
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import kenlm
from datasets import load_dataset

# Import các hàm xử lý dữ liệu ngoại vi của bạn
from spell_correction.dataset import process_dataset, split_data

# Import các module cấu trúc lại theo kiến trúc mới
from common.config import load_config
from spell_correction.processor import TeencodeProcessor, CandidateGenerator, FeatureExtractor, SpellCorrectionPipeline
from spell_correction.evaluator import Evaluator, Visualizer
from spell_correction.trainer import LightGBMRankerTrainer
from spell_correction.dataset import ResourceLoader

def main():
    # ==========================================
    # 1. KHỞI TẠO CẤU HÌNH ĐỘNG (CONFIG)
    # ==========================================
    cfg = load_config("configs/correction_config.yaml")

    if not hasattr(cfg, "DEVICE"):
        cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 2. TỰ ĐỘNG TẢI VÀ NẠP TÀI NGUYÊN (RESOURCE LOADER)
    # ==========================================
    # Khởi tạo bộ điều phối tài nguyên
    resource_loader = ResourceLoader(cfg)
    
    # Tự động kiểm tra file Skip-gram .npz cục bộ, nếu thiếu sẽ tự gdown về, sau đó nạp tất cả
    resources = resource_loader.load_all()
    
    # Giải nén nhanh các tài nguyên cần dùng cho các bước tiếp theo
    word_to_idx = resources["word_to_idx"]
    vocab = resources["vocab"]
    stopwords = resources["stopwords"]
    telex_dict = resources["telex_dict"]
    norm_embedding_matrix = resources["norm_embedding_matrix"]

    # Khởi tạo Teencode Processor từ tài nguyên đã nạp
    teencode_engine = TeencodeProcessor(cfg.paths.teen_code_file)

    # ==========================================
    # 3. TIỀN XỬ LÝ VÀ CHIA DATASET
    # ==========================================
    print("[*] Đang nạp và tiền xử lý Dataset từ HuggingFace...")
    dataset = load_dataset("yammdd/vietnamese-error-correction-corpus")
    
    df = dataset.map(
        process_dataset,
        batched=True,
        remove_columns=dataset['train'].column_names,
        fn_kwargs={"word_to_idx": word_to_idx}
    )

    df1, _ = split_data(df)
    df_train = pd.DataFrame(df['train'])
    df1_valid = pd.DataFrame(df1['validation'])

    # ==========================================
    # 4. KHỞI TẠO CÁC ENGINES SUY LUẬN CHÍNH
    # ==========================================
    print("[*] Đang nạp mô hình KenLM và khởi tạo Candidate Generator...")
    model_lm = kenlm.Model(cfg.paths.trigram_lm_file)

    # Khởi tạo bộ sinh ứng viên (Candidate Generator)
    generator = CandidateGenerator(vocab=vocab, telex_dict=telex_dict, cfg=cfg)
    print("[*] Đang thống kê tần suất N-gram từ tập huấn luyện...")
    generator.fit_ngram_counts(df_train['target'])

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
    
    X_train, y_train, group_train = ranker_trainer.load_training_data(cfg.paths.lightgbm_data_path)
    ranker = ranker_trainer.train(X_train, y_train, group_train)

    # ==========================================
    # 6. KHỞI TẠO PIPELINE SỬA LỖI CHÍNH TẢ TOÀN CỤC
    # ==========================================
    # Đóng gói toàn bộ các engine rời rạc phía trên vào một Pipeline thống nhất
    pipeline = SpellCorrectionPipeline(
        cfg=cfg,
        evaluator=evaluator,
        model_lm=model_lm,
        generator=generator,
        extractor_engine=extractor_engine,
        ranker=ranker,
        word_to_idx=word_to_idx
    )

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
        pipeline_correct_fn=pipeline.correct_sentence,  # Truyền method của pipeline vào thay cho hàm cũ
        word_to_idx=word_to_idx
    )

    # 7.4 Đánh giá hệ thống toàn cục (End-to-End WER/SER)
    evaluator.evaluate_end_to_end(
        validation_df=df1_valid,
        teencode_engine=teencode_engine,
        pipeline_correct_fn=pipeline.correct_sentence  # Truyền method của pipeline vào thay cho hàm cũ
    )

    # ==========================================
    # 8. TRỰC QUAN HÓA KẾT QUẢ SUY LUẬN THỰC TẾ
    # ==========================================
    print("\n" + "="*30 + " TRỰC QUAN HÓA KẾT QUẢ " + "="*30)
    
    # Khởi tạo bộ phân tích trực quan
    visualizer = Visualizer(
        pipeline=pipeline,
        teencode_engine=teencode_engine,
        evaluator=evaluator,
        word_to_idx=word_to_idx
    )
    
    # Tiến hành phân tích 200 mẫu dữ liệu validation
    exact_sentences, error_sentence, error_words = visualizer.analyze_predictions(
        validation_df=df1_valid,
        num_samples=200
    )

    # (Tùy chọn) Bạn có thể dùng exact_sentences, error_sentence để in ra màn hình hoặc ghi file tại đây nếu muốn.
    print(f"[+] Số câu sửa chính xác hoàn toàn: {len(exact_sentences)}")
    print(f"[+] Số câu bị sửa sai/sót lỗi: {len(error_sentence)}")
    print(f"[+] Tổng số từ lỗi ghi nhận: {len(error_words)}")

    print("\n[+] HOÀN TẤT PIPELINE KIỂM THỬ THÀNH CÔNG!")

if __name__ == "__main__":
    main()