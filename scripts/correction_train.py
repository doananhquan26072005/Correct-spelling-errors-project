# -*- coding: utf-8 -*-
from common.config import load_config

# Các import hiện tại ...
from spell_correction.processor import TeencodeProcessor, CandidateGenerator, FeatureExtractor, SpellCorrectionPipeline
from spell_correction.evaluator import Evaluator, Visualizer
from spell_correction.trainer import LightGBMRankerTrainer
from spell_correction.dataset import process_dataset, split_data, ResourceLoader
from datasets import load_dataset
import pandas as pd
import torch
import kenlm

def main():
    # 1. Config
    cfg = load_config("configs/correction_config.yaml")
    if not hasattr(cfg, "DEVICE"):
        cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load Resources (Áp dụng Class mới)
    print("[*] Đang nạp từ điển và tài nguyên hệ thống...")
    resource_loader = ResourceLoader(cfg)
    resources = resource_loader.load_all() 
    # resources chứa: word_to_idx, stopwords, telex_dict, norm_embedding_matrix

    # 3. Dataset
    print("[*] Đang tiền xử lý Dataset...")
    dataset = load_dataset("yammdd/vietnamese-error-correction-corpus")
    df = dataset.map(process_dataset, batched=True, remove_columns=dataset['train'].column_names, fn_kwargs={"word_to_idx": resources['word_to_idx']})
    df1, _ = split_data(df)
    df_train = pd.DataFrame(df['train'])
    df1_valid = pd.DataFrame(df1['validation'])

    # 4. Initialize Engines
    print("[*] Đang khởi tạo các cấu phần hệ thống...")
    teencode_engine = TeencodeProcessor(cfg.paths.teen_code_file)
    model_lm = kenlm.Model(cfg.paths.trigram_lm_file)
    
    generator = CandidateGenerator(vocab=resources['vocab'], telex_dict=resources['telex_dict'], cfg=cfg)
    generator.fit_ngram_counts(df_train['target'])

    extractor_engine = FeatureExtractor(
        word_to_idx=resources['word_to_idx'],
        norm_embedding_matrix=resources['norm_embedding_matrix'],
        stopwords=resources['stopwords'],
        cfg=cfg
    )
    evaluator = Evaluator(model_lm=model_lm, config=cfg)

    # 5. Train/Load Ranker
    print("[*] Đang nạp mô hình phân hạng LightGBM Ranker...")
    ranker_trainer = LightGBMRankerTrainer(cfg)
    X_train, y_train, group_train = ranker_trainer.load_training_data(cfg.paths.lightgbm_data_path)
    ranker = ranker_trainer.train(X_train, y_train, group_train)

    # 6. Pipeline (Áp dụng Class mới thay thế hàm đơn lẻ)
    pipeline = SpellCorrectionPipeline(
        cfg=cfg, evaluator=evaluator, model_lm=model_lm, generator=generator,
        extractor_engine=extractor_engine, ranker=ranker, word_to_idx=resources['word_to_idx']
    )

    # 7. Evaluation (Gom thành một hàm hoặc giữ nguyên vì đã gọi qua Evaluator)
    print("\n" + "="*30 + " BẮT ĐẦU CHU TRÌNH ĐÁNH GIÁ " + "="*30)
    evaluator.evaluate_error_detection(df1_valid, teencode_engine)
    evaluator.evaluate_ranking_performance(df1_valid, teencode_engine, extractor_engine, generator, ranker)
    evaluator.evaluate_word_accuracy(df1_valid, teencode_engine, pipeline.correct_sentence, resources['word_to_idx'])
    evaluator.evaluate_end_to_end(df1_valid, teencode_engine, pipeline.correct_sentence)

    # 8. Visualization & Analysis (Áp dụng Class mới)
    print("[*] Đang phân loại câu mẫu trực quan...")
    visualizer = Visualizer(pipeline, teencode_engine, evaluator, resources['word_to_idx'])
    exact_df, error_df, word_errors_df = visualizer.analyze_predictions(df1_valid, num_samples=200)

    print("\n[+] HOÀN TẤT PIPELINE KIỂM THỬ!")

if __name__ == "__main__":
    main()