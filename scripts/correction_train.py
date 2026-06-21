import time
import numpy as np
import pandas as pd
import torch
import kenlm
import gdown
from datasets import load_dataset

# Khởi tạo module logger chuyên dụng cho luồng chạy sửa lỗi chính tả
from common.logger import get_logger
from common.config import load_config
from spell_correction import (
    process_dataset,
    split_data,
    ResourceLoader,
    AbbreviationProcessor,
    CandidateGenerator,
    FeatureExtractor,
    SpellCorrectionPipeline,
    Evaluator,
    Visualizer,
    LightGBMRankerTrainer,
    SkipGramTrainer,
    SkipGram
)

logger = get_logger("SpellCorrectionMainPipeline")

def main():
    logger.info("=== BẮT ĐẦU PIPELINE SỬA LỖI CHÍNH TẢ TIẾNG VIỆT ===")
    pipeline_start_time = time.time()

    try:
        # ==========================================
        # 1. KHỞI TẠO CẤU HÌNH ĐỘNG (CONFIG)
        # ==========================================
        config_path = "configs/correction_config.yaml"
        logger.info(f"Loading correction configurations from: {config_path}")
        cfg = load_config(config_path)

        if not hasattr(cfg, "DEVICE"):
            cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Target compute hardware device allocated: {cfg.DEVICE.upper()}")

        # ==========================================
        # 2. TỰ ĐỘNG TẢI VÀ NẠP TÀI NGUYÊN (RESOURCE LOADER)
        # ==========================================
        logger.info("=== BƯỚC 1: TỰ ĐỘNG KIỂM TRA & NẠP TÀI NGUYÊN HỆ THỐNG ===")
        resource_loader = ResourceLoader(cfg)
        resources = resource_loader.load_vocab_and_dicts()
        
        # Giải nén tài nguyên
        word_to_idx = resources["word_to_idx"]
        vocab = resources["vocab"]
        stopwords = resources["stopwords"]
        telex_dict = resources["telex_dict"]

        # Khởi tạo Abbreviation Processor
        abbr_engine = AbbreviationProcessor(cfg.paths.teen_code_file)

        # ==========================================
        # 3. TIỀN XỬ LÝ VÀ CHIA DATASET
        # ==========================================
        logger.info("=== BƯỚC 2: TẢI HUGGINGFACE CORPUS & TIỀN XỬ LÝ DỮ LIỆU ===")
        corpus_name = "yammdd/vietnamese-error-correction-corpus"
        logger.info(f"Downloading corpus data framework: {corpus_name}")
        
        dataset = load_dataset(corpus_name)
        
        logger.info("Mapping clean token processors over parallel corpus batched matrices...")
        df = dataset.map(
            process_dataset,
            batched=True,
            remove_columns=dataset['train'].column_names,
            fn_kwargs={"word_to_idx": word_to_idx}
        )

        # Phân mảnh dữ liệu theo 2 luồng lỗi chuyên biệt
        df1, _ = split_data(df)
        df1_train = pd.DataFrame(df1['train'])
        df1_valid = pd.DataFrame(df1['validation'])


        # ==========================================
        # 4. KHỞI TẠO CÁC ENGINES SUY LUẬN CHÍNH
        # ==========================================
        logger.info("=== BƯỚC 3: KHỞI TẠO CÁC ENGINE SUY LUẬN NGỮ CẢNH ===")
        logger.info(f"Loading Language Model binary tree from: {cfg.paths.trigram_lm_file}")
        model_lm = kenlm.Model(cfg.paths.trigram_lm_file)

        generator = CandidateGenerator(vocab=vocab, telex_dict=telex_dict, cfg=cfg)
        
        logger.info("Fitting training targets for extracting background corpus n-gram frequencies...")
        generator.fit_ngram_counts(df1_train['target'])

        # Train skipgram
        # skipgram_trainer = SkipGramTrainer(cfg)
        # skipgram_train_dataset = skipgram_trainer.build_dataset(df1_train["target"])
        # skipgram_trainer.train(skipgram_train_dataset)
        # norm_embedding_matrix = skipgram_trainer.get_norm_embedding()

        # Load skipgram
        model_skipgram = SkipGram(len(vocab), cfg.model.embed_dim).to(cfg.DEVICE)
        gdown.download(id=cfg.paths.skipgram_model_url, output=cfg.paths.skipgram_model_file, quiet=False)
        model_skipgram.load_state_dict(torch.load(cfg.paths.skipgram_model_file, map_location=cfg.DEVICE, weights_only=True))
        embeddings = model_skipgram.embedding.weight.data
        embedding_matrix = embeddings.cpu().numpy()
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1 
        norm_embedding_matrix = embedding_matrix / norms

        extractor_engine = FeatureExtractor(
            word_to_idx=word_to_idx,
            norm_embedding_matrix=norm_embedding_matrix,
            model_lm=model_lm,
            counts_1=generator.counts_1,
            counts_2=generator.counts_2,
            counts_3=generator.counts_3,
            cfg=cfg
        )

        evaluator = Evaluator(model_lm=model_lm, config=cfg)

        # ==========================================
        # 5. KHỞI TẠO VÀ TẢI MÔ HÌNH LIGHTGBM RANKER
        # ==========================================
        logger.info("=== BƯỚC 4: NẠP DỮ LIỆU ĐẶC TRƯNG & HUẤN LUYỆN LIGHTGBM RANKER ===")
        ranker_trainer = LightGBMRankerTrainer(
            abbr_processor=abbr_engine,
            evaluator=evaluator,
            generator=generator,
            feature_extractor=extractor_engine,
            cfg=cfg
        )
        
        # X_train, y_train, group_train = ranker_trainer.build_dataset(df1_train, stopwords)
        X_train, y_train, group_train = ranker_trainer.load_training_data(cfg.paths.lightgbm_data_file)

        ranker = ranker_trainer.train(X_train, y_train, group_train)

        # ==========================================
        # 6. KHỞI TẠO PIPELINE SỬA LỖI CHÍNH TẢ TOÀN CỤC
        # ==========================================
        logger.info("=== BƯỚC 5: ĐÓNG GÓI HỆ THỐNG PIPELINE LIÊN KẾT THỐNG NHẤT ===")
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
        logger.info("=== BƯỚC 6: BẮT ĐẦU CHU TRÌNH ĐÁNH GIÁ ĐỊNH LƯỢNG (VALIDATION METRICS) ===")
        evaluation_start_time = time.time()

        # 7.1 Đánh giá cấu phần Detect Lỗi
        evaluator.evaluate_error_detection(
            validation_df=df1_valid, 
            abbr_engine=abbr_engine
        )

        # 7.2 Đánh giá cấu phần Ranker ứng viên (MRR, Hit@K)
        evaluator.evaluate_ranking_performance(
            validation_df=df1_valid,
            abbr_engine=abbr_engine,
            extractor_engine=extractor_engine,
            generator=generator,
            ranker=ranker,
            stopwords=stopwords,
        )

        # 7.3 Đánh giá chất lượng sửa từ (Word Accuracy)
        evaluator.evaluate_word_accuracy(
            validation_df=df1_valid,
            abbr_engine=abbr_engine,
            pipeline_correct_fn=pipeline.correct_sentence,
            stopwords=stopwords,
            word_to_idx=word_to_idx
        )

        # 7.4 Đánh giá hệ thống toàn cục (End-to-End WER/SER)
        evaluator.evaluate_end_to_end(
            validation_df=df1_valid,
            abbr_engine=abbr_engine,
            pipeline_correct_fn=pipeline.correct_sentence,
            stopwords=stopwords
        )
        logger.info(f"Toàn bộ chu trình đánh giá hoàn thành trong {time.time() - evaluation_start_time:.2f}s.")

        # ==========================================
        # 8. TRỰC QUAN HÓA KẾT QUẢ SUY LUẬN THỰC TẾ
        # ==========================================
        logger.info("=== BƯỚC 7: TRỰC QUAN HÓA & PHÂN LOẠI MẪU DỰ ĐOÁN QUAN SÁT ===")
        visualizer = Visualizer(
            pipeline=pipeline,
            abbr_engine=abbr_engine,
            evaluator=evaluator,
            word_to_idx=word_to_idx
        )
        
        exact_sentences, error_sentence, error_words = visualizer.analyze_predictions(
            validation_df=df1_valid,
            stopwords=stopwords,
            num_samples=200
        )

        # Thống kê kết xuất mẫu định lượng
        logger.info("-----------------------------------------------------------------------")
        logger.info(f" [+] Số câu sửa chính xác hoàn toàn (Exact Match) : {len(exact_sentences):,}")
        logger.info(f" [+] Số câu bị sửa sai / bỏ sót lỗi (Remained)      : {len(error_sentence):,}")
        logger.info(f" [+] Tổng số từ lỗi ghi nhận phân tích mẫu           : {len(error_words):,}")
        logger.info("-----------------------------------------------------------------------")

        total_pipeline_time = time.time() - pipeline_start_time
        logger.info(f"=== PIPELINE HOÀN THÀNG TOÀN DIỆN BƯỚC 2 TRONG {total_pipeline_time:.2f}s ===")

    except FileNotFoundError as e:
        logger.error(f"Critical data path target verification failed: {str(e)}", exc_info=True)
    except ValueError as e:
        logger.error(f"Matrix array value alignment breakdown or dynamic downsampling fail: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected structural failure within pipeline execution context: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()