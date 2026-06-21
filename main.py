import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import kenlm
import gdown
from datasets import load_dataset

# Import common modules
from common.logger import get_logger
from common.config import load_config

# Import Spell Correction modules
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

# Import Diacritic Restoration modules
from diacritic_restoration import (
    DiacriticDataLoaderFactory,
    build_model,
    DiacriticRestorer,
    DiacriticTrainer,
    build_allowed_token_mask
)

logger = get_logger("NLPPipeline")

def main():
    logger.info("=== BẮT ĐẦU PIPELINE HỆ THỐNG NLP TOÀN DIỆN ===")
    pipeline_start_time = time.time()

    try:
        # ==========================================
        # 1. KHỞI TẠO CẤU HÌNH (CONFIGS)
        # ==========================================
        spell_config_path = "configs/correction_config.yaml"
        dia_config_path = "configs/diacritic_config.yaml"
        
        logger.info(f"Loading configs from: {spell_config_path} & {dia_config_path}")
        spell_cfg = load_config(spell_config_path)
        dia_cfg = load_config(dia_config_path)

        # Cấu hình Device cho Spell Correction
        if not hasattr(spell_cfg, "DEVICE"):
            spell_cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Cấu hình Device cho Diacritic Restoration
        if dia_cfg.training.device == "auto":
            dia_cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        logger.info(f"Hardware allocated - Spell: {spell_cfg.DEVICE.upper()} | Diacritic: {dia_cfg.training.device.upper()}")

        # ==========================================
        # 2. NẠP TÀI NGUYÊN HỆ THỐNG (SPELL CORRECTION)
        # ==========================================
        logger.info("=== PHẦN 1: TỰ ĐỘNG KIỂM TRA & NẠP TÀI NGUYÊN HỆ THỐNG ===")
        resource_loader = ResourceLoader(spell_cfg)
        resources = resource_loader.load_vocab_and_dicts()
        
        word_to_idx = resources["word_to_idx"]
        vocab = resources["vocab"]
        stopwords = resources["stopwords"]
        telex_dict = resources["telex_dict"]

        abbr_engine = AbbreviationProcessor(spell_cfg.paths.teen_code_file)

        # ==========================================
        # 3. TIỀN XỬ LÝ VÀ CHIA DATASET (CHIA DF1 VÀ DF2)
        # ==========================================
        logger.info("=== PHẦN 2: TẢI HUGGINGFACE CORPUS & TIỀN XỬ LÝ DỮ LIỆU ===")
        corpus_name = "yammdd/vietnamese-error-correction-corpus"
        dataset = load_dataset(corpus_name)
        
        df = dataset.map(
            process_dataset,
            batched=True,
            remove_columns=dataset['train'].column_names,
            fn_kwargs={"word_to_idx": word_to_idx}
        )

        df_test = pd.DataFrame(df['test'])
        # Trích xuất df1 (Spell) và df2 (Diacritic)
        df1, df2 = split_data(df)
        
        df1_train = pd.DataFrame(df1['train'])
        df1_valid = pd.DataFrame(df1['validation'])
        df1_test = pd.DataFrame(df1['test'])

        # Ánh xạ df2 cho bài toán thêm dấu
        df2_train = pd.DataFrame(df2['train'])
        df2_valid = pd.DataFrame(df2['validation'])
        df2_test = pd.DataFrame(df2['test'])
        logger.info(f"Đã phân mảnh dữ liệu thành công: df1 (train: {len(df1_train)}), df2 (train: {len(df2_train)})")

        # ==========================================
        # 4. ENGINE SUY LUẬN & FEATURE EXTRACTOR (SPELL)
        # ==========================================
        logger.info("=== PHẦN 3: KHỞI TẠO CÁC ENGINE SUY LUẬN NGỮ CẢNH ===")
        model_lm = kenlm.Model(spell_cfg.paths.trigram_lm_file)

        generator = CandidateGenerator(vocab=vocab, telex_dict=telex_dict, cfg=spell_cfg)
        generator.fit_ngram_counts(df1_train['target'])

        # Train skipgram
        # skipgram_trainer = SkipGramTrainer(cfg)
        # skipgram_train_dataset = skipgram_trainer.build_dataset(df1_train["target"])
        # skipgram_trainer.train(skipgram_train_dataset)
        # norm_embedding_matrix = skipgram_trainer.get_norm_embedding()

        # Load skipgram
        model_skipgram = SkipGram(len(vocab), spell_cfg.model.embed_dim).to(spell_cfg.DEVICE)
        gdown.download(id=spell_cfg.paths.skipgram_model_url, output=spell_cfg.paths.skipgram_model_file, quiet=False)
        model_skipgram.load_state_dict(torch.load(spell_cfg.paths.skipgram_model_file, map_location=spell_cfg.DEVICE, weights_only=True))
        
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
            cfg=spell_cfg
        )

        evaluator = Evaluator(model_lm=model_lm, config=spell_cfg)

        # ==========================================
        # 5. LIGHTGBM RANKER (SPELL)
        # ==========================================
        logger.info("=== PHẦN 4: NẠP DỮ LIỆU ĐẶC TRƯNG & HUẤN LUYỆN LIGHTGBM RANKER ===")
        ranker_trainer = LightGBMRankerTrainer(
            abbr_processor=abbr_engine,
            evaluator=evaluator,
            generator=generator,
            feature_extractor=extractor_engine,
            cfg=spell_cfg
        )
        
        # X_train, y_train, group_train = ranker_trainer.build_dataset(df1_train, stopwords)
        X_train, y_train, group_train = ranker_trainer.load_training_data(spell_cfg.paths.lightgbm_data_file)
        ranker = ranker_trainer.train(X_train, y_train, group_train)

        # ==========================================
        # 6. PIPELINE CHÍNH TẢ & ĐÁNH GIÁ (SPELL)
        # ==========================================
        logger.info("=== PHẦN 5: ĐÁNH GIÁ PIPELINE CHÍNH TẢ ===")
        pipeline = SpellCorrectionPipeline(
            cfg=spell_cfg, abbr_processor=abbr_engine, evaluator=evaluator, 
            model_lm=model_lm, generator=generator, extractor_engine=extractor_engine,
            ranker=ranker, word_to_idx=word_to_idx
        )

        # evaluator.evaluate_error_detection(validation_df=df1_valid, abbr_engine=abbr_engine)
        # evaluator.evaluate_ranking_performance(
        #     validation_df=df1_valid, abbr_engine=abbr_engine,
        #     extractor_engine=extractor_engine, generator=generator,
        #     ranker=ranker, stopwords=stopwords
        # )
        # evaluator.evaluate_word_accuracy(
        #     validation_df=df1_valid, abbr_engine=abbr_engine,
        #     pipeline_correct_fn=pipeline.correct_sentence,
        #     stopwords=stopwords, word_to_idx=word_to_idx
        # )
        # evaluator.evaluate_end_to_end(
        #     validation_df=df1_valid, abbr_engine=abbr_engine,
        #     pipeline_correct_fn=pipeline.correct_sentence, stopwords=stopwords
        # )

        # visualizer = Visualizer(pipeline=pipeline, abbr_engine=abbr_engine, evaluator=evaluator, word_to_idx=word_to_idx)
        # exact_sentences, error_sentence, error_words = visualizer.analyze_predictions(
        #     validation_df=df1_valid, stopwords=stopwords, num_samples=200
        # )

        # ==========================================
        # 7. CHẠY KHÔI PHỤC DẤU TRÊN DF2 (DIACRITIC RESTORATION)
        # ==========================================
        logger.info("=== PHẦN 6: CHẠY PIPELINE KHÔI PHỤC DẤU TRÊN DF2 ===")
        
        # Khởi tạo data_factory và xây dựng môi trường từ điển cho diacritic
        data_factory = DiacriticDataLoaderFactory(dia_cfg)
        train_loader, valid_loader, test_loader, char_vocab, word_vocab = data_factory.build_loaders_and_vocabs(df2_train, df2_valid, df2_test)
        allowed_mask = build_allowed_token_mask(char_vocab, dia_cfg)

        restorer = DiacriticRestorer(
            checkpoint_path=dia_cfg.training.checkpoint_path,
            cfg=dia_cfg
        )
        
        logger.info("=== DEMO SUY LUẬN THỰC TẾ (INFERENCE TRÊN DF2) ===")
        # Lấy văn bản từ df2_valid (Giả sử cột chứa văn bản gốc không dấu là 'input')
        # Lưu ý: Chỉnh sửa lại tên cột 'input' thành tên cột phù hợp với dataset thực tế của bạn
        col_name = 'input' if 'input' in df2_valid.columns else df2_valid.columns[0]
        
        # Lấy 10 dòng đầu tiên của DF2 để demo suy luận tránh làm tràn log
        df2_examples = df2_valid[col_name].dropna().head(10).tolist()

        for idx, text in enumerate(df2_examples, 1):
            inference_start = time.time()
            predicted_text = restorer.process(text, stopwords)
            inference_time = time.time() - inference_start
            
            logger.info(f"Sample #{idx} | Inference time: {inference_time*1000:.2f}ms")
            logger.info(f"  > [IN]  : {text}")
            logger.info(f"  > [OUT] : {predicted_text}")

        logger.info("=== BẮT ĐẦU ĐÁNH GIÁ ĐỊNH LƯỢNG CHO KHÔI PHỤC DẤU ===")

        # Giả định df2_valid có cột 'text' (đầu vào không dấu) và 'target' (nhãn có dấu)
        # Chúng ta mượn luôn evaluator từ spell_correction

        # 1. Đánh giá Accuracy cấp độ từ
        # evaluator.evaluate_word_accuracy(
        #     validation_df=df2_valid,
        #     abbr_engine=abbr_engine, # Có thể None nếu khôi phục dấu không cần teen code
        #     pipeline_correct_fn=restorer.process, # Truyền hàm suy luận của Diacritic vào đây
        #     stopwords=stopwords,
        #     word_to_idx=word_to_idx
        # )

        # # 2. Đánh giá WER / SER đầu cuối
        # evaluator.evaluate_end_to_end(
        #     validation_df=df2_valid,
        #     abbr_engine=abbr_engine,
        #     pipeline_correct_fn=restorer.process, # Truyền hàm suy luận của Diacritic vào đây
        #     stopwords=stopwords
        # )
        logger.info("=== PHẦN 6: ĐÁNH GIÁ HỆ THỐNG TRÊN TẬP TEST CHUNG (KHÔNG CHIA TÁCH) ===")

        # 💡 Định nghĩa hàm chuỗi liên kết (Joint Pipeline Function)
        # Đầu vào không dấu/sai lỗi -> Khôi phục dấu -> Sửa lỗi chính tả & teen-code
        def joint_nlp_pipeline(text, stopwords=stopwords):
            if not text or not isinstance(text, str):
                return ""
            logger.debug(f"--------------------------------------------------------------------------------")
            logger.debug(f"Input text: {text}")
            for char in text:
                if char in "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ":
                    # Nếu phát hiện ký tự có dấu, bỏ qua khôi phục dấu
                    logger.debug(f"Input text already contains diacritics. Skipping diacritic restoration.")
                    final_clean_text = pipeline.correct_sentence(text, stopwords)
                    logger.debug(f"Final corrected text: {final_clean_text}")
                    return final_clean_text
            # Bước 1: Dựng lại dấu tiếng Việt chuẩn ngữ cảnh
            restored_text = restorer.process(text, stopwords)
            logger.debug(f"Restored text: {restored_text}")
            # Bước 2: Sửa các lỗi chính tả, gõ sai telex, viết tắt còn sót lại
            final_clean_text = pipeline.correct_sentence(restored_text, stopwords)
            logger.debug(f"Final corrected text: {final_clean_text}")
            
            return final_clean_text

        logger.info("--- Đang chạy đánh giá Chuỗi liên kết (Diacritic + Spell) trên df_test chung ---")
        
        # 7.1 Đánh giá Accuracy cấp độ từ trên tập test tổng hợp
        evaluator.evaluate_word_accuracy(
            validation_df=df_test[:100], # Truyền trực tiếp tập test thô vào đây
            abbr_engine=abbr_engine,
            pipeline_correct_fn=joint_nlp_pipeline, # Sử dụng hàm kết hợp chuỗi
            stopwords=stopwords,
            word_to_idx=word_to_idx
        )

        # 7.2 Đánh giá WER / SER (Word/Sentence Error Rate) cuối cùng trên toàn tập test
        evaluator.evaluate_end_to_end(
            validation_df=df_test[:100],
            abbr_engine=abbr_engine,
            pipeline_correct_fn=joint_nlp_pipeline,
            stopwords=stopwords
        )

        total_pipeline_time = time.time() - pipeline_start_time
        logger.info(f"=== PIPELINE TOÀN DIỆN HOÀN THÀNH TRONG {total_pipeline_time:.2f}s ===")

    except Exception as e:
        logger.error(f"Unexpected structural failure within pipeline: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()