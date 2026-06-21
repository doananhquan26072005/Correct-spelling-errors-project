import time
import numpy as np

from common.logger import get_logger

logger = get_logger(__name__)
class SpellCorrectionPipeline:
    def __init__(self, cfg, evaluator, model_lm, generator, extractor_engine, ranker, word_to_idx):
        """
        Đóng gói Pipeline sửa lỗi chính tả tổng thể.
        """
        self.cfg = cfg
        self.evaluator = evaluator
        self.model_lm = model_lm
        self.generator = generator
        self.extractor_engine = extractor_engine
        self.ranker = ranker
        self.word_to_idx = word_to_idx
        logger.info("SpellCorrectionPipeline deployed and verified.")

    def correct_sentence(self, sentence: str) -> str:
        """
        Pipeline khép kín nhận diện lỗi, trích xuất đặc trưng và sửa lỗi câu hoàn chỉnh.
        """
        logger.info(f"=== STARTING TYPO CORRECTION PIPELINE FOR: '{sentence}' ===")
        start_pipeline = time.time()
        
        error_indices = self.evaluator.detect_error(sentence)
        sentence_tokens = sentence.split()
        
        if error_indices:
            logger.debug(f"Detected potential typographical errors at token indices: {error_indices}")
        else:
            logger.debug("No clear typos identified by the initial heuristic evaluator.")

        for idx in error_indices:
            if idx >= len(sentence_tokens):
                continue
            error_word = sentence_tokens[idx]
            logger.info(f"Processing correction sequence for token suspect: '{error_word}' at index [{idx}]")

            # 1. Sinh ứng viên thô qua lookup
            k_candidates = self.cfg.candidate_generation.top_k_raw if hasattr(self.cfg, 'candidate_generation') else 2
            raw_candidates = self.generator.lookup(error_word, self.word_to_idx, k=k_candidates)

            if not raw_candidates:
                logger.warning(f"Unable to discover appropriate dictionary suggestions for token '{error_word}'. Skipping.")
                continue

            # 2. Trích xuất vector đặc trưng động
            candidates_with_features = self.extractor_engine.extract_candidates_and_features(
                error_word=error_word,
                sentence_words=sentence_tokens,
                error_idx=idx,
                error_indices=error_indices,
                candidates=raw_candidates,
                model_lm=self.model_lm,
                generator=self.generator
            )

            if not candidates_with_features:
                continue

            cand_words = [item[0] for item in candidates_with_features]
            X_infer = np.array([item[1] for item in candidates_with_features])

            # 3. Dự đoán phân hạng điểm số bằng LightGBM Ranker
            scores = self.ranker.predict(X_infer)
            best_idx = np.argmax(scores)
            best_candidate = cand_words[best_idx]
            
            logger.debug(f"Candidate: {best_candidate} achieved max LightGBM rank score: {scores[best_idx]:.4f}")

            # 4. Thay thế từ đúng ngữ cảnh vào chuỗi token phục vụ bước tiếp theo
            if error_word != best_candidate:
                logger.info(f"Successfully amended typo: '{error_word}' -> '{best_candidate}'")
                sentence_tokens[idx] = best_candidate
            else:
                logger.debug(f"Model decided to retain original spelling structure for token: '{error_word}'")

        corrected_sentence = " ".join(sentence_tokens)
        logger.info(f"=== PIPELINE COMPLETED. Elapsed execution time: {time.time() - start_pipeline:.4f}s ===")
        logger.info(f"Final Output String: '{corrected_sentence}'")
        
        return corrected_sentence