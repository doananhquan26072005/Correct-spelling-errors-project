import time
import numpy as np

from common.logger import get_logger

logger = get_logger(__name__)
class SpellCorrectionPipeline:
    def __init__(self, cfg, abbr_processor, evaluator, model_lm, generator, extractor_engine, ranker, word_to_idx):
        """
        Đóng gói Pipeline sửa lỗi chính tả tổng thể.
        """
        self.cfg = cfg
        self.abbr_processor = abbr_processor
        self.evaluator = evaluator
        self.model_lm = model_lm
        self.generator = generator
        self.extractor_engine = extractor_engine
        self.ranker = ranker
        self.word_to_idx = word_to_idx
        logger.info("SpellCorrectionPipeline deployed and verified.")

    def correct_sentence(self, sentence: str, stopwords) -> str:
        """
        Pipeline khép kín nhận diện lỗi, trích xuất đặc trưng và sửa lỗi câu hoàn chỉnh.
        """
        # logger.info(f"=== STARTING TYPO CORRECTION PIPELINE FOR: '{sentence}' ===")
        start_pipeline = time.time()
        sentence = self.abbr_processor.replace_abbreviations(sentence)

        error_indices = self.evaluator.detect_error(sentence)
        sentence_tokens = sentence.split()
        
        # if error_indices:
        #     logger.debug(f"Detected potential typographical errors at token indices: {error_indices}")
        # else:
        #     logger.debug("No clear typos identified by the initial heuristic evaluator.")

        for idx in error_indices:
            if idx >= len(sentence_tokens):
                continue
            error_word = sentence_tokens[idx]
            # logger.info(f"Processing correction sequence for token suspect: '{error_word}' at index [{idx}]")

            # 1. Sinh ứng viên thô qua lookup
            k_candidates = self.cfg.candidate_generation.top_k_raw if hasattr(self.cfg, 'candidate_generation') else 2
            raw_candidates = self.generator.lookup(error_word, self.word_to_idx, k=k_candidates)

            if not raw_candidates:
                # logger.warning(f"Unable to discover appropriate dictionary suggestions for token '{error_word}'. Skipping.")
                continue

            # 2. Trích xuất vector đặc trưng động
            candidates_with_features = extract_candidates_and_features(
                error_word=error_word,
                sentence_words=sentence_tokens,
                error_idx=idx,
                error_indices=error_indices,
                generator=self.generator,
                extractor=self.extractor_engine,
                stopwords=stopwords,
                cfg=self.cfg
            )

            if not candidates_with_features:
                continue

            cand_words = [item[0] for item in candidates_with_features]
            X_infer = np.array([item[1] for item in candidates_with_features])

            # 3. Dự đoán phân hạng điểm số bằng LightGBM Ranker
            scores = self.ranker.predict(X_infer)
            best_idx = np.argmax(scores)
            best_candidate = cand_words[best_idx]
            
            # logger.debug(f"Candidate: {best_candidate} achieved max LightGBM rank score: {scores[best_idx]:.4f}")

            # 4. Thay thế từ đúng ngữ cảnh vào chuỗi token phục vụ bước tiếp theo
            if error_word != best_candidate:
                # logger.info(f"Successfully amended typo: '{error_word}' -> '{best_candidate}'")
                sentence_tokens[idx] = best_candidate
            # else:
                # logger.debug(f"Model decided to retain original spelling structure for token: '{error_word}'")

        corrected_sentence = " ".join(sentence_tokens)
        # logger.info(f"=== PIPELINE COMPLETED. Elapsed execution time: {time.time() - start_pipeline:.4f}s ===")
        # logger.info(f"Final Output String: '{corrected_sentence}'")
        
        return corrected_sentence

from typing import List, Tuple, Set

def extract_candidates_and_features(
    error_word: str, 
    sentence_words: List[str], 
    error_idx: int, 
    error_indices: List[int],
    generator,           # Instance của CandidateGenerator
    extractor,           # Instance của FeatureExtractor
    stopwords: Set[str], 
    cfg=None,
    window_size: int = 3
) -> List[Tuple[str, List[float]]]:
    """
    Hàm điều phối: Đóng vai trò cầu nối giữa Generator (sinh ứng viên) 
    và Extractor (trích xuất đặc trưng), sau đó chấm điểm Heuristic để lọc.
    """
    n_words = len(sentence_words)
    
    # 1. Chuẩn bị ngữ cảnh cho KenLM (Tri-gram)
    local_start = max(0, error_idx - window_size)
    local_end = min(n_words, error_idx + window_size + 1)
    
    prefix_words = sentence_words[local_start:error_idx]
    suffix_words = sentence_words[error_idx + 1:local_end]
    prefix_str = " ".join(prefix_words) + " " if prefix_words else ""
    suffix_str = " " + " ".join(suffix_words) if suffix_words else ""

    # 2. Chuẩn bị ngữ cảnh cho N-gram tần suất
    prev_word = sentence_words[error_idx - 1].lower() if error_idx > 0 else "<s>"
    prev_2_word = sentence_words[error_idx - 2].lower() if error_idx > 1 else "<s>"
    next_word = sentence_words[error_idx + 1].lower() if error_idx < n_words - 1 else "</s>"
    next_2_word = sentence_words[error_idx + 2].lower() if error_idx < n_words - 2 else "</s>"

    # 3. Chọn ngữ cảnh và tính khoảng cách cho Word2Vec Similarity
    valid_context_words = []
    for i in range(local_start, local_end):
        if i == error_idx:
            continue
        if (i < error_idx or i not in error_indices) and sentence_words[i] not in stopwords:
            word = sentence_words[i]
            if word in extractor.word_to_idx:
                dist_weight = 1.0 / abs(i - error_idx) 
                valid_context_words.append((word, dist_weight))

    ctx_indices = [extractor.word_to_idx[ctx_word] for ctx_word, _ in valid_context_words]
    ctx_weights = [weight for _, weight in valid_context_words]

    # ==========================================
    # GIAI ĐOẠN ĐIỀU PHỐI (ORCHESTRATION)
    # ==========================================

    # Bước A: Gọi Generator để lấy ứng viên
    candidates = generator.lookup(error_word, extractor.word_to_idx)
    if not candidates:
        return []
    
    # Bước B: Gọi Extractor để tính Similarity (chạy 1 lần cho batch ứng viên)
    cand_to_sim = extractor.extract_similarity_features(candidates, ctx_indices, ctx_weights)

    top = []
    
    # Ép kiểu an toàn cho confusion_pairs từ cấu hình (tránh lỗi SimpleNamespace không lặp được)
    raw_pairs = cfg.confusion_pairs if (cfg and hasattr(cfg, 'confusion_pairs')) else {}
    if hasattr(raw_pairs, '__dict__'):
        confusion_pairs = raw_pairs.__dict__
    elif isinstance(raw_pairs, dict):
        confusion_pairs = raw_pairs
    elif isinstance(raw_pairs, (list, tuple, set)):
        confusion_pairs = set(raw_pairs)
    else:
        confusion_pairs = set()

    # Bước C: Trích xuất các đặc trưng còn lại và tính điểm Heuristic
    for candidate in candidates:
        candidate_lower = candidate.lower()
        
        # Đặc trưng 1: Similarity
        weighted_sim = cand_to_sim.get(candidate, 0.0)
        norm_sim = max(0.0, weighted_sim) 

        # Đặc trưng 2: KenLM
        ken_score, norm_ken = extractor.extract_kenlm_feature(candidate, prefix_str, suffix_str)

        # Đặc trưng 3: N-gram Counts
        c1, c2, c3, norm_c1, norm_c2, norm_c3 = extractor.extract_ngram_counts_feature(
            candidate_lower, prev_word, prev_2_word, next_word, next_2_word
        )

        # Đặc trưng 4: Edit Distance (Gọi qua Generator vì nó giữ telex_dict)
        dist_val = generator.compute_edit_distance_telex(error_word, candidate, confusion_pairs)
        norm_edit = 1.0 / (dist_val + 1)

        # Đặc trưng 5: Length Ratio
        length_ratio = extractor.extract_length_ratio_feature(error_word, candidate)

        # Tính Total Score (Heuristic Filter)
        total_score = extractor.compute_score(
            norm_ken=norm_ken, 
            norm_edit=norm_edit, 
            length_ratio=length_ratio, 
            norm_c1=norm_c1, 
            norm_c2=norm_c2, 
            norm_c3=norm_c3, 
            norm_sim=norm_sim
        )
        
        top.append((total_score, candidate, ken_score, weighted_sim, c1, c2, c3, dist_val, length_ratio))

    # ==========================================
    # GIAI ĐOẠN ĐÓNG GÓI OUTPUT
    # ==========================================
    
    # Sort để ưu tiên các ứng viên có điểm tổng hợp cao nhất
    top.sort(key=lambda x: x[0], reverse=True)
    
    mock_candidates = []
    for item in top:
        (_, candidate, ken_score, weighted_sim, c1, c2, c3, dist_val, length_ratio) = item
        
        # Đóng gói Vector đặc trưng chuẩn bị cho LightGBM
        feature_vector = [ken_score, weighted_sim, c1, c2, c3, dist_val, length_ratio]
        mock_candidates.append((candidate, feature_vector))

    return mock_candidates