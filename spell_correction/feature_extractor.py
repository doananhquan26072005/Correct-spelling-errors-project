import math
import numpy as np
from typing import List, Dict, Tuple
from common.logger import get_logger

logger = get_logger(__name__)

class FeatureExtractor:
    """
    Đóng gói các phương thức trích xuất đặc trưng (features) phục vụ cho mô hình xếp hạng.
    Khởi tạo một lần với các models và dictionaries để tái sử dụng xuyên suốt pipeline.
    """

    def __init__(self, 
                 word_to_idx: Dict[str, int], 
                 norm_embedding_matrix: np.ndarray, 
                 model_lm, 
                 counts_1: dict, 
                 counts_2: dict, 
                 counts_3: dict,
                 cfg=None):  # Bổ sung tham số cfg
        
        self.word_to_idx = word_to_idx
        self.norm_embedding_matrix = norm_embedding_matrix
        self.model_lm = model_lm
        self.counts_1 = counts_1
        self.counts_2 = counts_2
        self.counts_3 = counts_3
        self.cfg = cfg
        
        # Log thống kê dữ liệu đầu vào
        logger.info("Initializing FeatureExtractor framework...")
        logger.info(f" + Vocabulary mapping loaded: {len(self.word_to_idx):,} tokens.")
        logger.info(f" + Word2Vec embedding matrix shape: {self.norm_embedding_matrix.shape}")
        logger.info(f" + N-gram frequencies loaded - Unigrams: {len(self.counts_1):,} | Bigrams: {len(self.counts_2):,} | Trigrams: {len(self.counts_3):,}")
        
        if self.cfg is None:
            logger.warning("FeatureExtractor initialized WITHOUT config (cfg=None). compute_score will fallback to default hardcoded weights.")
        else:
            logger.info("Configuration applied successfully for heuristic scoring weights.")

    def extract_similarity_features(self, valid_candidates: List[str], ctx_indices: List[int], ctx_weights: List[float]) -> Dict[str, float]:
        """Trích xuất đặc trưng similarity từ model skip-gram."""
        cand_to_sim = {}
        if valid_candidates and ctx_indices:
            cand_indices = [self.word_to_idx[c] for c in valid_candidates if c in self.word_to_idx]
            
            if not cand_indices:
                logger.debug("No valid candidate indices matched in word_to_idx vocabulary. Skipping similarity extraction.")
                return cand_to_sim

            C = self.norm_embedding_matrix[cand_indices]
            W = self.norm_embedding_matrix[ctx_indices]
            
            S = np.dot(C, W.T)
            weights_array = np.array(ctx_weights)
            S_weighted = S * weights_array
            
            max_sims = np.max(S_weighted, axis=1)
            cand_to_sim = {cand: max_sims[i] for i, cand in enumerate(valid_candidates)}
            
        return cand_to_sim

    def extract_kenlm_feature(self, candidate: str, prefix_str: str, suffix_str: str) -> Tuple[float, float]:
        """Trích xuất đặc trưng log-probability từ model kenlm và chuẩn hóa."""
        local_sentence_str = f"{prefix_str}{candidate}{suffix_str}".strip()
        ken_score = self.model_lm.score(local_sentence_str)
        norm_ken = max(0.0, (ken_score + 15.0) / 15.0)
        return ken_score, norm_ken

    def extract_ngram_counts_feature(self, candidate_lower: str, prev_word: str, prev_2_word: str, next_word: str, next_2_word: str) -> Tuple[int, int, int, float, float, float]:
        """Trích xuất đặc trưng tần suất của unigram, bigram, trigram và chuẩn hóa log."""
        # Unigram
        c1 = self.counts_1.get(candidate_lower, 0)
        
        # Bigram gộp (Trái + Phải)
        c2_left  = self.counts_2.get(f"{prev_word} {candidate_lower}", 0)
        c2_right = self.counts_2.get(f"{candidate_lower} {next_word}", 0)
        c2 = c2_left + c2_right
        
        # Trigram gộp (Tâm + Trái + Phải)
        c3_center = self.counts_3.get(f"{prev_word} {candidate_lower} {next_word}", 0)
        c3_left   = self.counts_3.get(f"{candidate_lower} {next_word} {next_2_word}", 0)
        c3_right  = self.counts_3.get(f"{prev_2_word} {prev_word} {candidate_lower}", 0)
        c3 = c3_center + c3_left + c3_right
        
        # Chuẩn hóa log mượt dữ liệu cho màng lọc tuyển chọn Hard Negatives
        norm_c1 = min(1.0, math.log1p(c1) / 15.0)
        norm_c2 = min(1.0, math.log1p(c2) / 12.0)
        norm_c3 = min(1.0, math.log1p(c3) / 12.0)
        
        return c1, c2, c3, norm_c1, norm_c2, norm_c3

    @staticmethod
    def extract_length_ratio_feature(error_word: str, candidate: str) -> float:
        """Trích xuất đặc trưng tỉ lệ chiều dài giữa từ lỗi và ứng viên."""
        len_err = len(error_word)
        len_cand = len(candidate)
        length_ratio = min(len_err, len_cand) / max(len_err, len_cand) if max(len_err, len_cand) > 0 else 0
        return length_ratio
    
    def compute_score(self, norm_ken: float, norm_edit: float, length_ratio: float, 
                      norm_c1: float, norm_c2: float, norm_c3: float, norm_sim: float) -> float:
        """Tính điểm Heuristic tổng hợp dựa trên trọng số cấu hình."""
        w = self.cfg.feature_weights if self.cfg and hasattr(self.cfg, 'feature_weights') else None
        if w:
            return (
                (w.w_kenlm * norm_ken) +
                (w.w_edit * norm_edit) +
                (w.w_len * length_ratio) +
                (w.w_bigram * norm_c2) +
                (w.w_trigram * norm_c3) +
                (w.w_unigram * norm_c1) +
                (w.w_sim * norm_sim)
            )
        else:
            # Fallback weights nếu config không tồn tại
            return (0.30 * norm_ken) + (0.25 * norm_edit) + (0.10 * length_ratio) + (0.20 * norm_c2)