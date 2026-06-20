import os
import math
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
from spell_correction.utils import create_telex_form, compute_edit_distance, compute_edit_distance_telex
from spell_correction.evaluator import HeuristicScorer

# ==========================================
# 1. TEENCODE PROCESSOR MODULE
# ==========================================

class TeencodeProcessor:
    """Chịu trách nhiệm nạp và xử lý thay thế các từ viết tắt/teencode."""
    def __init__(self, teen_code_path: str):
        self.abbreviation_dict: Dict[str, str] = {}
        self._load_dictionary(teen_code_path)
        
    def _load_dictionary(self, path: str):
        if not os.path.exists(path):
            print(f"⚠️ Warning: Teen code file not found at {path}")
            return
            
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    shortcut = parts[0].lower()
                    full_word = parts[1].lower()
                    self.abbreviation_dict[shortcut] = full_word

    def replace_abbreviations(self, sentence: str) -> str:
        words = sentence.lower().split()
        for i, word in enumerate(words):
            if word in self.abbreviation_dict:
                # Chỉ thay thế nếu từ tương ứng là từ đơn để giữ nguyên cấu trúc index câu
                if len(self.abbreviation_dict[word].split()) == 1:
                    words[i] = self.abbreviation_dict[word]
        return " ".join(words)

# ==========================================
# 3. CANDIDATE GENERATOR MODULE
# ==========================================

class CandidateGenerator:
    """Chịu trách nhiệm quản lý Symmetric Delete và tra cứu ứng viên từ vựng sơ bộ."""
    def __init__(self, vocab: List[str], cfg=None):
        self.vocab = vocab
        self.cfg = cfg
        self.sym_dict = defaultdict(list)
        
        self.counts_1 = Counter()
        self.counts_2 = Counter()
        self.counts_3 = Counter()
        
        self._build_symmetric_delete_dictionary()

    def get_deletes(self, word: str, k: int = 2) -> Set[str]:
        """Tạo tập các biến thể xóa từ 0 đến k ký tự của một chuỗi."""
        queue = {word}
        variant_list = set()
        
        for _ in range(k):
            temp_queue = set()
            for w in queue:
                if len(w) > 1:
                    deletes = {w[:i] + w[i+1:] for i in range(len(w))}
                    variant_list.update(deletes)
                    temp_queue.update(deletes)
            queue = temp_queue
        return variant_list

    def _build_symmetric_delete_dictionary(self):
        """Ánh xạ toàn bộ biến thể xóa của Vocab vào sym_dict."""
        max_delete_k = self.cfg.candidate_generation.max_delete_k if self.cfg else 2

        for word in self.vocab:
            if ' ' in word:
                continue
            base_forms = [word] + create_telex_form(word, self.cfg.telex_dict)
            for form in base_forms:
                if word not in self.sym_dict[form]:
                    self.sym_dict[form].append(word)
                
                variant_list = self.get_deletes(form, k=max_delete_k)
                for variant in variant_list:
                    if word not in self.sym_dict[variant]:
                        self.sym_dict[variant].append(word)

    def fit_ngram_counts(self, train_targets: pd.Series):
        """Thống kê tần suất xuất hiện Uni/Bi/Tri-gram từ tập huấn luyện."""
        for sentence in train_targets:
            tokens = str(sentence).lower().split()
            if not tokens:
                continue
            self.counts_1.update(tokens)
            
            bigrams = [" ".join(p) for p in zip(tokens, tokens[1:])]
            self.counts_2.update(bigrams)
            
            trigrams = [" ".join(t) for t in zip(tokens, tokens[1:], tokens[2:])]
            self.counts_3.update(trigrams)

    def lookup(self, word: str, word_to_idx: Dict[str, int], k: int = 2) -> List[str]:
        """Tra cứu nhanh danh sách các ứng viên được xếp hạng sơ bộ theo khoảng cách và tần suất unigram."""
        max_delete_k = self.cfg.candidate_generation.max_delete_k if self.cfg else 2

        variant_list = [word] + list(self.get_deletes(word, k=max_delete_k))
        for telex_form in create_telex_form(word, self.cfg.telex_dict):
            variant_list += list(self.get_deletes(telex_form, k=max_delete_k))

        candidates = {}
        for variant in variant_list:
            if variant in self.sym_dict:
                for suggestion in self.sym_dict[variant]:
                    if suggestion in candidates:
                        continue

                    dist = compute_edit_distance_telex(
                        word, 
                        suggestion, 
                        self.cfg.telex_dict, 
                        self.cfg.confusion_pairs, 
                        self.cfg
                    )
                    if dist <= k and suggestion in word_to_idx and self.counts_1.get(suggestion, 0) > 0:
                        candidates[suggestion] = (dist, self.counts_1.get(suggestion, 0))

        result = sorted(candidates.items(), key=lambda x: (x[1][0], -x[1][1]))
        return [cand_word for cand_word, _ in result]


# ==========================================
# 5. FEATURE EXTRACTOR MODULE
# ==========================================

class FeatureExtractor:
    """Đảm nhiệm việc trích xuất vector đặc trưng đa nguồn từ danh sách ứng viên."""
    def __init__(self, word_to_idx: Dict[str, int], norm_embedding_matrix: np.ndarray, 
                 stopwords: Set[str], cfg=None):
        self.word_to_idx = word_to_idx
        self.norm_embedding_matrix = norm_embedding_matrix
        self.stopwords = stopwords
        self.cfg = cfg
        # Khởi tạo đối tượng chấm điểm chuyên biệt
        self.scorer = HeuristicScorer(self.cfg)

    def extract_similarity_features(self, valid_candidates: List[str], ctx_indices: List[int], ctx_weights: List[float]) -> Dict[str, float]:
        cand_to_sim = {}
        if valid_candidates and ctx_indices:
            cand_indices = [self.word_to_idx[c] for c in valid_candidates]
            C = self.norm_embedding_matrix[cand_indices]
            W = self.norm_embedding_matrix[ctx_indices]
            
            S = np.dot(C, W.T)
            weights_array = np.array(ctx_weights)
            S_weighted = S * weights_array
            
            max_sims = np.max(S_weighted, axis=1)
            cand_to_sim = {cand: max_sims[i] for i, cand in enumerate(valid_candidates)}
        return cand_to_sim

    def extract_kenlm_feature(self, candidate: str, prefix_str: str, suffix_str: str, model_lm) -> Tuple[float, float]:
        ken_ceiling = self.cfg.feature_normalization.kenlm_min_max_ceiling if self.cfg else 15.0
        local_sentence_str = f"{prefix_str}{candidate}{suffix_str}".strip()
        ken_score = model_lm.score(local_sentence_str)
        norm_ken = max(0.0, (ken_score + ken_ceiling) / ken_ceiling)
        return ken_score, norm_ken

    def extract_ngram_counts_feature(self, candidate_lower: str, prev_word: str, prev_2_word: str, 
                                     next_word: str, next_2_word: str, generator: CandidateGenerator) -> Tuple[int, int, int, float, float, float]:
        if self.cfg and hasattr(self.cfg, 'feature_normalization'):
            f_norm = self.cfg.feature_normalization
            norm_u, norm_b, norm_t = f_norm.log_norm_unigram, f_norm.log_norm_bigram, f_norm.log_norm_trigram
        else:
            norm_u, norm_b, norm_t = 15.0, 12.0, 12.0

        c1 = generator.counts_1.get(candidate_lower, 0)
        c2 = generator.counts_2.get(f"{prev_word} {candidate_lower}", 0) + generator.counts_2.get(f"{candidate_lower} {next_word}", 0)
        
        c3 = (generator.counts_3.get(f"{prev_word} {candidate_lower} {next_word}", 0) +
              generator.counts_3.get(f"{candidate_lower} {next_word} {next_2_word}", 0) +
              generator.counts_3.get(f"{prev_2_word} {prev_word} {candidate_lower}", 0))
        
        norm_c1 = min(1.0, math.log1p(c1) / norm_u)
        norm_c2 = min(1.0, math.log1p(c2) / norm_b)
        norm_c3 = min(1.0, math.log1p(c3) / norm_t)
        
        return c1, c2, c3, norm_c1, norm_c2, norm_c3

    def extract_length_ratio_feature(self, error_word: str, candidate: str) -> float:
        len_err, len_cand = len(error_word), len(candidate)
        return min(len_err, len_cand) / max(len_err, len_cand) if max(len_err, len_cand) > 0 else 0

    def extract_candidates_and_features(self, error_word: str, sentence_words: List[str], error_idx: int, 
                                        error_indices: List[int], candidates: List[str], model_lm, 
                                        generator: CandidateGenerator) -> List[Tuple[str, List[float]]]:
        """Pipeline trung tâm điều phối trích xuất và trả về danh sách vector đặc trưng ứng viên."""
        n_words = len(sentence_words)
        window = self.cfg.model.window_size if self.cfg else 3

        # 1. Chuẩn bị ngữ cảnh KenLM
        local_start, local_end = max(0, error_idx - window), min(n_words, error_idx + window + 1)
        prefix_str = " ".join(sentence_words[local_start:error_idx]) + " " if sentence_words[local_start:error_idx] else ""
        suffix_str = " " + " ".join(sentence_words[error_idx + 1:local_end]) if sentence_words[error_idx + 1:local_end] else ""

        # 2. Chuẩn bị ngữ cảnh N-gram
        prev_word = sentence_words[error_idx - 1].lower() if error_idx > 0 else "<s>"
        prev_2_word = sentence_words[error_idx - 2].lower() if error_idx > 1 else "<s>"
        next_word = sentence_words[error_idx + 1].lower() if error_idx < n_words - 1 else "</s>"
        next_2_word = sentence_words[error_idx + 2].lower() if error_idx < n_words - 2 else "</s>"

        # 3. Chuẩn bị ngữ cảnh Embedding Similarity
        valid_context_words = []
        for i in range(local_start, local_end):
            if i != error_idx and (i < error_idx or i not in error_indices) and sentence_words[i] not in self.stopwords:
                if sentence_words[i] in self.word_to_idx:
                    valid_context_words.append((sentence_words[i], 1.0 / abs(i - error_idx)))

        ctx_indices = [self.word_to_idx[w] for w, _ in valid_context_words]
        ctx_weights = [weight for _, weight in valid_context_words]
        cand_to_sim = self.extract_similarity_features(candidates, ctx_indices, ctx_weights)

        # 4. Trích xuất đặc trưng & Tính toán điểm số thành phần
        top_candidates = []
        for candidate in candidates:
            candidate_lower = candidate.lower()

            norm_sim = max(0.0, cand_to_sim.get(candidate, 0.0))
            ken_score, norm_ken = self.extract_kenlm_feature(candidate, prefix_str, suffix_str, model_lm)
            c1, c2, c3, norm_c1, norm_c2, norm_c3 = self.extract_ngram_counts_feature(candidate_lower, prev_word, prev_2_word, next_word, next_2_word, generator)
            
            # Sử dụng đúng hàm import từ utils.py
            dist_val = compute_edit_distance_telex(
                error_word, 
                candidate, 
                self.cfg.telex_dict, 
                self.cfg.confusion_pairs, 
                self.cfg
            )
            norm_edit = 1.0 / (dist_val + 1)
            length_ratio = self.extract_length_ratio_feature(error_word, candidate)

            # Ủy quyền tính toán điểm số cho lớp HeuristicScorer chuyên biệt đã được khởi tạo
            total_score = self.scorer.compute_score(norm_ken, norm_edit, length_ratio, norm_c1, norm_c2, norm_c3, norm_sim)
            top_candidates.append((total_score, candidate, ken_score, norm_sim, c1, c2, c3, dist_val, length_ratio))

        # 5. Sắp xếp và format đầu ra chuẩn hóa
        top_candidates.sort(key=lambda x: x[0], reverse=True)
        return [(item[1], list(item[2:])) for item in top_candidates]