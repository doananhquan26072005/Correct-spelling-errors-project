# teencode_processor.py
import os

class TeencodeProcessor:
    def __init__(self, teen_code_path: str):
        self.abbreviation_dict = {}
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
    
# candidate_generator.py
import re
from collections import defaultdict, Counter
import pandas as pd
from typing import List, Dict, Set

class CandidateGenerator:
    """Class chịu trách nhiệm xử lý các biến thể Telex, thuật toán Symmetric Delete 
    và tính khoảng cách Damerau-Levenshtein cải tiến để sinh danh sách ứng viên từ vựng."""
    
    def __init__(self, vocab: List[str], telex_dict: dict, cfg=None):
        self.vocab = vocab
        self.telex_dict = telex_dict
        self.cfg = cfg
        self.sym_dict = defaultdict(list)
        
        self.counts_1 = Counter()
        self.counts_2 = Counter()
        self.counts_3 = Counter()
        
        # Hằng số bàn phím & Cặp âm nhầm lẫn dùng cho Damerau-Levenshtein
        self.ADJACENT_KEYS = {
            'q': 'wea', 'w': 'qeasd', 'e': 'wrsdf', 'r': 'etdfg', 't': 'ryfgh', 'y': 'tughj', 'u': 'yihjk', 'i': 'uojkl', 'o': 'ipkl', 'p': 'ol',
            'a': 'qwsz', 's': 'weadzx', 'd': 'ersfxc', 'f': 'rtdgcv', 'g': 'tyfhvb', 'h': 'yugjbn', 'j': 'uihknm', 'k': 'iojlm', 'l': 'opk',
            'z': 'asx', 'x': 'sdzc', 'c': 'dfxv', 'v': 'fgcb', 'b': 'ghvn', 'n': 'hjbm', 'm': 'jkn'
        }
        self.CONFUSION_PAIRS = {
            ('s', 'x'), ('x', 's'), ('l', 'n'), ('n', 'l'), ('d', 'r'), ('r', 'd'),
            ('d', 'gi'), ('gi', 'd'), ('i', 'y'), ('y', 'i'), ('c', 'k'), ('k', 'c'),
            ('ch', 'tr'), ('tr', 'ch')
        }
        
        # Tự động build từ điển biến thể xóa khi khởi tạo đối tượng
        self._build_symmetric_delete_dictionary()

    def create_telex_form(self, word: str) -> List[str]:
        """Tạo nhiều biến thể Telex của 1 từ tiếng Việt dựa trên quy tắc Unicode."""
        word = word.lower()
        prefix = ""      # Phụ âm đầu
        vowel_base = ""  # Nguyên âm gốc
        suffix = ""      # Phụ âm cuối
        word_tone = ""   # Dấu thanh
        word_mod = ""    # Ký tự gõ mũ/móc

        VOWELS = "aeiouy"
        state = 0  # 0: phụ âm đầu, 1: nguyên âm

        i = 0
        while i < len(word):
            step = 1
            if i < len(word) - 1 and word[i:i+2] in self.telex_dict:
                char = word[i:i+2]
                step = 2
            else:
                char = word[i]

            if char in self.telex_dict:
                if char == 'đ':
                    if state == 0: prefix += 'dd'
                    else: suffix += 'dd'
                else:
                    vowel_base += self.telex_dict[char][0]
                    if self.telex_dict[char][1]: word_mod = self.telex_dict[char][1]
                    if self.telex_dict[char][2]: word_tone = self.telex_dict[char][2]
                    state = 1
            else:
                if char in VOWELS:
                    vowel_base += char
                    state = 1
                else:
                    if state == 0: prefix += char
                    else: suffix += char
            i += step

        variants = set()
        inline_vowel = vowel_base + word_mod

        variants.add(prefix + inline_vowel + word_tone + suffix)
        variants.add(prefix + inline_vowel + suffix + word_tone)

        if word_mod:
            variants.add(prefix + vowel_base + word_tone + suffix + word_mod)
            variants.add(prefix + vowel_base + suffix + word_mod + word_tone)
            variants.add(prefix + vowel_base + suffix + word_tone + word_mod)

        if vowel_base == 'uo' and word_mod == 'w':
            variants.add(prefix + 'uwow' + word_tone + suffix)
            variants.add(prefix + 'uwow' + suffix + word_tone)
            variants.add(prefix + 'uwo' + word_tone + suffix + 'w')
            variants.add(prefix + 'uwo' + suffix + 'w' + word_tone)
            variants.add(prefix + 'uwo' + suffix + word_tone + 'w')

        return list(v for v in variants if v)

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
        """Phương thức nội bộ: Ánh xạ toàn bộ biến thể xóa của Vocab vào sym_dict."""
        # Đọc tham số độ sâu xóa k động từ cấu hình (mặc định là 2)
        max_delete_k = self.cfg.candidate_generation.max_delete_k if self.cfg else 2

        for word in self.vocab:
            if ' ' in word:
                continue
            base_forms = [word] + self.create_telex_form(word)
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

    def edit_distance(self, s1: str, s2: str) -> float:
        """Tính khoảng cách Damerau-Levenshtein có hiệu chỉnh trọng số bàn phím và ngữ âm vùng miền."""
        # Đọc các trọng số phạt (costs) động từ file cấu hình
        if self.cfg and hasattr(self.cfg, 'candidate_generation'):
            c_gen = self.cfg.candidate_generation
            cost_confusion = c_gen.sub_cost_confusion
            cost_adjacent = c_gen.sub_cost_adjacent
            cost_transposition = c_gen.transposition_cost
        else:
            cost_confusion = 0.4
            cost_adjacent = 0.5
            cost_transposition = 0.5

        n, m = len(s1), len(s2)
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1): dp[i][0] = i
        for j in range(m + 1): dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                char1 = s1[i - 1]
                char2 = s2[j - 1]
                
                if char1 == char2:
                    sub_cost = 0.0
                elif (char1, char2) in self.CONFUSION_PAIRS:
                    sub_cost = cost_confusion
                elif char1 in self.ADJACENT_KEYS.get(char2, "") or char2 in self.ADJACENT_KEYS.get(char1, ""):
                    sub_cost = cost_adjacent
                else:
                    sub_cost = 1.0

                dp[i][j] = min(
                    dp[i - 1][j] + 1,                 # Xóa
                    dp[i][j - 1] + 1,                 # Thêm
                    dp[i - 1][j - 1] + sub_cost       # Thay thế
                )

                # Phép đổi chỗ ký tự kế cận (Transposition)
                if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                    dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + cost_transposition)

                # Các trường hợp tổ hợp âm lỗi chính tả tiếng Việt ghép 2 ký tự
                if i >= 2 and j >= 2:
                    if (s1[i-2:i], s2[j-2:j]) in self.CONFUSION_PAIRS:
                        dp[i][j] = min(dp[i][j], dp[i-2][j-2] + cost_confusion)
                
                if i >= 2 and j >= 1:
                    if (s1[i-2:i], s2[j-1:j]) in self.CONFUSION_PAIRS:
                        dp[i][j] = min(dp[i][j], dp[i-2][j-1] + cost_confusion)

                if i >= 1 and j >= 2:
                    if (s1[i-1:i], s2[j-2:j]) in self.CONFUSION_PAIRS:
                        dp[i][j] = min(dp[i][j], dp[i-1][j-2] + cost_confusion)

        return dp[n][m]

    def edit_distance_telex(self, s1: str, s2: str) -> float:
        """Tìm khoảng cách chỉnh sửa nhỏ nhất giữa mọi tổ hợp biến thể Telex của hai chuỗi."""
        min_dist = float('inf')
        string1 = self.create_telex_form(s1)
        string2 = self.create_telex_form(s2)

        for str1 in string1:
            for str2 in string2:
                dist = self.edit_distance(str1, str2)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def lookup(self, word: str, word_to_idx: Dict[str, int], k: int = 2) -> List[str]:
        """Tra cứu nhanh và trả về danh sách các ứng viên được sắp xếp theo độ ưu tiên khoảng cách và tần suất."""
        max_delete_k = self.cfg.candidate_generation.max_delete_k if self.cfg else 2

        variant_list = [word] + list(self.get_deletes(word, k=max_delete_k))
        for telex_form in self.create_telex_form(word):
            variant_list += list(self.get_deletes(telex_form, k=max_delete_k))

        candidates = {}

        for variant in variant_list:
            if variant in self.sym_dict:
                for suggestion in self.sym_dict[variant]:
                    if suggestion in candidates:
                        continue

                    dist = self.edit_distance_telex(word, suggestion)

                    # Chỉ chấp nhận nếu khoảng cách đạt yêu cầu, nằm trong từ điển và tần suất đơn > 0
                    if dist <= k and suggestion in word_to_idx and self.counts_1.get(suggestion, 0) > 0:
                        candidates[suggestion] = (dist, self.counts_1.get(suggestion, 0))

        # Sắp xếp: Khoảng cách tăng dần trước -> Tần suất giảm dần sau
        result = sorted(candidates.items(), key=lambda x: (x[1][0], -x[1][1]))
        return [cand_word for cand_word, _ in result]

# feature_extractor.py
import math
import numpy as np
from typing import List, Dict, Set, Tuple

class FeatureExtractor:
    """Class đảm nhiệm việc tính toán và trích xuất vector đặc trưng đa nguồn 
    (Context Embedding Similarity, KenLM Log-Prob, N-Gram Frequency, Length Ratio) 
    từ danh sách ứng viên phục vụ cho mô hình phân hạng LightGBM Ranker."""
    
    def __init__(self, word_to_idx: Dict[str, int], norm_embedding_matrix: np.ndarray, stopwords: Set[str], cfg):
        self.word_to_idx = word_to_idx
        self.norm_embedding_matrix = norm_embedding_matrix
        self.stopwords = stopwords
        self.cfg = cfg

    def extract_similarity_features(self, valid_candidates: List[str], ctx_indices: List[int], ctx_weights: List[float]) -> Dict[str, float]:
        """Tính toán đặc trưng độ tương đồng ngữ cảnh từ ma trận nhúng Skip-gram bằng phép toán vector hóa."""
        cand_to_sim = {}
        if valid_candidates and ctx_indices:
            cand_indices = [self.word_to_idx[c] for c in valid_candidates]
            
            C = self.norm_embedding_matrix[cand_indices]
            W = self.norm_embedding_matrix[ctx_indices]
            
            # Tính tích vô hướng Cosine Similarity cực nhanh bằng ma trận
            S = np.dot(C, W.T)
            weights_array = np.array(ctx_weights)
            S_weighted = S * weights_array
            
            max_sims = np.max(S_weighted, axis=1)
            cand_to_sim = {cand: max_sims[i] for i, cand in enumerate(valid_candidates)}
        return cand_to_sim

    def extract_kenlm_feature(self, candidate: str, prefix_str: str, suffix_str: str, model_lm) -> Tuple[float, float]:
        """Trích xuất điểm Log-probability từ mô hình KenLM và áp dụng chuẩn hóa min-max cục bộ."""
        # Đọc tham số trần dịch chuyển từ config
        ken_ceiling = self.cfg.feature_normalization.kenlm_min_max_ceiling if self.cfg else 15.0

        local_sentence_str = f"{prefix_str}{candidate}{suffix_str}".strip()
        ken_score = model_lm.score(local_sentence_str)
        norm_ken = max(0.0, (ken_score + ken_ceiling) / ken_ceiling)
        return ken_score, norm_ken

    def extract_ngram_counts_feature(
        self, candidate_lower: str, prev_word: str, prev_2_word: str, 
        next_word: str, next_2_word: str, generator
    ) -> Tuple[int, int, int, float, float, float]:
        """Thống kê tần suất xuất hiện và mượt log (log1p) dữ liệu cho Unigram, Bigram, Trigram."""
        # Đọc các hệ số chia chuẩn hóa log mượt từ config
        if self.cfg and hasattr(self.cfg, 'feature_normalization'):
            f_norm = self.cfg.feature_normalization
            norm_u = f_norm.log_norm_unigram
            norm_b = f_norm.log_norm_bigram
            norm_t = f_norm.log_norm_trigram
        else:
            norm_u = 15.0
            norm_b = 12.0
            norm_t = 12.0

        # Unigram
        c1 = generator.counts_1.get(candidate_lower, 0)
        
        # Bigram gộp (Trái + Phải)
        c2_left  = generator.counts_2.get(f"{prev_word} {candidate_lower}", 0)
        c2_right = generator.counts_2.get(f"{candidate_lower} {next_word}", 0)
        c2 = c2_left + c2_right
        
        # Trigram gộp (Tâm + Trái + Phải)
        c3_center = generator.counts_3.get(f"{prev_word} {candidate_lower} {next_word}", 0)
        c3_left   = generator.counts_3.get(f"{candidate_lower} {next_word} {next_2_word}", 0)
        c3_right  = generator.counts_3.get(f"{prev_2_word} {prev_word} {candidate_lower}", 0)
        c3 = c3_center + c3_left + c3_right
        
        # Chuẩn hóa log mượt dữ liệu động theo tham số cấu hình
        norm_c1 = min(1.0, math.log1p(c1) / norm_u)
        norm_c2 = min(1.0, math.log1p(c2) / norm_b)
        norm_c3 = min(1.0, math.log1p(c3) / norm_t)
        
        return c1, c2, c3, norm_c1, norm_c2, norm_c3

    def extract_length_ratio_feature(self, error_word: str, candidate: str) -> float:
        """Tính toán đặc trưng tỉ lệ chiều dài ký tự giữa từ gốc lỗi và ứng viên đúng gợi ý."""
        len_err = len(error_word)
        len_cand = len(candidate)
        length_ratio = min(len_err, len_cand) / max(len_err, len_cand) if max(len_err, len_cand) > 0 else 0
        return length_ratio

    def extract_candidates_and_features(
        self, error_word: str, sentence_words: List[str], error_idx: int, 
        error_indices: List[int], candidates: List[str], model_lm, generator
    ) -> List[Tuple[str, List[float]]]:
        """Pipeline trung tâm điều phối trích xuất toàn bộ vector đặc trưng cho tập ứng viên."""
        n_words = len(sentence_words)
        
        window = self.cfg.model.window_size if self.cfg else 3

        # 1. Chuẩn bị ngữ cảnh cho KenLM
        local_start = max(0, error_idx - window)
        local_end = min(n_words, error_idx + window + 1)
        prefix_str = " ".join(sentence_words[local_start:error_idx]) + " " if sentence_words[local_start:error_idx] else ""
        suffix_str = " " + " ".join(sentence_words[error_idx + 1:local_end]) if sentence_words[error_idx + 1:local_end] else ""

        # 2. Chuẩn bị ngữ cảnh cho N-gram Frequency Counts
        prev_word = sentence_words[error_idx - 1].lower() if error_idx > 0 else "<s>"
        prev_2_word = sentence_words[error_idx - 2].lower() if error_idx > 1 else "<s>"
        next_word = sentence_words[error_idx + 1].lower() if error_idx < n_words - 1 else "</s>"
        next_2_word = sentence_words[error_idx + 2].lower() if error_idx < n_words - 2 else "</s>"

        # 3. Chuẩn bị ngữ cảnh cho Word Embedding Similarity
        valid_context_words = []
        for i in range(local_start, local_end):
            if i == error_idx:
                continue
            if (i < error_idx or i not in error_indices) and sentence_words[i] not in self.stopwords:
                word = sentence_words[i]
                if word in self.word_to_idx:
                    dist_weight = 1.0 / abs(i - error_idx)
                    valid_context_words.append((word, dist_weight))

        ctx_indices = [self.word_to_idx[w] for w, _ in valid_context_words]
        ctx_weights = [weight for _, weight in valid_context_words]

        # Gọi phương thức tính toán similarity vector hóa
        cand_to_sim = self.extract_similarity_features(candidates, ctx_indices, ctx_weights)

        # Lấy dict trọng số đặc trưng từ config
        w = self.cfg.feature_weights if self.cfg else None

        top_candidates = []
        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Đặc trưng 1: Semantic Similarity
            weighted_sim = cand_to_sim.get(candidate, 0.0)
            norm_sim = max(0.0, weighted_sim)

            # Đặc trưng 2: KenLM Language Model
            ken_score, norm_ken = self.extract_kenlm_feature(candidate, prefix_str, suffix_str, model_lm)

            # Đặc trưng 3: N-gram Counts Frequency
            c1, c2, c3, norm_c1, norm_c2, norm_c3 = self.extract_ngram_counts_feature(
                candidate_lower, prev_word, prev_2_word, next_word, next_2_word, generator
            )

            # Đặc trưng 4: Edit Distance (Gọi thông qua đối tượng generator đã tích hợp)
            dist_val = generator.edit_distance_telex(error_word, candidate)
            norm_edit = 1.0 / (dist_val + 1)

            # Đặc trưng 5: Length Ratio
            length_ratio = self.extract_length_ratio_feature(error_word, candidate)

            # Tính toán Heuristic total score sử dụng trọng số cấu hình động từ SimpleNamespace
            if w:
                total_score = (
                    (w.w_kenlm * norm_ken) +
                    (w.w_edit * norm_edit) +
                    (w.w_len * length_ratio) +
                    (w.w_bigram * norm_c2) +
                    (w.w_trigram * norm_c3) +
                    (w.w_unigram * norm_c1) +
                    (w.w_sim * norm_sim)
                )
            else:
                total_score = (0.30 * norm_ken) + (0.25 * norm_edit) + (0.10 * length_ratio) + (0.20 * norm_c2)

            top_candidates.append((total_score, candidate, ken_score, weighted_sim, c1, c2, c3, dist_val, length_ratio))

        # Sắp xếp giảm dần dựa trên tổng điểm heuristics tốt nhất
        top_candidates.sort(key=lambda x: x[0], reverse=True)

        feature_records = []
        for item in top_candidates:
            (_, candidate, ken_score, weighted_sim, c1, c2, c3, dist_val, length_ratio) = item
            feature_vector = [ken_score, weighted_sim, c1, c2, c3, dist_val, length_ratio]
            feature_records.append((candidate, feature_vector))

        return feature_records