import time
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
from common.logger import get_logger

logger = get_logger(__name__)

class CandidateGenerator:
    """Chịu trách nhiệm quản lý Symmetric Delete và tra cứu ứng viên từ vựng sơ bộ."""
    
    # Hằng số cấu hình bàn phím vật lý QWERTY (Cố định, không đổi)
    _ADJACENT_KEYS: Dict[str, str] = {
        'q': 'wea', 'w': 'qeasd', 'e': 'wrsdf', 'r': 'etdfg', 't': 'ryfgh', 'y': 'tughj', 'u': 'yihjk', 'i': 'uojkl', 'o': 'ipkl', 'p': 'ol',
        'a': 'qwsz', 's': 'weadzx', 'd': 'ersfxc', 'f': 'rtdgcv', 'g': 'tyfhvb', 'h': 'yugjbn', 'j': 'uihknm', 'k': 'iojlm', 'l': 'opk',
        'z': 'asx', 'x': 'sdzc', 'c': 'dfxv', 'v': 'fgcb', 'b': 'ghvn', 'n': 'hjbm', 'm': 'jkn'
    }
    
    def __init__(self, vocab: List[str], telex_dict: Dict, cfg=None):
        self.vocab = vocab
        self.cfg = cfg
        self.telex_dict = telex_dict
        self.sym_dict = defaultdict(list)
        
        self.counts_1 = Counter()
        self.counts_2 = Counter()
        self.counts_3 = Counter()
        
        logger.info("Building internal Symmetric Delete inverted index framework...")
        start_time = time.time()
        self._build_symmetric_delete_dictionary()
        logger.info(f"Symmetric Delete index compiled successfully in {time.time() - start_time:.2f}s.")

    def create_telex_form(self, word: str) -> List[str]:
        word = word.lower()
        prefix, vowel_base, suffix, word_tone, word_mod = "", "", "", "", ""
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
        max_delete_k = self.cfg.candidate_generation.max_delete_k if self.cfg and hasattr(self.cfg, 'candidate_generation') else 2

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
        logger.debug(f"Total distinct keys loaded into sym_dict inverted index: {len(self.sym_dict):,}")

    def compute_edit_distance(self, s1: str, s2: str, confusion_pairs: Set[Tuple[str, str]]) -> float:
        """Tính khoảng cách Damerau-Levenshtein có hiệu chỉnh trọng số bàn phím và vùng miền."""
        if self.cfg and hasattr(self.cfg, 'candidate_generation'):
            c_gen = self.cfg.candidate_generation
            cost_confusion = c_gen.sub_cost_confusion
            cost_adjacent = c_gen.sub_cost_adjacent
            cost_transposition = c_gen.transposition_cost
        else:
            cost_confusion, cost_adjacent, cost_transposition = 0.4, 0.5, 0.5

        n, m = len(s1), len(s2)
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1): dp[i][0] = i
        for j in range(m + 1): dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                char1, char2 = s1[i - 1], s2[j - 1]
                
                if char1 == char2:
                    sub_cost = 0.0
                elif (char1, char2) in confusion_pairs:
                    sub_cost = cost_confusion
                elif char1 in self._ADJACENT_KEYS.get(char2, "") or char2 in self._ADJACENT_KEYS.get(char1, ""):
                    sub_cost = cost_adjacent
                else:
                    sub_cost = 1.0

                dp[i][j] = min(
                    dp[i - 1][j] + 1,                  # Xóa
                    dp[i][j - 1] + 1,                  # Thêm
                    dp[i - 1][j - 1] + sub_cost        # Thay thế
                )

                # Phép đổi chỗ ký tự kế cận (Transposition)
                if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                    dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + cost_transposition)

                # Các trường hợp tổ hợp âm lỗi chính tả tiếng Việt ghép 2 ký tự
                if i >= 2 and j >= 2 and (s1[i-2:i], s2[j-2:j]) in confusion_pairs:
                    dp[i][j] = min(dp[i][j], dp[i-2][j-2] + cost_confusion)
                if i >= 2 and j >= 1 and (s1[i-2:i], s2[j-1:j]) in confusion_pairs:
                    dp[i][j] = min(dp[i][j], dp[i-2][j-1] + cost_confusion)
                if i >= 1 and j >= 2 and (s1[i-1:i], s2[j-2:j]) in confusion_pairs:
                    dp[i][j] = min(dp[i][j], dp[i-1][j-2] + cost_confusion)

        return dp[n][m]

    def compute_edit_distance_telex(self, s1: str, s2: str, confusion_pairs: Set[Tuple[str, str]]) -> float:
        """Tìm khoảng cách chỉnh sửa nhỏ nhất giữa mọi tổ hợp biến thể Telex của hai chuỗi."""
        min_dist = float('inf')
        string1 = self.create_telex_form(s1)
        string2 = self.create_telex_form(s2)

        for str1 in string1:
            for str2 in string2:
                dist = self.compute_edit_distance(str1, str2, confusion_pairs)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def fit_ngram_counts(self, train_targets: pd.Series):
        logger.info("Computing Uni/Bi/Tri-gram contextual background frequency tables...")
        start_time = time.time()
        
        for sentence in train_targets:
            tokens = str(sentence).lower().split()
            if not tokens:
                continue
            self.counts_1.update(tokens)
            
            bigrams = [" ".join(p) for p in zip(tokens, tokens[1:])]
            self.counts_2.update(bigrams)
            
            trigrams = [" ".join(t) for t in zip(tokens, tokens[1:], tokens[2:])]
            self.counts_3.update(trigrams)
            
        logger.info(
            f"N-gram frequency mappings generated in {time.time() - start_time:.2f}s | "
            f"Unigrams: {len(self.counts_1):,} | Bigrams: {len(self.counts_2):,} | Trigrams: {len(self.counts_3):,}"
        )

    def lookup(self, word: str, word_to_idx: Dict[str, int], k: int = 2) -> List[str]:
        max_delete_k = self.cfg.candidate_generation.max_delete_k if self.cfg and hasattr(self.cfg, 'candidate_generation') else 2

        variant_list = [word] + list(self.get_deletes(word, k=max_delete_k))
        for telex_form in self.create_telex_form(word):
            variant_list += list(self.get_deletes(telex_form, k=max_delete_k))

        candidates = {}
        for variant in variant_list:
            if variant in self.sym_dict:
                for suggestion in self.sym_dict[variant]:
                    if suggestion in candidates:
                        continue

                    raw_pairs = self.cfg.confusion_pairs if (self.cfg and hasattr(self.cfg, 'confusion_pairs')) else {}
                    if hasattr(raw_pairs, '__dict__'):
                        confusion_dict = raw_pairs.__dict__
                    elif isinstance(raw_pairs, dict):
                        confusion_dict = raw_pairs
                    else:
                        confusion_dict = {}

                    dist = self.compute_edit_distance_telex(word, suggestion, confusion_dict)
                    
                    if dist <= k and suggestion in word_to_idx and self.counts_1.get(suggestion, 0) > 0:
                        candidates[suggestion] = (dist, self.counts_1.get(suggestion, 0))

        result = sorted(candidates.items(), key=lambda x: (x[1][0], -x[1][1]))
        candidate_words = [cand_word for cand_word, _ in result]
        
        logger.debug(f"Lookup for suspect '{word}': Generated {len(candidate_words)} early-stage candidate structures.")
        return candidate_words

    