# -*- coding: utf-8 -*-
import math
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

from common.logger import get_logger
from spell_correction.evaluator import HeuristicScorer
from spell_correction.utils import compute_edit_distance, compute_edit_distance_telex, create_telex_form

logger = get_logger(__name__)

# ==========================================
# 1. TEENCODE PROCESSOR MODULE
# ==========================================

class TeencodeProcessor:
    """Chịu trách nhiệm nạp và xử lý thay thế các từ viết tắt/teencode."""
    def __init__(self, teen_code_path: str):
        self.abbreviation_dict: Dict[str, str] = {}
        logger.info(f"Initializing TeencodeProcessor with dictionary target: {teen_code_path}")
        self._load_dictionary(teen_code_path)
        
    def _load_dictionary(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"Teencode map file NOT found at '{path}'. Skipping loading sequence.")
            return
            
        with open(path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    shortcut = parts[0].lower()
                    full_word = parts[1].lower()
                    self.abbreviation_dict[shortcut] = full_word
        logger.info(f"Successfully mapped {len(self.abbreviation_dict):,} teencode shorthand conversion pairs.")

    def replace_abbreviations(self, sentence: str) -> str:
        words = sentence.lower().split()
        replaced_count = 0
        for i, word in enumerate(words):
            if word in self.abbreviation_dict:
                if len(self.abbreviation_dict[word].split()) == 1:
                    logger.debug(f"Teencode replacement matched: '{word}' -> '{self.abbreviation_dict[word]}'")
                    words[i] = self.abbreviation_dict[word]
                    replaced_count += 1
        if replaced_count > 0:
            logger.debug(f"Total shorthand replacements executed in sentence: {replaced_count}")
        return " ".join(words)

# ==========================================
# 3. CANDIDATE GENERATOR MODULE
# ==========================================

class CandidateGenerator:
    """Chịu trách nhiệm quản lý Symmetric Delete và tra cứu ứng viên từ vựng sơ bộ."""
    
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
            
            base_forms = [word] + create_telex_form(word, self.telex_dict)
            for form in base_forms:
                if word not in self.sym_dict[form]:
                    self.sym_dict[form].append(word)
                
                variant_list = self.get_deletes(form, k=max_delete_k)
                for variant in variant_list:
                    if word not in self.sym_dict[variant]:
                        self.sym_dict[variant].append(word)
        logger.debug(f"Total distinct keys loaded into sym_dict inverted index: {len(self.sym_dict):,}")

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
        for telex_form in create_telex_form(word, self.telex_dict):
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

                    dist = compute_edit_distance_telex(
                        word, 
                        suggestion, 
                        self.telex_dict, 
                        confusion_dict,
                        self.cfg
                    )
                    if dist <= k and suggestion in word_to_idx and self.counts_1.get(suggestion, 0) > 0:
                        candidates[suggestion] = (dist, self.counts_1.get(suggestion, 0))

        result = sorted(candidates.items(), key=lambda x: (x[1][0], -x[1][1]))
        candidate_words = [cand_word for cand_word, _ in result]
        
        logger.debug(f"Lookup for suspect '{word}': Generated {len(candidate_words)} early-stage candidate structures.")
        return candidate_words

# ==========================================
# 5. FEATURE EXTRACTOR MODULE
# ==========================================

class FeatureExtractor:
    """Đảm nhiệm việc trích xuất vector đặc trưng đa nguồn từ danh sách ứng viên."""
    
    def __init__(self, word_to_idx: Dict[str, int], norm_embedding_matrix: np.ndarray, 
                 stopwords: Set[str], telex_dict: Dict = None, cfg=None):
        self.word_to_idx = word_to_idx
        self.norm_embedding_matrix = norm_embedding_matrix
        self.stopwords = stopwords
        self.cfg = cfg
        
        if telex_dict is not None:
            self.telex_dict = telex_dict
        else:
            self.telex_dict = cfg.telex_dict if (cfg and hasattr(cfg, 'telex_dict')) else {}
        logger.info("FeatureExtractor module online and fully calibrated.")

    def extract_similarity_features(self, valid_candidates: List[str], ctx_indices: List[int], ctx_weights: List[float]) -> Dict[str, float]:
        cand_to_sim = {c: 0.0 for c in valid_candidates}
        
        if valid_candidates and ctx_indices:
            actual_candidates = []
            cand_indices = []
            for c in valid_candidates:
                if c in self.word_to_idx:
                    actual_candidates.append(c)
                    cand_indices.append(self.word_to_idx[c])
            
            if not cand_indices:
                return cand_to_sim
                
            C = self.norm_embedding_matrix[cand_indices]
            W = self.norm_embedding_matrix[ctx_indices]
            
            S = np.dot(C, W.T)
            weights_array = np.array(ctx_weights)
            S_weighted = S * weights_array
            
            max_sims = np.max(S_weighted, axis=1)
            
            for i, cand_word in enumerate(actual_candidates):
                cand_to_sim[cand_word] = float(max_sims[i])
                
        return cand_to_sim

    def extract_kenlm_feature(self, candidate: str, prefix_str: str, suffix_str: str, model_lm) -> Tuple[float, float]:
        ken_ceiling = self.cfg.feature_normalization.kenlm_min_max_ceiling if (self.cfg and hasattr(self.cfg, 'feature_normalization')) else 15.0
        local_sentence_str = f"{prefix_str}{candidate}{suffix_str}".strip()
        ken_score = model_lm.score(local_sentence_str)
        norm_ken = max(0.0, (ken_score + ken_ceiling) / ken_ceiling)
        return ken_score, norm_ken

    def extract_ngram_counts_feature(self, candidate_lower: str, prev_word: str, prev_2_word: str, 
                                     next_word: str, next_2_word: str, generator) -> Tuple[int, int, int, float, float, float]:
        if self.cfg and hasattr(self.cfg, 'feature_normalization'):
            f_norm = self.cfg.feature_normalization
            norm_u = getattr(f_norm, 'log_norm_unigram', 15.0)
            norm_b = getattr(f_norm, 'log_norm_bigram', 12.0)
            norm_t = getattr(f_norm, 'log_norm_trigram', 12.0)
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

    def extract_candidates_and_features(
        self, error_word: str, sentence_words: List[str], error_idx: int, 
        error_indices: List[int], candidates: List[str], model_lm, generator
    ) -> List[Tuple[str, List[float]]]:
        
        if not candidates:
            return []

        candidates_with_features = []
        
        raw_pairs = self.cfg.confusion_pairs if (self.cfg and hasattr(self.cfg, 'confusion_pairs')) else {}
        if hasattr(raw_pairs, '__dict__'):
            confusion_dict = raw_pairs.__dict__
        elif isinstance(raw_pairs, dict):
            confusion_dict = raw_pairs
        else:
            confusion_dict = {}

        ctx_indices = []
        ctx_weights = []
        for i, w in enumerate(sentence_words):
            if i != error_idx and w not in self.stopwords and w in self.word_to_idx:
                ctx_indices.append(self.word_to_idx[w])
                weight = 1.0 / (abs(i - error_idx) + 1.0)
                ctx_weights.append(weight)

        sim_map = self.extract_similarity_features(candidates, ctx_indices, ctx_weights)

        prev_word = sentence_words[error_idx - 1].lower() if error_idx - 1 >= 0 else ""
        prev_2_word = sentence_words[error_idx - 2].lower() if error_idx - 2 >= 0 else ""
        next_word = sentence_words[error_idx + 1].lower() if error_idx + 1 < len(sentence_words) else ""
        next_2_word = sentence_words[error_idx + 2].lower() if error_idx + 2 < len(sentence_words) else ""

        prefix_str = " ".join(sentence_words[:error_idx]) + " " if error_idx > 0 else ""
        suffix_str = " " + " ".join(sentence_words[error_idx + 1:]) if error_idx + 1 < len(sentence_words) else ""

        for cand in candidates:
            features = []
            cand_lower = cand.lower()
            
            # 1. Similarity
            features.append(sim_map.get(cand, 0.0))

            # 2. N-gram counts
            _, _, _, norm_c1, norm_c2, norm_c3 = self.extract_ngram_counts_feature(
                candidate_lower=cand_lower, prev_word=prev_word, prev_2_word=prev_2_word,
                next_word=next_word, next_2_word=next_2_word, generator=generator
            )
            features.extend([norm_c1, norm_c2, norm_c3])

            # 3. Length Ratio
            features.append(self.extract_length_ratio_feature(error_word, cand))

            # 4. KenLM scores
            _, norm_ken = self.extract_kenlm_feature(cand, prefix_str, suffix_str, model_lm)
            features.append(norm_ken)

            # 5. Edit Distance
            edit_dist = compute_edit_distance_telex(error_word, cand, self.telex_dict, confusion_dict, self.cfg)
            features.append(float(edit_dist))

            candidates_with_features.append((cand, features))
            
        logger.debug(f"Feature vector compilation complete for {len(candidates)} candidates of error token '{error_word}'.")
        return candidates_with_features
    
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