import time
import numpy as np
from typing import List, Dict, Set, Tuple
from tqdm import tqdm

from common.logger import get_logger
from spell_correction.pipeline import extract_candidates_and_features

logger = get_logger(__name__)

class Evaluator:
    def __init__(self, model_lm, config=None):
        self.model_lm = model_lm
        self.config = config
        logger.info("Evaluator instance successfully mounted.")

    def detect_error(self, sentence: str) -> List[int]:
        scores = list(self.model_lm.full_scores(sentence))[:-1]
        words = sentence.split()

        error_indices: Set[int] = set()
        valid_probs: List[float] = []
        valid_indices: List[int] = []

        for i, (prob, length, is_oov) in enumerate(scores):
            if i >= len(words):
                continue

            if is_oov:
                error_indices.add(i)
                
            valid_probs.append(prob)
            valid_indices.append(i)

        if valid_probs:
            mean_prob = np.mean(valid_probs)
            std_prob = np.std(valid_probs) 
            
            if self.config and hasattr(self.config, 'detector_heuristics'):
                alpha = self.config.detector_heuristics.alpha
                hard_ceiling = self.config.detector_heuristics.hard_ceiling
                hard_floor = self.config.detector_heuristics.hard_floor
            else:
                alpha = 4.6
                hard_ceiling = -3.9
                hard_floor = -6.26
            
            dynamic_threshold = mean_prob - (alpha * std_prob)

            for idx, prob in zip(valid_indices, valid_probs):
                is_anomaly = (prob < dynamic_threshold) and (prob < hard_ceiling)
                is_absolute_error = (prob < hard_floor)
                
                if is_anomaly or is_absolute_error:
                    error_indices.add(idx)

        return sorted(list(error_indices))

    @staticmethod
    def find_misspelled_words_and_targets(input_sentence: str, target_sentence: str, word_to_idx: Dict[str, int]) -> Tuple[List[tuple], List[int]]:
        input_tokens = input_sentence.split()
        target_tokens = target_sentence.split()
        
        if len(input_tokens) != len(target_tokens):
            return [], []

        error_indices = []
        pairs = []
        for i in range(len(input_tokens)):
            if input_tokens[i] != target_tokens[i]:
                if word_to_idx is not None and hasattr(word_to_idx, 'stoi') and target_tokens[i] not in word_to_idx.stoi:
                    continue
                elif isinstance(word_to_idx, dict) and target_tokens[i] not in word_to_idx:
                    continue
                pairs.append((input_tokens[i], target_tokens[i]))
                error_indices.append(i)
        return pairs, error_indices

    @staticmethod
    def calculate_f05(tp: int, fp: int, fn: int) -> float:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall) if (precision + recall) > 0 else 0
    
    def evaluate_error_detection(self, validation_df, abbr_engine):
        logger.info("Executing Error Detection Performance Sweep...")
        start_time = time.time()
        
        total_TP = 0
        total_FP = 0
        total_FN = 0

        pbar = tqdm(validation_df.iterrows(), total=len(validation_df), desc="Evaluating Error Detection", leave=False)
        for _, row in pbar:
            input_sent = str(row['input'])
            target_sent = str(row['target'])

            input_sent = abbr_engine.replace_abbreviations(input_sent)
            _, error_indices = self.find_misspelled_words_and_targets(input_sent, target_sent, None)

            if not error_indices:
                continue 

            error_detect = self.detect_error(input_sent)

            set_true = set(error_indices)
            set_pred = set(error_detect)

            total_TP += len(set_true & set_pred)
            total_FP += len(set_pred - set_true)
            total_FN += len(set_true - set_pred)

        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        f05_score = self.calculate_f05(total_TP, total_FP, total_FN)

        elapsed = time.time() - start_time
        logger.info(f"Detection evaluation cycle finished in {elapsed:.2f}s.")

        logger.info("==================== ERROR DETECTION METRICS ====================")
        logger.info(f" • Actual Errors (Target)  : {total_TP + total_FN:,}")
        logger.info(f" • Predicted Errors (Model): {total_TP + total_FP:,}")
        logger.info(f"   + True Positives (TP)   : {total_TP:,}")
        logger.info(f"   + False Positives (FP)  : {total_FP:,}")
        logger.info(f"   + False Negatives (FN)  : {total_FN:,}")
        logger.info(f" • Precision               : {precision * 100:.2f}%")
        logger.info(f" • Recall                  : {recall * 100:.2f}%")
        logger.info(f" • F0.5 Score              : {f05_score * 100:.2f}%")
        logger.info("=================================================================")

        return {"precision": precision, "recall": recall, "f05": f05_score}

    def evaluate_ranking_performance(self, validation_df, abbr_engine, extractor_engine, generator, ranker, stopwords):
        logger.info("Executing Ranker Performance Sweep (MRR / Hit@K)...")
        start_time = time.time()

        count_error_all = 0
        mrr_sum = 0.0
        hit_at_1 = 0
        hit_at_3 = 0
        hit_at_5 = 0

        pbar = tqdm(validation_df.iterrows(), total=len(validation_df), desc="Evaluating Ranker", leave=False)
        for _, row in pbar:
            input_sent = str(row['input'])
            target_sent = str(row['target'])

            input_sent = abbr_engine.replace_abbreviations(input_sent)
            input_tokens = input_sent.split()
            target_tokens = target_sent.split()

            if len(input_tokens) != len(target_tokens):
                continue

            error_detect = self.detect_error(input_sent)

            for actual_error_idx in error_detect:
                if actual_error_idx >= len(target_tokens):
                    continue
                    
                correct_word = target_tokens[actual_error_idx]
                error_word = input_tokens[actual_error_idx]
                
                if (error_word == correct_word or 
                    correct_word not in extractor_engine.word_to_idx or 
                    correct_word.isdigit()):
                    continue

                candidates_with_features = extract_candidates_and_features(
                    error_word=error_word,
                    sentence_words=input_tokens,
                    error_idx=actual_error_idx,
                    error_indices=error_detect,
                    generator=generator,
                    extractor=extractor_engine,
                    stopwords=stopwords,
                    cfg=self.config
                )

                if not candidates_with_features:
                    count_error_all += 1 
                    continue

                count_error_all += 1

                cand_words = [item[0] for item in candidates_with_features]
                X_infer = np.array([item[1] for item in candidates_with_features])
                
                scores = ranker.predict(X_infer)
                ranked_candidates = [word for _, word in sorted(zip(scores, cand_words), reverse=True)]

                try:
                    rank = ranked_candidates.index(correct_word) + 1
                    mrr_sum += 1.0 / rank
                    
                    if rank == 1: hit_at_1 += 1
                    if rank <= 3: hit_at_3 += 1
                    if rank <= 5: hit_at_5 += 1
                        
                except ValueError:
                    pass 

        total_eval = max(1, count_error_all)
        metrics = {
            "mrr": mrr_sum / total_eval,
            "hit_at_1": hit_at_1 / total_eval,
            "hit_at_3": hit_at_3 / total_eval,
            "hit_at_5": hit_at_5 / total_eval,
            "total_errors": count_error_all
        }

        elapsed = time.time() - start_time
        logger.info(f"Ranking evaluation sweep complete in {elapsed:.2f}s.")

        logger.info("==================== RANKER CANDIDATE METRICS ====================")
        logger.info(f" • Total Evaluated Errors : {metrics['total_errors']:,}")
        logger.info(f" • MRR                    : {metrics['mrr']:.4f}")
        logger.info(f" • Hit@1 (Top-1 Accuracy) : {metrics['hit_at_1'] * 100:.2f}%")
        logger.info(f" • Hit@3                  : {metrics['hit_at_3'] * 100:.2f}%")
        logger.info(f" • Hit@5                  : {metrics['hit_at_5'] * 100:.2f}%")
        logger.info("==================================================================")

        return metrics

    def evaluate_word_accuracy(self, validation_df, abbr_engine, pipeline_correct_fn, stopwords, word_to_idx):
        logger.info("Evaluating Pipeline Word Accuracy metrics...")
        count_error = 0
        count_correct = 0

        pbar = tqdm(validation_df.iterrows(), total=len(validation_df), desc="Evaluating Word Accuracy", leave=False)
        for _, row in pbar:
            input_sent = str(row['input'])
            target_sent = str(row['target'])

            error_pairs, error_indices = self.find_misspelled_words_and_targets(input_sent, target_sent, word_to_idx)
            if not error_pairs:
                continue 

            fixed_sentence_str = pipeline_correct_fn(input_sent, stopwords)
            
            fixed_tokens = fixed_sentence_str.split()
            input_tokens = input_sent.split()
            target_tokens = target_sent.split()

            if not (len(input_tokens) == len(target_tokens) == len(fixed_tokens)):
                continue

            error_detect = self.detect_error(input_sent)
            for actual_error_idx in error_detect:
                if actual_error_idx >= len(target_tokens):
                    continue

                if (input_tokens[actual_error_idx] == target_tokens[actual_error_idx] or 
                    target_tokens[actual_error_idx] not in word_to_idx):
                    continue

                count_error += 1
                correct_word = target_tokens[actual_error_idx]
                fixed_word = fixed_tokens[actual_error_idx]

                if fixed_word == correct_word:
                    count_correct += 1
                    
        total_errors = max(1, count_error)
        accuracy = count_correct / total_errors
            
        logger.info("==================== PIPELINE WORD ACCURACY ====================")
        logger.info(f" • Total Evaluated Error Words : {total_errors:,}")
        logger.info(f" • Correctly Fixed Words       : {count_correct:,}")
        logger.info(f" • Word Accuracy               : {accuracy * 100:.2f}%")
        logger.info("================================================================")

        return {"word_accuracy": accuracy, "total_processed_errors": count_error}

    def evaluate_end_to_end(self, validation_df, abbr_engine, pipeline_correct_fn, stopwords):
        logger.info("Launching final End-to-End system validation sweep...")
        
        total_word_errors = 0
        total_reference_words = 0
        exact_match_sentences = 0

        pbar = tqdm(validation_df.iterrows(), total=len(validation_df), desc="Evaluating End-to-End", leave=False)
        for _, row in pbar:
            input_sent = str(row['input'])
            target_sent = str(row['target'])

            input_sent = abbr_engine.replace_abbreviations(input_sent)
            fixed_sentence_str = pipeline_correct_fn(input_sent, stopwords)
            
            target_tokens = target_sent.split()
            fixed_tokens = fixed_sentence_str.split()
            
            total_reference_words += len(target_tokens)
            errors_in_sentence = sum(1 for f, t in zip(fixed_tokens, target_tokens) if f != t)
            errors_in_sentence += abs(len(fixed_tokens) - len(target_tokens))
            
            total_word_errors += errors_in_sentence 
            
            if errors_in_sentence == 0:
                exact_match_sentences += 1

        total_sentences = len(validation_df)
        total_ref_words = max(1, total_reference_words)
        
        overall_wer = (total_word_errors / total_ref_words) * 100
        overall_ser = (exact_match_sentences / max(1, total_sentences)) * 100

        logger.info("==================== END-TO-END PIPELINE PERFORMANCE ====================")
        logger.info(f" • Total Evaluated Sentences : {total_sentences:,}")
        logger.info(f" • Total Reference Tokens    : {total_reference_words:,}")
        logger.info(f" • Total Remaining Errors    : {total_word_errors:,}")
        logger.info(f" • WER (Word Error Rate)     : {overall_wer:.2f}%") 
        logger.info(f" • SER (Sentence Error Rate) : {100 - overall_ser:.2f}% (Exact Match: {overall_ser:.2f}%)") 
        logger.info("=========================================================================")

        return {
            "wer": overall_wer,
            "ser": overall_ser,
            "total_word_errors": total_word_errors,
            "total_reference_words": total_reference_words
        }