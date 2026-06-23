import os
import time
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Tuple
from torch import nn
import torch
from tqdm import tqdm

from common.logger import get_logger
from spell_correction.abbr_processor import AbbreviationProcessor
from spell_correction.candidate_generator import CandidateGenerator
from spell_correction.evaluator import Evaluator
from spell_correction.feature_extractor import FeatureExtractor
from spell_correction.pipeline import extract_candidates_and_features

logger = get_logger(__name__)


class LightGBMRankerTrainer:
    
    def __init__(self, 
                 abbr_processor: AbbreviationProcessor, 
                 evaluator: Evaluator, 
                 generator: CandidateGenerator, 
                 feature_extractor: FeatureExtractor, 
                 cfg):
        self.abbr_processor = abbr_processor
        self.evaluator = evaluator
        self.generator = generator  
        self.feature_extractor = feature_extractor
        self.cfg = cfg
        self.ranker = None
        warnings.filterwarnings("ignore", category=UserWarning)
        logger.info("LightGBMRankerTrainer initialized.")

    def build_dataset(
        self,
        df_train: pd.DataFrame,
        stopwords: set,
        max_negatives: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logger.info("Building dataset and extracting features...")
        
        X_train_list = []
        y_train_list = []
        group_train_list = []

        pbar = tqdm(df_train.iterrows(), total=len(df_train), desc="Extracting Features")
        for idx, row in pbar:
            input_sent = str(row['input'])
            target_sent = str(row['target'])

            input_sent = self.abbr_processor.replace_abbreviations(input_sent)
            
            error_pairs, error_indices = self.evaluator.find_misspelled_words_and_targets(
                input_sent, 
                target_sent, 
                self.feature_extractor.word_to_idx
            )
            
            if not error_pairs:
                continue 
                
            sentence_words = input_sent.split()
                
            for i in range(len(error_indices)):
                error_word, correct_word = error_pairs[i]

                candidates_with_scores = extract_candidates_and_features(
                    error_word=error_word, 
                    sentence_words=sentence_words, 
                    error_idx=error_indices[i], 
                    error_indices=error_indices,
                    generator=self.generator,           
                    extractor=self.feature_extractor,
                    stopwords=stopwords,
                    cfg=self.cfg
                )
                
                if not candidates_with_scores:
                    continue

                positives = []
                negatives = []

                for candidate_word, feature_vector in candidates_with_scores:
                    if candidate_word == correct_word:
                        positives.append((candidate_word, feature_vector))
                    else:
                        negatives.append((candidate_word, feature_vector))

                if not positives:
                    continue 

                hard_negatives = negatives[:max_negatives]
                final_candidates = positives + hard_negatives

                group_X = []
                group_y = []

                for cand, feat in final_candidates:
                    label = 1 if cand == correct_word else 0
                    group_X.append(feat)
                    group_y.append(label)

                X_train_list.extend(group_X)
                y_train_list.extend(group_y)
                group_train_list.append(len(final_candidates))

                if error_indices[i] < len(sentence_words):
                    sentence_words[error_indices[i]] = correct_word

        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        group_train = np.array(group_train_list)

        logger.info(f"Dataset built successfully | X_train: {X_train.shape} | Query groups: {len(group_train):,}")
        return X_train, y_train, group_train

    def load_training_data(self, data_path: str) -> tuple:
        if not os.path.exists(data_path):
            logger.error(f"Feature dataset not discovered at path target: {data_path}")
            raise FileNotFoundError(f"Training data file not found at: {data_path}")
            
        logger.info(f"Loading feature matrices from: {data_path}")
        loaded_data = np.load(data_path)
        X_train = loaded_data['X_train']
        y_train = loaded_data['y_train']
        group_train = loaded_data['group_train']
        
        logger.info(f"Data matrices loaded | X_train shape: {X_train.shape} | Query Groups: {len(group_train):,}")
        return X_train, y_train, group_train

    def train(self, X_train: np.ndarray, y_train: np.ndarray, group_train: np.ndarray) -> lgb.LGBMRanker:
        logger.info("Configuring LightGBM LambdaRanker specifications...")
        
        if self.cfg and hasattr(self.cfg, 'ranker_training'):
            r_train = self.cfg.ranker_training
            lr = r_train.learning_rate
            num_leaves = r_train.num_leaves
            min_child_samples = r_train.min_child_samples
            random_state = r_train.random_state
            eval_at = r_train.eval_at
        else:
            lr = 0.05
            num_leaves = 31
            min_child_samples = 20
            random_state = 42
            eval_at = [1, 3, 5]

        self.ranker = lgb.LGBMRanker(
            objective='lambdarank',
            metric='ndcg',
            eval_at=eval_at,
            label_gain=[0, 1],
            learning_rate=lr,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            random_state=random_state
        )

        logger.info("Fitting LightGBMRanker on dataset...")
        start_time = time.time()
            
        self.ranker.fit(
            X=X_train,
            y=y_train,
            group=group_train
        )
        logger.info(f"Ranker training completed in {time.time() - start_time:.2f}s.")
        self._log_feature_importances()
        return self.ranker

    def _log_feature_importances(self):
        if self.ranker is None:
            return
            
        feature_names = [
            'ken_score', 'word2vec_sim', 'unigram_count', 
            'bigram_count', 'trigram_count', 'edit_dist', 'length_ratio'
        ]
        
        logger.info("==================== FEATURE IMPORTANCE STATISTICS ====================")
        for name, importance in zip(feature_names, self.ranker.feature_importances_):
            logger.info(f"  • {name:<15}: {importance}")
        logger.info("=======================================================================")

    def save_model(self, model_path: str):
        if self.ranker is None:
            logger.error("Ranker model is not trained yet. Aborting save.")
            raise ValueError("Model has not been trained. Cannot save checkpoint.")
        
        output_dir = os.path.dirname(model_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.ranker.booster_.save_model(model_path)
        logger.info(f"LightGBM ranker model saved successfully at: {model_path}")

    def load_model(self, model_path: str) -> lgb.LGBMRanker:
        if not os.path.exists(model_path):
            logger.error(f"LightGBM model text file not found at: {model_path}")
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
            
        logger.info(f"Loading LightGBM ranker from: {model_path}")
        bst = lgb.Booster(model_file=model_path)
        self.ranker = lgb.LGBMRanker()
        self.ranker.booster_ = bst
        logger.info("LightGBM ranker activated successfully.")
        return self.ranker