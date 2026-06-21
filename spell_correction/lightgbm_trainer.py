import os
import time
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Tuple
from tqdm import tqdm

from common.logger import get_logger
from spell_correction.abbr_processor import AbbreviationProcessor
from spell_correction.evaluator import Evaluator
from spell_correction.feature_extractor import FeatureExtractor

logger = get_logger(__name__)


class LightGBMRankerTrainer:
    """Class quản lý toàn bộ vòng đời cấu hình, huấn luyện, trích xuất dữ liệu 
    và đóng gói mô hình xếp hạng ứng viên (LightGBM LambdaRanker)."""
    
    def __init__(self, abbr_processor: AbbreviationProcessor, evaluator: Evaluator, feature_extractor: FeatureExtractor, cfg):
        self.abbr_processor = abbr_processor
        self.evaluator = evaluator
        self.feature_extractor = feature_extractor
        self.cfg = cfg
        self.ranker = None
        warnings.filterwarnings("ignore", category=UserWarning)
        logger.info("LightGBMRankerTrainer initialized.")

    def build_dataset(
        self,
        df_train: pd.DataFrame,
        max_negatives: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trích xuất đặc trưng và xây dựng tập dữ liệu huấn luyện (X, y, group) từ DataFrame.
        """
        logger.info("Bắt đầu xây dựng tập dữ liệu huấn luyện (Feature Extraction)...")
        
        X_train_list = []
        y_train_list = []
        group_train_list = []

        for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Trích xuất đặc trưng"):
            input_sent = str(row['input'])
            target_sent = str(row['target'])

            # Tiền xử lý chữ viết tắt
            input_sent = self.abbr_processor.replace_abbreviations(input_sent)
            
            # Tìm lỗi sai và vị trí
            error_pairs, error_indices = self.evaluator.find_misspelled_words_and_targets(input_sent, target_sent)
            
            if not error_pairs:
                continue 
                
            # Tách thành mảng token ngay từ đầu để truyền vào hàm
            sentence_words = input_sent.split()
                
            for i in range(len(error_indices)):
                error_word, correct_word = error_pairs[i]

                # Trích xuất ứng viên và vector đặc trưng
                candidates_with_scores = self.feature_extractor.extract_candidates_and_features(
                    error_word, 
                    sentence_words, 
                    error_indices[i], 
                    error_indices
                )
                
                if not candidates_with_scores:
                    continue

                # Tách riêng từ đúng (Positive) và từ sai (Negative)
                positives = []
                negatives = []

                for candidate_word, feature_vector in candidates_with_scores:
                    if candidate_word == correct_word:
                        positives.append((candidate_word, feature_vector))
                    else:
                        negatives.append((candidate_word, feature_vector))

                # Nếu không lấy được từ đúng -> Bỏ qua toàn bộ Group lỗi này
                if not positives:
                    continue 

                # Lấy tối đa N từ sai có điểm Heuristic cao nhất (khó phân biệt nhất)
                hard_negatives = negatives[:max_negatives]

                # Gộp từ đúng và các từ sai khó nhất lại thành danh sách training cho group này
                final_candidates = positives + hard_negatives

                group_X = []
                group_y = []

                for cand, feat in final_candidates:
                    label = 1 if cand == correct_word else 0
                    group_X.append(feat)
                    group_y.append(label)

                # Cập nhật vào mảng tổng của mô hình
                X_train_list.extend(group_X)
                y_train_list.extend(group_y)

                # Báo cho LightGBM biết Group này có bao nhiêu ứng viên
                group_train_list.append(len(final_candidates))

                # In-place Update cho từ lỗi tiếp theo
                if error_indices[i] < len(sentence_words):
                    sentence_words[error_indices[i]] = correct_word

        # Chuyển đổi sang numpy array để tương thích trực tiếp với LightGBM
        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        group_train = np.array(group_train_list)

        logger.info(
            f"Dataset xây dựng thành công | X_train: {X_train.shape} | "
            f"Tổng số query groups: {len(group_train):,}"
        )
        return X_train, y_train, group_train

    def load_training_data(self, data_path: str) -> tuple:
        """Đọc tập dữ liệu trích xuất đặc trưng (.npz) được chuẩn bị sẵn."""
        if not os.path.exists(data_path):
            logger.error(f"Feature dataset not discovered at path target: {data_path}")
            raise FileNotFoundError(f"❌ Không tìm thấy tệp dữ liệu huấn luyện tại: {data_path}")
            
        logger.info(f"Loading engineered feature matrix files from: {data_path}")
        loaded_data = np.load(data_path)
        X_train = loaded_data['X_train']
        y_train = loaded_data['y_train']
        group_train = loaded_data['group_train']
        
        logger.info(f"Data matrices successfully loaded | X_train shape: {X_train.shape} | Total Query Groups: {len(group_train):,}")
        return X_train, y_train, group_train

    def train(self, X_train: np.ndarray, y_train: np.ndarray, group_train: np.ndarray) -> lgb.LGBMRanker:
        """Khởi tạo cấu hình và kích hoạt chu trình huấn luyện mô hình Ranker."""
        logger.info("Configuring LightGBM LambdaRanker core specifications...")
        
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

        logger.info("Fitting LightGBMRanker on feature datasets...")
        start_time = time.time()
        self.ranker.fit(
            X=X_train,
            y=y_train,
            group=group_train
        )
        logger.info(f"LightGBM ranking estimator optimization completed in {time.time() - start_time:.2f}s.")
        self._log_feature_importances()
        return self.ranker

    def _log_feature_importances(self):
        """Phương thức nội bộ: Thống kê và hiển thị mức độ đóng góp của từng đặc trưng."""
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
        """Lưu trữ mô hình đã huấn luyện ra file text của LightGBM."""
        if self.ranker is None:
            logger.error("Ranker estimator has no optimized parameter trees. Aborting.")
            raise ValueError("❌ Mô hình chưa được huấn luyện. Không thể lưu checkpoint.")
        
        output_dir = os.path.dirname(model_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.ranker.booster_.save_model(model_path)
        logger.info(f"LightGBM ranker booster model trees secured text-file format at: {model_path}")

    def load_model(self, model_path: str) -> lgb.LGBMRanker:
        """Tải mô hình từ checkpoint có sẵn để phục vụ cho Inference/Evaluation nhanh."""
        if not os.path.exists(model_path):
            logger.error(f"LightGBM configuration text file not found at: {model_path}")
            raise FileNotFoundError(f"❌ Không tìm thấy checkpoint mô hình tại: {model_path}")
            
        logger.info(f"Loading pre-compiled LightGBM ranker assets from: {model_path}")
        bst = lgb.Booster(model_file=model_path)
        self.ranker = lgb.LGBMRanker()
        self.ranker.booster_ = bst
        logger.info("LightGBM ranker components re-activated successfully.")
        return self.ranker