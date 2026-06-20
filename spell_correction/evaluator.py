# spell_correction/evaluator.py
import re
import numpy as np
from typing import List, Dict, Set, Tuple

class HeuristicScorer:
    """Chuyên trách việc tính toán điểm Heuristic tổng hợp dựa trên cấu hình trọng số."""
    def __init__(self, cfg=None):
        self.cfg = cfg

    def compute_score(self, norm_ken: float, norm_edit: float, length_ratio: float, 
                      norm_c1: float, norm_c2: float, norm_c3: float, norm_sim: float) -> float:
        w = self.cfg.feature_weights if self.cfg else None
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
            # Fallback heuristic mặc định
            return (0.30 * norm_ken) + (0.25 * norm_edit) + (0.10 * length_ratio) + (0.20 * norm_c2)


class Evaluator:
    def __init__(self, model_lm, config=None):
        """
        Khởi tạo Evaluator với mô hình ngôn ngữ KenLM và đối tượng cấu hình.
        
        Args:
            model_lm: Mô hình KenLM đã được load.
            config: Đối tượng SimpleNamespace chứa các tham số từ file config.yaml.
        """
        self.model_lm = model_lm
        self.config = config

    def detect_error(self, sentence: str) -> List[int]:
        """
        Phương thức dò tìm và trả về danh sách các index của từ bị nghi ngờ lỗi 
        dựa trên cơ chế lọc OOV và phân tích ngưỡng phân phối xác suất KenLM.
        """
        # Trích xuất điểm full_scores từ mô hình KenLM, bỏ qua ký tự kết thúc chuỗi </s>
        scores = list(self.model_lm.full_scores(sentence))[:-1]
        words = sentence.split()

        error_indices: Set[int] = set()
        valid_probs: List[float] = []
        valid_indices: List[int] = []

        for i, (prob, length, is_oov) in enumerate(scores):
            if i >= len(words):
                continue

            # 1. Lọc lập tức các từ Out Of Vocabulary (Từ lạ, teencode biến dị, lỗi nặng)
            if is_oov:
                error_indices.add(i)
                
            valid_probs.append(prob)
            valid_indices.append(i)

        # 2. Áp dụng cơ chế ngưỡng động kết hợp ngưỡng cứng từ Config (Heuristic KenLM)
        if valid_probs:
            mean_prob = np.mean(valid_probs)
            std_prob = np.std(valid_probs) 
            
            # Đọc tham số từ SimpleNamespace (kèm cấu hình fallback an toàn)
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
                # Bất thường động: Xác suất thấp hơn hẳn trung bình câu VÀ vượt ngưỡng trần cho phép
                is_anomaly = (prob < dynamic_threshold) and (prob < hard_ceiling)
                # Lỗi tuyệt đối: Xác suất quá thấp, chắc chắn sai ngữ cảnh
                is_absolute_error = (prob < hard_floor)
                
                if is_anomaly or is_absolute_error:
                    error_indices.add(idx)

        return sorted(list(error_indices))

    @staticmethod
    def find_misspelled_words_and_targets(input_sentence: str, target_sentence: str, word_to_idx: Dict[str, int]) -> Tuple[List[tuple], List[int]]:
        """Tìm các cặp từ lỗi thực tế dựa trên nhãn Target chuẩn."""
        input_tokens = input_sentence.split()
        target_tokens = target_sentence.split()
        
        if len(input_tokens) != len(target_tokens):
            return [], []

        error_indices = []
        pairs = []
        for i in range(len(input_tokens)):
            if input_tokens[i] != target_tokens[i] and target_tokens[i] in word_to_idx:
                pairs.append((input_tokens[i], target_tokens[i]))
                error_indices.append(i)
        return pairs, error_indices

    @staticmethod
    def calculate_f05(tp: int, fp: int, fn: int) -> float:
        """Tính điểm F0.5 ưu tiên Precision."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall) if (precision + recall) > 0 else 0
    
    def evaluate_error_detection(self, validation_df, teencode_engine):
        """Quét qua tập dữ liệu validation để đánh giá độ chính xác giai đoạn phát hiện lỗi (Detect)."""
        import pandas as pd
        from tqdm import tqdm

        total_TP = 0
        total_FP = 0
        total_FN = 0

        for _, row in tqdm(validation_df.iterrows(), total=len(validation_df), desc="Đánh giá độ bắt lỗi trên tập Valid"):
            input_sent = str(row['input'])
            target_sent = str(row['target'])

            # Sử dụng đối tượng teencode engine đã truyền từ ngoài
            input_sent = teencode_engine.replace_abbreviations(input_sent)

            # Gọi phương thức static tìm vị trí lỗi gốc nhãn target
            _, error_indices = self.find_misspelled_words_and_targets(input_sent, target_sent, self.model_lm)

            if not error_indices:
                continue 

            # Dự đoán lỗi qua phương thức detect_error nội bộ của Evaluator
            error_detect = self.detect_error(input_sent)

            set_true = set(error_indices)
            set_pred = set(error_detect)

            # Tính toán ma trận lỗi câu hiện tại
            total_TP += len(set_true & set_pred)
            total_FP += len(set_pred - set_true)
            total_FN += len(set_true - set_pred)

        # Tính toán các chỉ số cuối cùng
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        f05_score = self.calculate_f05(total_TP, total_FP, total_FN)

        print("\n" + "="*20 + " KẾT QUẢ GIAI ĐOẠN PHÁT HIỆN LỖI (DETECTION) " + "="*20)
        print(f" Số lỗi thực tế (Actual) : {total_TP + total_FN}")
        print(f" Số lỗi dự đoán (Predict): {total_TP + total_FP}")
        print(f"  + Bắt trúng (True Positives) : {total_TP}")
        print(f"  + Bắt nhầm (False Positives): {total_FP}")
        print(f"  + Bỏ sót (False Negatives)  : {total_FN}\n")
        print(f" Precision (Độ chính xác bắt lỗi): {precision * 100:.2f}%")
        print(f" Recall (Độ phủ bắt lỗi)         : {recall * 100:.2f}%")
        print(f" F0.5 (Điểm đánh giá cuối cùng)  : {f05_score * 100:.2f}%")
        print("=" * 76 + "\n")

        return {"precision": precision, "recall": recall, "f05": f05_score}

    def evaluate_ranking_performance(
        self, validation_df, teencode_engine, extractor_engine, generator, ranker
    ):
        """
        Quét qua tập validation để đánh giá khả năng sắp xếp ứng viên của LightGBM Ranker
        thông qua các chỉ số MRR (Mean Reciprocal Rank) và Hit@K Rate.
        """
        import numpy as np
        from tqdm import tqdm

        # Khởi tạo các bộ đếm metrics
        count_error_all = 0
        mrr_sum = 0.0
        hit_at_1 = 0
        hit_at_3 = 0
        hit_at_5 = 0

        for _, row in tqdm(validation_df.iterrows(), total=len(validation_df), desc="Đánh giá Ranking (MRR, Hit@K)"):
            input_sent = str(row['input'])
            target_sent = str(row['target'])

            # 1. Tiền xử lý teencode
            input_sent = teencode_engine.replace_abbreviations(input_sent)

            input_tokens = input_sent.split()
            target_tokens = target_sent.split()

            # Bảo vệ cấu trúc: Tránh lệch index giữa hai chuỗi văn bản
            if len(input_tokens) != len(target_tokens):
                continue

            # 2. Dò tìm các vị trí nghi ngờ lỗi thông qua KenLM
            error_detect = self.detect_error(input_sent)

            for actual_error_idx in error_detect:
                if actual_error_idx >= len(target_tokens):
                    continue
                    
                correct_word = target_tokens[actual_error_idx]
                error_word = input_tokens[actual_error_idx]
                
                # Lọc nhiễu: Bỏ qua nếu từ gốc đã đúng, từ nhãn nằm ngoài vocab, hoặc là chữ số
                if (error_word == correct_word or 
                    correct_word not in extractor_engine.word_to_idx or 
                    correct_word.isdigit()):
                    continue

                # 3. Tạo danh sách thô các ứng viên tiềm năng từ bộ generator (Đọc k từ Config)
                if self.config and hasattr(self.config, 'candidate_generation'):
                    k_candidates = self.config.candidate_generation.top_k_raw
                else:
                    k_candidates = 2

                raw_candidates = generator.lookup(error_word, extractor_engine.word_to_idx, k=k_candidates)
                if not raw_candidates:
                    count_error_all += 1 
                    continue

                # 4. Trích xuất ma trận đặc trưng cho tập ứng viên
                candidates_with_features = extractor_engine.extract_candidates_and_features(
                    error_word=error_word,
                    sentence_words=input_tokens,
                    error_idx=actual_error_idx,
                    error_indices=error_detect,
                    candidates=raw_candidates,
                    model_lm=self.model_lm,
                    generator=generator
                )

                if not candidates_with_features:
                    count_error_all += 1 
                    continue

                count_error_all += 1

                # Tách riêng danh sách từ và ma trận X đầu vào cho mô hình
                cand_words = [item[0] for item in candidates_with_features]
                X_infer = np.array([item[1] for item in candidates_with_features])
                
                # 5. Sử dụng LightGBM Ranker để dự đoán điểm thứ hạng
                scores = ranker.predict(X_infer)

                # Sắp xếp lại danh sách ứng viên theo điểm số giảm dần từ Ranker
                ranked_candidates = [word for _, word in sorted(zip(scores, cand_words), reverse=True)]

                # 6. Tính toán MRR và Hit@K dựa trên thứ hạng thực tế của correct_word
                try:
                    # Trả về chỉ mục từ 0 nên cần cộng 1 để ra thứ hạng (Rank) thực tế
                    rank = ranked_candidates.index(correct_word) + 1
                    
                    mrr_sum += 1.0 / rank
                    
                    if rank == 1:
                        hit_at_1 += 1
                    if rank <= 3:
                        hit_at_3 += 1
                    if rank <= 5:
                        hit_at_5 += 1
                        
                except ValueError:
                    # Xử lý ngoại lệ nếu correct_word không xuất hiện trong top ứng viên (Recall thất bại)
                    pass 

        # Tránh lỗi chia cho 0 nếu tập valid không ghi nhận lỗi hợp lệ nào
        total_eval = max(1, count_error_all)
        
        metrics = {
            "mrr": mrr_sum / total_eval,
            "hit_at_1": hit_at_1 / total_eval,
            "hit_at_3": hit_at_3 / total_eval,
            "hit_at_5": hit_at_5 / total_eval,
            "total_errors": count_error_all
        }

        # Báo cáo kết quả hiển thị ra terminal
        print("\n" + "="*20 + " BÁO CÁO RANKING METRICS TRÊN TẬP VALID " + "="*20)
        print(f" • Tổng số lỗi đưa vào đánh giá : {metrics['total_errors']:,}")
        print(f" • MRR (Mean Reciprocal Rank)    : {metrics['mrr']:.4f}")
        print(f" • Hit@1 (Top-1 Accuracy)        : {metrics['hit_at_1'] * 100:.2f}%")
        print(f" • Hit@3 (Có trong Top 3)        : {metrics['hit_at_3'] * 100:.2f}%")
        print(f" • Hit@5 (Có trong Top 5)        : {metrics['hit_at_5'] * 100:.2f}%")
        print("=" * 80 + "\n")

        return metrics

    def evaluate_word_accuracy(
        self, validation_df, teencode_engine, pipeline_correct_fn, word_to_idx
    ):
        """
        Đánh giá độ chính xác cấp độ từ (Word Accuracy) của các từ sai được sửa 
        khi chạy qua toàn bộ Pipeline thực tế.
        """
        from tqdm import tqdm

        count_error = 0
        count_correct = 0

        for _, row in tqdm(validation_df.iterrows(), total=len(validation_df), desc="Đánh giá Word Accuracy trên tập Valid"):
            input_sent = str(row['input'])
            target_sent = str(row['target'])

            # 1. Tiền xử lý teencode
            input_sent = teencode_engine.replace_abbreviations(input_sent)

            # 2. Kiểm tra câu gốc có lỗi thực tế theo nhãn target hay không
            error_pairs, error_indices = self.find_misspelled_words_and_targets(
                input_sent, target_sent, word_to_idx
            )
            if not error_pairs:
                continue  # Bỏ qua câu không có lỗi

            # 3. Đưa CẢ CÂU qua toàn bộ pipeline thực tế để sửa lỗi
            fixed_sentence_str = pipeline_correct_fn(input_sent)
            
            fixed_tokens = fixed_sentence_str.split()
            input_tokens = input_sent.split()
            target_tokens = target_sent.split()

            # Bảo vệ cấu trúc câu
            if not (len(input_tokens) == len(target_tokens) == len(fixed_tokens)):
                continue

            # 4. Duyệt qua các vị trí mô hình phát hiện lỗi để chấm điểm khâu sửa (Correction)
            error_detect = self.detect_error(input_sent)
            for actual_error_idx in error_detect:
                if actual_error_idx >= len(target_tokens):
                    continue

                # Bỏ qua nếu: máy bắt nhầm (từ gốc vốn đã đúng), hoặc từ target OOV
                if (input_tokens[actual_error_idx] == target_tokens[actual_error_idx] or 
                    target_tokens[actual_error_idx] not in word_to_idx):
                    continue

                count_error += 1

                correct_word = target_tokens[actual_error_idx]
                fixed_word = fixed_tokens[actual_error_idx]

                # Kiểm tra xem từ mô hình sửa có khớp chính xác với nhãn Target không
                if fixed_word == correct_word:
                    count_correct += 1
                    
        # Sửa lỗi logic chia sai biến đếm ở mã nguồn cũ
        total_errors = max(1, count_error)
        accuracy = count_correct / total_errors
            
        print("\n" + "="*20 + " KẾT QUẢ ĐÁNH GIÁ WORD ACCURACY " + "="*20)
        print(f" • Tổng số từ lỗi mô hình thực xử lý : {total_errors:,}")
        print(f" • Số từ sửa trúng đáp án (Correct)   : {count_correct:,}")
        print(f" • Word Accuracy (Độ chính xác từ)  : {accuracy * 100:.2f}%")
        print("=" * 72 + "\n")

        return {"word_accuracy": accuracy, "total_processed_errors": count_error}

    def evaluate_end_to_end(self, validation_df, teencode_engine, pipeline_correct_fn):
        """
        Đánh giá hiệu năng toàn cục End-to-End của hệ thống sửa lỗi chính tả
        thông qua các chỉ số WER (Word Error Rate) và SER (Sentence Error Rate/Exact Match).
        """
        from tqdm import tqdm

        total_word_errors = 0
        total_reference_words = 0
        exact_match_sentences = 0  # Đếm số câu sửa hoàn hảo 100%

        for _, row in tqdm(validation_df.iterrows(), total=len(validation_df), desc="Đánh giá End-to-End (WER/SER)"):
            input_sent = str(row['input'])
            target_sent = str(row['target'])

            # 1. Tiền xử lý chuẩn hóa teencode/viết tắt
            input_sent = teencode_engine.replace_abbreviations(input_sent)

            # 2. Đưa toàn bộ câu chạy qua Pipeline sửa lỗi thực tế
            fixed_sentence_str = pipeline_correct_fn(input_sent)
            
            # Tokenize (Tách chuỗi từ theo khoảng trắng)
            target_tokens = target_sent.split()
            fixed_tokens = fixed_sentence_str.split()
            
            # Đếm tổng số từ tham chiếu chuẩn (N)
            total_reference_words += len(target_tokens)
            
            # 3. Tính toán số từ bị sai sót (S) bằng zip để tránh lỗi IndexError nếu lệch độ dài
            errors_in_sentence = sum(1 for f, t in zip(fixed_tokens, target_tokens) if f != t)
            
            # Cộng thêm phần chênh lệch độ dài chuỗi từ (nếu mô hình làm mất/thêm từ lãng phí)
            errors_in_sentence += abs(len(fixed_tokens) - len(target_tokens))
            
            total_word_errors += errors_in_sentence 
            
            # Nếu câu không chứa bất kỳ từ sai nào -> Sửa đổi hoàn hảo
            if errors_in_sentence == 0:
                exact_match_sentences += 1

        # 4. Tính toán các chỉ số phần trăm cuối cùng
        total_sentences = len(validation_df)
        total_ref_words = max(1, total_reference_words)
        
        overall_wer = (total_word_errors / total_ref_words) * 100
        overall_ser = (exact_match_sentences / max(1, total_sentences)) * 100

        # Báo cáo thống kê ra màn hình console
        print("\n" + "="*20 + " BÁO CÁO TOÀN CỤC END-TO-END " + "="*20)
        print(f" • Tổng số câu đưa vào đánh giá      : {total_sentences:,}")
        print(f" • Tổng số từ trong tập đích (N)     : {total_reference_words:,}")
        print(f" • Tổng số từ lỗi mô hình để sót/sinh : {total_word_errors:,}")
        print(f" • WER (Word Error Rate - Càng thấp càng tốt) : {overall_wer:.2f}%") 
        print(f" • SER (Sentence Exact Match - Càng cao càng tốt): {overall_ser:.2f}%") 
        print("=" * 70 + "\n")

        return {
            "wer": overall_wer,
            "ser": overall_ser,
            "total_word_errors": total_word_errors,
            "total_reference_words": total_reference_words
        }
    
# -*- coding: utf-8 -*-
import pandas as pd
from tqdm import tqdm

class Visualizer:
    def __init__(self, pipeline, teencode_engine, evaluator, word_to_idx):
        """
        Class hỗ trợ phân loại và đánh giá trực quan kết quả sửa lỗi của hệ thống.
        """
        self.pipeline = pipeline
        self.teencode_engine = teencode_engine
        self.evaluator = evaluator
        self.word_to_idx = word_to_idx

    def analyze_predictions(self, validation_df, num_samples=200):
        """
        Quét qua tập dữ liệu kiểm thử, chạy pipeline và phân loại câu/từ.
        :param validation_df: DataFrame chứa tập Validation (có cột 'input' và 'target')
        :param num_samples: Số lượng mẫu muốn chạy phân tích trực quan
        :return: (exact_sentences_df, error_sentence_df, error_words_df)
        """
        exact_sentences = pd.DataFrame(columns=['Input', 'Fixed', 'Target'])
        error_sentence = pd.DataFrame(columns=['Input', 'Fixed', 'Target'])
        error_words = pd.DataFrame(columns=['Error', 'Correct'])

        # Giới hạn số lượng mẫu để tối ưu thời gian chạy trực quan
        target_df = validation_df.head(num_samples)

        for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc="Phân tích trực quan mẫu"): 
            input_sent = str(row['input'])
            target_sent = str(row['target'])

            # Tiền xử lý teencode
            cleaned_input = self.teencode_engine.replace_abbreviations(input_sent)

            # Quét tìm nhãn lỗi gốc từ cặp câu (cleaned_input, target_sent)
            _, error_indices = self.evaluator.find_misspelled_words_and_targets(
                cleaned_input, target_sent, self.word_to_idx
            )

            # Chạy qua Pipeline chính đã được đóng gói
            fixed_sentence_str = self.pipeline.correct_sentence(cleaned_input)

            target_tokens = target_sent.split()
            fixed_tokens = fixed_sentence_str.split()

            if len(target_tokens) != len(fixed_tokens):
                continue

            errors_in_sentence = 0
            for idx in error_indices:
                if fixed_tokens[idx] != target_tokens[idx]:
                    errors_in_sentence += 1
                    error_words.loc[error_words.shape[0]] = [fixed_tokens[idx], target_tokens[idx]]

            # Phân loại câu sửa đúng hoàn toàn vs câu vẫn còn sót lỗi
            if errors_in_sentence == 0 and not error_indices:
                exact_sentences.loc[exact_sentences.shape[0]] = [input_sent, fixed_sentence_str, target_sent]
            
            if errors_in_sentence != 0:
                error_sentence.loc[error_sentence.shape[0]] = [input_sent, fixed_sentence_str, target_sent]

        return exact_sentences, error_sentence, error_words