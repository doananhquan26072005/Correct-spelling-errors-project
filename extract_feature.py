import math
import numpy as np

# Đặc trưng similarity từ model skip-gram
def extract_similarity_features(valid_candidates, ctx_indices, ctx_weights, word_to_idx, norm_embedding_matrix):
    cand_to_sim = {}
    if valid_candidates and ctx_indices:
        cand_indices = [word_to_idx[c] for c in valid_candidates]
        
        C = norm_embedding_matrix[cand_indices]
        W = norm_embedding_matrix[ctx_indices]
        
        S = np.dot(C, W.T)
        weights_array = np.array(ctx_weights)
        S_weighted = S * weights_array
        
        max_sims = np.max(S_weighted, axis=1)
        cand_to_sim = {cand: max_sims[i] for i, cand in enumerate(valid_candidates)}
    return cand_to_sim

# Đặc trưng log-probability từ model kenlm và chuẩn hóa
def extract_kenlm_feature(candidate, prefix_str, suffix_str, model_lm):

    local_sentence_str = f"{prefix_str}{candidate}{suffix_str}".strip()
    ken_score = model_lm.score(local_sentence_str)
    norm_ken = max(0.0, (ken_score + 15.0) / 15.0)
    return ken_score, norm_ken

# Đặc trưng tần suất của unigram, bigram, trigram
def extract_ngram_counts_feature(candidate_lower, prev_word, prev_2_word, next_word, next_2_word, counts_1, counts_2, counts_3):
    # Unigram
    c1 = counts_1.get(candidate_lower, 0)
    
    # Bigram gộp (Trái + Phải)
    c2_left  = counts_2.get(f"{prev_word} {candidate_lower}", 0)
    c2_right = counts_2.get(f"{candidate_lower} {next_word}", 0)
    c2 = c2_left + c2_right
    
    # Trigram gộp (Tâm + Trái + Phải)
    c3_center = counts_3.get(f"{prev_word} {candidate_lower} {next_word}", 0)
    c3_left   = counts_3.get(f"{candidate_lower} {next_word} {next_2_word}", 0)
    c3_right  = counts_3.get(f"{prev_2_word} {prev_word} {candidate_lower}", 0)
    c3 = c3_center + c3_left + c3_right
    
    # Chuẩn hóa log mượt dữ liệu cho màng lọc tuyển chọn Hard Negatives
    norm_c1 = min(1.0, math.log1p(c1) / 15.0)
    norm_c2 = min(1.0, math.log1p(c2) / 12.0)
    norm_c3 = min(1.0, math.log1p(c3) / 12.0)
    
    return c1, c2, c3, norm_c1, norm_c2, norm_c3

# Đặc trưng tỉ lệ chiều dài từ lỗi và ứng viên
def extract_length_ratio_feature(error_word, candidate):
    len_err = len(error_word)
    len_cand = len(candidate)
    length_ratio = min(len_err, len_cand) / max(len_err, len_cand) if max(len_err, len_cand) > 0 else 0
    return length_ratio
