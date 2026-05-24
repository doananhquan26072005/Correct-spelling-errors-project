import re
import numpy as np

# Hàm tìm vị trí lỗi của câu bằng model tri-gram
def detect_error(sentence, model_lm):
    scores = list(model_lm.full_scores(sentence))[:-1]
    words = sentence.split()

    error_indices = set()
    valid_probs = []
    valid_indices = []

    for i, (prob, length, is_oov) in enumerate(scores):
        if i >= len(words):
            continue
            
        # Lọc số
        if re.search(r'[0-9]', words[i]):
            continue

        # Lọc các từ out of vocabulary
        if is_oov:
            error_indices.add(i)
            
        valid_probs.append(prob)
        valid_indices.append(i)

    # Ngưỡng ken_lm
    if valid_probs:
        mean_prob = np.mean(valid_probs)
        std_prob = np.std(valid_probs) 
        
        alpha = 1.4
        dynamic_threshold = mean_prob - (alpha * std_prob)
        
        hard_ceiling = -5.52
        
        hard_floor = -5.93

        for idx, prob in zip(valid_indices, valid_probs):
            is_anomaly = (prob < dynamic_threshold) and (prob < hard_ceiling)
            is_absolute_error = (prob < hard_floor)
            
            if is_anomaly or is_absolute_error:
                error_indices.add(idx)

    return sorted(list(error_indices))

# Hàm tính độ liên quan bằng model skip-gram
def calculate_similarity(word1, word2, embedding_matrix, word_to_idx):
    if word1 not in word_to_idx or word2 not in word_to_idx:
        return 0.0
    
    # Lấy vector của từng từ
    idx1 = word_to_idx[word1]
    idx2 = word_to_idx[word2]
    vec1 = embedding_matrix[idx1]
    vec2 = embedding_matrix[idx2]

    # Tính Cosine Similarity
    # Công thức: (A . B) / (||A|| * ||B||)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)