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

        # Lọc các từ out of vocabulary
        if is_oov:
            error_indices.add(i)
            
        valid_probs.append(prob)
        valid_indices.append(i)

    # Ngưỡng ken_lm
    if valid_probs:
        mean_prob = np.mean(valid_probs)
        std_prob = np.std(valid_probs) 
        
        alpha = 4.6
        dynamic_threshold = mean_prob - (alpha * std_prob)
        
        hard_ceiling = -3.9
        
        hard_floor = -6.26

        for idx, prob in zip(valid_indices, valid_probs):
            is_anomaly = (prob < dynamic_threshold) and (prob < hard_ceiling)
            is_absolute_error = (prob < hard_floor)
            
            if is_anomaly or is_absolute_error:
                error_indices.add(idx)

    return sorted(list(error_indices))