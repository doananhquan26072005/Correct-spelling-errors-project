"""
File: create_data_LightGBM.py
Mô tả: Tạo data cho LightGBM bằng duyệt các câu trong tập train,
        tìm lỗi sai của câu input và target và tạo các candidates.
"""

import json
import math
import re
from collections import Counter, defaultdict

import kenlm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm

# 1. KHỞI TẠO & LOAD DATASET / VOCABULARY

# Load dataset lỗi chính tả tiếng Việt
dataset = load_dataset("yammdd/vietnamese-error-correction-corpus")

# Tạo vocab và ánh xạ chỉ số từ file text
vocab = []
with open(
    r"/kaggle/input/datasets/anhquanjerryus/all-file/vocabulary.txt",
    "r",
    encoding="utf-8",
) as f:
    vocab = f.read().splitlines()

# Lọc các từ trùng lặp
vocab = list(dict.fromkeys(vocab))

# Ánh xạ word <-> idx
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}


# 2. TIỀN XỬ LÝ & PHÂN TÁCH DỮ LIỆU
def process_dataset(examples):
    inputs = []
    targets = []

    for inp, tgt in zip(examples["input"], examples["target"]):
        # Viết thường
        inp_str = str(inp).lower()
        tgt_str = str(tgt).lower()

        # Xóa dấu câu và chữ số
        inp_clean = re.sub(r"[^\w\s_]|\d+", "", inp_str)
        tgt_clean = re.sub(r"[^\w\s_]|\d+", "", tgt_str)

        inp_tokens = inp_clean.split()
        tgt_tokens = tgt_clean.split()

        # Điều kiện độ dài: Chỉ giữ lại nếu bằng nhau và khác rỗng
        if len(inp_tokens) != len(tgt_tokens) or len(inp_tokens) == 0:
            continue

        # Thay thế các từ không phải tiếng Việt mà bị lỗi
        new_inp_tokens = []
        for inp_word, tgt_word in zip(inp_tokens, tgt_tokens):
            if tgt_word not in word_to_idx:
                new_inp_tokens.append(tgt_word)
            else:
                new_inp_tokens.append(inp_word)

        inputs.append(" ".join(new_inp_tokens))
        targets.append(" ".join(tgt_tokens))

    return {"input": inputs, "target": targets}


# Map tiền xử lý lên toàn bộ DatasetDict
df = dataset.map(
    process_dataset, batched=True, remove_columns=dataset["train"].column_names
)

# Phân tách loại lỗi dựa trên dấu thanh tiếng Việt
VIETNAMESE_DIACRITICS = re.compile(
    r"[àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]"
)


def split_data(dataset):
    # Lỗi chính tả (còn giữ ký tự dấu thanh)
    def error_1(example):
        inp = str(example["input"])
        return bool(VIETNAMESE_DIACRITICS.search(inp))

    # Lỗi gõ không dấu hoàn toàn
    def error_2(example):
        inp = str(example["input"])
        return not bool(VIETNAMESE_DIACRITICS.search(inp))

    df_error1 = dataset.filter(error_1)
    df_error2 = dataset.filter(error_2)

    return df_error1, df_error2


# Tách và chuyển thành DataFrame
df1, df2 = split_data(df)

df_train = pd.DataFrame(df["train"])
df_test = pd.DataFrame(df["test"])
df_valid = pd.DataFrame(df["validation"])

df1_train = pd.DataFrame(df1["train"])
df1_test = pd.DataFrame(df1["test"])
df1_valid = pd.DataFrame(df1["validation"])

df2_train = pd.DataFrame(df2["train"])
df2_test = pd.DataFrame(df2["test"])
df2_valid = pd.DataFrame(df2["validation"])


# 3. THỐNG KÊ N-GRAM & LOAD RESOURCES (TELEX, STOPWORDS, TEENCODE)

# Tính tần suất xuất hiện Uni/Bi/Tri-gram từ tập target (chuẩn) làm tài nguyên heuristic
counts_1 = Counter()
counts_2 = Counter()
counts_3 = Counter()

for sentence in df_train["target"]:
    sentence = sentence.lower()
    sentence = re.sub(r"\s+([.,!?;:\)\]\}])", r"\1", sentence)

    tokens = str(sentence).lower().split()
    if not tokens:
        continue

    counts_1.update(tokens)

    bigrams = [" ".join(p) for p in zip(tokens, tokens[1:])]
    counts_2.update(bigrams)

    trigrams = [" ".join(t) for t in zip(tokens, tokens[1:], tokens[2:])]
    counts_3.update(trigrams)

# Load danh sách Stopwords
with open(
    r"/kaggle/input/datasets/anhquanjerryus/all-file/vietnamese-stopwords.txt",
    "r",
    encoding="utf-8",
) as f:
    stopwords = f.read().splitlines()
stopword = set(stopwords)

# Load bộ gõ Telex mapping phục vụ sinh biến thể
with open(
    r"/kaggle/input/datasets/anhquanjerryus/all-file/telex.txt",
    "r",
    encoding="utf-8",
) as f:
    telex = f.read()

telex = re.sub(r",\s*}", "\n}", telex)
telex = json.loads(telex)

# Load từ điển teen-code / từ viết tắt
abbreviation_dict = {}
with open(
    r"/kaggle/input/datasets/anhquanjerryus/teen-code/teen_code.txt",
    "r",
    encoding="utf-8",
) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        shortcut, full_word = parts[0].lower(), parts[1].lower()
        abbreviation_dict[shortcut] = full_word


def replace_abbreviations(sentence):
    words = sentence.lower().split()
    for i, word in enumerate(words):
        if word in abbreviation_dict:
            words[i] = abbreviation_dict[word]
    return " ".join(words)


# 4. THUẬT TOÁN SINH BIẾN THỂ & EDIT DISTANCE CUSTOM


def create_telex_form(word):
    word = word.lower()
    prefix = ""  # Phụ âm đầu
    vowel_base = ""  # Nguyên âm gốc
    suffix = ""  # Phụ âm cuối
    word_tone = ""  # Dấu thanh
    word_mod = ""  # Ký tự gõ mũ/móc

    VOWELS = "aeiouy"
    state = 0  # 0: phụ âm đầu, 1: nguyên âm

    i = 0
    while i < len(word):
        step = 1
        if i < len(word) - 1 and word[i : i + 2] in telex:
            char = word[i : i + 2]
            step = 2
        else:
            char = word[i]

        if char in telex:
            if char == "đ":
                if state == 0:
                    prefix += "dd"
                else:
                    suffix += "dd"
            else:
                vowel_base += telex[char][0]
                if telex[char][1]:
                    word_mod = telex[char][1]
                if telex[char][2]:
                    word_tone = telex[char][2]
                state = 1
        else:
            if char in VOWELS:
                vowel_base += char
                state = 1
            else:
                if state == 0:
                    prefix += char
                else:
                    suffix += char
        i += step

    variants = set()
    inline_vowel = vowel_base + word_mod

    variants.add(prefix + inline_vowel + word_tone + suffix)
    variants.add(prefix + inline_vowel + suffix + word_tone)

    if word_mod:
        variants.add(prefix + vowel_base + word_tone + suffix + word_mod)
        variants.add(prefix + vowel_base + suffix + word_mod + word_tone)
        variants.add(prefix + vowel_base + suffix + word_tone + word_mod)

    if vowel_base == "uo" and word_mod == "w":
        variants.add(prefix + "uwow" + word_tone + suffix)
        variants.add(prefix + "uwow" + suffix + word_tone)
        variants.add(prefix + "uwo" + word_tone + suffix + "w")
        variants.add(prefix + "uwo" + suffix + "w" + word_tone)
        variants.add(prefix + "uwo" + suffix + word_tone + "w")

    return list(v for v in variants if v)


def get_deletes(word, k=2):
    queue = {word}
    variant_list = set()

    for _ in range(k):
        temp_queue = set()
        for w in queue:
            if len(w) > 1:
                deletes = {w[:i] + w[i + 1 :] for i in range(len(w))}
                variant_list.update(deletes)
                temp_queue.update(deletes)
        queue = temp_queue
    return variant_list


# Khởi tạo bản đồ biến thể đảo ngược (Hash Map) phục vụ tra cứu ngược candidate nhanh
sym_dict = defaultdict(list)
for word in vocab:
    length = word.split(" ")
    if len(length) > 1:
        continue
    base_forms = [word] + create_telex_form(word)

    for form in base_forms:
        if word not in sym_dict[form]:
            sym_dict[form].append(word)

        variant_list = get_deletes(form)
        for variant in variant_list:
            if word not in sym_dict[variant]:
                sym_dict[variant].append(word)

# Định nghĩa các phím liền kề và cặp âm hay nhầm lẫn vùng miền
ADJACENT_KEYS = {
    "q": "wea",
    "w": "qeasd",
    "e": "wrsdf",
    "r": "etdfg",
    "t": "ryfgh",
    "y": "tughj",
    "u": "yihjk",
    "i": "uojkl",
    "o": "ipkl",
    "p": "ol",
    "a": "qwsz",
    "s": "weadzx",
    "d": "ersfxc",
    "f": "rtdgcv",
    "g": "tyfhvb",
    "h": "yugjbn",
    "j": "uihknm",
    "k": "iojlm",
    "l": "opk",
    "z": "asx",
    "x": "sdzc",
    "c": "dfxv",
    "v": "fgcb",
    "b": "ghvn",
    "n": "hjbm",
    "m": "jkn",
}
TELEX_KEYS = set("sfrxjwaeod")
CONFUSION_PAIRS = {
    ("s", "x"),
    ("x", "s"),
    ("l", "n"),
    ("n", "l"),
    ("d", "r"),
    ("r", "d"),
    ("d", "gi"),
    ("gi", "d"),
    ("i", "y"),
    ("y", "i"),
    ("c", "k"),
    ("ch", "tr"),
    ("tr", "ch"),
}


def edit_distance(s1, s2):
    n, m = len(s1), len(s2)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            char1 = s1[i - 1]
            char2 = s2[j - 1]

            if char1 == char2:
                sub_cost = 0.0
            elif (char1, char2) in CONFUSION_PAIRS:
                sub_cost = 0.4
            elif char1 in ADJACENT_KEYS.get(
                char2, ""
            ) or char2 in ADJACENT_KEYS.get(char1, ""):
                sub_cost = 0.5
            else:
                sub_cost = 1.0

            del_cost = 1.0
            ins_cost = 1.0

            dp[i][j] = min(
                dp[i - 1][j] + del_cost,  # Xóa
                dp[i][j - 1] + ins_cost,  # Thêm
                dp[i - 1][j - 1] + sub_cost,  # Thay thế
            )

            # Phép đổi chỗ 2 ký tự liền kề (Transposition)
            if (
                i > 1
                and j > 1
                and s1[i - 1] == s2[j - 2]
                and s1[i - 2] == s2[j - 1]
            ):
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 0.5)

            if i >= 2 and j >= 2:
                if (s1[i - 2 : i], s2[j - 2 : j]) in CONFUSION_PAIRS:
                    dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 0.4)

            if i >= 2 and j >= 1:
                if (s1[i - 2 : i], s2[j - 1 : j]) in CONFUSION_PAIRS:
                    dp[i][j] = min(dp[i][j], dp[i - 2][j - 1] + 0.4)

            if i >= 1 and j >= 2:
                if (s1[i - 1 : i], s2[j - 2 : j]) in CONFUSION_PAIRS:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 2] + 0.4)

    return dp[n][m]


def edit_distance_telex(s1, s2):
    min_dist = float("inf")
    string1 = create_telex_form(s1)
    string2 = create_telex_form(s2)

    for str1 in string1:
        for str2 in string2:
            dist = edit_distance(str1, str2)
            if dist < min_dist:
                min_dist = dist
    return min_dist


def lookup(word, k=2):
    variant_list = [word] + list(get_deletes(word))
    for tx in create_telex_form(word):
        variant_list += list(get_deletes(tx))

    candidates = {}
    for variant in variant_list:
        if variant in sym_dict:
            for suggestion in sym_dict[variant]:
                if suggestion in candidates:
                    continue

                dist = edit_distance_telex(word, suggestion)
                if (
                    dist <= k
                    and suggestion in vocab
                    and counts_1.get(suggestion, 0) > 0
                ):
                    candidates[suggestion] = (dist, counts_1.get(suggestion, 0))

    result = sorted(candidates.items(), key=lambda x: (x[1][0], -x[1][1]))
    return [cand_word for cand_word, _ in result]


# 5. KHỞI TẠO CÁC MÔ HÌNH (LANGUAGE MODEL & SKIPGRAM EMBEDDING)

# Khởi tạo KenLM Tri-gram model detect lỗi anomal
model_lm = kenlm.Model(r"/kaggle/input/datasets/anhquanjerryus/all-file/trigram.bin")


def detect_error_word(sentence):
    scores = list(model_lm.full_scores(sentence))[:-1]
    words = sentence.split()

    error_indices = set()
    valid_probs = []
    valid_indices = []

    for i, (prob, length, is_oov) in enumerate(scores):
        if i >= len(words):
            continue
        if re.search(r"[0-9]", words[i]):
            continue

        if is_oov:
            error_indices.add(i)

        valid_probs.append(prob)
        valid_indices.append(i)

    if valid_probs:
        mean_prob = np.mean(valid_probs)
        std_prob = np.std(valid_probs)

        alpha = 1.4
        dynamic_threshold = mean_prob - (alpha * std_prob)

        hard_ceiling = -5.52
        hard_floor = -5.93

        for idx, prob in zip(valid_indices, valid_probs):
            is_anomaly = (prob < dynamic_threshold) and (prob < hard_ceiling)
            is_absolute_error = prob < hard_floor

            if is_anomaly or is_absolute_error:
                error_indices.add(idx)

    return sorted(list(error_indices))


# Định nghĩa cấu trúc mô hình SkipGram kế thừa từ PyTorch
class SkipGram(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        out = self.linear(embed)
        return out


# Khởi tạo mô hình trên CPU để load weights pre-trained
device = torch.device("cpu")
EMBED_DIM = 300
VOCAB_SIZE = len(vocab)
model_skipgram = SkipGram(VOCAB_SIZE, EMBED_DIM).to(device)

model_path = r"/kaggle/input/datasets/anhquanjerryus/all-file/model_skipgram.pth"
model_skipgram.load_state_dict(
    torch.load(model_path, map_location=device, weights_only=True)
)

# Chuẩn bị ma trận nhúng chuẩn hóa (Normalized Matrix Vector) để tính Cosine Similarity nhanh bằng Numpy
embeddings = model_skipgram.embedding.weight.data
embedding_matrix = embeddings.cpu().numpy()
norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
norms[norms == 0] = 1.0
norm_embedding_matrix = embedding_matrix / norms


# 6. TRÍCH XUẤT ĐẶC TRƯNG HỖN HỢP & RANKING HEURISTIC


def extract_candidates_and_features(
    error_word, sentence_words, error_idx, error_indices, window_size=3
):
    n_words = len(sentence_words)

    if error_idx >= n_words or error_idx < 0:
        return []

    local_start = max(0, error_idx - window_size)
    local_end = min(n_words, error_idx + window_size + 1)

    prefix_words = sentence_words[local_start:error_idx]
    suffix_words = sentence_words[error_idx + 1 : local_end]

    prefix_str = " ".join(prefix_words) + " " if prefix_words else ""
    suffix_str = " " + " ".join(suffix_words) if suffix_words else ""

    start = max(0, error_idx - window_size)
    end = min(n_words, error_idx + window_size + 1)

    valid_context_words = []
    for i in range(start, end):
        if i == error_idx:
            continue
        if (
            i < error_idx or i not in error_indices
        ) and sentence_words[i] not in stopwords:
            word = sentence_words[i]
            if word in word_to_idx:
                dist_weight = 1.0 / abs(i - error_idx)
                valid_context_words.append((word, dist_weight))

    prev_word = sentence_words[error_idx - 1].lower() if error_idx > 0 else "<s>"
    prev_2_word = (
        sentence_words[error_idx - 2].lower() if error_idx > 1 else "<s>"
    )

    next_word = (
        sentence_words[error_idx + 1].lower() if error_idx < n_words - 1 else "</s>"
    )
    next_2_word = (
        sentence_words[error_idx + 2].lower() if error_idx < n_words - 2 else "</s>"
    )

    ctx_indices = []
    ctx_weights = []
    for ctx_word, weight in valid_context_words:
        ctx_indices.append(word_to_idx[ctx_word])
        ctx_weights.append(weight)

    mock_candidates = []
    top = []

    candidates = lookup(error_word)
    valid_candidates = [c for c in candidates if c in word_to_idx]

    cand_to_sim = {}
    if valid_candidates and ctx_indices:
        cand_indices = [word_to_idx[c] for c in valid_candidates]

        C = norm_embedding_matrix[cand_indices]
        W = norm_embedding_matrix[ctx_indices]

        S = np.dot(C, W.T)
        weights_array = np.array(ctx_weights)
        S_weighted = S * weights_array

        max_sims = np.max(S_weighted, axis=1)
        cand_to_sim = {
            cand: max_sims[i] for i, cand in enumerate(valid_candidates)
        }

    for rank_idx, candidate in enumerate(candidates):
        candidate_lower = candidate.lower()

        # Feature 1: Similarity
        weighted_sim = cand_to_sim.get(candidate, 0.0)
        norm_sim = max(0.0, weighted_sim)

        # Feature 2: KenLM Score
        local_sentence_str = f"{prefix_str}{candidate}{suffix_str}".strip()
        ken_score = model_lm.score(local_sentence_str)
        norm_ken = max(0.0, (ken_score + 15.0) / 15.0)

        # Feature 3: N-gram frequency log-smooth
        count_val_1 = counts_1.get(candidate, 0)

        c2_left = counts_2.get(f"{prev_word} {candidate_lower}", 0)
        c2_right = counts_2.get(f"{candidate_lower} {next_word}", 0)
        count_val_2 = c2_left + c2_right

        c3_center = counts_3.get(
            f"{prev_word} {candidate_lower} {next_word}", 0
        )
        c3_left = counts_3.get(f"{candidate_lower} {next_word} {next_2_word}", 0)
        c3_right = counts_3.get(f"{prev_2_word} {prev_word} {candidate_lower}", 0)
        count_val_3 = c3_center + c3_left + c3_right

        norm_count_1 = min(1.0, math.log1p(count_val_1) / 15.0)
        norm_count_2 = min(1.0, math.log1p(count_val_2) / 12.0)
        norm_count_3 = min(1.0, math.log1p(count_val_3) / 12.0)

        # Feature 4: Edit Distance Info
        dist = edit_distance_telex(error_word, candidate)
        norm_edit = 1.0 / (dist + 1)

        # Feature 5: Length Ratio
        len_err = len(error_word)
        len_cand = len(candidate)
        length_ratio = (
            min(len_err, len_cand) / max(len_err, len_cand)
            if max(len_err, len_cand) > 0
            else 0
        )

        # Tính toán điểm tích hợp Heuristic để lọc Hard Negative
        total_score = (
            (0.30 * norm_ken)
            + (0.25 * norm_edit)
            + (0.10 * length_ratio)
            + (0.20 * norm_count_2)
            + (0.05 * norm_count_3)
            + (0.05 * norm_count_1)
            + (0.05 * norm_sim)
        )

        top.append(
            (
                total_score,
                candidate,
                ken_score,
                weighted_sim,
                count_val_1,
                count_val_2,
                count_val_3,
                dist,
                length_ratio,
                rank_idx,
            )
        )

    # Hard Negative Mining
    top.sort(key=lambda x: x[0], reverse=True)

    for item in top:
        (
            _,
            candidate,
            ken_score,
            weighted_sim,
            c1,
            c2,
            c3,
            dist_val,
            length_ratio,
            sym_rank,
        ) = item

        feature_vector = [
            ken_score,  # 0
            weighted_sim,  # 1
            c1,  # 2
            c2,  # 3
            c3,  # 4
            dist_val,  # 5
            length_ratio,  # 6
        ]
        mock_candidates.append((candidate, feature_vector))

    return mock_candidates


def find_misspelled_words_and_targets(input_sentence, target_sentence):
    input_tokens = input_sentence.split()
    target_tokens = target_sentence.split()

    error_indices = []
    pairs = []
    if len(input_tokens) != len(target_tokens):
        return [], []

    for i in range(len(input_tokens)):
        if input_tokens[i] != target_tokens[i] and target_tokens[i] in word_to_idx:
            pairs.append((input_tokens[i], target_tokens[i]))
            error_indices.append(i)

    return pairs, error_indices


# 7. TRÍCH XUẤT ĐẶC TRƯNG TẬP TRAIN & LƯU FILE NPZ

X_train_list = []
y_train_list = []
group_train_list = []

for idx, row in tqdm(
    df1_train.iterrows(), total=len(df1_train), desc="Trích xuất đặc trưng"
):
    input_sent = str(row["input"])
    target_sent = str(row["target"])

    input_sent = replace_abbreviations(input_sent)

    error_pairs, error_indices = find_misspelled_words_and_targets(
        input_sent, target_sent
    )

    if not error_pairs:
        continue

    sentence_words = input_sent.split()

    for i in range(len(error_indices)):
        error_word, correct_word = error_pairs[i]

        candidates_with_scores = extract_candidates_and_features(
            error_word, sentence_words, error_indices[i], error_indices
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

        # Mining tối đa 20 Hard Negatives có Heuristic cao nhất để tạo tính cạnh tranh dữ liệu
        max_negatives = 20
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

        # Lưu lại độ dài nhóm (g-bound) cho định dạng LambdaMART nhóm của LightGBM
        group_train_list.append(len(final_candidates))

        # In-place Update cho từ đã sửa đổi để tối ưu ngữ cảnh bước sau
        if error_indices[i] < len(sentence_words):
            sentence_words[error_indices[i]] = correct_word

# Chuyển đổi sang định dạng mảng Numpy phục vụ lưu trữ cấu trúc màng lọc
X_train = np.array(X_train_list)
y_train = np.array(y_train_list)
group_train = np.array(group_train_list)

# Tiến hành nén lưu trữ dữ liệu cuối cùng
np.savez_compressed(
    "dataset_spell_correction_final.npz",
    X_train=X_train,
    y_train=y_train,
    group_train=group_train,
)
