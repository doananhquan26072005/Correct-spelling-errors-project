# spell_correction/utils.py
from typing import List, Dict, Set, Tuple

# Hằng số cấu hình bàn phím vật lý QWERTY (Cố định, không đổi)
_ADJACENT_KEYS: Dict[str, str] = {
    'q': 'wea', 'w': 'qeasd', 'e': 'wrsdf', 'r': 'etdfg', 't': 'ryfgh', 'y': 'tughj', 'u': 'yihjk', 'i': 'uojkl', 'o': 'ipkl', 'p': 'ol',
    'a': 'qwsz', 's': 'weadzx', 'd': 'ersfxc', 'f': 'rtdgcv', 'g': 'tyfhvb', 'h': 'yugjbn', 'j': 'uihknm', 'k': 'iojlm', 'l': 'opk',
    'z': 'asx', 'x': 'sdzc', 'c': 'dfxv', 'v': 'fgcb', 'b': 'ghvn', 'n': 'hjbm', 'm': 'jkn'
}

def create_telex_form(word: str, telex_dict: dict) -> List[str]:
    """Tạo nhiều biến thể Telex của 1 từ tiếng Việt dựa trên quy tắc Unicode."""
    word = word.lower()
    prefix, vowel_base, suffix, word_tone, word_mod = "", "", "", "", ""
    VOWELS = "aeiouy"
    state = 0  # 0: phụ âm đầu, 1: nguyên âm

    i = 0
    while i < len(word):
        step = 1
        if i < len(word) - 1 and word[i:i+2] in telex_dict:
            char = word[i:i+2]
            step = 2
        else:
            char = word[i]

        if char in telex_dict:
            if char == 'đ':
                if state == 0: prefix += 'dd'
                else: suffix += 'dd'
            else:
                vowel_base += telex_dict[char][0]
                if telex_dict[char][1]: word_mod = telex_dict[char][1]
                if telex_dict[char][2]: word_tone = telex_dict[char][2]
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


def compute_edit_distance(s1: str, s2: str, confusion_pairs: Set[Tuple[str, str]], cfg=None) -> float:
    """Tính khoảng cách Damerau-Levenshtein có hiệu chỉnh trọng số bàn phím và vùng miền."""
    if cfg and hasattr(cfg, 'candidate_generation'):
        c_gen = cfg.candidate_generation
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
            elif char1 in _ADJACENT_KEYS.get(char2, "") or char2 in _ADJACENT_KEYS.get(char1, ""):
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


def compute_edit_distance_telex(s1: str, s2: str, telex_dict: dict, confusion_pairs: Set[Tuple[str, str]], cfg=None) -> float:
    """Tìm khoảng cách chỉnh sửa nhỏ nhất giữa mọi tổ hợp biến thể Telex của hai chuỗi."""
    min_dist = float('inf')
    string1 = create_telex_form(s1, telex_dict)
    string2 = create_telex_form(s2, telex_dict)

    for str1 in string1:
        for str2 in string2:
            dist = compute_edit_distance(str1, str2, confusion_pairs, cfg)
            if dist < min_dist:
                min_dist = dist
    return min_dist