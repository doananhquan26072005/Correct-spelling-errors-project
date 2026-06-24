# Hàm tạo nhiều biến thể telex của 1 từ
def create_telex_form(word, telex):
    word = word.lower()
    prefix = ""      # Phụ âm đầu
    vowel_base = ""  # Nguyên âm gốc
    suffix = ""      # Phụ âm cuối
    word_tone = ""   # Dấu thanh
    word_mod = ""    # Ký tự gõ mũ/móc

    VOWELS = "aeiouy" # Các nguyên âm tiếng Việt
    state = 0 # 0: phụ âm đầu, 1: nguyên âm

    i = 0
    while i < len(word):
        step = 1
        # Trường hợp 'ươ'
        if i < len(word) - 1 and word[i:i+2] in telex:
            char = word[i:i+2]
            step = 2
        else:
            char = word[i]

        # Nếu ký tự nằm trong từ điển
        if char in telex:
            if char == 'đ':
                if state == 0: prefix += 'dd'
                else: suffix += 'dd'
            else:
                vowel_base += telex[char][0]
                if telex[char][1]: word_mod = telex[char][1]
                if telex[char][2]: word_tone = telex[char][2]
                state = 1

        # Nếu ký tự là chữ bình thường
        else:
            if char in VOWELS:
                vowel_base += char
                state = 1
            else:
                if state == 0: prefix += char
                else: suffix += char

        i += step

    # Tạo biến thể
    variants = set()
    inline_vowel = vowel_base + word_mod

    # Kiểu 1 & 2: Gõ chuẩn ngay sau nguyên âm hoặc ném dấu thanh ra cuối
    variants.add(prefix + inline_vowel + word_tone + suffix)
    variants.add(prefix + inline_vowel + suffix + word_tone)

    # Kiểu 3: Phím bổ nghĩa (w, a, e, o) ném ra cuối từ
    if word_mod:
        variants.add(prefix + vowel_base + word_tone + suffix + word_mod)
        variants.add(prefix + vowel_base + suffix + word_mod + word_tone)
        variants.add(prefix + vowel_base + suffix + word_tone + word_mod)

    # Trường hợp đặc biệt của "ươ"
    if vowel_base == 'uo' and word_mod == 'w':
        # Tách w hai lần ngay sau nguyên âm
        variants.add(prefix + 'uwow' + word_tone + suffix)
        variants.add(prefix + 'uwow' + suffix + word_tone)

        # Tách w đầu, w cuối
        variants.add(prefix + 'uwo' + word_tone + suffix + 'w')
        variants.add(prefix + 'uwo' + suffix + 'w' + word_tone)
        variants.add(prefix + 'uwo' + suffix + word_tone + 'w')

    return list(v for v in variants if v)

# Hàm tạo các biến thể xóa từ 0 đến k kí tự của từ
def get_deletes(word, k = 2):
    queue = {word}
    variant_list = set()
    
    for _ in range(k):
        temp_queue = set()
        for w in queue:
            if len(w) > 1:
                # Tạo deletes cho vòng hiện tại
                deletes = {w[:i] + w[i+1:] for i in range(len(w))}
                variant_list.update(deletes)
                temp_queue.update(deletes)
        queue = temp_queue
    return variant_list

# Hàm tính khoảng cách của xâu 1 và xâu 2 bằng thuật toán Damerau-Levenshtein
# Là hàm edit_distance nhưng có thêm phép đổi chỗ các kí tự (gõ lộn thứ tự)
# Cải thiện thêm bằng khoảng cách bàn phím cho trường hợp gõ nhầm
def edit_distance(s1, s2):
    # Các phím liền kề trên bàn phím để tính trọng số
    ADJACENT_KEYS = {
        'q': 'wea', 'w': 'qeasd', 'e': 'wrsdf', 'r': 'etdfg', 't': 'ryfgh', 'y': 'tughj', 'u': 'yihjk', 'i': 'uojkl', 'o': 'ipkl', 'p': 'ol',
        'a': 'qwsz', 's': 'weadzx', 'd': 'ersfxc', 'f': 'rtdgcv', 'g': 'tyfhvb', 'h': 'yugjbn', 'j': 'uihknm', 'k': 'iojlm', 'l': 'opk',
        'z': 'asx', 'x': 'sdzc', 'c': 'dfxv', 'v': 'fgcb', 'b': 'ghvn', 'n': 'hjbm', 'm': 'jkn'
    }

    # Các cặp âm dễ nhầm lẫn trong phát âm tiếng Việt (Lỗi ngữ âm)
    CONFUSION_PAIRS = {
        ('s', 'x'), ('x', 's'),
        ('l', 'n'), ('n', 'l'),
        ('d', 'r'), ('r', 'd'),
        ('d', 'gi'), ('gi', 'd'),
        ('i', 'y'), ('y', 'i'),
        ('c', 'k'), ('c', 'k'),
        ('ch', 'tr'), ('tr', 'ch')
    }

    n, m = len(s1), len(s2)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    # Khởi tạo giá trị hàng và cột đầu tiên
    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            char1 = s1[i - 1]
            char2 = s2[j - 1]
            # Tính chi phí thay thế (0 nếu giống nhau, 0.5 nếu khác mà gần nhau trên bàn phím, 1 nếu khác và ở xa trên bàn phím)
            if char1 == char2:
                sub_cost = 0.0
            elif (char1, char2) in CONFUSION_PAIRS:
            # Lỗi phát âm vùng miền (VD: s vs x). Phạt rất nhẹ vì đây là lỗi cực kỳ phổ biến.
                sub_cost = 0.4
            elif char1 in ADJACENT_KEYS.get(char2, "") or char2 in ADJACENT_KEYS.get(char1, ""):
                # Nếu gõ nhầm 2 phím cạnh nhau (VD: a và s), chi phí chỉ là 0.6
                sub_cost = 0.5
            else:
                # Lỗi gõ nhầm phím xa nhau, chi phí 1.0
                sub_cost = 1

            # Tính chi phí thêm / xóa
            del_cost = 1
            ins_cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + del_cost,       # Xóa
                dp[i][j - 1] + ins_cost,       # Thêm
                dp[i - 1][j - 1] + sub_cost    # Thay thế
            )

            # Phép Đổi chỗ (Transposition)
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 0.5)

            if i >= 2 and j >= 2:
                sub1 = s1[i-2:i]
                sub2 = s2[j-2:j]
                if (sub1, sub2) in CONFUSION_PAIRS:
                    dp[i][j] = min(dp[i][j], dp[i-2][j-2] + 0.4)
            
            if i >= 2 and j >= 1:
                sub1 = s1[i-2:i]
                sub2 = s2[j-1:j]
                if (sub1, sub2) in CONFUSION_PAIRS:
                    dp[i][j] = min(dp[i][j], dp[i-2][j-1] + 0.4)

            if i >= 1 and j >= 2:
                sub1 = s1[i-1:i]
                sub2 = s2[j-2:j]
                if (sub1, sub2) in CONFUSION_PAIRS:
                    dp[i][j] = min(dp[i][j], dp[i-1][j-2] + 0.4)

    return dp[n][m]

# Hàm tính min tất cả edit_distance của các cách gõ unicode của 2 từ
def edit_distance_telex(s1, s2, telex):
    min_dist = float('inf')

    string1 = create_telex_form(s1, telex)
    string2 = create_telex_form(s2, telex)

    for str1 in string1:
        for str2 in string2:
            dist = edit_distance(str1, str2)
            if dist < min_dist:
                min_dist = dist
    return min_dist

# Hàm tìm từ gần nhất với từ nhập vào, hàm trả về các từ gần nhất và khoảng cách của nó
def lookup(word, sym_dict, telex, word_to_idx, counts_1, k=2):
    variant_list = [word] + list(get_deletes(word))
    for telex_form in create_telex_form(word, telex):
        variant_list += list(get_deletes(telex_form))

    # Lưu đáp án là các từ gần nhất và khoảng cách của nó
    candidates = {}

    # Tra cứu các biến thể
    for variant in variant_list:
        if variant in sym_dict:
            for suggestion in sym_dict[variant]:
                if suggestion in candidates:
                    continue

                dist = edit_distance_telex(word, suggestion, telex)

                # Nhận khi khoảng cách <= k, tồn tại trong từ điển và có tần suất > 0
                if dist <= k and suggestion in word_to_idx and counts_1.get(suggestion, 0) > 0:
                    candidates[suggestion] = (dist, counts_1.get(suggestion, 0))

    # Xếp hạng ưu tiên: Distance nhỏ trước -> Tần suất cao trước
    result = sorted(candidates.items(),
                    key=lambda x: (x[1][0], -x[1][1]))

    ans = []
    # Giải nén thẳng tuple cho dễ đọc, đổi tên biến tránh ghi đè tham số 'word'
    for cand_word, (dist, count) in result:
        ans.append(cand_word)

    return ans