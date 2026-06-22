import json
import re

from common.logger import get_logger

logger = get_logger(__name__)


class ResourceLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        logger.info("ResourceLoader initialized.")

    def load_vocab_and_dicts(self):
        logger.info("Loading dictionaries and telex configurations...")

        try:
            with open(self.cfg.paths.vocab_file, 'r', encoding='utf-8') as f:
                vocab = list(dict.fromkeys(f.read().splitlines()))
            word_to_idx = {word: i for i, word in enumerate(vocab)}
            logger.debug(f"Loaded vocab. Unique words: {len(vocab):,}")

            with open(self.cfg.paths.stopwords_file, 'r', encoding='utf-8') as f:
                stopwords = set(f.read().splitlines())
            logger.debug(f"Loaded stopwords. Total count: {len(stopwords):,}")

            with open(self.cfg.paths.telex_file, "r", encoding="utf-8") as f:
                telex_raw = f.read()
                telex_raw = re.sub(r',\s*}', '\n}', telex_raw)
                telex_dict = json.loads(telex_raw)
            logger.debug(f"Loaded telex dictionary rules: {len(telex_dict):,}")

            logger.info("All resource assets loaded successfully.")
            return {
                "vocab": vocab,
                "word_to_idx": word_to_idx,
                "stopwords": stopwords,
                "telex_dict": telex_dict
            }
        except FileNotFoundError as e:
            logger.error(f"Critical resource file missing: {e.filename}", exc_info=True)
            raise e
        except json.JSONDecodeError as e:
            logger.error("Malformed JSON structure detected in telex mapping file.", exc_info=True)
            raise e


def process_dataset(examples, word_to_idx):
    inputs = []
    targets = []
    skipped_count = 0
    raw_samples = len(examples['input'])

    for inp, tgt in zip(examples['input'], examples['target']):
        inp_str = str(inp).lower()
        tgt_str = str(tgt).lower()

        inp_clean = re.sub(r'[^\w\s_]|\d+', '', inp_str)
        tgt_clean = re.sub(r'[^\w\s_]|\d+', '', tgt_str)

        inp_tokens = inp_clean.split()
        tgt_tokens = tgt_clean.split()

        if len(inp_tokens) != len(tgt_tokens) or len(inp_tokens) == 0:
            skipped_count += 1
            continue

        new_inp_tokens = []
        for inp_word, tgt_word in zip(inp_tokens, tgt_tokens):
            if tgt_word not in word_to_idx:
                new_inp_tokens.append(tgt_word)
            else:
                new_inp_tokens.append(inp_word)

        inputs.append(" ".join(new_inp_tokens))
        targets.append(" ".join(tgt_tokens))

    logger.debug(
        f"Processed batch size: {raw_samples} | Retained: {len(inputs)} | Skipped: {skipped_count}"
    )
    return {"input": inputs, "target": targets}


def split_data(dataset):
    """Splits dataset into Typo/Teencode pipeline (Error 1) and Unmarked pipeline (Error 2)."""
    logger.info("Splitting dataset into two distinct error pipelines...")
    
    VIETNAMESE_DIACRITICS = re.compile(
        r'[àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]'
    )

    def error_1(example):
        return bool(VIETNAMESE_DIACRITICS.search(str(example['input'])))

    def error_2(example):
        return not bool(VIETNAMESE_DIACRITICS.search(str(example['input'])))

    df_error1 = dataset.filter(error_1)
    df_error2 = dataset.filter(error_2)
    
    for split_name in dataset.keys():
        total_len = len(dataset[split_name])
        len_e1 = len(df_error1[split_name])
        len_e2 = len(df_error2[split_name])
        
        logger.info(f"Split results for [{split_name}]:")
        logger.info(f"  > Total samples : {total_len:,}")
        logger.info(f"  > Error 1 (Typo): {len_e1:,} ({len_e1/max(1, total_len)*100:.2f}%)")
        logger.info(f"  > Error 2 (None): {len_e2:,} ({len_e2/max(1, total_len)*100:.2f}%)")
        
    return df_error1, df_error2