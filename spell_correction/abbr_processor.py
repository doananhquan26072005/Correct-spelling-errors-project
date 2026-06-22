import os
from typing import Dict
from common.logger import get_logger

logger = get_logger(__name__)

class AbbreviationProcessor:
    def __init__(self, teen_code_path: str):
        self.abbreviation_dict: Dict[str, str] = {}
        logger.info(f"Initializing AbbreviationProcessor with dict: {teen_code_path}")
        self._load_dictionary(teen_code_path)
        
    def _load_dictionary(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"Teencode file not found at '{path}'. Skipping load.")
            return
            
        with open(path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    shortcut = parts[0].lower()
                    full_word = parts[1].lower()
                    self.abbreviation_dict[shortcut] = full_word
        logger.info(f"Loaded {len(self.abbreviation_dict):,} teencode pairs.")

    def replace_abbreviations(self, sentence: str) -> str:
        words = sentence.lower().split()
        replaced_count = 0
        for i, word in enumerate(words):
            if word in self.abbreviation_dict:
                if len(self.abbreviation_dict[word].split()) == 1:
                    logger.debug(f"Teencode match: '{word}' -> '{self.abbreviation_dict[word]}'")
                    words[i] = self.abbreviation_dict[word]
                    replaced_count += 1
        if replaced_count > 0:
            logger.debug(f"Executed {replaced_count} shorthand replacements in sentence.")
        return " ".join(words)