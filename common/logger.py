import logging
import os
import sys
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"spelling_checker_{datetime.now().strftime('%Y%m%d')}.log")

def get_logger(module_name):
    logger = logging.getLogger(module_name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger