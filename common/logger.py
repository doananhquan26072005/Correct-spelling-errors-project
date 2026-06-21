import logging
import os
import sys
from datetime import datetime

# Tạo thư mục logs nếu chưa tồn tại
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Tạo tên file log theo ngày
LOG_FILE = os.path.join(LOG_DIR, f"spelling_checker_{datetime.now().strftime('%Y%m%d')}.log")

def get_logger(module_name):
    logger = logging.getLogger(module_name)
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 1. File Handler
        file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # 2. Console Handler: Ép ghi vào sys.stdout để đồng bộ với print/thư viện khác
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger