import logging
import os
from datetime import datetime

# Tạo thư mục logs nếu chưa tồn tại
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Tạo tên file log theo ngày
LOG_FILE = os.path.join(LOG_DIR, f"spelling_checker_{datetime.now().strftime('%Y%m%d')}.log")

def get_logger(module_name):
    """
    Hàm khởi tạo logger cho từng module chuyên biệt
    """
    logger = logging.getLogger(module_name)
    
    # Tránh việc add trùng handler nếu hàm này bị gọi lại
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Định dạng log: Thời gian - Tên Module - Mức độ log - Thông điệp
        formatter = logging.Formatter(
            '%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 1. Handler để ghi ra File (Lưu từ mức INFO trở lên)
        file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # 2. Handler để in ra Console (Màn hình terminal - Hiện cả DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        # Thêm các handler vào logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger