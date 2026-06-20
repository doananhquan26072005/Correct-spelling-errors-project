import yaml
import re
from types import SimpleNamespace

def dict_to_namespace(d):
    """Đệ quy chuyển đổi hoàn toàn Dictionary lồng nhau sang SimpleNamespace Object."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def load_config(config_path: str) -> SimpleNamespace:
    """Đọc file YAML tĩnh và nạp vào cấu trúc Class ẩn danh."""
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    # BẮT BUỘC: Phải bọc config_dict qua hàm convert trước khi return
    return dict_to_namespace(config_dict)

