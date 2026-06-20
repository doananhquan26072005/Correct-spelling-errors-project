# test_config.py
from common.config import load_config
from types import SimpleNamespace

def test_system_config():
    print("=== BẮT ĐẦU KIỂM TRA HÀM LOAD_CONFIG ===")
    
    # 1. Đường dẫn tới file config bước 1 của bạn
    config_path = "configs/diacritic_config.yaml"
    
    try:
        # 2. Chạy hàm load
        cfg = load_config(config_path)
        print(f"✓ Đọc file '{config_path}' thành công.")
        
        # 3. KIỂM TRA 1: Check xem có phải là Object (SimpleNamespace) không
        assert isinstance(cfg, SimpleNamespace), "Lỗi: Kết quả trả về phải là một Namespace Object!"
        print("✓ Kiểm tra kiểu dữ liệu gốc: Đúng cấu trúc Object.")
        
        # 4. KIỂM TRA 2: Check xem các nhánh con có bị lồng thành Object luôn chưa
        assert isinstance(cfg.model, SimpleNamespace), "Lỗi: Nhóm 'model' chưa được convert thành Object!"
        assert isinstance(cfg.training, SimpleNamespace), "Lỗi: Nhóm 'training' chưa được convert thành Object!"
        print("✓ Kiểm tra cấu trúc lồng nhau (Nested sub-groups): Đạt chuẩn.")
        
        # 5. KIỂM TRA 3: In thử một vài giá trị thực tế để mắt thường xác nhận
        print("\n--- Giá trị thực tế đọc từ file YAML ---")
        print(f"  • Độ dài chuỗi tối đa (cfg.model.max_len) : {cfg.model.max_len}")
        print(f"  • Kích thước Batch (cfg.training.batch_size): {cfg.training.batch_size}")
        print(f"  • Đường dẫn CSV (cfg.data.csv_path)        : {cfg.data.csv_path}")
        print(f"  • Tỷ lệ học (cfg.training.learning_rate)   : {cfg.training.learning_rate}")
        print("----------------------------------------")
        
        # 6. KIỂM TRA 4: Check xem IDE có gợi ý code tự động không
        # Khi bạn gõ "cfg.model." trong VS Code hoặc PyCharm, nó sẽ tự hiện gợi ý "max_len", "d_model"...
        
        print("\n🎉 CHÚC MỪNG: Hàm load_config hoạt động HOÀN HẢO 100%!")
        print(cfg)
        
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file tại '{config_path}'. Hãy chắc chắn bạn đã tạo file YAML trong thư mục configs/")
    except AttributeError as e:
        print(f"❌ LỖI cấu trúc: Bạn đang gọi sai tên biến hoặc cấu trúc YAML không đồng bộ! Chi tiết: {e}")
    except AssertionError as e:
        print(f"❌ LỖI logic hàm: {e}")

if __name__ == "__main__":
    test_system_config()