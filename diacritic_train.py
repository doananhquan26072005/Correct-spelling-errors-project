import torch
import torch.nn as nn
from utils import load_config
from diacritic_restoration.dataset import DiacriticDataLoaderFactory
from diacritic_restoration.networks import build_model
from diacritic_restoration.trainer import DiacriticTrainer
from diacritic_restoration.processor import DiacriticRestorer

from diacritic_restoration.vocab import CharVocab, WordVocab, build_allowed_token_mask

if __name__ == "__main__":
    print("=== BẮT ĐẦU PIPELINE KHÔI PHỤC DẤU TIẾNG VIỆT ===")

    # 1. Nạp cấu hình từ file YAML tĩnh thành Object Class ẩn danh
    cfg = load_config("configs/diacritic_config.yaml")
    
    # Xử lý động thiết bị phần cứng (Device auto-detection)
    if cfg.training.device == "auto":
        cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Sử dụng thiết bị tính toán: {cfg.training.device}")

    # 2. Chuẩn bị dữ liệu và từ điển thông qua DataLoaderFactory (Đã đóng gói dạng Class)
    print("\n[*] Giai đoạn 1: Đang chuẩn bị dữ liệu và xây dựng từ điển...")
    data_factory = DiacriticDataLoaderFactory(cfg)
    train_loader, valid_loader, test_loader, char_vocab, word_vocab = data_factory.build_loaders_and_vocabs()
    
    # Xây dựng ma trận mặt nạ ràng buộc đầu ra cho chữ có dấu
    allowed_mask = build_allowed_token_mask(char_vocab, cfg)

    # 3. Khởi tạo mô hình mạng nơ-ron Transformer
    print("\n[*] Giai đoạn 2: Khởi tạo kiến trúc mạng nơ-ron...")
    model = build_model(char_vocab, word_vocab, cfg)
    
    # Hỗ trợ huấn luyện đa GPU nếu có
    if torch.cuda.device_count() > 1:
        print(f"    -> Phát hiện {torch.cuda.device_count()} GPUs. Kích hoạt nn.DataParallel.")
        model = nn.DataParallel(model)

    # 4. Kích hoạt tiến trình huấn luyện thông qua lớp Trainer chuyên trách
    print("\n[*] Giai đoạn 3: Bắt đầu quá trình huấn luyện mô hình (Training Loop)...")
    trainer = DiacriticTrainer(
        model=model,
        char_vocab=char_vocab,
        word_vocab=word_vocab,
        allowed_mask=allowed_mask,
        cfg=cfg
    )
    # Tiến hành fit dữ liệu qua các epoch
    model = trainer.fit(train_loader, valid_loader)

    # 5. Load lại checkpoint tốt nhất và thực hiện đánh giá trên tập Test
    print("\n[*] Giai đoạn 4: Đang nạp checkpoint tối ưu để đánh giá kiểm thử...")
    # Khởi tạo đối tượng Restorer độc lập (Tự động load weight bên trong class)
    restorer = DiacriticRestorer(
        checkpoint_path=cfg.training.checkpoint_path,
        cfg=cfg
    )
    
    # Tạo bộ đo đạc metrics
    trainer.evaluate_dataset(test_loader)
    
    # 6. Thử nghiệm suy luận thực tế (Inference Demo) với các chuỗi tự do
    examples = [
        "toi ten la",
        "mot nguoi mat tich o ho chua nuoc nuoc trong.",
        "nhung hinh anh dac sac nhat ve vong chung ket world cup.",
        "apple se ra mat ipad duoc thiet ke lai trong thang 4?.",
    ]

    print("\n=== DEMO SUY LUẬN THỰC TẾ (INFERENCE) ===")
    for text in examples:
        # Gọi qua phương thức API duy nhất .process() cực kỳ tinh gọn
        predicted_text = restorer.process(text)
        print("-" * 60)
        print("Đầu vào không dấu:", text)
        print("Kết quả thêm dấu :", predicted_text)
    print("\n=== PIPELINE HOÀN THÀNH BIÊN DỊCH BƯỚC 1 ===")