import time
import torch
import torch.nn as nn

from common.config import load_config
from common.logger import get_logger

from diacritic_restoration import (
    DiacriticDataLoaderFactory,
    build_model,
    DiacriticRestorer,
    DiacriticTrainer,
    build_allowed_token_mask
)

logger = get_logger("DiacriticMainPipeline")

if __name__ == "__main__":
    logger.info("=== BẮT ĐẦU PIPELINE KHÔI PHỤC DẤU TIẾNG VIỆT ===")
    pipeline_start_time = time.time()

    try:
        config_path = "configs/diacritic_config.yaml"
        logger.info(f"Loading system configurations from: {config_path}")
        cfg = load_config(config_path)
        
        if cfg.training.device == "auto":
            cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Target compute hardware device auto-detected: {cfg.training.device.upper()}")

        logger.info("=== GIAI ĐOẠN 1: CHUẨN BỊ DỮ LIỆU & XÂY DỰNG TỪ ĐIỂN ===")

        data_factory = DiacriticDataLoaderFactory(cfg)
        train_loader, valid_loader, test_loader, char_vocab, word_vocab = data_factory.build_loaders_and_vocabs()
        
        allowed_mask = build_allowed_token_mask(char_vocab, cfg)

        # logger.info("=== GIAI ĐOẠN 2: KHỞI TẠO KIẾN TRÚC MẠNG NƠ-RON ===")
        # model = build_model(char_vocab, word_vocab, cfg)
        
        # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        #     logger.info(f"Multi-GPU environment detected: {torch.cuda.device_count()} GPUs available. Activating nn.DataParallel.")
        #     model = nn.DataParallel(model)

        # logger.info("=== GIAI ĐOẠN 3: BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN MÔ HÌNH (TRAINING LOOP) ===")
        # trainer = DiacriticTrainer(
        #     model=model,
        #     char_vocab=char_vocab,
        #     word_vocab=word_vocab,
        #     allowed_mask=allowed_mask,
        #     cfg=cfg
        # )
        # model = trainer.fit(train_loader, valid_loader, test_loader)

        logger.info("=== GIAI ĐOẠN 4: NẠP CHECKPOINT TỐI ƯU ĐỂ ĐÁNH GIÁ KIỂM THỬ ===")
        restorer = DiacriticRestorer(
            checkpoint_path=cfg.training.checkpoint_path,
            cfg=cfg
        )
        
        examples = [
            "toi ten la",
            "mot nguoi mat tich o ho chua nuoc nuoc trong.",
            "nhung hinh anh dac sac nhat ve vong chung ket world cup.",
            "apple se ra mat ipad duoc thiet ke lai trong thang 4?.",
        ]

        logger.info("=== DEMO SUY LUẬN THỰC TẾ (INFERENCE) ===")
        for idx, text in enumerate(examples, 1):
            inference_start = time.time()
            predicted_text = restorer.process(text)
            inference_time = time.time() - inference_start
            
            logger.info(f"Sample #{idx} | Inference time: {inference_time*1000:.2f}ms")
            logger.info(f"  > [IN]  : {text}")
            logger.info(f"  > [OUT] : {predicted_text}")
            
        total_pipeline_time = time.time() - pipeline_start_time
        logger.info(f"=== PIPELINE HOÀN THÀNH BIÊN DỊCH BƯỚC 1 TRONG {total_pipeline_time:.2f}s ===")

    except KeyError as e:
        logger.error(f"Critical configuration key mismatch error in YAML schema: {str(e)}", exc_info=True)
    except RuntimeError as e:
        logger.error(f"Hardware or CUDA Execution Environment failed during pipeline execution: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected structural failure within pipeline execution context: {str(e)}", exc_info=True)