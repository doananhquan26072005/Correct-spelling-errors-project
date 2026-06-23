import time
import torch
import torch.nn as nn
import pandas as pd

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
    logger.info("=== STARTING VIETNAMESE DIACRITIC RESTORATION PIPELINE ===")
    pipeline_start_time = time.time()

    try:
        config_path = "configs/diacritic_config.yaml"
        logger.info(f"Loading configurations from: {config_path}")
        cfg = load_config(config_path)
        
        if cfg.training.device == "auto":
            cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Target compute hardware device allocated: {cfg.training.device.upper()}")

        logger.info("=== DATA PREPARATION AND VOCABULARY BUILDING ===")
        full_df = pd.read_csv(cfg.data.csv_path)

        seed = cfg.training.seed if hasattr(cfg, "training") else 42
        full_df = full_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        train_ratio = cfg.data.train_ratio if hasattr(cfg, "data") else 0.8
        valid_ratio = cfg.data.valid_ratio if hasattr(cfg, "data") else 0.1

        total_samples = len(full_df)
        n_train = int(total_samples * train_ratio)
        n_valid = int(total_samples * valid_ratio)

        df_train = full_df.iloc[:n_train].reset_index(drop=True)
        df_valid = full_df.iloc[n_train:n_train + n_valid].reset_index(drop=True)
        df_test = full_df.iloc[n_train + n_valid:].reset_index(drop=True)

        logger.info(f"CSV split completed | Train: {len(df_train):,} | Valid: {len(df_valid):,} | Test: {len(df_test):,}")

        data_factory = DiacriticDataLoaderFactory(cfg)
        train_loader, valid_loader, test_loader, char_vocab, word_vocab = data_factory.build_loaders_and_vocabs(df_train, df_valid, df_test)
        
        allowed_mask = build_allowed_token_mask(char_vocab, cfg)

        logger.info("=== INITIALIZING NEURAL NETWORK ARCHITECTURE ===")
        model = build_model(char_vocab, word_vocab, cfg)
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logger.info(f"Multi-GPU environment detected: {torch.cuda.device_count()} GPUs available. Activating DataParallel.")
            model = nn.DataParallel(model)

        logger.info("=== STARTING MODEL TRAINING LOOP ===")
        trainer = DiacriticTrainer(
            model=model,
            char_vocab=char_vocab,
            word_vocab=word_vocab,
            allowed_mask=allowed_mask,
            cfg=cfg
        )
        model = trainer.fit(train_loader, valid_loader, test_loader)
        
        stopwords = set()
        logger.info("=== LOADING BEST CHECKPOINT FOR EVALUATION ===")
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

        logger.info("=== LIVE INFERENCE DEMO ===")
        for idx, text in enumerate(examples, 1):
            inference_start = time.time()
            predicted_text = restorer.process(text, stopwords)
            inference_time = time.time() - inference_start
            
            logger.info(f"Sample #{idx} | Inference time: {inference_time*1000:.2f}ms")
            logger.info(f"  > [IN]  : {text}")
            logger.info(f"  > [OUT] : {predicted_text}")
            
        total_pipeline_time = time.time() - pipeline_start_time
        logger.info(f"=== PIPELINE COMPLETED SUCCESSFULLY IN {total_pipeline_time:.2f}s ===")

    except KeyError as e:
        logger.error(f"Critical configuration key mismatch in YAML schema: {str(e)}", exc_info=True)
    except RuntimeError as e:
        logger.error(f"Hardware or CUDA execution failure: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected structural failure within pipeline context: {str(e)}", exc_info=True)