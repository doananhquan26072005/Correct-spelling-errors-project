import os
import time
import numpy as np
import pandas as pd
import torch
import kenlm
import gdown
from datasets import load_dataset

from common.logger import get_logger
from common.config import load_config
from spell_correction import (
    process_dataset,
    split_data,
    ResourceLoader,
    AbbreviationProcessor,
    CandidateGenerator,
    FeatureExtractor,
    SpellCorrectionPipeline,
    Evaluator,
    Visualizer,
    LightGBMRankerTrainer,
    SkipGramTrainer,
    SkipGram
)

logger = get_logger("SpellCorrectionMainPipeline")

def main():
    logger.info("=== STARTING VIETNAMESE SPELL CORRECTION PIPELINE ===")
    pipeline_start_time = time.time()

    try:
        config_path = "configs/correction_config.yaml"
        logger.info(f"Loading configurations from: {config_path}")
        cfg = load_config(config_path)

        if not hasattr(cfg, "DEVICE"):
            cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Target compute hardware device allocated: {cfg.DEVICE.upper()}")

        logger.info("=== LOADING SYSTEM RESOURCES ===")
        resource_loader = ResourceLoader(cfg)
        resources = resource_loader.load_vocab_and_dicts()
        
        word_to_idx = resources["word_to_idx"]
        vocab = resources["vocab"]
        stopwords = resources["stopwords"]
        telex_dict = resources["telex_dict"]

        abbr_engine = AbbreviationProcessor(cfg.paths.teen_code_file)

        logger.info("=== LOADING AND PREPROCESSING CORPUS ===")
        corpus_name = "yammdd/vietnamese-error-correction-corpus"
        logger.info(f"Downloading corpus: {corpus_name}")
        
        dataset = load_dataset(corpus_name)
        
        logger.info("Mapping dataset tokens...")
        df = dataset.map(
            process_dataset,
            batched=True,
            remove_columns=dataset['train'].column_names,
            fn_kwargs={"word_to_idx": word_to_idx}
        )

        df1, _ = split_data(df)
        df1_train = pd.DataFrame(df1['train'])
        df1_valid = pd.DataFrame(df1['validation'])

        logger.info("=== INITIALIZING INFERENCE ENGINES ===")
        logger.info(f"Loading KenLM model from: {cfg.paths.trigram_lm_file}")
        model_lm = kenlm.Model(cfg.paths.trigram_lm_file)

        generator = CandidateGenerator(vocab=vocab, telex_dict=telex_dict, cfg=cfg)
        
        logger.info("Fitting training targets for N-gram frequencies...")
        generator.fit_ngram_counts(df1_train['target'])

        skipgram_trainer = SkipGramTrainer(cfg)
        skipgram_train_dataset = skipgram_trainer.build_dataset(df1_train["target"])
        skipgram_trainer.train(skipgram_train_dataset)
        norm_embedding_matrix = skipgram_trainer.get_norm_embedding()

        extractor_engine = FeatureExtractor(
            word_to_idx=word_to_idx,
            norm_embedding_matrix=norm_embedding_matrix,
            model_lm=model_lm,
            counts_1=generator.counts_1,
            counts_2=generator.counts_2,
            counts_3=generator.counts_3,
            cfg=cfg
        )

        evaluator = Evaluator(model_lm=model_lm, config=cfg)

        logger.info("=== TRAINING LIGHTGBM RANKER ===")
        ranker_trainer = LightGBMRankerTrainer(
            abbr_processor=abbr_engine,
            evaluator=evaluator,
            generator=generator,
            feature_extractor=extractor_engine,
            cfg=cfg
        )
        
        X_train, y_train, group_train = ranker_trainer.build_dataset(df1_train, stopwords)
        ranker = ranker_trainer.train(X_train, y_train, group_train)

        logger.info("=== PACKAGING INTEGRATED PIPELINE ===")
        pipeline = SpellCorrectionPipeline(
            cfg=cfg,
            evaluator=evaluator,
            model_lm=model_lm,
            generator=generator,
            extractor_engine=extractor_engine,
            ranker=ranker,
            word_to_idx=word_to_idx
        )

        logger.info("=== EXECUTING EVALUATION METRICS LOOP ===")
        evaluation_start_time = time.time()

        evaluator.evaluate_error_detection(
            validation_df=df1_valid, 
            abbr_engine=abbr_engine
        )

        evaluator.evaluate_ranking_performance(
            validation_df=df1_valid,
            abbr_engine=abbr_engine,
            extractor_engine=extractor_engine,
            generator=generator,
            ranker=ranker,
            stopwords=stopwords,
        )

        evaluator.evaluate_word_accuracy(
            validation_df=df1_valid,
            abbr_engine=abbr_engine,
            pipeline_correct_fn=pipeline.correct_sentence,
            stopwords=stopwords,
            word_to_idx=word_to_idx
        )

        evaluator.evaluate_end_to_end(
            validation_df=df1_valid,
            abbr_engine=abbr_engine,
            pipeline_correct_fn=pipeline.correct_sentence,
            stopwords=stopwords
        )
        logger.info(f"Evaluation cycle completed in {time.time() - evaluation_start_time:.2f}s.")

        logger.info("=== VISUALIZING PREDICTION SAMPLES ===")
        visualizer = Visualizer(
            pipeline=pipeline,
            abbr_engine=abbr_engine,
            evaluator=evaluator,
            word_to_idx=word_to_idx
        )
        
        exact_sentences, error_sentence, error_words = visualizer.analyze_predictions(
            validation_df=df1_valid,
            stopwords=stopwords,
            num_samples=200
        )

        logger.info("-----------------------------------------------------------------------")
        logger.info(f" [+] Exact sentences (Exact Match) : {len(exact_sentences):,}")
        logger.info(f" [+] Sentences with errors remaining : {len(error_sentence):,}")
        logger.info(f" [+] Total error tokens registered  : {len(error_words):,}")
        logger.info("-----------------------------------------------------------------------")

        total_pipeline_time = time.time() - pipeline_start_time
        logger.info(f"=== PIPELINE COMPLETED SUCCESSFULLY IN {total_pipeline_time:.2f}s ===")

    except FileNotFoundError as e:
        logger.error(f"Critical data path verification failed: {str(e)}", exc_info=True)
    except ValueError as e:
        logger.error(f"Value alignment or formatting failure: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected structural failure within pipeline context: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()