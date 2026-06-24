import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import kenlm
import gdown
from datasets import load_dataset
from typing import List, Tuple, Set

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

from diacritic_restoration import (
    DiacriticDataLoaderFactory,
    build_model,
    DiacriticRestorer,
    DiacriticTrainer,
    build_allowed_token_mask
)

logger = get_logger("NLPPipeline")

def main():
    logger.info("=== STARTING INTEGRATED NLP SYSTEM PIPELINE ===")
    pipeline_start_time = time.time()

    try:
        spell_config_path = "configs/correction_config.yaml"
        dia_config_path = "configs/diacritic_config.yaml"
        
        logger.info(f"Loading configs from: {spell_config_path} & {dia_config_path}")
        spell_cfg = load_config(spell_config_path)
        dia_cfg = load_config(dia_config_path)

        if not hasattr(spell_cfg, "DEVICE"):
            spell_cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            
        if dia_cfg.training.device == "auto":
            dia_cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        logger.info(f"Hardware allocated - Spell: {spell_cfg.DEVICE.upper()} | Diacritic: {dia_cfg.training.device.upper()}")

        logger.info("=== LOADING SYSTEM RESOURCES ===")
        resource_loader = ResourceLoader(spell_cfg)
        resources = resource_loader.load_vocab_and_dicts()
        
        word_to_idx = resources["word_to_idx"]
        vocab = resources["vocab"]
        stopwords = resources["stopwords"]
        telex_dict = resources["telex_dict"]

        abbr_engine = AbbreviationProcessor(spell_cfg.paths.teen_code_file)

        logger.info("=== LOADING AND PREPROCESSING CORPUS ===")
        corpus_name = "yammdd/vietnamese-error-correction-corpus"
        dataset = load_dataset(corpus_name)
        
        df = dataset.map(
            process_dataset,
            batched=True,
            remove_columns=dataset['train'].column_names,
            fn_kwargs={"word_to_idx": word_to_idx}
        )

        df_test = pd.DataFrame(df['test'])
        df1, df2 = split_data(df)
        
        df1_train = pd.DataFrame(df1['train'])
        df1_valid = pd.DataFrame(df1['validation'])
        df1_test = pd.DataFrame(df1['test'])

        df2_train = pd.DataFrame(df2['train'])
        df2_valid = pd.DataFrame(df2['validation'])
        df2_test = pd.DataFrame(df2['test'])
        logger.info(f"Corpus split completed | df1 train: {len(df1_train)} | df2 train: {len(df2_train)}")

        logger.info("=== INITIALIZING INFERENCE ENGINES ===")
        model_lm = kenlm.Model(spell_cfg.paths.trigram_lm_file)

        generator = CandidateGenerator(vocab=vocab, telex_dict=telex_dict, cfg=spell_cfg)
        generator.fit_ngram_counts(df1_train['target'])

        model_skipgram = SkipGram(len(vocab), spell_cfg.model.embed_dim).to(spell_cfg.DEVICE)
        gdown.download(id=spell_cfg.paths.skipgram_model_url, output=spell_cfg.paths.skipgram_model_file, quiet=False)
        model_skipgram.load_state_dict(torch.load(spell_cfg.paths.skipgram_model_file, map_location=spell_cfg.DEVICE, weights_only=True))
        
        embeddings = model_skipgram.embedding.weight.data
        embedding_matrix = embeddings.cpu().numpy()
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1 
        norm_embedding_matrix = embedding_matrix / norms

        extractor_engine = FeatureExtractor(
            word_to_idx=word_to_idx,
            norm_embedding_matrix=norm_embedding_matrix,
            model_lm=model_lm,
            counts_1=generator.counts_1,
            counts_2=generator.counts_2,
            counts_3=generator.counts_3,
            cfg=spell_cfg
        )

        evaluator = Evaluator(model_lm=model_lm, config=spell_cfg)

        logger.info("=== LOADING FEATURE MATRIX & TRAINING LIGHTGBM RANKER ===")
        ranker_trainer = LightGBMRankerTrainer(
            abbr_processor=abbr_engine,
            evaluator=evaluator,
            generator=generator,
            feature_extractor=extractor_engine,
            cfg=spell_cfg
        )
        
        X_train, y_train, group_train = ranker_trainer.load_training_data(spell_cfg.paths.lightgbm_data_file)
        ranker = ranker_trainer.train(X_train, y_train, group_train)

        logger.info("=== EVALUATING SPELL CORRECTION PIPELINE ===")
        pipeline = SpellCorrectionPipeline(
            cfg=spell_cfg, abbr_processor=abbr_engine, evaluator=evaluator, 
            model_lm=model_lm, generator=generator, extractor_engine=extractor_engine,
            ranker=ranker, word_to_idx=word_to_idx
        )

        # evaluator.evaluate_error_detection(validation_df=df1_valid, abbr_engine=abbr_engine)
        # evaluator.evaluate_ranking_performance(
        #     validation_df=df1_valid, abbr_engine=abbr_engine,
        #     extractor_engine=extractor_engine, generator=generator,
        #     ranker=ranker, stopwords=stopwords
        # )
        # evaluator.evaluate_word_accuracy(
        #     validation_df=df1_valid, abbr_engine=abbr_engine,
        #     pipeline_correct_fn=pipeline.correct_sentence,
        #     stopwords=stopwords, word_to_idx=word_to_idx
        # )
        # evaluator.evaluate_end_to_end(
        #     validation_df=df1_valid, abbr_engine=abbr_engine,
        #     pipeline_correct_fn=pipeline.correct_sentence, stopwords=stopwords
        # )

        visualizer = Visualizer(pipeline=pipeline.correct_sentence, abbr_engine=abbr_engine, evaluator=evaluator, word_to_idx=word_to_idx)
        exact_sentences, error_sentence, error_words = visualizer.analyze_predictions(
            validation_df=df1_valid, stopwords=stopwords, num_samples=10
        )

        logger.info("=== RUNNING DIACRITIC RESTORATION PIPELINE ON DF2 ===")
        data_factory = DiacriticDataLoaderFactory(dia_cfg)
        train_loader, valid_loader, test_loader, char_vocab, word_vocab = data_factory.build_loaders_and_vocabs(df2_train, df2_valid, df2_test)
        allowed_mask = build_allowed_token_mask(char_vocab, dia_cfg)

        restorer = DiacriticRestorer(
            checkpoint_path=dia_cfg.training.checkpoint_path,
            cfg=dia_cfg
        )
        
        # logger.info("=== LIVE INFERENCE DEMO ON DF2 ===")
        # col_name = 'input' if 'input' in df2_valid.columns else df2_valid.columns[0]
        # df2_examples = df2_valid[col_name].dropna().head(10).tolist()

        # for idx, text in enumerate(df2_examples, 1):
        #     inference_start = time.time()
        #     predicted_text = restorer.process(text, stopwords)
        #     inference_time = time.time() - inference_start
            
        #     logger.info(f"Sample #{idx} | Inference time: {inference_time*1000:.2f}ms")
        #     logger.info(f"  > [IN]  : {text}")
        #     logger.info(f"  > [OUT] : {predicted_text}")

        # logger.info("=== QUANTITATIVE EVALUATION FOR DIACRITIC RESTORATION ===")
        # evaluator.evaluate_word_accuracy(
        #     validation_df=df2_valid,
        #     abbr_engine=abbr_engine, 
        #     pipeline_correct_fn=restorer.process, 
        #     stopwords=stopwords,
        #     word_to_idx=word_to_idx
        # )

        # evaluator.evaluate_end_to_end(
        #     validation_df=df2_valid,
        #     abbr_engine=abbr_engine,
        #     pipeline_correct_fn=restorer.process, 
        #     stopwords=stopwords
        # )

        logger.info("=== EVALUATING JOINT PIPELINE ON UNSPLIT GLOBAL TEST SET ===")

        def joint_pipeline(text, stopwords=stopwords):
            if not text or not isinstance(text, str):
                return ""
            logger.debug(f"Input text: {text}")
            for char in text:
                if char in "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ":
                    logger.debug("Input text already contains diacritics. Skipping diacritic restoration.")
                    final_clean_text = pipeline.correct_sentence(text, stopwords)
                    logger.debug(f"Final corrected text: {final_clean_text}")
                    return final_clean_text

            restored_text = restorer.process(text, stopwords)
            logger.debug(f"Restored text: {restored_text}")
            final_clean_text = pipeline.correct_sentence(restored_text, stopwords)
            logger.debug(f"Final corrected text: {final_clean_text}")
            
            return final_clean_text

        logger.info("Running joint evaluation loop (Diacritic + Spell) on df_test...")
        
        # evaluator.evaluate_word_accuracy(
        #     validation_df=df_test, 
        #     abbr_engine=abbr_engine,
        #     pipeline_correct_fn=joint_pipeline, 
        #     stopwords=stopwords,
        #     word_to_idx=word_to_idx
        # )

        # evaluator.evaluate_end_to_end(
        #     validation_df=df_test,
        #     abbr_engine=abbr_engine,
        #     pipeline_correct_fn=joint_pipeline,
        #     stopwords=stopwords
        # )
        visualizer = Visualizer(pipeline=joint_pipeline, abbr_engine=abbr_engine, evaluator=evaluator, word_to_idx=word_to_idx)
        exact_sentences, error_sentence, error_words = visualizer.analyze_predictions(
            validation_df=df_test, stopwords=stopwords, num_samples=10
        )
        total_pipeline_time = time.time() - pipeline_start_time
        logger.info(f"=== COMPREHENSIVE PIPELINE COMPLETED IN {total_pipeline_time:.2f}s ===")

    except Exception as e:
        logger.error(f"Unexpected structural failure within pipeline: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
