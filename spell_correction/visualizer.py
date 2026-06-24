import pandas as pd
from tqdm import tqdm
from common.logger import get_logger

logger = get_logger(__name__)


class Visualizer:
    def __init__(self, pipeline, abbr_engine, evaluator, word_to_idx):
        self.pipeline = pipeline
        self.abbr_engine = abbr_engine
        self.evaluator = evaluator
        self.word_to_idx = word_to_idx
        logger.info("Visualizer module online for qualitative output sampling.")

    def analyze_predictions(self, validation_df, stopwords, num_samples=200):
        logger.info(f"Sampling {num_samples} validation queries for performance breakdown...")
        
        exact_sentences = pd.DataFrame(columns=['Input', 'Fixed', 'Target'])
        error_sentence = pd.DataFrame(columns=['Input', 'Fixed', 'Target'])
        error_words = pd.DataFrame(columns=['Error', 'Correct'])

        target_df = validation_df.head(num_samples)

        pbar = tqdm(target_df.iterrows(), total=len(target_df), desc="Analyzing Sample Predictions", leave=False)
        for _, row in pbar: 
            input_sent = str(row['input'])
            target_sent = str(row['target'])

            cleaned_input = self.abbr_engine.replace_abbreviations(input_sent)
            _, error_indices = self.evaluator.find_misspelled_words_and_targets(cleaned_input, target_sent, self.word_to_idx)

            fixed_sentence_str = self.pipeline(cleaned_input, stopwords)

            target_tokens = target_sent.split()
            fixed_tokens = fixed_sentence_str.split()

            if len(target_tokens) != len(fixed_tokens):
                logger.debug(f"Dropped sentence due to token length mismatch: {input_sent}")
                continue

            errors_in_sentence = 0
            for idx in error_indices:
                if fixed_tokens[idx] != target_tokens[idx]:
                    errors_in_sentence += 1
                    error_words.loc[error_words.shape[0]] = [fixed_tokens[idx], target_tokens[idx]]

            if errors_in_sentence == 0 and not error_indices:
                exact_sentences.loc[exact_sentences.shape[0]] = [input_sent, fixed_sentence_str, target_sent]
            
            if errors_in_sentence != 0:
                error_sentence.loc[error_sentence.shape[0]] = [input_sent, fixed_sentence_str, target_sent]

        logger.info(f"Analysis completed | Exact sentences: {len(exact_sentences)} | Sentences with errors: {len(error_sentence)}")
        return exact_sentences, error_sentence, error_words
