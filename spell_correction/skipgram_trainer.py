import os
from typing import List, Dict, Set
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from common.logger import get_logger

logger = get_logger(__name__)


class SkipGram(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        out = self.linear(embed)
        return out
    

class SkipGramTrainer:
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = getattr(cfg, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.vocab: List[str] = []
        self.word_to_idx: Dict[str, int] = {}
        self.stopword: Set[str] = set()
        
        self.model: nn.Module = None
        self.embedding_weights: np.ndarray = None
        self.norm_embedding_matrix: np.ndarray = None
        
        logger.info("Initializing SkipGramTrainer...")
        self._load_resources()

    def _load_resources(self):
        vocab_path = self.cfg.paths.vocab_file
        stopwords_path = self.cfg.paths.stopwords_file

        logger.info(f"Loading resources | Vocab: {vocab_path} | Stopwords: {stopwords_path}")
        if not os.path.exists(vocab_path):
            logger.error(f"Vocabulary file missing at: {vocab_path}")
            raise FileNotFoundError(f"Vocab file not found at: {vocab_path}")
        if not os.path.exists(stopwords_path):
            logger.error(f"Stopwords file missing at: {stopwords_path}")
            raise FileNotFoundError(f"Stopwords file not found at: {stopwords_path}")

        with open(vocab_path, 'r', encoding='utf-8') as f:
            raw_vocab = f.read().splitlines()
        self.vocab = list(dict.fromkeys(raw_vocab))
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            self.stopword = set(f.read().splitlines())
            
        logger.info(f"Resources loaded | Unique Vocab Size: {len(self.vocab):,}")

    def build_dataset(self, target_sentences: pd.Series) -> DataLoader:
        window_size = self.cfg.model.window_size
        batch_size = self.cfg.skipgram_training.batch_size if (self.cfg and hasattr(self.cfg, 'skipgram_training')) else 512
        
        logger.info(f"Generating Center-Context pairs | Window: {window_size} | Batch: {batch_size}")
        start_time = time.time()
        skipgram_data = []

        for sentence in target_sentences:
            words = str(sentence).lower().split()
            filtered_words = [
                w for w in words 
                if w in self.word_to_idx and w not in self.stopword
            ]

            n_words = len(filtered_words)
            for i, word in enumerate(filtered_words):
                center = self.word_to_idx[word]

                for j in range(-window_size, window_size + 1):
                    if j == 0:
                        continue
                    if 0 <= i + j < n_words:
                        context_word = filtered_words[i + j]
                        context = self.word_to_idx[context_word]
                        skipgram_data.append((center, context))

        elapsed = time.time() - start_time
        logger.info(f"Dataset generated in {elapsed:.2f}s | Total instances: {len(skipgram_data):,}")
        
        centers = torch.LongTensor([pair[0] for pair in skipgram_data])
        contexts = torch.LongTensor([pair[1] for pair in skipgram_data])
        
        train_dataset = TensorDataset(centers, contexts)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def train(self, loader: DataLoader):
        vocab_size = len(self.vocab)
        embed_dim = self.cfg.model.embed_dim
        
        if self.cfg and hasattr(self.cfg, 'skipgram_training'):
            epochs = self.cfg.skipgram_training.epochs
            lr = self.cfg.skipgram_training.learning_rate
        else:
            epochs = 10
            lr = 0.001

        self.model = SkipGram(vocab_size, embed_dim).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        logger.info(f"Starting Skip-gram training on: {self.device.upper()} | Total Epochs: {epochs}")
        total_start_time = time.time()
        self.model.train()

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            total_loss = 0.0

            pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{epochs} [SkipGram]", leave=False)
            for batch_centers, batch_contexts in pbar:
                batch_centers = batch_centers.to(self.device)
                batch_contexts = batch_contexts.to(self.device)

                outputs = self.model(batch_centers)
                loss = criterion(outputs, batch_contexts)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_elapsed = time.time() - epoch_start_time
            avg_loss = total_loss / len(loader)
            logger.info(f"Epoch {epoch:02d}/{epochs} finished | Avg Loss: {avg_loss:.4f} | Time: {epoch_elapsed:.1f}s")
            
        total_train_time = time.time() - total_start_time
        logger.info(f"Training completed successfully in {total_train_time:.2f}s")
        self._extract_and_normalize_embeddings()

    def _extract_and_normalize_embeddings(self):
        logger.info("Extracting and normalizing embedding matrices...")
        self.model.eval()
        with torch.no_grad():
            self.embedding_weights = self.model.embedding.weight.data.cpu().numpy()

        norms = np.linalg.norm(self.embedding_weights, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  
        self.norm_embedding_matrix = self.embedding_weights / norms
        logger.info(f"L2 normalized matrix secured | shape: {self.norm_embedding_matrix.shape}")

    def save_model(self, output_path: str):
        if self.model is None:
            logger.error("Model is not trained yet. Aborting save.")
            raise ValueError("Model has not been trained. Cannot save checkpoint.")

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "vocab_size": len(self.vocab),
            "embed_dim": self.cfg.model.embed_dim,
            "word_to_idx": self.word_to_idx
        }
        torch.save(checkpoint, output_path)
        logger.info(f"Model checkpoint saved at: {output_path}")

        matrix_path = output_path.replace(".pth", "_matrix.npz")
        np.savez_compressed(
            matrix_path,
            embedding_matrix=self.embedding_weights,
            norm_embedding_matrix=self.norm_embedding_matrix,
            vocab=np.array(self.vocab)
        )
        logger.info(f"Compressed numpy embedding matrix saved at: {matrix_path}")

    def sanity_check(self, target_word: str, top_k: int = 5):
        if self.norm_embedding_matrix is None:
            logger.warning("No normalized matrix structure found. Skipping sanity check.")
            return
        if target_word not in self.word_to_idx:
            logger.warning(f"Target word '{target_word}' is out of vocabulary (OOV).")
            return

        word_idx = self.word_to_idx[target_word]
        word_vector = self.norm_embedding_matrix[word_idx]
        
        scores = np.dot(self.norm_embedding_matrix, word_vector)
        top_indices = np.argsort(scores)[::-1][:top_k + 1]
        
        logger.info(f"Vector Space Sanity Check for token: '{target_word}'")
        count = 0
        for idx in top_indices:
            sim_word = self.vocab[idx]
            if sim_word == target_word:
                continue
            logger.info(f"  -> Neighbor: {sim_word:<15} | Cosine Similarity: {scores[idx]:.4f}")
            count += 1
            if count >= top_k:
                break
    
    def load_model(self, model_path: str):
        matrix_path = model_path.replace(".pth", "_matrix.npz")
        if not os.path.exists(matrix_path):
            logger.error(f"Compressed matrix file not found at: {matrix_path}")
            raise FileNotFoundError(f"Matrix file not found at: {matrix_path}")

        logger.info(f"Loading pre-trained embedding matrices from: {matrix_path}")
        loaded_data = np.load(matrix_path)
        
        self.embedding_weights = loaded_data['embedding_matrix']
        self.norm_embedding_matrix = loaded_data['norm_embedding_matrix']
        
        loaded_vocab = loaded_data['vocab'].tolist()
        if len(self.vocab) != len(loaded_vocab):
            logger.warning("Current vocab size differs from model file. Overriding with model vocab.")
            self.vocab = loaded_vocab
            self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}

        logger.info(f"Loaded L2 normalized embedding matrix | shape: {self.norm_embedding_matrix.shape}")

    def get_norm_embedding(self) -> np.ndarray:
        if self.norm_embedding_matrix is None:
            logger.error("norm_embedding_matrix is None. Model must be trained or loaded first.")
            raise ValueError("Normalized matrix not loaded. Please train or load model first.")
        return self.norm_embedding_matrix