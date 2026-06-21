import os
from typing import List, Dict, Set
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
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
    """Class chịu trách nhiệm quản lý toàn bộ vòng đời của mô hình Word Embedding Skip-gram:
    Từ tiền xử lý dữ liệu, tạo cặp Center-Context, huấn luyện, đến trích xuất ma trận chuẩn hóa."""
    
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
        # Tải tài nguyên ngay khi khởi tạo
        self._load_resources()

    def _load_resources(self):
        """Phương thức nội bộ: Nạp từ điển Vocab và Stopwords từ đường dẫn cấu hình."""
        vocab_path = self.cfg.paths.vocab_file
        stopwords_path = self.cfg.paths.stopwords_file

        logger.info(f"Loading external dictionary components | Vocab: {vocab_path} | Stopwords: {stopwords_path}")
        if not os.path.exists(vocab_path):
            logger.error(f"Vocabulary file missing at path: {vocab_path}")
            raise FileNotFoundError(f"Không tìm thấy file vocab tại: {vocab_path}")
        if not os.path.exists(stopwords_path):
            logger.error(f"Stopwords file missing at path: {stopwords_path}")
            raise FileNotFoundError(f"Không tìm thấy file stopwords tại: {stopwords_path}")

        with open(vocab_path, 'r', encoding='utf-8') as f:
            raw_vocab = f.read().splitlines()
        self.vocab = list(dict.fromkeys(raw_vocab))
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            self.stopword = set(f.read().splitlines())
            
        logger.info(f"Resources loaded successfully. Unique Vocab Size: {len(self.vocab):,}")

    def build_dataset(self, target_sentences: pd.Series) -> DataLoader:
        """Xử lý thô văn bản đích và chuyển đổi thành DataLoader chứa các cặp Center-Context."""
        window_size = self.cfg.model.window_size
        batch_size = self.cfg.skipgram_training.batch_size if (self.cfg and hasattr(self.cfg, 'skipgram_training')) else 512
        
        logger.info(f"Generating Center-Context word training pairs (Window Size: {window_size}, Batch Size: {batch_size})...")
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
        logger.info(f"Dataset generated in {elapsed:.2f}s | Total generated training instances: {len(skipgram_data):,}")
        
        centers = torch.LongTensor([pair[0] for pair in skipgram_data])
        contexts = torch.LongTensor([pair[1] for pair in skipgram_data])
        
        train_dataset = TensorDataset(centers, contexts)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def train(self, loader: DataLoader):
        """Khởi tạo trọng số mạng và kích hoạt vòng lặp huấn luyện chính."""
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
        
        logger.info(f"Activating Skip-gram optimization on hardware: {self.device.upper()} | Total Epochs: {epochs}")
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
        logger.info(f"Neural network optimization finished. Total optimization time: {total_train_time:.2f}s")
        self._extract_and_normalize_embeddings()

    def _extract_and_normalize_embeddings(self):
        """Phương thức nội bộ: Rút trích và thực hiện chuẩn hóa L2 cho ma trận vector từ."""
        logger.info("Extracting raw embedding matrices and calculating L2 normalizations...")
        self.model.eval()
        with torch.no_grad():
            self.embedding_weights = self.model.embedding.weight.data.cpu().numpy()

        norms = np.linalg.norm(self.embedding_weights, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  
        self.norm_embedding_matrix = self.embedding_weights / norms
        logger.info(f"L2 Normalization matrix secured with structural shape: {self.norm_embedding_matrix.shape}")

    def save_model(self, output_path: str):
        """Lưu trữ checkpoint mô hình PyTorch và ma trận nén Numpy."""
        if self.model is None:
            logger.error("Model has not been optimized yet. Aborting serialization.")
            raise ValueError("Mô hình chưa được huấn luyện. Không thể lưu trữ.")

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
        logger.info(f"PyTorch model checkpoint weight dict safely stored at: {output_path}")

        matrix_path = output_path.replace(".pth", "_matrix.npz")
        np.savez_compressed(
            matrix_path,
            embedding_matrix=self.embedding_weights,
            norm_embedding_matrix=self.norm_embedding_matrix,
            vocab=np.array(self.vocab)
        )
        logger.info(f"Compressed compressed embedding numpy matrix blocks secured at: {matrix_path}")

    def sanity_check(self, target_word: str, top_k: int = 5):
        """Kiểm tra nhanh chất lượng không gian vector ngữ nghĩa dựa trên Cosine Similarity."""
        if self.norm_embedding_matrix is None:
            logger.warning("No normalized matrix structures found. Execution dropped.")
            return
        if target_word not in self.word_to_idx:
            logger.warning(f"Target verification token '{target_word}' is out of vocabulary (OOV).")
            return

        word_idx = self.word_to_idx[target_word]
        word_vector = self.norm_embedding_matrix[word_idx]
        
        scores = np.dot(self.norm_embedding_matrix, word_vector)
        top_indices = np.argsort(scores)[::-1][:top_k + 1]
        
        logger.info(f"Qualitative Vector Space Sanity Check for token: '{target_word}'")
        count = 0
        for idx in top_indices:
            sim_word = self.vocab[idx]
            if sim_word == target_word:
                continue
            logger.info(f"  -> NearNeighbor: {sim_word:<15} | Cosine Similarity Score: {scores[idx]:.4f}")
            count += 1
            if count >= top_k:
                break
    
    def load_model(self, model_path: str):
        """Tải lại trọng số mô hình và ma trận embedding đã nén từ ổ cứng."""
        matrix_path = model_path.replace(".pth", "_matrix.npz")
        
        if not os.path.exists(matrix_path):
            logger.error(f"Compressed matrix file not found at: {matrix_path}")
            raise FileNotFoundError(f"❌ Không tìm thấy file ma trận nén tại: {matrix_path}")

        logger.info(f"Loading pre-trained embedding matrices from: {matrix_path}")
        loaded_data = np.load(matrix_path)
        
        self.embedding_weights = loaded_data['embedding_matrix']
        self.norm_embedding_matrix = loaded_data['norm_embedding_matrix']
        
        # Cập nhật lại vocab nếu vocab từ config và file npz có sự sai lệch
        loaded_vocab = loaded_data['vocab'].tolist()
        if len(self.vocab) != len(loaded_vocab):
            logger.warning("Kích thước vocab hiện tại khác với vocab trong file model. Đang ghi đè bằng vocab của model.")
            self.vocab = loaded_vocab
            self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}

        logger.info(f"Successfully loaded L2 normalized embedding matrix with shape: {self.norm_embedding_matrix.shape}")

    def get_norm_embedding(self) -> np.ndarray:
        """Phương thức getter để trích xuất ma trận chuẩn hóa truyền cho các module khác."""
        if self.norm_embedding_matrix is None:
            logger.error("norm_embedding_matrix is None. Please train or load the model first.")
            raise ValueError("❌ Ma trận chuẩn hóa chưa được nạp. Hãy gọi train() hoặc load_model() trước.")
        return self.norm_embedding_matrix