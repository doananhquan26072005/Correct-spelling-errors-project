# train_skipgram.py
import os
import json
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Dict, Set, Tuple

class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        
        # Tải tài nguyên ngay khi khởi tạo
        self._load_resources()

    def _load_resources(self):
        """Phương thức nội bộ: Nạp từ điển Vocab và Stopwords từ đường dẫn cấu hình."""
        print("[*] Loading Vocab and Stopwords resources...")
        vocab_path = self.cfg.paths.vocab_file
        stopwords_path = self.cfg.paths.stopwords_file

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Không tìm thấy file vocab tại: {vocab_path}")
        if not os.path.exists(stopwords_path):
            raise FileNotFoundError(f"Không tìm thấy file stopwords tại: {stopwords_path}")

        with open(vocab_path, 'r', encoding='utf-8') as f:
            raw_vocab = f.read().splitlines()
        self.vocab = list(dict.fromkeys(raw_vocab))
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            self.stopword = set(f.read().splitlines())
            
        print(f"  • Kích thước từ vựng (Vocab Size): {len(self.vocab):,}")

    def build_dataset(self, target_sentences: pd.Series) -> DataLoader:
        """Xử lý thô văn bản đích và chuyển đổi thành DataLoader chứa các cặp Center-Context."""
        window_size = self.cfg.model.window_size
        
        # Đọc động thông số batch_size từ nhánh skipgram_training trong config
        if self.cfg and hasattr(self.cfg, 'skipgram_training'):
            batch_size = self.cfg.skipgram_training.batch_size
        else:
            batch_size = 512
        
        print(f"[*] Đang sinh các cặp Center - Context (Window Size = {window_size})...")
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

        print(f"  • Tổng số lượng cặp Skip-gram tạo ra: {len(skipgram_data):,}")
        
        centers = torch.LongTensor([pair[0] for pair in skipgram_data])
        contexts = torch.LongTensor([pair[1] for pair in skipgram_data])
        
        train_dataset = TensorDataset(centers, contexts)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def train(self, loader: DataLoader):
        """Khởi tạo trọng số mạng và kích hoạt vòng lặp huấn luyện chính."""
        vocab_size = len(self.vocab)
        embed_dim = self.cfg.model.embed_dim
        
        # Đọc động thông số chu kỳ học và lr từ cấu hình
        if self.cfg and hasattr(self.cfg, 'skipgram_training'):
            epochs = self.cfg.skipgram_training.epochs
            lr = self.cfg.skipgram_training.learning_rate
        else:
            epochs = 10
            lr = 0.001

        self.model = SkipGram(vocab_size, embed_dim).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"[*] Kích hoạt huấn luyện Skip-gram trên {self.device} trong {epochs} Epochs...")
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for batch_centers, batch_contexts in loader:
                batch_centers = batch_centers.to(self.device)
                batch_contexts = batch_contexts.to(self.device)

                outputs = self.model(batch_centers)
                loss = criterion(outputs, batch_contexts)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"  • Epoch {epoch+1:02d}/{epochs} -> Total Loss: {total_loss:.4f}")
            
        print("[+] Quá trình huấn luyện mạng nơ-ron hoàn tất!")
        self._extract_and_normalize_embeddings()

    def _extract_and_normalize_embeddings(self):
        """Phương thức nội bộ: Rút trích và thực hiện chuẩn hóa L2 cho ma trận vector từ."""
        self.model.eval()
        with torch.no_grad():
            self.embedding_weights = self.model.embedding.weight.data.cpu().numpy()

        norms = np.linalg.norm(self.embedding_weights, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  
        self.norm_embedding_matrix = self.embedding_weights / norms

    def save_model(self, output_path: str):
        """Lưu trữ checkpoint mô hình PyTorch và ma trận nén Numpy."""
        if self.model is None:
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
        print(f"[+] Đã lưu checkpoint State Dict tại: {output_path}")

        matrix_path = output_path.replace(".pth", "_matrix.npz")
        np.savez_compressed(
            matrix_path,
            embedding_matrix=self.embedding_weights,
            norm_embedding_matrix=self.norm_embedding_matrix,
            vocab=np.array(self.vocab)
        )
        print(f"[+] Đã lưu ma trận nén Embedding tại: {matrix_path}")

    def sanity_check(self, target_word: str, top_k: int = 5):
        """Kiểm tra nhanh chất lượng không gian vector ngữ nghĩa dựa trên Cosine Similarity."""
        if self.norm_embedding_matrix is None:
            print("⚠️ Chưa có ma trận embedding để kiểm tra.")
            return
        if target_word not in self.word_to_idx:
            print(f"⚠️ Từ '{target_word}' không tồn tại trong từ điển (OOV).")
            return

        word_idx = self.word_to_idx[target_word]
        word_vector = self.norm_embedding_matrix[word_idx]
        
        scores = np.dot(self.norm_embedding_matrix, word_vector)
        top_indices = np.argsort(scores)[::-1][:top_k + 1]
        
        print(f"\n[Thử nghiệm ngữ nghĩa] Top {top_k} từ gần nhất với '{target_word}':")
        count = 0
        for idx in top_indices:
            sim_word = self.vocab[idx]
            if sim_word == target_word:
                continue
            print(f"  • {sim_word:<15} -> Cosine Score: {scores[idx]:.4f}")
            count += 1
            if count >= top_k:
                break

# trainer.py
import os
import warnings
import numpy as np
import lightgbm as lgb
from typing import Dict, Any

class LightGBMRankerTrainer:
    """Class quản lý toàn bộ vòng đời cấu hình, huấn luyện và đóng gói 
    mô hình xếp hạng ứng viên (LightGBM LambdaRanker)."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.ranker = None
        warnings.filterwarnings("ignore", category=UserWarning)

    def load_training_data(self, data_path: str) -> tuple:
        """Đọc tập dữ liệu trích xuất đặc trưng (.npz) được chuẩn bị sẵn."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"❌ Không tìm thấy tệp dữ liệu huấn luyện tại: {data_path}")
            
        print(f"[*] Đang nạp dữ liệu từ {data_path}...")
        loaded_data = np.load(data_path)
        X_train = loaded_data['X_train']
        y_train = loaded_data['y_train']
        group_train = loaded_data['group_train']
        
        print(f"  • X_train shape: {X_train.shape}")
        print(f"  • Số lượng nhóm (queries): {len(group_train):,}")
        return X_train, y_train, group_train

    def train(self, X_train: np.ndarray, y_train: np.ndarray, group_train: np.ndarray) -> lgb.LGBMRanker:
        """Khởi tạo cấu hình và kích hoạt chu trình huấn luyện mô hình Ranker."""
        print("[*] Khởi tạo mô hình LGBMRanker (lambdarank)...")
        
        # Đọc động toàn bộ siêu tham số của mô hình cây phân hạng từ ranker_training trong config
        if self.cfg and hasattr(self.cfg, 'ranker_training'):
            r_train = self.cfg.ranker_training
            lr = r_train.learning_rate
            num_leaves = r_train.num_leaves
            min_child_samples = r_train.min_child_samples
            random_state = r_train.random_state
            eval_at = r_train.eval_at
        else:
            lr = 0.05
            num_leaves = 31
            min_child_samples = 20
            random_state = 42
            eval_at = [1, 3, 5]

        self.ranker = lgb.LGBMRanker(
            objective='lambdarank',
            metric='ndcg',
            eval_at=eval_at,
            label_gain=[0, 1],
            learning_rate=lr,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            random_state=random_state
        )

        print("[*] Đang fit mô hình trên tập dữ liệu đặc trưng...")
        self.ranker.fit(
            X=X_train,
            y=y_train,
            group=group_train
        )
        print("[+] Huấn luyện LightGBM Ranker hoàn tất thành công!")
        self._log_feature_importances()
        return self.ranker

    def _log_feature_importances(self):
        """Phương thức nội bộ: Thống kê và hiển thị mức độ đóng góp của từng đặc trưng."""
        if self.ranker is None:
            return
            
        feature_names = [
            'ken_score', 'word2vec_sim', 'unigram_count', 
            'bigram_count', 'trigram_count', 'edit_dist', 'length_ratio'
        ]
        
        print("\n" + "-"*15 + " ĐÓNG GÓP CỦA CÁC ĐẶC TRƯNG (FEATURE IMPORTANCE) " + "-"*15)
        for name, importance in zip(feature_names, self.ranker.feature_importances_):
            print(f"  • {name:<15}: {importance}")
        print("-" * 76 + "\n")

    def save_model(self, model_path: str):
        """Lưu trữ mô hình đã huấn luyện ra file text của LightGBM hoặc checkpoint binary."""
        if self.ranker is None:
            raise ValueError("❌ Mô hình chưa được huấn luyện. Không thể lưu checkpoint.")
        
        output_dir = os.path.dirname(model_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.ranker.booster_.save_model(model_path)
        print(f"[+] Đã lưu checkpoint mô hình tốt nhất vào {model_path}")

    def load_model(self, model_path: str) -> lgb.LGBMRanker:
        """Tải mô hình từ checkpoint có sẵn để phục vụ cho Inference/Evaluation nhanh."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Không tìm thấy checkpoint mô hình tại: {model_path}")
            
        print(f"[*] Đang tải mô hình từ {model_path}...")
        bst = lgb.Booster(model_file=model_path)
        self.ranker = lgb.LGBMRanker()
        self.ranker.booster_ = bst
        return self.ranker