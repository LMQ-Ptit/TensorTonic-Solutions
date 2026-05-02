import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Tạo một tầng Embedding chuẩn trong PyTorch.
    """
    # Khởi tạo ma trận embedding với kích thước vocab x d_model
    return nn.Embedding(vocab_size, d_model)

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Chuyển đổi token indices thành scaled embeddings.
    Công thức: x = Embedding(tokens) * sqrt(d_model)
    """
    # 1. Look up vector từ ma trận embedding
    x = embedding(tokens)
    
    # 2. Nhân với căn bậc hai của d_model (theo Paper 'Attention Is All You Need')
    # Việc này giúp ổn định gradient và scale giá trị embedding
    x = x * math.sqrt(d_model)
    
    return x