import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Thực hiện Layer Normalization.
    x shape: (batch, seq_len, d_model)
    gamma, beta shape: (d_model,)
    """
    
    # 1. Tính giá trị trung bình (mean) theo chiều cuối cùng (d_model)
    # axis=-1 và keepdims=True để dễ dàng thực hiện phép trừ sau đó
    mean = np.mean(x, axis=-1, keepdims=True)
    
    # 2. Tính phương sai (variance) theo chiều cuối cùng
    var = np.var(x, axis=-1, keepdims=True)
    
    # 3. Chuẩn hóa x: (x - mean) / sqrt(var + eps)
    # eps giúp tránh lỗi chia cho 0 khi phương sai quá nhỏ
    x_normalized = (x - mean) / np.sqrt(var + eps)
    
    # 4. Scale bằng gamma và shift bằng beta (các tham số có thể học được)
    output = gamma * x_normalized + beta
    
    return output