import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Áp dụng mạng Feed-Forward theo từng vị trí (position-wise).
    Công thức: FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    # 1. Tầng tuyến tính thứ nhất (thường mở rộng chiều dữ liệu từ d_model lên d_ff)
    # x shape: (batch, seq, d_model), W1 shape: (d_model, d_ff)
    z1 = np.dot(x, W1) + b1
    
    # 2. Hàm kích hoạt ReLU: giữ lại các giá trị dương, biến các giá trị âm thành 0
    a1 = np.maximum(0, z1)
    
    # 3. Tầng tuyến tính thứ hai (nén chiều dữ liệu từ d_ff quay về d_model)
    # a1 shape: (batch, seq, d_ff), W2 shape: (d_ff, d_model)
    output = np.dot(a1, W2) + b2
    
    return output