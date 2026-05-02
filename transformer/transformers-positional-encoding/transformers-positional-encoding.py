import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Tạo ma trận Positional Encoding sử dụng hàm sin và cos.
    """
    # 1. Khởi tạo ma trận toàn số 0 với kích thước (seq_length, d_model)
    pe = np.zeros((seq_length, d_model))
    
    # 2. Tạo vector vị trí (pos) từ 0 đến seq_length - 1
    # Chuyển thành shape (seq_length, 1) để nhân ma trận
    position = np.arange(seq_length)[:, np.newaxis]
    
    # 3. Tính toán mẫu số (div_term) cho các chỉ số 2i
    # Công thức: 10000^(2i / d_model)
    # Sử dụng log-space để tính toán ổn định hơn: exp(2i * -log(10000) / d_model)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # 4. Áp dụng sin cho các cột chẵn (0, 2, 4, ...)
    pe[:, 0::2] = np.sin(position * div_term)
    
    # 5. Áp dụng cos cho các cột lẻ (1, 3, 5, ...)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe