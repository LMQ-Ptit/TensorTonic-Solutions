import numpy as np

def init_hidden(batch_size: int, hidden_dim: int) -> np.ndarray:
    """
    Khởi tạo trạng thái ẩn (hidden state) cho RNN với giá trị 0.
    
    Tham số:
    - batch_size: Số lượng mẫu trong một batch.
    - hidden_dim: Kích thước của vector trạng thái ẩn.
    
    Trả về:
    - h_0: Mảng NumPy kích thước (batch_size, hidden_dim) chứa toàn số 0.
    """
    
    # Sử dụng np.zeros để tạo ma trận 0 với kích thước tương ứng
    h_0 = np.zeros((batch_size, hidden_dim))
    
    return h_0