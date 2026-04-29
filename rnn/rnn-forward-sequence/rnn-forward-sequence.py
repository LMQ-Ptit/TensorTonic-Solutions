import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass cho RNN hỗ trợ Batch Processing.
    """
    # Lấy thông tin kích thước
    # X shape: (batch_size, seq_len, input_dim)
    batch_size, seq_len, input_dim = X.shape
    hidden_dim = h_0.shape[1]
    
    # Khởi tạo H để lưu tất cả hidden states
    # Shape mong muốn: (batch_size, seq_len, hidden_dim)
    H = np.zeros((batch_size, seq_len, hidden_dim))
    
    h_prev = h_0
    
    for t in range(seq_len):
        # Lấy input tại bước t cho toàn bộ batch: (batch_size, input_dim)
        x_t = X[:, t, :]
        
        # Tính toán với transpose của W để khớp với Batch Dimension
        # h_t = tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b_h)
        # Kết quả z sẽ có shape: (batch_size, hidden_dim)
        z = np.dot(x_t, W_xh.T) + np.dot(h_prev, W_hh.T) + b_h
        h_t = np.tanh(z)
        
        # Lưu vào H tại bước thời gian t
        H[:, t, :] = h_t
        
        # Cập nhật h_prev cho bước kế tiếp
        h_prev = h_t
        
    # h_n là trạng thái cuối cùng: (batch_size, hidden_dim)
    h_n = h_prev
    
    return H, h_n
