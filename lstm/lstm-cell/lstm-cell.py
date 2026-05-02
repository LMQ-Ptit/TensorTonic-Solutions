import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> tuple:
    """
    Thực hiện forward pass hoàn chỉnh cho một LSTM cell.
    """
    
    # 1. Kết hợp trạng thái ẩn trước đó và đầu vào hiện tại
    # Shape: (batch_size, hidden_dim + input_dim)
    concat = np.concatenate((h_prev, x_t), axis=-1)

    # 2. Tính toán các Cổng (Gates) và Bộ nhớ ứng viên
    # Sử dụng .T để phù hợp với phép nhân ma trận cho dữ liệu dạng Batch
    f_t = sigmoid(np.dot(concat, W_f.T) + b_f)      # Forget gate
    i_t = sigmoid(np.dot(concat, W_i.T) + b_i)      # Input gate
    c_tilde = np.tanh(np.dot(concat, W_c.T) + b_c)  # Candidate cell state
    o_t = sigmoid(np.dot(concat, W_o.T) + b_o)      # Output gate

    # 3. Cập nhật Cell State (Bộ nhớ dài hạn)
    # C_t = (quên đi phần cũ) + (nạp thêm phần mới)
    C_t = f_t * C_prev + i_t * c_tilde

    # 4. Tính toán Hidden State (Đầu ra/Bộ nhớ ngắn hạn)
    # h_t = (cổng ra) * tanh(bộ nhớ dài hạn đã cập nhật)
    h_t = o_t * np.tanh(C_t)

    return h_t, C_t