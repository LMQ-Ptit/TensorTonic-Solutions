import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def output_gate(h_prev: np.ndarray, x_t: np.ndarray, C_t: np.ndarray,
                W_o: np.ndarray, b_o: np.ndarray) -> tuple:
    """
    Tính toán cổng ra (output gate) và trạng thái ẩn hiện tại (hidden state).
    
    Trả về:
    - o_t: Giá trị cổng ra (0 đến 1)
    - h_t: Trạng thái ẩn mới (-1 đến 1)
    """
    # 1. Ghép nối h_prev và x_t dọc theo trục đặc trưng
    concat = np.concatenate((h_prev, x_t), axis=-1)
    
    # 2. Tính Output Gate (o_t)
    # Công thức: o_t = sigmoid(W_o @ [h_prev, x_t] + b_o)
    # Sử dụng W_o.T để xử lý dữ liệu dạng Batch
    o_t = sigmoid(np.dot(concat, W_o.T) + b_o)
    
    # 3. Tính Hidden State (h_t)
    # Kết hợp cổng ra với Cell State đã được nén qua hàm tanh
    # Công thức: h_t = o_t * tanh(C_t)
    h_t = o_t * np.tanh(C_t)
    
    return o_t, h_t