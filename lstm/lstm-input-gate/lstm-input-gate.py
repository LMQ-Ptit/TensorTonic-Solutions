import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> tuple:
    """
    Tính toán cổng vào (input gate) và bộ nhớ ứng viên (candidate memory).
    
    Trả về:
    - i_t: Giá trị cổng vào (0 đến 1)
    - g_t: Giá trị bộ nhớ ứng viên (-1 đến 1)
    """
    # 1. Ghép nối h_prev và x_t (giống bài trước)
    concat = np.concatenate((h_prev, x_t), axis=-1)
    
    # 2. Tính Input Gate (i_t) - Sử dụng hàm sigmoid
    # Công thức: i_t = sigmoid(W_i @ [h, x] + b_i)
    i_t = sigmoid(np.dot(concat, W_i.T) + b_i)
    
    # 3. Tính Candidate Memory (g_t hoặc č_t) - Sử dụng hàm tanh
    # Công thức: g_t = tanh(W_c @ [h, x] + b_c)
    g_t = np.tanh(np.dot(concat, W_c.T) + b_c)
    
    return i_t, g_t