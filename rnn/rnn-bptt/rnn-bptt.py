import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Thực hiện lan truyền ngược cho một bước thời gian RNN duy nhất.
    
    Tham số:
    - dh_next: Gradient của hàm mất mát đối với h_t, hình dạng (batch_size, hidden_dim)
    - h_t: Trạng thái ẩn tại bước t, hình dạng (batch_size, hidden_dim)
    - h_prev: Trạng thái ẩn tại bước t-1, hình dạng (batch_size, hidden_dim)
    - x_t: Đầu vào tại bước t, hình dạng (batch_size, input_dim)
    - W_hh: Trọng số hidden-to-hidden, hình dạng (hidden_dim, hidden_dim)
    
    Trả về:
    - (dh_prev, dW_hh): Gradient đối với h_prev và W_hh
    """
    
    # 1. Gradient qua hàm kích hoạt tanh
    # Đạo hàm của tanh(z) là (1 - tanh^2(z))
    # Vì h_t = tanh(z), đạo hàm là (1 - h_t^2)
    # Chúng ta dùng phép nhân element-wise (*) với dh_next
    dtanh = (1 - h_t**2) * dh_next # Shape: (batch_size, hidden_dim)
    
    # 2. Gradient đối với W_hh
    # Theo quy ước batch (h_t = tanh(h_prev @ W_hh.T + ...))
    # dW_hh = dtanh.T @ h_prev
    dW_hh = np.dot(dtanh.T, h_prev) # Shape: (hidden_dim, hidden_dim)
    
    # 3. Gradient đối với h_prev
    # dh_prev = dtanh @ W_hh
    dh_prev = np.dot(dtanh, W_hh) # Shape: (batch_size, hidden_dim)
    
    return dh_prev, dW_hh