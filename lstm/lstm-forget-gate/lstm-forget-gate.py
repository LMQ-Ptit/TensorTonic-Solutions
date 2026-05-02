import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """
    Tính toán cổng quên: f_t = sigmoid(W_f @ [h_prev, x_t] + b_f)
    """
    
    # 1. Ghép nối (Concatenate) h_prev và x_t dọc theo trục đặc trưng (feature dimension)
    # Giả sử shape là (batch_size, dim), ta ghép theo axis=1 (hoặc -1)
    # Kết quả concat sẽ có kích thước (batch_size, hidden_dim + input_dim)
    concat = np.concatenate((h_prev, x_t), axis=-1)
    
    # 2. Thực hiện phép tính tuyến tính
    # Vì chúng ta đang xử lý batch, công thức W_f @ v sẽ trở thành concat @ W_f.T
    z = np.dot(concat, W_f.T) + b_f
    
    # 3. Đi qua hàm kích hoạt sigmoid để đưa giá trị về khoảng (0, 1)
    f_t = sigmoid(z)
    
    return f_t