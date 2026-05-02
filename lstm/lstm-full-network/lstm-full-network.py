import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """
        Forward pass qua toàn bộ chuỗi.
        X shape: (batch, T, input_dim)
        Trả về: (y_seq, h_last, C_last)
        """
        batch_size, T, input_dim = X.shape
        
        # 1. Khởi tạo h_t và C_t ban đầu là các ma trận 0
        h_t = np.zeros((batch_size, self.hidden_dim))
        C_t = np.zeros((batch_size, self.hidden_dim))
        
        y_list = []
        
        # 2. Lặp qua từng bước thời gian từ 1 đến T
        for t in range(T):
            x_t = X[:, t, :]
            
            # Ghép nối (concatenate) h_prev và x_t: shape (batch, hidden_dim + input_dim)
            concat = np.concatenate((h_t, x_t), axis=1)
            
            # 3. Tính toán các cổng (Gates)
            # f: forget, i: input, o: output, c_tilde: candidate
            f = sigmoid(concat @ self.W_f.T + self.b_f)
            i = sigmoid(concat @ self.W_i.T + self.b_i)
            c_tilde = np.tanh(concat @ self.W_c.T + self.b_c)
            o = sigmoid(concat @ self.W_o.T + self.b_o)
            
            # 4. Cập nhật Cell State và Hidden State
            C_t = f * C_t + i * c_tilde
            h_t = o * np.tanh(C_t)
            
            # 5. Tính toán đầu ra (output projection) cho bước thời gian này
            y_t = h_t @ self.W_y.T + self.b_y
            y_list.append(y_t)
            
        # 6. Gộp danh sách các y_t thành mảng 3D: (batch, T, output_dim)
        y_seq = np.stack(y_list, axis=1)
        
        return y_seq, h_t, C_t