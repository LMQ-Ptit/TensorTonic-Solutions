import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Thực hiện forward pass qua toàn bộ chuỗi.
        X shape: (batch, T, input_dim)
        Trả về: (y_seq, h_final)
        """
        batch_size, T, input_dim = X.shape
        
        # Hint 1: Khởi tạo h_0 là ma trận không nếu không được cung cấp
        if h_0 is None:
            h_0 = np.zeros((batch_size, self.hidden_dim))
        
        h_t = h_0
        y_list = []
        
        # Hint 2: Lặp qua từng bước thời gian
        for t in range(T):
            x_t = X[:, t, :] # Lấy input tại bước t cho toàn bộ batch
            
            # Tính trạng thái ẩn: h_t = tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b_h)
            h_t = np.tanh(x_t @ self.W_xh.T + h_t @ self.W_hh.T + self.b_h)
            
            # Tính đầu ra (Output Projection): y_t = h_t @ W_hy.T + b_y
            # Lưu ý: Không dùng hàm kích hoạt (activation) ở đây theo yêu cầu
            y_t = h_t @ self.W_hy.T + self.b_y
            
            y_list.append(y_t)
            
        # Hint 3: Gộp danh sách các y_t thành một mảng 3D
        # y_seq shape: (batch, T, output_dim)
        y_seq = np.stack(y_list, axis=1)
        
        # h_final là trạng thái ẩn tại bước cuối cùng
        h_final = h_t
        
        return y_seq, h_final