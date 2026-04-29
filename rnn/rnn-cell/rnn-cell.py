import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass (Column Vector Convention).
    """
    # Thực hiện nhân ma trận Trọng số trước, Vector sau
    # h_t = tanh(W_xh * x_t + W_hh * h_prev + b_h)
    
    z = np.dot(W_xh, x_t) + np.dot(W_hh, h_prev) + b_h
    h_next = np.tanh(z)
    
    return h_next