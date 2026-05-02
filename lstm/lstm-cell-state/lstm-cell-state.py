import numpy as np

def update_cell_state(C_prev: np.ndarray, f_t: np.ndarray,
                      i_t: np.ndarray, c_tilde: np.ndarray) -> np.ndarray:
    """
    Cập nhật trạng thái tế bào (cell state): 
    C_t = f_t * C_prev + i_t * c_tilde
    """
    
    # Sử dụng toán tử * để thực hiện phép nhân từng phần tử (Hadamard product)
    # 1. f_t * C_prev: Quyết định giữ lại bao nhiêu phần trăm bộ nhớ cũ
    # 2. i_t * c_tilde: Quyết định nạp bao nhiêu phần trăm thông tin mới vào
    
    C_t = (f_t * C_prev) + (i_t * c_tilde)
    
    return C_t