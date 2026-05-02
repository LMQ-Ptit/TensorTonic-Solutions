import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Mô phỏng sự suy giảm của gradient norm bằng cách sử dụng Spectral Norm.
    Bám sát các gợi ý từ TensorTonic.
    """
    # Hint 1: Tính Spectral Norm (chuẩn bậc 2 - giá trị suy biến lớn nhất)
    spectral_norm = np.linalg.norm(W_hh, ord=2)
    
    norms = []
    # Hint 2: Bắt đầu với gradient norm là 1.0
    g_norm = 1.0
    
    for t in range(T):
        # Hint 3: Thu thập giá trị trước khi nhân (để phần tử đầu luôn là 1.0)
        norms.append(float(g_norm))
        
        # Nhân với spectral norm cho bước tiếp theo
        g_norm *= spectral_norm
        
    return norms