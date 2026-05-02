import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Tính toán Scaled Dot-Product Attention.
    
    Tham số:
    - Q: Queries, shape (..., seq_q, d_k)
    - K: Keys, shape (..., seq_k, d_k)
    - V: Values, shape (..., seq_k, d_v)
    
    Trả về:
    - Output: Kết quả attention, shape (..., seq_q, d_v)
    """
    
    # 1. Lấy kích thước d_k (chiều của vector key)
    d_k = K.size(-1)
    
    # 2. Tính điểm số (Scores) bằng tích vô hướng giữa Q và K chuyển vị
    # K.transpose(-2, -1) đảo ngược hai chiều cuối cùng của K
    # scores shape: (..., seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # 3. Chia tỉ lệ (Scaling) bằng căn bậc hai của d_k
    scaled_scores = scores / math.sqrt(d_k)
    
    # 4. Áp dụng hàm Softmax để tính trọng số Attention (Attention Weights)
    # Dim=-1 để tính softmax trên chiều của các Keys
    weights = F.softmax(scaled_scores, dim=-1)
    
    # 5. Nhân trọng số với các giá trị Values
    # output shape: (..., seq_q, d_v)
    output = torch.matmul(weights, V)
    
    return output