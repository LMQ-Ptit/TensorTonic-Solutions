import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Chuẩn hóa lớp (Layer Normalization)."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """Cơ chế Multi-head attention bám sát quy ước np.dot."""
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    
    # 1. Projections
    q_proj = np.dot(Q, W_q)
    k_proj = np.dot(K, W_k)
    v_proj = np.dot(V, W_v)
    
    # 2. Split heads
    def split(x):
        return x.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    q, k, v = split(q_proj), split(k_proj), split(v_proj)
    
    # 3. Scaled Dot-Product Attention
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    attn_output = np.matmul(weights, v)
    
    # 4. Concatenate and Project
    concat = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    return np.dot(concat, W_o)

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Mạng Feed-forward theo từng vị trí."""
    z1 = np.dot(x, W1) + b1
    a1 = np.maximum(0, z1) # ReLU
    return np.dot(a1, W2) + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Hợp nhất toàn bộ Encoder Block:
    - Sub-layer 1: Multi-Head Attention + Residual + LayerNorm
    - Sub-layer 2: Feed-Forward + Residual + LayerNorm
    """
    
    # --- Sub-layer 1: Attention ---
    # Tính Attention (Self-attention nên Q=K=V=x)
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    
    # Kết nối tắt (Residual) và Layer Normalization 1
    # x = LayerNorm(x + Sublayer(x))
    out1 = layer_norm(x + attn_out, gamma1, beta1)
    
    # --- Sub-layer 2: Feed Forward ---
    # Tính Feed-Forward dựa trên đầu ra của tầng trước
    ffn_out = feed_forward(out1, W1, b1, W2, b2)
    
    # Kết nối tắt (Residual) và Layer Normalization 2
    # x = LayerNorm(x + Sublayer(x))
    out2 = layer_norm(out1 + ffn_out, gamma2, beta2)
    
    return out2