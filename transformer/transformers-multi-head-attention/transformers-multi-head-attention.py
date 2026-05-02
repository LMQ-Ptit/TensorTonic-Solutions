import numpy as np

def softmax(x, axis=-1):
    # Hàm softmax đã được cung cấp
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Tính toán Multi-Head Attention bám sát Hints của TensorTonic.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    
    # Hint 1: Project Q, K, V bằng np.dot (không dùng .T)
    # Shape sau project: (batch, seq, d_model)
    Q_proj = np.dot(Q, W_q)
    K_proj = np.dot(K, W_k)
    V_proj = np.dot(V, W_v)
    
    # Hint 2: Reshape sang (batch, seq, num_heads, d_k) 
    # rồi transpose sang (batch, num_heads, seq, d_k)
    def split_heads(x):
        # reshape: (batch, seq, num_heads, d_k)
        x = x.reshape(batch_size, seq_len, num_heads, d_k)
        # transpose: (batch, num_heads, seq, d_k)
        return x.transpose(0, 2, 1, 3)

    Q_heads = split_heads(Q_proj)
    K_heads = split_heads(K_proj)
    V_heads = split_heads(V_proj)
    
    # Hint 3: Tính scores bằng np.matmul(Q, K.transpose(0, 1, 3, 2)) / sqrt(d_k)
    # Q_heads: (batch, heads, seq, d_k)
    # K_heads_T: (batch, heads, d_k, seq)
    scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    
    # Áp dụng softmax để lấy trọng số attention
    weights = softmax(scores, axis=-1)
    
    # Tính output của từng head: (batch, num_heads, seq, d_k)
    attention_output = np.matmul(weights, V_heads)
    
    # Gộp các head lại (Concatenate)
    # (batch, num_heads, seq, d_k) -> (batch, seq, num_heads, d_k) -> (batch, seq, d_model)
    concat_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    # Cuối cùng: Project qua W_o bằng np.dot
    output = np.dot(concat_output, W_o)
    
    return output