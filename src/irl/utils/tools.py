import numpy as np

class AdamOptimizer:
    def __init__(self, params_shape, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m = np.zeros(params_shape)
        self.v = np.zeros(params_shape)
        self.t = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def step(self, params, grads):
        """
        Updates params using Adam Gradient Ascent (maximizing reward).
        If minimizing loss, flip the sign of the update.
        """
        self.t += 1
        
        # 1. Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        
        # 2. Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        # 3. Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        
        # 4. Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # 5. Update parameters
        # Note: We use += because we are doing Gradient Ascent (Maximizing Likelihood)
        params += self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params
    
def compute_dtw(series_a, series_b):
    """
    Computes the Dynamic Time Warping distance between two 1D arrays.
    """
    n, m = len(series_a), len(series_b)
    
    # Initialize the cost matrix with infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    
    # The starting point (0,0) has 0 accumulated cost
    dtw_matrix[0, 0] = 0
    
    # Fill the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Calculate squared Euclidean distance between points
            cost = (series_a[i - 1] - series_b[j - 1]) ** 2
            
            # Take the cost + the minimum of the three neighbors
            last_min = np.min([
                dtw_matrix[i - 1, j],     # Insertion
                dtw_matrix[i, j - 1],     # Deletion
                dtw_matrix[i - 1, j - 1]  # Match
            ])
            
            dtw_matrix[i, j] = cost + last_min

    return np.sqrt(dtw_matrix[n, m])

def compute_mae(soc_a, soc_b):
    """MAE between two SoC trajectories, resampling to the longer length."""
    n = max(len(soc_a), len(soc_b))
    if len(soc_a) != n:
        soc_a = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(soc_a)), soc_a)
    if len(soc_b) != n:
        soc_b = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(soc_b)), soc_b)
    return float(np.mean(np.abs(soc_a - soc_b)))

def resample_trajectory(values, target_len):
    """Linearly resample a 1-D array to target_len points."""
    values = np.asarray(values, dtype=np.float32)
    if len(values) == target_len:
        return values
    if len(values) <= 1:
        return np.full(target_len, values[0] if len(values) == 1 else 0.0, dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(dst_x, src_x, values).astype(np.float32)