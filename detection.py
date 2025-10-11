import numpy as np


def detection(image, watermarked, alpha=None, mark_size=1024, v=None, **kwargs):
    """LSB watermark extraction - extracts watermark from least significant bits"""
    # Convert to uint8
    image = image.astype(np.uint8)
    watermarked = watermarked.astype(np.uint8)
    
    # Flatten watermarked image
    flat_watermarked = watermarked.flatten()
    
    # Extract LSB from first mark_size pixels
    w_ex = np.zeros(mark_size, dtype=np.uint8)
    
    for i in range(mark_size):
        # Extract LSB
        w_ex[i] = flat_watermarked[i] & 0x01
    
    return w_ex


def similarity(X, X_star):
    """Compute bit error rate (BER) based similarity for binary watermarks"""
    # Convert to binary if needed
    X = (X > 0.5).astype(np.uint8)
    X_star = (X_star > 0.5).astype(np.uint8)
    
    # Calculate number of matching bits
    matches = np.sum(X == X_star)
    total = len(X)
    
    # Similarity: 1.0 = perfect match, 0.0 = all bits different
    similarity = matches / total
    
    return similarity


def compute_threshold(mark_size, w, N=1000):
    """Compute detection threshold using Monte Carlo simulation for binary watermarks"""
    np.random.seed(42)
    SIM = np.zeros(N)
    
    # Convert watermark to binary
    w_binary = (w > 0.5).astype(np.uint8)
    
    for i in range(N):
        # Generate random binary sequence
        r = np.random.randint(0, 2, mark_size, dtype=np.uint8)
        SIM[i] = similarity(w_binary, r)
    
    SIMs = SIM.copy()
    SIM.sort()
    
    # Threshold: mean + 3*std (for binary, expected ~0.5 for random)
    mean_sim = np.mean(SIM)
    std_sim = np.std(SIM)
    T = mean_sim + 3 * std_sim
    
    
    return T, SIMs