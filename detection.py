import numpy as np
from scipy.fft import dct


# def detection(image, watermarked, alpha=None, mark_size=1024, v=None, **kwargs):
#     """6th bit watermark extraction - extracts watermark from 6th bit"""
#     # Convert to uint8
#     image = image.astype(np.uint8)
#     watermarked = watermarked.astype(np.uint8)
    
#     # Flatten watermarked image
#     flat_watermarked = watermarked.flatten()
    
#     # Calculate replication factor
#     total_pixels = len(flat_watermarked)
#     replication_factor = int(np.ceil(total_pixels / mark_size))
    
#     # Extract 6th bit from all pixels
#     extracted_bits = np.zeros(total_pixels, dtype=np.uint8)
#     for i in range(total_pixels):
#         # Extract 6th bit (bit position 5) and shift it to LSB position
#         extracted_bits[i] = (flat_watermarked[i] >> 6) & 0x01
    
#     # Average replicated watermark bits to get original watermark
#     w_ex = np.zeros(mark_size, dtype=np.uint8)
#     for i in range(mark_size):
#         # Collect all replicated instances of this watermark bit
#         replicated_values = []
#         for rep in range(replication_factor):
#             idx = i + rep * mark_size
#             if idx < total_pixels:
#                 replicated_values.append(extracted_bits[idx])
        
#         # Use majority voting
#         w_ex[i] = 1 if np.mean(replicated_values) >= 0.5 else 0
    
#     return w_ex


def detection(image, watermarked, alpha, mark_size, v='multiplicative'):
    ori_dct = dct(dct(image,axis=0, norm='ortho'),axis=1, norm='ortho')
    wat_dct = dct(dct(watermarked,axis=0, norm='ortho'),axis=1, norm='ortho')

    # Get the locations of the most perceptually significant components
    ori_dct = abs(ori_dct)
    wat_dct = abs(wat_dct)
    locations = np.argsort(-ori_dct,axis=None) # - sign is used to get descending order
    rows = image.shape[0]
    locations = [(val//rows, val%rows) for val in locations] # locations as (x,y) coordinates

    # Generate a watermark
    w_ex = np.zeros(mark_size, dtype=np.float64)

    # Embed the watermark
    for idx, loc in enumerate(locations[1:mark_size+1]):
        if v=='additive':
            w_ex[idx] =  (wat_dct[loc] - ori_dct[loc]) /alpha
        elif v=='multiplicative':
            w_ex[idx] =  (wat_dct[loc] - ori_dct[loc]) / (alpha*ori_dct[loc])
            
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