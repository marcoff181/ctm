import numpy as np
import pywt
from scipy.fft import dct
from wpsnr import wpsnr

def detection(original, watermarked, attacked,Uw,Vw):
    extracted_wm = extraction(original,attacked,Uw ,Vw )
    original_wm = extraction(original,watermarked,Uw ,Vw )

    # Measure quality
    wpsnr_attack = wpsnr(watermarked, attacked)
    
    # Detect watermark
    sim = similarity(original_wm, extracted_wm)


    #  TODO: use ROC to compute final threshold
    detected =  1 if sim > 0.7  else 0

    return detected , wpsnr_attack

def extraction(original,watermarked, Uw, Vw, alpha=0.2):
    # DWT decomposition
    image_coeffs = pywt.dwt2(original, wavelet='haar')
    attacked_coeffs = pywt.dwt2(watermarked, wavelet='haar')

    LLi, (LHi, HLi, HHi) = image_coeffs
    LLa, (LHa, HLa, HHa) = attacked_coeffs

    # SVD on HH subbands
    Ui, Si, Vi = np.linalg.svd(HHi)
    Ua, Sa, Va = np.linalg.svd(HHa)

    # Extract watermark singular values
    # According to embedding: S_new[i] = Si[i] + alpha * Swm[i]
    # Therefore: Swm[i] = (S_new[i] - Si[i]) / alpha
    Sw = (Sa - Si) / alpha
    
    # Recompose watermark from singular values
    # Use the size of Uw/Vw to determine watermark dimensions
    watermark_sv_count = min(Uw.shape[1], Vw.shape[0])
    extracted_watermark = Uw.dot(np.diag(Sw[:watermark_sv_count])).dot(Vw)

    # Flatten and binarize for similarity/BER
    extracted_watermark = extracted_watermark.flatten()
    extracted_watermark = (extracted_watermark > 0.5).astype(np.uint8)

    return extracted_watermark


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
