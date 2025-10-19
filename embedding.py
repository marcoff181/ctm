import numpy as np
import cv2
import pywt
import time

from scipy.fft import dct, idct


def get_watermark_svd(watermark_path):
    """Load watermark and compute its SVD decomposition."""
    watermark = np.load(watermark_path)
    watermark_matrix = watermark.reshape(32, 32)
    U, S, V = np.linalg.svd(watermark_matrix)
    return watermark, U, S, V


def embedding(original_image_path, watermark_path, alpha, type="multiplicative", dwt_level=2):
    """Embed watermark by modifying singular values of LL subband at specified DWT level."""
    # Load image and watermark
    image = cv2.imread(original_image_path, 0)
    watermark, Uwm, Swm, Vwm = get_watermark_svd(watermark_path)

    # Start timing
    start = time.time()

    # Multi-level DWT decomposition
    coeffs = pywt.wavedec2(image, wavelet="haar", level=dwt_level)
    
    # coeffs structure for level 3:
    # [LL3, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1)]
    # We want to modify LL3 (the deepest approximation)
    
    LL = coeffs[0]  # Deepest LL subband
    
    print(f"DWT Level {dwt_level}: LL subband shape = {LL.shape}")

    # -------- Apply DCT to the LL subband ---------
    LL_dct = dct(dct(LL, axis=0, norm='ortho'), axis=1, norm='ortho')
    # ----------------------------------------------

    # SVD on DCT-transformed LL subband
    Ui, Si, Vi = np.linalg.svd(LL_dct)

    # Embed watermark in singular values
    S_new = Si.copy()
    num_watermark_values = len(Swm)

    if type == "multiplicative":
        S_new[:num_watermark_values] = (
            Si[:num_watermark_values] + alpha * Si[:num_watermark_values] * Swm
        )
    else:
        S_new[:num_watermark_values] = Si[:num_watermark_values] + alpha * Swm

    # Reconstruct LL subband: SVD -> inverse DCT
    LL_dct_new = Ui.dot(np.diag(S_new)).dot(Vi)
    LL_new = idct(idct(LL_dct_new, axis=1, norm='ortho'), axis=0, norm='ortho')
    
    # Replace the LL subband in the coefficient list
    coeffs[0] = LL_new
    
    # Multi-level inverse DWT reconstruction
    watermarked_image = pywt.waverec2(coeffs, wavelet="haar")

    # Ensure same size as original (waverec2 might add padding)
    watermarked_image = watermarked_image[:image.shape[0], :image.shape[1]]

    # Clip values to [0, 255] and convert to uint8
    np.clip(watermarked_image, 0, 255, out=watermarked_image)
    watermarked_image = watermarked_image.astype(np.uint8)

    end = time.time()
    print(f"Embedding time (DWT level {dwt_level}): {end - start:.4f}s")

    return watermarked_image, watermark, Uwm, Vwm


def detection(original, watermarked, attacked, Uw, Vw, alpha=0.05, dwt_level=2):
    """Detect watermark in attacked image."""
    extracted_wm = extraction(original, attacked, Uw, Vw, alpha=alpha, dwt_level=dwt_level)
    original_wm = extraction(original, watermarked, Uw, Vw, alpha=alpha, dwt_level=dwt_level)

    # Measure quality
    wpsnr_attack = wpsnr(watermarked, attacked)

    # Detect watermark
    sim = similarity(original_wm, extracted_wm)

    # TODO: use ROC to compute final threshold
    detected = 1 if sim > 0.7 else 0

    return detected, wpsnr_attack


def extraction(original, watermarked, Uw, Vw, alpha=0.05, type="multiplicative", dwt_level=2):
    """Extract watermark from watermarked image."""

    # Multi-level DWT decomposition
    coeffs_original = pywt.wavedec2(original, wavelet="haar", level=dwt_level)
    coeffs_watermarked = pywt.wavedec2(watermarked, wavelet="haar", level=dwt_level)

    LLi = coeffs_original[0]  # Deepest LL from original
    LLa = coeffs_watermarked[0]  # Deepest LL from watermarked/attacked

    # Apply DCT to LL subbands (CRITICAL: must match embedding!)
    LLi_dct = dct(dct(LLi, axis=0, norm='ortho'), axis=1, norm='ortho')
    LLa_dct = dct(dct(LLa, axis=0, norm='ortho'), axis=1, norm='ortho')

    # SVD on DCT-transformed LL subbands
    Ui, Si, Vi = np.linalg.svd(LLi_dct)
    Ua, Sa, Va = np.linalg.svd(LLa_dct)

    # Extract watermark singular values
    # Embedding formula: S_new = Si + alpha * Si * Swm (multiplicative)
    # Therefore: Swm = (S_new - Si) / (alpha * Si)

    # Use the size of Uw/Vw to determine watermark dimensions
    watermark_sv_count = min(Uw.shape[1], Vw.shape[0])

    # Extract using the inverse of embedding formula
    if type == "multiplicative":
        epsilon = 1e-10  # small constant to avoid division by zero
        Sw = (Sa[:watermark_sv_count] - Si[:watermark_sv_count]) / (
            alpha * Si[:watermark_sv_count] + epsilon
        )
    else:
        # Additive formula: S_new = Si + alpha * Swm
        # Therefore: Swm = (S_new - Si) / alpha
        Sw = (Sa[:watermark_sv_count] - Si[:watermark_sv_count]) / alpha

    # Recompose watermark from singular values
    extracted_watermark = Uw.dot(np.diag(Sw)).dot(Vw)

    # Flatten and binarize for similarity/BER
    extracted_watermark = extracted_watermark.flatten()
    extracted_watermark = (extracted_watermark > 0.5).astype(np.uint8)

    return extracted_watermark


def similarity(X, X_star):
    """Compute bit error rate (BER) based similarity for binary watermarks"""
    X = X.astype(np.uint8)
    X_star = X_star.astype(np.uint8)

    # Calculate number of matching bits
    matches = np.sum(X == X_star)
    total = len(X)

    # Similarity: 1.0 = perfect match, 0.0 = all bits different
    similarity_score = matches / total

    return similarity_score


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
