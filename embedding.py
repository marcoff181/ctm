import numpy as np
import cv2
import pywt
import time


def get_watermark_svd(watermark_path):
    """Load watermark and compute its SVD decomposition."""
    watermark = np.load(watermark_path)
    watermark_matrix = watermark.reshape(32, 32)
    U, S, V = np.linalg.svd(watermark_matrix)
    return watermark, U, S, V


def embedding(original_image_path, watermark_path, alpha=0.05):
    """
    Embed watermark by modifying singular values of HH subband.
    
    Args:
        original_image_path: Path to the original image
        watermark_path: Path to the watermark .npy file
        alpha: Embedding strength (default: 0.05)
    
    Returns:
        watermarked_image: Image with embedded watermark
        watermark: Original watermark array
        U, V: SVD matrices needed for detection
    """
    # Load image and watermark
    image = cv2.imread(original_image_path, 0)
    watermark, Uwm, Swm, Vwm = get_watermark_svd(watermark_path)

    # Start timing
    start = time.time()

    # DWT decomposition
    coeffs2 = pywt.dwt2(image, wavelet='haar')
    LL, (LH, HL, HH) = coeffs2

    # SVD on HH subband
    Ui, Si, Vi = np.linalg.svd(HH)

    # Embed watermark in singular values
    S_new = Si.copy()
    num_watermark_values = len(Swm)
    for i in range(num_watermark_values):
        S_new[i] = Si[i] + alpha * Swm[i]

    # Reconstruct and inverse DWT
    HH_new = Ui.dot(np.diag(S_new)).dot(Vi)
    coeffs2_new = LL, (LH, HL, HH_new)
    watermarked_image = pywt.idwt2(coeffs2_new, wavelet='haar')

    end = time.time()
    print(f"Embedding time: {end - start:.4f}s")

    return watermarked_image, watermark, Uwm, Vwm