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


def embedding(original_image_path, watermark_path, alpha, beta, type="additive"):
    """
    Embed watermark by modifying singular values of LL, LH and HL subbands.
    
    Args:
        original_image_path: Path to original image
        watermark_path: Path to watermark file
        alpha: Embedding strength for LH and HL subbands
        beta: Embedding strength for LL subband
        type: Embedding type ("multiplicative" or additive)
    """
    # Load image and watermark
    image = cv2.imread(original_image_path, 0)
    watermark, Uwm, Swm, Vwm = get_watermark_svd(watermark_path)

    # Start timing
    start = time.time()

    # Multi-level DWT decomposition
    coeffs = pywt.dwt2(image, wavelet="haar")
    
    LL, (LH, HL, HH) = coeffs  

    # SVD on LL, LH and HL subbands
    Ui_LL, Si_LL, Vi_LL = np.linalg.svd(LL)
    Ui_LH, Si_LH, Vi_LH = np.linalg.svd(LH)
    Ui_HL, Si_HL, Vi_HL = np.linalg.svd(HL)

    # Embed watermark in singular values
    S_new_LL = Si_LL.copy()
    S_new_LH = Si_LH.copy()
    S_new_HL = Si_HL.copy()
    num_watermark_values = len(Swm)

    if type == "multiplicative":
        # LL with beta
        S_new_LL[:num_watermark_values] = Si_LL[:num_watermark_values] * (1 + beta * Swm)
        # LH and HL with alpha
        S_new_LH[:num_watermark_values] = Si_LH[:num_watermark_values] * (1 + alpha * Swm)
        S_new_HL[:num_watermark_values] = Si_HL[:num_watermark_values] * (1 + alpha * Swm)
    else:
        # LL with beta
        S_new_LL[:num_watermark_values] = Si_LL[:num_watermark_values] + beta * Swm
        # LH and HL with alpha
        S_new_LH[:num_watermark_values] = Si_LH[:num_watermark_values] + alpha * Swm
        S_new_HL[:num_watermark_values] = Si_HL[:num_watermark_values] + alpha * Swm

    # Reconstruct subbands with SVD
    LL_new = Ui_LL.dot(np.diag(S_new_LL)).dot(Vi_LL)
    LH_new = Ui_LH.dot(np.diag(S_new_LH)).dot(Vi_LH)
    HL_new = Ui_HL.dot(np.diag(S_new_HL)).dot(Vi_HL)

    # Replace all three subbands in the coefficient list
    coeffs = LL_new, (LH_new, HL_new, HH)
    
    # Multi-level inverse DWT reconstruction
    watermarked_image = pywt.idwt2(coeffs, wavelet="haar")

    # Ensure same size as original
    watermarked_image = watermarked_image[:image.shape[0], :image.shape[1]]

    # Clip values to [0, 255] and convert to uint8
    np.clip(watermarked_image, 0, 255, out=watermarked_image)
    watermarked_image = watermarked_image.astype(np.uint8)

    end = time.time()

    return watermarked_image, watermark, Uwm, Vwm