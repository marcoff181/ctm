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


# TODO: change signature to be (original_image_name?, watermark_name) -> watermarked image
def embedding(original_image_path, watermark_path, alpha, beta):
    """
    Embed watermark by modifying singular values of LL, LH and HL subbands.

    Args:
        original_image_path: Path to original image
        watermark_path: Path to watermark file
        alpha: Embedding strength for LH and HL subbands
        beta: Embedding strength for LL subband
    """
    # Load image and watermark
    image = cv2.imread(original_image_path, 0)
    watermark, _, Sw, _ = get_watermark_svd(watermark_path)

    # Multi-level DWT decomposition
    coeffs = pywt.dwt2(image, wavelet="haar")

    LL, (LH, HL, HH) = coeffs

    # SVD on LL, LH and HL subbands
    Ui_LL, Si_LL, Vi_LL = np.linalg.svd(LL)
    Ui_LH, Si_LH, Vi_LH = np.linalg.svd(LH)

    # Embed watermark in singular values
    S_new_LL = Si_LL.copy()
    S_new_LH = Si_LH.copy()
    num_watermark_values = len(Sw)

    # LL with beta
    S_new_LL[:num_watermark_values] = Si_LL[:num_watermark_values] + beta * Sw
    # LH and HL with alpha
    S_new_LH[:num_watermark_values] = Si_LH[:num_watermark_values] + alpha * Sw

    # Reconstruct subbands with SVD
    LL = Ui_LL.dot(np.diag(S_new_LL)).dot(Vi_LL)
    LH = Ui_LH.dot(np.diag(S_new_LH)).dot(Vi_LH)

    # Replace all three subbands in the coefficient list
    coeffs = LL, (LH, HL, HH)

    # Multi-level inverse DWT reconstruction
    watermarked_image = pywt.idwt2(coeffs, wavelet="haar")

    # Ensure same size as original
    watermarked_image = watermarked_image[: image.shape[0], : image.shape[1]]

    # Clip values to [0, 255] and convert to uint8
    np.clip(watermarked_image, 0, 255, out=watermarked_image)
    watermarked_image = watermarked_image.astype(np.uint8)

    return watermarked_image, watermark
