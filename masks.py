import heapq
import warnings
import cv2
import numpy as np


def edges_mask(
    img: np.ndarray,
    low_threshold: int = 100,
    high_threshold: int = 200,
    dilate_iter: int = 1,
) -> np.ndarray:
    """
    Generates a boolean mask of edges in an image using the Canny algorithm.
    """
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
        
    edges = cv2.Canny(img_gray.astype(np.uint8), low_threshold, high_threshold)
    
    if dilate_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, k, iterations=dilate_iter)
        
    return edges.astype(bool)


def noisy_mask(
    img: np.ndarray, 
    window: int = 7, 
    percentile: int = 90, 
    dilate_iter: int = 0
) -> np.ndarray:
    """
    Generates a boolean mask of high-variance ("noisy") regions.
    """
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
        
    img_float = img_gray.astype(np.float32)
    sq = img_float * img_float
    k = (window, window)
    
    mean = cv2.boxFilter(img_float, ddepth=-1, ksize=k, normalize=True)
    mean_sq = cv2.boxFilter(sq, ddepth=-1, ksize=k, normalize=True)
    var = mean_sq - mean * mean
    
    thr = np.percentile(var, percentile)
    mask = var > thr
    
    if dilate_iter > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask.astype(np.uint8), se, iterations=dilate_iter).astype(bool)
        
    return mask


# --- Frequency Domain Mask for DWT/DCT attacks ---
def frequency_mask(
    img: np.ndarray,
    method: str = "dct",
    band: str = "mid",
    block_size: int = 8,
    keep_ratio: float = 0.1,
) -> np.ndarray:
    """
    Masks regions based on their frequency content in DCT or DWT domain.
    """
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
        
    img_float = img_gray.astype(np.float32)
    h, w = img_float.shape
    mask = np.zeros_like(img_float, dtype=bool)
    scores = []
    
    pywt = None
    if method == "dwt":
        try:
            import pywt
        except ImportError:
            raise ImportError(
                "PyWavelets is required for DWT frequency mask. "
                "Please install it with 'pip install PyWavelets'"
            )

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img_float[i : i + block_size, j : j + block_size]
            if block.shape[0] < block_size or block.shape[1] < block_size:
                continue
                
            score = 0.0
            if method == "dct":
                dct_block = cv2.dct(block)
                # Mid-band mask: select coefficients away from DC and corners
                mid_mask = np.zeros_like(dct_block, dtype=bool)
                N = block_size
                # A simple definition for mid-band
                u_start, u_end = N // 6, N * 5 // 6
                v_start, v_end = N // 6, N * 5 // 6
                
                mid_mask[u_start:u_end, v_start:v_end] = True
                
                if np.any(mid_mask):
                    score = np.mean(np.abs(dct_block[mid_mask]))
                else:
                    score = 0.0 # Handle case where block_size is too small

            elif method == "dwt" and pywt:
                coeffs2 = pywt.dwt2(block, 'haar')
                LL, (LH, HL, HH) = coeffs2
                # Mid-band: use LH and HL
                score = (np.mean(np.abs(LH)) + np.mean(np.abs(HL))) / 2
            else:
                continue
                
            scores.append((score, i, j)) # Store score first for easier heap processing

    # Sort blocks by score (mid-frequency energy) and select top N
    n_keep = int(len(scores) * keep_ratio)
    if n_keep > 0:
        top_scores = heapq.nlargest(n_keep, scores) # More efficient than sorting all
        for score, i, j in top_scores:
            mask[i : i + block_size, j : j + block_size] = True

    # show_images(img,mask)

    return mask


# --- Saliency Mask ---
# this attack is like a hidden gem, if we change the threshold to check for the top 20% we
# can attack images that embed in intelligent area.
# if instead we search for the low 20 (change the '>' to '<' for the end threshold, and also the,
# percentile parameter) and we are checking for really simple watermarking techniques.
def saliency_mask(img: np.ndarray, percentile: int = 80) -> np.ndarray:
    """
    Masks regions with lowest visual saliency (likely watermark embedding zones).
    Uses OpenCV's saliency API. Requires 'opencv-contrib-python'.
    Falls back to a Laplacian if the saliency module fails.
    """
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    img_gray = img_gray.astype(np.uint8)
    
    saliencyMap = None
    try:
        # StaticSaliencyFineGrained_create is in opencv-contrib-python
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(img_gray)
        if not success:
            raise RuntimeError("cv2.saliency.computeSaliency() failed")
        saliencyMap = (saliencyMap * 255).astype(np.uint8)
        
    except (cv2.error, AttributeError, RuntimeError) as e:
        warnings.warn(
            f"Failed to use cv2.saliency ({e}). "
            "Falling back to simple Laplacian proxy. "
            "For better results, please install 'opencv-contrib-python'."
        )
        # Fallback: use Laplacian as a simple saliency proxy (high values = high saliency)
        saliencyMap_raw = np.abs(cv2.Laplacian(img_gray, cv2.CV_64F))
        max_val = saliencyMap_raw.max()
        if max_val > 0:
            saliencyMap = (saliencyMap_raw / max_val * 255).astype(np.uint8)
        else:
            saliencyMap = np.zeros_like(img_gray, dtype=np.uint8)

    # Mask least salient regions
    thr = np.percentile(saliencyMap, percentile)
    mask = saliencyMap > thr

    # show_images(img,mask)
    return mask

def entropy_mask(
    img: np.ndarray,
    block_size: int = 8,
    entropy_exp: float = 3.0,
    energy_thr: float = 50.0,
    keep_ratio: float = 0.007,
    n_candidates: int = 16,
) -> np.ndarray:
    """
    Masks blocks with highest SVD flatness score (similar to embedding block selection).
    """
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    img_float = img_gray.astype(np.float32)
    h, w = img_float.shape
    mask = np.zeros_like(img_float, dtype=bool)
    scores = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img_float[i : i + block_size, j : j + block_size]
            if block.shape[0] < block_size or block.shape[1] < block_size:
                continue

            U, S, V = np.linalg.svd(block, full_matrices=False)

            # Prevent division by zero if sum is 0
            S_sum = S.sum()
            if S_sum < 1e-8:
                S_norm = S # All are zero
            else:
                S_norm = S / S_sum

            # Prevent log(0)
            S_log = np.log2(S_norm + 1e-9) 
            entropy = -np.sum(S_norm * S_log) / np.log2(len(S))

            energy = np.var(block)
            score = (entropy**entropy_exp) * np.exp(-energy / energy_thr)
            scores.append((score, i, j)) # Score first for heapq

    # Select blocks with highest scores (most likely to be used for embedding)
    n_keep = int(len(scores) * keep_ratio)
    if n_keep == 0 and len(scores) > 0: # Ensure at least one block is selected
        n_keep = 1

    if n_keep > 0:
        top_scores = heapq.nlargest(n_keep, scores)
        for score, i, j in top_scores:
            mask[i : i + block_size, j : j + block_size] = True

    # TODO: uncomment, and comment before to => Always select at least n_blocks, optionally up to n_candidates for analysis
    # n_blocks = 16
    # top_scores = heapq.nlargest(max(n_blocks, n_candidates), scores)
    # for idx, (score, i, j) in enumerate(top_scores):
    #     if idx < n_blocks:
    #         mask[i : i + block_size, j : j + block_size] = True

    return mask


def first_blocks_mask(
    img: np.ndarray
) -> np.ndarray:
    m = np.zeros_like(img, dtype=bool)
    m[:8, :8] = True
    return m


# In utilities.py, aggiungi:
def border_mask(img: np.ndarray, border_percent: float = 0.1) -> np.ndarray:
    """Maschera che attacca solo i bordi (simula crop)"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    border_h = int(h * border_percent)
    border_w = int(w * border_percent)

    # Attacca solo bordi
    mask[:border_h, :] = True  # Top
    mask[-border_h:, :] = True  # Bottom
    mask[:, :border_w] = True  # Left
    mask[:, -border_w:] = True  # Right
    return mask