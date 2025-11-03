from matplotlib import pyplot as plt
import numpy as np
import os
import cv2

# --- New imports for improved mask functions ---
import heapq
import warnings
from typing import Tuple, Optional

from wpsnr import *
from detection_crispymcmark import *

def show_images(img, watermarked):
    plt.subplot(121)
    plt.title("Original")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(122)
    plt.title("Watermarked")
    plt.imshow(watermarked, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def save_comparison(original, watermarked, attacked, attack_name, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    images = [
        (original, "Original"),
        (watermarked, "Watermarked"),
        (attacked, f"After {attack_name}"),
    ]

    for ax, (img, title) in zip(axes, images):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    filename = (
        attack_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    )
    plt.savefig(
        os.path.join(output_dir, f"comparison_{filename}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# --- IMPROVED MASK FUNCTIONS ---

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


# --- UNCHANGED FUNCTIONS ---

def verify_watermark_extraction(
    original, watermarked, attacked, mark_path, dwt_level=1, output_prefix=""
):
    """Verify that the embedded watermark can be correctly extracted."""
    print("\n" + "=" * 80)
    print("WATERMARK EXTRACTION VERIFICATION")
    print("=" * 80)

    # Load original watermark
    original_watermark = np.load(mark_path)

    # Extract watermark from watermarked image (no attack)
    extracted_watermark = extraction(original, watermarked, watermarked)

    # Compute similarity
    sim = similarity(original_watermark, extracted_watermark)

    # Bit statistics
    total_bits = len(original_watermark)
    matching_bits = np.sum(original_watermark == extracted_watermark)
    differing_bits = total_bits - matching_bits

    # Bit pattern analysis
    both_zero = np.sum((original_watermark == 0) & (extracted_watermark == 0))
    both_one = np.sum((original_watermark == 1) & (extracted_watermark == 1))
    orig_one_ext_zero = np.sum((original_watermark == 1) & (extracted_watermark == 0))
    orig_zero_ext_one = np.sum((original_watermark == 0) & (extracted_watermark == 1))

    print(f"\nExtraction Statistics:")
    print(f"  Total bits:        {total_bits}")
    print(f"  Matching bits:     {matching_bits} ({matching_bits/total_bits*100:.2f}%)")
    print(
        f"  Differing bits:    {differing_bits} ({differing_bits/total_bits*100:.2f}%)"
    )
    print(f"  Similarity:        {sim:.4f}")

    # Hamming distance
    hamming_dist = differing_bits
    normalized_hamming = hamming_dist / total_bits
    # print(f"\nHamming Distance:  {hamming_dist}")
    # print(f"Normalized:        {normalized_hamming:.4f}")

    # Status
    if sim >= 0.95:
        status = "EXCELLENT"
        status_symbol = "[OK]"
        bbox_color = "lightgreen"
    elif sim >= 0.7:
        status = "GOOD"
        status_symbol = "[OK]"
        bbox_color = "lightblue"
    elif sim >= 0.5:
        status = "WARNING"
        status_symbol = "[!]"
        bbox_color = "lightyellow"
    else:
        status = "ERROR"
        status_symbol = "[X]"
        bbox_color = "lightcoral"

    print(f"\nExtraction Status: {status_symbol} {status}")

    # Prepare watermark visualizations
    size = int(np.sqrt(total_bits))
    wm_orig = original_watermark.reshape(size, size)
    wm_extracted = extracted_watermark.reshape(size, size)
    error_2d = (
        (original_watermark != extracted_watermark).reshape(size, size).astype(float)
    )

    # ========== PLOT 1: Overview (Watermarks + Images) ==========
    fig1 = plt.figure(figsize=(18, 10))
    gs1 = fig1.add_gridspec(2, 4, hspace=0.25, wspace=0.25)

    # Top row: Watermarks
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.imshow(wm_orig, cmap="binary", vmin=0, vmax=1, interpolation="nearest")
    ax1.set_title(
        f"Original Watermark\n{np.sum(original_watermark)} ones",
        fontsize=12,
        fontweight="bold",
    )
    ax1.axis("off")

    ax2 = fig1.add_subplot(gs1[0, 1])
    ax2.imshow(wm_extracted, cmap="binary", vmin=0, vmax=1, interpolation="nearest")
    ax2.set_title(
        f"Extracted Watermark\n{np.sum(extracted_watermark)} ones",
        fontsize=12,
        fontweight="bold",
    )
    ax2.axis("off")

    ax3 = fig1.add_subplot(gs1[0, 2])
    im3 = ax3.imshow(error_2d, cmap="RdYlGn_r", interpolation="nearest", vmin=0, vmax=1)
    ax3.set_title(
        f"Error Map\n{differing_bits} errors ({normalized_hamming*100:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )
    ax3.axis("off")
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label("Error", fontsize=10)

    # Bottom row: Images
    ax5 = fig1.add_subplot(gs1[1, 0])
    ax5.imshow(original, cmap="gray")
    ax5.set_title("Original Image", fontsize=12, fontweight="bold")
    ax5.axis("off")

    ax6 = fig1.add_subplot(gs1[1, 1])
    ax6.imshow(watermarked, cmap="gray")
    wpsnr_val = wpsnr(original, watermarked)
    ax6.set_title(
        f"Watermarked Image\nWPSNR: {wpsnr_val:.2f} dB", fontsize=12, fontweight="bold"
    )
    ax6.axis("off")

    ax7 = fig1.add_subplot(gs1[1, 2])
    diff_img = np.abs(watermarked.astype(float) - original.astype(float))
    im7 = ax7.imshow(diff_img, cmap="hot")
    ax7.set_title(
        f"Embedding Difference\nMax: {diff_img.max():.1f}",
        fontsize=12,
        fontweight="bold",
    )
    ax7.axis("off")
    cbar7 = plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    # Statistics panel
    ax8 = fig1.add_subplot(gs1[1, 3])
    ax8.axis("off")

    stats_text = f"""QUALITY METRICS
{'='*28}

    Similarity:      {sim:.4f}
    BER:             {normalized_hamming:.4f}
    Hamming Dist:    {hamming_dist}

    MATCHING BITS:   {matching_bits:4d}
    Both 0:        {both_zero:4d}
    Both 1:        {both_one:4d}

    ERROR BITS:      {differing_bits:4d}
    False Pos:     {orig_zero_ext_one:4d}
    False Neg:     {orig_one_ext_zero:4d}

    PARAMETERS
    ALPHA (LL):   {ALPHA:.2f}

    STATUS:          {status}
"""

    ax8.text(
        0.05,
        0.95,
        stats_text,
        transform=ax8.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round",
            facecolor=bbox_color,
            alpha=0.4,
            edgecolor="black",
            linewidth=1.5,
        ),
    )

    fig1.suptitle(
        f"Watermark Extraction Verification | Similarity: {sim:.4f} | Status: {status}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    output_path1 = os.path.join("./", f"{output_prefix}extraction_overview.png")
    plt.savefig(
        output_path1, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"\nSaved overview plot to: {output_path1}")
    plt.close(fig1)

    print("=" * 80 + "\n")

    return {
        "similarity": sim,
        "matching_bits": matching_bits,
        "differing_bits": differing_bits,
        "hamming_distance": hamming_dist,
        "status": status,
        "original_watermark": original_watermark,
        "extracted_watermark": extracted_watermark,
    }
