from matplotlib import pyplot as plt
import numpy as np
import os
import cv2

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


def edges_mask(img, low_threshold=100, high_threshold=200, dilate_iter=1):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img.astype(np.uint8), low_threshold, high_threshold)
    if dilate_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, k, iterations=dilate_iter)
    return edges.astype(bool)


def noisy_mask(img, window=7, percentile=90, dilate_iter=0):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    sq = img * img
    k = (window, window)
    mean = cv2.boxFilter(img, ddepth=-1, ksize=k, normalize=True)
    mean_sq = cv2.boxFilter(sq, ddepth=-1, ksize=k, normalize=True)
    var = mean_sq - mean * mean
    thr = np.percentile(var, percentile)
    mask = var > thr
    if dilate_iter > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask.astype(np.uint8), se, iterations=dilate_iter).astype(
            bool
        )
    return mask


# TODO: note the percenile paramter can be tweaked, currently 3%-5% covers more than enough
def entropy_mask(img, block_size=16, entropy_exp=3, energy_thr=50, percentile=3):
    """
    Mask blocks with highest SVD flatness score (similar to embedding block selection).
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    h, w = img.shape
    mask = np.zeros_like(img, dtype=bool)
    scores = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i : i + block_size, j : j + block_size]
            if block.shape[0] < block_size or block.shape[1] < block_size:
                continue
            U, S, V = np.linalg.svd(block, full_matrices=False)
            S /= S.sum() + 1e-8
            entropy = -np.sum(S * np.log2(S + 1e-8)) / np.log2(len(S))
            energy = np.var(block)
            score = (entropy**entropy_exp) * np.exp(-energy / energy_thr)
            scores.append((i, j, score))

    # Select blocks with highest scores (most likely to be used for embedding)
    scores = sorted(scores, key=lambda x: x[2], reverse=True)[:16]
    for i, j, _ in scores:
        mask[i : i + block_size, j : j + block_size] = True

    return mask

    # Generate watermark if needed
    # if not os.path.exists(mark_path):
    #     mark = np.random.uniform(0.0, 1.0, mark_size)
    #     mark = np.uint8(np.rint(mark))
    #     np.save(mark_path, mark)


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

    # print(f"\nBit Pattern Distribution:")
    # print(f"  Both 0:            {both_zero} ({both_zero/total_bits*100:.2f}%)")
    # print(f"  Both 1:            {both_one} ({both_one/total_bits*100:.2f}%)")
    # print(
    #     f"  Orig=1, Ext=0:     {orig_one_ext_zero} ({orig_one_ext_zero/total_bits*100:.2f}%)"
    # )
    # print(
    #     f"  Orig=0, Ext=1:     {orig_zero_ext_one} ({orig_zero_ext_one/total_bits*100:.2f}%)"
    # )

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
