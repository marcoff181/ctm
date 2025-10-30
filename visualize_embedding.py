import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from matplotlib.patches import Patch
from embedding import svd_flat_score, select_best_blocks, BLOCK_SIZE, N_BLOCKS
from detection_crispymcmark import identify_watermarked_blocks, extraction


def plot_full_overview(
    image, watermarked, attacked, block_mask, save_path=None
):
    diff_img = np.abs(watermarked.astype(float) - image.astype(float))
    h, w = image.shape
    sv_changes = np.zeros((h // BLOCK_SIZE, w // BLOCK_SIZE))
    sv_attacked_diff = np.zeros((h // BLOCK_SIZE, w // BLOCK_SIZE))

    # Compute singular value changes for all blocks
    for i in range(0, h, BLOCK_SIZE):
        for j in range(0, w, BLOCK_SIZE):
            block_orig = image[i : i + BLOCK_SIZE, j : j + BLOCK_SIZE]
            block_wm = watermarked[i : i + BLOCK_SIZE, j : j + BLOCK_SIZE]
            block_att = attacked[i : i + BLOCK_SIZE, j : j + BLOCK_SIZE]
            if block_orig.shape != (BLOCK_SIZE, BLOCK_SIZE):
                continue
            LL_orig = pywt.wavedec2(block_orig, wavelet="haar", level=1)[0]
            LL_wm = pywt.wavedec2(block_wm, wavelet="haar", level=1)[0]
            LL_att = pywt.wavedec2(block_att, wavelet="haar", level=1)[0]
            S_wm = np.linalg.svd(LL_wm, full_matrices=False)[1][0]
            S_att = np.linalg.svd(LL_att, full_matrices=False)[1][0]
            sv_changes[i // BLOCK_SIZE, j // BLOCK_SIZE] = np.abs(S_wm - np.linalg.svd(LL_orig, full_matrices=False)[1][0])
            sv_attacked_diff[i // BLOCK_SIZE, j // BLOCK_SIZE] = S_att - S_wm

    # Mask for embedded blocks only
    mask_matrix = np.zeros_like(sv_attacked_diff, dtype=bool)
    for x, y in block_mask:
        mask_matrix[x // BLOCK_SIZE, y // BLOCK_SIZE] = True

    # Set non-embedded blocks to np.nan for visualization
    sv_attacked_diff_masked = np.where(mask_matrix, sv_attacked_diff, np.nan)

    # --- Only calculate for embedded blocks (block_mask) ---
    sv_changes_masked = np.full((h // BLOCK_SIZE, w // BLOCK_SIZE), np.nan)
    sv_attacked_diff_masked = np.full((h // BLOCK_SIZE, w // BLOCK_SIZE), np.nan)
    sv_extracted = np.full((h // BLOCK_SIZE, w // BLOCK_SIZE), np.nan)

    for x, y in block_mask:
        block_orig = image[x : x + BLOCK_SIZE, y : y + BLOCK_SIZE]
        block_wm = watermarked[x : x + BLOCK_SIZE, y : y + BLOCK_SIZE]
        block_att = attacked[x : x + BLOCK_SIZE, y : y + BLOCK_SIZE]
        LL_orig = pywt.wavedec2(block_orig, wavelet="haar", level=1)[0]
        LL_wm = pywt.wavedec2(block_wm, wavelet="haar", level=1)[0]
        LL_att = pywt.wavedec2(block_att, wavelet="haar", level=1)[0]
        S_orig = np.linalg.svd(LL_orig, full_matrices=False)[1][0]
        S_wm = np.linalg.svd(LL_wm, full_matrices=False)[1][0]
        S_att = np.linalg.svd(LL_att, full_matrices=False)[1][0]
        sv_changes_masked[x // BLOCK_SIZE, y // BLOCK_SIZE] = np.abs(S_wm - S_orig)
        sv_attacked_diff_masked[x // BLOCK_SIZE, y // BLOCK_SIZE] = S_att - S_wm
        from embedding import ALPHA
        sv_extracted[x // BLOCK_SIZE, y // BLOCK_SIZE] = (S_att - S_orig) / ALPHA

    # Find common color scale for both heatmaps (masked only)
    min_val = np.nanmin([np.nanmin(sv_changes_masked), np.nanmin(sv_attacked_diff_masked)])
    max_val = np.nanmax([np.nanmax(sv_changes_masked), np.nanmax(sv_attacked_diff_masked)])

    # Create a 2x4 grid for all plots
    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    plt.subplots_adjust(hspace=0.25, wspace=0.25)

    # Top row
    axes[0, 0].imshow(watermarked, cmap="gray")
    axes[0, 0].set_title("Watermarked Image", fontsize=15, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image, cmap="gray")
    for loc in block_mask:
        rect = plt.Rectangle(
            (loc[1], loc[0]),
            BLOCK_SIZE,
            BLOCK_SIZE,
            linewidth=2.5,
            edgecolor="blue",
            facecolor="none",
        )
        axes[0, 1].add_patch(rect)
    axes[0, 1].set_title("Embedded Blocks (blue)", fontsize=15, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(attacked, cmap="gray")
    for loc in block_mask:
        rect = plt.Rectangle(
            (loc[1], loc[0]),
            BLOCK_SIZE,
            BLOCK_SIZE,
            linewidth=2.5,
            edgecolor="red",
            facecolor="none",
        )
        axes[0, 2].add_patch(rect)
    axes[0, 2].set_title("Attacked Image + Embedded Blocks", fontsize=15, fontweight="bold")
    axes[0, 2].axis("off")

    # Leave axes[0, 3] empty or use for legend
    axes[0, 3].axis("off")
    legend_elements = [
        Patch(edgecolor="blue", facecolor="none", label="Embedded Block", linewidth=2.5),
        Patch(edgecolor="red", facecolor="none", label="Embedded Block on Attacked", linewidth=2.5),
    ]
    axes[0, 3].legend(handles=legend_elements, loc="center", fontsize=14, frameon=True)

    # Bottom row
    im1 = axes[1, 0].imshow(diff_img, cmap="hot")
    axes[1, 0].set_title("Block Difference Heatmap", fontsize=15, fontweight="bold")
    axes[1, 0].axis("off")
    cbar1 = plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar1.set_label("Absolute Difference", fontsize=12, fontweight="bold")

    im2 = axes[1, 1].imshow(sv_changes_masked, cmap="coolwarm", vmin=min_val, vmax=max_val)
    axes[1, 1].set_title("Singular Value Change (LL[0])\nEmbedded Blocks", fontsize=15, fontweight="bold")
    axes[1, 1].axis("off")
    cbar2 = plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar2.set_label("ΔS₀ (LL)", fontsize=12, fontweight="bold")

    im3 = axes[1, 2].imshow(sv_attacked_diff_masked, cmap="viridis", vmin=min_val, vmax=max_val)
    axes[1, 2].set_title("Singular Value Diff (Attacked - Watermarked)\nEmbedded Blocks", fontsize=15, fontweight="bold")
    axes[1, 2].axis("off")
    cbar3 = plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar3.set_label("ΔS₀ (Attacked - Watermarked)", fontsize=12, fontweight="bold")

    im4 = axes[1, 3].imshow(sv_extracted, cmap="plasma", vmin=-1, vmax=1)
    axes[1, 3].set_title("Extracted Watermark Value\n(Attacked - Original) / ALPHA", fontsize=15, fontweight="bold")
    axes[1, 3].axis("off")
    cbar4 = plt.colorbar(im4, ax=axes[1, 3], fraction=0.046, pad=0.04)
    cbar4.set_label("Extracted Value", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Watermark Embedding & Detection: Full Overview",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def main(image_path, watermarked_path, attacked_path):
    image = cv2.imread(image_path, 0)
    watermarked = cv2.imread(watermarked_path, 0)
    attacked = cv2.imread(attacked_path, 0)
    print(f"Loaded image: {image.shape}")

    # Get block_mask from extraction
    _, block_mask = extraction(image, watermarked, watermarked)
    # block_mask is a list of (x, y) tuples

    plot_full_overview(
        image,
        watermarked,
        attacked,
        block_mask,
        save_path="full_embedding_detection_overview.png",
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python visualize_embedding.py <image_path> <watermarked_path> <attacked_path>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
