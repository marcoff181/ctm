import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from matplotlib.patches import Patch
from embedding import svd_flat_score, select_best_blocks, BLOCK_SIZE, N_BLOCKS
from detection import identify_watermarked_blocks


def plot_full_overview(
    image, watermarked, embedded_blocks, detected_blocks, save_path=None
):
    """Plot overview with: watermarked image, block overview, block difference heatmap, singular value change heatmap."""
    # Prepare block difference heatmap
    diff_img = np.abs(watermarked.astype(float) - image.astype(float))

    # Prepare singular value change heatmap
    h, w = image.shape
    sv_changes = np.zeros((h // BLOCK_SIZE, w // BLOCK_SIZE))
    for i in range(0, h, BLOCK_SIZE):
        for j in range(0, w, BLOCK_SIZE):
            block_orig = image[i : i + BLOCK_SIZE, j : j + BLOCK_SIZE]
            block_wm = watermarked[i : i + BLOCK_SIZE, j : j + BLOCK_SIZE]
            if block_orig.shape != (BLOCK_SIZE, BLOCK_SIZE):
                continue
            LL_orig = pywt.wavedec2(block_orig, wavelet="haar", level=1)[0]
            LL_wm = pywt.wavedec2(block_wm, wavelet="haar", level=1)[0]
            S_orig = np.linalg.svd(LL_orig, full_matrices=False)[1][0]
            S_wm = np.linalg.svd(LL_wm, full_matrices=False)[1][0]
            sv_changes[i // BLOCK_SIZE, j // BLOCK_SIZE] = np.abs(S_wm - S_orig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    plt.subplots_adjust(hspace=0.15, wspace=0.15)

    # Top left: Watermarked image
    axes[0, 0].imshow(watermarked, cmap="gray")
    axes[0, 0].set_title("Watermarked Image", fontsize=15, fontweight="bold")
    axes[0, 0].axis("off")

    # Top right: Block overview (embedding/detection)
    block_overview = np.zeros_like(image, dtype=np.uint8)
    axes[0, 1].imshow(image, cmap="gray")
    for loc in embedded_blocks:
        rect = plt.Rectangle(
            (loc[1], loc[0]),
            BLOCK_SIZE,
            BLOCK_SIZE,
            linewidth=2.5,
            edgecolor="blue",
            facecolor="none",
        )
        axes[0, 1].add_patch(rect)
    for loc in detected_blocks:
        color = "orange"
        if loc in embedded_blocks:
            color = "green"
        rect = plt.Rectangle(
            (loc[1], loc[0]),
            BLOCK_SIZE,
            BLOCK_SIZE,
            linewidth=2.5,
            edgecolor=color,
            facecolor="none",
        )
        axes[0, 1].add_patch(rect)
    axes[0, 1].set_title(
        "Block Overview\nEmbedded (blue), Detected (orange), Both (green)",
        fontsize=15,
        fontweight="bold",
    )
    axes[0, 1].axis("off")
    legend_elements = [
        Patch(
            edgecolor="blue", facecolor="none", label="Embedded Block", linewidth=2.5
        ),
        Patch(
            edgecolor="orange", facecolor="none", label="Detected Block", linewidth=2.5
        ),
        Patch(
            edgecolor="green",
            facecolor="none",
            label="Embedded & Detected",
            linewidth=2.5,
        ),
    ]
    axes[0, 1].legend(
        handles=legend_elements, loc="upper right", fontsize=12, frameon=True
    )

    # Bottom left: Block difference heatmap
    im1 = axes[1, 0].imshow(diff_img, cmap="hot")
    axes[1, 0].set_title("Block Difference Heatmap", fontsize=15, fontweight="bold")
    axes[1, 0].axis("off")
    cbar1 = plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar1.set_label("Absolute Difference", fontsize=12, fontweight="bold")
    for loc in embedded_blocks:
        rect = plt.Rectangle(
            (loc[1], loc[0]),
            BLOCK_SIZE,
            BLOCK_SIZE,
            linewidth=2,
            edgecolor="blue",
            facecolor="none",
        )
        axes[1, 0].add_patch(rect)
    for loc in detected_blocks:
        rect = plt.Rectangle(
            (loc[1], loc[0]),
            BLOCK_SIZE,
            BLOCK_SIZE,
            linewidth=2,
            edgecolor="orange",
            facecolor="none",
        )
        axes[1, 0].add_patch(rect)

    # Bottom right: Singular value change heatmap
    im2 = axes[1, 1].imshow(sv_changes, cmap="coolwarm")
    axes[1, 1].set_title(
        "Singular Value Change (LL[0])", fontsize=15, fontweight="bold"
    )
    axes[1, 1].axis("off")
    cbar2 = plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar2.set_label("ΔS₀ (LL)", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Watermark Embedding & Detection: Full Overview",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def main(image_path, watermarked_path):
    image = cv2.imread(image_path, 0)
    watermarked = cv2.imread(watermarked_path, 0)
    print(f"Loaded image: {image.shape}")

    strength_map = np.zeros_like(image)
    embedded_blocks = select_best_blocks(image, strength_map)
    detected_blocks = identify_watermarked_blocks(image, watermarked)

    plot_full_overview(
        image,
        watermarked,
        embedded_blocks,
        detected_blocks,
        save_path="full_embedding_detection_overview.png",
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python visualize_embedding.py <image_path> <watermarked_path>")
    else:
        main(sys.argv[1], sys.argv[2])
