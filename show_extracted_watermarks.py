import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from detection_crispymcmark import extraction, similarity
from embedding import BLOCK_SIZE, ALPHA
from attack import attack_config, param_converters

def parse_attack_from_filename(filename):
    # Example: crispymcmark_0000_JPEG_0.90_destroyed.bmp
    basename = os.path.basename(filename)
    parts = basename.split("_")
    # Find attack name and parameter
    for i in range(len(parts)):
        if parts[i] in attack_config:
            attack_name = parts[i]
            try:
                param = float(parts[i+1])
            except Exception:
                param = None
            return attack_name, param
    return None, None

def attack_image(original, attack_name, param):
    """Apply the attack specified by attack_name and param to the original image."""
    if attack_name in attack_config and param is not None:
        attack_func = attack_config[attack_name]
        return attack_func(original.copy(), param)
    return original.copy()

def show_watermarks(original_path, watermarked_path, attacked_path, watermark_path="crispymcmark.npy"):
    original = cv2.imread(original_path, 0)
    watermarked = cv2.imread(watermarked_path, 0)
    attacked = cv2.imread(attacked_path, 0)

    if original is None or watermarked is None or attacked is None:
        raise ValueError("One or more input images could not be loaded. Check the file paths.")

    # Load real watermark
    real_watermark = np.load(watermark_path)

    # Parse attack info from filename
    attack_name, param = parse_attack_from_filename(attacked_path)
    if attack_name is None or param is None:
        print("Could not parse attack info from filename, using original as is.")
        original_attacked = original.copy()
    else:
        original_attacked = attack_image(original, attack_name, param)

    # Extract watermarks
    wm_from_watermarked, mask = extraction(original, watermarked, watermarked)
    wm_from_watermarked_attacked, _ = extraction(original, watermarked, attacked, mask)
    wm_from_original, _ = extraction(original, original, original, mask)
    wm_from_original_attacked, _ = extraction(original, original, original_attacked, mask)

    size = int(np.sqrt(wm_from_watermarked.size))

    # Calculate similarities
    sim_watermarked = similarity(real_watermark, wm_from_watermarked)
    sim_attacked = similarity(real_watermark, wm_from_watermarked_attacked)
    sim_original = similarity(real_watermark, wm_from_original)
    sim_original_attacked = similarity(real_watermark, wm_from_original_attacked)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    image_titles = [
        "Watermarked Image",
        f"Attacked Image (Watermarked)\n{attack_name}({param})",
        "Original Image",
        f"Original + {attack_name}({param})"
    ]
    watermark_titles = [
        f"Extracted from Watermarked\nSimilarity: {sim_watermarked:.4f}",
        f"Extracted from Watermarked + Attacked\nSimilarity: {sim_attacked:.4f}",
        f"Extracted from Original\nSimilarity: {sim_original:.4f}",
        f"Extracted from Original + Attacked\nSimilarity: {sim_original_attacked:.4f}"
    ]
    images = [watermarked, attacked, original, original_attacked]
    watermarks = [
        wm_from_watermarked,
        wm_from_watermarked_attacked,
        wm_from_original,
        wm_from_original_attacked
    ]

    # Display images
    for i in range(4):
        axes[0, i].imshow(images[i], cmap="gray")
        axes[0, i].set_title(image_titles[i], fontsize=13, fontweight="bold")
        axes[0, i].axis("off")

    # Display extracted watermarks with similarity
    for i in range(4):
        axes[1, i].imshow(watermarks[i].reshape(size, size), cmap="plasma", vmin=-1, vmax=1)
        axes[1, i].set_title(watermark_titles[i], fontsize=13, fontweight="bold")
        axes[1, i].axis("off")

    fig.suptitle("Watermark Extraction Overview", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python show_extracted_watermarks.py <original> <watermarked> <attacked>")
    else:
        try:
            show_watermarks(sys.argv[1], sys.argv[2], sys.argv[3])
        except Exception as e:
            print(f"Error: {e}")