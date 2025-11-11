import cv2

from plot.utilities import verify_watermark_extraction
from embedding import embedding, ALPHA as alpha
from detection_crispymcmark import detection
from wpsnr import wpsnr


def main(image_name):
    group_name = "crispymcmark"
    mark_path = f"{group_name}.npy"

    original_path = f"./challenge_images/{image_name}"
    watermarked_path = f"./watermarked_groups_images/{group_name}_{image_name}"

    print("\n" + "=" * 50)
    print(f"{group_name} Embedder")
    print("=" * 50 + "\n")

    print(f"Alpha: {alpha}")
    print(f"Image: {image_name}")

    watermarked = embedding(original_path, mark_path)
    cv2.imwrite(watermarked_path, watermarked)

    original = cv2.imread(original_path, 0)

    # verification
    wpsnr_val = wpsnr(original, watermarked)
    print("\nWatermark Quality:")
    print(f"  WPSNR: {wpsnr_val:6.2f} dB")

    _ = verify_watermark_extraction(original, watermarked, watermarked, mark_path)

    # sanity check
    found_in_original, _ = detection(original_path, watermarked_path, original_path)
    if found_in_original:
        print("ERROR: watermark detected in original image")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python crispy_embedder.py <image_name>")
    else:
        main(sys.argv[1])
