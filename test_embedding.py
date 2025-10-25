import cv2
import numpy as np
from embedding import embedding
from detection import (
    detection,
    verify_watermark_extraction,
)
from wpsnr import wpsnr
import attack

def main():
    alpha = 5.0  # Match the ALPHA constant in embedding.py
    mark_path = "crispymcmark.npy"
    image_index = "0005"
    original_path = f"./challenge_images/{image_index}.bmp"
    watermarked_path = f"./watermarked_groups_images/crispymcmark_{image_index}.bmp"

    print("===============================================")
    print("CrispyMcMark Embedding Tester")
    print("===============================================")
    print(f"Alpha: {alpha}")
    print(f"Image: {image_index}")

    # embedding
    watermarked= embedding(original_path, mark_path)
    watermarked = watermarked.astype(np.uint8)
    cv2.imwrite(watermarked_path,watermarked)

    original = cv2.imread(original_path, 0)

    # verification
    wpsnr_val = wpsnr(original, watermarked)
    print(f"\nWatermark Quality:")
    print(f"  WPSNR: {wpsnr_val:6.2f} dB")

    extraction_results = verify_watermark_extraction(
        original, 
        watermarked, 
        watermarked,
        mark_path
    )

    # sanity check
    found_in_original, _ = detection(original_path,watermarked_path,original_path)
    if found_in_original:
        print("ERROR: watermark detected in original image")
        return
    
    # run attacks
    detection_functions = {
        "crispymcmark": lambda i1, i2, i3: detection(i1, i2, i3),
    } 
    attack.full_attack(detection_functions)

if __name__ == "__main__":
    main()
