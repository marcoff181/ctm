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
    alpha = 5.11  # Match the ALPHA constant in embedding.py
    mark_path = "crispymcmark.npy"

    print("===============================================")
    print("CrispyMcMark Embedding Tester")
    print("===============================================")
    print(f"Alpha: {alpha}")

    # =================================================================
    #                       IMAGE GENERATION
    # =================================================================
    image_index = "0002"
    print(f"Image: {image_index}")
    
    # embedding() signature: (original_image_path, watermark_path, alpha, dwt_level)
    # Returns: (watermarked_image, watermark, Uwm, Vwm)
    watermarked, watermark, Uwm, Vwm = embedding(
        f"./challenge_images/{image_index}.bmp", 
        mark_path, 
        alpha,
        dwt_level=1  # Use 1-level DWT as per embedding.py
    )
    watermarked = watermarked.astype(np.uint8)
    original = cv2.imread(f"./challenge_images/{image_index}.bmp", 0)

    # =================================================================
    #                      EMBEDDING VERIFICATION 
    # =================================================================
    wpsnr_val = wpsnr(original, watermarked)
    print(f"\nWatermark Quality:")
    print(f"  WPSNR: {wpsnr_val:6.2f} dB")

    # verify_watermark_extraction() signature: (original, watermarked, alpha, mark_path, dwt_level, output_prefix)
    # But looking at detection.py, it takes: (original, watermarked, alpha, mark_path)
    extraction_results = verify_watermark_extraction(
        original, 
        watermarked, 
        watermarked,
        alpha, 
        mark_path
    )
    
    if extraction_results['similarity'] < 0.7:
        print("\n[WARNING] Watermark extraction is not working properly!")
        print("  Skipping attacks as detection may not be reliable.")
        return

    # =================================================================
    #                      RUN ATTACK SCRIPT 
    # =================================================================
    output_path = f"./watermarked_groups_images/crispymcmark_{image_index}.bmp"
    cv2.imwrite(output_path, watermarked)
    print(f"\nWatermarked image saved to: {output_path}")
    
    # detection() signature: (input1, input2, input3)
    # where input1=original, input2=watermarked, input3=attacked
    detection_functions = {
        "crispymcmark": lambda i1, i2, i3: detection(i1, i2, i3),
    } 
    attack.full_attack(detection_functions)

if __name__ == "__main__":
    main()
