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
    alpha = 20  # Embedding strength for LH and HL subbands
    beta = 30 # Embedding strength for LL subband (typically lower than alpha)
    mark_path = "crispymcmark.npy"

    print("===============================================")
    print("CrispyMcMark Embedding Tester")
    print("===============================================")
    print(f"Alpha (LH/HL): {alpha}")
    print(f"Beta (LL):     {beta}")

    # =================================================================
    #                       IMAGE GENERATION
    # =================================================================
    image_index = "0002"
    print(f"Image:         {image_index}")
    watermarked, watermark = embedding(
        "./challenge_images/"+image_index+".bmp", mark_path, alpha, beta
    )
    watermarked = watermarked.astype(np.uint8)
    original = cv2.imread("./images/"+image_index+".bmp", 0)

    # =================================================================
    #                      EMBEDDING VERIFICATION 
    # =================================================================
    wpsnr_val = wpsnr(original, watermarked)
    print(f"\nWatermark Quality:")
    print(f"  WPSNR: {wpsnr_val:6.2f} dB")

    # VERIFY WATERMARK EXTRACTION FIRST
    extraction_results = verify_watermark_extraction(
        original, watermarked, alpha, beta, mark_path, 
        output_prefix=f"ciao"
    )
    
    if extraction_results['similarity'] < 0.7:
        print("\n[WARNING] Watermark extraction is not working properly!")
        print("  Skipping attacks as detection may not be reliable.")
        return
 

    # =================================================================
    #                      RUN ATTACK SCRIPT 
    # =================================================================
    cv2.imwrite("./watermarked_groups_images/crispymcmark_"+image_index+".bmp", watermarked)
    detection_functions = {
        "crispymcmark": lambda i1, i2, i3: detection(i1, i2, i3,alpha,beta),
    } 
    attack.full_attack(detection_functions)

if __name__ == "__main__":
    main()
