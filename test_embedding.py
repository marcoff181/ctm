import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from skimage.transform import rescale

from embedding import embedding
from detection import (
    detection,
    extraction,
    similarity,
    compute_threshold,
    verify_watermark_extraction,
)
from wpsnr import wpsnr
import pandas as pd
from itertools import combinations
from attack_functions import awgn, blur, sharpening, median, resizing, jpeg_compression
from utilities import edges_mask, noisy_mask


# Configuration
# images_path = "./images"
alpha = 20  # Embedding strength for LH and HL subbands
beta = 30 # Embedding strength for LL subband (typically lower than alpha)
mark_size = 1024
mark_path = "./mark.npy"
input_dir = "./watermarked_groups_images/"
output_dir = "./attacked_groups_images/"
originals_dir = "./challenge_images/"
attacked_wpsnr_lower_bound = 35

# conversion between the input value 0.0..1.0 and the actual parameters of each attack function
param_converters = {
    "JPEG": lambda x: int(round((1 - x) * 100)),
    "Blur": lambda x: x * 10,
    "AWGN": lambda x: x * 50,
    "Resize": lambda x: max(0.001, 0.5 ** (x * 10)),
    "Median": lambda x: [1, 3, 5, 7][int(round(x * 3))],
    "Sharp": lambda x: x * 3,
}

# attacks that take as input a strenght value `x` between 0.0 and 1.0
attack_config = {
    "JPEG": lambda img, x: jpeg_compression(img, quality=param_converters["JPEG"](x)),
    "Blur": lambda img, x: blur(img, sigma=param_converters["Blur"](x)),
    "AWGN": lambda img, x: awgn(img, std=param_converters["AWGN"](x)),
    "Resize": lambda img, x: resizing(img, scale=param_converters["Resize"](x)),
    "Median": lambda img, x: median(img, kernel_size=param_converters["Median"](x)),
    "Sharp": lambda img, x: sharpening(
        img, sigma=1.0, alpha=param_converters["Sharp"](x)
    ),
}




# TODO: tweak iterations to find balance between speed and accuracy
def bin_search_attack(original, watermarked, detection, Uwm, Vwm, mask, alpha, beta, mark_path, iterations=6):
    results = []

    for attack_name, attack_func in attack_config.items():
        low, high = 0.0, 1.0
        best_param, best_wpsnr = None, -np.inf
        best_attacked = None

        for _ in range(iterations):
            mid = (low + high) / 2
            # converged
            if abs(high - low) < 1e-6:
                break

            try:
                full_attacked_img = attack_func(watermarked.copy(), mid)
                attacked_img = np.where(mask, full_attacked_img, watermarked)
                detected, wpsnr_val = detection(
                    original, watermarked, attacked_img, Uwm, Vwm, alpha=alpha, beta=beta
                )
                actual_param = param_converters[attack_name](mid)

                if not detected:
                    best_param, best_wpsnr = mid, wpsnr_val
                    best_attacked = attacked_img.copy()
                    high = mid
                else:
                    low = mid

            except Exception as e:
                print(f"Error during {attack_name} with param {mid}: {e}")
                break

        if best_param is not None:
            actual_param = param_converters[attack_name](best_param)
            
            # Verify watermark extraction after attack
            original_watermark = np.load(mark_path)
            extracted_after_attack = extraction(original, best_attacked, Uwm, Vwm, alpha=alpha, beta=beta)
            sim_after_attack = similarity(original_watermark, extracted_after_attack)
            
            print(
                f"  ✓ {attack_name}: Optimal param = {actual_param:.4f} | WPSNR: {best_wpsnr:.2f} dB"
            )
            results.append(
                {
                    "Attack": attack_name,
                    "Best_Parameter": actual_param,
                    "WPSNR": best_wpsnr,
                    "Similarity": sim_after_attack,
                    "Status": "Removed",
                }
            )
        else:
            print(f"  ✗ {attack_name}: Could not remove watermark")
            results.append(
                {
                    "Attack": attack_name,
                    "Best_Parameter": np.nan,
                    "WPSNR": np.nan,
                    "Similarity": np.nan,
                    "Status": "Not Removed",
                }
            )
    return pd.DataFrame(results)


def main():
    print("===============================================")
    print("CrispyMcMark Attack Suite")
    print("===============================================")
    print(f"Alpha (LH/HL): {alpha}")
    print(f"Beta (LL):     {beta}")

    # Setup directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(originals_dir, exist_ok=True)

    # TODO: hardcode Uwm and Vwm inside the detection func
    _, _, Uwm, Vwm = embedding("./challenge_images/0002.bmp", mark_path, alpha, beta)

    # TODO: remove ------------------------------------------------------------------
    # generate an image to simulate having images to attack
    if len(os.listdir(input_dir)) == 0:
        watermarked, watermark, Uwm, Vwm = embedding(
            "./challenge_images/0002.bmp", mark_path, alpha, beta
        )
        watermarked = watermarked.astype(np.uint8)
        cv2.imwrite("./watermarked_groups_images/cryspymcmark_0002.bmp", watermarked)
    # --------------------------------------------------------------------------------
    
    # Load image
    for filename in sorted(os.listdir(input_dir)):
        image_path = os.path.join(input_dir, filename)

        # expected image name is groupName_imageName.bmp
        group_name, image_name = os.path.splitext(filename)[0].split("_")

        watermarked = cv2.imread(image_path, 0)
        print(f"\nAttacking image: {image_name}")
        print(f"Group: {group_name}")

        # find out which is the original challenge image to compare to
        original_path = os.path.join(originals_dir, image_name + ".bmp")
        original = cv2.imread(original_path, 0)

        # Print quality metrics
        # TODO: leave to check what is the original WPSNR of the watermaked image
        wpsnr_val = wpsnr(original, watermarked)
        print(f"\nWatermark Quality:")
        print(f"  WPSNR: {wpsnr_val:6.2f} dB")


        # TODO: remove 
        # VERIFY WATERMARK EXTRACTION FIRST
        extraction_results = verify_watermark_extraction(
            original, watermarked, Uwm, Vwm, alpha, beta, mark_path, 
            output_prefix=f"{group_name}_{image_name}_"
        )
        
        if extraction_results['similarity'] < 0.7:
            print("\n[WARNING] Watermark extraction is not working properly!")
            print("  Skipping attacks as detection may not be reliable.")
            continue
        # -------------

        print("\nBinary search with no mask...")
        mask = original >= 0
        res = bin_search_attack(original, watermarked, detection, Uwm, Vwm, mask, alpha, beta, mark_path)
        print(f"\nResults:\n{res.to_string()}\n")

        print("Binary search with edges mask...")
        emask = edges_mask(original)
        res = bin_search_attack(original, watermarked, detection, Uwm, Vwm, emask, alpha, beta, mark_path)
        print(f"\nResults:\n{res.to_string()}\n")

        print("Binary search with noisy mask...")
        nmask = noisy_mask(original)
        res = bin_search_attack(original, watermarked, detection, Uwm, Vwm, nmask, alpha, beta, mark_path)

        # TODO: find best attack and save it in output

        # remove to run for all images
        # TODO: add parallelization
        return


if __name__ == "__main__":
    main()
