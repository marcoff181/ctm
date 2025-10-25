import os
import cv2
import numpy as np
from detection_crispymcmark import detection as crispy_detection

import pandas as pd
from attack_functions import awgn, blur, sharpening, median, resizing, jpeg_compression
from utilities import edges_mask, noisy_mask, entropy_mask

MIN_WPSNR = 35.00

# hardcoded stuff
input_dir = "./watermarked_groups_images/"
output_dir = "./attacked_groups_images/"
originals_dir = "./challenge_images/"
attacked_wpsnr_lower_bound = 35

# conversion between the input value 0.0..1.0 and the actual parameters of each attack function
param_converters = {
    "JPEG": lambda x: int(round((1 - x) * 100)),
    "Blur": lambda x: (x + 0.15) * 1.2,
    "AWGN": lambda x: x * 30,
    # pick closest number that is divisible by 512 so that when upscaling we come back to the same image size
    "Resize": lambda x: np.round(((1 - x) + 0.4) * 512) / 512,
    "Median": lambda x: [[1, 3], [3, 1], [3, 3], [3, 5], [5, 3]][int(round(x * 4))],
    "Sharp": lambda x: (x * 0.07) + 0.035,
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

detection_functions = {
    "crispymcmark": lambda i1, i2, i3: crispy_detection(i1, i2, i3, 10000, 1000),
    # "group_name" : lambda (orig,water,attack) : their function
}


# TODO: tweak iterations to find balance between speed and accuracy
def bin_search_attack(original_path, watermarked_path, detection, mask, iterations):
    original = cv2.imread(original_path, 0).copy()
    watermarked = cv2.imread(watermarked_path, 0).copy()

    results = []
    best_attacks = []

    for attack_name, attack_func in attack_config.items():
        low, high = 0.0, 1.0
        best_param, best_wpsnr = None, -np.inf
        best_attacked = None

        for _ in range(iterations):
            # converged
            if abs(high - low) < 1e-6:
                break

            mid = (low + high) / 2

            full_attacked_img = attack_func(watermarked.copy(), mid)
            attacked_img = np.where(mask, full_attacked_img, watermarked)

            attack_img_path = f"./tmp_attacks/{original_path.split('/')[-1].split('.')[0]}-{attack_name}-{mid}.bmp"
            cv2.imwrite(attack_img_path, attacked_img)

            detected, wpsnr_val = detection(
                original_path, watermarked_path, attack_img_path
            )
            actual_param = param_converters[attack_name](mid)

            if not detected and wpsnr_val > MIN_WPSNR:
                best_param, best_wpsnr = mid, wpsnr_val
                best_attacked = attacked_img.copy()
                high = mid
            else:
                low = mid

        if best_attacked is not None:
            best_attacks.append(best_attacked)

        if best_param is not None:
            actual_param = param_converters[attack_name](best_param)
            print(
                f"  ✓ {attack_name}: Optimal param = {actual_param:.4f} | WPSNR: {best_wpsnr:.2f} dB"
            )
            results.append(
                {
                    "Attack": attack_name,
                    "Best_Parameter": actual_param,
                    "WPSNR": best_wpsnr,
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
                    "Status": "Not Removed",
                }
            )
    return best_attacks


def full_attack(detection_functions):
    print("===============================================")
    print("CrispyMcMark Attack Suite")
    print("===============================================")

    # Setup directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(originals_dir, exist_ok=True)

    # Load image
    for filename in sorted(os.listdir(input_dir)):
        watermarked_path = os.path.join(input_dir, filename)

        # expected image name is groupName_imageName.bmp
        group_name, image_name = os.path.splitext(filename)[0].split("_")
        det_fun = detection_functions[group_name]

        watermarked = cv2.imread(watermarked_path, 0)
        print(f"\nAttacking image: {image_name}")
        print(f"Group: {group_name}")

        # find out which is the original challenge image to compare to
        original_path = os.path.join(originals_dir, image_name + ".bmp")
        original = cv2.imread(original_path, 0)

        detected, wpsnr_val = det_fun(original_path, watermarked_path, watermarked_path)
        print(f"\nNon-attacked image:")
        print(f"  WPSNR: {wpsnr_val:6.2f} dB")
        print(f"  detected: {detected}")
        if detected != 1:
            print(
                "\n[WARNING] Detection did not detect watermark in non-attcked image, skipping..."
            )
            continue

        bin_search_iterations = 6
        double_bin_search_iterations = 10

        print("\nBinary search with no mask...")
        mask = original >= 0
        res = bin_search_attack(
            original_path, watermarked_path, det_fun, mask, bin_search_iterations
        )
        # print(f"\nResults:\n{res}\n")

        print("Binary search with edges mask...")
        emask = edges_mask(original)
        res = bin_search_attack(
            original_path, watermarked_path, det_fun, emask, bin_search_iterations
        )
        # print(f"\nResults:\n{res}\n")

        print("Binary search with noisy mask...")
        nmask = noisy_mask(original)
        res = bin_search_attack(
            original_path, watermarked_path, det_fun, nmask, bin_search_iterations
        )

        print("Binary search with entropy mask...")
        emask = entropy_mask(original)

        # note for debugging
        # diff_img = np.abs(watermarked.astype(float) - original.astype(float))
        # show_images(diff_img, emask)
        res = bin_search_attack(
            original_path, watermarked_path, det_fun, emask, bin_search_iterations
        )

        # TODO: add parallelization
        # TODO: find best attack and save it in output

        # remove to run for all images
        return


def main():
    full_attack(detection_functions)


if __name__ == "__main__":
    main()
