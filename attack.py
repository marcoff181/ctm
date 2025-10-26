import os
import cv2
import numpy as np
import importlib.util
import sys
import glob

import pandas as pd
from attack_functions import awgn, blur, sharpening, median, resizing, jpeg_compression
from utilities import edges_mask, noisy_mask, entropy_mask

MIN_WPSNR = 35.00
BIN_SEARCH_ITERATIONS = 6

# hardcoded stuff
WATERMARKED_IMAGES_PATH = "./watermarked_groups_images/"
ATTACKED_IMAGES_PATH = "./attacked_groups_images/"
ORIGINAL_IMAGES_PATH = "./challenge_images/"

CSV_ATTACKS_LOG_PATH = "./attacks_log.csv"


# conversion between the input value 0.0..1.0 and the actual parameters of each attack function
param_converters = {
    "JPEG": lambda x: int(round((1 - x) * 100)),
    "Blur": lambda x: (x + 0.15) * 1.2,
    "AWGN": lambda x: x * 30,
    "Resize": lambda x: np.round(((1 - x) + 0.4) * 512) / 512,
    "Median": lambda x: [[1, 3], [3, 1], [3, 3], [3, 5], [5, 3]][int(round(x * 4))],
    "Sharp": lambda x: (x * 0.07) + 0.035,
}

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


def discover_detection_functions():
    """
    Automatically discover and import detection functions from detection_*.py or detection_*.pyc files.

    Returns:
        dict: Dictionary mapping group names to their detection functions.
              Example: {"crispymcmark": <function>, "luigi": <function>}
    """
    detection_functions = {}
    found_groups = set()

    # Search for detection_*.py files in current directory
    detection_files = glob.glob("*/detection_*.py*")

    # Also search for detection_*.pyc files in current directory
    # pyc_files = glob.glob("detection_*.pyc")
    # detection_files.extend(pyc_files)

    # # Search in subdirectory
    # py_files = glob.glob("*/detection_*.py*")
    # detection_files.extend(py_files)

    for detection_file in detection_files:
        base_name = os.path.basename(detection_file)

        if base_name.startswith("detection_") and base_name.endswith(".py"):
            group_name = base_name[10:-3]
        elif base_name.startswith("detection_") and base_name.endswith(".pyc"):
            group_name = base_name[10:].split(".")[0]
        else:
            continue

        if group_name in found_groups:
            continue

        try:
            # Dynamically import the detection module
            module_name = f"detection_{group_name}"
            spec = importlib.util.spec_from_file_location(module_name, detection_file)

            if spec is None or spec.loader is None:
                print(f"[WARNING] Could not load module spec from {detection_file}")
                continue

            detection_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = detection_module
            spec.loader.exec_module(detection_module)

            # Check if detection function exists
            if not hasattr(detection_module, "detection"):
                print(
                    f"[WARNING] Module {module_name} does not have 'detection' function"
                )
                continue

            # Store the detection function
            detection_functions[group_name] = detection_module.detection
            found_groups.add(group_name)
            print(f"[INFO] Loaded detection function for group: {group_name}")

        except Exception as e:
            print(f"[ERROR] Failed to import {detection_file}: {e}")
            continue

    return detection_functions


def log_attack(detected, path, wpsnr, attack_name, params, mask):
    with open(CSV_ATTACKS_LOG_PATH, "a") as log:
        # Image name, Group attacked,  WPSNR, Attack(s) with parameters
        base = os.path.splitext(os.path.basename(path))[0]
        if "_" in base:
            group, image = base.split("_", 1)
        else:
            group, image = base, base

        # write header if file is empty/new
        if log.tell() == 0:
            log.write(
                "Watermark detected,Image,Group,WPSNR,Attack(s) with parameters\n"
            )

        wpsnr_str = "" if wpsnr is None else f"{wpsnr:.2f}"
        if mask is not None:
            log.write(
                f"{detected},{image},{group},{wpsnr_str},{attack_name}({params}),{mask}\n"
            )
        else:
            log.write(
                f"{detected},{image},{group},{wpsnr_str},{attack_name}({params})\n"
            )


def clear_tmp():
    # remove any remaining files in tmp_attacks (cleanup)
    tmp_dir = "./tmp_attacks"
    for tmp_file in glob.glob(os.path.join(tmp_dir, "*")):
        try:
            os.remove(tmp_file)
        except Exception as e:
            print(f"[WARNING] Could not remove temp file {tmp_file}: {e}")


def move_best_to_attacked_folder(img_path):
    if img_path and os.path.exists(img_path):
        dst = os.path.join(
            ATTACKED_IMAGES_PATH,
            "crispymcmark_" + os.path.basename(img_path).split("-")[0] + ".bmp",
        )
        try:
            os.replace(img_path, dst)
            print(f"\n[INFO] Choosed attacked: {img_path}")
            print(f"[INFO] Moving {img_path} to {dst}")
        except Exception as e:
            print(f"[WARNING] Could not move {img_path} -> {dst}: {e}")


def bin_search_attack(
    original_path, watermarked_path, detection, mask, iterations, mask_name=None
):
    watermarked = cv2.imread(watermarked_path, 0).copy()

    results = []

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

            params = param_converters[attack_name](mid)

            attack_img_path = f"./tmp_attacks/{watermarked_path.split('/')[-1].split('.')[0]}-{attack_name}-{params}.bmp"
            cv2.imwrite(attack_img_path, attacked_img)

            detected, wpsnr_val = detection(
                original_path, watermarked_path, attack_img_path
            )

            log_attack(
                detected,
                watermarked_path,
                wpsnr_val,
                attack_name,
                params,
                mask_name,
            )

            if not detected and wpsnr_val >= MIN_WPSNR:
                best_param, best_wpsnr = mid, wpsnr_val
                best_attacked = attack_img_path
                high = mid
            else:
                low = mid

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
                    "Image": best_attacked,
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

    # move best attacked image, based on wpsnr, to ATTACKED_IMAGES_PATH
    sorted_results = sorted(
        [r for r in results if r.get("WPSNR") is not np.nan],
        key=lambda k: k.get("WPSNR"),
    )

    if len(sorted_results) != 0:
        img_path = sorted_results[-1].get("Image")
        move_best_to_attacked_folder(img_path)

    clear_tmp()
    return results


def setup():
    # Setup directories
    os.makedirs(ATTACKED_IMAGES_PATH, exist_ok=True)
    os.makedirs(WATERMARKED_IMAGES_PATH, exist_ok=True)
    os.makedirs(ORIGINAL_IMAGES_PATH, exist_ok=True)


def full_attack(detection_functions):
    """Start the attack suite"""

    print("\n" + "=" * 50)
    print("CrispyMcMark Attack Suite")
    print("=" * 50 + "\n")

    if not detection_functions:
        print(
            "[ERROR] No detection functions found. Make sure detection_*.py/c files exist."
        )
        return

    # Load image
    img_list = sorted(os.listdir(WATERMARKED_IMAGES_PATH))
    print(f"[INFO] Start attacking:\n{', '.join(img_list)}")

    for filename in img_list:
        watermarked_path = os.path.join(WATERMARKED_IMAGES_PATH, filename)

        # expected image name is groupName_imageName.bmp
        name_without_ext = os.path.splitext(filename)[0]

        if "_" not in name_without_ext:
            print(
                f"[WARNING] Skipping {filename}: filename does not match pattern 'groupName_imageName.bmp'"
            )
            continue

        group_name, image_name = name_without_ext.split("_", 1)

        # Check if detection function exists for this group
        if group_name not in detection_functions:
            print(f"[ERROR] No detection function found for group: {group_name}")
            print(f"[ERROR] Available groups: {list(detection_functions.keys())}")
            continue

        det_fun = detection_functions[group_name]

        print("\n" + "-" * 50)
        print(f"Attacking image: {image_name}")
        print(f"Group: {group_name}")

        # find out which is the original challenge image to compare to
        original_path = os.path.join(ORIGINAL_IMAGES_PATH, image_name + ".bmp")
        original = cv2.imread(original_path, 0)

        detected, wpsnr_val = det_fun(original_path, watermarked_path, watermarked_path)

        print("\nNon-attacked image:")
        print(f"  WPSNR: {wpsnr_val:6.2f} dB")
        print(f"  detected: {detected}")
        if detected != 1:
            print(
                "\n[WARNING] Detection did not detect watermark in non-attacked image, skipping..."
            )
            continue

        print("\nBinary search with no mask...")
        mask = original >= 0
        res = bin_search_attack(
            original_path, watermarked_path, det_fun, mask, BIN_SEARCH_ITERATIONS
        )

        print("Binary search with edges mask...")
        emask = edges_mask(original)
        res = bin_search_attack(
            original_path,
            watermarked_path,
            det_fun,
            emask,
            BIN_SEARCH_ITERATIONS,
            "edges mask",
        )

        print("Binary search with noisy mask...")
        nmask = noisy_mask(original)
        res = bin_search_attack(
            original_path,
            watermarked_path,
            det_fun,
            nmask,
            BIN_SEARCH_ITERATIONS,
            "noisy mask",
        )

        print("Binary search with entropy mask...")
        emask = entropy_mask(original)
        res = bin_search_attack(
            original_path,
            watermarked_path,
            det_fun,
            emask,
            BIN_SEARCH_ITERATIONS,
            "entropy mask",
        )

        # TODO: add parallelization

        print("-" * 50 + "\n")


def main():
    # Automatically discover all detection functions
    detection_functions = discover_detection_functions()

    if not detection_functions:
        print("[ERROR] No detection modules found. Please create detection_*.py files.")
        return

    full_attack(detection_functions)


if __name__ == "__main__":
    main()
