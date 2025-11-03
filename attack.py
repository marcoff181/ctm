import os
import cv2
import numpy as np
import importlib.util
import sys
import glob
import pandas as pd
import tempfile
import concurrent.futures
from typing import Dict, Callable, Any, List, Tuple, Optional
import matplotlib.pyplot as plt

# --- Attack Function Imports ---
from attack_functions import awgn, blur, sharpening, median, resizing, jpeg_compression
from utilities import edges_mask, frequency_mask, noisy_mask, entropy_mask, saliency_mask, border_mask, show_images

# --- Configuration ---
MIN_WPSNR = 35.0
BIN_SEARCH_ITERATIONS = 6
MAX_WORKERS = os.cpu_count() # // 2  # Use half of available CPU cores

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WATERMARKED_IMAGES_PATH = os.path.join(BASE_DIR, "watermarked_groups_images")
ATTACKED_IMAGES_PATH = os.path.join(BASE_DIR, "attacked_groups_images")
ORIGINAL_IMAGES_PATH = os.path.join(BASE_DIR, "challenge_images")
TMP_ATTACK_DIR = os.path.join(BASE_DIR, "tmp_attacks")
CSV_ATTACKS_LOG_PATH = os.path.join(BASE_DIR, "attacks_log.csv")
CSV_BEST_ATTACKS_LOG_PATH = os.path.join(BASE_DIR, "best_attacks_log.csv")


# ==============================================================================
# --- MODIFIED SECTION: Attack Definitions ---
#
# We must use named 'def' functions here instead of 'lambda'.
# Lambda functions are anonymous and cannot be "pickled"
# to be sent to other processes in the ProcessPoolExecutor.
# ==============================================================================

# --- FIX: Define param_converters as top-level 'def' functions ---
def param_jpeg(x: float) -> int:
    return int(round((1 - x) * 100))

def param_blur(x: float) -> float:
    return (x + 0.15) * 3 
    
def param_awgn(x: float) -> float:
    return (x+ 0.01) * 40
    
def param_resize(x: float) -> float:
    return max(1,np.round((1 - x) * 512)) / 512
    
def param_median(x: float) -> list:
    return [[1,3], [3,1], [3,3], [3,5], [5,3], [5,5], [5,7], [7,5], [7,7]][(int(np.floor(x * 8.999)))]
    
def param_sharp(x: float) -> float:
    return (x+0.1) * 0.2

# This dictionary is now pickleable (string -> function reference)
param_converters = {
    "JPEG": param_jpeg,
    "Blur": param_blur,
    "AWGN": param_awgn,
    "Resize": param_resize,
    "Median": param_median,
    "Sharp": param_sharp,
}

# --- FIX: Define attack functions as top-level 'def' ---
def attack_jpeg(img: np.ndarray, x: float) -> np.ndarray:
    return jpeg_compression(img, quality=param_converters["JPEG"](x))

def attack_blur(img: np.ndarray, x: float) -> np.ndarray:
    return blur(img, sigma=param_converters["Blur"](x))

def attack_awgn(img: np.ndarray, x: float) -> np.ndarray:
    return awgn(img, std=param_converters["AWGN"](x))

def attack_resize(img: np.ndarray, x: float) -> np.ndarray:
    return resizing(img, scale=param_converters["Resize"](x))

def attack_median(img: np.ndarray, x: float) -> np.ndarray:
    return median(img, kernel_size=param_converters["Median"](x))

def attack_sharp(img: np.ndarray, x: float) -> np.ndarray:
    return sharpening(img, sigma=1.0, alpha=param_converters["Sharp"](x))

# This dictionary is also pickleable (string -> function reference)
attack_config: Dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "JPEG": attack_jpeg,
    "Blur": attack_blur,
    "AWGN": attack_awgn,
    "Resize": attack_resize,
    "Median": attack_median,
    "Sharp": attack_sharp,
}

# ==============================================================================
# --- END OF MODIFIED SECTION ---
# ==============================================================================


# --- Restore log_attack for GUI import ---
def log_attack(detected, path, wpsnr, attack_name, params, mask):
    with open(CSV_ATTACKS_LOG_PATH, "a") as log:
        base = os.path.splitext(os.path.basename(path))[0]
        if "_" in base:
            group, image = base.split("_", 1)
        else:
            group, image = base, base
        if log.tell() == 0:
            log.write("Watermark detected,Image,Group,WPSNR,Attack(s) with parameters\n")
        wpsnr_str = "" if wpsnr is None else f"{wpsnr:.2f}"
        if mask is not None:
            log.write(f'{detected},{image},{group},{wpsnr_str},"{attack_name}({params}),{mask}"\n')
        else:
            log.write(f'{detected},{image},{group},{wpsnr_str},"{attack_name}({params})"\n')


def attack_mask(img: np.ndarray, percentile: int = 20) -> np.ndarray:
    """Masks blocks with least probability of being attacked."""
    strengths = np.linspace(0.0, 1.0, 20)  # Use a static alpha to get consistent result

    # Collect absolute difference for each attack
    diffs = []
    for name, func in attack_config.items():
        # awgn is random so it just skews the results
        if name == "AWGN":
            continue
        for s in strengths:
            attacked = func(img, s)
            diff = np.abs(attacked.astype(np.float32) - img.astype(np.float32))
            diffs.append(diff)
    
    # Average difference across all attacks
    avg_diff = np.mean(diffs, axis=0)
    
    # Find zones with lowest change (least affected)
    threshold = np.percentile(avg_diff, percentile)
    mask = avg_diff <= threshold
    # mask = avg_diff

    # show_images(img,mask)
    
    return mask

# --- Core Logic ---

def discover_detection_functions() -> Dict[str, str]:
    """
    Automatically discover detection function *paths* from detection_*.py* files
    in the current directory and all subdirectories.
    
    Returns:
        Dict[str, str]: A dictionary mapping group_name to the file path.
    """
    detection_function_paths = {}
    found_groups = set()

    detection_files = glob.glob(os.path.join(BASE_DIR, "**/detection_*.py*"), recursive=True)

    for detection_file in detection_files:
        base_name = os.path.basename(detection_file)

        if base_name.startswith("detection_") and (base_name.endswith(".py") or base_name.endswith(".pyc")):
            if base_name.endswith(".py"):
                group_name = base_name[10:-3]
            else:
                group_name = base_name[10:].split(".")[0]
        else:
            continue

        if group_name in found_groups:
            continue
            
        detection_function_paths[group_name] = detection_file
        found_groups.add(group_name)
        print(f"[INFO] Found detection script for group: {group_name} at {detection_file}")

    return detection_function_paths


def run_attack_job(
    original_path: str,
    watermarked_path: str,
    watermarked_img: np.ndarray,
    detection_file_path: str,
    mask: np.ndarray,
    mask_name: str,
    attack_name: str,
    attack_func: Callable, # This is now a pickleable 'def' function
    group_name: str,
    image_name: str
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Performs a binary search for a single attack/mask combination.
    This function is run in a separate process.
    """
    
    # --- Dynamically load the detection function *inside the worker* ---
    try:
        module_name = f"detection_modules.{group_name}"
        spec = importlib.util.spec_from_file_location(module_name, detection_file_path)
        if spec is None or spec.loader is None:
             raise ImportError(f"Could not create spec for {detection_file_path}")
        detection_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(detection_module)
        detection_func = detection_module.detection
    except Exception as e:
        print(f"[FATAL WORKER ERROR] Could not load {detection_file_path}: {e}")
        log_entry = {
            "Watermark detected": -1,
            "Image": image_name,
            "Group": group_name,
            "WPSNR": np.nan,
            "Attack(s) with parameters": f"{attack_name}(LOAD_ERROR),{mask_name}"
        }
        return [log_entry], None 
    # --- End of dynamic loading ---

    log_entries = []
    best_result = None
    low, high = 0.0, 1.0

    for _ in range(BIN_SEARCH_ITERATIONS):
        if abs(high - low) < 1e-6:
            break
        
        mid = (low + high) / 2
        
        full_attacked_img = attack_func(watermarked_img.copy(), mid)
        attacked_img = np.where(mask, full_attacked_img, watermarked_img)
        
        # Use the globally-defined pickleable param_converters
        params = param_converters[attack_name](mid)

        fd, attack_img_path = tempfile.mkstemp(suffix=".bmp", dir=TMP_ATTACK_DIR)
        os.close(fd) 
        
        try:
            cv2.imwrite(attack_img_path, attacked_img)
            
            detected, wpsnr_val = detection_func(
                original_path, watermarked_path, attack_img_path
            )
        except Exception as e:
            print(f"[ERROR] Detection failed for {attack_name} on {image_name}: {e}")
            detected, wpsnr_val = -1, -1.0
        finally:
            if os.path.exists(attack_img_path):
                os.remove(attack_img_path)

        log_entry = {
            "Watermark detected": detected,
            "Image": image_name,
            "Group": group_name,
            "WPSNR": f"{wpsnr_val:.2f}" if wpsnr_val is not None and wpsnr_val >= 0 else np.nan,
            "Attack(s) with parameters": f"{attack_name}({params}),{mask_name}"
        }
        log_entries.append(log_entry)

        if not detected and wpsnr_val is not None and wpsnr_val >= MIN_WPSNR:
            best_result = {
                "wpsnr": wpsnr_val,
                "attacked_img": attacked_img.copy(), 
                "attack_name": attack_name,
                "params": params,
                "mask_name": mask_name,
            }
            high = mid 
        else:
            low = mid  

    return log_entries, best_result


def setup_directories():
    """Creates all necessary directories."""
    os.makedirs(ATTACKED_IMAGES_PATH, exist_ok=True)
    os.makedirs(WATERMARKED_IMAGES_PATH, exist_ok=True)
    os.makedirs(ORIGINAL_IMAGES_PATH, exist_ok=True)
    os.makedirs(TMP_ATTACK_DIR, exist_ok=True)


def clear_tmp():
    """Removes any remaining files in tmp_attacks (cleanup)."""
    for tmp_file in glob.glob(os.path.join(TMP_ATTACK_DIR, "*")):
        try:
            os.remove(tmp_file)
        except Exception as e:
            print(f"[WARNING] Could not remove temp file {tmp_file}: {e}")


def full_attack(detection_function_paths: Dict[str, str]):
    """Starts the parallel attack suite."""
    print("\n" + "=" * 50)
    print("CrispyMcMark Parallel Attack Suite")
    print(f"Running with up to {MAX_WORKERS} parallel processes.")
    print("=" * 50 + "\n")

    if not detection_function_paths:
        print("[ERROR] No detection scripts found. Exiting.")
        return

    all_log_entries = []
    best_for_each = []
    if os.path.exists(CSV_ATTACKS_LOG_PATH):
        try:
            df_existing = pd.read_csv(CSV_ATTACKS_LOG_PATH)
            all_log_entries = df_existing.to_dict('records')
            print(f"[INFO] Loaded {len(all_log_entries)} existing log entries.")
        except Exception as e:
            print(f"[WARNING] Could not read existing log file: {e}. Starting new log.")

    img_list = sorted(os.listdir(WATERMARKED_IMAGES_PATH))
    print(f"[INFO] Found {len(img_list)} images to attack.")


    for filename in img_list:
        watermarked_path = os.path.join(WATERMARKED_IMAGES_PATH, filename)
        name_without_ext = os.path.splitext(filename)[0]

        if "_" not in name_without_ext:
            print(f"[WARNING] Skipping {filename}: Must match 'groupName_imageName.bmp'")
            continue

        group_name, image_name = name_without_ext.split("_", 1)
        
        if group_name not in detection_function_paths:
            print(f"[ERROR] No detection script for group: {group_name}. Skipping.")
            continue
            
        detection_file_path = detection_function_paths[group_name]

        original_path = os.path.join(ORIGINAL_IMAGES_PATH, image_name + ".bmp")
        if not os.path.exists(original_path):
            print(f"[ERROR] Original image not found: {original_path}. Skipping.")
            continue

        print("\n" + "-" * 50)
        print(f"Attacking image: {image_name} (Group: {group_name})")

        original_img = cv2.imread(original_path, 0)
        watermarked_img = cv2.imread(watermarked_path, 0)

        # 1. Initial Check
        try:
            module_name = f"detection_modules.{group_name}_main"
            spec = importlib.util.spec_from_file_location(module_name, detection_file_path)
            if spec is None or spec.loader is None:
                 raise ImportError(f"Could not create spec for {detection_file_path}")
            detection_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(detection_module)
            det_fun = detection_module.detection
            
            detected, wpsnr_val = det_fun(original_path, watermarked_path, watermarked_path)
            print(f"  Initial check: Detected = {detected}, WPSNR = {wpsnr_val:6.2f} dB")
            if not detected:
                print("[WARNING] Watermark not detected on original file. Skipping attacks.")
                continue
        except Exception as e:
            print(f"[ERROR] Initial detection failed: {e}. Skipping.")
            continue
            
        print("[INFO] Generating masks...")
        masks_to_try = {
            "no_mask": np.ones_like(original_img, dtype=bool),
            "edges_mask": edges_mask(original_img),
            "noisy_mask": noisy_mask(original_img),
            "entropy_mask": entropy_mask(original_img),
            "attack_mask": attack_mask(original_img.copy()),
            "frequency_mask": frequency_mask(original_img),
            "saliency_mask": saliency_mask(original_img),
            "border_mask": border_mask(original_img),
        }

        # show attack masks

        # plt.figure(figsize=(10, 6))
        # for i, (name, mask) in enumerate(masks_to_try.items(), 1):
        #     plt.subplot(2, 3, i)
        #     plt.imshow(mask if mask.ndim == 2 else mask[..., 0], cmap='gray')
        #     plt.title(name)
        #     plt.axis('off')
        # plt.tight_layout()
        # plt.show()

        jobs = []
        for mask_name, mask_data in masks_to_try.items():
            for attack_name, attack_func in attack_config.items():
                jobs.append((
                    original_path, watermarked_path, watermarked_img,
                    detection_file_path, 
                    mask_data, mask_name,
                    attack_name, 
                    attack_func, # This is now a pickleable 'def' function
                    group_name, image_name
                ))

        print(f"[INFO] Starting {len(jobs)} attack jobs in parallel...")
        
        image_best_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_job = {executor.submit(run_attack_job, *job): job for job in jobs}
            
            for future in concurrent.futures.as_completed(future_to_job):
                job_params = future_to_job[future]
                attack_name = job_params[6] 
                mask_name = job_params[5]
                try:
                    logs, best_res = future.result()
                    all_log_entries.extend(logs)
                    if best_res:
                        image_best_results.append(best_res)
                        print(f"  ✓ Success: {attack_name} ({mask_name})")
                    else:
                        print(f"  ✗ Failed:  {attack_name} ({mask_name})")
                except Exception as e:
                    print(f"[ERROR] Job {attack_name}/{mask_name} failed: {e}")


        if image_best_results:
            overall_best = max(image_best_results, key=lambda r: r["wpsnr"])

            wpsnr = overall_best["wpsnr"]
            atk_name = overall_best["attack_name"]
            mask_name = overall_best["mask_name"]
            img_data = overall_best["attacked_img"]

            best_entry = np.array([
                image_name,
                group_name,
                f"{wpsnr}",
                f"{atk_name}({overall_best['params']}),{mask_name}"
            ])
            best_for_each.append(best_entry)
            
            final_filename = f"crispymcmark_{group_name}_{image_name}.bmp"
            final_path = os.path.join(ATTACKED_IMAGES_PATH, final_filename)
            cv2.imwrite(final_path, img_data)
            
            print(f"\n[SUCCESS] Best attack for {image_name}:")
            print(f"  Attack: {atk_name} ({mask_name})")
            print(f"  WPSNR:  {wpsnr:.2f} dB")
            print(f"  Saved to: {final_path}")
        else:
            print(f"\n[FAILURE] No attack could remove the watermark for {image_name}.")

        print("-" * 50 + "\n")

    if all_log_entries:
        print(f"[INFO] Writing {len(all_log_entries)} total log entries to {CSV_ATTACKS_LOG_PATH}...")
        df_logs = pd.DataFrame(all_log_entries)
        df_logs = df_logs.drop_duplicates(subset=["Image", "Group", "Attack(s) with parameters"], keep='last')
        df_logs.to_csv(CSV_ATTACKS_LOG_PATH, index=False)

    if best_for_each:
        print(f"[INFO] Writing only best attacks to {CSV_BEST_ATTACKS_LOG_PATH}...")
        np.savetxt(CSV_BEST_ATTACKS_LOG_PATH, best_for_each, fmt='%s', delimiter=',')
        # df_logs = df_logs.drop_duplicates(subset=["Image", "Group", "Attack(s) with parameters"], keep='last')
        # df_logs.to_csv(CSV_BEST_ATTACKS_LOG_PATH, index=False)

    
    print("[INFO] Attack suite finished.")


def main():
    setup_directories()
    clear_tmp() 
    
    detection_function_paths = discover_detection_functions()
    if not detection_function_paths:
        print("[ERROR] No detection modules found. Please create detection_*.py files.")
        return

    full_attack(detection_function_paths)


if __name__ == "__main__":
    main()
