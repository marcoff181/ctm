import os
import cv2
import numpy as np
from attack import attack_config
from detection_crispymcmark import detection, extraction, similarity
from wpsnr import wpsnr

WATERMARKED_DIR = "./watermarked_groups_images/"
DESTROYED_DIR = "./destroyed_images/"
CHALLENGE_DIR = "./challenge_images/"
WPSNR_THRESHOLD = 25.0

os.makedirs(DESTROYED_DIR, exist_ok=True)

def get_original_path(watermarked_name):
    # Assumes watermarked_name is like "crispymcmark_0005.bmp"
    suffix = watermarked_name.split("_", 1)[1]
    orig_path = os.path.join(CHALLENGE_DIR, suffix)
    if not os.path.exists(orig_path):
        orig_path = os.path.join(CHALLENGE_DIR, suffix.split(".")[0] + ".bmp")
    return orig_path

error_count = 0
total = 0

# Only use a subset of attacks for simplicity
selected_attacks = ["JPEG", "AWGN", "Blur"]

for fname in os.listdir(WATERMARKED_DIR):
    if not fname.endswith(".bmp"):
        continue
    watermarked_path = os.path.join(WATERMARKED_DIR, fname)
    original_path = get_original_path(fname)
    if not os.path.exists(original_path):
        print(f"Original not found for {fname}, skipping.")
        continue

    watermarked_img = cv2.imread(watermarked_path, 0)
    original_img = cv2.imread(original_path, 0)

    for attack_name in selected_attacks:
        attack_func = attack_config[attack_name]
        found = False
        for strength in np.linspace(0.0, 20.0, 20):
            attacked_img = attack_func(watermarked_img.copy(), strength)
            wpsnr_val = wpsnr(watermarked_img, attacked_img)
            if wpsnr_val <= WPSNR_THRESHOLD:
                destroyed_name = f"{fname[:-4]}_{attack_name}_{strength:.2f}_destroyed.bmp"
                destroyed_path = os.path.join(DESTROYED_DIR, destroyed_name)
                cv2.imwrite(destroyed_path, attacked_img)
                detected, wpsnr_val = detection(original_path, watermarked_path, destroyed_path)
                # Extract similarity for reporting
                original_image = cv2.imread(original_path, 0)
                watermarked_image = cv2.imread(watermarked_path, 0)
                attacked_image = cv2.imread(destroyed_path, 0)
                wm_ref, block_mask = extraction(original_image, watermarked_image, watermarked_image)
                wm_ext, _ = extraction(original_image, watermarked_image, attacked_image, block_mask)
                sim_val = similarity(wm_ref, wm_ext)
                total += 1
                if detected:
                    error_count += 1
                    print(f"[ERROR] Watermark detected in destroyed image: {destroyed_name} (WPSNR={wpsnr_val:.2f}, Similarity={sim_val:.4f})")
                found = True
                break
        if not found:
            print(f"[INFO] Could not destroy {fname} with {attack_name}")

print(f"\nSummary: {error_count} out of {total} destroyed images still detected watermark (should be 0).")