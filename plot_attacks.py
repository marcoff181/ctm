import os
import cv2
import numpy as np
from detection_crispymcmark import detection
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from wpsnr import wpsnr

import pandas as pd
from attack_functions import awgn, blur, sharpening, median, resizing, jpeg_compression
from utilities import edges_mask, noisy_mask

from attack import attack_config, param_converters

attacked_wpsnr_lower_bound = 35


def study_attack_parameter_ranges(
    images_dir="./images/", num_images=1, alpha_steps=50
):
    print("--- Starting Attack Parameter Range Study ---")

    # Validate the images directory and load image paths
    if not os.path.isdir(images_dir):
        print(f"Error: The directory '{images_dir}' was not found.")
        return

    image_files = sorted(
        [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith(".bmp")
        ]
    )

    if not image_files:
        print(f"Error: No BMP images found in '{images_dir}'.")
        return

    # Load the specified number of images in grayscale
    image_paths = image_files[:num_images]
    images = [cv2.imread(p, 0) for p in image_paths]
    images = [img for img in images if img is not None]  # Filter out any failed loads

    if not images:
        print(
            "Error: Could not load any images. Please check the image paths and files."
        )
        return

    print(f"Successfully loaded {len(images)} images for the study.")

    # we of course start from 1/512 to help our brother resize
    alpha_values = np.linspace(0.0, 1.0, alpha_steps)
    results = {}

    for attack_name, attack_func in attack_config.items():
        print(f"  Analyzing attack: {attack_name}...")
        attack_wpsnrs = []
        for alpha in alpha_values:
            print(round(alpha * alpha_steps), end="\r")
            current_alpha_wpsnrs = []
            for img in images:
                # Apply the attack and calculate WPSNR against the original
                attacked_img = attack_func(img.copy(), alpha)
                current_alpha_wpsnrs.append(min(100, wpsnr(img, attacked_img)))

            # Average the WPSNR across all images for the current alpha
            if num_images == 1:
                avg_wpsnr = current_alpha_wpsnrs[0]
            else:
                avg_wpsnr = np.mean(current_alpha_wpsnrs)
            attack_wpsnrs.append(avg_wpsnr)

        results[attack_name] = attack_wpsnrs

    # --- Plotting Results ---
    num_attacks = len(attack_config)
    ncols = 3
    nrows = (num_attacks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, nrows * 5))
    axes = axes.flatten()

    for i, (attack_name, wpsnr_values) in enumerate(results.items()):
        ax = axes[i]
        ax.plot(
            alpha_values, wpsnr_values, marker=".", linestyle="-", label="Avg. WPSNR"
        )

        # Plot the critical threshold line
        ax.axhline(
            y=attacked_wpsnr_lower_bound,
            color="r",
            linestyle="--",
            label=f"Destroyed Threshold ({attacked_wpsnr_lower_bound} dB)",
        )

        ax.set_title(f"Attack: {attack_name}", fontsize=12)
        ax.set_xlabel("Alpha (Attack Strength)")
        ax.set_ylabel("WPSNR (dB)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        # Set sensible y-axis limits
        min_wpsnr_val = min(wpsnr_values) if wpsnr_values else 20
        max_wpsnr_val = max(wpsnr_values) if wpsnr_values else 60
        ax.set_ylim(bottom=min(25, min_wpsnr_val - 5), top=max_wpsnr_val + 5)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("WPSNR Degradation vs. Attack Strength (Alpha)", fontsize=16, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("attack_parameter_study_results.png", dpi=300)
    plt.show()

    print("\n--- Study Complete ---")
    print("Plot saved to 'attack_parameter_study_results.png'.")


def visualize_attack_combination(original, watermarked, detection, grid_size=20):
    attack1_name, attack2_name = list(attack_config.keys())[:2]
    attack1_func = attack_config[attack1_name]
    attack2_func = attack_config[attack2_name]

    def get_range(name):
        return (0.0, 0.07) if "blur" in name.lower() else (0.0, 1.0)

    range1 = get_range(attack1_name)
    range2 = get_range(attack2_name)
    params1 = np.linspace(*range1, grid_size)
    params2 = np.linspace(*range2, grid_size)

    results = []
    d1, d2, u1, u2 = [], [], [], []
    dwpsnr, uwpsnr = [], []

    ct = 0
    iterations = grid_size * grid_size
    for p1 in params1:
        a1 = attack1_func(watermarked.copy(), p1)
        for p2 in params2:
            ct += 1
            if ct % 10 == 0:
                print(f"{ct}/{iterations}")
            img = attack2_func(a1.copy(), p2)
            detected, wpsnr = detection(original, watermarked, img)
            if detected:
                continue
            ap1 = param_converters[attack1_name](p1)
            ap2 = param_converters[attack2_name](p2)
            results.append(
                {
                    f"{attack1_name}_param": ap1,
                    f"{attack2_name}_param": ap2,
                    "detected": detected,
                    "WPSNR": wpsnr,
                }
            )
            u1.append(p1)
            u2.append(p2)
            uwpsnr.append(wpsnr)

    vmin, vmax = min(uwpsnr), max(uwpsnr)

    fig, ax = plt.subplots(figsize=(10, 8))
    if dwpsnr:
        ax.scatter(
            d1,
            d2,
            c=dwpsnr,
            cmap="Reds",
            s=70,
            alpha=0.7,
            marker="x",
            vmin=vmin,
            vmax=vmax,
            edgecolors="darkred",
            linewidths=1.2,
            label="Detected",
        )
    if uwpsnr:
        sc = ax.scatter(
            u1,
            u2,
            c=uwpsnr,
            cmap="RdYlGn_r",
            s=90,
            alpha=0.8,
            marker="o",
            vmin=vmin,
            vmax=vmax,
            edgecolors="black",
            linewidths=0.5,
            label="Undetected",
        )
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("WPSNR (dB)", rotation=270, labelpad=15)

    pad_x = 0.05 * (range1[1] - range1[0])
    pad_y = 0.05 * (range2[1] - range2[0])
    ax.set_xlim(range1[0] - pad_x, range1[1] + pad_x)
    ax.set_ylim(range2[0] - pad_y, range2[1] + pad_y)

    ax.set_xlabel(f"{attack1_name} param")
    ax.set_ylabel(f"{attack2_name} param")
    ax.set_title(
        f"{attack1_name} + {attack2_name} Attack Grid ({grid_size}×{grid_size})"
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"attack_combination_{attack1_name}_{attack2_name}.png", dpi=300)
    plt.show()

    return pd.DataFrame(results)


if __name__ == "__main__":
    study_attack_parameter_ranges()
