# CrispyMcMark Watermarking & Attack Suite

This is the repository related to the *Capture the Mark* competion held in the University of Trento for the Multimedia Data Security course, for the academic year 2025/2026.

Group name: **crispymcmark**

## Overview

This repository provides a complete framework for robust image watermarking, attack simulation, and detection. It includes:

- **Watermark embedding and detection algorithms** (DWT-SVD, LSB)
- **Attack simulation** (AWGN, blur, sharpening, resizing, JPEG compression, median filtering)
- **Visualization tools** for embedding/detection logic and attack effects
- **ROC analysis** and quality metrics (WPSNR, BER)
- **GUI** for interactive attack and detection experiments

## Directory Structure

```
attack_functions.py         # Attack implementations
attack.py                  # Attack orchestration and logging
crispy_embedder.py         # Main embedder for DWT-SVD watermarking
crispymcmark.npy           # Default watermark
detection_crispymcmark.py  # Detection logic for DWT-SVD watermark
embedding.py               # Embedding logic for DWT-SVD watermark
example_python.py          # Example usage
gui_roba.py                # Tkinter GUI for attacks/detection
plot_attacks.py            # Attack parameter visualization
roc_crispymcmark.py        # ROC curve computation
utilities.py               # Helper functions (masks, metrics, visualization)
visualize_embedding.py     # Embedding/detection visualization
wpsnr.py                   # WPSNR metric
lsb/                       # LSB watermarking (embedding/detection)
challenge_images/          # Original images for watermarking
watermarked_groups_images/ # Watermarked images
attacked_groups_images/    # Attacked images
tmp_attacks/               # Temporary attack results
attack_results/            # Attack logs and results
```

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/crispymcmark.git
   cd crispymcmark
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

   Main dependencies:
   - numpy
   - scipy
   - matplotlib
   - scikit-image
   - opencv-python
   - Pillow
   - scikit-learn

## Usage

### 1. Watermark Embedding

Embed a watermark into an image using DWT-SVD:

```sh
python crispy_embedder.py 0005.bmp
```
- `0005.bmp` is the image filename in `challenge_images/`.

### 2. Watermark Detection

Detect watermark in an attacked image:

```python
from detection_crispymcmark import detection

detected, wpsnr_val = detection(
    "./challenge_images/0005.bmp",
    "./watermarked_groups_images/crispymcmark_0005.bmp",
    "./attacked_groups_images/crispymcmark_crispymcmark_0005.bmp"
)
print(f"Detected: {detected}, WPSNR: {wpsnr_val:.2f} dB")
```

### 3. Simulate Attacks

Run attacks and log results:

```sh
python attack.py
```

### 4. Visualize Embedding & Detection

Generate visualizations:

```sh
python visualize_embedding.py <original_image_path> <watermarked_image_path> 
```

### 5. ROC Analysis

Compute ROC curve for watermark detection:

```sh
python roc_crispymcmark.py
```

### 6. GUI

Launch the GUI for interactive attack/detection:

```sh
python gui_roba.py
```

## Strategy used

We started exploring the current SOTA regarding digital watermarking techniques for binary watermarks. We noticed that in the literature the majority of scientific papers was regarding the **Discrete Wavelet Transform**, with the addition of the **Singular Value Decomposition**, which is a method for matrix factorization. While developing the current solution we moved away from the implementation described in the papers and focused on a simpler implementation that was more tailored for the challenge rules.

### Embedding

This watermarking process first prepares the watermark by converting it into a compact "fingerprint", by reshaping the watermark to a  $32 \times 32$ image. We then performs a Singular Value Decomposition (SVD), and extracts just its singular values. This 32-value vector is the data that will be hidden.

Next, the algorithm intelligently selects the best places in the host image to embed this data. We scan the image to find the top 13 "best" $8 \times 8$ blocks. The "best" blocks are determined by favoring **textured, complex blocks** (which have high entropy) and avoiding simple, flat, or high-contrast blocks where any changes would be visually obvious. These 13 chosen blocks are then sorted by their location to ensure they are always processed in the same deterministic order.

Finally, the embedding function loops through these 13 selected blocks to insert the watermark. For each block, it applies a Discrete Wavelet Transform (DWT) to isolate the $LL$ sub-band, which is the most robust, low-frequency approximation of the block. It then performs an SVD on this $LL$ sub-band to get *its* singular values. The **core embedding** happens here: the block's largest singular value (The first) is modified by adding the value from the watermark's fingerprint, scaled by the $\alpha$ strength. The block is then rebuilt using an inverse SVD and inverse DWT, and this newly watermarked block replaces the original in the image.

![Embedding](./presentation/embedding.png)

### Detection (Non-Blind technique)

The detection itself begins by locating the 13 hidden data blocks. This is done simply by comparing the original image with the watermarked image; the blocks that show a difference are the ones that carry the data. With these locations identified, the core data retrieval begins. For each block, the algorithm mathematically reverses the embedding process: it decomposes both the original block and the corresponding attacked block into their robust, low-frequency $LL$ components using a DWT. After applying SVD to both, it isolates the embedded data by calculating the **difference** between the largest singular value of the attacked block and that of the original block. This difference, when scaled down, reveals the single piece of watermark data hidden in that block.

This process yields a 32-value "fingerprint" (the singular values). To reconstruct the full watermark image, this fingerprint is mathematically combined with two hardcoded matrices (`Uwm` and `Vwm`), which represent the original watermark's structural SVD components. The resulting matrix is then flattened and binarized (converted to 0s and 1s) to create the final digital signature.

Finally, the detection is a two-part test. First, the retrieved watermark signature is compared to the "clean" reference watermark signature to get a bit-matching **similarity score**. Second, the perceptual **image quality** of the attacked image is measured using a **Weighted PSNR (WPSNR)**. The watermark is only considered "detected" if the similarity score is high (above 0.70, computed using the **ROC**).

![Detection](./presentation/extraction.png)

### Attack

The core of the attack is a **binary search**. For each of the six attack types (like JPEG, Blur, or Noise), the script doesn't just apply one strong, fixed attack. Instead, it intelligently searches for the "sweet spot": the **strongest attack parameter** (e.g., the lowest JPEG quality) that causes the detection function to *just* fail, while simultaneously ensuring the image's perceptual quality (measured by **WPSNR**) remains above a minimum threshold. This process is repeated with **masks**, applying the attack only to specific regions (like edges or noisy areas) where the watermark is likely hidden and the visual changes are less noticeable.

![attack-plot](./presentation/attack_plot_new.png)

### ROC calculation

The computation of the ROC curve differs from the implementation explained in the challenge rules by adding new null hypothesis ($H_{0}$) which are: 
  
- The extraction of the watermark from an attacked original (unwatermarked) image.
- The extraction of the watermark from destroyed images ($wPSNR <= 25.0dB$)

We felt the need to add this step as it would create a AUC of 1.0 without it and it helps make the ROC curve both, more representative of real world performance and also comply to the challenge rules.

![ROC](roc_crispymcmark.png)

## Example Workflow

Before embedding a watermark, ensure that your images are present in the `challenge_images` folder.  
You can copy all (or one) images from the `images` directory using:

```sh
cp images/*.bmp challenge_images/
```

```sh
cp images/0005.bmp challenge_images/
```

1. **Embed watermark:**  
   `python crispy_embedder.py 0005.bmp`

2. **Apply attack:**  
   `python attack.py`

3. **Detect watermark:**  
   `python detection_crispymcmark.py`

4. **Visualize results:**  
   `python visualize_embedding.py challenge_images/0005.bmp watermarked_groups_images/crispymcmark_0005.bmp`

## Example Python Usage

You can also use the main watermarking, attack, and detection functions directly from Python scripts, without calling them from the command line.

### Embedding a Watermark

```python
from embedding import embedding

# Embed watermark into an image
watermarked = embedding(
    "./challenge_images/0005.bmp",      # Path to original image
    "crispymcmark.npy"                  # Path to watermark file
)

# Save the watermarked image
import cv2
cv2.imwrite("./watermarked_groups_images/crispymcmark_0005.bmp", watermarked)
```

### Detecting a Watermark

```python
from detection_crispymcmark import detection

# Run detection on an attacked image
detected, wpsnr_val = detection(
    "./challenge_images/image_name.bmp",                      # Original image
    "./watermarked_groups_images/image_name_watermarked.bmp", # Watermarked image
    "./attacked_groups_images/image_name_attacked.bmp"        # Attacked image
)
print(f"Detected: {detected}, WPSNR: {wpsnr_val:.2f} dB")
```

### Applying an Attack

```python
from attack import attack_config, param_converters

import cv2
img = cv2.imread("./watermarked_groups_images/crispymcmark_tree.bmp", 0)

# Choose attack by name
attack_name = "JPEG"  # or any key from attack_config
strength = 0.9        # value between 0.0 and 1.0

attack_func = attack_config[attack_name]
params = param_converters[attack_name](strength)
attacked = attack_func(img, strength)

cv2.imwrite(f"./attacked_groups_images/crispymcmark_crispymcmark_tree_{attack_name}_{params}.bmp", attacked)
```

## License

MIT License