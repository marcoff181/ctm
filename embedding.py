import numpy as np
import cv2
import pywt
import time
import os
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt

from wpsnr import wpsnr


# embedded paramters:
ALPHA = 5.11
BLOCKS_TO_EMBED = 32
BLOCK_SIZE = 4
SPATIAL_WEIGHT = 0.33 # 0: no spatial domain, 1: only spatial domain
ATTACK_WEIGHT = 1.0 - SPATIAL_WEIGHT
BRIGHTNESS_THRESHOLD = 230
DARKNESS_THRESHOLD = 10

def get_watermark_svd(watermark_path):
    """Load watermark and compute its SVD decomposition."""
    watermark = np.load(watermark_path)
    watermark_matrix = watermark.reshape(32, 32)
    U, S, V = np.linalg.svd(watermark_matrix, full_matrices=False)
    return watermark, U, S, V

#------- DUE TO SLEF CONTAINED NATURE ------------------
def jpeg_compression(img, QF):
    cv2.imwrite('tmp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])
    attacked = cv2.imread('tmp.jpg', 0)
    os.remove('tmp.jpg')
    return attacked

def blur(img, sigma):
    attacked = gaussian_filter(img, sigma)
    return attacked

def awgn(img, std, seed):
    mean = 0.0
    # np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked

def sharpening(img, sigma, alpha):
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    return attacked

def median(img, kernel_size):
    attacked = medfilt(img, kernel_size)
    return attacked

def resizing(img, scale):
  from skimage.transform import rescale
  x, y = img.shape
  attacked = rescale(img, scale)
  attacked = rescale(attacked, 1/scale)
  attacked = attacked[:x, :y]
  return attacked

#------- DUE TO SLEF CONTAINED NATURE ------------------

# TODO: instead of import copy function here
from attack import bin_search_attack

def attack_image(original_image):
    """
    Intelligent attack phase: uses binary search to find the most effective attack parameters.
    For each attack, finds the parameter that maximizes the difference while keeping the image valid.
    Resizes attacked images back to original shape if needed.
    """
    blank_image = np.zeros((512, 512), dtype=np.uint8)

    start = time.time()
    # these are the attacks that manage to get the wpsnr as close as 35 though, might want to change that
    best_attacks = bin_search_attack(blank_image,blank_image,lambda a,b,c : (1,wpsnr(b,c)), np.ones((512, 512), dtype=np.uint8),4)
    for attacked in best_attacks:
        blank_image += np.abs(attacked.astype(np.float64) - original_image)

    end = time.time()
    print(f"[EMBEDDING] Intelligent Attack Phase (binary search) duration: {end - start:.2f}s")

    return blank_image

def IsTooBrightorTooDark(block):
    mean_val = np.mean(block)
    return DARKNESS_THRESHOLD < mean_val < BRIGHTNESS_THRESHOLD

def select_best_blocks(original_image, attacked_image, n_blocks,  block_size):
    """
    Select best blocks based on edge content.
    
    Args:
        image: Input grayscale image
        n_blocks: Number of blocks to select
        block_size: Size of each block (4x4 for better LL subband size)
    
    Returns:
        list: Locations (x, y) of selected blocks sorted by edge content
    """

    selected_blocks_tmp = []

    for i in range(0, original_image.shape[0], block_size):
        for j in range(0, original_image.shape[1], block_size):
            block = original_image[i:i + block_size, j:j + block_size]
            if IsTooBrightorTooDark(block):
                # choosen to average over all of the values inside the block
                spatial_value = np.average(block)

                block_tmp = {
                    'locations': (i,j),
                    'spatial_value': spatial_value,
                    'attack_value': np.average(attacked_image[i:i + block_size, j:j + block_size])
                }
                selected_blocks_tmp.append(block_tmp)
    
    # 1. Sort all of the blocks based on the spatial value (average on brightness)
    selected_blocks_tmp = sorted(selected_blocks_tmp, key=lambda k: k['spatial_value'], reverse=True)
    # Normalize each block and score it based on the brightness value (the more the better)
    for i in range(len(selected_blocks_tmp)):
        selected_blocks_tmp[i]['merit'] = i*SPATIAL_WEIGHT
    
    # 2. We next want to sort them based on how much they where affected by the attacks, choosing the less affected
    selected_blocks_tmp = sorted(selected_blocks_tmp, key=lambda k: k['attack_value'], reverse=False)
    # we value more the one attacked less normalizing the result
    for i in range(len(selected_blocks_tmp)):
        selected_blocks_tmp[i]['merit'] += i*ATTACK_WEIGHT
    
    # 3. In the end we select the blocks with the highest merit
    selected_blocks_tmp = sorted(selected_blocks_tmp, key=lambda k: k['merit'], reverse=True)

    selected_blocks = []
    for i in range(n_blocks):
        tmp = selected_blocks_tmp.pop()
        selected_blocks.append(tmp)
        attacked_image[tmp['locations'][0]:tmp['locations'][0] + BLOCK_SIZE,
                        tmp['locations'][1]:tmp['locations'][1] + BLOCK_SIZE] = 1
        
    selected_blocks = sorted(selected_blocks, key=lambda k: k['locations'], reverse=False)

    return selected_blocks


def embedding(original_image_path, watermark_path, alpha, dwt_level):
    """Embed watermark using DWT-SVD with block selection."""
    
    # Load image and watermark
    image = cv2.imread(original_image_path, 0)
    watermark, Uwm, Swm, Vwm = get_watermark_svd(watermark_path)
    
    start = time.time()

    # Compute attack resistance map
    attacked_image = attack_image(image)

    # Select best blocks
    selected_blocks = select_best_blocks(image, attacked_image, BLOCKS_TO_EMBED, BLOCK_SIZE)

    n_blocks_in_image = image.shape[0] / BLOCK_SIZE
    shape_LL_tmp = np.uint8(np.floor(image.shape[0] / (2*n_blocks_in_image))) 
    
    # Prepare output images
    watermarked_image = image.astype(np.float64)
    binary_mask = np.zeros_like(image, dtype=np.float64)

    # Embed watermark in selected blocks
    for idx, block_info in enumerate(selected_blocks):
        block_x, block_y = block_info['locations']
        block = image[block_x:block_x + BLOCK_SIZE, block_y:block_y + BLOCK_SIZE]

        # DWT and SVD
        coeffs = pywt.wavedec2(block, wavelet='haar', level=1)
        LL = coeffs[0]
        U, S, V = np.linalg.svd(LL)
        S_new = S.copy()

        # Watermark singular values for this block
        wm_start = idx * shape_LL_tmp
        wm_end = wm_start + shape_LL_tmp

        if wm_end <= len(Swm):
            S_new += Swm[wm_start:wm_end] * alpha
            LL_new = U @ np.diag(S_new) @ V
            coeffs[0] = LL_new
            block_watermarked = pywt.waverec2(coeffs, wavelet='haar')

            # Place watermarked block and update mask
            watermarked_image[block_x:block_x + BLOCK_SIZE, block_y:block_y + BLOCK_SIZE] = block_watermarked
            binary_mask[block_x:block_x + BLOCK_SIZE, block_y:block_y + BLOCK_SIZE] = 1

    # Finalize watermarked image
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
    difference = (-watermarked_image + image) * binary_mask.astype(np.uint8)
    watermarked_image = image + difference
    watermarked_image += binary_mask.astype(np.uint8)

    end = time.time()
    w = wpsnr(image, watermarked_image)
    print("[EMBEDDING] wPSNR: %.2fdB" % w)
    print(f"[EMBEDDING] Time: {end - start:.2f}s")
    print(f"[EMBEDDING] Embedded in {len(selected_blocks)} blocks of size {BLOCK_SIZE}x{BLOCK_SIZE}")

    return watermarked_image, watermark, Uwm, Vwm
