import numpy as np
import cv2
import pywt
import time
import os
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
from skimage.transform import rescale
from PIL import Image

from wpsnr import wpsnr


# embedded paramters:
ALPHA = 5.0
BLOCKS_TO_EMBED = 32
BLOCK_SIZE = 4
# BRIGHTNESS_WEIGHT = 0.33 # 0: no spatial domain, 1: only spatial domain
# ATTACK_WEIGHT = 1.0 - BRIGHTNESS_WEIGHT
BRIGHTNESS_THRESHOLD = 230
DARKNESS_THRESHOLD = 10

def get_watermark_svd(watermark_path):
    """Load watermark and compute its SVD decomposition."""
    watermark = np.load(watermark_path)
    watermark_matrix = watermark.reshape(32, 32)
    U, S, V = np.linalg.svd(watermark_matrix, full_matrices=False)
    return watermark, U, S, V


# TODO: instead of import copy function here
from attack import attack_config

def attack_strength_map(original_image):
    """evenly sample attacks and find out where they affect the original image the most"""
    strength_map = np.zeros((512, 512), dtype=np.uint64)

    steps= 10
    attack_range =np.linspace(0.0,1.0,steps) 
    n_of_attacks = len(attack_config) * steps

 
    for attack in attack_config.values():
        for x in attack_range:
            attacked = attack(original_image.copy(),x)
            diff = attacked - original_image 
            strength_map +=  diff

    #divide by n_of_attacks to get back to the uint8 scale
    strength_map = np.astype(strength_map/n_of_attacks,np.uint8)
    cv2.imwrite(f"./attack_diffs/embedding_attack_tests_sum.bmp",strength_map)

    return strength_map

def select_best_blocks(original_image, strength_map, n_blocks,  block_size):
    """Select best blocks based on how much they are attacked by using `strength_map`"""

    blocks = []

    for i in range(0, original_image.shape[0], block_size):
        for j in range(0, original_image.shape[1], block_size):
            block = original_image[i:i + block_size, j:j + block_size]
            avg_brightness = np.average(block)
            # if DARKNESS_THRESHOLD < avg_brightness < BRIGHTNESS_THRESHOLD:
            block_tmp = {
                'locations': (i,j),
                # 'avg_brightness': avg_brightness,
                'attack_value': np.average(strength_map[i:i + block_size, j:j + block_size])
            }
            blocks.append(block_tmp)

    # select first 32 blocks with lower attack strength
    blocks = sorted(blocks, key=lambda k: k['attack_value'])[:n_blocks]
        
    # order blocks based on their location, so they can be retrieved in a deterministic order
    return sorted(blocks, key=lambda k: k['locations'])

def embedding(original_image_path, watermark_path, alpha, dwt_level):
    """Embed watermark using DWT-SVD with block selection."""
    
    # Load image and watermark
    image = cv2.imread(original_image_path, 0)
    watermark, Uwm, Swm, Vwm = get_watermark_svd(watermark_path)

    # Swm = ((Swm - 0.1) / (15 - 0.1)) + 1 # Normalization step
    # print(Swm)
    # print(Swm*100)

    start = time.time()

    # Compute attack resistance map
    strength_map = attack_strength_map(image)

    # Select best blocks
    selected_blocks = select_best_blocks(image, strength_map, BLOCKS_TO_EMBED, BLOCK_SIZE)

    n_blocks_in_image = image.shape[0] / BLOCK_SIZE # 512 / 4 = 128
    shape_LL_tmp = np.uint8(np.floor(image.shape[0] / (2*n_blocks_in_image))) # 512 / 256 = 2 
    
    # Prepare output images
    watermarked_image = image.copy()
    # binary_mask = np.zeros_like(image, dtype=np.float64)

    # Embed watermark in selected blocks
    # print(f"Location in embedding:")
    for idx, block_info in enumerate(selected_blocks):
        x, y = block_info['locations']
        # print(f"x:{x}, y:{y}")
        block = image[x:x + BLOCK_SIZE, y:y + BLOCK_SIZE]

        # DWT and SVD
        coeffs = pywt.wavedec2(block, wavelet='haar', level=1)
        LL = coeffs[0]
        U, S, V = np.linalg.svd(LL)
        S_new = S.copy()

        # Watermark singular values for this block
        # wm_start = idx * shape_LL_tmp
        # wm_end = wm_start + shape_LL_tmp
        # print(f"wm_start: {wm_start}, wm_end: {wm_end}, len(Swm): {len(Swm)}")

        # if wm_end <= len(Swm):
        #TODO: Need to change the logic if we want more than 32 blocks
        S_new[0] += Swm[idx] * alpha
        LL_new = U.dot(np.diag(S_new)).dot(V)
        coeffs[0] = LL_new
        block_watermarked = pywt.waverec2(coeffs, wavelet='haar')

        # Place watermarked block and67.18 update mask
        watermarked_image[x:x + BLOCK_SIZE, y:y + BLOCK_SIZE] = block_watermarked
            # binary_mask[x:x + BLOCK_SIZE, y:y + BLOCK_SIZE] = 1

    # Finalize watermarked image
    # watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
    
    # watermarked_image = np.uint8(watermarked_image)

    # difference = (-watermarked_image + image) * binary_mask.astype(np.uint8)
    # watermarked_image = image + difference
    # watermarked_image += binary_mask.astype(np.uint8)

    end = time.time()
    # w = wpsnr(image, watermarked_image)
    # print("[EMBEDDING] wPSNR: %.2fdB" % w)
    print(f"Time to embed: {end - start:.2f}s")
    # print(f"[EMBEDDING] Embedded in {len(selected_blocks)} blocks of size {BLOCK_SIZE}x{BLOCK_SIZE}")

    return watermarked_image, watermark, Uwm, Vwm
