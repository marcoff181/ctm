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
ALPHA = 0.5
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


# TODO: instead of import copy function here
from attack import attack_config

def attack_strength_map(original_image):
    strength_map = np.zeros((512, 512), dtype=np.uint64)

    steps= 10
    attack_range =np.linspace(0.0,1.0,steps) 
    n_of_attacks = len(attack_config) * steps

    # evenly sample attacks and find out where they affect the original image the most
    for name,attack in attack_config.items():
        for x in attack_range:
            attacked = attack(original_image.copy(),x)
            diff = attacked - original_image 
            strength_map +=  diff
            # cv2.imwrite(f"./attack_diffs/embedding_attack_tests_{name}_{x}.bmp",diff)

    #divide by n_of_attacks to get back to the uint8 scale
    strength_map = np.astype(strength_map/n_of_attacks,np.uint8)
    cv2.imwrite(f"./attack_diffs/embedding_attack_tests_sum.bmp",strength_map)

    # TODO: decide if the output format is correct
    return strength_map

def IsTooBrightorTooDark(block):
    mean_val = np.mean(block)
    return DARKNESS_THRESHOLD < mean_val < BRIGHTNESS_THRESHOLD

def select_best_blocks(original_image, strength_map, n_blocks,  block_size):
    """
    Select best blocks based on edge content.
    
    Args:
        image: Input grayscale image
        attacked image: the blanked image attacked
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
                    'attack_value': np.average(strength_map[i:i + block_size, j:j + block_size])
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
        strength_map[tmp['locations'][0]:tmp['locations'][0] + BLOCK_SIZE,
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
    strength_map = attack_strength_map(image)

    # Select best blocks
    selected_blocks = select_best_blocks(image, strength_map, BLOCKS_TO_EMBED, BLOCK_SIZE)

    n_blocks_in_image = image.shape[0] / BLOCK_SIZE # 512 / 4 = 128
    shape_LL_tmp = np.uint8(np.floor(image.shape[0] / (2*n_blocks_in_image))) # 512 / 256 = 2 
    
    # Prepare output images
    watermarked_image = image.astype(np.float64)
    binary_mask = np.zeros_like(image, dtype=np.float64)

    # Embed watermark in selected blocks
    # print(f"Location in embedding:")
    for idx, block_info in enumerate(selected_blocks):
        block_x, block_y = block_info['locations']
        # print(f"x:{block_x}, y:{block_y}")
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
            LL_new = U.dot(np.diag(S_new)).dot(V)
            coeffs[0] = LL_new
            block_watermarked = pywt.waverec2(coeffs, wavelet='haar')

            # Place watermarked block and update mask
            watermarked_image[block_x:block_x + BLOCK_SIZE, block_y:block_y + BLOCK_SIZE] = block_watermarked
            binary_mask[block_x:block_x + BLOCK_SIZE, block_y:block_y + BLOCK_SIZE] = 1

    # Finalize watermarked image
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
    # difference = (-watermarked_image + image) * binary_mask.astype(np.uint8)
    # watermarked_image = image + difference
    # watermarked_image += binary_mask.astype(np.uint8)

    end = time.time()
    w = wpsnr(image, watermarked_image)
    print("[EMBEDDING] wPSNR: %.2fdB" % w)
    print(f"[EMBEDDING] Time: {end - start:.2f}s")
    print(f"[EMBEDDING] Embedded in {len(selected_blocks)} blocks of size {BLOCK_SIZE}x{BLOCK_SIZE}")

    return watermarked_image, watermark, Uwm, Vwm
