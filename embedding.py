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


# TODO: instead of import copy function here
from attack import attack_config
from wpsnr import wpsnr

# embedded parameters:
ALPHA = 5.0
BLOCKS_TO_EMBED = 32
BLOCK_SIZE = 4
# BRIGHTNESS_THRESHOLD = 230
# DARKNESS_THRESHOLD = 10

def get_watermark_S(watermark_path):
    watermark = np.load(watermark_path)
    watermark = watermark.reshape(32, 32)
    _, S, _ = np.linalg.svd(watermark, full_matrices=False)
    return S


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
            # TODO: for now seemed like it was making no difference, if we want to try it properly we need to decide on the threshold values
            # NOTE: when enabled for now, WPSNR does not even change second decimal value
            # avg_brightness = np.average(block)
            # if DARKNESS_THRESHOLD < avg_brightness < BRIGHTNESS_THRESHOLD:
            block_attack_strength = np.average(strength_map[i:i + block_size, j:j + block_size])
            blocks.append({
                'locations': (i,j),
                'attack_strength': np.average(strength_map[i:i + block_size, j:j + block_size])
            })

    # select first 32 blocks with lowest attack strength
    best_blocks = sorted(blocks, key=lambda k: k['attack_strength'])[:n_blocks]
    block_positions = [block['locations'] for block in best_blocks]
        
    # order blocks based on their location, so they can be retrieved in a deterministic order
    return sorted(block_positions)

def embedding(image_path, watermark_path, alpha, dwt_level):
    """Embed watermark using DWT-SVD with block selection."""
    
    image = cv2.imread(image_path, 0)
    Swm = get_watermark_S(watermark_path)

    # TODO: try more attempts at using normalization to remove huge first watermark
    # Swm = ((Swm - 0.1) / (15 - 0.1)) + 1 # Normalization step
    # print(Swm)
    # print(Swm*100)

    strength_map = attack_strength_map(image)

    blocks = select_best_blocks(image, strength_map, BLOCKS_TO_EMBED, BLOCK_SIZE)

    for idx, (x,y) in enumerate(blocks):
        block_location = (slice(x, x + BLOCK_SIZE), slice(y, y + BLOCK_SIZE))

        block = image[block_location]

        # DWT 
        coeffs = pywt.wavedec2(block, wavelet='haar', level=1)
        LLb = coeffs[0]
        # SVD
        Ub, Sb, Vb = np.linalg.svd(LLb)

        # Watermark singular values for this block
        # wm_start = idx * shape_LL_tmp
        # wm_end = wm_start + shape_LL_tmp
        # print(f"wm_start: {wm_start}, wm_end: {wm_end}, len(Swm): {len(Swm)}")

        #TODO: Need to change the logic if we want more than 32 blocks
        Sb[0] += Swm[idx] * ALPHA
        LLnew = Ub.dot(np.diag(Sb)).dot(Vb)
        coeffs[0] = LLnew
        block_watermarked = pywt.waverec2(coeffs, wavelet='haar')
        
        image[block_location] = block_watermarked

    return image
