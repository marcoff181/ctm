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
ALPHA = 10.0
N_BLOCKS =  16
BLOCK_SIZE = 16


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

def select_best_blocks(original_image, strength_map):
    """Select best blocks based on how much they are attacked by using `strength_map`"""

    blocks = []

    for i in range(0, original_image.shape[0], BLOCK_SIZE):
        for j in range(0, original_image.shape[1], BLOCK_SIZE):
            blocks.append({
                'locations': (i,j),
                'attack_strength': np.average(strength_map[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE])
            })

    # select first x blocks with lowest attack strength
    best_blocks = sorted(blocks, key=lambda k: k['attack_strength'])[:N_BLOCKS]
    block_positions = [block['locations'] for block in best_blocks]
        
    # order blocks based on their location, so they can be retrieved in a deterministic order
    return sorted(block_positions)

def embedding(image_path, watermark_path):
    """Embed watermark using DWT-SVD with block selection."""
    
    image = cv2.imread(image_path, 0)
    Swm = get_watermark_S(watermark_path)

    strength_map = attack_strength_map(image)

    blocks = select_best_blocks(image, strength_map)

    for idx, (x,y) in enumerate(blocks):
        block_location = (slice(x, x + BLOCK_SIZE), slice(y, y + BLOCK_SIZE))

        block = image[block_location]

        # DWT 
        coeffs = pywt.wavedec2(block, wavelet='haar', level=1)
        LLb = coeffs[0]
        # SVD
        Ub, Sb, Vb = np.linalg.svd(LLb)

        Sb[0] += Swm[idx] * ALPHA

        # iSVD
        LLnew = Ub.dot(np.diag(Sb)).dot(Vb)
        # iDWT
        coeffs[0] = LLnew
        block_watermarked = pywt.waverec2(coeffs, wavelet='haar')
        
        image[block_location] = block_watermarked

    return image
