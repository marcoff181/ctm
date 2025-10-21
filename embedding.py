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


def attack_image(original_image):

    blank_image = np.float64(np.zeros((512,512)))

    start = time.time()

    blur_sigma_values = [0.1, 0.5, 1, 2, [1, 1], [2, 1]]
    for sigma in blur_sigma_values:
        attacked_image_tmp = blur(original_image, sigma)
        blank_image += np.abs(attacked_image_tmp - original_image)

    kernel_size = [3, 5, 7, 9, 11]
    for k in kernel_size:
        attacked_image_tmp = median(original_image, k)
        blank_image += np.abs(attacked_image_tmp - original_image)

    awgn_std = [0.1, 0.5, 2, 5, 10]
    for std in awgn_std:
        attacked_image_tmp = awgn(original_image, std, 0)
        blank_image += np.abs(attacked_image_tmp - original_image)

    sharpening_sigma_values = [0.1, 0.5, 2, 100]
    sharpening_alpha_values = [0.1, 0.5, 1, 2]
    for sharpening_sigma in sharpening_sigma_values:
        for sharpening_alpha in sharpening_alpha_values:
            attacked_image_tmp = sharpening(original_image, sharpening_sigma, sharpening_alpha)
            blank_image += np.abs(attacked_image_tmp - original_image)

    resizing_scale_values = [0.5, 0.75, 0.9, 1.1, 1.5]
    for scale in resizing_scale_values:
        attacked_image_tmp = cv2.resize(original_image, (0, 0), fx=scale, fy=scale)
        attacked_image_tmp = cv2.resize(attacked_image_tmp, (512, 512))
        blank_image += np.abs(attacked_image_tmp - original_image)

    end = time.time()

    print(f"[EMBEDDING] Attack Phase duration: {str(end-start)}")

    return blank_image

def IsTooBrightorTooDark(block):
    return np.mean(block) < BRIGHTNESS_THRESHOLD and np.mean(block) > DARKNESS_THRESHOLD

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
    for i in range(BLOCKS_TO_EMBED):
        tmp = selected_blocks_tmp.pop()
        selected_blocks.append(tmp)
        attacked_image[tmp['locations'][0]:tmp['locations'][0] + BLOCK_SIZE,
                        tmp['locations'][0]:tmp['locations'][1] + BLOCK_SIZE] = 1
        
    selected_blocks = sorted(selected_blocks, key=lambda k: k['locations'], reverse=False)

    return selected_blocks


def embedding(original_image_path, watermark_path, alpha, dwt_level):
    """
    Embed watermark by distributing singular values across blocks using additive embedding.
    
    Strategy: Each block embeds a slice of the watermark's singular values.
    For 32x32 watermark and 32 blocks with 4x4 size:
    - Each 4x4 block -> 2x2 LL after 1-level DWT
    - Each LL has 2 singular values
    - Embed watermark S values: block_i gets Swm[i*2:(i+1)*2]
    Add 1 to selected blocks for some
    Args:
        original_image_path: Path to original image
        watermark_path: Path to watermark file
        alpha: Embedding strength for LL subband
        dwt_level: DWT decomposition level (should be 1 for 4x4 blocks)
    
    Returns:
        tuple: (watermarked_image, watermark, Uwm, Vwm)
    """
    # Load image and watermark
    image = cv2.imread(original_image_path, 0)
    watermark, Uwm, Swm, Vwm = get_watermark_svd(watermark_path)
    
    start = time.time()

    attacked_image = attack_image(image)

    # After 1-level DWT on 4x4 block, we get 2x2 LL
    # 2x2 LL gives us 2 singular values
    # So 32 blocks × 2 values/block = 64 positions (we use first 32)
    
    # Select best blocks based on edge content
    selected_blocks = select_best_blocks(image, attacked_image, BLOCKS_TO_EMBED, BLOCK_SIZE)

    n_blocks_in_image = image.shape[0] / BLOCK_SIZE

    shape_LL_tmp = np.floor(image.shape[0] / (2*n_blocks_in_image)).astype(np.uint8) # try to keep float64
    
    # Create a copy of the image for watermarking
    watermarked_image = image.astype(np.float64)

    for i in range(len(selected_blocks)):
        x = selected_blocks[i]['locations'][0]
        y = selected_blocks[i]['locations'][1]

        # get the block from the original image
        block_original = image[x:x + BLOCK_SIZE, y:y + BLOCK_SIZE]
        # compute the LL of the block
        coeffs = pywt.wavedec2(block_original, wavelet='haar', level=1)
        LL_tmp = coeffs[0]
        # apply SVD
        Ui, Si, Vi = np.linalg.svd(LL_tmp)
        Sw = Si.copy()

        # Additive Embedding (Linear combination)
        Sw = Sw + Swm[(i*shape_LL_tmp)%32: (shape_LL_tmp+(i*shape_LL_tmp)%32)] * alpha

        LL_new = np.zeros((shape_LL_tmp, shape_LL_tmp))
        LL_new = (Ui).dot(np.diag(Sw)).dot(Vi)

        # compute the new block
        coeffs[0] = LL_new
        block_new = pywt.waverec2(coeffs, wavelet='haar')
        # replace the block in the original image
        watermarked_image[x:x + BLOCK_SIZE, y:y + BLOCK_SIZE] = block_new

    
    watermarked_image = watermarked_image.astype(np.uint8)

    difference = (-watermarked_image + image) * attacked_image.astype(np.uint8)
    watermarked_image = image + difference + attacked_image.astype(np.uint8)

    end = time.time()
        
    #compute the quality and timings
    w = wpsnr(image,watermarked_image)
    print("[EMEDDING] wPSNR: %.2fdB" % w)
    print(f"[EMBEDDING] Time: {end - start:.2f}s")
    print(f"[EMBEDDING] Embedded in {len(selected_blocks)} blocks of size {BLOCK_SIZE}x{BLOCK_SIZE}")
    
    return watermarked_image, watermark, Uwm, Vwm