import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from skimage.transform import rescale

def awgn(img, std=5.0):
    np.random.seed(123)
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def blur(img, sigma=3.0):
    result = gaussian_filter(img, sigma)
    return np.clip(result, 0, 255).astype(np.uint8)


def sharpening(img, sigma=1.0, alpha=1.5):
    blurred = gaussian_filter(img, sigma)
    result = img + alpha * (img - blurred)
    return np.clip(result, 0, 255).astype(np.uint8)


def median(img, kernel_size=3):
    result = medfilt(img, kernel_size)
    return result.astype(np.uint8)


def resizing(img, scale=0.9):
    h, w = img.shape
    downscaled = rescale(img, scale, anti_aliasing=True)
    upscaled = rescale(downscaled, 1/scale, anti_aliasing=True)
    
    upscaled_h, upscaled_w = upscaled.shape
    if upscaled_h >= h and upscaled_w >= w:
        result = upscaled[:h, :w]
    else:
        result = np.zeros((h, w))
        result[:upscaled_h, :upscaled_w] = upscaled
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def jpeg_compression(img, quality=70):
    temp_path = 'tmp.jpg'
    img_pil = Image.fromarray(img)
    img_pil.save(temp_path, "JPEG", quality=quality)
    result = Image.open(temp_path)
    result_array = np.asarray(result, dtype=np.uint8)
    os.remove(temp_path)
    return result_array
