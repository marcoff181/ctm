import os
import numpy as np
import cv2


def embedding(image, mark_size, alpha=None, v=None, mark=None, **kwargs):
    """LSB watermark embedding - embeds watermark in least significant bits of pixels"""
    image = image.astype(np.uint8)
    h, w = image.shape
    
    # Generate binary watermark if not provided
    if mark is None:
        np.random.seed(42)
        mark = np.random.randint(0, 2, mark_size, dtype=np.uint8)
    
    # Check if image has enough pixels for watermark
    total_pixels = h * w
    if mark_size > total_pixels:
        raise ValueError(f"Watermark size {mark_size} exceeds image capacity {total_pixels}")
    
    # Create watermarked image (copy)
    watermarked = image.copy()
    
    # Flatten image for easier access
    flat_image = watermarked.flatten()
    
    # Embed watermark in LSB of pixels
    for i in range(mark_size):
        # Clear LSB and set it to watermark bit
        flat_image[i] = (flat_image[i] & 0xFE) | mark[i]
    
    # Reshape back to original dimensions
    watermarked = flat_image.reshape(h, w)
    
    return mark, watermarked