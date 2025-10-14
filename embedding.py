import os
import numpy as np
import cv2
from scipy.fft import dct, idct


# def embedding(image, mark_size, alpha=None, v=None, mark=None, **kwargs):
#     """6th bit watermark embedding with replication"""
#     image = image.astype(np.uint8)
#     h, w = image.shape
    
#     # Generate binary watermark if not provided
#     if mark is None:
#         np.random.seed(42)
#         mark = np.random.randint(0, 2, mark_size, dtype=np.uint8)
    
#     # Calculate replication factor needed
#     total_pixels = h * w
#     replication_factor = int(np.ceil(total_pixels / mark_size))
    
#     # Replicate watermark to fill entire image
#     replicated_mark = np.tile(mark, replication_factor)[:total_pixels]
    
#     # Create watermarked image (copy)
#     watermarked = image.copy()
    
#     # Flatten image for easier access
#     flat_image = watermarked.flatten()
    
#     # Embed watermark in 6th bit (bit position 5, counting from 0)
#     # Bit mask: 0xDF = 11011111 (clears 6th bit)
#     # Shift watermark bit to 6th position
#     for i in range(total_pixels):
#         # Clear 6th bit and set it to watermark bit
#         flat_image[i] = (flat_image[i] & 0xBF) | (replicated_mark[i] << 6)
    
#     # Reshape back to original dimensions
#     watermarked = flat_image.reshape(h, w)
    
#     return mark, watermarked


def embedding(image, mark, alpha, v='multiplicative'):
    # Get the DCT transform of the image
    ori_dct = dct(dct(image,axis=0, norm='ortho'),axis=1, norm='ortho')

    # Get the locations of the most perceptually significant components
    sign = np.sign(ori_dct)
    ori_dct = abs(ori_dct)
    locations = np.argsort(-ori_dct,axis=None) # - sign is used to get descending order
    rows = image.shape[0]
    locations = [(val//rows, val%rows) for val in locations] # locations as (x,y) coordinates
    

    # Embed the watermark
    watermarked_dct = ori_dct.copy()
    for idx, (loc,mark_val) in enumerate(zip(locations[1:], mark)):
        if v == 'additive':
            watermarked_dct[loc] += (alpha * mark_val)
        elif v == 'multiplicative':
            watermarked_dct[loc] *= 1 + (alpha * mark_val)

    # Restore sign and o back to spatial domain
    watermarked_dct *= sign
    watermarked = np.uint8(idct(idct(watermarked_dct,axis=1, norm='ortho'),axis=0, norm='ortho'))

    return watermarked