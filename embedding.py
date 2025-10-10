#YOUR CODE
import os
from scipy.fft import dct, idct
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
from math import sqrt


def embedding(image, mark_size, alpha, v='multiplicative', mark=None, freq_range='high'):
    """DCT-based embedding with frequency selection"""
    ori_dct = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    # Use absolute values for location selection only
    abs_dct = abs(ori_dct)
    locations = np.argsort(-abs_dct, axis=None)
    rows = image.shape[0]
    locations = [(val // rows, val % rows) for val in locations]
    
    # Select frequency range
    if freq_range == 'high':
        selected_locs = locations[1:mark_size + 1]
    elif freq_range == 'mid':
        selected_locs = locations[500:500 + mark_size]
    elif freq_range == 'low':
        selected_locs = locations[100:100 + mark_size]
    else:
        selected_locs = locations[1:mark_size + 1]
    
    if mark is None:
        mark = np.random.uniform(0.0, 1.0, mark_size)
        mark = np.uint8(np.rint(mark))
    
    watermarked_dct = ori_dct.copy()
    for loc, mark_val in zip(selected_locs, mark):
        if v == 'additive':
            watermarked_dct[loc] += (alpha * mark_val)
        elif v == 'multiplicative':
            watermarked_dct[loc] *= 1 + (alpha * mark_val)
    
    watermarked = np.uint8(idct(idct(watermarked_dct, axis=1, norm='ortho'), axis=0, norm='ortho'))
    
    return mark, watermarked