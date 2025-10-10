import cv2
import numpy as np
from scipy.signal import convolve2d


def psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio."""
    return cv2.PSNR(img1, img2)


def wpsnr(img1, img2):
    """Compute Weighted Peak Signal-to-Noise Ratio using CSF."""
    img1 = np.float32(img1) / 255.0
    img2 = np.float32(img2) / 255.0
    difference = img1 - img2
    
    if not np.any(difference):
        return 9999999
    
    w = np.genfromtxt('csf.csv', delimiter=',')
    ew = convolve2d(difference, np.rot90(w, 2), mode='valid')
    decibels = 20.0 * np.log10(1.0 / np.sqrt(np.mean(ew**2)))
    
    return decibels