import cv2
import numpy as np
from scipy.signal import convolve2d
from math import sqrt


def psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio."""
    return cv2.PSNR(img1, img2)


def wpsnr(img1, img2):
    """Compute Weighted Peak Signal-to-Noise Ratio using CSF."""
    # Convert to float in range [0, 1]
    img1 = np.float32(img1) / 255.0
    img2 = np.float32(img2) / 255.0
    difference = img1 - img2
    
    # Check if images are identical
    if not np.any(difference):
        return 9999999  # Return infinity for identical images
    
    # Load CSF (Contrast Sensitivity Function) filter
    w = np.genfromtxt('csf.csv', delimiter=',')
    ew = convolve2d(difference, np.rot90(w, 2), mode='valid')

    # Compute WPSNR
    decibels = 20.0*np.log10(1.0/sqrt(np.mean(np.mean(ew**2))))
    
    # Cap unreasonably high values
    return decibels