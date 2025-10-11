import cv2
import numpy as np
from scipy.signal import convolve2d


def psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio."""
    return cv2.PSNR(img1, img2)


def wpsnr(img1, img2):
    """Compute Weighted Peak Signal-to-Noise Ratio using CSF."""
    # Convert to float in range [0, 1]
    img1 = np.float64(img1) / 255.0
    img2 = np.float64(img2) / 255.0
    difference = img1 - img2
    
    # Check if images are identical
    if not np.any(difference):
        return float('inf')  # Return infinity for identical images
    
    # Load CSF (Contrast Sensitivity Function) filter
    w = np.genfromtxt('csf.csv', delimiter=',')
    
    # Apply CSF filter using 'same' mode to preserve image size
    ew = convolve2d(difference, np.rot90(w, 2), mode='same')
    
    # Compute mean squared error in weighted domain
    mse_weighted = np.mean(ew**2)
    
    # Avoid log of very small numbers
    if mse_weighted < 1e-10:
        return 100.0  # Cap at reasonable maximum
    
    # Compute WPSNR
    decibels = 20.0 * np.log10(1.0 / np.sqrt(mse_weighted))
    
    # Cap unreasonably high values
    return decibels