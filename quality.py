import cv2
import numpy as np
from scipy.signal import convolve2d
from math import sqrt


def psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio."""
    print(f"tipo1: {img1.dtype}, tipo2: {img2.dtype}")
    return cv2.PSNR(img1, img2)