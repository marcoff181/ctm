import os
from matplotlib import pyplot as plt
import cv2
import numpy as np

def show_images(img, watermarked):
    plt.subplot(121)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(122)
    plt.title('Watermarked')
    plt.imshow(watermarked, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def save_comparison(original, watermarked, attacked, attack_name, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    images = [(original, 'Original'), (watermarked, 'Watermarked'), 
              (attacked, f'After {attack_name}')]
    
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    filename = attack_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
    plt.savefig(os.path.join(output_dir, f"comparison_{filename}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

def edges_mask(img, low_threshold=100, high_threshold=200, dilate_iter=1):
    if img.ndim==3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img.astype(np.uint8), low_threshold, high_threshold)
    if dilate_iter>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        edges = cv2.dilate(edges, k, iterations=dilate_iter)
    return edges.astype(bool)

def noisy_mask(img, window=7, percentile=90, dilate_iter=0):
    if img.ndim==3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    sq = img*img
    k = (window, window)
    mean = cv2.boxFilter(img, ddepth=-1, ksize=k, normalize=True)
    mean_sq = cv2.boxFilter(sq, ddepth=-1, ksize=k, normalize=True)
    var = mean_sq - mean*mean
    thr = np.percentile(var, percentile)
    mask = var > thr
    if dilate_iter>0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.dilate(mask.astype(np.uint8), se, iterations=dilate_iter).astype(bool)
    return mask
    



    # Generate watermark if needed
    # if not os.path.exists(mark_path):
    #     mark = np.random.uniform(0.0, 1.0, mark_size)
    #     mark = np.uint8(np.rint(mark))
    #     np.save(mark_path, mark)
