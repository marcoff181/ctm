import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from skimage.transform import rescale
from PIL import Image
from embedding import embedding
from detection import compute_threshold, detection, similarity
from quality import psnr, wpsnr


def awgn(img, std=5.0):
    """Add Additive White Gaussian Noise."""
    np.random.seed(123)
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def blur(img, sigma=3.0):
    """Apply Gaussian blur."""
    result = gaussian_filter(img, sigma)
    return np.clip(result, 0, 255).astype(np.uint8)


def sharpening(img, sigma=1.0, alpha=1.5):
    """Apply sharpening filter."""
    blurred = gaussian_filter(img, sigma)
    result = img + alpha * (img - blurred)
    return np.clip(result, 0, 255).astype(np.uint8)


def median(img, kernel_size=5):
    """Apply median filter."""
    result = medfilt(img, kernel_size)
    return result.astype(np.uint8)


def resizing(img, scale=0.7):
    """Apply resizing attack."""
    h, w = img.shape
    downscaled = rescale(img, scale, anti_aliasing=True)
    upscaled = rescale(downscaled, 1/scale, anti_aliasing=True)
    
    # Ensure exact dimensions
    upscaled_h, upscaled_w = upscaled.shape
    if upscaled_h >= h and upscaled_w >= w:
        result = upscaled[:h, :w]
    else:
        # Pad if needed
        result = np.zeros((h, w))
        result[:upscaled_h, :upscaled_w] = upscaled
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def jpeg_compression(img, quality=100):
    """Apply JPEG compression."""
    img = Image.fromarray(img)
    img.save('tmp.jpg', "JPEG", quality=quality)
    result = Image.open('tmp.jpg')
    result = np.asarray(result, dtype=np.uint8)
    os.remove('tmp.jpg')
    return result


def main():
    """Test attacks on first image."""
    images_path = "./images"
    alpha = 0.3
    mark_size = 1024
    v = 'multiplicative'
    
    # Load first image
    filename = sorted(os.listdir(images_path))[0]
    image_path = os.path.join(images_path, filename)
    image = cv2.imread(image_path, 0)
    
    print(f"Testing image: {filename}")
    print(f"{'='*60}\n")
    
    # Embed watermark
    mark, watermarked = embedding(image, mark_size, alpha, v)
    threshold, _ = compute_threshold(mark_size, mark, N=1000)
    
    print(f"Threshold: {threshold:.4f}\n")
    print(f"{'='*60}")
    print(f"ATTACK RESULTS")
    print(f"{'='*60}\n")
    
    # Test all attacks
    attacks = {
        'AWGN': awgn,
        'Blur': blur,
        'Sharpening': sharpening,
        'Median': median,
        'Resizing': resizing,
        'JPEG': jpeg_compression
    }
    
    for attack_name, attack_func in attacks.items():
        # Apply attack
        attacked = attack_func(watermarked)
        
        # Measure quality
        psnr_val = psnr(watermarked, attacked)
        wpsnr_val = wpsnr(watermarked, attacked)
        
        # Detect watermark
        extracted = detection(watermarked, attacked, alpha, mark_size, v)
        sim = similarity(mark, extracted)
        detected = sim > threshold
        
        # Print result
        status = "DETECTED" if detected else "NOT DETECTED"
        icon = "✗" if detected else "✓"
        
        print(f"{attack_name:12s} {icon}")
        print(f"  PSNR:       {psnr_val:6.2f} dB")
        print(f"  WPSNR:      {wpsnr_val:6.2f} dB")
        print(f"  Similarity: {sim:6.4f}")
        print(f"  Status:     {status}\n")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()