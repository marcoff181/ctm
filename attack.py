import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from skimage.transform import rescale

from embedding import embedding
from detection import detection, similarity, compute_threshold
from wpsnr import wpsnr


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Simple configuration."""
    images_path = "./images"
    alpha = 0.005
    mark_size = 1024
    mark_path = "mark.npy"
    image_index = 2
    output_dir = "./attack_results"
    save_images = True


# ============================================================================
# Attack Functions
# ============================================================================

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


def median(img, kernel_size=3):
    """Apply median filter."""
    result = medfilt(img, kernel_size)
    return result.astype(np.uint8)


def resizing(img, scale=0.9):
    """Apply resizing attack."""
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
    """Apply JPEG compression."""
    temp_path = 'tmp.jpg'
    img_pil = Image.fromarray(img)
    img_pil.save(temp_path, "JPEG", quality=quality)
    result = Image.open(temp_path)
    result_array = np.asarray(result, dtype=np.uint8)
    os.remove(temp_path)
    return result_array


# ============================================================================
# Attack Suite
# ============================================================================

def get_attacks():
    """Return list of attacks to test."""
    return [
        ('AWGN (std=3)', lambda img: awgn(img, std=3.0)),
        ('AWGN (std=5)', lambda img: awgn(img, std=5.0)),
        ('AWGN (std=10)', lambda img: awgn(img, std=10.0)),
        ('Blur (σ=1)', lambda img: blur(img, sigma=1.0)),
        ('Blur (σ=3)', lambda img: blur(img, sigma=3.0)),
        ('Blur (σ=5)', lambda img: blur(img, sigma=5.0)),
        ('Sharp (α=1.0)', lambda img: sharpening(img, sigma=1.0, alpha=1.0)),
        ('Sharp (α=1.5)', lambda img: sharpening(img, sigma=1.0, alpha=1.5)),
        ('Sharp (α=2.0)', lambda img: sharpening(img, sigma=1.0, alpha=2.0)),
        ('Median (k=3)', lambda img: median(img, kernel_size=3)),
        ('Median (k=5)', lambda img: median(img, kernel_size=5)),
        ('Median (k=7)', lambda img: median(img, kernel_size=7)),
        ('Resize (0.5x)', lambda img: resizing(img, scale=0.5)),
        ('Resize (0.7x)', lambda img: resizing(img, scale=0.7)),
        ('Resize (0.9x)', lambda img: resizing(img, scale=0.9)),
        ('JPEG (Q=50)', lambda img: jpeg_compression(img, quality=50)),
        ('JPEG (Q=70)', lambda img: jpeg_compression(img, quality=70)),
        ('JPEG (Q=90)', lambda img: jpeg_compression(img, quality=90)),
    ]


# ============================================================================
# Visualization
# ============================================================================

def show_images(img, watermarked):
    """Display original and watermarked images."""
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
    """Save comparison image."""
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


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function."""
    config = Config()
    
    # Setup output directory
    if config.save_images:
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"Saving images to: {config.output_dir}\n")
    
    # Generate watermark if needed
    if not os.path.exists(config.mark_path):
        mark = np.random.uniform(0.0, 1.0, config.mark_size)
        mark = np.uint8(np.rint(mark))
        np.save(config.mark_path, mark)
    
    # Load image
    filename = sorted(os.listdir(config.images_path))[config.image_index]
    image_path = os.path.join(config.images_path, filename)
    original = cv2.imread(image_path, 0)
    print(f"Testing image: {filename}")
    print(f"Alpha: {config.alpha}\n")
    
    # Embed watermark
    print("Embedding watermark...")
    watermarked, watermark, Uwm, Vwm = embedding(image_path, config.mark_path, config.alpha)
    watermarked = watermarked.astype(np.uint8)
    
    # Save base images
    if config.save_images:
        cv2.imwrite(os.path.join(config.output_dir, "original.png"), original)
        cv2.imwrite(os.path.join(config.output_dir, "watermarked.png"), watermarked)
    
    # Compute threshold
    threshold, _ = compute_threshold(config.mark_size, watermark, N=1000)
    
    # Print quality metrics
    psnr = cv2.PSNR(original, watermarked)
    wpsnr_val = wpsnr(original, watermarked)
    print(f"\nWatermark Quality:")
    print(f"  PSNR:  {psnr:6.2f} dB")
    print(f"  WPSNR: {wpsnr_val:6.2f} dB")
    print(f"\nDetection Threshold: {threshold:.4f}\n")
    
    # Show images
    show_images(original, watermarked)
    
    # Test clean detection
    extracted_clean = detection(original, watermarked, watermarked, Uwm, Vwm, config.alpha)
    sim_clean = similarity(watermark, extracted_clean)
    print(f"Clean watermarked similarity: {sim_clean:.4f} (should be ~1.0)\n")
    
    # Run attacks
    print("=" * 60)
    print(f"{'ATTACK RESULTS':^60}")
    print("=" * 60 + "\n")
    
    attacks = get_attacks()
    detected_count = 0
    
    for attack_name, attack_func in attacks:
        # Apply attack
        attacked = attack_func(watermarked)
        
        # Measure quality
        psnr_att = cv2.PSNR(watermarked, attacked)
        wpsnr_att = wpsnr(watermarked, attacked)
        
        # Detect watermark
        extracted = detection(original, watermarked, attacked, Uwm, Vwm, config.alpha)
        sim = similarity(watermark, extracted)
        detected = sim > threshold
        
        # Save comparison
        if config.save_images:
            save_comparison(original, watermarked, attacked, attack_name, config.output_dir)
            clean_name = attack_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
            cv2.imwrite(os.path.join(config.output_dir, f"attacked_{clean_name}.png"), attacked)
        
        # Print results
        icon = "✓" if detected else "✗"
        status = "DETECTED" if detected else "NOT DETECTED"
        print(f"{attack_name:20s} {icon}")
        print(f"  PSNR:       {psnr_att:6.2f} dB")
        print(f"  WPSNR:      {wpsnr_att:6.2f} dB")
        print(f"  Similarity: {sim:6.4f} (threshold: {threshold:.4f})")
        print(f"  Status:     {status}\n")
        
        if detected:
            detected_count += 1
    
    # Print summary
    total = len(attacks)
    rate = (detected_count / total) * 100
    
    print("=" * 60)
    print(f"{'SUMMARY':^60}")
    print("=" * 60)
    print(f"Total Attacks: {total}")
    print(f"Watermark Detected: {detected_count}/{total} ({rate:.1f}%)")
    print(f"Watermark NOT Detected: {total - detected_count}/{total} ({100 - rate:.1f}%)")
    
    if config.save_images:
        print(f"\nAll images saved to: {config.output_dir}/")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
