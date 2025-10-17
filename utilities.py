import os
from matplotlib import pyplot as plt

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
