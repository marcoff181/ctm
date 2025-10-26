from cv2 import imread
import numpy as np


def embedding(image_path, watermark_path):
    """Embed using lsb"""

    image = imread(image_path, 0)
    watermark = np.load(watermark_path)

    # lsb
    # Flatted the images for easy embedding
    pixels = image.flatten()
    flat_im_size = np.shape(pixels)

    # Embedding
    idx_watermark_bit = 0
    for i in range(flat_im_size[0]):
        im_pixel_bytes = format(pixels[i], "08b")

        # concate all the pixels until the LSB from the image
        # and the MSB of the watermark as LSB
        watermarked_data = im_pixel_bytes[:7] + str(watermark[idx_watermark_bit])

        # encoding bytes back to pixels
        pixels[i] = int(watermarked_data, 2)
        idx_watermark_bit += 1
        if idx_watermark_bit > 1023:
            idx_watermark_bit = 0

    watermarked_img = pixels.reshape(512, 512)

    return watermarked_img
