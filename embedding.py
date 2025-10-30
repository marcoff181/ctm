from pywt import dwt2, wavedec2, waverec2
from cv2 import imread
import numpy as np

# embedded parameters
ALPHA = 10.0
N_BLOCKS = 8 
BLOCK_SIZE = 8  


def get_watermark_S(watermark_path):
    """Load watermark and compute its singular values."""
    watermark = np.load(watermark_path)
    watermark = watermark.reshape(32, 32)
    _, S, _ = np.linalg.svd(watermark, full_matrices=False)
    return S


def svd_flat_score(block):
    # get LL
    block = block.astype(np.float32)
    LL, _ = dwt2(block, "haar")

    # SVD of LL
    S = np.linalg.svd(LL, full_matrices=False)[1]
    # normalize singular values
    S /= S.sum() + 1e-8
    # shannon entropy normalized: high entropy means noisyer block, better for invisibility but our method does not embed well into really noisy blocks
    entropy = -np.sum(S * np.log2(S + 1e-8)) / np.log2(len(S))
    # computes variance: high variance -> bright/dark extremes
    energy = np.var(LL)

    # compromise between the two, use entropy exponent to tweak the final score
    return float((entropy**3) * np.exp(-energy / 50))


def select_best_blocks(original_image):
    """Select best blocks based on how much they are attacked by using `strength_map`"""

    blocks = []

    for i in range(0, original_image.shape[0], BLOCK_SIZE):
        for j in range(0, original_image.shape[1], BLOCK_SIZE):
            block_location = slice(i, i + BLOCK_SIZE), slice(j, j + BLOCK_SIZE)
            blocks.append(
                {
                    "locations": (i, j),
                    "entropy": svd_flat_score(original_image[block_location]),
                }
            )

    # select blocks with highest entropy score
    best_blocks = sorted(blocks, key=lambda k: k["entropy"], reverse=True)[:N_BLOCKS]
    block_positions = [block["locations"] for block in best_blocks]

    # order blocks based on their location, so they can be retrieved in a deterministic order
    return sorted(block_positions)


def embedding(image_path, watermark_path):
    """Embed watermark using DWT-SVD with block selection.

    Args:
        image_path (str): Path to the input image.
        watermark_path (str): Path to the watermark image.

    Returns:
        np.ndarray: Watermarked image.
    """

    image = imread(image_path, 0)
    Swm = get_watermark_S(watermark_path)

    blocks = select_best_blocks(image)

    for idx, (x, y) in enumerate(blocks):
        block_location = (slice(x, x + BLOCK_SIZE), slice(y, y + BLOCK_SIZE))

        # DWT
        block = image[block_location]
        coeffs = wavedec2(block, wavelet="haar", level=1)
        LLb = coeffs[0]

        # SVD
        Ub, Sb, Vb = np.linalg.svd(LLb)
        Sb[0] += Swm[idx] * ALPHA

        # iSVD
        LLnew = Ub.dot(np.diag(Sb)).dot(Vb)
        # iDWT
        coeffs[0] = LLnew
        block_watermarked = waverec2(coeffs, wavelet="haar")

        image[block_location] = block_watermarked

    return image
