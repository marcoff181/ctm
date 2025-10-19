import os
import numpy as np
import pywt
from scipy.fft import dct, idct
from wpsnr import wpsnr
from matplotlib import pyplot as plt


def detection(original, watermarked, attacked, Uw, Vw, alpha=0.05, dwt_level=2):
    """Detect watermark in attacked image."""
    extracted_wm = extraction(original, attacked, Uw, Vw, alpha=alpha, dwt_level=dwt_level)
    original_wm = extraction(original, watermarked, Uw, Vw, alpha=alpha, dwt_level=dwt_level)

    # Measure quality
    wpsnr_attack = wpsnr(watermarked, attacked)

    # Detect watermark
    sim = similarity(original_wm, extracted_wm)

    # TODO: use ROC to compute final threshold
    detected = 1 if sim > 0.7 else 0

    return detected, wpsnr_attack


def extraction(original, watermarked, Uw, Vw, alpha=0.05, type="multiplicative", dwt_level=2):
    """Extract watermark from watermarked image."""

    # Multi-level DWT decomposition
    coeffs_original = pywt.wavedec2(original, wavelet="haar", level=dwt_level)
    coeffs_watermarked = pywt.wavedec2(watermarked, wavelet="haar", level=dwt_level)

    LLi = coeffs_original[0]  # Deepest LL from original
    LLa = coeffs_watermarked[0]  # Deepest LL from watermarked/attacked

    # Apply DCT to LL subbands (CRITICAL: must match embedding!)
    LLi_dct = dct(dct(LLi, axis=0, norm='ortho'), axis=1, norm='ortho')
    LLa_dct = dct(dct(LLa, axis=0, norm='ortho'), axis=1, norm='ortho')

    # SVD on DCT-transformed LL subbands
    Ui, Si, Vi = np.linalg.svd(LLi_dct)
    Ua, Sa, Va = np.linalg.svd(LLa_dct)

    # Extract watermark singular values
    # Embedding formula: S_new = Si + alpha * Si * Swm (multiplicative)
    # Therefore: Swm = (S_new - Si) / (alpha * Si)

    # Use the size of Uw/Vw to determine watermark dimensions
    watermark_sv_count = min(Uw.shape[1], Vw.shape[0])

    # Extract using the inverse of embedding formula
    if type == "multiplicative":
        epsilon = 1e-10  # small constant to avoid division by zero
        Sw = (Sa[:watermark_sv_count] - Si[:watermark_sv_count]) / (
            alpha * Si[:watermark_sv_count] + epsilon
        )
    else:
        # Additive formula: S_new = Si + alpha * Swm
        # Therefore: Swm = (S_new - Si) / alpha
        Sw = (Sa[:watermark_sv_count] - Si[:watermark_sv_count]) / alpha

    # Recompose watermark from singular values
    extracted_watermark = Uw.dot(np.diag(Sw)).dot(Vw)

    # Flatten and binarize for similarity/BER
    extracted_watermark = extracted_watermark.flatten()
    extracted_watermark = (extracted_watermark > 0.5).astype(np.uint8)

    return extracted_watermark


def similarity(X, X_star):
    """Compute bit error rate (BER) based similarity for binary watermarks"""
    X = X.astype(np.uint8)
    X_star = X_star.astype(np.uint8)

    # Calculate number of matching bits
    matches = np.sum(X == X_star)
    total = len(X)

    # Similarity: 1.0 = perfect match, 0.0 = all bits different
    similarity_score = matches / total

    return similarity_score


def compute_threshold(mark_size, w, N=1000):
    """Compute detection threshold using Monte Carlo simulation for binary watermarks"""
    np.random.seed(42)
    SIM = np.zeros(N)

    # Convert watermark to binary
    w_binary = (w > 0.5).astype(np.uint8)

    for i in range(N):
        # Generate random binary sequence
        r = np.random.randint(0, 2, mark_size, dtype=np.uint8)
        SIM[i] = similarity(w_binary, r)

    SIMs = SIM.copy()
    SIM.sort()

    # Threshold: mean + 3*std (for binary, expected ~0.5 for random)
    mean_sim = np.mean(SIM)
    std_sim = np.std(SIM)
    T = mean_sim + 3 * std_sim

    return T, SIMs

def verify_watermark_extraction(original, watermarked, Uwm, Vwm, alpha, mark_path, output_prefix="", dwt_level=3):
    """Verify that the embedded watermark can be correctly extracted."""
    print("\n" + "=" * 80)
    print("WATERMARK EXTRACTION VERIFICATION")
    print("=" * 80)
    
    # Load original watermark
    original_watermark = np.load(mark_path)
    
    # Extract watermark from watermarked image (no attack)
    extracted_watermark = extraction(original, watermarked, Uwm, Vwm, alpha=alpha, dwt_level=dwt_level)
    
    # Compute similarity
    sim = similarity(original_watermark, extracted_watermark)
    
    # Bit statistics
    total_bits = len(original_watermark)
    matching_bits = np.sum(original_watermark == extracted_watermark)
    differing_bits = total_bits - matching_bits
    
    # Bit pattern analysis
    both_zero = np.sum((original_watermark == 0) & (extracted_watermark == 0))
    both_one = np.sum((original_watermark == 1) & (extracted_watermark == 1))
    orig_one_ext_zero = np.sum((original_watermark == 1) & (extracted_watermark == 0))
    orig_zero_ext_one = np.sum((original_watermark == 0) & (extracted_watermark == 1))
    
    print(f"\nExtraction Statistics:")
    print(f"  Total bits:        {total_bits}")
    print(f"  Matching bits:     {matching_bits} ({matching_bits/total_bits*100:.2f}%)")
    print(f"  Differing bits:    {differing_bits} ({differing_bits/total_bits*100:.2f}%)")
    print(f"  Similarity:        {sim:.4f}")
    
    print(f"\nBit Pattern Distribution:")
    print(f"  Both 0:            {both_zero} ({both_zero/total_bits*100:.2f}%)")
    print(f"  Both 1:            {both_one} ({both_one/total_bits*100:.2f}%)")
    print(f"  Orig=1, Ext=0:     {orig_one_ext_zero} ({orig_one_ext_zero/total_bits*100:.2f}%)")
    print(f"  Orig=0, Ext=1:     {orig_zero_ext_one} ({orig_zero_ext_one/total_bits*100:.2f}%)")
    
    # Hamming distance
    hamming_dist = differing_bits
    normalized_hamming = hamming_dist / total_bits
    print(f"\nHamming Distance:  {hamming_dist}")
    print(f"Normalized:        {normalized_hamming:.4f}")
    
    # Status
    if sim >= 0.95:
        status = "EXCELLENT"
        status_symbol = "[OK]"
        bbox_color = 'lightgreen'
    elif sim >= 0.7:
        status = "GOOD"
        status_symbol = "[OK]"
        bbox_color = 'lightblue'
    elif sim >= 0.5:
        status = "WARNING"
        status_symbol = "[!]"
        bbox_color = 'lightyellow'
    else:
        status = "ERROR"
        status_symbol = "[X]"
        bbox_color = 'lightcoral'
    
    print(f"\nExtraction Status: {status_symbol} {status}")
    
    # Prepare watermark visualizations
    size = int(np.sqrt(total_bits))
    wm_orig = original_watermark.reshape(size, size)
    wm_extracted = extracted_watermark.reshape(size, size)
    error_2d = (original_watermark != extracted_watermark).reshape(size, size).astype(float)
    
    # ========== PLOT 1: Overview (Watermarks + Images) ==========
    fig1 = plt.figure(figsize=(18, 10))
    gs1 = fig1.add_gridspec(2, 4, hspace=0.25, wspace=0.25)
    
    # Top row: Watermarks
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.imshow(wm_orig, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    ax1.set_title(f'Original Watermark\n{np.sum(original_watermark)} ones', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig1.add_subplot(gs1[0, 1])
    ax2.imshow(wm_extracted, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    ax2.set_title(f'Extracted Watermark\n{np.sum(extracted_watermark)} ones', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig1.add_subplot(gs1[0, 2])
    im3 = ax3.imshow(error_2d, cmap='RdYlGn_r', interpolation='nearest', vmin=0, vmax=1)
    ax3.set_title(f'Error Map\n{differing_bits} errors ({normalized_hamming*100:.1f}%)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Error', fontsize=10)
    
    # Confusion Matrix
    ax4 = fig1.add_subplot(gs1[0, 3])
    confusion = np.array([[both_zero, orig_zero_ext_one],
                         [orig_one_ext_zero, both_one]])
    im4 = ax4.imshow(confusion, cmap='Blues', aspect='auto')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Extr: 0', 'Extr: 1'], fontsize=10)
    ax4.set_yticklabels(['Orig: 0', 'Orig: 1'], fontsize=10)
    ax4.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    for i in range(2):
        for j in range(2):
            color = "white" if confusion[i, j] > confusion.max()/2 else "black"
            ax4.text(j, i, f'{confusion[i, j]}\n({confusion[i, j]/total_bits*100:.1f}%)',
                    ha="center", va="center", color=color, fontsize=10, fontweight='bold')
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # Bottom row: Images
    ax5 = fig1.add_subplot(gs1[1, 0])
    ax5.imshow(original, cmap='gray')
    ax5.set_title('Original Image', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    ax6 = fig1.add_subplot(gs1[1, 1])
    ax6.imshow(watermarked, cmap='gray')
    wpsnr_val = wpsnr(original, watermarked)
    ax6.set_title(f'Watermarked Image\nWPSNR: {wpsnr_val:.2f} dB', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    ax7 = fig1.add_subplot(gs1[1, 2])
    diff_img = np.abs(watermarked.astype(float) - original.astype(float))
    im7 = ax7.imshow(diff_img, cmap='hot')
    ax7.set_title(f'Embedding Difference\nMax: {diff_img.max():.1f}', fontsize=12, fontweight='bold')
    ax7.axis('off')
    cbar7 = plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    
    # Statistics panel
    ax8 = fig1.add_subplot(gs1[1, 3])
    ax8.axis('off')
    
    stats_text = f"""QUALITY METRICS
{'='*28}

    Similarity:      {sim:.4f}
    BER:             {normalized_hamming:.4f}
    Hamming Dist:    {hamming_dist}

    MATCHING BITS:   {matching_bits:4d}
    Both 0:        {both_zero:4d}
    Both 1:        {both_one:4d}

    ERROR BITS:      {differing_bits:4d}
    False Pos:     {orig_zero_ext_one:4d}
    False Neg:     {orig_one_ext_zero:4d}

    PARAMETERS
    Alpha:           {alpha:.6f}
    DWT Level:       {dwt_level}

    STATUS:          {status}
"""
    
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.4, edgecolor='black', linewidth=1.5))
    
    fig1.suptitle(f'Watermark Extraction Verification | Similarity: {sim:.4f} | Status: {status}', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    output_path1 = os.path.join("./", f"{output_prefix}extraction_overview.png")
    plt.savefig(output_path1, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\nSaved overview plot to: {output_path1}")
    plt.close(fig1)
    
    print("=" * 80 + "\n")
    
    return {
        'similarity': sim,
        'matching_bits': matching_bits,
        'differing_bits': differing_bits,
        'hamming_distance': hamming_dist,
        'status': status,
        'original_watermark': original_watermark,
        'extracted_watermark': extracted_watermark
    }