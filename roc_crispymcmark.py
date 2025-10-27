import numpy as np
import random
import time
import sys
import os

from detection_crispymcmark import extraction
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from attack import attack_config
from embedding import embedding
from cv2 import imread, imwrite

WATERMARKED_IMAGES_PRE = "./roc_calculation_images/crispymcmark_"
WATERMARK_PATH = "crispymcmark.npy"
SAMPLE_IMAGES_FOLDER = "images"
SAMPLE_PER_IMAGE = 25


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


def random_attack(img):
    """Select a random attack from the attack_config and apply it to the image."""
    attack_name = random.choice(list(attack_config.keys()))
    attack_func = attack_config[attack_name]
    # Random strength between 0.0 and 1.0
    strength = random.uniform(0.0, 1.0)
    attacked = attack_func(img, strength)
    return attacked


def sorted_rocs(rocs, threshold):
    return sorted(
        [r if r[0] < threshold else (0, 0, 0) for r in rocs], key=lambda k: k[1]
    )[-1]


def _progress_bar(current, total, prefix="Progress", bar_length=40):
    """Mostra una progress bar nel terminale"""
    percent = float(current) / float(total)
    filled_length = int(bar_length * percent)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    sys.stdout.write(f"\r{prefix}: |{bar}| {percent*100:.1f}% ({current}/{total})")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def compute_roc(save_watermarked_images=False):
    # start time
    start = time.time()

    sample_images = []
    # loop for importing images from sample_images folder
    for filename in os.listdir(SAMPLE_IMAGES_FOLDER):
        if filename.endswith(".bmp"):
            path_tmp = os.path.join(SAMPLE_IMAGES_FOLDER, filename)
            sample_images.append(path_tmp)

    # sample_images.sort()

    # load the watermark
    watermark = np.load(WATERMARK_PATH)
    watermark_size = watermark.shape[0]

    # scores and labels are two lists we will use to append the values of similarity and their labels
    # In scores we will append the similarity between our watermarked image and the attacked one,
    # or  between the attacked watermark and a random watermark
    # In labels we will append the 1 if the scores was computed between the watermarked image and the attacked one,
    # and 0 otherwise
    scores = []
    labels = []

    total_images = len(sample_images)
    total_samples = total_images * (SAMPLE_PER_IMAGE + 1)
    current_sample_global = 0

    for idx, image_path in enumerate(sample_images, 1):

        # 1. Embed the watermark into the original image
        watermarked_image = embedding(image_path, WATERMARK_PATH)

        if save_watermarked_images:
            name = image_path.split("/")[1]
            imwrite(WATERMARKED_IMAGES_PRE + name, watermarked_image)

        # ================== ADDITIONS NOT REQUIRED IN CHALLENGE RULES ==========
        # othermark = np.random.uniform(0.0, 1.0, watermark_size)
        # othermark = np.uint8(np.rint(othermark))
        # np.save("./tpm-attacks/othermark.npy", othermark)
        # othermarked_image = embedding.embedding(original_image,"./tmp_attacks/othermark.npy")
        # ====================================================================

        # read original image
        original_image = imread(image_path, 0)

        # 2. Apply random attacks and extract watermark
        sample = 0
        while sample <= SAMPLE_PER_IMAGE:
            # Aggiorna la progress bar
            current_sample_global += 1
            _progress_bar(
                current_sample_global,
                total_samples,
                f"Calculating ROC (Image {idx - 1}/{total_images - 1})",
            )

            # fakemark is the watermark for H0
            fakemark = np.random.randint(0, 2, size=watermark_size, dtype=np.uint8)

            # random attack to watermarked image
            attacked_image = random_attack(watermarked_image)
            attacked_original_image = random_attack(original_image)

            # check if we are extracting the correct mark from an attacked image
            extracted_watermark = extraction(
                original_image, watermarked_image, attacked_image
            )

            scores.append(similarity(watermark, extracted_watermark))
            labels.append(1)

            scores.append(similarity(fakemark, extracted_watermark))
            labels.append(0)

            # ================== ADDITIONS NOT REQUIRED IN CHALLENGE RULES ==========
            # check if we are able to embed and extract a different watermark
            # shows how much information about our watermark we are hardcoding
            # still, not needed for the challenge
            # extracted_othermark = detection.extraction(original_image, othermarked_image, attacked_image)
            #
            # scores.append(similarity(othermark, extracted_othermark))
            # labels.append(1)
            #
            # scores.append(similarity(fakemark, extracted_othermark))
            # labels.append(0)

            # check that passing original(modified by attacks too) as attacked does not find watermark
            extracted_watermark = extraction(
                original_image, watermarked_image, attacked_original_image
            )

            scores.append(similarity(watermark, extracted_watermark))
            labels.append(0)

            # =======================================================================

            sample += 1

    # compute ROC
    fpr, tpr, tau = roc_curve(
        np.asarray(labels), np.asarray(scores), drop_intermediate=False
    )

    rocs = zip(fpr, tpr, tau)

    fp, tp, T = sorted_rocs(rocs, 0.0001)
    print(f"FPR<0.0001  = FPR {fp:.4f} -> TPR: {tp:.2f} threshold: {T:.2f}")

    rocs = zip(fpr, tpr, tau)
    fp, tp, T = sorted_rocs(rocs, 0.01)
    print(f"FPR<0.01    = FPR {fp:.4f} -> TPR: {tp:.2f} threshold: {T:.2f}")

    rocs = zip(fpr, tpr, tau)
    fp, tp, T = sorted_rocs(rocs, 0.1)
    print(f"FPR<0.1     = FPR {fp:.4f} -> TPR: {tp:.2f} threshold: {T:.2f}")

    rocs = zip(fpr, tpr, tau)
    fp, tp, T = sorted_rocs(rocs, 1.0)
    print(f"Highest TPR = FPR {fp:.4f} -> TPR: {tp:.2f} threshold: {T:.2f}")

    # end time
    end = time.time()
    print("[COMPUTE ROC] Time: %0.2f seconds" % (end - start))

    # compute AUC
    roc_auc = auc(fpr, tpr)
    print("AUC: %0.4f" % roc_auc)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="AUC = %0.3f" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - CrispyMcMark")
    plt.legend(loc="lower right")
    plt.savefig("roc_crispymcmark.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    compute_roc()
