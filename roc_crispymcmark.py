import time
import random
import cv2
import os
import numpy as np
from attack_functions import awgn, blur, jpeg_compression, median, resizing, sharpening
import embedding, detection_crispymcmark as detection_crispymcmark
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt

from attack import attack_config


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
    # Select a random attack from attack_config
    attack_name = random.choice(list(attack_config.keys()))
    attack_func = attack_config[attack_name]
    # Random strength between 0.0 and 1.0
    strength = random.uniform(0.9, 1.0)
    attacked = attack_func(img, strength)
    # print(f"Applied attack: {attack_name} with strength {strength:.2f}")
    return attacked


# def attack_strength_map(original_image):
#     """evenly sample attacks and find out where they affect the original image the most"""
#     strength_map = np.zeros((512, 512), dtype=np.uint64)
#
#     steps = 10
#     attack_range = np.linspace(0.0,1.0,steps)
#     n_of_attacks = len(attack_config) * steps
#
#
#     for attack in attack_config.values():
#         for x in attack_range:
#             attacked = attack(original_image.copy(),x)
#             diff = attacked - original_image
#             strength_map +=  diff
#
#     #divide by n_of_attacks to get back to the uint8 scale
#     strength_map = np.astype(strength_map/n_of_attacks,np.uint8)
#     cv2.imwrite(f"./attack_diffs/embedding_attack_tests_sum.bmp",strength_map)
#
#     return strength_map


def compute_roc():
    # start time
    start = time.time()
    from sklearn.metrics import roc_curve, auc

    sample_images = []
    # loop for importing images from sample_images folder
    for filename in os.listdir("images"):
        if filename.endswith(".bmp"):
            path_tmp = os.path.join("images", filename)
            sample_images.append(path_tmp)

    sample_images.sort()

    # generate your watermark (if it is necessary)
    watermark_size = 1024
    watermark_path = "crispymcmark.npy"
    watermark = np.load(watermark_path)

    # scores and labels are two lists we will use to append the values of similarity and their labels
    # In scores we will append the similarity between our watermarked image and the attacked one,
    # or  between the attacked watermark and a random watermark
    # In labels we will append the 1 if the scores was computed between the watermarked image and the attacked one,
    # and 0 otherwise
    scores = []
    labels = []

    for i in range(0, len(sample_images)):
        # for i in range(0, 5):

        original_image = sample_images[i]

        watermarked_image = embedding.embedding(original_image, watermark_path)
        # just to check how visible is the mark
        name = original_image.split("/")[1]
        cv2.imwrite(
            "./watermarked_groups_images/crispymcmark_" + name, watermarked_image
        )

        # ================== ADDITIONS NOT REQUIRED IN CHALLENGE RULES ==========
        # othermark = np.random.uniform(0.0, 1.0, watermark_size)
        # othermark = np.uint8(np.rint(othermark))
        # np.save("./tpm-attacks/othermark.npy", othermark)
        # othermarked_image = embedding.embedding(original_image,"./tmp_attacks/othermark.npy")
        # ====================================================================

        original_image = cv2.imread(original_image, 0)
        print(sample_images[i], end="\r")
        # plot original and watermarked image
        # plt.subplot(1, 2, 1)
        # plt.imshow(original_image, cmap='gray')
        # plt.title('Original image')
        # plt.subplot(1, 2, 2)
        # plt.imshow(watermarked_image, cmap='gray')
        # plt.title('Watermarked image')
        # plt.show()

        sample = 0
        while sample <= 25:
            # fakemark is the watermark for H0
            fakemark = np.random.uniform(0.0, 1.0, watermark_size)
            fakemark = np.uint8(np.rint(fakemark))

            # random attack to watermarked image (you can modify it)
            attacked_image = random_attack(watermarked_image)

            # check we are extracting the correct mark from an attacked image
            extracted_watermark = detection_crispymcmark.extraction(
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

            # check that passing original as attacked does not find watermark
            extracted_watermark = detection_crispymcmark.extraction(
                original_image, watermarked_image, original_image
            )

            scores.append(similarity(watermark, extracted_watermark))
            labels.append(0)

            # =======================================================================

            sample += 1

    # print the scores and labels
    # print('Scores:', scores)
    # print('Labels:', labels)

    # compute ROC
    fpr, tpr, tau = roc_curve(
        np.asarray(labels), np.asarray(scores), drop_intermediate=False
    )
    # compute AUC
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="AUC = %0.2f" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig("roc_crispymcmark.png", dpi=300, bbox_inches="tight")

    rocs = zip(fpr, tpr, tau)

    fp, tp, T = sorted(
        [r if r[0] < 0.0001 else (0, 0, 0) for r in rocs], key=lambda k: k[1]
    )[-1]
    print(f"Choose this = FPR {fp:.4f} -> TPR: {tp:.2f} threshold: {T:.2f}")
    rocs = zip(fpr, tpr, tau)
    fp, tp, T = sorted(
        [r if r[0] < 0.01 else (0, 0, 0) for r in rocs], key=lambda k: k[1]
    )[-1]
    print(f"FPR<0.01    = FPR {fp:.4f} -> TPR: {tp:.2f} threshold: {T:.2f}")
    rocs = zip(fpr, tpr, tau)
    fp, tp, T = sorted(
        [r if r[0] < 0.1 else (0, 0, 0) for r in rocs], key=lambda k: k[1]
    )[-1]
    print(f"FPR<0.1     = FPR {fp:.4f} -> TPR: {tp:.2f} threshold: {T:.2f}")
    rocs = zip(fpr, tpr, tau)
    fp, tp, T = sorted(rocs, key=lambda k: k[1])[-1]
    print(f"Highest TPR = FPR {fp:.4f} -> TPR: {tp:.2f} threshold: {T:.2f}")
    plt.show()

    # end time
    end = time.time()
    print("[COMPUTE ROC] Time: %0.2f seconds" % (end - start))


compute_roc()
