from attack import attack_config, param_converters

import cv2
img = cv2.imread("./watermarked_groups_images/crispymcmark_tree.bmp", 0)

# Choose attack by name
attack_name = "JPEG"  # or any key from attack_config
strength = 0        # value between 0.0 and 1.0

attack_func = attack_config[attack_name]
params = param_converters[attack_name](strength)
attacked = attack_func(img, strength)

cv2.imwrite(f"./attacked_groups_images/crispymcmark_crispymcmark_tree_{attack_name}_{params}.bmp", attacked)


