import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from skimage.transform import rescale

from embedding import embedding
from detection import detection, extraction, similarity, compute_threshold
from wpsnr import wpsnr
import pandas as pd
from itertools import combinations
from attack_functions import awgn, blur, sharpening, median, resizing, jpeg_compression
from utilities import edges_mask , noisy_mask 


# Configuration
# images_path = "./images"
alpha = 2.605
mark_size = 1024
mark_path = "./mark.npy"
input_dir = "./watermarked_groups_images/"
output_dir = "./attacked_groups_images/"
originals_dir = "./challenge_images/"
attacked_wpsnr_lower_bound = 35

# conversion between the input value 0.0..1.0 and the actual parameters of each attack function
param_converters = {
    'JPEG':  lambda x: int(round((1 - x) * 100)),                 
    'Blur':  lambda x: x * 10,                                   
    'AWGN':  lambda x: x * 50,                                   
    'Resize':lambda x: max(0.001, 0.5 ** (x * 10)),             
    'Median':lambda x: [1, 3, 5, 7][int(round(x * 3))],        
    'Sharp': lambda x: x * 3         
}

# attacks that take as input a strenght value `x` between 0.0 and 1.0
attack_config = {
    'JPEG':  lambda img, x: jpeg_compression(img, quality=param_converters['JPEG'](x)),
    'Blur':  lambda img, x: blur(img, sigma=param_converters['Blur'](x)),
    'AWGN':  lambda img, x: awgn(img, std=param_converters['AWGN'](x)),
    'Resize':lambda img, x: resizing(img, scale=param_converters['Resize'](x)),
    'Median':lambda img, x: median(img, kernel_size=param_converters['Median'](x)),
    'Sharp': lambda img, x: sharpening(img,sigma=1.0,alpha=param_converters['Sharp'](x))
}

# TODO: tweak iterations to find balance between speed and accuracy
def bin_search_attack(original, watermarked, detection, Uwm, Vwm ,mask,iterations=6):
    results = []
    
    for attack_name, attack_func in attack_config.items():
        low, high = 0.0, 1.0
        best_param, best_wpsnr = None, -np.inf
        
        for _ in range(iterations):
            mid = (low + high) / 2
            # converged
            if abs(high - low) < 1e-6:
                break
            
            try:
                full_attacked_img = attack_func(watermarked.copy(), mid)
                attacked_img = np.where(mask, full_attacked_img, watermarked)
                detected, wpsnr_val = detection(original, watermarked, attacked_img, Uwm, Vwm)
                actual_param = param_converters[attack_name](mid)
                # print(f"{attack_name} det:{detected} WPSNR:{wpsnr_val:.2f}dB x:{mid:.4f} par:{actual_param}")
                
                if not detected:
                    best_param, best_wpsnr = mid, wpsnr_val
                    high = mid
                else:
                    low = mid
                    
            except Exception as e:
                print(f"Error during {attack_name} with param {mid}: {e}")
                break
        
        if best_param is not None:
            actual_param = param_converters[attack_name](mid)
            print(f"  ✓ {attack_name}: Optimal param = {actual_param:.4f} | WPSNR: {best_wpsnr:.2f} dB")
            results.append({'Attack': attack_name, 'Best_Parameter': actual_param, 'WPSNR': best_wpsnr, 'Status': 'Removed'})
        else:
            print(f"  ✗ {attack_name}: Could not remove watermark")
            results.append({'Attack': attack_name, 'Best_Parameter': np.nan, 'WPSNR': np.nan, 'Status': 'Not Removed'})
    return pd.DataFrame(results)

def main():
    print("===============================================")
    print("CrispyMcMark Attack Suite")
    print("===============================================")

    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)

    # TODO: hardcode Uwm and Vwm inside the detection func
    _, _, Uwm, Vwm = embedding("./challenge_images/0002.bmp", mark_path, alpha)

    # TODO: remove
    # generate an image to simulate having images to attack: to use put an image (exs. 0002.bmp) inside the challenge_images folder
    # and then update the paths below to generate a corresponding watermarked image
    if len(os.listdir(input_dir)) == 0:
        watermarked, watermark, Uwm, Vwm = embedding("./challenge_images/0002.bmp", mark_path, alpha)
        watermarked = watermarked.astype(np.uint8)
        cv2.imwrite("./watermarked_groups_images/cryspymcmark_0002.bmp", watermarked)
    
    # Load image
    for filename in sorted(os.listdir(input_dir)): 
        image_path = os.path.join(input_dir, filename)

        # expected image name is groupName_imageName.bmp
        group_name, image_name = os.path.splitext(filename)[0].split('_')

        watermarked = cv2.imread(image_path, 0)
        print(f"Attacking image: {image_name}")
        print(f"of group: {group_name}")

        # find out which is the original challenge image to compare to
        original_path = os.path.join(originals_dir, image_name + ".bmp")
        original = cv2.imread(original_path, 0)
    
        # Print quality metrics
        wpsnr_val = wpsnr(original, watermarked)
        print(f"\nWatermark Quality:")
        print(f"  WPSNR: {wpsnr_val:6.2f} dB")

        print("binary search with no mask...")
        mask = original >= 0 
        res = bin_search_attack(original,watermarked,detection,Uwm,Vwm,mask)

        print("binary search with edges mask...")
        emask = edges_mask(original) 
        res = bin_search_attack(original,watermarked,detection,Uwm,Vwm,emask)

        print("binary search with noisy mask...")
        nmask = noisy_mask(original) 
        res = bin_search_attack(original,watermarked,detection,Uwm,Vwm,nmask)

        # df = pd.DataFrame(res)
        # df.plot.bar(x='Attack', y='WPSNR', color=df['Status'].map({'Removed': 'green', 'Not Removed': 'red'}))
        # plt.ylabel('WPSNR (dB)')
        # plt.title('Attack Effectiveness')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.show()
         
        # TODO: find best attack and save it in output
        
        # remove to run for all images
        # TODO: add parallelization
        return


if __name__ == "__main__":
    main()


    # combined_results = test_combined_attacks(
    #     original=original,
    #     watermarked=watermarked,
    #     detection=detection,
    #     Uwm=Uwm,
    #     Vwm=Vwm,
    #     output_dir=output_dir
    # )
    # # Access results
    # best = combined_results['best_attack']
    # df_all = combined_results['results_df']
    # df_removed = combined_results['removed_df']
# def test_combined_attacks(original, watermarked, detection, Uwm, Vwm, output_dir):
#     """
#     Test all possible pairs of attacks with different strength combinations.
#     Find the attack with best WPSNR that still removes the watermark.
#
#     Args:
#         original: Original image
#         watermarked: Watermarked image
#         detection: the group's detection function
#         Uwm, Vwm: SVD components from embedding
#         output_dir: Directory to save results
#
#     Returns:
#         dict: Best attack configuration and results DataFrame
#     """
#
#     # Get all category pairs
#     category_names = list(attack_categories.keys())
#     category_pairs = list(combinations(category_names, 2))
#
#     print(f"Testing {len(category_pairs)} attack category combinations...")
#     print(f"Categories: {', '.join(category_names)}\n")
#
#     results = []
#     best_attack = None
#     best_wpsnr = -np.inf
#
#     # Test each category pair
#     for cat1, cat2 in category_pairs:
#         print(f"\nTesting {cat1} + {cat2} combinations...")
#
#         # Test all strength combinations for this category pair
#         for attack1_name, attack1_func in attack_categories[cat1]:
#             for attack2_name, attack2_func in attack_categories[cat2]:
#                 # Apply attacks in sequence: attack1 -> attack2
#                 try:
#                     attacked = attack1_func(watermarked.copy())
#                     attacked = attack2_func(attacked)
#
#                     detected, wpsnr_val = detection(original, watermarked, attacked, Uwm, Vwm)
#
#                     # Store result
#                     combo_name = f"{attack1_name} + {attack2_name}"
#                     result = {
#                         'Attack_1': attack1_name,
#                         'Attack_2': attack2_name,
#                         'Combined_Name': combo_name,
#                         'WPSNR': wpsnr_val,
#                         'Detected': detected,
#                         'Removed': not detected
#                     }
#                     results.append(result)
#
#                     # Check if this is the best attack that removes watermark
#                     if not detected and wpsnr_val > best_wpsnr:
#                         best_wpsnr = wpsnr_val
#                         best_attack = {
#                             'name': combo_name,
#                             'attack1': (attack1_name, attack1_func),
#                             'attack2': (attack2_name, attack2_func),
#                             'wpsnr': wpsnr_val,
#                             'image': attacked.copy()
#                         }
#
#                     # Print progress for successful removals
#                     if not detected:
#                         print(f"  ✓ {combo_name:50s} WPSNR: {wpsnr_val:6.2f} dB (REMOVED)")
#                     else:
#                         print(f"  ✗ {combo_name:50s} WPSNR: {wpsnr_val:6.2f} dB (DETECTED)")
#
#                 except Exception as e:
#                     print(f"  ✗ Error with {attack1_name} + {attack2_name}: {str(e)}")
#                     continue
#
#     # Convert results to DataFrame
#     df = pd.DataFrame(results)
#
#     # Sort by WPSNR (descending) for removed watermarks
#     df_removed = df[df['Removed'] == True].sort_values('WPSNR', ascending=False)
#     df_detected = df[df['Removed'] == False].sort_values('WPSNR', ascending=False)
#
#     # Print summary
#     print("\n" + "=" * 80)
#     print(f"{'COMBINED ATTACK RESULTS SUMMARY':^80}")
#     print("=" * 80)
#     print(f"\nTotal combinations tested: {len(results)}")
#     print(f"Watermark removed: {len(df_removed)} ({len(df_removed)/len(results)*100:.1f}%)")
#     print(f"Watermark survived: {len(df_detected)} ({len(df_detected)/len(results)*100:.1f}%)")
#
#     print(f"{'BEST ATTACK (Highest WPSNR that removes watermark)':^80}")
#     print("=" * 80)
#     if best_attack:
#         print(f"Attack: {best_attack['name']}")
#         print(f"WPSNR:  {best_attack['wpsnr']:.2f} dB")
#         print(f"Status: WATERMARK REMOVED ✓")
#
#         # Save best attack result
#         if output_dir:
#             os.makedirs(output_dir, exist_ok=True)
#             best_path = os.path.join(output_dir, "best_combined_attack.png")
#             cv2.imwrite(best_path, best_attack['image'])
#             print(f"\nBest attack image saved to: {best_path}")
#     else:
#         print("\n⚠ Warning: No combined attack successfully removed the watermark!")
#
#     return {
#         'best_attack': best_attack,
#         'results_df': df,
#         'removed_df': df_removed,
#         'detected_df': df_detected
#     }
