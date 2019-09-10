import os
import cv2
import numpy as np


def count_classes(mask_dir, mask_wh, num_classes, mask_format=".png"):
    total_counts = np.zeros(num_classes)

    for fname in sorted(os.listdir(mask_dir)):
        if fname.endswith(mask_format):
            mask = cv2.imread(os.path.join(masks_dir, fname), 0)
            mask = cv2.resize(mask, (mask_wh, mask_wh),
                              interpolation=cv2.INTER_NEAREST)

            counts = np.zeros(32)
            labels, img_counts = np.unique(mask, return_counts=True)
            counts[labels] = img_counts
            total_counts += np.array(counts)





masks_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded_small_glob_rot/masks/train"
