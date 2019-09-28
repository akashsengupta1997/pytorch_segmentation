import os
import cv2
import numpy as np
import torch


def pad_to_square_mask(mask):
    height, width = mask.shape[:2]

    if width < height:
        border_width = (height - width) // 2
        padded_mask = cv2.copyMakeBorder(mask, 0, 0, border_width, border_width,
                                         cv2.BORDER_CONSTANT, value=0)
    else:
        border_width = (width - height) // 2
        padded_mask = cv2.copyMakeBorder(mask, border_width, border_width, 0, 0,
                                         cv2.BORDER_CONSTANT, value=0)
    return padded_mask


def count_classes(mask_dir, mask_h, mask_w, num_classes, mask_format=".png", padding=True):
    total_counts = np.zeros(num_classes)

    for fname in sorted(os.listdir(mask_dir)):
        if fname.endswith(mask_format):
            mask = cv2.imread(os.path.join(mask_dir, fname), 0)
            mask = pad_to_square_mask(mask)
            mask = cv2.resize(mask, (mask_h, mask_w), interpolation=cv2.INTER_NEAREST)
            counts = np.zeros(32)
            labels, img_counts = np.unique(mask, return_counts=True)
            counts[labels] = img_counts
            total_counts += np.array(counts)

    return total_counts


def enet_style_bounded_log_weighting(mask_dir, mask_h, mask_w, num_classes, mask_format=".png",
                                     c=1.02):
    class_counts = count_classes(mask_dir, mask_h, mask_w, num_classes,
                                 mask_format=mask_format)
    class_probabilities = np.divide(class_counts, np.sum(class_counts))
    class_weights = np.divide(1.0, np.log(c + class_probabilities))
    class_weights = torch.from_numpy(class_weights.astype(np.float32))

    return class_weights


def median_frequency_balancing(mask_dir, mask_h, mask_w, num_classes, mask_format=".png"):
    class_frequencies = count_classes(mask_dir, mask_h, mask_w, num_classes,
                                      mask_format=mask_format)
    median_freq = np.median(class_frequencies)
    class_weights = np.divide(median_freq, class_frequencies)
    class_weights = torch.from_numpy(class_weights.astype(np.float32))

    return class_weights


def simple_bg_down_weight(bg_weight, num_classes):
    class_weights = np.ones(num_classes)
    class_weights[0] = bg_weight
    class_weights = torch.from_numpy(class_weights.astype(np.float32))

    return class_weights