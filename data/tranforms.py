import torch
from skimage import transform
import numpy as np
import cv2


class Resize(object):
    def __init__(self, image_height, image_width, mask_height, mask_width):
        self.new_image_height = image_height
        self.new_image_width = image_width
        self.new_mask_height = mask_height
        self.new_mask_width = mask_width

    def __call__(self, sample):
        if isinstance(sample, dict):
            image, mask = sample['image'], sample['mask']
            image = cv2.resize(image, (self.new_image_height, self.new_image_width),
                               interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.new_mask_height, self.new_mask_width),
                              interpolation=cv2.INTER_NEAREST)
            image = image.astype(np.float32)
            mask = mask.astype(np.int64)
            return {'image': image, 'mask': mask}
        else:
            image = sample
            image = cv2.resize(image, (self.new_image_height, self.new_image_width),
                               interpolation=cv2.INTER_LINEAR)
            return image


class ToTensor(object):
    def __call__(self, sample, transpose_channels=True):
        if isinstance(sample, dict):
            image, mask = sample['image'], sample['mask']
            image = image.astype(np.float32)
            mask = mask.astype(np.int64)
            if transpose_channels:
                image = np.transpose(image, [2, 0, 1])
            return {'image': torch.from_numpy(image),
                    'mask': torch.from_numpy(mask)}
        else:
            image = sample
            image = image.astype(np.float32)
            if transpose_channels:
                image = np.transpose(image, [2, 0, 1])
            return torch.from_numpy(image)


class PadToSquare(object):
    def __call__(self, sample):
        if isinstance(sample, dict):
            image, mask = sample['image'], sample['mask']
            image = image.astype(np.float32)
            mask = mask.astype(np.int64)
            assert image.shape[:2] == mask.shape[:2]
            height, width = image.shape[:2]

            if width < height:
                border_width = (height - width) // 2
                padded_image = cv2.copyMakeBorder(image, 0, 0, border_width, border_width,
                                                  cv2.BORDER_CONSTANT, value=0)
                padded_mask = cv2.copyMakeBorder(mask, 0, 0, border_width, border_width,
                                                 cv2.BORDER_CONSTANT, value=0)
            else:
                border_width = (width - height) // 2
                padded_image = cv2.copyMakeBorder(image, border_width, border_width, 0, 0,
                                                  cv2.BORDER_CONSTANT, value=0)
                padded_mask = cv2.copyMakeBorder(mask, border_width, border_width, 0, 0,
                                                 cv2.BORDER_CONSTANT, value=0)

            return {'image': padded_image, 'mask': padded_mask}
        else:
            image = sample
            image = image.astype(np.float32)
            height, width = image.shape[:2]
            if width < height:
                border_width = (height - width) // 2
                padded_image = cv2.copyMakeBorder(image, 0, 0, border_width, border_width,
                                                  cv2.BORDER_CONSTANT, value=0)
            else:
                border_width = (width - height) // 2
                padded_image = cv2.copyMakeBorder(image, border_width, border_width, 0, 0,
                                                  cv2.BORDER_CONSTANT, value=0)
            return padded_image