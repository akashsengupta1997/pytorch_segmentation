import torch
import os
from skimage import io, transform
import cv2
import numpy as np
from torch.utils.data import Dataset


class Resize(object):
    def __init__(self, height, width):
        self.new_height = height
        self.new_width = width

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        image = transform.resize(image, (self.new_height, self.new_width),
                                 order=1)
        mask = transform.resize(mask, (self.new_height, self.new_width),
                                order=0,
                                preserve_range=True)
        image = image.astype(np.float32)
        mask = mask.astype(np.int64)
        return {'image': image, 'mask': mask}


class ToTensor(object):
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = np.transpose(image, [2, 0, 1])
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}

