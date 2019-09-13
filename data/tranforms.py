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


class RandomRotate(object):
    def __init__(self, rotate_range):
        self.rotate_range = rotate_range

    def __call__(self, sample):
        if isinstance(sample, dict):
            image, mask = sample['image'], sample['mask']
            image = image.astype(np.float32)
            mask = mask.astype(np.int64)

            rotation_angle = np.random.uniform(-self.rotate_range, self.rotate_range)
            rotated_image = transform.rotate(image, rotation_angle, order=1, resize=True)
            rotated_mask = transform.rotate(mask, rotation_angle, order=0, preserve_range=True,
                                            resize=True)

            return {'image': rotated_image, 'mask': rotated_mask}


class RandomCrop(object):
    def __init__(self, min_height_scale, min_width_scale):
        self.min_height_scale = min_height_scale
        self.min_width_scale = min_width_scale

    def __call__(self, sample):
        if isinstance(sample, dict):
            image, mask = sample['image'], sample['mask']
            image = image.astype(np.float32)
            mask = mask.astype(np.int64)
            assert image.shape[:2] == mask.shape[:2]
            height, width = image.shape[:2]

            new_height = int(np.random.uniform(height*self.min_height_scale, height))
            new_width = int(np.random.uniform(width*self.min_width_scale, width))
            crop_origin_x = int(np.random.uniform(0, height-new_height))
            crop_origin_y = int(np.random.uniform(0, width-new_width))

            cropped_image = image[crop_origin_x:crop_origin_x+new_height,
                                  crop_origin_y:crop_origin_y+new_width]

            cropped_mask = mask[crop_origin_x:crop_origin_x + new_height,
                                  crop_origin_y:crop_origin_y + new_width]

            return {'image': cropped_image, 'mask': cropped_mask}
