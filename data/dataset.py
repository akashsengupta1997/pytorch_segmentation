import torch
import os
from skimage import io
import cv2
from torch.utils.data import Dataset
import numpy as np


class UPS31Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, image_format=".png",
                 label_format=".png", use_surreal_labels=False):
        self.transform = transform
        self.image_format = image_format
        self.use_surreal_labels=use_surreal_labels

        self.image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))
                            if f.endswith(image_format)]
        self.label_paths = [os.path.join(label_dir, f) for f in sorted(os.listdir(label_dir))
                            if f.endswith(label_format)]

        assert len(self.image_paths) == len(self.label_paths), "Number of images and masks " \
                                                               "do not match!"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = io.imread(self.image_paths[index])
        image = image.astype(float)/255.0
        mask = cv2.imread(self.label_paths[index], 0)

        if self.use_surreal_labels:
            pass  #TODO ups31 to surreal labels conversion

        sample = {"image": image, "mask": mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ImageFolder(Dataset):
    def __init__(self, image_dir, transform=None, image_format=".png"):
        self.transform = transform
        self.image_format = image_format

        self.image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))
                            if f.endswith(image_format)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = io.imread(self.image_paths[index])
        image = image.astype(float)/255.0

        if self.transform:
            image = self.transform(image)

        return image






