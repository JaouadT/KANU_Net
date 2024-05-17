import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import os
from PIL import Image, ImageOps
import cv2

import os

import cv2
import numpy as np
import torch
import torch.utils.data

      
# Create a pytorch dataset class for the data and use albumentations for data augmentation
class Dataset(torch.utils.data.Dataset):
    def __init__(self, images_list, masks_list, transform=None):
        self.images_list = images_list
        self.masks_list = masks_list
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.images_list[idx])
        image = cv2.resize(image, (224, 224))
        image = np.array(image)


        mask = cv2.imread(self.masks_list[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224, 224))
        mask = np.array(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        #print("Augmented image", image.shape)
        #print("Augmented mask", mask.shape)

        image = image.transpose(2, 0, 1)
        mask = np.expand_dims(mask, axis=2)
        mask = mask.transpose(2, 0, 1)

        image = image/255.0
        mask = mask/255.0

        # return image, mask, self.images_list[idx]
        return image, mask