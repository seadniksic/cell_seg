import os
import cv2
import pyvips
import torch
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class VSDataset(Dataset):

    def __init__(self, image_path, label_path, transform=None, label_transform=None):
        self.image_path = image_path

        self.img_names = [d[:-4] for d in os.listdir(image_path)]
        self.label_path = label_path

        self.transform = transform
        self.label_transform = label_transform

    def __getitem__(self, idx):

        image = cv2.imread(os.path.join(self.image_path, self.img_names[idx]) + ".png")
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
        image = np.array(image, dtype=np.float32)
        image = torch.from_numpy(image)
        image = torch.permute(image, (2,0,1))

        mask = np.array(cv2.resize(cv2.imread(f"{os.path.join(self.label_path, self.img_names[idx])}.png", cv2.IMREAD_GRAYSCALE), (256,256), interpolation=cv2.INTER_LINEAR), dtype=np.int64)
        mask[mask > 0] = 1
        mask = torch.from_numpy(mask)

        if self.transform:
            image=self.transform(image)

        if self.label_transform:
            mask = self.label_transform(mask)

        return image, mask

    def __len__(self):
        return len(self.img_names)
