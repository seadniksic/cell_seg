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

        self.img_names = os.listdir(image_path)[:-5]
        self.label_path = label_path

        self.transform = transform
        self.label_transform = label_transform

    def __getitem__(self, idx):

        image = torch.from_numpy(pyvips.Image.new_from_file(os.path.join(self.image_path, self.img_names[idx])).numpy(dtype=np.float32))
        image = torch.unsqueeze(image, 0)
        mask = np.array(cv2.imread(f"{os.path.join(self.label_path, self.img_names[idx][:-5])}.png", cv2.IMREAD_GRAYSCALE), dtype=np.int64)
        mask[mask > 0] = 1
        mask = torch.from_numpy(mask)

        if self.transform:
            image=self.transform(image)

        if self.label_transform:
            mask = self.label_transform(mask)

        return image, mask

    def __len__(self):
        return len(self.img_names)
