import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageOps
import PIL
import torch, torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math
import random
import os

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    mask = mask.astype(int)
    return mask

mask = create_circular_mask(128,128)

class ansimDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,img_list_csv, seq_csv, root_dir, step=20, random_rotate = True, mask = mask, transform=None):
        """
        Args:
            image_csv (string): Path to the csv file with image path.
            seq_csv (string): Path to the csv file with indices of heads of sequence.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_list = pd.read_csv(img_list_csv)
        self.seq_list = pd.read_csv(seq_csv)
        self.root_dir = root_dir
        self.transform = transform
        self.step = step
        self.random_rotate = random_rotate
        self.mask = mask

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):
        seq_head = self.seq_list.iloc[idx,0]
        seq = torch.empty(self.step, 1, 128,128, dtype=torch.float)
        angle = 360 * np.random.uniform(0, 1)
        for i in np.arange(self.step):
            img_idx = seq_head + i
            img_name = os.path.join(self.root_dir, self.img_list.iloc[img_idx, 0])
            image = Image.open(img_name)
            image = image.convert('L')
            image_resized = torchvision.transforms.functional.resize(image, (128,128), interpolation=2)
            if self.random_rotate:
                image_resized = torchvision.transforms.functional.rotate(image_resized, angle, resample=False, expand=False, center=None)
            image_resized = image_resized * self.mask
            image_tensor = torch.from_numpy(image_resized)
            seq[i][0] = image_tensor
        if self.transform:
            seq = self.transform(seq)
        return seq
