"""
Example of how to create custom dataset in Pytorch. In this case
we have images of cats and dogs in a separate folder and a csv
file containing the name to the jpg file as well as the target
label (0 for cat, 1 for dog).
"""

import os
import pandas as pd
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset

class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
        return (image, y_label)