import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset

class ImageRestorationDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.image_filenames = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        
        noisy_path = os.path.join(self.noisy_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name)
        
        noisy_arr = np.load(noisy_path)
        gt_arr = np.load(gt_path)
        
        noisy_tensor = torch.from_numpy(noisy_arr).float()
        gt_tensor = torch.from_numpy(gt_arr).float()
        
        if noisy_tensor.ndim == 2:
            noisy_tensor = noisy_tensor.unsqueeze(0)
        if gt_tensor.ndim == 2:
            gt_tensor = gt_tensor.unsqueeze(0)
            
        # --- DATA AUGMENTATION ---
        # 1. Random Horizontal Flip (50% chance)
        if random.random() > 0.5:
            noisy_tensor = torch.flip(noisy_tensor, dims=[2])
            gt_tensor = torch.flip(gt_tensor, dims=[2])
            
        # 2. Random Vertical Flip (50% chance)
        if random.random() > 0.5:
            noisy_tensor = torch.flip(noisy_tensor, dims=[1])
            gt_tensor = torch.flip(gt_tensor, dims=[1])
            
        # 3. Random 90-degree Rotations (0, 90, 180, or 270 degrees)
        k = random.randint(0, 3)
        if k > 0:
            noisy_tensor = torch.rot90(noisy_tensor, k, dims=[1, 2])
            gt_tensor = torch.rot90(gt_tensor, k, dims=[1, 2])
            
        return noisy_tensor, gt_tensor
