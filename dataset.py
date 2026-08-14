import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SemiconductorDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir):
        super().__init__()
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        
        all_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.npy')])
        
        self.valid_files = [f for f in all_files if os.path.exists(os.path.join(gt_dir, f))]
        
        if len(self.valid_files) == 0:
            raise FileNotFoundError(f"No paired .npy files found in {noisy_dir} and {gt_dir}")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        filename = self.valid_files[idx]
        
        noisy_arr = np.load(os.path.join(self.noisy_dir, filename)).astype(np.float32)
        gt_arr = np.load(os.path.join(self.gt_dir, filename)).astype(np.float32)

        if noisy_arr.ndim == 2:
            noisy_arr = np.expand_dims(noisy_arr, axis=0)
        if gt_arr.ndim == 2:
            gt_arr = np.expand_dims(gt_arr, axis=0)

        noisy_tensor = torch.from_numpy(noisy_arr)
        gt_tensor = torch.from_numpy(gt_arr)

        return noisy_tensor, gt_tensor