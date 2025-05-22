# datasets/kvasir.py
import os
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np

class KvasirSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transforms=None):
        self.img_paths  = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img  = cv2.imread(self.img_paths[idx])[:,:,::-1]  # BGR→RGB
        mask = cv2.imread(self.mask_paths[idx], 0)        # grayscale
        if self.transforms:
            data = self.transforms(image=img, mask=mask)
            img, mask = data["image"], data["mask"]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.  # HWC → CHW

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        mask = (mask.unsqueeze(0).float() > 127).float()

        return {'image': img, 'mask': mask, "image_meta_dict": {"filename_or_obj": self.img_paths[idx]} }

        # return img, mask
