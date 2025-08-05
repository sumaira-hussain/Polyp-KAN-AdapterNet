# datasets/kvasir.py
import os
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np


# class KvasirSegDataset(Dataset):
#     def __init__(self, img_dir, mask_dir, transforms=None):
#         self.img_paths  = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
#         self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, idx):
#         # Load mask first with proper thresholding
#         img_path = self.image_paths[idx] #return path
#         mask_path = self.mask_paths[idx] #return path

#         mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
#         mask = (mask > 127).astype(np.float32)  # Threshold EARLY at numpy level
#         mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Match albumentations input format
#         img  = cv2.imread(self.img_paths[idx])[:,:,::-1]  # BGR→RGB
#         # mask = cv2.imread(self.mask_paths[idx], 0)        # grayscale

#         # Apply transforms (if any)
#         if self.transforms:
#             # Note: Albumentations expects image and mask in specific formats
#             data = self.transforms(image=img, mask=mask)
#             img, mask = data["image"], data["mask"]
#         # Ensure mask has proper shape [1, H, W]
#         if mask.dim() == 3 and mask.shape[0] == 1:
#             # Already in [1, H, W] format
#             pass
#         elif mask.dim() == 2:
#             mask = mask.unsqueeze(0)  # Add channel dimension
#         else:
#             # Handle unexpected format
#             mask = mask[:1]  # Take first channel if multiple exist

#         # if isinstance(img, np.ndarray):
#         #     img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.  # HWC → CHW
#         #
#         # if isinstance(mask, np.ndarray):
#         #     mask = torch.from_numpy(mask)
#         #
#         # mask = (mask.unsqueeze(0).float() > 127).float()
#         return img, mask, img_path  # Return tuple: (image, mask, image_path)

#         # return {'image': img, 'mask': mask,
#         #         'image_meta_dict': { 'filename_or_obj': os.path.basename(self.img_paths[idx])}
#         #          }

#                 # "image_meta_dict": {"filename_or_obj": self.img_paths[idx]}


#         # return img, mask
class KvasirSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transforms=None):
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load mask first with proper thresholding
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # Read image and mask
        img = cv2.imread(img_path)[:, :, ::-1]  # BGR to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        
        # Threshold mask to binary (0 or 1)
        mask = (mask > 127).astype(np.float32)
        
        # Apply transforms if any
        if self.transforms:
            data = self.transforms(image=img, mask=mask)
            img, mask = data["image"], data["mask"]
        
        # Convert to tensors if transforms didn't handle it
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        
        # Ensure mask has proper shape [1, H, W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension
        
        return {
            'image': img,
            'mask': mask,
            'image_meta_dict': {'filename_or_obj': os.path.basename(img_path)}
        }


class CachedKvasir(KvasirSegDataset):
    def __init__(self, img_dir, mask_dir, transforms=None, cache_policy='train'):
        super().__init__(img_dir, mask_dir, transforms)
        self.cache_policy = cache_policy
        self._init_cache()

    def _init_cache(self):
        if self.cache_policy == 'train':
            self.cache_size = len(self)  # Cache entire dataset
            self.pin_memory = True
        elif self.cache_policy == 'val':
            self.cache_size = min(512, len(self))  # Cap at 512
            self.pin_memory = True
        else:  # test
            self.cache_size = 0
            self.pin_memory = False

        self.cache = {}
        self.cache_order = []

    def __getitem__(self, idx):
        try:
            return super().__getitem__(idx)
        except:
            return self._load_random_valid_sample()

        if idx in self.cache:
            return self.cache[idx]

        # Get from parent class
        img, mask, img_path = super().__getitem__(idx)
        # Cache handling
        if self.pin_memory and isinstance(img, torch.Tensor):
            img = img.pin_memory()
        if self.pin_memory and isinstance(mask, torch.Tensor):
            mask = mask.pin_memory()

        # Create tuple to return
        sample = (img, mask, img_path)
        # sample = super().__getitem__(idx)
        # if self.pin_memory:
        #     sample['image'] = sample['image'].pin_memory()
        #     sample['mask'] = sample['mask'].pin_memory()

        if self.cache_size > 0:
            if len(self.cache) >= self.cache_size:
                # Remove oldest cache entry
                del self.cache[self.cache_order.pop(0)]
            self.cache[idx] = sample
            self.cache_order.append(idx)


        return sample
