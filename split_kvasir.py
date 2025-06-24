#!/usr/bin/env python3
# split_kvasir.py

import os
import shutil
from sklearn.model_selection import train_test_split

# Path to your extracted Kvasir-SEG root
SOURCE_ROOT = r"D:\2025-research\Polyp-KAN-AdapterNet\datasets\Kvasir-SEG"
# Adjust these if your extracted folder structure differs:
IMG_DIR  = os.path.join(SOURCE_ROOT, "images")
MASK_DIR = os.path.join(SOURCE_ROOT, "masks")

# Where to write the split
TARGET_DIR = r"D:\2025-research\Polyp-KAN-AdapterNet\datasets\kvasir"

# Ratios
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1  # test will implicitly be 1 - TRAIN_RATIO - VAL_RATIO

def split_dataset(img_dir, mask_dir, target_dir,
                  train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):

    imgs  = sorted(f for f in os.listdir(img_dir)  if f.lower().endswith((".png", ".jpg", ".jpeg")))
    masks = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    assert len(imgs) == len(masks), f"Image/mask count mismatch: {len(imgs)} vs {len(masks)}"

    img_paths  = [os.path.join(img_dir,  f) for f in imgs]
    mask_paths = [os.path.join(mask_dir, f) for f in masks]

    # Split into train and remainder
    train_imgs, rem_imgs, train_masks, rem_masks = train_test_split(
        img_paths, mask_paths, train_size=train_ratio, random_state=42
    )
    # Split remainder into val and test
    val_fraction = val_ratio / (1 - train_ratio)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        rem_imgs, rem_masks, train_size=val_fraction, random_state=42
    )

    splits = {
        "train": (train_imgs, train_masks),
        "val":   (val_imgs,   val_masks),
        "test":  (test_imgs,  test_masks),
    }

    for split, (i_paths, m_paths) in splits.items():
        tgt_img_dir  = os.path.join(target_dir, "images", split)
        tgt_mask_dir = os.path.join(target_dir, "masks",  split)
        os.makedirs(tgt_img_dir,  exist_ok=True)
        os.makedirs(tgt_mask_dir, exist_ok=True)
        print(f"Creating {split} split with {len(i_paths)} samples")
        for ip, mp in zip(i_paths, m_paths):
            shutil.copy(ip,  tgt_img_dir)
            shutil.copy(mp, tgt_mask_dir)

    print(f"\nAll splits created under `{target_dir}`:")
    print(f"  images/train ({len(splits['train'][0])})")
    print(f"  images/val   ({len(splits['val'][0])})")
    print(f"  images/test  ({len(splits['test'][0])})")

if __name__ == "__main__":
    split_dataset(IMG_DIR, MASK_DIR, TARGET_DIR)
