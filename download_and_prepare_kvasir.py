#!/usr/bin/env python3
# download_and_prepare_kvasir.py

import os
import requests
import zipfile
import shutil
from sklearn.model_selection import train_test_split

# Two possible URLs
URLS = [
    "https://datasets.simula.no/kvasir-seg/kvasir-seg.zip",
    "https://github.com/pranavraikwal/Kvasir-SEG/archive/refs/heads/master.zip"
]

DOWNLOAD_PATH = "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir-seg.zip"
EXTRACT_DIR   = "D:/2025-research/Polyp-KAN-AdapterNet/datasets/Kvasir-SEG"
TARGET_DIR    = "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir"

def try_download(url, download_path):
    try:
        print(f"Trying to download from {url} ...")
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(download_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download succeeded.")
        return True
    except Exception as e:
        print(f"  → Failed: {e}")
        return False

def download_and_extract():
    os.makedirs(os.path.dirname(DOWNLOAD_PATH), exist_ok=True)
    # Try each URL until one succeeds
    for url in URLS:
        if try_download(url, DOWNLOAD_PATH):
            break
    else:
        print("\nERROR: All download attempts failed.")
        print(f"Please manually download Kvasir-SEG and place it at:\n  {os.path.abspath(DOWNLOAD_PATH)}")
        return False

    # Extract
    if not os.path.exists(EXTRACT_DIR):
        print("Extracting archive...")
        with zipfile.ZipFile(DOWNLOAD_PATH, "r") as z:
            z.extractall(os.path.dirname(EXTRACT_DIR))
        print("Extraction complete.")
    else:
        print("Extraction folder exists, skipping extraction.")
    return True

def split_dataset(img_dir, mask_dir, target_dir, train_ratio=0.8, val_ratio=0.1):
    imgs  = sorted(f for f in os.listdir(img_dir)  if f.lower().endswith((".png",".jpg")))
    masks = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith((".png",".jpg")))
    assert len(imgs) == len(masks), "Image/mask count mis-match"

    img_paths  = [os.path.join(img_dir,  f) for f in imgs]
    mask_paths = [os.path.join(mask_dir, f) for f in masks]

    train_i, rem_i, train_m, rem_m = train_test_split(
        img_paths, mask_paths, train_size=train_ratio, random_state=42
    )
    val_frac = val_ratio / (1 - train_ratio)
    val_i, test_i, val_m, test_m = train_test_split(
        rem_i, rem_m, train_size=val_frac, random_state=42
    )

    for split, (ips, mps) in {
        "train": (train_i, train_m),
        "val":   (val_i,   val_m),
        "test":  (test_i,  test_m),
    }.items():
        im_tgt = os.path.join(target_dir, "images", split)
        ms_tgt = os.path.join(target_dir, "masks",  split)
        os.makedirs(im_tgt, exist_ok=True)
        os.makedirs(ms_tgt, exist_ok=True)
        for src_i, src_m in zip(ips, mps):
            shutil.copy(src_i, im_tgt)
            shutil.copy(src_m, ms_tgt)
    print(f"Splits created under `{target_dir}`.")

if __name__ == "__main__":
    if not download_and_extract():
        exit(1)

    # After extraction, some folders differ depending on source:
    # - Simula archive: EXTRACT_DIR/images & EXTRACT_DIR/masks
    # - GitHub mirror: Kvasir-SEG-master/images & .../masks
    # Let’s detect which:
    if os.path.exists(os.path.join(EXTRACT_DIR, "images")):
        img_src = os.path.join(EXTRACT_DIR, "images")
        msk_src = os.path.join(EXTRACT_DIR, "masks")
    else:
        # e.g. data/Kvasir-SEG/Kvasir-SEG-master/images
        alt = next(d for d in os.listdir(EXTRACT_DIR) if "Kvasir-SEG" in d)
        base = os.path.join(EXTRACT_DIR, alt)
        img_src = os.path.join(base, "images")
        msk_src = os.path.join(base, "masks")

    split_dataset(img_src, msk_src, TARGET_DIR)
