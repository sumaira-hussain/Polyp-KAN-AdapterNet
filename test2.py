import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

import cfg
from models.polyp_kan import UKAN  # Make sure this matches your correct model file
from datasets.kvasir import KvasirSegDataset


def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = image.clone()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(image, 0, 1)


def create_overlay(image, prediction):
    pred_mask = torch.zeros_like(image)
    pred_mask[0] = prediction.float()
    overlay = torch.clamp(image + pred_mask, 0, 1)
    return overlay


def test(checkpoint_path, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model according to the version in your repo
    model = UKAN(
        num_classes=1,
        input_channels=3,
        deep_supervision=False,
        img_size=352,
        patch_size=16,
        embed_dims=[128, 256, 512],
        no_kan=False
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Load with strict=False to allow partial loading
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path} successfully!")

    # Test-time data augmentation
    test_transforms = A.Compose([
        A.Resize(352, 352),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    test_ds = KvasirSegDataset(
        "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/images/test",
        "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/masks/test",
        transforms=test_transforms
    )
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    pred_dir = os.path.join(save_dir, "predictions")
    overlay_dir = os.path.join(save_dir, "overlays")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    with torch.no_grad():
        for images, masks, img_paths in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5

            for i in range(len(img_paths)):
                orig_name = Path(img_paths[i]).stem
                denorm_img = denormalize(images[i].cpu())
                pred_path = os.path.join(pred_dir, f"{orig_name}_pred.png")
                torchvision.utils.save_image(predictions[i].float().cpu(), pred_path)

                overlay = create_overlay(denorm_img, predictions[i].cpu())
                overlay_path = os.path.join(overlay_dir, f"{orig_name}_overlay.png")
                torchvision.utils.save_image(overlay, overlay_path)

    print(f"Predictions saved to: {pred_dir}")
    print(f"Overlays saved to: {overlay_dir}")

if __name__ == '__main__':
    checkpoint_path = 'D:/2025-research/Polyp-KAN-AdapterNet/checkpoint/polyp_kan/20250604_171032/best_dice_epoch21.pth'
    save_dir = 'D:/2025-research/Polyp-KAN-AdapterNet/test_results'
    test(checkpoint_path, save_dir)
