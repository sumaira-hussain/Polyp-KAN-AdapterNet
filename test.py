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
from models.polyp_kan import UKAN
from datasets.kvasir import KvasirSegDataset  # Ensure this returns (img, mask, path)


def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a tensor image"""
    # Clone to avoid modifying original
    image = image.clone()

    # Denormalize each channel
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)

    return torch.clamp(image, 0, 1)


def create_overlay(image, prediction):
    """Create image + prediction overlay"""
    # Create prediction mask (red)
    pred_mask = torch.zeros_like(image)
    pred_mask[0] = prediction.float()  # Red channel

    # Blend image and prediction
    overlay = torch.clamp(image + pred_mask, 0, 1)
    return overlay


def test(checkpoint_path, save_dir):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
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
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Data transforms - should match validation transforms
    test_transforms = A.Compose([
        A.Resize(352, 352),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Dataset and loader
    test_ds = KvasirSegDataset(
        "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/images/test",
        "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/masks/test",
        transforms=test_transforms
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Create directories for saving results
    pred_dir = os.path.join(save_dir, "predictions")
    overlay_dir = os.path.join(save_dir, "overlays")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # Inference loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images, masks, img_paths = batch
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5

            # Process each item in the batch
            for i in range(len(img_paths)):
                # Get original filename
                orig_name = Path(img_paths[i]).stem

                # Denormalize image for visualization
                denorm_img = denormalize(images[i].cpu())

                # Save prediction mask
                pred_path = os.path.join(pred_dir, f"{orig_name}_pred.png")
                torchvision.utils.save_image(
                    predictions[i].float().cpu(),
                    pred_path
                )

                # Create and save overlay
                overlay = create_overlay(denorm_img, predictions[i].cpu())
                overlay_path = os.path.join(overlay_dir, f"{orig_name}_overlay.png")
                torchvision.utils.save_image(overlay, overlay_path)

    print(f"Predictions saved to: {pred_dir}")
    print(f"Overlays saved to: {overlay_dir}")


if __name__ == '__main__':
    # Example usage
    checkpoint_path = 'D:/2025-research/Polyp-KAN-AdapterNet/checkpoint/polyp_kan/20250604_171032/best_dice_epoch21.pth'
    save_dir = 'D:/2025-research/Polyp-KAN-AdapterNet/test_results'

    test(checkpoint_path, save_dir)