#!/usr/bin/env python3
# train.py

import os
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

import cfg               # your arg parser
import function          # with train_sam, validation_sam
from conf import settings
from models.polyp_kan import UKAN  # or PolypKANNet
from datasets.kvasir import KvasirSegDataset, CachedKvasir
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler
from lion_pytorch import Lion

import GPUtil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torchvision

class EarlyStopper:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_dice = 0

    def __call__(self, current_dice):
        if current_dice > self.best_dice + self.min_delta:
            self.best_dice = current_dice
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def create_overlay(image, prediction):
    """Create image + prediction overlay"""
    # Denormalize image
    image = image.cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)

    # Create prediction mask (red)
    pred_mask = torch.zeros_like(image)
    pred_mask[0] = prediction.cpu().float()  # Red channel

    # Blend image and prediction
    overlay = torch.clamp(image + pred_mask, 0, 1)
    return overlay

def main():
    args = cfg.parse_args()
    device = torch.device(f"cuda:{args.gpu_device}" if args.gpu and torch.cuda.is_available() else "cpu")
    #
    # # ____ Data Paths ___________#

    # train_ds = KvasirSegDataset("D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/images/train", "data/kvasir/masks/train", transforms=train_transforms)
    # val_ds = KvasirSegDataset("D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/images/val", "data/kvasir/masks/val", transforms=val_transforms)
    # test_ds = KvasirSegDataset("D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/images/test", "data/kvasir/masks/test", transforms=val_transforms)

    # ─── Data Transforms & Loaders ─────────────────────────────────────────────

    # train_transforms = T.Compose([
    #     T.ToPILImage(),
    #     T.Resize((352,352)),
    #     T.RandomHorizontalFlip(),
    #     T.ToTensor(),
    # ])

    train_transforms = A.Compose([
        A.Resize(352, 352),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ElasticTransform(p=0.3),  # Simulate tissue deformation
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(var_limit=(0.001, 0.005)), # Simulate endoscopic noise
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})  # Explicitly handle mask target


    val_transforms = A.Compose([
        A.Resize(352, 352),
        # A.Normalize(mean=(0.5,), std=(0.5,)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})  # Explicitly handle mask target
    # val_transforms = T.Compose([
    #     T.ToPILImage(),
    #     T.Resize((352,352)),
    #     T.ToTensor(),
    # ])
    # test_transforms = T.Compose([
    #     T.ToPILImage(),
    #     T.Resize((352, 352)),
    #     T.ToTensor(),
    # ])
    test_transforms = A.Compose([
        A.Resize(352, 352),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    # train_ds = KvasirSegDataset(
    #     "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/images/train", "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/masks/train",
    #     transforms=train_transforms
    # )
    train_ds = CachedKvasir(
        "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/images/train",
        "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/masks/train",
        transforms=train_transforms, cache_policy='train'
    )
    # val_ds = KvasirSegDataset(
    #     "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/images/val", "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/masks/val",
    #     transforms=val_transforms
    # )
    val_ds = CachedKvasir(
        "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/images/val",
        "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/masks/val",
        transforms=val_transforms,  cache_policy='val'
    )
    test_ds = KvasirSegDataset(
        "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/images/test","D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/masks/test",
        transforms=val_transforms
    )

    train_loader = DataLoader(train_ds, batch_size=args.b, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.b//2, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.b// 2, shuffle=False, num_workers=2, pin_memory=True)

    # After dataset creation
    if getattr(args, 'cache_warmup', False):  # Handles missing argument
        print("Warming up caches...")

        # Training cache
        for i in tqdm(range(len(train_ds)), desc="Training Cache"):
            try:
                _ = train_ds[i]
            except Exception as e:
                print(f"Error loading training sample {i}: {str(e)}")

        # Validation cache
        for i in tqdm(range(len(val_ds)), desc="Validation Cache"):
            try:
                _ = val_ds[i]
            except Exception as e:
                print(f"Error loading validation sample {i}: {str(e)}")

        print(f"Train cache: {len(train_ds.cache)}/{len(train_ds)}")
        print(f"Val cache: {len(val_ds.cache)}/{len(val_ds)}")

    # ─── Model Instantiation ───────────────────────────────────────────────────
    net = UKAN(
        num_classes=1,
        input_channels=3,
        deep_supervision=False,
        img_size=352,
        patch_size=16,
        embed_dims=[128, 256, 512],       # embed_dims=[256,320,512],
        no_kan=False
    )
    net.to(device)

    # ─── Load Pretrained Weights (if any) ─────────────────────────────────────
    if args.pretrain:
        print(f"[INFO] Loading pretrained weights from {args.pretrain}")
        state = torch.load(args.pretrain, map_location=device)
        net.load_state_dict(state, strict=False)

    # ─── Freeze All Except Adapters & KAN Layers ───────────────────────────────
    # for name, param in net.named_parameters():
    #     if "adapt" not in name.lower() and "kan" not in name.lower():
    #         param.requires_grad = False
    # After (include decoder/final layers)
    trainable_modules  = ['adapt1', 'adapt2', 'adapt3',
    'decoder1', 'decoder2', 'decoder3', 'decoder4', 'decoder5',
    'block1', 'block2', 'dblock1', 'dblock2',
    'final']
    for name, param in net.named_parameters():
        if any(module in name for module in trainable_modules):
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in net.parameters())
    print(f"[FREEZE] Trainable params: {trainable}/{total}")

    # ─── Optimizer & Scheduler ────────────────────────────────────────────────
    # optimizer = optim.AdamW(
    #     filter(lambda p: p.requires_grad, net.parameters()),
    #     lr=args.lr, weight_decay=1e-5
    # )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=5
    # )

    try:
        scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    except TypeError:  # Fallback for older versions
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)  # Add AMP flag to arguments
    optimizer = Lion(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=1e-5
    )

    # Change scheduler to cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,  # Match your dataset characteristics
        eta_min=1e-6
    )

    # #cosine annealing scheduler isn't validation sensitive so it can be changed later to
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='max',  # Track Dice (higher is better)
    #     factor=0.5,
    #     patience=5,
    #     verbose=True
    # )
    #
    # # Then in training loop:
    # scheduler.step(val_dice)  # Step with validation metric

    # ─── Checkpoint & TensorBoard Setup ─────────────────────────────────────┐
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(settings.LOG_DIR, args.exp_name, now)
    ckpt_dir = os.path.join(settings.CHECKPOINT_PATH, args.exp_name, now)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    # ─────────────────────────────────────────────────────────────────────────┘

    start_epoch = 1
    best_dice = 0.0
    early_stopper = EarlyStopper(patience=15)

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        print(f"Resuming training from epoch {start_epoch}")

    # ─── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, settings.EPOCH+1):
        # GPU monitoring
        if epoch % 5 == 0:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                writer.add_scalar(f"GPU/Usage_{i}", gpu.load * 100, epoch)
                writer.add_scalar(f"GPU/Memory_{i}", gpu.memoryUsed, epoch)
                print(f"GPU {i}: {gpu.load * 100:.1f}% load, "
                      f"{gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f} MB")

        # print(f"\n=== Epoch {epoch}/{settings.EPOCH} ===")

        # --- Training ---
        net.train()
        start = time.time()
        train_loss = function.train_sam(
            args, net, optimizer, scaler, train_loader, epoch, writer, vis=args.vis
        )
        print(f"[TRAIN]  Loss: {train_loss:.4f}  Time: {time.time()-start:.1f}s")

        # --- Validation ---
        net.eval()
        with torch.no_grad():
            val_tol, (val_iou, val_dice) = function.validation_sam(
                args, val_loader, epoch, net, writer
            )
        print(f"[VAL] IOU: {val_iou:.4f}  Dice: {val_dice:.4f}  Tol: {val_tol:.4f}")

        # Add TensorBoard image visualization
        if epoch % 10 == 0:
            # Get a validation batch
            sample_images, sample_masks, sample_paths = next(iter(val_loader))
            sample_images = sample_images.to(device)

            with torch.no_grad():
                sample_outputs = net(sample_images)

            # Log to TensorBoard
            writer.add_images("Val/Input", sample_images[:4], epoch)
            writer.add_images("Val/Prediction",
                              torch.sigmoid(sample_outputs[:4]) > 0.5, epoch)
            writer.add_images("Val/GroundTruth",
                              sample_masks[:4].unsqueeze(1), epoch)

        # Early stopping check
        if early_stopper(val_dice):
            print(f"Early stopping triggered at epoch {epoch}!")
            break

        # --- Scheduler Step ---
        # scheduler.step(train_loss)
        scheduler.step()  # CosineAnnealingWarmRestarts doesn't take a metric argument

        # --- Checkpointing on Best Dice ---
        if val_dice > best_dice:
            best_dice = val_dice
            ckpt_path = os.path.join(ckpt_dir, f"best_dice_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dice': best_dice,
            }, ckpt_path)
            print(f"[CHECKPOINT] Saved best model to {ckpt_path}")

    writer.close()
    print("Training complete.")

    # ─── Test Functionality ────────────────────────────────────────────────
    print("\nStarting testing...")
    test_results_dir = os.path.join(ckpt_dir, "test_results")
    os.makedirs(test_results_dir, exist_ok=True)

    net.eval()
    with torch.no_grad():
        for images, masks, img_paths in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = net(images)
            predictions = torch.sigmoid(outputs) > 0.5

            # Save results with original filenames
            for i in range(len(img_paths)):
                orig_name = Path(img_paths[i]).stem

                # Save prediction
                pred_path = os.path.join(test_results_dir, f"{orig_name}_pred.png")
                torchvision.utils.save_image(predictions[i].float(), pred_path)

                # Save overlay (optional)
                overlay = create_overlay(images[i], predictions[i])
                overlay_path = os.path.join(test_results_dir, f"{orig_name}_overlay.png")
                torchvision.utils.save_image(overlay, overlay_path)

    print(f"Test results saved to {test_results_dir}")

if __name__ == "__main__":

    main()
