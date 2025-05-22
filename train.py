#!/usr/bin/env python3
# train.py

import os
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import cfg               # your arg parser
import function          # with train_sam, validation_sam
from conf import settings
from models.polyp_kan import UKAN  # or PolypKANNet
from datasets.kvasir import KvasirSegDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2




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
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    val_transforms = A.Compose([
        A.Resize(352, 352),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])
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

    train_ds = KvasirSegDataset(
        "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/images/train", "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/masks/train",
        transforms=train_transforms
    )
    val_ds = KvasirSegDataset(
        "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/images/val", "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/masks/val",
        transforms=val_transforms
    )
    test_ds = KvasirSegDataset(
        "D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/images/test","D:/2025-research/Polyp-KAN-AdapterNet/datasets/kvasir/masks/test",
        transforms=val_transforms
    )
    train_loader = DataLoader(train_ds, batch_size=args.b, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.b//2, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.b// 2, shuffle=False, num_workers=2)

    # ─── Model Instantiation ───────────────────────────────────────────────────
    net = UKAN(
        num_classes=1,
        input_channels=3,
        deep_supervision=False,
        img_size=352,
        patch_size=16,
        embed_dims=[256,320,512],
        no_kan=False
    )
    net.to(device)

    # ─── Load Pretrained Weights (if any) ─────────────────────────────────────
    if args.pretrain:
        print(f"[INFO] Loading pretrained weights from {args.pretrain}")
        state = torch.load(args.pretrain, map_location=device)
        net.load_state_dict(state, strict=False)

    # ─── Freeze All Except Adapters & KAN Layers ───────────────────────────────
    for name, param in net.named_parameters():
        if "adapt" not in name.lower() and "kan" not in name.lower():
            param.requires_grad = False
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
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.lr, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ─── Checkpoint & TensorBoard Setup ─────────────────────────────────────┐
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(settings.LOG_DIR, args.exp_name, now)
    ckpt_dir = os.path.join(settings.CHECKPOINT_PATH, args.exp_name, now)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    # ─────────────────────────────────────────────────────────────────────────┘

    best_dice = 0.0

    # ─── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(1, settings.EPOCH+1):
        print(f"\n=== Epoch {epoch}/{settings.EPOCH} ===")

        # --- Training ---
        net.train()
        start = time.time()
        train_loss = function.train_sam(
            args, net, optimizer, train_loader, epoch, writer, vis=args.vis
        )
        print(f"[TRAIN]  Loss: {train_loss:.4f}  Time: {time.time()-start:.1f}s")

        # --- Validation ---
        net.eval()
        with torch.no_grad():
            val_tol, (val_iou, val_dice) = function.validation_sam(
                args, val_loader, epoch, net, writer
            )
        print(f"[VAL] IOU: {val_iou:.4f}  Dice: {val_dice:.4f}  Tol: {val_tol:.4f}")

        # --- Scheduler Step ---
        scheduler.step(train_loss)

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

if __name__ == "__main__":

    main()
