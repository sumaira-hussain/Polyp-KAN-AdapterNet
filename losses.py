import torch
import torch.nn as nn
import torch.nn.functional as F

# Dice Loss for binary segmentation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: (B,1,H,W) logits or probabilities
        preds = torch.sigmoid(preds)
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (preds * targets).sum(1)
        dice = (2. * intersection + self.smooth) / (preds.sum(1) + targets.sum(1) + self.smooth)
        return 1 - dice.mean()

# Combined Loss
class ComboLoss(nn.Module):
    def __init__(self, weight_bce=1.0, weight_dice=1.0):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.wb, self.wd = weight_bce, weight_dice

    def forward(self, preds, targets):
        return self.wb*self.bce(preds, targets) + self.wd*self.dice(preds, targets)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        return self.alpha * (1 - pt) ** self.gamma * bce.mean()


class SurfaceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # Calculate boundary weights
        kernel = torch.tensor([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]], dtype=torch.float32, device=pred.device)
        kernel = kernel.view(1, 1, 3, 3)

        boundary = F.conv2d(target, kernel, padding=1)
        boundary = (boundary > 0) & (boundary < 9)  # Edge detection
        return F.binary_cross_entropy_with_logits(pred, target,
                                                  weight=boundary.float() + self.eps)