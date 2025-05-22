import torch
import torch.nn as nn

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
