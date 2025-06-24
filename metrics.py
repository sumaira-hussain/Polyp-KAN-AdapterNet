import torch

def iou_score(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    union        = (preds + targets - preds * targets).sum()
    return (intersection + 1e-6) / (union + 1e-6)
