import torch
import numpy as np

def compute_iou(pred, target, n_classes=3):
    ious = []
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)

    for cls in range(n_classes):
        y_true = np.multiply(target.eq(cls).detach().cpu().numpy(), 1)
        y_pred = np.multiply(pred.eq(cls).detach().cpu().numpy(), 1)

        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection

        iou = intersection / (union + 1e-6)
        ious.append(iou)

    return np.array(ious)