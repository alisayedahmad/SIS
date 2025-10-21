from __future__ import annotations
import torch
import torch.nn.functional as F

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits) if logits.shape[1] == 1 else torch.softmax(logits, dim=1)
    if logits.shape[1] == 1:
        probs = probs.squeeze(1)
        targets = targets.float()
        inter = (probs*targets).sum(dim=(-1,-2))
        denom = probs.sum(dim=(-1,-2)) + targets.sum(dim=(-1,-2))
        dice = (2*inter + eps)/(denom + eps)
        return 1 - dice.mean()
    else:
        # one-hot
        n,c,h,w = logits.shape
        targets_oh = F.one_hot(targets, num_classes=c).permute(0,3,1,2).float()
        inter = (probs*targets_oh).sum(dim=(-1,-2))
        denom = probs.sum(dim=(-1,-2)) + targets_oh.sum(dim=(-1,-2))
        dice = (2*inter + eps)/(denom + eps)
        return 1 - dice.mean()

def bce_dice_loss(logits, targets):
    if logits.shape[1] == 1:
        bce = F.binary_cross_entropy_with_logits(logits.squeeze(1), targets.float())
    else:
        bce = F.cross_entropy(logits, targets)
    return bce + dice_loss(logits, targets)

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    if logits.shape[1] == 1:
        p = torch.sigmoid(logits).squeeze(1)
        t = targets.float()
        ce = F.binary_cross_entropy(p, t, reduction='none')
        p_t = p*t + (1-p)*(1-t)
        loss = (alpha*(1-p_t)**gamma)*ce
        return loss.mean()
    else:
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = (alpha*(1-pt)**gamma)*ce
        return loss.mean()

def tversky_loss(logits, targets, alpha=0.5, beta=0.5, eps=1e-6):
    if logits.shape[1] == 1:
        p = torch.sigmoid(logits).squeeze(1)
        t = targets.float()
        tp = (p*t).sum(dim=(-1,-2))
        fp = (p*(1-t)).sum(dim=(-1,-2))
        fn = ((1-p)*t).sum(dim=(-1,-2))
        tv = (tp + eps)/(tp + alpha*fp + beta*fn + eps)
        return 1 - tv.mean()
    else:
        probs = torch.softmax(logits, dim=1)
        n,c,h,w = logits.shape
        targets_oh = F.one_hot(targets, num_classes=c).permute(0,3,1,2).float()
        tp = (probs*targets_oh).sum(dim=(-1,-2))
        fp = (probs*(1-targets_oh)).sum(dim=(-1,-2))
        fn = ((1-probs)*targets_oh).sum(dim=(-1,-2))
        tv = (tp + eps)/(tp + alpha*fp + beta*fn + eps)
        return 1 - tv.mean()
