from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class BCELoss(nn.Module):
    """Binary cross-entropy avec support multi-classes (one-vs-rest).

    Comme DiceLoss/FocalLoss/TverskyLoss, gère deux conventions:
    - logits à 1 canal: BCE binaire classique (sigmoid).
    - logits à C canaux (C>1) avec cibles d'index de classe (B,H,W): BCE
      "one-vs-rest" par canal sur les cibles one-hot. Nécessaire car les
      têtes UNet/DeepLabV3+ de ce dépôt produisent toujours `num_classes`
      canaux (y compris pour les tâches binaires, num_classes=2), ce que
      nn.BCEWithLogitsLoss brut ne peut pas consommer directement.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            return self.bce(logits.squeeze(1), targets.float())
        c = logits.shape[1]
        targets_oh = F.one_hot(targets, num_classes=c).permute(0, 3, 1, 2).float()
        return self.bce(logits, targets_oh)


class DiceLoss(nn.Module):
    """Dice Loss avec support multi-classes et poids de classes."""
    def __init__(self, smooth: float = 1e-6, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            # Binaire
            probs = torch.sigmoid(logits).squeeze(1)
            targets_f = targets.float()
            inter = (probs * targets_f).sum(dim=(-1, -2))
            denom = probs.sum(dim=(-1, -2)) + targets_f.sum(dim=(-1, -2))
            dice = (2 * inter + self.smooth) / (denom + self.smooth)
            return 1 - dice.mean()
        else:
            # Multi-classes
            probs = torch.softmax(logits, dim=1)
            n, c, h, w = logits.shape
            targets_oh = F.one_hot(targets, num_classes=c).permute(0, 3, 1, 2).float()
            
            inter = (probs * targets_oh).sum(dim=(-1, -2))
            denom = probs.sum(dim=(-1, -2)) + targets_oh.sum(dim=(-1, -2))
            dice = (2 * inter + self.smooth) / (denom + self.smooth)
            
            if self.class_weights is not None:
                dice = dice * self.class_weights.to(dice.device)
            
            return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss pour gérer le déséquilibre de classes."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            # Binaire
            p = torch.sigmoid(logits).squeeze(1)
            t = targets.float()
            ce = F.binary_cross_entropy(p, t, reduction='none')
            p_t = p * t + (1 - p) * (1 - t)
            loss = self.alpha * (1 - p_t) ** self.gamma * ce
        else:
            # Multi-classes
            ce = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
            loss = self.alpha * (1 - pt) ** self.gamma * ce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class TverskyLoss(nn.Module):
    """Tversky Loss - généralisation de Dice Loss."""
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            p = torch.sigmoid(logits).squeeze(1)
            t = targets.float()
            tp = (p * t).sum(dim=(-1, -2))
            fp = (p * (1 - t)).sum(dim=(-1, -2))
            fn = ((1 - p) * t).sum(dim=(-1, -2))
            tv = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            return 1 - tv.mean()
        else:
            probs = torch.softmax(logits, dim=1)
            n, c, h, w = logits.shape
            targets_oh = F.one_hot(targets, num_classes=c).permute(0, 3, 1, 2).float()
            
            tp = (probs * targets_oh).sum(dim=(-1, -2))
            fp = (probs * (1 - targets_oh)).sum(dim=(-1, -2))
            fn = ((1 - probs) * targets_oh).sum(dim=(-1, -2))
            tv = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            return 1 - tv.mean()


class ComboLoss(nn.Module):
    """Combinaison pondérée de plusieurs pertes."""
    def __init__(self, losses: dict):
        super().__init__()
        # nn.ModuleDict ne peut contenir que des nn.Module: on sépare les
        # fonctions de perte (modules) de leurs poids scalaires (floats).
        self.loss_fns = nn.ModuleDict({name: fn for name, (fn, _) in losses.items()})
        self.weights = {name: float(weight) for name, (_, weight) in losses.items()}
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple:
        total_loss = 0
        loss_dict = {}
        
        for name, loss_fn in self.loss_fns.items():
            loss = loss_fn(logits, targets)
            total_loss = total_loss + self.weights[name] * loss
            loss_dict[name] = loss.item()
        
        return total_loss, loss_dict


def get_loss_function(name: str, **kwargs):
    """Factory pour créer les fonctions de perte."""
    losses = {
        'bce': lambda: BCELoss(**kwargs),
        'ce': lambda: nn.CrossEntropyLoss(**kwargs),
        'dice': lambda: DiceLoss(**kwargs),
        'focal': lambda: FocalLoss(**kwargs),
        'tversky': lambda: TverskyLoss(**kwargs),
        'bce_dice': lambda: ComboLoss({
            'bce': (BCELoss(), 0.5),
            'dice': (DiceLoss(), 0.5)
        }),
        'focal_dice': lambda: ComboLoss({
            'focal': (FocalLoss(), 0.5),
            'dice': (DiceLoss(), 0.5)
        })
    }
    
    if name not in losses:
        raise ValueError(f"Perte inconnue: {name}. Disponibles: {list(losses.keys())}")
    
    return losses[name]()