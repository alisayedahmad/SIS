import torch
from src.metrics import iou_score, dice_score, bfscore, apls_score

def test_metrics_shapes():
    logits = torch.randn(2,2,64,64)
    target = torch.randint(0,2,(2,64,64))
    iou = iou_score(logits, target, num_classes=2)
    dice = dice_score(logits, target, num_classes=2)
    bf = bfscore(logits, target)
    apls = apls_score(logits, target)
    for t in [iou, dice, bf, apls]:
        assert torch.is_tensor(t)
