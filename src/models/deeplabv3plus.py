from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=2, in_channels=3):
        super().__init__()
        self.model = deeplabv3_resnet101(weights=None, num_classes=num_classes, aux_loss=None)
        if in_channels != 3:
            # Remplace la première conv si besoin
            conv1 = self.model.backbone.conv1
            new = nn.Conv2d(in_channels, conv1.out_channels, kernel_size=conv1.kernel_size, stride=conv1.stride, padding=conv1.padding, bias=False)
            with torch.no_grad():
                if in_channels==1:
                    new.weight.copy_(conv1.weight.sum(dim=1, keepdim=True))
                else:
                    nn.init.kaiming_normal_(new.weight, mode='fan_out', nonlinearity='relu')
            self.model.backbone.conv1 = new

    def forward(self, x):
        out = self.model(x)["out"]
        return out
