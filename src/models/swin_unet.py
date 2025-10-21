from __future__ import annotations
import torch
import torch.nn as nn
import timm

class FPN(nn.Module):
    def __init__(self, in_channels, out_ch=128):
        super().__init__()
        self.lats = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_channels])
        self.smooth = nn.ModuleList([nn.Conv2d(out_ch, out_ch, 3, padding=1) for _ in in_channels])
    def forward(self, feats):
        # feats: [C1..C4], low->high resolution
        c2,c3,c4,c5 = feats
        p5 = self.lats[3](c5)
        p4 = self.lats[2](c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lats[1](c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lats[0](c2) + nn.functional.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = self.smooth[0](p2); p3 = self.smooth[1](p3); p4 = self.smooth[2](p4); p5 = self.smooth[3](p5)
        return p2,p3,p4,p5

class SwinUNet(nn.Module):
    """Swin-UNet léger: backbone Swin + FPN + tête simple."""
    def __init__(self, in_channels=3, num_classes=2, backbone='swin_tiny_patch4_window7_224'):
        super().__init__()
        self.backbone = timm.create_model(backbone, features_only=True, in_chans=in_channels, pretrained=True)
        chs = self.backbone.feature_info.channels()  # [c2,c3,c4,c5]
        self.fpn = FPN(chs, out_ch=128)
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes if num_classes>1 else 1, 1)
        )
    def forward(self, x):
        feats = self.backbone(x)
        p2,_,_,_ = self.fpn(feats)
        x = nn.functional.interpolate(p2, scale_factor=4, mode="bilinear", align_corners=False)
        return self.head(x)
