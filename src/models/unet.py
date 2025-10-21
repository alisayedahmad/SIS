from __future__ import annotations
import torch
import torch.nn as nn
import timm

class ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    """U-Net avec encodeur TIMM (ResNet34/50)."""
    def __init__(self, encoder='resnet34', in_channels=3, num_classes=2, decoder_channels=(256,128,64,32,16)):
        super().__init__()
        self.encoder = timm.create_model(encoder, features_only=True, in_chans=in_channels, pretrained=True)
        enc_channels = self.encoder.feature_info.channels()  # [c1..c5]
        self.center = ConvRelu(enc_channels[-1], decoder_channels[0])
        self.dec4 = ConvRelu(decoder_channels[0]+enc_channels[-2], decoder_channels[1])
        self.dec3 = ConvRelu(decoder_channels[1]+enc_channels[-3], decoder_channels[2])
        self.dec2 = ConvRelu(decoder_channels[2]+enc_channels[-4], decoder_channels[3])
        self.dec1 = ConvRelu(decoder_channels[3]+enc_channels[-5], decoder_channels[4])
        self.head = nn.Conv2d(decoder_channels[4], num_classes if num_classes>1 else 1, 1)

    def forward(self, x):
        feats = self.encoder(x)  # list 5 scales
        x = self.center(feats[-1])
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec4(torch.cat([x, feats[-2]], dim=1))
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, feats[-3]], dim=1))
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, feats[-4]], dim=1))
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, feats[-5]], dim=1))
        return self.head(x)
