from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List
import logging

logger = logging.getLogger(__name__)


class PatchExpanding(nn.Module):
    """Expansion de patches pour upsampling."""
    
    def __init__(self, input_resolution: tuple, dim: int, dim_scale: int = 2):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        
        assert L == H * W, "Input feature has wrong size"
        
        x = self.expand(x)
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 2)
        x = x.view(B, -1, C // 2)
        x = self.norm(x)
        
        return x


class FPN(nn.Module):
    """Feature Pyramid Network pour fusion multi-échelle."""
    
    def __init__(self, in_channels: List[int], out_ch: int = 128):
        super().__init__()
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_ch, kernel_size=1, bias=False)
            for c in in_channels
        ])
        
        # Top-down pathway
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            for _ in in_channels
        ])
    
    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        # Bottom-up
        laterals = [conv(f) for f, conv in zip(feats, self.lateral_convs)]
        
        # Top-down
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )
        
        # Smooth
        outputs = [conv(lat) for lat, conv in zip(laterals, self.fpn_convs)]
        
        return outputs


class SwinUNet(nn.Module):
    """
    Swin Transformer U-Net avec FPN.
    
    Args:
        in_channels: Canaux d'entrée
        num_classes: Nombre de classes
        backbone: Nom du modèle Swin (swin_tiny, swin_small, swin_base)
        pretrained: Utiliser poids pré-entraînés
        fpn_channels: Canaux FPN
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        backbone: str = 'swin_tiny_patch4_window7_224',
        pretrained: bool = True,
        fpn_channels: int = 128
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        logger.info(f"Création Swin-UNet avec {backbone} (pretrained={pretrained})")
        
        # Backbone Swin
        self.backbone = timm.create_model(
            backbone,
            features_only=True,
            in_chans=in_channels,
            pretrained=pretrained
        )
        
        # Canaux des features
        feat_channels = self.backbone.feature_info.channels()
        logger.info(f"Canaux features: {feat_channels}")
        
        # FPN
        self.fpn = FPN(feat_channels, out_ch=fpn_channels)
        
        # Décodeur
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels // 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels // 2),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
            )
            for _ in range(len(feat_channels))
        ])
        
        # Tête de segmentation
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(fpn_channels // 2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes if num_classes > 1 else 1, kernel_size=1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialisation des poids."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        
        # Encoder
        features = self.backbone(x)
        
        # FPN
        fpn_features = self.fpn(features)
        
        # Prendre la feature la plus haute résolution
        x = fpn_features[0]
        
        # Décodeur
        x = self.decoder[0](x)
        
        # Upsampling à la résolution d'entrée
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        # Segmentation head
        x = self.segmentation_head(x)
        
        return x