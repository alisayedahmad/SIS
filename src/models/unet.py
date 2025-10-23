from __future__ import annotations
import torch
import torch.nn as nn
import timm
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Bloc de convolution avec BatchNorm et activation."""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_batchnorm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=not use_batchnorm)
        ]
        
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        
        layers.append(nn.ReLU(inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        layers.append(
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding, bias=not use_batchnorm)
        )
        
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        
        layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate pour améliorer la fusion des features."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet(nn.Module):
    """
    U-Net avec encodeur pré-entraîné (TIMM) et attention gates optionnels.
    
    Args:
        encoder: Nom de l'encodeur TIMM (resnet34, resnet50, efficientnet_b0, etc.)
        in_channels: Nombre de canaux d'entrée
        num_classes: Nombre de classes de sortie
        decoder_channels: Canaux pour chaque niveau du décodeur
        use_attention: Activer les attention gates
        dropout: Dropout rate dans les blocs de convolution
        pretrained: Utiliser les poids pré-entraînés ImageNet
    """
    
    def __init__(
        self,
        encoder: str = 'resnet34',
        in_channels: int = 3,
        num_classes: int = 2,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        use_attention: bool = False,
        dropout: float = 0.0,
        pretrained: bool = True
    ):
        super().__init__()
        
        self.encoder_name = encoder
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Encodeur TIMM
        logger.info(f"Création encodeur: {encoder} (pretrained={pretrained})")
        self.encoder = timm.create_model(
            encoder,
            features_only=True,
            in_chans=in_channels,
            pretrained=pretrained
        )
        
        # Canaux de l'encodeur
        enc_channels = self.encoder.feature_info.channels()
        logger.info(f"Canaux encodeur: {enc_channels}")
        
        # Centre (bottleneck)
        self.center = ConvBlock(
            enc_channels[-1],
            decoder_channels[0],
            dropout=dropout
        )
        
        # Attention gates
        if use_attention:
            self.att4 = AttentionGate(decoder_channels[0], enc_channels[-2], decoder_channels[1])
            self.att3 = AttentionGate(decoder_channels[1], enc_channels[-3], decoder_channels[2])
            self.att2 = AttentionGate(decoder_channels[2], enc_channels[-4], decoder_channels[3])
            self.att1 = AttentionGate(decoder_channels[3], enc_channels[-5], decoder_channels[4])
        
        # Décodeur
        self.dec4 = ConvBlock(
            decoder_channels[0] + enc_channels[-2],
            decoder_channels[1],
            dropout=dropout
        )
        self.dec3 = ConvBlock(
            decoder_channels[1] + enc_channels[-3],
            decoder_channels[2],
            dropout=dropout
        )
        self.dec2 = ConvBlock(
            decoder_channels[2] + enc_channels[-4],
            decoder_channels[3],
            dropout=dropout
        )
        self.dec1 = ConvBlock(
            decoder_channels[3] + enc_channels[-5],
            decoder_channels[4],
            dropout=dropout
        )
        
        # Tête de classification
        self.head = nn.Conv2d(
            decoder_channels[4],
            num_classes if num_classes > 1 else 1,
            kernel_size=1
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialisation Xavier pour le décodeur."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encodage
        feats = self.encoder(x)  # [C1, C2, C3, C4, C5]
        
        # Bottleneck
        x = self.center(feats[-1])
        
        # Décodage avec skip connections
        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        skip4 = self.att4(x, feats[-2]) if self.use_attention else feats[-2]
        x = self.dec4(torch.cat([x, skip4], dim=1))
        
        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        skip3 = self.att3(x, feats[-3]) if self.use_attention else feats[-3]
        x = self.dec3(torch.cat([x, skip3], dim=1))
        
        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        skip2 = self.att2(x, feats[-4]) if self.use_attention else feats[-4]
        x = self.dec2(torch.cat([x, skip2], dim=1))
        
        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        skip1 = self.att1(x, feats[-5]) if self.use_attention else feats[-5]
        x = self.dec1(torch.cat([x, skip1], dim=1))
        
        # Tête
        x = self.head(x)
        
        return x