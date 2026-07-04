from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50
from torchvision.models._utils import IntermediateLayerGetter
from typing import Optional
import logging
logger = logging.getLogger(__name__)

class ASPPConv(nn.Sequential):
    """Convolution ASPP avec dilatation."""
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*modules)

class ASPPPooling(nn.Sequential):
    """Pooling global avec projection"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),  # bias=True car pas de BN
            nn.ReLU(inplace=True)

        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling."""
    
    def __init__(
        self,
        in_channels: int,
        atrous_rates: list = [6, 12, 18],
        out_channels: int = 256
    ):
        super().__init__()
        
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Convolutions atrous
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        
        # Global pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # Projection finale
        self.project = nn.Sequential(
            nn.Conv2d(
                len(self.convs) * out_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ avec ASPP et décodeur amélioré.
    
    Args:
        num_classes: Nombre de classes
        in_channels: Canaux d'entrée
        backbone: 'resnet50' ou 'resnet101'
        output_stride: 8 ou 16
        pretrained: Utiliser poids pré-entraînés
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        backbone: str = 'resnet101',
        output_stride: int = 16,
        pretrained: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Backbone
        if backbone == 'resnet101':
            logger.info("Création DeepLabV3+ avec ResNet101")
            base_model = deeplabv3_resnet101(
                weights=None,
                weights_backbone='DEFAULT' if pretrained else None
            )
        elif backbone == 'resnet50':
            logger.info("Création DeepLabV3+ avec ResNet50")
            base_model = deeplabv3_resnet50(
                weights=None,
                weights_backbone='DEFAULT' if pretrained else None
            )
        else:
            raise ValueError(f"Backbone non supporté: {backbone}")

        if output_stride != 8:

            logger.warning(
                f"output_stride={output_stride} demandé mais ignoré: le backbone "
                f"torchvision utilisé est toujours construit avec un stride "
                f"effectif de 8 pour la branche 'out'."
            )
        
       
        self.backbone = IntermediateLayerGetter(
            base_model.backbone,
            return_layers={'layer1': 'low_level', 'layer4': 'out'}
        )
        
        # Adapter premier conv si nécessaire
        if in_channels != 3:
            self._modify_first_conv(in_channels)
        
        # ASPP
        self.aspp = ASPP(2048, atrous_rates=[6, 12, 18], out_channels=256)
        
        # Décodeur low-level features
        self.decoder_low = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Décodeur final
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Tête de classification
        self.classifier = nn.Conv2d(
            256,
            num_classes if num_classes > 1 else 1,
            kernel_size=1
        )
        
        self._initialize_weights()
    
    def _modify_first_conv(self, in_channels: int):
        """Modifie la première convolution pour gérer un nombre différent de canaux."""
        conv1 = self.backbone.conv1
        
        new_conv1 = nn.Conv2d(
            in_channels,
            conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=False
        )
        
        with torch.no_grad():
            if in_channels == 1:
                # Moyenne des canaux RGB
                new_conv1.weight.copy_(conv1.weight.sum(dim=1, keepdim=True))
            else:
                # Initialisation Kaiming
                nn.init.kaiming_normal_(
                    new_conv1.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
        
        self.backbone.conv1 = new_conv1
        logger.info(f"Premier conv adapté pour {in_channels} canaux")
    
    def _initialize_weights(self):
        """Initialisation des poids du décodeur."""
        for m in [self.decoder, self.classifier]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        module.weight,
                        mode='fan_out',
                        nonlinearity='relu'
                    )
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        
        # Encoder
        features = self.backbone(x)
        
        # Low-level features (stride 4)
        low_level_feat = features['low_level']
        
        # High-level features (stride 16)
        x = features['out']
        
        # ASPP
        x = self.aspp(x)
        
        # Upsampling
        x = F.interpolate(
            x,
            size=low_level_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Fusion avec low-level features
        low_level_feat = self.decoder_low(low_level_feat)
        x = torch.cat([x, low_level_feat], dim=1)
        
        # Décodeur
        x = self.decoder(x)
        
        # Classification
        x = self.classifier(x)
        
        # Upsampling final
        x = F.interpolate(
            x,
            size=input_shape,
            mode='bilinear',
            align_corners=False
        )
        
        return x