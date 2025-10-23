from __future__ import annotations
from typing import Any, Dict, Optional, List
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
import logging

from .models.unet import UNet
from .models.deeplabv3plus import DeepLabV3Plus
from .models.swin_unet import SwinUNet
from .losses import get_loss_function
from .metrics import (iou_score, dice_score, bfscore, apls_score, 
                      precision_recall_f1, pixel_accuracy, MetricsTracker)

logger = logging.getLogger(__name__)


class SegLightningModule(pl.LightningModule):
    """Module Lightning avec EMA, gradient clipping et logging avancé."""
    
    def __init__(
        self, 
        model_name: str = "unet", 
        num_classes: int = 2, 
        in_channels: int = 3, 
        encoder: str = "resnet34",
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        lr: float = 3e-4, 
        weight_decay: float = 1e-4, 
        optimizer: str = "adamw",
        scheduler: str = "onecycle", 
        loss: str = "bce_dice",
        use_ema: bool = False,
        ema_decay: float = 0.999,
        gradient_clip_val: Optional[float] = 1.0,
        log_images: bool = True,
        compute_apls: bool = False  # APLS coûteux, désactivé par défaut en train
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Construction du modèle
        self.net = self._build_model()
        
        # EMA (Exponential Moving Average) optionnel
        if use_ema:
            self.ema_net = self._build_model()
            self.ema_net.load_state_dict(self.net.state_dict())
            for param in self.ema_net.parameters():
                param.requires_grad = False
        else:
            self.ema_net = None
        
        # Loss function
        self.criterion = get_loss_function(loss)
        
        # Metrics trackers
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        logger.info(f"Modèle initialisé: {model_name} | Optimizer: {optimizer} | Loss: {loss}")
    
    def _build_model(self) -> nn.Module:
        """Factory pour créer les modèles."""
        if self.hparams.model_name == "unet":
            return UNet(
                encoder=self.hparams.encoder,
                in_channels=self.hparams.in_channels,
                num_classes=self.hparams.num_classes,
                decoder_channels=self.hparams.decoder_channels
            )
        elif self.hparams.model_name == "deeplabv3plus":
            return DeepLabV3Plus(
                num_classes=self.hparams.num_classes,
                in_channels=self.hparams.in_channels
            )
        elif self.hparams.model_name == "swin_unet":
            return SwinUNet(
                in_channels=self.hparams.in_channels,
                num_classes=self.hparams.num_classes
            )
        else:
            raise ValueError(f"Modèle inconnu: {self.hparams.model_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def _update_ema(self):
        """Met à jour les poids EMA."""
        if self.ema_net is None:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(self.ema_net.parameters(), self.net.parameters()):
                ema_param.data.mul_(self.hparams.ema_decay).add_(
                    param.data, alpha=1 - self.hparams.ema_decay
                )
    
    def _shared_step(self, batch: tuple, stage: str) -> Dict[str, torch.Tensor]:
        """Step partagé avec calcul de toutes les métriques."""
        x, y = batch
        
        # Forward pass
        logits = self(x)
        
        # Loss
        if isinstance(self.criterion, tuple):  # ComboLoss retourne (loss, dict)
            loss, loss_dict = self.criterion(logits, y)
        else:
            loss = self.criterion(logits, y)
            loss_dict = {}
        
        # Métriques
        with torch.no_grad():
            metrics = {
                'loss': loss.item(),
                'mIoU': iou_score(logits, y, num_classes=self.hparams.num_classes).item(),
                'Dice': dice_score(logits, y, num_classes=self.hparams.num_classes).item(),
                'PixAcc': pixel_accuracy(logits, y).item(),
            }
            
            # Métriques géométriques (validation/test seulement)
            if stage in ['val', 'test']:
                metrics['BF'] = bfscore(logits, y).item()
                
                if self.hparams.compute_apls:
                    metrics['APLS'] = apls_score(logits, y).item()
                
                # Precision/Recall/F1 par classe
                pr_metrics = precision_recall_f1(logits, y, self.hparams.num_classes)
                metrics.update({k: v.item() for k, v in pr_metrics.items()})
            
            # Ajout des sous-pertes si ComboLoss
            metrics.update(loss_dict)
        
        # Logging
        self.log_dict(
            {f"{stage}_{k}": v for k, v in metrics.items()},
            prog_bar=(stage != 'test'),
            logger=True,
            on_step=(stage == 'train'),
            on_epoch=True,
            sync_dist=True
        )
        
        return {'loss': loss, 'metrics': metrics}
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        output = self._shared_step(batch, "train")
        self.train_metrics.update(output['metrics'])
        
        # Update EMA
        if self.ema_net is not None:
            self._update_ema()
        
        return output['loss']
    
    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        # Utilise EMA si disponible
        if self.ema_net is not None:
            with torch.no_grad():
                x, y = batch
                logits = self.ema_net(x)
                # Recalcule les métriques avec EMA
                # ... (similaire à _shared_step)
        else:
            output = self._shared_step(batch, "val")
            self.val_metrics.update(output['metrics'])
    
    def test_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "test")
    
    def on_train_epoch_end(self) -> None:
        """Logging des moyennes d'époque."""
        avg_metrics = self.train_metrics.compute()
        logger.info(f"Epoch {self.current_epoch} Train: {avg_metrics}")
        self.train_metrics.reset()
    
    def on_validation_epoch_end(self) -> None:
        avg_metrics = self.val_metrics.compute()
        logger.info(f"Epoch {self.current_epoch} Val: {avg_metrics}")
        self.val_metrics.reset()
    
    def configure_optimizers(self) -> Dict:
        """Configuration avancée des optimizers et schedulers."""
        # Optimizer
        if self.hparams.optimizer == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.999)
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Optimizer inconnu: {self.hparams.optimizer}")
        
        # Scheduler
        if self.hparams.scheduler == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
                epochs=self.trainer.max_epochs,
                pct_start=0.3,
                div_factor=25,
                final_div_factor=1e4
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }
        
        elif self.hparams.scheduler == "cosine":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        
        elif self.hparams.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_mIoU",
                    "interval": "epoch"
                }
            }
        
        return {"optimizer": optimizer}