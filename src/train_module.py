from __future__ import annotations
from typing import Any, Dict
import torch
import lightning.pytorch as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from .models.unet import UNet
from .models.deeplabv3plus import DeepLabV3Plus
from .models.swin_unet import SwinUNet
from .losses import bce_dice_loss, tversky_loss, focal_loss
from .metrics import iou_score, dice_score, bfscore, apls_score

class SegLightningModule(pl.LightningModule):
    def __init__(self, model_name:str="unet", num_classes:int=2, in_channels:int=3, lr:float=3e-4, weight_decay:float=1e-4, scheduler:str="onecycle", loss:str="bce_dice"):
        super().__init__()
        self.save_hyperparameters()
        if model_name=="unet":
            self.net = UNet(encoder="resnet34", in_channels=in_channels, num_classes=num_classes)
        elif model_name=="deeplabv3plus":
            self.net = DeepLabV3Plus(num_classes=num_classes, in_channels=in_channels)
        elif model_name=="swin_unet":
            self.net = SwinUNet(in_channels=in_channels, num_classes=num_classes)
        else:
            raise ValueError(f"Modèle inconnu: {model_name}")

    def forward(self, x): return self.net(x)

    def _loss(self, logits, y):
        name = self.hparams.loss
        if name=="bce_dice":
            return bce_dice_loss(logits, y)
        if name=="tversky":
            return tversky_loss(logits, y)
        if name=="focal_dice":
            return focal_loss(logits, y) + 0.5*bce_dice_loss(logits, y)
        raise ValueError(name)

    def _shared_step(self, batch, stage:str):
        x,y = batch
        logits = self(x)
        loss = self._loss(logits, y)
        with torch.no_grad():
            miou = iou_score(logits, y, num_classes=self.hparams.num_classes)
            dsc  = dice_score(logits, y, num_classes=self.hparams.num_classes)
            bfs  = bfscore(logits, y)
            apls = apls_score((logits>0).float() if self.hparams.num_classes==2 else logits, y) if stage!="train" else torch.tensor(0.0, device=logits.device)
        self.log_dict({f"{stage}_loss": loss, f"{stage}_mIoU": miou, f"{stage}_Dice": dsc, f"{stage}_BF": bfs, f"{stage}_APLS": apls}, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx): return self._shared_step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._shared_step(batch, "val")
    def test_step(self, batch, batch_idx): return self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler == "onecycle":
            sch = OneCycleLR(opt, max_lr=self.hparams.lr, steps_per_epoch=self.trainer.datamodule.train_dataloader().__len__(), epochs=self.trainer.max_epochs)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}
        return {"optimizer": opt}
