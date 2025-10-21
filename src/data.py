from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import os, json, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio
import lightning.pytorch as pl

class TileDataset(Dataset):
    """Dataset de tuiles GeoTIFF (images/masks)."""
    def __init__(self, root:str, split:str, mean, std, augment:Dict, num_classes:int):
        super().__init__()
        self.root = root
        self.mean, self.std = mean, std
        self.num_classes = num_classes
        with open(os.path.join(root, "manifest.json"), "r", encoding="utf-8") as f:
            items = json.load(f)
        # split simple par ratio
        items = sorted(items, key=lambda x: x["image"])
        n = len(items)
        if split == "train":
            self.items = items[: int(0.8*n)]
        elif split == "val":
            self.items = items[int(0.8*n): int(0.9*n)]
        else:
            self.items = items[int(0.9*n):]
        self.tf = self.build_tf(augment)

    def build_tf(self, aug:Dict):
        transforms = []
        if aug.get("hflip", False): transforms.append(A.HorizontalFlip(p=0.5))
        if aug.get("vflip", False): transforms.append(A.VerticalFlip(p=0.5))
        if (r := aug.get("rotate", 0))>0: transforms.append(A.Rotate(limit=r, p=0.5, border_mode=0))
        if aug.get("brightness", 0) or aug.get("contrast", 0):
            transforms.append(A.RandomBrightnessContrast(brightness_limit=aug.get("brightness",0),
                                                         contrast_limit=aug.get("contrast",0), p=0.5))
        transforms += [A.Normalize(mean=self.mean, std=self.std), ToTensorV2()]
        return A.Compose(transforms)

    def __len__(self): return len(self.items)

    def _read(self, p):
        with rasterio.open(p) as src:
            arr = src.read()
        return arr

    def __getitem__(self, idx):
        it = self.items[idx]
        img = self._read(it["image"])[:3]  # RGB
        mask = self._read(it["mask"])  # (1,H,W)
        img = np.moveaxis(img, 0, -1)
        mask = mask[0]
        data = self.tf(image=img, mask=mask)
        img_t = data["image"]
        mask_t = data["mask"].long()
        if self.num_classes == 2:
            mask_t = (mask_t > 0).long()
        return img_t, mask_t

class TilesDataModule(pl.LightningDataModule):
    def __init__(self, root:str, batch_size:int, num_workers:int, mean, std, augment, num_classes:int):
        super().__init__()
        self.save_hyperparameters()
    def setup(self, stage=None):
        self.train_ds = TileDataset(self.hparams.root, "train", self.hparams.mean, self.hparams.std, self.hparams.augment, self.hparams.num_classes)
        self.val_ds   = TileDataset(self.hparams.root, "val",   self.hparams.mean, self.hparams.std, self.hparams.augment, self.hparams.num_classes)
        self.test_ds  = TileDataset(self.hparams.root, "test",  self.hparams.mean, self.hparams.std, self.hparams.augment, self.hparams.num_classes)
    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True,  num_workers=self.hparams.num_workers, pin_memory=True)
    def val_dataloader(self):   return DataLoader(self.val_ds,   batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)
    def test_dataloader(self):  return DataLoader(self.test_ds,  batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)
