# Ajouter au début
import logging
from typing import Optional, Dict
import json
from pathlib import Path
import numpy as np
import rasterio
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

class TileDataset(Dataset):
    """Dataset de tuiles GeoTIFF avec cache et stratification."""
    def __init__(
        self, 
        root: str, 
        split: str, 
        mean: list, 
        std: list, 
        augment: Dict, 
        num_classes: int,
        cache_size: int = 100,
        stratify: bool = False
    ):
        super().__init__()
        self.root = Path(root)
        self.mean, self.std = mean, std
        self.num_classes = num_classes
        self.cache = {}
        self.cache_size = cache_size
        
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest introuvable: {manifest_path}")
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        # save_tiles_manifest() écrit {'version', 'num_tiles', 'items': [...]},
        # pas une simple liste. On reste tolérant aux deux formats.
        items = manifest["items"] if isinstance(manifest, dict) else manifest
        
        # Les chemins du manifest sont relatifs à la racine du dataset
        # (cf. `img_out.relative_to(self.output)` dans les scripts
        # prepare_*.py), pas au répertoire courant du process.
        items = [
            it for it in items
            if (self.root / it['image']).exists() and (self.root / it['mask']).exists()
        ]
        logger.info(f"Dataset chargé: {len(items)} tuiles valides trouvées")
        
        # Split stratifié optionnel
        if stratify:
            items = self._stratified_split(items, split)
        else:
            items = sorted(items, key=lambda x: x["image"])
            n = len(items)
            if split == "train":
                self.items = items[:int(0.8*n)]
            elif split == "val":
                self.items = items[int(0.8*n):int(0.9*n)]
            else:
                self.items = items[int(0.9*n):]
        
        self.split = split
        self.tf = self.build_tf(augment if split == "train" else {})
        
        logger.info(f"Split '{split}': {len(self.items)} échantillons")
        self._analyze_dataset()
    
    def _stratified_split(self, items: list, split: str) -> list:
        """Split stratifié basé sur la distribution des classes."""
        # À implémenter selon vos besoins spécifiques
        return items
    
    def _analyze_dataset(self):
        """Analyse statistique du dataset."""
        if len(self.items) > 0:
            sample_img = self._read(self.items[0]['image'])
            logger.info(f"Dimensions tuiles: {sample_img.shape}")
    
    def build_tf(self, aug: Dict):
        """Construit la chaîne de transformations avec augmentations avancées."""
        transforms = []
        
        # Augmentations géométriques
        if aug.get("hflip", False): 
            transforms.append(A.HorizontalFlip(p=0.5))
        if aug.get("vflip", False): 
            transforms.append(A.VerticalFlip(p=0.5))
        if (r := aug.get("rotate", 0)) > 0: 
            transforms.append(A.Rotate(limit=r, p=0.5, border_mode=0))
        if aug.get("shift_scale", False):
            transforms.append(A.ShiftScaleRotate(
                shift_limit=0.0625, 
                scale_limit=0.1, 
                rotate_limit=0, 
                p=0.5
            ))
        
        # Augmentations photométriques
        if aug.get("brightness", 0) or aug.get("contrast", 0):
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=aug.get("brightness", 0),
                contrast_limit=aug.get("contrast", 0), 
                p=0.5
            ))
        
        if aug.get("blur", False):
            transforms.append(A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0)
            ], p=0.3))
        
        if aug.get("noise", False):
            transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.3))
        
        # Normalisation et conversion
        transforms += [
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ]
        
        return A.Compose(transforms)

    def _read(self, p: str, use_cache: bool = True) -> np.ndarray:
        """Lecture avec cache LRU. `p` est un chemin relatif à self.root
        (tel que stocké dans le manifest) ou un chemin déjà absolu."""
        if use_cache and p in self.cache:
            return self.cache[p]
        
        full_path = self.root / p
        with rasterio.open(full_path) as src:
            arr = src.read()
        
        if use_cache and len(self.cache) < self.cache_size:
            self.cache[p] = arr
        
        return arr

    def __len__(self) -> int: 
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple:
        it = self.items[idx]
        img = self._read(it["image"])[:3]  # RGB seulement
        mask = self._read(it["mask"])
        
        # Validation des données
        if img.shape[0] != 3:
            raise ValueError(f"Image doit avoir 3 canaux, trouvé: {img.shape[0]}")
        
        img = np.moveaxis(img, 0, -1)
        mask = mask[0]
        
        # Transformations
        data = self.tf(image=img, mask=mask)
        img_t = data["image"]
        mask_t = data["mask"].long()
        
        # Binarisation pour tâches binaires
        if self.num_classes == 2:
            mask_t = (mask_t > 0).long()
        
        return img_t, mask_t


class TilesDataModule(pl.LightningDataModule):
    """DataModule Lightning avec préchargement et validation."""
    def __init__(
        self, 
        root: str, 
        batch_size: int, 
        num_workers: int, 
        mean: list, 
        std: list, 
        augment: dict, 
        num_classes: int,
        prefetch_factor: int = 2,
        persistent_workers: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage: Optional[str] = None):
        """Initialise les datasets avec logging."""
        logger.info(f"Configuration DataModule - Stage: {stage}")
        
        self.train_ds = TileDataset(
            self.hparams.root, "train", 
            self.hparams.mean, self.hparams.std, 
            self.hparams.augment, self.hparams.num_classes
        )
        self.val_ds = TileDataset(
            self.hparams.root, "val",   
            self.hparams.mean, self.hparams.std, 
            {}, self.hparams.num_classes
        )
        self.test_ds = TileDataset(
            self.hparams.root, "test",  
            self.hparams.mean, self.hparams.std, 
            {}, self.hparams.num_classes
        )
        
        logger.info(f"Datasets créés: Train={len(self.train_ds)}, Val={len(self.val_ds)}, Test={len(self.test_ds)}")
    
    def _worker_kwargs(self) -> Dict:
        """prefetch_factor / persistent_workers ne sont valides que si
        num_workers > 0 (PyTorch lève une ValueError sinon)."""
        if self.hparams.num_workers > 0:
            return {
                'prefetch_factor': self.hparams.get('prefetch_factor', 2),
                'persistent_workers': self.hparams.get('persistent_workers', True),
            }
        return {}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, 
            batch_size=self.hparams.batch_size, 
            shuffle=True,  
            num_workers=self.hparams.num_workers, 
            pin_memory=True,
            drop_last=True,  # Pour stabiliser le batch norm
            **self._worker_kwargs()
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,   
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True,
            **self._worker_kwargs()
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,  
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True
        )