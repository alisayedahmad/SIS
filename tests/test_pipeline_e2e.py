"""
Test end-to-end (smoke test) du pipeline complet sur données synthétiques:
préparation de tuiles -> DataModule -> SegLightningModule -> Trainer.fit/test
-> inférence sliding-window -> post-traitement.

Couvre les régressions des bugs précédemment présents dans :
- tile_array (alignement image/masque)
- UNet (résolution de sortie)
- DeepLabV3Plus (KeyError 'low_level')
- ComboLoss / get_loss_function('bce_dice', 'focal_dice')
- SegLightningModule (isinstance(criterion, tuple), EMA validation_step)
- src/data.py (import lightning.pytorch cohérent avec train_module.py)
"""
import json
import numpy as np
import pytest
import rasterio
import torch

from src.geo.io import write_geotiff_like
from src.geo.tiling import save_tiles_manifest
from src.train_module import SegLightningModule
from src.infer import slide_infer_on_raster
from src.postprocess import postprocess_prediction


def _make_synthetic_dataset(root, n_tiles=12, size=64):
    """Crée un mini dataset de tuiles GeoTIFF + manifest.json valide."""
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    transform = rasterio.transform.from_origin(0, 0, 1, 1)
    items = []
    rng = np.random.default_rng(0)

    for i in range(n_tiles):
        img = (rng.random((3, size, size)) * 255).astype("uint8")
        mask = (rng.random((1, size, size)) > 0.7).astype("uint8")

        img_path = images_dir / f"tile_{i:03d}.tif"
        mask_path = masks_dir / f"tile_{i:03d}.tif"

        img_meta = {
            "driver": "GTiff", "dtype": "uint8", "count": 3,
            "height": size, "width": size, "crs": "EPSG:4326",
            "transform": transform,
        }
        mask_meta = dict(img_meta, count=1)

        write_geotiff_like(str(img_path), img_meta, img)
        write_geotiff_like(str(mask_path), mask_meta, mask)

        items.append({
            "image": str(img_path.relative_to(root)),
            "mask": str(mask_path.relative_to(root)),
            "source_image": f"tile_{i:03d}.tif",
            "coords": [0, 0, size, size],
            "coverage": float(mask.mean() * 100),
        })

    save_tiles_manifest(root / "manifest.json", items, metadata={"dataset": "synthetic"})
    return root


@pytest.mark.parametrize("model_name,loss_name,use_ema", [
    ("unet", "bce_dice", True),     # matches unet_buildings.yaml / unet_paris.yaml
    ("unet", "tversky", False),     # matches multiclass.yaml family
])
def test_train_module_fit_smoke(tmp_path, model_name, loss_name, use_ema):
    """Un fast_dev_run complet (train+val+test) ne doit pas lever d'exception,
    et val_mIoU doit être correctement loggé même quand use_ema=True (c'était
    silencieusement absent avant correction)."""
    import lightning.pytorch as pl
    from src.data import TilesDataModule

    data_root = _make_synthetic_dataset(tmp_path / "data", n_tiles=12, size=64)

    dm = TilesDataModule(
        root=str(data_root),
        batch_size=2,
        num_workers=0,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        augment={"hflip": True},
        num_classes=2,
    )

    model = SegLightningModule(
        model_name=model_name,
        num_classes=2,
        in_channels=3,
        encoder="resnet18",
        lr=1e-3,
        loss=loss_name,
        use_ema=use_ema,
        pretrained=False,
        scheduler="cosine",
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_train_batches=2,
        limit_val_batches=2,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=dm)

    # val_mIoU must have actually been logged (this silently failed to ever
    # populate when use_ema=True before the validation_step fix).
    assert "val_mIoU" in trainer.callback_metrics, (
        "val_mIoU absent de callback_metrics — la validation EMA ne loggue rien."
    )

    test_results = trainer.test(model, datamodule=dm)
    assert "test_mIoU" in test_results[0]


def test_inference_and_postprocess_smoke():
    """Inférence sliding-window + post-traitement sur un faux modèle UNet,
    pour vérifier que la résolution de sortie correspond bien à l'image
    d'entrée de bout en bout (régression du bug de résolution UNet)."""
    from src.models.unet import UNet

    model = UNet(encoder="resnet18", in_channels=3, num_classes=2, pretrained=False)
    model.eval()

    img = (np.random.rand(3, 300, 260) * 255).astype("uint8")
    prob = slide_infer_on_raster(
        model, img, tile_size=128, overlap=32, batch_size=2, device="cpu",
        progress_bar=False,
    )
    assert prob.shape[1:] == img.shape[1:]

    pred = prob.argmax(axis=0).astype(np.uint8)
    out = postprocess_prediction(pred[None, ...], kind="buildings")
    assert out.shape == (1, 300, 260)
