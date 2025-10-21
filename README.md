# sat-segmentation

Pipeline complet et professionnel de **segmentation sémantique** pour **bâtiments** et **routes** sur imagerie satellite (SpaceNet, DeepGlobe).

## ✨ Fonctionnalités
- Prétraitement géospatial (lecture/écriture GeoTIFF, CRS, transform) avec `rasterio`, `geopandas`.
- Rasterisation (GeoJSON/SHP → masque) et vectorisation inverse.
- Tuilage 512×512 avec overlap et **mosaïque inverse par blending**.
- Datasets et DataModule Lightning (PyTorch Lightning).
- Modèles: **U-Net (ResNet34/50)**, **DeepLabV3+ (ResNet101)**, **Swin-UNet (expérimental)**.
- Pertes: BCE+Dice, Tversky, Focal+Dice. Métriques: IoU, Dice, BFScore, **APLS** (routes, approx conformes SpaceNet).
- Entraînement avec AdamW, OneCycleLR, AMP (precision=16), checkpoint sur mIoU.
- Inférence large échelle: sliding window + blending, export GeoTIFF & GeoJSON.
- Post-traitements: fermeture morphologique (bâtiments), squelettisation (routes).
- App **Streamlit** interactive.
- Reproductibilité: Docker, seeds fixés, configs YAML, tests unitaires.

## 📦 Installation rapide (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Sous Linux, `rasterio`/`geopandas` nécessitent GDAL/GEOS. Voir `Dockerfile` si besoin d’un environnement reproductible.

## 🐳 Docker
```bash
# Build
docker build -t sat-seg:latest .
# Entraînement (exemple)
docker run --rm -it -v $PWD:/workspace sat-seg:latest   python -m src.cli.train --config configs/unet_buildings.yaml
```

## 🗂 Données (exemples)
Organisez les données sous `data/raw/` (SpaceNet/DeepGlobe). Les scripts de préparation créent `data/tiles/`:
```bash
python scripts/prepare_spacenet2_buildings.py --root data/raw/spacenet2 --out data/tiles/spacenet2 --size 512 --overlap 64
python scripts/prepare_spacenet3_roads.py     --root data/raw/spacenet3 --out data/tiles/spacenet3 --size 512 --overlap 64
python scripts/prepare_deepglobe.py           --root data/raw/deepglobe --out data/tiles/deepglobe --size 512 --overlap 64
```

## 🚀 Entraînement
```bash
python -m src.cli.train --config configs/unet_buildings.yaml
# ou
python -m src.cli.train --config configs/deeplab_roads.yaml
```

## 🔮 Inférence large échelle
```bash
python -m src.cli.predict --checkpoint path/to/ckpt.ckpt --image path/to/image.tif --out outputs/pred.tif --geojson outputs/pred.geojson
```

## 🖥️ App Streamlit
```bash
streamlit run app/streamlit_app.py
```

## 🎯 Objectifs de performance (indicatifs)
| Modèle | Jeu | mIoU | Dice |
|---|---|---|---|
| U-Net (bâtiments) | SpaceNet 2 | ≥ 0.75 | ≥ 0.80 |
| DeepLabV3+ (routes) | SpaceNet 3 | ≥ 0.60 | ≥ 0.70 |

## 🧪 Tests
```bash
pytest -q
```

## 📁 Structure
Voir l’arborescence dans ce dépôt. Chaque module possède des docstrings détaillées.
