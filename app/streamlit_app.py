import io
import numpy as np
import streamlit as st
import torch
import rasterio
from rasterio.io import MemoryFile
from PIL import Image

from src.infer import slide_infer_on_raster
from src.train_module import SegLightningModule
from src.postprocess import postprocess_prediction
from src.geo.io import read_geotiff, write_geotiff_like
from src.utils import set_seed

st.set_page_config(page_title="Satellite Segmentation", layout="wide")
st.title("🛰️ Segmentation de bâtiments & routes")

ckpt = st.sidebar.file_uploader("Checkpoint Lightning (.ckpt)", type=["ckpt"])
threshold = st.sidebar.slider("Seuil", 0.0, 1.0, 0.5, 0.05)
task = st.sidebar.selectbox("Tâche", ["buildings", "roads", "multi"])
tile = st.sidebar.slider("Taille de tuile", 256, 1024, 512, 64)
overlap = st.sidebar.slider("Overlap", 0, 256, 64, 16)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.markdown(f"**Device:** `{device}`")

uploaded = st.file_uploader("GeoTIFF d'entrée", type=["tif", "tiff"])

if uploaded and ckpt:
    set_seed(42)
    with MemoryFile(uploaded.getvalue()) as mem:
        with mem.open() as src:
            img = src.read()  # C,H,W
            meta = src.meta.copy()

    model = SegLightningModule.load_from_checkpoint(io.BytesIO(ckpt.getvalue()), map_location=device)
    model.eval().to(device)

    with torch.inference_mode():
        prob = slide_infer_on_raster(model, img, tile_size=tile, overlap=overlap, batch_size=4, device=device)

    pred = (prob >= threshold).astype(np.uint8)

    if task == "buildings":
        pred = postprocess_prediction(pred, kind="buildings")
    elif task == "roads":
        pred = postprocess_prediction(pred, kind="roads")

    # Affichage
    rgb = np.moveaxis(img[:3], 0, -1)
    rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6), 0, 1)
    st.image([
        Image.fromarray((rgb * 255).astype(np.uint8)),
        Image.fromarray((pred.squeeze() * 255).astype(np.uint8))
    ], caption=["Image", "Masque"], width=512)

    # Export GeoTIFF
    out = io.BytesIO()
    write_geotiff_like(out, meta, pred.astype(np.uint8))
    st.download_button("Télécharger GeoTIFF", data=out.getvalue(), file_name="prediction.tif", mime="image/tiff")
else:
    st.info("Chargez un GeoTIFF et un checkpoint pour lancer l'inférence.")
