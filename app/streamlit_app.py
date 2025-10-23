import io
import numpy as np
import streamlit as st
import torch
import rasterio
from rasterio.io import MemoryFile
from PIL import Image
import logging
from pathlib import Path

from src.infer import slide_infer_on_raster
from src.train_module import SegLightningModule
from src.postprocess import postprocess_prediction
from src.geo.io import read_geotiff, write_geotiff_like
from src.utils import set_seed

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de la page
st.set_page_config(
    page_title="Satellite Segmentation Pro",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# En-tête
st.markdown('<h1 class="main-header">🛰️ Segmentation Satellite Professionnelle</h1>', unsafe_allow_html=True)
st.markdown("**Segmentation sémantique de bâtiments et routes sur imagerie satellite**")

# Sidebar - Configuration
st.sidebar.title("⚙️ Configuration")

# Upload du checkpoint
st.sidebar.subheader("1️⃣ Modèle")
ckpt_file = st.sidebar.file_uploader(
    "Checkpoint Lightning (.ckpt)",
    type=["ckpt"],
    help="Charger un modèle pré-entraîné"
)

# Paramètres de segmentation
st.sidebar.subheader("2️⃣ Paramètres")
threshold = st.sidebar.slider(
    "Seuil de confiance",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Seuil pour la binarisation des prédictions"
)

task = st.sidebar.selectbox(
    "Type de tâche",
    ["buildings", "roads", "multi"],
    help="Type d'objets à segmenter"
)

# Paramètres d'inférence
st.sidebar.subheader("3️⃣ Inférence")
tile_size = st.sidebar.select_slider(
    "Taille de tuile",
    options=[256, 512, 768, 1024],
    value=512,
    help="Taille des tuiles pour l'inférence (plus grand = plus rapide mais plus de mémoire)"
)

overlap = st.sidebar.select_slider(
    "Chevauchement",
    options=[0, 32, 64, 128, 256],
    value=64,
    help="Chevauchement entre tuiles (plus grand = meilleurs résultats)"
)

use_tta = st.sidebar.checkbox(
    "Test-Time Augmentation",
    value=False,
    help="Améliore la précision mais ralentit l'inférence (8x plus lent)"
)

apply_postprocess = st.sidebar.checkbox(
    "Post-traitement",
    value=True,
    help="Applique les post-traitements morphologiques"
)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
device_emoji = "🚀" if device == "cuda" else "🐌"
st.sidebar.markdown(f"**Device:** {device_emoji} `{device.upper()}`")

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    st.sidebar.markdown(f"*{gpu_name} ({gpu_memory:.1f} GB)*")

# Divider
st.sidebar.markdown("---")

# Upload de l'image
st.subheader("📁 Upload Image")
uploaded_file = st.file_uploader(
    "Choisir un GeoTIFF",
    type=["tif", "tiff"],
    help="Image satellite au format GeoTIFF"
)

# Colonnes principales
col1, col2 = st.columns(2)

if uploaded_file and ckpt_file:
    try:
        # Initialisation
        set_seed(42)
        
        with st.spinner("Chargement de l'image..."):
            # Lecture de l'image
            with MemoryFile(uploaded_file.getvalue()) as mem:
                with mem.open() as src:
                    img = src.read()  # (C, H, W)
                    meta = src.meta.copy()
            
            # Informations sur l'image
            with st.expander("ℹ️ Informations sur l'image", expanded=False):
                st.write(f"**Dimensions:** {img.shape[1]} × {img.shape[2]} pixels")
                st.write(f"**Canaux:** {img.shape[0]}")
                st.write(f"**CRS:** {meta.get('crs', 'N/A')}")
                st.write(f"**Dtype:** {img.dtype}")
                st.write(f"**Taille:** {uploaded_file.size / 1024 / 1024:.2f} MB")
        
        # Chargement du modèle
        with st.spinner("Chargement du modèle..."):
            model = SegLightningModule.load_from_checkpoint(
                io.BytesIO(ckpt_file.getvalue()),
                map_location=device
            )
            model.eval().to(device)
            
            st.success(f"✅ Modèle chargé: {model.hparams.model_name}")
        
        # Bouton d'inférence
        if st.button("🚀 Lancer la segmentation", type="primary"):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Inférence
            status_text.text("Inférence en cours...")
            progress_bar.progress(30)
            
            with torch.inference_mode():
                prob = slide_infer_on_raster(
                    model=model,
                    img=img,
                    tile_size=tile_size,
                    overlap=overlap,
                    batch_size=4,
                    device=device,
                    use_tta=use_tta,
                    progress_bar=False
                )
            
            progress_bar.progress(60)
            status_text.text("Binarisation...")
            
            # Binarisation
            pred = (prob >= threshold).astype(np.uint8)
            
            # Post-traitement
            if apply_postprocess and task in ["buildings", "roads"]:
                progress_bar.progress(80)
                status_text.text("Post-traitement...")
                pred = postprocess_prediction(pred, kind=task)
            
            progress_bar.progress(100)
            status_text.text("✅ Terminé!")
            
            # Visualisation
            with col1:
                st.subheader("🖼️ Image originale")
                rgb = np.moveaxis(img[:3], 0, -1)
                rgb = np.clip(
                    (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6),
                    0, 1
                )
                st.image(
                    Image.fromarray((rgb * 255).astype(np.uint8)),
                    use_container_width=True
                )
            
            with col2:
                st.subheader("🎯 Prédiction")
                pred_vis = pred.squeeze()
                st.image(
                    Image.fromarray((pred_vis * 255).astype(np.uint8)),
                    use_container_width=True
                )
            
            # Overlay
            st.subheader("🔀 Superposition")
            overlay = rgb.copy()
            mask_colored = np.zeros_like(overlay)
            mask_colored[pred_vis > 0] = [1, 0, 0]  # Rouge
            overlay = 0.7 * overlay + 0.3 * mask_colored
            st.image(
                Image.fromarray((overlay * 255).astype(np.uint8)),
                use_container_width=True
            )
            
            # Statistiques
            st.subheader("📊 Statistiques")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            total_pixels = pred_vis.size
            positive_pixels = pred_vis.sum()
            coverage = (positive_pixels / total_pixels) * 100
            
            with col_stat1:
                st.metric("Pixels totaux", f"{total_pixels:,}")
            with col_stat2:
                st.metric("Pixels positifs", f"{positive_pixels:,}")
            with col_stat3:
                st.metric("Couverture", f"{coverage:.2f}%")
            with col_stat4:
                st.metric("Seuil utilisé", f"{threshold:.2f}")
            
            # Export GeoTIFF
            st.subheader("💾 Export")
            out = io.BytesIO()
            meta_out = meta.copy()
            meta_out.update(count=1, dtype='uint8')
            write_geotiff_like(out, meta_out, pred.astype(np.uint8))
            
            st.download_button(
                label="📥 Télécharger GeoTIFF",
                data=out.getvalue(),
                file_name=f"prediction_{task}.tif",
                mime="image/tiff"
            )
    
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        logger.exception("Erreur durant l'inférence")

else:
    # Instructions
    st.info("👆 Veuillez charger un checkpoint et une image pour commencer")
    
    with st.expander("📖 Guide d'utilisation", expanded=True):
        st.markdown("""
        ### Instructions
        
        1. **Chargez un checkpoint** dans la barre latérale
        2. **Chargez une image GeoTIFF** ci-dessus
        3. **Ajustez les paramètres** selon vos besoins
        4. **Lancez la segmentation** et visualisez les résultats
        5. **Téléchargez** le résultat au format GeoTIFF
        
        ### Paramètres
        
        - **Seuil**: Plus élevé = prédictions plus conservatrices
        - **Taille de tuile**: Plus grand = plus rapide mais nécessite plus de mémoire
        - **Chevauchement**: Plus grand = meilleurs résultats aux bords des tuiles
        - **TTA**: Améliore la précision de ~2-3% mais 8× plus lent
        - **Post-traitement**: Applique des opérations morphologiques pour nettoyer les résultats
        
        ### Support
        
        - Formats supportés: GeoTIFF (.tif, .tiff)
        - Types de tâches: bâtiments, routes, multi-classes
        - Device: CPU ou GPU (CUDA)
        """)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: gray;">'
    '🛰️ Satellite Segmentation Pro | Powered by PyTorch Lightning & Streamlit'
    '</div>',
    unsafe_allow_html=True
)