import argparse, os, io, json
import numpy as np
import rasterio
import torch
from ..infer import slide_infer_on_raster
from ..train_module import SegLightningModule
from ..postprocess import postprocess_prediction
from ..geo.io import read_geotiff, write_geotiff_like

def main(ckpt, image, out, geojson=None, tile_size=512, overlap=64, threshold=0.5, task="buildings"):
    arr, meta = read_geotiff(image)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegLightningModule.load_from_checkpoint(ckpt, map_location=device)
    model.eval().to(device)
    with torch.inference_mode():
        prob = slide_infer_on_raster(model, arr, tile_size=tile_size, overlap=overlap, batch_size=4, device=device)
    pred = (prob >= threshold).astype(np.uint8)
    if task in ["buildings", "roads"]:
        pred = postprocess_prediction(pred, kind=task)
    write_geotiff_like(out, meta, pred.astype(np.uint8))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--geojson", default=None)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--task", choices=["buildings","roads","multi"], default="buildings")
    args = ap.parse_args()
    main(args.checkpoint, args.image, args.out, args.geojson, args.tile, args.overlap, args.threshold, args.task)
