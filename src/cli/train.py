import argparse, yaml, os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from ..utils import set_seed
from ..data import TilesDataModule
from ..train_module import SegLightningModule

def main(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))

    dm = TilesDataModule(root=cfg["data"]["root"],
                         batch_size=cfg["train"]["batch_size"],
                         num_workers=cfg["train"]["num_workers"],
                         mean=cfg["data"]["mean"], std=cfg["data"]["std"],
                         augment=cfg["data"]["augment"],
                         num_classes=cfg["num_classes"])

    model = SegLightningModule(model_name=cfg["model"]["name"],
                               num_classes=cfg["num_classes"],
                               in_channels=cfg["model"].get("in_channels", 3),
                               lr=cfg["train"]["lr"],
                               weight_decay=cfg["train"]["weight_decay"],
                               scheduler=cfg["train"].get("scheduler","onecycle"),
                               loss=cfg.get("loss","bce_dice"))

    ckpt_cb = ModelCheckpoint(monitor=cfg.get("monitor","val_mIoU"), mode="max", save_top_k=1)
    lrmon = LearningRateMonitor(logging_interval='step')
    tb = TensorBoardLogger(save_dir="lightning_logs", name=cfg["model"]["name"]+"_"+cfg["task"])

    trainer = pl.Trainer(max_epochs=cfg["train"]["max_epochs"],
                         precision=cfg["train"].get("precision", 32),
                         logger=tb, callbacks=[ckpt_cb, lrmon])
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
