import argparse
import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint, LearningRateMonitor, EarlyStopping,
    RichProgressBar, ModelSummary, DeviceStatsMonitor
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler

from ..utils import set_seed, setup_logging, load_config, get_device
from ..data import TilesDataModule
from ..train_module import SegLightningModule

logger = logging.getLogger(__name__)


def setup_callbacks(cfg: Dict[str, Any]) -> list:
    """Configure tous les callbacks Lightning."""
    callbacks = []
    
    # Checkpoint - sauvegarde le meilleur modèle
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"checkpoints/{cfg['model']['name']}_{cfg['task']}",
        filename='{epoch:02d}-{val_mIoU:.4f}',
        monitor=cfg.get("metrics", {}).get("monitor", "val_mIoU"),
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_cb)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Early stopping
    if cfg['train'].get('early_stopping', False):
        early_stop = EarlyStopping(
            monitor=cfg.get("metrics", {}).get("monitor", "val_mIoU"),
            patience=cfg['train'].get('early_stopping_patience', 10),
            mode='max',
            verbose=True,
            min_delta=1e-4
        )
        callbacks.append(early_stop)
        logger.info(f"Early stopping activé (patience={cfg['train'].get('early_stopping_patience', 10)})")
    
    # Progress bar enrichie
    progress_bar = RichProgressBar(leave=True)
    callbacks.append(progress_bar)
    
    # Model summary
    model_summary = ModelSummary(max_depth=2)
    callbacks.append(model_summary)
    
    # Device stats (GPU monitoring)
    if cfg['train'].get('monitor_gpu', False):
        device_stats = DeviceStatsMonitor()
        callbacks.append(device_stats)
    
    return callbacks


def setup_logger(cfg: Dict[str, Any]):
    """Configure le logger (TensorBoard ou W&B)."""
    log_type = cfg.get('logging', {}).get('type', 'tensorboard')
    
    if log_type == 'wandb':
        try:
            wandb_logger = WandbLogger(
                project=cfg.get('logging', {}).get('project', 'sat-segmentation'),
                name=f"{cfg['model']['name']}_{cfg['task']}",
                config=cfg,
                log_model=True
            )
            logger.info("Logging avec Weights & Biases")
            return wandb_logger
        except Exception as e:
            logger.warning(f"Impossible d'initialiser W&B: {e}. Fallback sur TensorBoard.")
    
    # TensorBoard par défaut
    tb_logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=f"{cfg['model']['name']}_{cfg['task']}",
        default_hp_metric=False
    )
    logger.info("Logging avec TensorBoard")
    return tb_logger


def validate_config(cfg: Dict[str, Any]) -> None:
    """Valide la configuration avant l'entraînement."""
    # Vérifier que les données existent
    data_root = Path(cfg['data']['root'])
    if not data_root.exists():
        raise FileNotFoundError(f"Dossier de données introuvable: {data_root}")
    
    manifest = data_root / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest introuvable: {manifest}")
    
    # Vérifier cohérence des paramètres
    if cfg['num_classes'] < 2:
        raise ValueError(f"num_classes doit être >= 2, trouvé: {cfg['num_classes']}")
    
    if cfg['train']['batch_size'] < 1:
        raise ValueError(f"batch_size invalide: {cfg['train']['batch_size']}")
    
    logger.info("✓ Configuration validée")


def main(cfg_path: str, resume_from: str = None, test_only: bool = False, fast_dev_run: bool = False):
    """
    Fonction principale d'entraînement.
    
    Args:
        cfg_path: Chemin vers le fichier de configuration YAML
        resume_from: Chemin vers un checkpoint pour reprendre l'entraînement
        test_only: Exécuter uniquement le test (pas d'entraînement)
        fast_dev_run: Mode développement rapide (1 batch train/val, pas de checkpoint/logger)
    """
    # Configuration du logging
    setup_logging(log_dir="logs", level=logging.INFO)
    logger.info("=" * 80)
    logger.info("DÉBUT DE L'ENTRAÎNEMENT")
    logger.info("=" * 80)
    
    # Chargement de la configuration
    cfg = load_config(cfg_path)
    logger.info(f"Configuration chargée depuis: {cfg_path}")
    
    # Validation
    validate_config(cfg)
    
    # Seed pour reproductibilité
    set_seed(cfg.get("seed", 42))
    logger.info(f"Seed fixé à: {cfg.get('seed', 42)}")
    
    # DataModule
    logger.info("Initialisation du DataModule...")
    dm = TilesDataModule(
        root=cfg["data"]["root"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
        augment=cfg["data"]["augment"],
        num_classes=cfg["num_classes"],
        prefetch_factor=cfg["train"].get("prefetch_factor", 2),
        persistent_workers=cfg["train"].get("persistent_workers", True)
    )
    
    # Model
    logger.info("Initialisation du modèle...")
    model = SegLightningModule(
        model_name=cfg["model"]["name"],
        num_classes=cfg["num_classes"],
        in_channels=cfg["model"].get("in_channels", 3),
        encoder=cfg["model"].get("encoder", "resnet34"),
        decoder_channels=cfg["model"].get("decoder_channels", [256, 128, 64, 32, 16]),
        use_attention=cfg["model"].get("use_attention", False),
        dropout=cfg["model"].get("dropout", 0.0),
        pretrained=cfg["model"].get("pretrained", True),
        backbone=cfg["model"].get("backbone", "resnet101"),
        output_stride=cfg["model"].get("output_stride", 16),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        optimizer=cfg["train"].get("optimizer", "adamw"),
        scheduler=cfg["train"].get("scheduler", "onecycle"),
        loss=cfg.get("loss", "bce_dice"),
        use_ema=cfg["train"].get("use_ema", False),
        ema_decay=cfg["train"].get("ema_decay", 0.999),
        gradient_clip_val=cfg["train"].get("gradient_clip_val", 1.0),
        compute_apls=cfg.get("metrics", {}).get("compute_apls", False)
    )
    
    # Callbacks
    callbacks = setup_callbacks(cfg)
    
    # Logger
    pl_logger = setup_logger(cfg)
    
    # Profiler (optionnel)
    profiler = None
    if cfg.get('profiling', {}).get('enabled', False):
        profiler = PyTorchProfiler(
            dirpath="profiling",
            filename="profile",
            profile_memory=True,
            with_stack=True
        )
        logger.info("Profiling activé")
    
    # Trainer
    logger.info("Configuration du Trainer...")
    trainer = pl.Trainer(
        max_epochs=cfg["train"]["max_epochs"],
        precision=cfg["train"].get("precision", "32-true"),
        accelerator="auto",
        devices="auto",
        strategy="auto",
        logger=pl_logger,
        callbacks=callbacks,
        profiler=profiler,
        gradient_clip_val=cfg["train"].get("gradient_clip_val", 1.0),
        accumulate_grad_batches=cfg["train"].get("accumulate_grad_batches", 1),
        val_check_interval=cfg["train"].get("val_check_interval", 1.0),
        log_every_n_steps=cfg["train"].get("log_every_n_steps", 50),
        deterministic=True,
        benchmark=False,  # Pour reproductibilité
        enable_model_summary=True,
        enable_progress_bar=True,
        enable_checkpointing=True,
        fast_dev_run=fast_dev_run
    )
    
    # Log des hyperparamètres
    trainer.logger.log_hyperparams(cfg)
    
    if not test_only:
        # Entraînement
        logger.info("=" * 80)
        logger.info("DÉBUT DE L'ENTRAÎNEMENT")
        logger.info("=" * 80)
        
        trainer.fit(model, datamodule=dm, ckpt_path=resume_from)
        
        logger.info("=" * 80)
        logger.info("ENTRAÎNEMENT TERMINÉ")
        logger.info("=" * 80)
        
        # Meilleur checkpoint (peut être absent en fast_dev_run, ou si aucune
        # époque de validation complète n'a eu lieu)
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_model_score = trainer.checkpoint_callback.best_model_score
        if best_model_path:
            logger.info(f"Meilleur modèle sauvegardé: {best_model_path}")
        if best_model_score is not None:
            logger.info(f"Meilleur score: {best_model_score:.4f}")
    
    # Test
    logger.info("=" * 80)
    logger.info("DÉBUT DU TEST")
    logger.info("=" * 80)
    
    if resume_from and test_only:
        test_results = trainer.test(model, datamodule=dm, ckpt_path=resume_from)
    elif not test_only and trainer.checkpoint_callback.best_model_path:
        test_results = trainer.test(model, datamodule=dm, ckpt_path="best")
    else:
        # Pas de checkpoint "best" disponible (fast_dev_run, ou test_only sans
        # --resume): on teste avec les poids actuellement en mémoire plutôt
        # que de planter sur ckpt_path="best".
        logger.warning(
            "Aucun checkpoint 'best' disponible - test avec les poids actuels en mémoire."
        )
        test_results = trainer.test(model, datamodule=dm)
    
    logger.info("=" * 80)
    logger.info("RÉSULTATS DU TEST")
    logger.info("=" * 80)
    for key, value in test_results[0].items():
        try:
            logger.info(f"{key}: {value:.4f}")
        except (TypeError, ValueError):
            logger.info(f"{key}: {value}")
    
    logger.info("=" * 80)
    logger.info("PIPELINE TERMINÉ")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement de modèle de segmentation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Chemin vers le fichier de configuration YAML"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Reprendre depuis un checkpoint"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Exécuter uniquement le test"
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Mode développement rapide (1 batch par epoch)"
    )
    
    args = parser.parse_args()
    
    try:
        main(
            args.config,
            resume_from=args.resume,
            test_only=args.test_only,
            fast_dev_run=args.fast_dev_run
        )
    except Exception as e:
        logger.exception("Erreur fatale durant l'entraînement")
        raise e