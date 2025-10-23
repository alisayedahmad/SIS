from __future__ import annotations
import random, os, logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch
import yaml

def set_seed(seed: int):
    """Fixe tous les seeds pour reproductibilité totale."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Configure un logger professionnel avec rotation de fichiers."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"{log_dir}/pipeline.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(path: str) -> Dict[str, Any]:
    """Charge et valide un fichier de configuration YAML."""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Validation basique
    required = ['seed', 'task', 'num_classes', 'model', 'train', 'data']
    for key in required:
        if key not in cfg:
            raise ValueError(f"Configuration manquante: {key}")
    
    return cfg

def get_device(preferred: Optional[str] = None) -> torch.device:
    """Détermine le meilleur device disponible."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

class EarlyStopping:
    """Early stopping pour éviter le surapprentissage."""
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        improved = (score - self.best_score > self.min_delta) if self.mode == 'max' else (self.best_score - score > self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop