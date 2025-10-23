from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial.distance import directed_hausdorff
import networkx as nx
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracker pour accumuler et moyenner les métriques."""
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float]):
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            self.metrics[name] += value
            self.counts[name] += 1
    
    def compute(self) -> Dict[str, float]:
        return {name: self.metrics[name] / self.counts[name] 
                for name in self.metrics}
    
    def reset(self):
        self.metrics.clear()
        self.counts.clear()


def iou_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2, eps: float = 1e-6, ignore_bg: bool = True) -> torch.Tensor:
    """IoU avec support multi-classes et option d'ignorer le background."""
    if pred.ndim == 4:  # logits -> class
        pred = pred.argmax(1)
    
    ious = []
    start_class = 1 if ignore_bg else 0
    
    for c in range(start_class, num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        
        if union == 0:  # Pas d'annotations pour cette classe
            iou = torch.tensor(1.0, device=pred.device)
        else:
            iou = (inter + eps) / (union + eps)
        
        ious.append(iou)
    
    return torch.stack(ious).mean() if ious else torch.tensor(0.0, device=pred.device)


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2, eps: float = 1e-6, ignore_bg: bool = True) -> torch.Tensor:
    """Dice score avec gestion robuste des cas edge."""
    if pred.ndim == 4:
        pred = pred.argmax(1)
    
    dices = []
    start_class = 1 if ignore_bg else 0
    
    for c in range(start_class, num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        
        if denom == 0:
            dice = torch.tensor(1.0, device=pred.device)
        else:
            dice = (2 * inter + eps) / (denom + eps)
        
        dices.append(dice)
    
    return torch.stack(dices).mean() if dices else torch.tensor(0.0, device=pred.device)


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Précision pixel à pixel."""
    if pred.ndim == 4:
        pred = pred.argmax(1)
    correct = (pred == target).sum().float()
    total = target.numel()
    return correct / total


def precision_recall_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    """Calcul de précision, rappel et F1-score par classe."""
    if pred.ndim == 4:
        pred = pred.argmax(1)
    
    metrics = {}
    
    for c in range(1, num_classes):  # Ignore background
        p = (pred == c).float()
        t = (target == c).float()
        
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        metrics[f'class_{c}_precision'] = precision
        metrics[f'class_{c}_recall'] = recall
        metrics[f'class_{c}_f1'] = f1
    
    return metrics


def bfscore(pred: torch.Tensor, target: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
    """Boundary F-score amélioré avec gestion d'erreurs."""
    if pred.ndim == 4:
        pred = pred.argmax(1)
    
    pred_np = pred.detach().cpu().numpy().astype(bool)
    target_np = target.detach().cpu().numpy().astype(bool)
    
    scores = []
    
    for p, t in zip(pred_np, target_np):
        try:
            # Squelettisation
            ps = skeletonize(p)
            ts = skeletonize(t)
            
            a = np.argwhere(ps)
            b = np.argwhere(ts)
            
            # Cas particuliers
            if len(a) == 0 and len(b) == 0:
                scores.append(1.0)
                continue
            if len(a) == 0 or len(b) == 0:
                scores.append(0.0)
                continue
            
            # Hausdorff bidirectionnel
            d1 = directed_hausdorff(a, b)[0]
            d2 = directed_hausdorff(b, a)[0]
            
            # Precision et recall basés sur le seuil
            precision = (d1 < threshold).astype(float).mean() if len(a) > 0 else 0.0
            recall = (d2 < threshold).astype(float).mean() if len(b) > 0 else 0.0
            
            if precision + recall > 0:
                f_score = 2 * precision * recall / (precision + recall)
            else:
                f_score = 0.0
            
            scores.append(f_score)
            
        except Exception as e:
            logger.warning(f"Erreur BFScore: {e}")
            scores.append(0.0)
    
    return torch.tensor(scores).mean()


def apls_score(pred: torch.Tensor, target: torch.Tensor, num_samples: int = 200) -> torch.Tensor:
    """APLS amélioré avec meilleure gestion des graphes."""
    if pred.ndim == 4:
        pred = pred.argmax(1)
    
    pred_np = pred.detach().cpu().numpy().astype(bool)
    target_np = target.detach().cpu().numpy().astype(bool)
    
    def build_graph(mask: np.ndarray) -> nx.Graph:
        """Construit un graphe pondéré depuis un squelette."""
        sk = skeletonize(mask)
        G = nx.Graph()
        
        ys, xs = np.where(sk)
        if len(ys) == 0:
            return G
        
        coords = list(zip(ys, xs))
        node_map = {coord: i for i, coord in enumerate(coords)}
        
        for i, (y, x) in enumerate(coords):
            G.add_node(i, pos=(y, x))
        
        # 8-connectivité
        offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for (y, x), i in node_map.items():
            for dy, dx in offsets:
                neighbor = (y + dy, x + dx)
                if neighbor in node_map:
                    j = node_map[neighbor]
                    if not G.has_edge(i, j):
                        weight = np.hypot(dy, dx)
                        G.add_edge(i, j, weight=weight)
        
        return G
    
    def sample_path_lengths(G: nx.Graph, k: int = 200) -> np.ndarray:
        """Échantillonne k longueurs de chemins."""
        if G.number_of_nodes() < 2:
            return np.array([0.0])
        
        nodes = list(G.nodes())
        rng = np.random.default_rng(42)  # Seed fixe pour reproductibilité
        
        dists = []
        attempts = min(k * 2, len(nodes) * (len(nodes) - 1))  # Limite les tentatives
        
        for _ in range(attempts):
            if len(dists) >= k:
                break
            
            a, b = rng.choice(nodes, size=2, replace=False)
            try:
                length = nx.shortest_path_length(G, a, b, weight="weight")
                dists.append(length)
            except nx.NetworkXNoPath:
                continue
        
        return np.array(dists) if dists else np.array([0.0])
    
    def earth_movers_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
        """EMD approximé pour distributions 1D."""
        if len(a) == 0 or len(b) == 0:
            return 1.0
        
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        
        max_val = max(a_sorted[-1], b_sorted[-1], 1.0)
        bins = np.linspace(0, max_val, 100)
        
        cdf_a = np.searchsorted(a_sorted, bins, side='right') / max(len(a), 1)
        cdf_b = np.searchsorted(b_sorted, bins, side='right') / max(len(b), 1)
        
        emd = np.mean(np.abs(cdf_a - cdf_b))
        
        # Score de similarité (1 = identique, 0 = très différent)
        return 1.0 / (1.0 + emd)
    
    scores = []
    
    for p, t in zip(pred_np, target_np):
        try:
            G_pred = build_graph(p)
            G_target = build_graph(t)
            
            paths_pred = sample_path_lengths(G_pred, k=num_samples)
            paths_target = sample_path_lengths(G_target, k=num_samples)
            
            similarity = earth_movers_distance_1d(paths_pred, paths_target)
            scores.append(similarity)
            
        except Exception as e:
            logger.warning(f"Erreur APLS: {e}")
            scores.append(0.0)
    
    return torch.tensor(scores).mean()


def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Matrice de confusion pour analyse détaillée."""
    if pred.ndim == 4:
        pred = pred.argmax(1)
    
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long, device=pred.device)
    
    for c_pred in range(num_classes):
        for c_true in range(num_classes):
            conf_matrix[c_true, c_pred] = ((pred == c_pred) & (target == c_true)).sum()
    
    return conf_matrix