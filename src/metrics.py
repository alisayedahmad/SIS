from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial.distance import directed_hausdorff
import networkx as nx

def iou_score(pred, target, num_classes=2, eps=1e-6):
    if pred.ndim == 4:  # logits -> class
        pred = pred.argmax(1)
    ious = []
    for c in range(1, num_classes):  # ignore background
        p = (pred==c).float()
        t = (target==c).float()
        inter = (p*t).sum()
        union = p.sum()+t.sum()-inter
        ious.append((inter+eps)/(union+eps))
    return torch.stack(ious).mean()

def dice_score(pred, target, num_classes=2, eps=1e-6):
    if pred.ndim == 4:
        pred = pred.argmax(1)
    dices = []
    for c in range(1, num_classes):
        p = (pred==c).float()
        t = (target==c).float()
        inter = (2*(p*t).sum()+eps)/((p.sum()+t.sum())+eps)
        dices.append(inter)
    return torch.stack(dices).mean()

def bfscore(pred, target):
    """Boundary F-score basé sur les squelettes (approx)."""
    if pred.ndim == 4:
        pred = pred.argmax(1)
    pred = pred.detach().cpu().numpy().astype(bool)
    target = target.detach().cpu().numpy().astype(bool)
    scores = []
    for p, t in zip(pred, target):
        ps = skeletonize(p)
        ts = skeletonize(t)
        # Hausdorff dirigé approx
        a = np.argwhere(ps); b = np.argwhere(ts)
        if len(a)==0 and len(b)==0:
            scores.append(1.0); continue
        if len(a)==0 or len(b)==0:
            scores.append(0.0); continue
        d1 = directed_hausdorff(a, b)[0]
        d2 = directed_hausdorff(b, a)[0]
        # map to (0,1] with soft decay
        s = 1.0/ (1.0 + (d1+d2)/2.0)
        scores.append(s)
    return torch.tensor(scores).mean()

def apls_score(pred, target):
    """Approx APLS (Average Path Length Similarity) pour routes.
    1) squelette -> graphe non orienté.
    2) compare distributions de longueurs de plus courts chemins via EMD-like simplifié.
    """
    if pred.ndim == 4:
        pred = pred.argmax(1)
    pred = pred.detach().cpu().numpy().astype(bool)
    target = target.detach().cpu().numpy().astype(bool)

    def graph_from_mask(m):
        sk = skeletonize(m)
        G = nx.Graph()
        ys, xs = np.where(sk)
        coords = list(zip(ys, xs))
        for i,(y,x) in enumerate(coords):
            G.add_node(i, pos=(y,x))
        # 8-neigh
        off = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        idx = {(y,x):i for i,(y,x) in enumerate(coords)}
        for (y,x), i in idx.items():
            for dy,dx in off:
                j = idx.get((y+dy, x+dx), None)
                if j is not None:
                    G.add_edge(i,j,weight=np.hypot(dy,dx))
        return G

    def path_lengths(G, k=200):
        if G.number_of_nodes()==0:
            return np.array([0.0])
        nodes = list(G.nodes())
        rng = np.random.default_rng(0)
        pairs = rng.choice(len(nodes), size=(k,2), replace=True)
        dists = []
        for a,b in pairs:
            try:
                d = nx.shortest_path_length(G, nodes[a], nodes[b], weight="weight")
                dists.append(d)
            except nx.NetworkXNoPath:
                continue
        return np.array(dists) if len(dists)>0 else np.array([0.0])

    def dist_diff(a,b):
        a = np.sort(a); b = np.sort(b)
        # Discrétisation linéaire et distance L1 entre CDFs (EMD 1D)
        xs = np.linspace(0, max(a.max(), b.max(), 1.0), 100)
        cdfa = np.searchsorted(a, xs, side="right")/max(len(a),1)
        cdfb = np.searchsorted(b, xs, side="right")/max(len(b),1)
        emd = np.mean(np.abs(cdfa-cdfb))
        return 1.0/(1.0+emd)

    scores = []
    for p,t in zip(pred, target):
        Gp = graph_from_mask(p); Gt = graph_from_mask(t)
        dp = path_lengths(Gp); dt = path_lengths(Gt)
        scores.append(dist_diff(dp, dt))
    return torch.tensor(scores).mean()
