"""Shared evaluation metrics, so every model is scored the same way.

Three families (brief):
  * Integration / clustering : ari, nmi, asw, batch_entropy (patient mixing).
  * Subtype prediction       : auprc (macro, one-vs-rest), ece (calibration).
  * Generated data           : gene-wise mean / variance / zero-fraction error.

IMPORTANT: compute these on the SAME cell set for every model
(same cells, same genes), otherwise the numbers are not comparable.
"""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, average_precision_score)
from sklearn.neighbors import NearestNeighbors


# ---------- integration / clustering (latent space vs true cell types) ----------
def clustering_scores(Z, cell_type, n_clusters=None, seed=42):
    """ARI/NMI of k-means(Z) vs true cell types, and cell-type silhouette (ASW)."""
    y = np.asarray(cell_type)
    k = n_clusters or len(np.unique(y))
    lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
    return {"ARI": adjusted_rand_score(y, lab),
            "NMI": normalized_mutual_info_score(y, lab),
            "ASW_celltype": (silhouette_score(Z, y) + 1) / 2}  # rescaled to [0,1]


def batch_entropy(Z, batch, k=30, seed=42):
    """Patient mixing: mean normalised entropy of neighbour `batch` labels.

    0 = neighbours all from the same patient, 1 = well mixed. Higher is better
    for cell-type integration (the goal of patient correction).
    """
    b = np.asarray(batch)
    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    _, idx = nn.kneighbors(Z)
    uniq = {v: i for i, v in enumerate(np.unique(b))}
    codes = np.array([uniq[v] for v in b])
    ent = []
    for row in idx:
        counts = np.bincount(codes[row], minlength=len(uniq))
        p = counts[counts > 0] / counts.sum()
        h = -(p * np.log(p)).sum()
        ent.append(h / np.log(len(uniq)))      # normalise by max possible entropy
    return float(np.mean(ent))


# ---------- subtype prediction (patient level) ----------
def auprc_macro(y_true, proba, classes):
    """Macro one-vs-rest average precision (robust to the 5-HER2+ imbalance)."""
    y_true = np.asarray(y_true)
    aps = []
    for j, c in enumerate(classes):
        ap = average_precision_score((y_true == c).astype(int), proba[:, j])
        aps.append(ap)
    return {"AUPRC_macro": float(np.mean(aps)),
            "AUPRC_per_class": {c: float(a) for c, a in zip(classes, aps)}}


def expected_calibration_error(y_true, proba, classes, n_bins=10):
    """ECE of the predicted top-class probability (lower is better)."""
    y_true = np.asarray(y_true)
    conf = proba.max(1)
    pred = np.array(classes)[proba.argmax(1)]
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# ---------- generated data (posterior predictive check) ----------
def ppc_gene_errors(real, sim):
    """Gene-wise error between real and simulated counts (lower is better).

    Returns mean |Δ| across genes for mean, variance and zero-fraction.
    `real`/`sim`: arrays (cells x genes), same genes.
    """
    real, sim = np.asarray(real), np.asarray(sim)
    def stats(X):
        return X.mean(0), X.var(0), (X == 0).mean(0)
    rm, rv, rz = stats(real)
    sm, sv, sz = stats(sim)
    return {"mean_err": float(np.mean(np.abs(rm - sm))),
            "var_err": float(np.mean(np.abs(rv - sv))),
            "zero_frac_err": float(np.mean(np.abs(rz - sz)))}
