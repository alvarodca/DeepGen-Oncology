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
from scipy.stats import entropy


# ---------- integration / clustering (latent space vs true cell types) ----------
def compute_entropy(X, labels, k=50, normalize=True):
    _, codes = np.unique(labels, return_inverse=True)
    n_classes = codes.max() + 1
    if n_classes < 2:
        return np.nan

    nn = NearestNeighbors(n_neighbors=min(k, len(X) - 1) + 1, n_jobs=-1).fit(X)
    indices = nn.kneighbors(X, return_distance=False)[:, 1:]

    counts = np.column_stack([(codes[indices] == c).sum(axis=1) for c in range(n_classes)])
    base = n_classes if normalize else None
    return float(entropy(counts, base=base, axis=1).mean())

def _safe_silhouette(Z, labels, sample_size=None, seed=42):
    labels = np.asarray(labels)
    n_unique = len(np.unique(labels))
    if n_unique < 2 or n_unique >= len(Z):
        return float("nan")
    return float(silhouette_score(Z, labels, sample_size=sample_size, random_state=seed))

def clustering_scores(Z, cell_type, n_clusters=None, asw_sample_size=20000, seed=42):
    """ARI/NMI of k-means(Z) vs true cell types, and cell-type silhouette (ASW)."""
    y = np.asarray(cell_type)
    k = n_clusters or len(np.unique(y))
    lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
    asw = _safe_silhouette(Z, y)
    return {
        "ARI": float(adjusted_rand_score(y, lab)),
        "NMI": float(normalized_mutual_info_score(y, lab)),
        "ASW_celltype": float((asw + 1) / 2) if not np.isnan(asw) else float("nan"),
    }

def evaluate_embedding(
    Z,
    cell_type,
    patient,
    batch_run=None,
    n_clusters=None,
    k=50,
    asw_sample_size=20000, # Pairwise distance matrix, thats why 
    seed=42,
):
    """All unsupervised bio-conservation & batch-mixing metrics on the SAME cells."""
    Z = np.asarray(Z)
    y_ct = np.asarray(cell_type)
    y_pat = np.asarray(patient)

    scores = clustering_scores(
        Z, y_ct, n_clusters=n_clusters, asw_sample_size=asw_sample_size, seed=seed
    )

    scores["ASW_patient"] = _safe_silhouette(
        Z, y_pat, sample_size=asw_sample_size, seed=seed
    )

    scores["Patient_Entropy"] = compute_entropy(Z, y_pat, k=k, normalize=True)
    if batch_run is not None:
        scores["Batch_Run_Entropy"] = compute_entropy(Z, batch_run, k=k, normalize=True)
    return scores

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
