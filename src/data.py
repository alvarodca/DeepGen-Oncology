"""Load the atlas and attach the shared folds; get train/test masks per fold.

The .h5ad is gitignored and lives outside the repo. Edit CANDIDATES if needed.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad

from . import splits as S

REPO = Path(__file__).resolve().parents[1]   # paths work from any notebook folder
CANDIDATES = [REPO.parent / "wu2021_brca.h5ad",
              REPO / "data" / "wu_2021_breast_cancer.h5ad",
              REPO / "wu2021_brca.h5ad"]


def find_h5ad() -> Path:
    p = next((c for c in CANDIDATES if c.exists()), None)
    if p is None:
        raise FileNotFoundError(f"h5ad not found; tried {CANDIDATES}")
    return p


def load_adata(path: str | Path | None = None, backed: str | None = None):
    """Read the atlas (backed='r' to read obs/metadata without the counts).

    In the h5ad, `.X` is NOT raw counts: it is log1p(counts / total * 1e4)
    (each cell's expm1 sums to 10,000). The integer counts live in `.raw.X`.
    When loaded in memory we normalise the layout so every model sees the same:
        adata.X                  -> raw integer counts (scVI / MrVI / seurat_v3 HVG)
        adata.layers["lognorm"]  -> log1p CP10k (PCA and other Gaussian baselines)
    Do NOT apply log1p again on either of them.
    """
    adata = ad.read_h5ad(path or find_h5ad(), backed=backed)
    if backed or adata.raw is None:
        return adata   # backed mode: obs/metadata only, X left as stored
    if not adata.raw.var_names.equals(adata.var_names):
        raise ValueError("raw and X have different genes; cannot swap safely")
    adata.layers["lognorm"] = adata.X
    adata.X = adata.raw.X
    adata.raw = None   # counts are now in X; avoid keeping a third copy
    return adata


def attach_folds(adata, splits_path=REPO / "data" / "splits.parquet"):
    """Map patient-level folds onto every cell via donor_id."""
    sp = S.load(splits_path)
    for col in ("fold_5", "fold_lopo", "fold_batch"):
        adata.obs[col] = adata.obs[S.KEY_PATIENT].astype(str).map(dict(zip(sp.patient, sp[col])))
    return adata


def fold_mask(adata, scheme: str, test_value):
    """Boolean train/test masks for one fold of a scheme (no cell leakage)."""
    f = adata.obs[scheme].astype(str).values
    test = f == str(test_value)
    return ~test, test   # train_mask, test_mask


def iter_folds(adata, scheme="fold_5"):
    """Yield (test_value, train_mask, test_mask) for every fold of a scheme."""
    for v in pd.unique(adata.obs[scheme].astype(str)):
        tr, te = fold_mask(adata, scheme, v)
        yield v, tr, te
