"""Patient-level data splits for the Wu 2021 breast atlas.

Unit of analysis = the patient (`donor_id`), n = 26 (not the ~100k cells).
Three schemes, all at patient level, so no cell of a patient is ever in both
train and test:
  * fold_5     : 5 folds, stratified by subtype (for slow generative models).
  * fold_lopo  : leave-one-patient-out (26 folds) -> subtype-prediction protocol.
  * fold_batch : leave-one-batch-run-out (robustness to technical batch signal).

Real obs columns used: donor_id, subtype, batch_run, celltype_major,
grade, cryo_state, treatment_status.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
KEY_PATIENT, KEY_SUBTYPE, KEY_BATCH, KEY_CT = "donor_id", "subtype", "batch_run", "celltype_major"


def patient_table(obs: pd.DataFrame) -> pd.DataFrame:
    """One row per patient with subtype, batch and confounders (from adata.obs)."""
    t = (obs.groupby(KEY_PATIENT, observed=True)
         .agg(subtype=(KEY_SUBTYPE, "first"), batch_run=(KEY_BATCH, "first"),
              grade=("grade", "first"), cryo=("cryo_state", "first"),
              treatment=("treatment_status", "first"),
              n_cells=(KEY_SUBTYPE, "size"),
              n_cancer_epi=(KEY_CT, lambda s: int((s == "Cancer Epithelial").sum())))
         .reset_index().rename(columns={KEY_PATIENT: "patient"}))
    for c in ("patient", "subtype", "batch_run", "grade", "cryo", "treatment"):
        t[c] = t[c].astype(str)
    return t.sort_values("patient").reset_index(drop=True)


def build_splits(patient: pd.DataFrame, n_splits: int = 5, seed: int = SEED) -> pd.DataFrame:
    """Add fold_5 / fold_lopo / fold_batch columns to the patient table."""
    p = patient.copy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p["fold_5"] = -1
    for k, (_, te) in enumerate(sgkf.split(p, p["subtype"], groups=p["patient"])):
        p.loc[p.index[te], "fold_5"] = k
    p["fold_lopo"] = np.arange(len(p))
    p["fold_batch"] = p["batch_run"]           # iterate leaving one run out
    return p


def inner_folds(p: pd.DataFrame, train_patients, n_splits: int = 4, seed: int = SEED):
    """Stratified-by-subtype inner CV over the TRAIN patients of an outer fold."""
    sub = p[p.patient.isin(train_patients)]
    n = min(n_splits, int(sub.subtype.value_counts().min()))
    skf = StratifiedGroupKFold(n_splits=max(2, n), shuffle=True, random_state=seed)
    return [(sub.patient.iloc[tr].tolist(), sub.patient.iloc[va].tolist())
            for tr, va in skf.split(sub, sub.subtype, groups=sub.patient)]


def save(p: pd.DataFrame, path="data/splits.parquet"):
    """Write a REAL parquet (the previous file was a CSV used by no model)."""
    cols = ["patient", "subtype", "batch_run", "grade", "cryo", "treatment",
            "n_cells", "n_cancer_epi", "fold_5", "fold_lopo", "fold_batch"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    p[cols].to_parquet(path, engine="pyarrow", index=False)
    return path


def load(path="data/splits.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)
