"""Highly-variable gene selection on TRAIN cells only (no test leakage).

In LOPO this is redone once per left-out patient (26x). seurat_v3 works on raw
integer counts. Call inside each training round, never on the whole dataset.
"""
from __future__ import annotations


def select_hvg_train(adata, train_cell_mask, n_top: int = 2000, flavor: str = "seurat_v3"):
    """Return the `n_top` HVG names computed on the train cells only."""
    import scanpy as sc
    sub = adata[train_cell_mask]
    sub = sub.to_memory() if getattr(adata, "isbacked", False) else sub.copy()
    if flavor == "seurat_v3":
        d = sub.X.data if hasattr(sub.X, "data") else sub.X
        if not (d[:10000] == d[:10000].round()).all():
            raise ValueError("seurat_v3 needs raw counts in adata.X; load with data.load_adata()")
    sc.pp.highly_variable_genes(sub, n_top_genes=n_top, flavor=flavor)
    return sub.var_names[sub.var["highly_variable"]].tolist()
