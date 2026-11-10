"""Highly-variable gene selection on TRAIN cells only (no test leakage).

In LOPO this is redone once per left-out patient (26x). seurat_v3 works on raw
integer counts. Call inside each training round, never on the whole dataset.

Per-patient selection (`batch_key="donor_id"`): variability is measured inside
each patient and genes are ranked first by how many patients they are variable
in. Genes that only shift level between patients (technical or patient-specific)
stop dominating; genes that separate cell types within patients are kept.
`span=0.5`: with the scanpy default (0.3) the loess fit fails on some patients.

The subtype receptor genes (ESR1 = ER, PGR = PR, ERBB2 = HER2) sit right at the
cut-off: depending on the cells, per-patient selection drops ESR1 or ERBB2, and
PGR is never selected. They are always added on top (`force_genes`), so the
subtype signal is never lost. The list can therefore have up to n_top + 3 genes.
"""
from __future__ import annotations

SUBTYPE_GENES = ("ESR1", "PGR", "ERBB2")


def select_hvg_train(adata, train_cell_mask, n_top: int = 2000, flavor: str = "seurat_v3",
                     batch_key: str | None = "donor_id", span: float = 0.5,
                     force_genes=SUBTYPE_GENES):
    """Return the `n_top` HVG names computed on the train cells only, plus `force_genes`.

    batch_key=None gives the old global selection (all cells pooled).
    """
    import scanpy as sc
    sub = adata[train_cell_mask]
    sub = sub.to_memory() if getattr(adata, "isbacked", False) else sub.copy()
    if flavor == "seurat_v3":
        d = sub.X.data if hasattr(sub.X, "data") else sub.X
        if not (d[:10000] == d[:10000].round()).all():
            raise ValueError("seurat_v3 needs raw counts in adata.X; load with data.load_adata()")
    if batch_key is not None:
        sub.obs[batch_key] = sub.obs[batch_key].astype(str)   # drop unused categories (test patients)
    sc.pp.highly_variable_genes(sub, n_top_genes=n_top, flavor=flavor,
                                batch_key=batch_key, span=span)
    genes = sub.var_names[sub.var["highly_variable"]].tolist()
    forced = symbols_to_ids(adata, force_genes)
    return genes + [g for g in forced if g not in genes]


def symbols_to_ids(adata, symbols):
    """Gene symbols (e.g. 'ESR1') -> var_names (Ensembl IDs, e.g. 'ENSG00000091831')."""
    if not len(symbols):
        return []
    names = adata.var["feature_name"].astype(str) if "feature_name" in adata.var else None
    out = []
    for s in symbols:
        if s in adata.var_names:
            out.append(s)
            continue
        hits = [] if names is None else adata.var_names[names.values == s].tolist()
        if len(hits) != 1:
            raise KeyError(f"gene {s!r}: {len(hits)} matches in var['feature_name']")
        out.append(hits[0])
    return out
