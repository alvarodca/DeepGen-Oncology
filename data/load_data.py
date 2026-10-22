from pathlib import Path
import numpy as np
import json
import urllib.request
import ssl
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc 
import anndata as ad

DATA_DIR = Path(__file__).resolve().parent
H5AD_PATH = DATA_DIR / "wu_2021_breast_cancer.h5ad"
GEO_DIR = DATA_DIR / "geo_data"  # Backup source if CELLxGENE fails

CXG_COLLECTION_ID = "dea97145-f712-431c-a223-6b5f565f362a"
CXG_DATASET_ID    = "9fddb063-056d-4202-8b8a-4b0ee531d3ce"

# Set to None to keep the full dataset for EDA (skips subsampling)
N_CELLS = None 
SEED = 42

#  Helper Functions 
def to_dense(x):
    if sp.issparse(x):
        return x.toarray()
    if hasattr(x, "todense"):
        return np.asarray(x.todense())
    return np.asarray(x)

# We need raw counts as input to scVI and for scanpy functions
def is_integer_counts(X, n=500):
    sub = to_dense(X[:n])
    return bool(np.all(sub >= 0) and np.allclose(sub, np.round(sub)))

def ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

def download_from_cellxgene():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api_url = (f"https://api.cellxgene.cziscience.com/curation/v1/collections/"
               f"{CXG_COLLECTION_ID}/datasets/{CXG_DATASET_ID}")
    ctx = ssl_context()
    with urllib.request.urlopen(api_url, context=ctx) as r:
        meta = json.load(r)
    url = next((a["url"] for a in meta.get("assets", []) if a.get("filetype") == "H5AD"), None)
    if url is None:
        raise ValueError("No H5AD file found in the CELLxGENE dataset metadata.")

    print(f"Downloading {url} (~850 MB), this can take a while...")
    tmp_path = H5AD_PATH.with_suffix(".h5ad.part")
    with urllib.request.urlopen(url, context=ctx) as r, open(tmp_path, "wb") as f:
        shutil.copyfileobj(r, f, length=16 * 1024 * 1024)
    tmp_path.rename(H5AD_PATH)
    print("Download complete.")

def load_wu2021():
    if not H5AD_PATH.exists() and not GEO_DIR.exists():
        print("Data not found locally, downloading it from CELLxGENE.")
        try:
            download_from_cellxgene()
        except Exception as e:
            raise FileNotFoundError(f"Automatic download failed ({e}). Download manually to {H5AD_PATH}.") from e

    if H5AD_PATH.exists():
        adata = sc.read_h5ad(H5AD_PATH)
        if not is_integer_counts(adata.X):
            assert adata.raw is not None, "X is not raw counts and there is no .raw"
            adata = adata.raw.to_adata()
        
        if "feature_name" in adata.var: # CELLxGENE uses Ensembl IDs as the index
            adata.var["ensembl_id"] = adata.var_names
            adata.var_names = adata.var["feature_name"].astype(str)
            
    elif GEO_DIR.exists():
        X = sp.mmread(GEO_DIR / "count_matrix_sparse.mtx").tocsr()
        genes = pd.read_csv(GEO_DIR / "count_matrix_genes.tsv", header=None, sep="\t")[0].astype(str).values
        barcodes = pd.read_csv(GEO_DIR / "count_matrix_barcodes.tsv", header=None, sep="\t")[0].astype(str).values
        if X.shape[0] == len(genes) and X.shape[1] == len(barcodes):
            X = X.T.tocsr()
        meta = pd.read_csv(GEO_DIR / "metadata.csv", index_col=0)
        adata = ad.AnnData(X=X, obs=meta.loc[barcodes].copy(), var=pd.DataFrame(index=genes))
    else:
        raise FileNotFoundError("Data file not found.")

    adata.var_names_make_unique()
    assert is_integer_counts(adata.X), "The matrix doesn't look like raw counts"
    return adata

def first_col(obs, candidates):
    for c in candidates:
        if c in obs.columns:
            return c
    raise KeyError(f"None of {candidates} found in obs: {list(obs.columns)}")

if __name__ == "__main__":
    adata = load_wu2021()

    # Standardizing metadata column names
    COLMAP = {
        "patient":        first_col(adata.obs, ["orig.ident", "donor_id", "Patient", "patient"]),
        "subtype":        first_col(adata.obs, ["subtype", "Subtype", "clinical_subtype", "disease"]),
        "celltype":       first_col(adata.obs, ["celltype_major", "author_cell_type", "cell_type"]),
        "celltype_minor": first_col(adata.obs, ["celltype_minor", "celltype_major", "author_cell_type", "cell_type"]),
    }

    for std, orig in COLMAP.items():
        adata.obs[std] = adata.obs[orig].astype(str).astype("category")

    # Optional subsampling
    if N_CELLS is not None and adata.n_obs > N_CELLS:
        sc.pp.subsample(adata, n_obs=N_CELLS, random_state=SEED)

