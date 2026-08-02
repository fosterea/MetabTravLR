import sys
from pathlib import Path

# Make the repo root (and src/) importable regardless of CWD or machine: walk up from
# this file until we hit a repo marker, then put that dir on sys.path.
_root = next(
    (p for p in Path(__file__).resolve().parents
     if (p / ".git").exists() or (p / "setup.py").exists()),
    Path(__file__).resolve().parent,
)
for _p in (str(_root), str(_root / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import scanpy as sc
# import matplotlib.pyplot as plt
import scvi
from pathlib import Path
import os
import random

from metab_processing.metab_travlr_config import PROJECT_DATA_DIR, DATA_DIR as METAB_DATA_DIR

# Consts
XENIUM2_DATA_DIR = f'{METAB_DATA_DIR}/Xenium2_HVG_Rep'
N_LATENT = 45
DEVICE = 'gpu'
N_DEVICES = 1
# DEVICE = 'cpu'
# N_DEVICES = 'auto'
SCVI_LATENT_KEY = "X_scVI"
DATASET_NAME = 'Primary_Dermal_Melanoma'

def run_scvi_params(adata_path, model_dir, save_adata_dir, overwrite_model=True):
    print(f'running on {adata_path}')
    adata = sc.read_h5ad(adata_path)
    print('adata loaded')

    # --- STEP 1: CALCULATE HIGHLY VARIABLE GENES ---
    # flavor="seurat_v3" expects raw count integers in the passed matrix
    sc.pp.highly_variable_genes(
        adata, 
        n_top_genes=2000, 
        flavor="seurat_v3",
        subset=False
    )
    print('HVG selection complete')

    # --- STEP 2: CREATE TEMPORARY SUBSET FOR TRAINING ---
    adata_hvg = adata[:, adata.var['highly_variable']].copy()

    scvi.model.SCVI.setup_anndata(
        adata_hvg,
        # batch_key="sample",
    )
    
    # Train the model on the HVG subset only
    model = scvi.model.SCVI(adata_hvg, n_latent=N_LATENT)
    print('starting training')

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(20000, 60000))
    
    model.train(accelerator=DEVICE, devices=N_DEVICES)
    print('training done')

    model.save(model_dir, overwrite=overwrite_model)
    
    # --- STEP 3: EXTRACT LATENT AND WRITE BACK TO FULL ADATA ---
    latent = model.get_latent_representation()
    adata.obsm[SCVI_LATENT_KEY] = latent

    # Save the original, full-sized object containing all genes
    adata.write_h5ad(save_adata_dir)
    print(f'Full adata saved with shape: {adata.shape}')

    return adata

resolutions = [2.5, 2, 1.5, 1, 0.75, 0.65, 0.5, 0.375, 0.25, 0.1, 0.05]
def res_label(res):
    return f'leiden_scVI_res_{res}'
res_labels = [res_label(res) for res in resolutions]


def add_umap(adata, adata_path=None):
    # if not "neighbors" in adata.uns:
    print('neighbors start')
    sc.pp.neighbors(adata, use_rep="X_scVI")
    # else:
    #     print('Neighbors already exists')

    for res in resolutions:
        
        leiden_key = res_label(res)
        # if leiden_key in adata.obs:
        #     print(f"The Leiden clustering run for '{leiden_key}' has already been made.")
        # else:
        print(f"Running clustering '{leiden_key}'.")
        sc.tl.leiden(adata, resolution=res, key_added=leiden_key, flavor="igraph")
        if adata.n_obs > 300000 and adata_path:
            adata.write_h5ad(adata_path)
    
    umap_key = 'X_umap' 

    # if umap_key in adata.obsm:
    #     print(f"UMAP coordinates found under '{umap_key}'.")
    #     print(f"Shape of coordinates: {adata.obsm[umap_key].shape}")
    # else:
    print(f"Running umap '{umap_key}'.")
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=res_labels)


run_scvi_params(f'{PROJECT_DATA_DIR}/{DATASET_NAME}/adata.h5ad',f'{XENIUM2_DATA_DIR}/{DATASET_NAME}/scvi_model', f'{XENIUM2_DATA_DIR}/{DATASET_NAME}/adata.h5ad')

adata_path = f'{XENIUM2_DATA_DIR}/{DATASET_NAME}/adata.h5ad'
adata = sc.read_h5ad(adata_path)
add_umap(adata, adata_path=adata_path)
adata.write_h5ad(adata_path)

