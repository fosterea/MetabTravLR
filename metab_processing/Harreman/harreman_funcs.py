import sys
from pathlib import Path

# Make the repo root importable regardless of CWD or machine: walk up from this
# file until we hit a repo marker, then put that dir on sys.path.
_root = next(
    (p for p in Path(__file__).resolve().parents
     if (p / ".git").exists() or (p / "setup.py").exists()),
    Path(__file__).resolve().parent,
)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from metab_processing.metab_travlr_config import PROJECT_DATA_DIR
XENIUM_DATA_DIR = PROJECT_DATA_DIR

import harreman
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import warnings
import json
import os
warnings.filterwarnings("ignore")
from pathlib import Path

import nbhd_scores


class HarremanRunner():

    def __init__(self, data_path, compute_nbhd_scores=True):
        self.easy_download_path = f'{data_path}/easy_download/harreman_outputs'
        self.data_path = data_path
        self.adata_path = f'{data_path}/adata.h5ad'
        self.compute_nbhd_scores = compute_nbhd_scores

    def load_adata(self):
        self.adata = sc.read_h5ad(self.adata_path)
        # Make sure raw counts are stored

        adata = self.adata

        adata.layers['counts'] = adata.X.copy()

        # Normalize and log-transform, store as a layer
        adata.X = adata.layers['counts'].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.layers['log_norm'] = adata.X.copy()

        # Restore raw counts as X
        adata.X = adata.layers['counts'].copy()

    def save_harreman_network(self):
        """Run this before doing anything else to get a baseline of the availible pathways.
        You will need to redo compute gene pairs if you filter after."""
        # Start fresh with transporter database only
        save_dir = self.easy_download_path
        adata = self.adata
        harreman.pp.extract_interaction_db(
            adata,
            species='human',
            database='transporter',   # metabolic transporters only, not LR pairs
            extracellular_only=True,
        )

        # Recompute KNN graph (in case it needs refreshing)
        harreman.tl.compute_knn_graph(
            adata,
            compute_neighbors_on_key='spatial',
            n_neighbors=5,
            weighted_graph=False,
            # sample_key='sample',
        )

        harreman.tl.compute_gene_pairs(adata, ct_specific=False, verbose=False)
        
        db_key = adata.uns['database_varm_key']
        db = pd.DataFrame(adata.varm[db_key], index=adata.var_names)
        transporter_genes = db.index[db.any(axis=1)].tolist()
        num_transporter_genes = len(transporter_genes)
        gp_per_metabolite = adata.uns['gene_pairs_per_metabolite']
        gp = adata.uns['gene_pairs']
        num_metab = len(gp_per_metabolite)
        num_gp = len(gp)
        metabolite_pair_counts = {metabolite: len(info['gene_pair']) for metabolite, info in gp_per_metabolite.items()}
        to_save = {
            'num_transporter_genes': num_transporter_genes,
            'num_metab': num_metab,
            'num_gp': num_gp,
            'metabolite_pair_counts': metabolite_pair_counts,
            'transporter_gense': transporter_genes,
            'gp': gp,
            'gp_per_metabolite': gp_per_metabolite,
        }
        save_path = f'{save_dir}/harreman_network.json'
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w") as file:
            json.dump(to_save, file)
        # flage = self.data_path
        # # Creates the file and closes it immediately
        # with open(filename, "w") as file:
        #     pass


    def run_harreman(self, cell_type_col, cell_type_indep_fdr=0.05, cell_type_dep_fdr=0.05, recompute_cell_indep=True):
        self.run_cell_independent(fdr_threshold=cell_type_indep_fdr, recompute_cell_indep=recompute_cell_indep)
        self.run_cell_aware(cell_type_col, fdr_threshold=cell_type_dep_fdr)
        self.save_harreman_outputs(cell_type_col, recompute_cell_indep=recompute_cell_indep)


    def run_cell_independent(self, n_permutations=1000, fdr_threshold=0.05, recompute_cell_indep=True):
        adata = self.adata

        if 'ccc_results' in adata.uns.keys():
            print('Already have ccc_results')
            return

        cell_type_indep_path = Path(self.easy_download_path) / '[ccc_results][cell_com_df_gp_sig].csv'
        if cell_type_indep_path.is_file() and not recompute_cell_indep:
            print('Cell indep already saved, not running')
            return
        
        print('running cell independent')
        

        # Start fresh with transporter database only
        harreman.pp.extract_interaction_db(
            adata,
            species='human',
            database='transporter',   # metabolic transporters only, not LR pairs
            extracellular_only=True,
            verbose=True
        )

        # Rerun gene filtering
        harreman.tl.apply_gene_filtering(
            adata,
            layer_key='counts',
            model='danb',
            autocorrelation_filt=False,
            verbose=True
        )

        # Recompute KNN graph (in case it needs refreshing)
        harreman.tl.compute_knn_graph(
            adata,
            compute_neighbors_on_key='spatial',
            n_neighbors=5,
            weighted_graph=False,
            # sample_key='sample',
            verbose=True
        )

        # Compute gene pairs with transporter database only
        harreman.tl.compute_gene_pairs(adata, ct_specific=False, verbose=True)
        print(f"Gene pairs to test: {len(adata.uns.get('gene_pairs', []))}")

        # Run communication
        harreman.tl.compute_cell_communication(
            adata,
            model='danb',
            M=n_permutations,
            test='both',
            layer_key_p_test='counts',
            layer_key_np_test='log_norm',
            verbose=True
        )

        harreman.tl.select_significant_interactions(adata, test='non-parametric', threshold=fdr_threshold)

        if self.compute_nbhd_scores:
            print('computing neighborhood scores')
            nbhd_scores.compute_nbhd_scores(adata, M=n_permutations, verbose=True)

    def run_cell_aware(self, cell_type_col, n_permutations=1000, fdr_threshold=0.05):

        adata = self.adata

        # Get the significant gene pairs from step 5 to focus on
        cell_type_indep_path = Path(self.easy_download_path) / '[ccc_results][cell_com_df_gp_sig].csv'
        if cell_type_indep_path.is_file():
            print('loading cell indep results for filtering')
            cell_communication_df = pd.read_csv(cell_type_indep_path)
            
        elif 'ccc_results' in adata.uns:
            print('using computed cell type indep results for filtering')
            cell_communication_df = adata.uns['ccc_results']['cell_com_df_gp_sig'].copy()
        else:
            raise Exception('Didn\'t load sig genes for filtering')
        gene_pairs_filt = list(zip(cell_communication_df['Gene 1'], cell_communication_df['Gene 2']))

        # compute_gene_pairs writes these with uns.setdefault, so a second tier would keep the
        # first tier's cell type labels and KeyError in create_weights_ct_pairs.
        for key in ('cell_type_pairs', 'gene_pairs_per_ct_pair'):
            adata.uns.pop(key, None)

        # Compute cell-type-specific gene pairs
        harreman.tl.compute_gene_pairs(adata, cell_type_key=cell_type_col, verbose=True)

        # Run Test 8 — cell-type-specific metabolite crosstalk
        harreman.tl.compute_ct_cell_communication(
            adata,
            model='danb',
            cell_type_key=cell_type_col,
            M=n_permutations,
            test='both',
            layer_key_p_test='counts',
            layer_key_np_test='log_norm',
            subset_gene_pairs=gene_pairs_filt,
            fix_gp=False,
            verbose=True
        )

        harreman.tl.select_significant_interactions(
            adata,
            test='non-parametric',
            ct_aware=True,
            threshold=fdr_threshold
        )

    def save_harreman_outputs(self, cell_type_col, recompute_cell_indep=True):

        adata = self.adata

        Path(f'{self.easy_download_path}/{cell_type_col}').mkdir(parents=True, exist_ok=True)

        cell_type_indep_path = Path(self.easy_download_path) / '[ccc_results][cell_com_df_gp_sig].csv'
        if cell_type_indep_path.is_file() and not recompute_cell_indep:
            print('Not saving cell type indep becasue file already exists.')
        else:
            adata.uns['ccc_results']['cell_com_df_gp_sig'].to_csv(cell_type_indep_path)
            adata.uns['ccc_results']['cell_com_df_m_sig'].to_csv(Path(self.easy_download_path) / '[ccc_results][cell_com_df_m_sig].csv')
            adata.uns['ccc_results']['cell_com_df_m'].to_csv(Path(self.easy_download_path) / '[ccc_results][cell_com_df_m].csv')

        adata.uns['ct_ccc_results']['cell_com_df_m'].to_csv(f'{self.easy_download_path}/{cell_type_col}/[ct_ccc_results][cell_com_df_m].csv')
        adata.uns['ct_ccc_results']['cell_com_df_m_sig'].to_csv(f'{self.easy_download_path}/{cell_type_col}/[ct_ccc_results][cell_com_df_m_sig].csv')
        adata.uns['ct_ccc_results']['cell_com_df_gp_sig'].to_csv(f'{self.easy_download_path}/{cell_type_col}/[ct_ccc_results][cell_com_df_gp_sig].csv')

        if self.compute_nbhd_scores:
            for grain in ('m', 'gp'):
                summary = nbhd_scores.summarize_nbhd_scores(adata, cell_type_col, grain=grain)
                summary.to_csv(f'{self.easy_download_path}/{cell_type_col}/[nbhd_scores][summary_{grain}].csv', index=False)

    def save_adata(self, filename):
        adata = self.adata
        backup = dict(adata.uns)
        for k, v in backup.items():
            if isinstance(v, pd.DataFrame):
                adata.uns[k] = v.fillna('')
            elif isinstance(v, dict):
                adata.uns[k] = {("__".join(map(str, key)) if isinstance(key, tuple) else str(key)): val for key, val in v.items()}
        adata.write_h5ad(filename)
        adata.uns.clear()
        adata.uns.update(backup)


