#                                _..._
#                            .'     '.      _
#                            /    .-""-\   _/ \
#                        .-|   /:.   |  |   |
#                        |  \  |:.   /.-'-./
#                        | .-'-;:__.'    =/
#                        .'=  *=|     _.='
#                        /   _.  |    ;
#                        ;-.-'|    \   |
#                        /   | \    _\  _\
#                        \__/'._;.  ==' ==\
#                                \    \   |
#                                /    /   /
#                                /-._/-._/
#                                \   `\  \
#                                `-._/._/


import os
import sys 
import pickle
import functools
import time
from typing import Any

import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import enlighten

from datetime import timedelta
from tqdm import tqdm
from collections import defaultdict
from simple_slurm import Slurm  # pyright: ignore[reportMissingImports]

from SpaceTravLR.tools.network import expand_paired_interactions, get_cellchat_db

from enum import Enum

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class Status(Enum):
    BORN        =   "Newly born"
    BORED       =   "Ready but not doing anything"
    RUNNING     =   "Running"
    SUCCESS     =   "Completed everything gracefully"
    FUBAR       =   "F- Up Beyond Repair"

""" 
default output directory is 'output' 
'output/input_data' stores all the inputs
'output/logs' stores logs
'output/betadata'  stores the spatial gene-gene networks

methods with trailing underscores have side-effects but return Nothing
code philosophy is to fail early and loudly
"""


def catch_and_retry(retry=1):
    def wrapper(f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            for i in range(0, retry):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    # print(e)
                    raise e
                    time.sleep(i+1)
        return inner
    return wrapper

catch_errors = catch_and_retry(retry=1) #alias

class SpaceShip:
    def __init__(self, name: str = 'AlienTissue', outdir: str = './output'):
        self.name = name
        self.outdir = outdir.rstrip("/\\")
        self.manager = None
        self.status_bar = None
        
        self.status = Status.BORN
     
    @catch_errors
    def process_adata_(self, adata: ad.AnnData, annot: str = 'cell_type'):
        
        from .oracles import BaseTravLR
        from .tools.utils import scale_adata, is_mouse_data
        from .tools.network import encode_labels
        
        if self.status_bar:
            self.status_bar.update('📊 Processing AnnData: Validating input...')
        
        assert isinstance(adata, ad.AnnData)
        assert annot in adata.obs.columns
        assert 'spatial' in adata.obsm
        
        adata = adata.copy()
        
        if 'normalized_count' not in adata.layers:
            if self.status_bar:
                self.status_bar.update('📊 Processing AnnData: Creating normalized count layer...')
            adata.layers['normalized_count'] = adata.X.copy()
        
        self.species = 'mouse' if is_mouse_data(adata) else 'human'
        
        if self.status_bar:
            self.status_bar.update('📊 Processing AnnData: Scaling data...')
        adata = scale_adata(adata)
        
        if self.status_bar:
            self.status_bar.update('📊 Processing AnnData: Encoding cell types...')
        adata.obs['cell_type_int'] = adata.obs[annot].apply(
            lambda x: encode_labels(adata.obs[annot], reverse_dict=True)[x])
        
        if 'X_umap' not in adata.obsm:
            if self.status_bar:
                self.status_bar.update('📊 Processing AnnData: Computing PCA, neighbors, and UMAP...')
            sc.pp.pca(adata)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            
        if 'imputed_count' not in adata.layers:
            if self.status_bar:
                self.status_bar.update('📊 Processing AnnData: Imputing cluster-wise counts...')
            BaseTravLR.impute_clusterwise(
                adata, 
                annot=annot, 
                layer='normalized_count', 
                layer_added='imputed_count'
            )
        
        self.annot = annot
        
        if self.status_bar:
            self.status_bar.update('📊 Processing AnnData: Saving processed data...')
        adata.write_h5ad(f'{self.outdir}/input_data/_adata.h5ad')
        self.adata = adata
        
        if self.status_bar:
            self.status_bar.update('✅ Processing AnnData: Complete')
            
    def load_base_cell_thresholds(self) -> pd.DataFrame:
        df_ligrec = get_cellchat_db(self.species) 
        df_ligrec['name'] = df_ligrec['ligand'] + '-' + df_ligrec['receptor']
        expanded = expand_paired_interactions(df_ligrec)
        genes = set(expanded.ligand) | set(expanded.receptor)
        genes = list(genes)

        return pd.DataFrame(
            columns=genes, 
            index=self.adata.obs_names
        ).fillna(1).astype(int)
        
    @staticmethod 
    def load_base_GRN(species) -> pd.DataFrame:
        assert species in ['human', 'mouse']

        data_path = os.path.join(
            os.path.dirname(__file__), '..', 'SpaceTravLR_data', f'{species}_base_grn.parquet')
        df = pd.read_parquet(data_path)
        return df
    
    @catch_errors  
    def run_celloracle_(self, alpha=5):
        if self.status_bar:
            self.status_bar.update('Building base GRN...')
        
        import celloracle_tmp as co

        adata = self.adata
        
        oracle = co.Oracle()
        adata.X = adata.layers["raw_count"].copy()
        
        oracle.import_anndata_as_raw_count(
            adata=adata,
            cluster_column_name=self.annot,
            embedding_name="X_umap"
        )
        oracle.pcs = [True]
        oracle.k_knn_imputation = 1
        oracle.knn = 1
        
        base_GRN = self.load_base_GRN(self.species)

        oracle.import_TF_data(TF_info_matrix=base_GRN)
        
        if self.status_bar:
            self.status_bar.update('Computing & filtering TF links...')
        
        links = oracle.get_links(
            cluster_name_for_GRN_unit=self.annot, 
            alpha=alpha,
            verbose_level=0
        )

        links.filter_links()
        oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
        
        self.links = links.links_dict
        
        with open(f'{self.outdir}/input_data/celloracle_links.pkl', 'wb') as f:
            pickle.dump(links.links_dict, f)

    
    @catch_errors
    def run_commot_(self, radius=350):
        from .tools.network import expand_paired_interactions
        from .tools.network import get_cellchat_db
        import commot as ct
        
        adata = self.adata

        if 'cell_thresholds' not in adata.uns:

            if self.status_bar:
                self.status_bar.update('Loading ligand-receptor database...')
            df_ligrec = get_cellchat_db(self.species) 
            df_ligrec['name'] = df_ligrec['ligand'] + '-' + df_ligrec['receptor']
            
            if self.status_bar:
                self.status_bar.update('🔬 Commot: Expanding paired interactions...')
            expanded = expand_paired_interactions(df_ligrec)
            genes = set(expanded.ligand) | set(expanded.receptor)
            genes = list(genes)

            expanded = expanded[
                expanded.ligand.isin(adata.var_names) & expanded.receptor.isin(adata.var_names)]
            
            adata.X = adata.layers['normalized_count']
            
            if self.status_bar:
                self.status_bar.update('🔬 COMMOT: Computing spatial communication...')
            ct.tl.spatial_communication(adata,
                database_name='user_database', 
                df_ligrec=expanded, 
                dis_thr=radius, 
                heteromeric=False
            )
            
            expanded['rename'] = expanded['ligand'] + '-' + expanded['receptor']
            
            if self.status_bar:
                self.status_bar.update(f'Computing cluster communication for {len(expanded["rename"].unique())} pathways...')
            unique_pathways = expanded['rename'].unique()
            for idx, name in enumerate(unique_pathways):
                if self.status_bar:
                    self.status_bar.update(f'🔬 Commot: Cluster communication {idx+1}/{len(unique_pathways)}: {name[:30]}...')
                ct.tl.cluster_communication(
                    adata, 
                    database_name='user_database', 
                    pathway_name=name, 
                    clustering='cell_type',
                    random_seed=42, 
                    n_permutations=100
                )
                
            data_dict = defaultdict(dict)

            for name in expanded['rename']:
                data_dict[name]['communication_matrix'] = adata.uns[
                    f'commot_cluster-cell_type-user_database-{name}']['communication_matrix']
                data_dict[name]['communication_pvalue'] = adata.uns[
                    f'commot_cluster-cell_type-user_database-{name}']['communication_pvalue']

            with open(f'{self.outdir}/input_data/communication.pkl', 'wb') as f:
                pickle.dump(data_dict, f)
                
            info = data_dict
            
            def get_sig_interactions(value_matrix, p_matrix, pval=0.3):
                p_matrix = np.where(p_matrix < pval, 1, 0)
                return value_matrix * p_matrix
            
            if self.status_bar:
                self.status_bar.update('Processing significant interactions...')
            interactions = {}
            for lig, rec in tqdm(zip(expanded['ligand'], expanded['receptor'])):
                name = lig + '-' + rec
                if name in info.keys():
                    value_matrix = info[name]['communication_matrix']
                    p_matrix = info[name]['communication_pvalue']
                    sig_matrix = get_sig_interactions(value_matrix, p_matrix)
                    if sig_matrix.sum().sum() > 0:
                        interactions[name] = sig_matrix
                        
            if self.status_bar:
                self.status_bar.update('Computing ligand-receptor thresholds...')
            ct_masks = {cell_type: adata.obs[self.annot] == cell_type for cell_type in adata.obs[self.annot].unique()}
            df = pd.DataFrame(index=adata.obs_names, columns=genes)
            df = df.fillna(0)
            for name in tqdm(interactions.keys(), total=len(interactions)):
                lig, rec = name.rsplit('-', 1)
                tmp = interactions[name].sum(axis=1)
                for cell_type, val in zip(interactions[name].index, tmp):
                    df.loc[ct_masks[cell_type], lig] += tmp[cell_type]
                tmp = interactions[name].sum(axis=0)
                for cell_type, val in zip(interactions[name].columns, tmp):
                    df.loc[ct_masks[cell_type], rec] += tmp[cell_type]
                    
            perc_filtered = np.where(df > 0, 1, 0).sum().sum() / (df.shape[0] * df.shape[1])      
            
            df.to_parquet(f'{self.outdir}/input_data/LRs.parquet')
            
            adata.uns['cell_thresholds'] = df.copy()
        else:
            print('Cell thresholds already computed, skipping COMMOT...')
            df = adata.uns['cell_thresholds']
        
        if self.status_bar:
            self.status_bar.update('Caching received ligands...')
        
        keys = list(adata.obsm.keys())
        for key in keys:
            if 'commot' in key:
                del adata.obsm[key]
                
        keys = list(adata.uns.keys())
        for key in keys:
            if 'commot' in key:
                del adata.uns[key]
                
        keys = list(adata.obsp.keys())
        for key in keys:
            if 'commot' in key:
                del adata.obsp[key]

        self.adata = adata.copy()
        adata.write_h5ad(f'{self.outdir}/input_data/_adata.h5ad')
        self.status = Status.BORED

    def setup_(
        self, 
        adata: ad.AnnData, 
        overwrite=False,
        run_celloracle=True,
        run_commot=True,
    ):
        if os.path.exists(self.outdir) and not overwrite:
            print("Warning: output directory already exists. Will not overwrite.")
            self.status = Status.FUBAR
            return
        
        self.manager = enlighten.get_manager()
        self.status_bar = self.manager.status_bar(
            f'🚀 SpaceShip {self.name}: Initializing...',
            color='black_on_cyan',
            justify=enlighten.Justify.CENTER,
            auto_refresh=True
        )
        
        if self.status_bar:
            self.status_bar.update('🚀 SpaceShip: Creating output directories...')
        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(f'{self.outdir}/betadata', exist_ok=True)
        os.makedirs(f'{self.outdir}/input_data', exist_ok=True)
        os.makedirs(f'{self.outdir}/logs', exist_ok=True)
        
        self.status = Status.RUNNING
        
        self.process_adata_(adata)

        self.setup_tf_modulators_(run_celloracle=run_celloracle)
        self.setup_lr_modulators_(run_commot=run_commot)
        self.setup_tfl_modulators_()

        if self.status_bar:
            self.status_bar.update('✅ SpaceShip: Setup complete!')
        self.status = Status.BORED
        
        return self

    def setup_tf_modulators_(self, run_celloracle=True):
        if run_celloracle:
            self.run_celloracle_()
        else:
            from itertools import product

            base_grn = self.load_base_GRN(self.species)
            tfs = base_grn.columns
            tfs = list(set(tfs) & set(self.adata.var_names) - {'peak_id', 'gene_short_name'})
            targets = base_grn['gene_short_name'].unique()
            targets = list(set(targets) & set(self.adata.var_names))

            # this is inefficient but uses the same structure as with TF-target priors
            
            pairs = list(product(tfs, targets))
            df = pd.DataFrame(pairs, columns=['source', 'target'])
            df['coef_mean'] = 1
            df['coef_abs'] = 1
            df['p'] = 1e-5
            df['-logp'] = 5

            links_dict = {ct: df for ct in self.adata.obs[self.annot].unique()}
            self.links = links_dict
        
            with open(f'{self.outdir}/input_data/celloracle_links.pkl', 'wb') as f:
                pickle.dump(links_dict, f)

    def setup_lr_modulators_(self, run_commot=True):
        if run_commot:
            self.run_commot_()
        else:
            return
    
    def setup_tfl_modulators_(self):
        # this is species-specific, not dataset specific
        self.get_nichenet_links_()
    
    def get_nichenet_links_(self):
        if self.status_bar:
            self.status_bar.update('🔗 NicheNet: Downloading ligand-target links...')
        
        data_path = f'https://zenodo.org/records/17594271/files/ligand_target_{self.species}.parquet'
        nichenet_lt = pd.read_parquet(data_path)
        
        if self.status_bar:
            self.status_bar.update('🔗 NicheNet: Saving links...')
        nichenet_lt.to_parquet(f'{self.outdir}/input_data/tflinks.parquet')
        
        if self.status_bar:
            self.status_bar.update('✅ NicheNet: Complete')
        return nichenet_lt
        
        
    def spawn_worker(
        self, 
        partition='preempt', 
        clusters='gpu', 
        gres='gpu:1', 
        memory='20G',
        job_name='SpaceTravLR',
        lifespan=3, # hours
        python_path='python',
        ):
        
        outlog = f'{self.outdir}/logs/training_{str(time.strftime("%Y%m%d_%H%M%S"))}.log'
        
        slurm = Slurm(
            cpus_per_task=1,
            partition=partition,
            clusters=clusters,
            gres=gres,
            mem=memory,
            ignore_pbs=True,
            job_name=job_name+'_'+self.name,
            output=outlog,
            time=timedelta(hours=lifespan),
        ) 
        
        slurm.sbatch(python_path + ' launch.py')
        
    @catch_errors
    def run_spacetravlr(
        self, 
        max_epochs: int = 150, 
        learning_rate: float = 5e-3, 
        spatial_dim: int = 64, 
        batch_size: int = 512, 
        radius: int = 300, 
        contact_distance: int = 50,
        extra_modulators: list[str] = None,
        extra_lr: list[tuple[str, str]] = None,
    ):
        
        from .oracles import SpaceTravLR
        from .tools.network import RegulatoryFactory
        from .models.parallel_estimators import init_received_ligands
        
        base_dir = f'{self.outdir}/betadata/'
        adata = sc.read_h5ad(f'{self.outdir}/input_data/_adata.h5ad')
        tflinks = pd.read_parquet(f'{self.outdir}/input_data/tflinks.parquet')
        links = pickle.load(open(f'{self.outdir}/input_data/celloracle_links.pkl', 'rb'))

        co_grn = RegulatoryFactory(links=links)

        adata = init_received_ligands(
            adata, 
            radius=radius, 
            cell_threshes=adata.uns.get('cell_thresholds', None)
        )
        
        space_travlr = SpaceTravLR(
            adata=adata,
            max_epochs=max_epochs, 
            learning_rate=learning_rate, 
            spatial_dim=spatial_dim,
            batch_size=batch_size,
            grn=co_grn,
            radius=radius,
            contact_distance=contact_distance,
            save_dir=base_dir,
            tflinks=tflinks,
            extra_modulators=extra_modulators, 
            extra_lr=extra_lr
        )

        space_travlr.run()

    #@alias
    def fit(self, **kwargs): return self.run_spacetravlr(**kwargs)
    

    def is_everything_ok(self) -> bool:
        assert os.path.isfile(f'{self.outdir}/input_data/_adata.h5ad'), "AnnData file not found"
        _adata = sc.read_h5ad(f'{self.outdir}/input_data/_adata.h5ad')
        _links = pickle.load(open(f'{self.outdir}/input_data/celloracle_links.pkl', 'rb'))
        
        assert 'normalized_count' in _adata.layers, "Normalized count layer not found"
        assert 'imputed_count' in _adata.layers, "Imputed count layer not found"
        assert 'X_umap' in _adata.obsm, "UMAP embedding not found"
        assert 'cell_type_int' in _adata.obs.columns, "Cell type integer column not found"
        assert 'spatial' in _adata.obsm, "Spatial coordinates not found"
        
        assert os.path.isdir(self.outdir), "Output directory not found"
        assert os.path.isdir(f'{self.outdir}/betadata'), "Betadata directory not found"
        assert os.path.isdir(f'{self.outdir}/input_data'), "Input data directory not found"
        assert os.path.isfile(f'{self.outdir}/input_data/celloracle_links.pkl'), "Base links file not found"
        assert os.path.isdir(f'{self.outdir}/logs'), "Logs directory not found"
        assert os.path.isfile('launch.py'), "Launch script not found"
        
        print("We're going on a trip in our favorite rocket ship 🚀️")
        
        return True



