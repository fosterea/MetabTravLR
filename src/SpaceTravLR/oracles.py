from abc import ABC
from easydict import EasyDict
import numpy as np
import enlighten
import time
import pandas as pd
import pickle
import torch
from tqdm import tqdm
import os
import datetime
import re
import glob
import pickle
import io
import json
import warnings
from sklearn.decomposition import PCA
import warnings
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

from .tools.network import DayThreeRegulatoryNetwork
from .tools.knn_smooth import knn_smoothing
from .tools.utils import deprecated
from .models.spatial_map import xyc2spatial, xyc2spatial_fast
from .models.parallel_estimators import SpatialCellularProgramsEstimator

from .tools.utils import (
    clean_up_adata,
    knn_distance_matrix,
    _adata_to_matrix,
    connectivity_to_weights,
    convolve_by_sparse_weights,
    prune_neighbors
)

import warnings
warnings.filterwarnings("ignore")


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class BaseTravLR(ABC):
    
    def __init__(self, adata, fields_to_keep=['cell_type', 'cell_type_int', 'cell_thresholds']):
        assert 'normalized_count' in adata.layers or 'imputed_count' in adata.layers
        
        self.settings = EasyDict()
        
        self.adata = adata.copy()
        # self.adata.layers['normalized_count'] = self.adata.X.copy()
        self.gene2index = dict(zip(self.adata.var_names, range(len(self.adata.var_names))))
        self.pcs = None
        
        if 'imputed_count' not in self.adata.layers:
            self.pcs = self.perform_PCA(self.adata)
            self.knn_imputation(self.adata, self.pcs, method='MAGIC')

        clean_up_adata(self.adata, fields_to_keep=fields_to_keep)

    ## cannibalized from CellOracle
    @staticmethod
    def perform_PCA(adata, n_components=None, div_by_std=False):
        X = _adata_to_matrix(adata, "normalized_count")

        pca = PCA(n_components=n_components)
        if div_by_std:
            pcs = pca.fit_transform(X.T / X.std(0))
        else:
            pcs = pca.fit_transform(X.T)
        
        n_comps = np.where(np.diff(np.diff(np.cumsum(pca.explained_variance_ratio_))>0.002))[0][0]

        return pcs[:, :n_comps]
    
    
    @staticmethod
    def impute_clusterwise(adata, annot='cell_type', layer='normalized_count', layer_added='imputed_count'):
        import magic
        import warnings
        import enlighten
        warnings.filterwarnings("ignore")
        
        X = _adata_to_matrix(adata, layer)
        X = X.T
        X = pd.DataFrame(X, columns=adata.var_names, index=adata.obs_names)

        X_magic_list = []
        pbar = enlighten.get_manager().counter(
            total=len(adata.obs[annot].unique()),
            desc='Imputing clusterwise',
            unit='clusters',
            color='green',
            auto_refresh=True
        )

        for cell_type in adata.obs[annot].unique():
            magic_operator = magic.MAGIC(verbose=0)
            
            mask = adata.obs[annot] == cell_type
            X_subset = X.loc[mask]
            X_magic_subset = magic_operator.fit_transform(X_subset, genes='all_genes')
            X_magic_list.append(X_magic_subset)
            pbar.update()
            
        X_magic = pd.concat(X_magic_list)
        X_magic = X_magic.loc[adata.obs_names]
        
        adata.layers[layer_added] = X_magic.values

    @staticmethod
    @deprecated(instructions="Use impute_clusterwise instead")
    def knn_imputation(adata, pcs, k=None, metric="euclidean", diag=1,
                       n_pca_dims=50, maximum=False,
                       balanced=True, b_sight=None, b_maxl=None,
                       method='MAGIC', n_jobs=8) -> None:
        
        supported_methods = ['CellOracle', 'MAGIC', 'knn-smoothing']
        assert method in supported_methods, f'method is not implemented, choose from {supported_methods}'
        
        X = _adata_to_matrix(adata, "normalized_count")

        N = adata.shape[0] # cell number

        if k is None:
            k = int(N * 0.025)
        if b_sight is None and balanced:
            b_sight = int(k * 8)
        if b_maxl is None and balanced:
            b_maxl = int(k * 4)

        n_pca_dims = min(n_pca_dims, pcs.shape[1])
        space = pcs[:, :n_pca_dims]

        if method == 'CellOracle':
            if balanced:
                nn = NearestNeighbors(n_neighbors=b_sight + 1, metric=metric, n_jobs=n_jobs, leaf_size=30)
                nn.fit(space)

                dist, dsi = nn.kneighbors(space, return_distance=True)
                knn = prune_neighbors(dsi, dist, b_maxl)
            
            else:
                knn = knn_distance_matrix(space, metric=metric, k=k,
                                                mode="distance", n_jobs=n_jobs)
            
            connectivity = (knn > 0).astype(float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                connectivity.setdiag(diag)
            knn_smoothing_w = connectivity_to_weights(connectivity)

            Xx = convolve_by_sparse_weights(X, knn_smoothing_w)
            adata.layers["imputed_count"] = Xx.transpose().copy()
            
        elif method == 'MAGIC':
            import magic
            
            X = X.T
            magic_operator = magic.MAGIC()
            X = pd.DataFrame(X, columns=adata.var_names, index=adata.obs_names)
            X_magic = magic_operator.fit_transform(X, genes='all_genes')

            adata.layers['imputed_count'] = X_magic
        
        elif method == 'knn-smoothing':

            d = 10          # n pcs default 10
            dither = 0.03   # default 0.03 
            k = 32          # number of neighbors 

            matrix = adata.layers['raw_count'].T 
            S = knn_smoothing(matrix, k, d=d, dither=dither, seed=1334)

            adata.layers['imputed_count'] = S.T

            

class OracleQueue:

    """
    A jobs manager for training gene models in parallel
    Ideal for HPC environments
    """

    def __init__(self, model_dir, all_genes, priority_genes=None,lock_timeout=3600):
        """
        Args:
            model_dir (str): Directory to store the trained model weights.
            all_genes (list): List of all genes to train models for.
            priority_genes (list, optional): List of genes to train models for first.
            lock_timeout (int, optional): Timeout for job locks in seconds.
        """

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        self.model_dir = model_dir
        self.all_genes = all_genes
        self.orphans = []
        self.lock_timeout = lock_timeout
        self.priority_genes = priority_genes
        
        self.created_on = datetime.datetime.now()
        self.last_refresh_on = datetime.datetime.now()
        
    @property
    def age(self):
        """Return the age of the queue in seconds."""
        return (datetime.datetime.now() - self.created_on).total_seconds()
    
    def last_refresh_age(self):
        """Return the age of the last refresh in seconds."""
        return (datetime.datetime.now() - self.last_refresh_on).total_seconds()
    
        
    @property
    def regulated_genes(self):
        if not self.orphans:
            return self.all_genes
        return list(set(self.all_genes).difference(set(self.orphans)))
    
    def __getitem__(self, index):
        return self.remaining_genes[index]

    def __iter__(self):
        return self

    def __next__(self):
        """
        Get a gene to train for
        """

        if self.is_empty:
            raise StopIteration
        
        if self.priority_genes is not None:
            self.priority_genes = np.intersect1d(self.priority_genes, self.remaining_genes)
        
            if len(self.priority_genes) > 0:
                return self.priority_genes[0]
        
        return np.random.choice(self.remaining_genes)

    def __len__(self):
        return len(self.remaining_genes)
        
    @property
    def is_empty(self):
        return self.__len__() == 0

    @property
    def completed_genes(self):
        completed_paths = glob.glob(f'{self.model_dir}/*.parquet')
        return list(filter(None, map(self.extract_gene_name, completed_paths)))

    @property
    def num_orphans(self):
        return len(glob.glob(f'{self.model_dir}/*.orphan'))
    
    @property
    def agents(self):
        return len(glob.glob(f'{self.model_dir}/*.lock'))
    
    @property
    def remaining_genes(self):
        completed_paths = glob.glob(f'{self.model_dir}/*.parquet')
        locked_paths = glob.glob(f'{self.model_dir}/*.lock')
        orphan_paths = glob.glob(f'{self.model_dir}/*.orphan')
        completed_genes = list(filter(None, map(self.extract_gene_name, completed_paths)))
        locked_genes = list(filter(None, map(self.extract_gene_name, locked_paths)))
        orphan_genes = list(filter(None, map(self.extract_gene_name, orphan_paths)))
        return list(set(self.regulated_genes).difference(
            set(completed_genes+locked_genes+orphan_genes)))

    def create_lock(self, gene):
        """
        Create a lock for a gene while the model is being trained.
        This prevents multiple processes from training the same gene.
        """
        now = str(datetime.datetime.now())
        pid = os.getpid()
        with open(f'{self.model_dir}/{gene}.lock', 'w') as f:
            f.write(f'{now} {pid}')

    def delete_lock(self, gene):
        try:
            assert os.path.exists(f'{self.model_dir}/{gene}.lock')
            os.remove(f'{self.model_dir}/{gene}.lock')
        except Exception as e:
            print(f'Error deleting lock for {gene}: {e}')
    
    def kill_old_locks(self):
        locked_paths = glob.glob(f'{self.model_dir}/*.lock')

        for path in locked_paths:
            try:
                with open(path, 'r') as f:
                    data = f.read()
                lock = datetime.datetime.strptime(
                    ' '.join(data.split()[:2]), "%Y-%m-%d %H:%M:%S.%f")

                if (datetime.datetime.now() - lock).total_seconds() > self.lock_timeout:
                    gene = self.extract_gene_name(path)
                    self.delete_lock(gene)
                    print(f'Deleted lock for {gene} after {self.lock_timeout} seconds')
            
            except Exception as e:
                print(f'Error deleting lock for {path}: {e}')


    def add_orphan(self, gene):
        """
        Creates a .orphan file to mark a gene as untrainable.
        Used when a gene-gene network is too sparse to train a model.
        For example when a gene has no known TFs.
        """
        now = str(datetime.datetime.now())
        pid = os.getpid()
        with open(f'{self.model_dir}/{gene}.orphan', 'w') as f:
            f.write(f'{now} {pid}')
        self.orphans.append(gene)

    @staticmethod
    def extract_gene_name(path):
        patterns = {
            'betadata': r'([^/]+)_betadata\.parquet$',
            'lock': r'([^/]+)\.lock$',
            'orphan': r'([^/]+)\.orphan$',
            'perturbed': r'([^/]+)_\d+n_[\d\.]+x\.parquet$',
            'maxx': r'([^/]+)_\d+n_maxx\.parquet$'
        }
        
        for pattern in patterns.values():
            match = re.search(pattern, path)
            if match:
                return match.group(1)
        return None
    

    def __str__(self):
        return f'OracleQueue with {len(self.remaining_genes)} remaining genes'
    
    def __repr__(self):
        return self.__str__()



class SpaceTravLR(BaseTravLR):

    def __init__(
        self, adata, 
        save_dir='./models', annot='cell_type_int', grn=None,
        max_epochs=50, spatial_dim=64, learning_rate=5e-3, 
        batch_size=512, rotate_maps=True, 
        layer='imputed_count', alpha=0.05,
        threshold_lambda=1e-6, 
        tflinks=None,
        tf_ligand_cutoff=0.01, 
        receptor_thresh=0.01,
        radius=200, 
        contact_distance=30,
        skip_clusters=None,
        activation='identity', # Foster had to add
        scale_factor=1,
        genes=None,
        metab_pairs=None):

        super().__init__(adata, fields_to_keep=[annot, 'cell_thresholds'])
        if grn is None:
            self.grn = DayThreeRegulatoryNetwork() # CellOracle GRN
        else:
            self.grn = grn

        if genes is not None:
            if isinstance(genes, str):
                raise TypeError(
                    "genes must be a list of gene names, not a string"
                )
            if len(genes) == 0:
                raise ValueError(
                    "genes was passed as an empty list; pass genes=None to train all genes"
                )
            all_genes = list(dict.fromkeys(genes))
            var_set = set(self.adata.var_names)
            missing = [g for g in all_genes if g not in var_set]
            if missing:
                # Drop genes absent from the data (warn loudly) rather than erroring, so a
                # focus list can be reused across datasets with slightly different panels.
                print(
                    f"⚠️  SpaceTravLR: dropping {len(missing)} focus gene(s) not in "
                    f"adata.var_names: {missing}"
                )
                all_genes = [g for g in all_genes if g in var_set]
            if len(all_genes) == 0:
                raise ValueError(
                    "none of the requested focus genes are present in adata.var_names"
                )
        else:
            all_genes = list(self.adata.var_names)

        self.save_dir = save_dir
        self.queue = OracleQueue(save_dir, all_genes=all_genes)

        self.annot = annot
        self.max_epochs = max_epochs
        self.spatial_dim = spatial_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rotate_maps = rotate_maps
        self.layer = layer
        self.alpha = alpha
        self.threshold_lambda = threshold_lambda
        self.tf_ligand_cutoff = tf_ligand_cutoff
        self.receptor_thresh = receptor_thresh
        self.beta_dict = None
        self.coef_matrix = None
        self.radius = radius
        self.contact_distance = contact_distance
        self.scale_factor = scale_factor
        self.activation = activation
        self.tflinks = tflinks

        self.estimator_models = {}
        self.ligands = set()

        if metab_pairs is not None and (
            not isinstance(metab_pairs, list)
            or not all(
                isinstance(p, (tuple, list)) and len(p) == 2
                and all(isinstance(g, str) for g in p)
                for p in metab_pairs
            )
        ):
            # Fail fast at construction (not only when the first gene trains) --
            # matters for unattended Savio runs. The estimator does its own
            # (more thorough) validation too; this is a cheap up-front guard.
            raise ValueError(
                "metab_pairs must be a list of (export, import) gene-name tuples, or None"
            )

        self.genes = all_genes
        self.trained_genes = []
        self.skip_clusters = skip_clusters
        self.metab_pairs = metab_pairs

        if not os.path.exists(self.save_dir+'/run_params.json'):
            with open(self.save_dir+'/run_params.json', 'w') as f:
                json.dump({
                    'timestamp': str(datetime.datetime.now()),
                    'max_epochs': max_epochs,
                    'spatial_dim': spatial_dim,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'rotate_maps': rotate_maps,
                    'threshold_lambda': threshold_lambda,
                    'tf_ligand_cutoff': tf_ligand_cutoff,
                    'radius': radius,
                    'contact_distance': contact_distance,
                    'annot': annot,
                    'layer': layer,
                    'save_dir': save_dir,
                    'n_genes': len(self.genes),
                    'scale_factor': scale_factor,
                    'activation': activation,
                    'n_metab_pairs': len(metab_pairs) if metab_pairs else 0
                }, f, indent=4)

    
    def run(self):

        _manager = enlighten.get_manager()

        gene_bar = _manager.counter(
            total=len(self.queue.all_genes), 
            desc=f'... initializing ...', 
            unit='genes',
            color='green',
            autorefresh=True,
        )

        train_bar = _manager.counter(
            total=self.adata.shape[0]*self.max_epochs, 
            desc=f'Ready...', unit='cells',
            color='red',
            auto_refresh=True
        )


        while not self.queue.is_empty and not os.path.exists(self.save_dir+'/process.kill'):
            
            # Remove old locks from other models
            self.queue.kill_old_locks()

            gene = next(self.queue)

            estimator = SpatialCellularProgramsEstimator(
                adata=self.adata,
                target_gene=gene,
                layer=self.layer,
                cluster_annot=self.annot,
                spatial_dim=self.spatial_dim,
                radius=self.radius,
                contact_distance=self.contact_distance,
                tf_ligand_cutoff=self.tf_ligand_cutoff,
                grn=self.grn,
                scale_factor=self.scale_factor,
                activation=self.activation,
                tflinks=self.tflinks,
                receptor_thresh=self.receptor_thresh,
                metab_pairs=self.metab_pairs,
            )

            estimator.test_mode = False

            # A gene with metabolite (or other) modulators but no TF regulators should still
            # train. Historically we orphaned on TF regulators alone; only relax that when the
            # gene actually has metabolite modulators, so behavior is unchanged when metab_pairs
            # is not supplied (n_metab == 0 for every gene).
            n_metab = len(getattr(estimator, 'metab_pairs', []))
            if len(estimator.regulators) == 0 and n_metab == 0:
                self.queue.add_orphan(gene)
                continue

            else:
                gene_bar.count = len(self.queue.all_genes) - len(self.queue.remaining_genes)
                gene_bar.desc = f'🕵️️  {self.queue.agents+1} workers'
                gene_bar.refresh()

                if os.path.exists(f'{self.queue.model_dir}/{gene}.lock'):
                    continue

                self.queue.create_lock(gene)

                estimator.fit(
                    num_epochs=self.max_epochs, 
                    threshold_lambda=self.threshold_lambda, 
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    pbar=train_bar,
                    skip_clusters=self.skip_clusters
                )
                
                ## filter out columns with all zeros
                betadata = estimator.betadata
                nonzero_betadata = betadata.loc[:, (betadata != 0).any(axis=0)]
                if nonzero_betadata.shape[1] > 0:
                    nonzero_betadata.to_parquet(
                        f'{self.save_dir}/{gene}_betadata.parquet')
                else:
                    self.queue.add_orphan(gene)

                self.trained_genes.append(gene)
                self.queue.delete_lock(gene)
                
                if self.queue.last_refresh_age() > self.queue.lock_timeout:
                    self.queue.kill_old_locks()
                    self.queue.last_refresh_on = datetime.datetime.now()
                    

            gene_bar.count = len(self.queue.all_genes) - len(self.queue.remaining_genes)
            gene_bar.refresh()

            train_bar.count = 0
            train_bar.start = time.time()
            
        gene_bar.desc = 'All done! 🎉️'
        gene_bar.refresh()

    @staticmethod
    def imbue_adata_with_space(adata, annot='cell_type_int', 
            spatial_dim=64, in_place=False, method='fast'):
        """
        Generate and cache 2D spatial maps for each cell location
        """
        clusters = np.array(adata.obs[annot])
        xy = np.array(adata.obsm['spatial'])

        if method == 'fast':
            sp_maps = xyc2spatial_fast(
                xyc = np.column_stack([xy, clusters]),
                m=spatial_dim,
                n=spatial_dim,
            ).astype(np.float32)

        else:
            sp_maps = xyc2spatial(
                xy[:, 0], 
                xy[:, 1], 
                clusters,
                spatial_dim, spatial_dim, 
                disable_tqdm=False
            ).astype(np.float32)

        if in_place:
            adata.obsm['spatial_maps'] = sp_maps
            return

        return sp_maps