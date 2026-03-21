from collections import defaultdict
from functools import partialmethod
import os
import pandas as pd
import numpy as np
import glob
from dataclasses import dataclass
from typing import List, Optional, Tuple
from numba import jit, prange
import numpy as np
from tqdm import tqdm as tqdm_mock
import pyarrow.parquet as pq
tqdm_mock.__init__ = partialmethod(tqdm_mock.__init__, disable=True)
import warnings
import enlighten

warnings.filterwarnings('ignore')

import warnings
warnings.filterwarnings("ignore", message="Pandas doesn't allow columns to be created via a new attribute name")


@dataclass
class BetaOutput:
    betas: np.ndarray
    modulator_genes: List[str]
    modulator_gene_indices: List[int]
    ligands: Optional[List[str]] = None
    receptors: Optional[List[str]] = None
    tfl_ligands: Optional[List[str]] = None
    tfl_regulators: Optional[List[str]] = None
    ligand_receptor_pairs: Optional[List[Tuple[str, str]]] = None
    tfl_pairs: Optional[List[Tuple[str, str]]] = None
    wbetas: Optional[Tuple[str, pd.DataFrame]] = None


@jit(nopython=True, parallel=True)
def compute_all_derivatives(tf_vals, lr_betas, lr_ligs, lr_recs, tfl_betas, tfl_ligs, tfl_regs):
    n_samples = tf_vals.shape[0]
    
    # Compute all products in parallel
    rec_derivs = np.zeros((n_samples, lr_betas.shape[1]))
    lig_lr_derivs = np.zeros((n_samples, lr_betas.shape[1]))
    lig_tfl_derivs = np.zeros((n_samples, tfl_betas.shape[1]))
    tf_tfl_derivs = np.zeros((n_samples, tfl_betas.shape[1]))
    
    for i in prange(n_samples):
        # Compute all derivatives in parallel
        rec_derivs[i] = lr_betas[i] * lr_ligs[i]
        lig_lr_derivs[i] = lr_betas[i] * lr_recs[i]
        lig_tfl_derivs[i] = tfl_betas[i] * tfl_regs[i]
        tf_tfl_derivs[i] = tfl_betas[i] * tfl_ligs[i]
    
    return rec_derivs, lig_lr_derivs, lig_tfl_derivs, tf_tfl_derivs
    

class BetaFrame(pd.DataFrame):

    @classmethod
    def from_path(cls, path, obs_names=None, float16=False, randomize=False):
        if randomize:
            columns = pq.read_schema(path).names
            columns = [c for c in columns if c.startswith('beta')]
            df = pd.DataFrame(columns=columns, index=obs_names)
            df = df.apply(lambda x: np.random.randn(len(x)))
        else:
            df = pd.read_parquet(path, engine='pyarrow')
        df.index.name = path.split('/')[-1].split('_')[0]

        if float16:
            df = df.astype(np.float16)
        
        if obs_names is not None:
            df = df.loc[obs_names]


            
        return cls(df)

    def reindex(self, *args, **kwargs):
        result = super().reindex(*args, **kwargs)
        result = BetaFrame(result)
        
        for attr, value in vars(self).items():
            if attr != '_mgr':
                setattr(result, attr, value)

        return result
    
    def set_index(self, *args, **kwargs):
        result = super().set_index(*args, **kwargs)
        result = BetaFrame(result)
        
        for attr, value in vars(self).items():
            if attr != '_mgr':
                setattr(result, attr, value)

        return result

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prefix = 'beta_'
        self.tfs = []
        self.lr_pairs = []
        self.tfl_pairs = []

        # to be filled in later
        self.modulator_gene_indices = None
        self.wbetas = None

        self.is_random = False
        
        for col in self.columns:
            if col.startswith(self.prefix):
                modulator = col[len(self.prefix):]
                if '$' in modulator:
                    self.lr_pairs.append(modulator)
                elif '#' in modulator:
                    self.tfl_pairs.append(modulator) 
                else:
                    self.tfs.append(modulator)


        self.ligands, self.receptors = zip(
            *[p.split('$') for p in self.lr_pairs]) if self.lr_pairs else ([], [])
        self.tfl_ligands, self.tfl_regulators = zip(
            *[p.split('#') for p in self.tfl_pairs]) if self.tfl_pairs else ([], [])
        self.ligands = list(self.ligands)
        self.receptors = list(self.receptors)
        self.tfl_ligands = list(self.tfl_ligands)
        self.tfl_regulators = list(self.tfl_regulators)
        self.modulators_genes = [f'beta_{m}' for m in np.unique(
                self.tfs + self.ligands + self.receptors + \
                self.tfl_ligands + self.tfl_regulators)
            ]
        
        self._ligands = np.unique(list(self.ligands))
        self._tfl_ligands = np.unique(list(self.tfl_ligands))
        self._all_ligands = np.unique(list(self.ligands) + list(self.tfl_ligands))

        # self.df_lr_columns = [f'beta_{r}' for r in self.receptors]+ \
        #     [f'beta_{l}' for l in self.ligands]
        # self.df_tfl_columns = [f'beta_{r}' for r in self.tfl_regulators]+ \
        #     [f'beta_{l}' for l in self.tfl_ligands]
        
        self.tf_columns = [f'beta_{t}' for t in self.tfs]

        self.lr_pairs = [pair.split('$') for pair in self.lr_pairs]
        self.tfl_pairs = [pair.split('#') for pair in self.tfl_pairs]
    

    def splash(self, rw_ligands, rw_ligands_tfl, gex_df, scale_factor=1, beta_cap=None, grn_tfs=None):
        ## wL is the amount of ligand 'received' at each location
        ## assuming ligands and receptors expression are independent, dL/dR = 0
        ## y = b0 + b1*TF1 + b2*wL1R1 + b3*wL1R2
        ## dy/dTF1 = b1
        ## dy/dwL1 = b2[wL1*dR1/dwL1 + R1] + b3[wL1*dR2/dwL1 + R2]
        ##         = b2*R1 + b3*R2
        ## dy/dR1 = b2*[wL1 + R1*dwL1/dR1] = b2*wL1
        
        
        # _df = pd.DataFrame(
        #     np.concatenate([
        #         self[self.tf_columns].to_numpy(),
        #         self[[f'beta_{a}${b}' for a, b in zip(self.ligands, self.receptors)]*2].to_numpy() * \
        #             rw_ligands[self.ligands].join(gex_df[self.receptors]).to_numpy(),
        #         self[[f'beta_{a}#{b}' for a, b in zip(self.tfl_ligands, self.tfl_regulators)]*2].to_numpy() * \
        #             rw_ligands[self.tfl_ligands].join(gex_df[self.tfl_regulators]).to_numpy()
        #     ], axis=1),
        #     index=self.index,
        #     columns=self.tf_columns + self.df_lr_columns + self.df_tfl_columns
        # ).groupby(lambda x: x, axis=1).sum()

        # return _df[self.modulators_genes]

        lr_betas = self.filter(like='$', axis=1)
        tfl_betas = self.filter(like='#', axis=1)

        rec_derivatives = pd.DataFrame(
            np.where(
                gex_df[self.receptors].values > 0, # LR receptor betas only present if receptor is important to cell   
                lr_betas.values * rw_ligands[self.ligands].values,
                0
            ), 
            index=self.index, 
            columns=self.receptors
        ).astype(float) * scale_factor

        lig_lr_derivatives = pd.DataFrame(
            lr_betas.values * gex_df[self.receptors].values, 
            index=self.index, 
            columns=self.ligands
        ).astype(float) * scale_factor

        lig_tfl_derivatives = pd.DataFrame(
            tfl_betas.values * gex_df[self.tfl_regulators].values, 
            index=self.index, 
            columns=self.tfl_ligands
        ).astype(float) * scale_factor

        tf_derivatives = pd.DataFrame(
            self[self.tf_columns].values,
            index=self.index,
            columns=self.tfs
        ).astype(float)

        # if provided, enforce links to also appear in co_grn_links
        if grn_tfs is not None:
            grn_tfs = [f'beta_{t}' for t in grn_tfs]
            tf_derivatives.loc[:, ~tf_derivatives.columns.isin(grn_tfs)] = 0

        tf_tfl_derivatives = pd.DataFrame(
            tfl_betas.values * rw_ligands_tfl[self.tfl_ligands].values,
            index=self.index,
            columns=self.tfl_regulators
        ).astype(float) * scale_factor

        _df = pd.concat(
            [
                rec_derivatives, 
                lig_lr_derivatives, 
                lig_tfl_derivatives,
                tf_derivatives,
                tf_tfl_derivatives
            ], axis=1).groupby(level=0, axis=1).sum()
            # ], axis=1).groupby(level=0).sum()
                
        
        if beta_cap is not None:
            _df = _df.clip(lower=-beta_cap, upper=beta_cap)


        _df.columns = 'beta_' + _df.columns.astype(str)
        return _df[self.modulators_genes]

        
    def _repr_html_(self):
        info = f"BetaFrame with {len(self.modulators_genes)} modulator genes<br>"
        info += f"{len(set(self.tfs))} transcription factors<br>"
        info += f"{len(set(self.ligands))} ligands <br>"
        info += f"{len(set(self.receptors))} receptors <br>"
        info += f"{len(np.unique(self.lr_pairs))} ligand-receptor pairs<br>" 
        info += f"{len(np.unique(self.tfl_pairs))} tfl pairs<br>"
        df_html = super()._repr_html_()
        return f"<div><p>{info}</p>{df_html}</div>"


class Betabase:
    """
    Holds a collection of BetaFrames for each gene.
    """
    def __init__(
        self, 
        adata, 
        folder, 
        gene_subset=None, 
        subsample=None, 
        float16=True, 
        obs_names=None,
        genes=None,
        randomize=False,
        auto_load=True):
        
        assert os.path.exists(folder), f'Folder {folder} does not exist'
        # self.adata = adata
        self.xydf = pd.DataFrame(
            adata.obsm['spatial'], index=adata.obs_names)
        self.folder = folder
        self.gene2index = dict(
            zip(
                adata.var_names, 
                range(len(adata.var_names))
            )
        )
        self.gene_subset = gene_subset
        self.obs = adata.obs.copy()
        self.beta_paths = glob.glob(f'{self.folder}/*_betadata.parquet')
        
        if genes is not None:
            self.beta_paths = [path for path in self.beta_paths if any(gene in path for gene in genes)]
        
        if subsample is not None:
            self.beta_paths = self.beta_paths[:subsample]

        self.data = {}
        self.ligands_set = set()
        self.receptors_set = set()
        self.tfl_ligands_set = set()
        self.tfs_set = set()
        self.float16 = float16
        self.randomize = randomize
        
        if auto_load:
            self.load_betas_from_disk(obs_names=obs_names)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, gene_name):
        return self.data.get(gene_name, None)
    
    
    def collect_interactions(self, cell_type, annot='cell_type', aggregate='mean'):
        assert cell_type in self.obs[annot].unique()
        assert aggregate in ['mean', 'min', 'max', 'sum', 'positive', 'negative']
        
        beta_lr = defaultdict(list)
        beta_tfl = defaultdict(list)
        beta_tfs = defaultdict(list)
        
        manager = enlighten.get_manager()
        progress_bar = manager.counter(
            total=len(self.beta_paths),
            desc=f'Unraveling genes in {cell_type}',
            unit='parquet',
            color='orange',
            autorefresh=True,
        )   
        
        for j, f in enumerate(self.beta_paths):
            gene_name = f.split('/')[-1].replace('_betadata.parquet', '')
            beta = pd.read_parquet(f)
            beta = beta.join(self.obs[annot]).query(f'{annot}==@cell_type').drop(columns=[annot])

            if aggregate == 'mean':
                beta_ = beta.mean()
            elif aggregate == 'min':
                beta_ = beta.min()
            elif aggregate == 'max':    
                beta_ = beta.max()
            elif aggregate == 'sum':
                beta_ = beta.sum()
            elif aggregate == 'positive':
                beta_ = beta[beta > 0].fillna(0).mean()
            elif aggregate == 'negative':
                beta_ = beta[beta < 0].fillna(0).mean()
            
            for k, v in beta_.to_dict().items():
                if abs(v) > 0:
                    if '$' in k:
                        beta_lr[k].append((gene_name, v))
                    elif '#' in k:
                        beta_tfl[k].append((gene_name, v))
                    else:
                        beta_tfs[k].append((gene_name, v))

            progress_bar.update()
            
            
        beta_tf_out = pd.DataFrame(
            [(k, gene, beta) for k, gene_beta_pairs in beta_tfs.items() for gene, beta in gene_beta_pairs], 
                    columns=['interaction', 'gene', 'beta'])
        beta_tf_out.index.name = cell_type
        beta_tf_out['interaction_type'] = 'tf'
            
        beta_lr_out = pd.DataFrame(
            [(k, gene, beta) for k, gene_beta_pairs in beta_lr.items() for gene, beta in gene_beta_pairs], 
                    columns=['interaction', 'gene', 'beta'])
        beta_lr_out.index.name = cell_type
        beta_lr_out['interaction_type'] = 'ligand-receptor'
        
        beta_tfl_out = pd.DataFrame(
            [(k, gene, beta) for k, gene_beta_pairs in beta_tfl.items() for gene, beta in gene_beta_pairs], 
                    columns=['interaction', 'gene', 'beta'])
        beta_tfl_out.index.name = cell_type
        beta_tfl_out['interaction_type'] = 'ligand-tf'
        out_df = pd.concat([beta_tf_out, beta_lr_out, beta_tfl_out])
        out_df = out_df.query('interaction != "beta0"')
        
        return out_df
        

    def load_betadata(self, gene_name):
        return BetaFrame.from_path(f'{self.folder}/{gene_name}_betadata.parquet')

    def load_betas_from_disk(self, obs_names=None):
        "obs_names are the str cell index from adata.obs_names"
        
        manager = enlighten.get_manager()
        progress_bar = manager.counter(
            total=len(self.beta_paths),
            desc='Reading betadata files',
            unit='parquet',
            color='lightblue',
            autorefresh=True,
        )   
        for path in self.beta_paths:
            gene_name = path.split('/')[-1].split('_')[0]
            
            if self.gene_subset is not None and gene_name not in self.gene_subset:
                continue

            self.data[gene_name] = BetaFrame.from_path(
                path, obs_names=obs_names, randomize=self.randomize)
            
            self.ligands_set.update(self.data[gene_name]._ligands)
            self.tfl_ligands_set.update(self.data[gene_name]._tfl_ligands)
            self.receptors_set.update(self.data[gene_name].receptors)
            self.tfs_set.update(self.data[gene_name].tfs)

            progress_bar.update()
        
        for gene_name, betadata in self.data.items():
            betadata.modulator_gene_indices = [
                self.gene2index[g.replace('beta_', '')] for g in betadata.modulators_genes
            ]
            
        progress_bar.close()
        
        
