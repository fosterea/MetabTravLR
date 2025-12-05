
from collections import defaultdict
from functools import partial
import glob
import statistics
import numpy as np
import pandas as pd
from tqdm import tqdm
# import commot as ct
import gc
from .tools.network import expand_paired_interactions, get_cellchat_db
from .models.parallel_estimators import get_filtered_df, received_ligands
from .oracles import OracleQueue, BaseTravLR
from .beta import BetaFrame, Betabase
from .tools.utils import is_mouse_data
import enlighten
from pqdm.threads import pqdm
import datetime
import os
import warnings
warnings.filterwarnings('ignore')


            
class GeneFactory(BaseTravLR):
    def __init__(
        self, 
        adata, 
        models_dir, 
        annot='cell_type_int', 
        radius=200, 
        contact_distance=30,
        scale_factor=1,
        beta_scale_factor=1,
        beta_cap=None, 
        co_grn=None
        ):
        
        super().__init__(adata, fields_to_keep=[annot])
        
        self.adata = adata.copy()
        self.annot = annot
        self.save_dir = models_dir
        self.radius = radius
        self.contact_distance = contact_distance
        self.species = 'mouse' if is_mouse_data(adata) else 'human'
        self.scale_factor = scale_factor

        self.queue = OracleQueue(models_dir, all_genes=self.adata.var_names)
        self.ligands = []
        self.genes = list(self.adata.var_names)
        self.trained_genes = []
        self.beta_dict = None
        self._name = 'GeneFactory'
        self.beta_scale_factor = beta_scale_factor
        self.beta_cap = beta_cap
        
        if co_grn is not None:
            flat_links = pd.concat([co_grn.links[ct] for ct in co_grn.links.keys()], axis=0)
            flat_links.sort_values(by='coef_mean', ascending=False, inplace=True)
            flat_links.drop_duplicates(subset=['source', 'target'], inplace=True, keep='first')
            self.co_grn_links = flat_links
        else:
            self.co_grn_links = None

        self.manager = enlighten.get_manager()
        
        self._logo = '🐭️🧬️️' if self.species == 'mouse' else '🙅‍♂️🧬️️'
        self._logo = f'{self._logo} {self._name}'
        
        self.status = self.manager.status_bar(
            f'{self._logo}: [Ready] | {adata.shape[0]} cells / {len(self.genes)} genes',
            color='black_on_green',
            justify=enlighten.Justify.CENTER,
            auto_refresh=True,
            width=30
        )
        
        self.xy = pd.DataFrame(
            self.adata.obsm['spatial'], 
            index=self.adata.obs_names, 
            columns=['x', 'y']
        )
    
        # df_ligrec = ct.pp.ligand_receptor_database(
        #         database='CellChat', 
        #         species=self.species, 
        #         signaling_type=None
        #     )
            
        # df_ligrec.columns = ['ligand', 'receptor', 'pathway', 'signaling']
        
        df_ligrec = get_cellchat_db(self.species)  
        
        self.lr = expand_paired_interactions(df_ligrec)
        self.lr = self.lr[
            self.lr.ligand.isin(self.adata.var_names) & (
                self.lr.receptor.isin(self.adata.var_names))]
        self.lr['radius'] = np.where(
            self.lr['signaling'] == 'Secreted Signaling', 
            self.radius, self.contact_distance
        )
        
        
    @classmethod
    def from_json(cls, adata, json_path, override_params=None, 
                  beta_scale_factor=1, beta_cap=None, co_grn=None):
        import json
        
        with open(json_path, 'r') as f:
            params = json.load(f)
            
        if override_params is not None:
            params.update(override_params)
            
        return cls(
            adata, 
            models_dir=params['save_dir'], 
            annot=params['annot'], 
            radius=params['radius'], 
            contact_distance=params['contact_distance'],
            scale_factor=params.get('scale_factor', 1),
            beta_scale_factor=beta_scale_factor,
            beta_cap=beta_cap,
            co_grn=co_grn
        )
        
    ## backwards compatibility
    def compute_betas(self, **kwargs):
        self.load_betas(**kwargs)

    def load_betas(self, subsample=None, float16=False, obs_names=None):
        self.beta_dict = None
        del self.beta_dict
        gc.collect()
        
        obs_names = obs_names if obs_names is not None else self.adata.obs_names
        
        self.status.update(
            '💾️ Loading betas from disk' + f' {len(obs_names)} cells')
        self.status.color = 'black_on_salmon'
        self.status.refresh()

        self.beta_dict = self._get_spatial_betas_dict(
            subsample=subsample, 
            float16=float16, 
            obs_names=obs_names
        )
        
        self.obs_names = obs_names
        
        self.status.update('Loading betas - Done')
        self.status.color = 'black_on_green'
        self.status.refresh()
        
    @staticmethod
    def load_betadata(gene, save_dir, obs_names=None):
        return BetaFrame.from_path(f'{save_dir}/{gene}_betadata.parquet', obs_names=obs_names)
    
    def _compute_weighted_ligands(self, gene_mtx, cell_thresholds, genes):
        self.update_status(f'{self.current_target} >> Computing received ligands', color='black_on_cyan')
        gex_df = pd.DataFrame(
            gene_mtx, 
            index=self.obs_names, 
            columns=self.adata.var_names
        )
        
        if len(genes) > 0:
            weighted_ligands = received_ligands(
                xy=self.adata[self.obs_names].obsm['spatial'], 
                ligands_df=get_filtered_df(gex_df, cell_thresholds, genes),
                lr_info=self.lr,
                scale_factor=self.scale_factor
        )
        else:
            weighted_ligands = pd.DataFrame(index=self.obs_names)
        
        return weighted_ligands
    
    def update_status(self, msg='', color='black_on_green'):
        self.status.update(msg)
        self.status.color = color
        self.status.refresh()

    def _get_wbetas_dict(
        self, 
        betas_dict, 
        weighted_ligands, 
        weighted_ligands_tfl, 
        gene_mtx, 
        cell_thresholds):

        gex_df = get_filtered_df(       # mask out receptors too
            counts_df=pd.DataFrame(
                gene_mtx, 
                index=self.obs_names, 
                columns=self.adata.var_names
            ),
            cell_thresholds=cell_thresholds,
            genes=self.adata.var_names
        )[self.adata.var_names] 
        
        self.update_status(
            f'[{self.iter}/{self.max_iter}] | Computing Ligand interactions', 
            color='black_on_salmon')
        
        out_dict = {}
        
        for i, (gene, betadata) in enumerate(betas_dict.data.items()):

            if self.co_grn_links is not None:
                grn_tfs = self.co_grn_links.loc[self.co_grn_links['source'] == gene, 'target'].values
            else:
                grn_tfs = None

            out_dict[gene] = self._combine_gene_wbetas(
                weighted_ligands, weighted_ligands_tfl, gex_df, betadata, grn_tfs=grn_tfs)
            
            self.update_status(
                f'{self.current_target} | {i}/{len(betas_dict.data)} | [{self.iter}/{self.max_iter}] | Computing Ligand interactions', color='black_on_salmon')
            
        self.update_status(f'Ligand interactions - Done')

        return out_dict

    def _combine_gene_wbetas(self, rw_ligands, rw_ligands_tfl, filtered_df, betadata, grn_tfs=None):
        betas_df = betadata.splash(
            rw_ligands, 
            rw_ligands_tfl, 
            filtered_df,
            scale_factor=self.beta_scale_factor,
            beta_cap=self.beta_cap,
            grn_tfs=grn_tfs
        )
        
        return betas_df
        
    def _get_spatial_betas_dict(self, subsample=None, float16=False, obs_names=None):
        bdb = Betabase(self.adata, self.save_dir, subsample=subsample, float16=float16, obs_names=obs_names)
        self.ligands = list(bdb.ligands_set)
        self.tfl_ligands = list(bdb.tfl_ligands_set)

        return bdb
    
    def splash_betas(self, gene, obs_names=None):
        
        assert gene in self.adata.var_names
        if obs_names is None:
            obs_names = self.adata.obs_names
        
        rw_ligands = self.adata.uns.get('received_ligands').loc[obs_names]
        rw_tfligands = self.adata.uns.get('received_ligands_tfl').loc[obs_names]
        gene_mtx = self.adata.to_df(layer='imputed_count').loc[obs_names].values
        cell_thresholds = self.adata.uns.get('cell_thresholds').loc[obs_names]
        
        filtered_df = get_filtered_df(
            counts_df=pd.DataFrame(
                gene_mtx, 
                index=obs_names, 
                columns=self.adata.var_names
            ),
            cell_thresholds=cell_thresholds,
            genes=self.adata.var_names
        )[self.adata.var_names].loc[obs_names]
        
        betadata = self.load_betadata(gene, self.save_dir, obs_names=obs_names)
        
        return self._combine_gene_wbetas(
            rw_ligands, rw_tfligands, filtered_df, betadata)
    
    
    def _perturb_all_cells(self, gex_delta, betas_dict):
        n_obs, n_genes = gex_delta.shape
        result = np.zeros((n_obs, n_genes))
        n_vars = len(self.adata.var_names)

        for i, gene in enumerate(self.adata.var_names):
            self.update_status(
                f'[{self.iter}/{self.max_iter}] | Perturbing 🧬️🐝️ {i+1}/{n_vars} ', 
                color='black_on_cyan'
            )
            
            _beta_out = betas_dict.get(gene, None)

            if _beta_out is not None:
                mod_idx = self.beta_dict.data[gene].modulator_gene_indices
                result[:, i] = np.sum(_beta_out.values * gex_delta[:, mod_idx], axis=1)
                
        assert not np.isnan(result).any(), "NaN values found in delta_simulated"
        
        return result

    def perturb(
        self, 
        target, 
        n_propagation=4, 
        gene_expr=0, 
        cells=None, 
        save_layer=False,
        delta_dir=None,
        ):
        
        payload_dict = {}
        output_name = None
        
        if isinstance(target, str):
            assert isinstance(gene_expr, (int, float))
            assert target in self.adata.var_names
            payload_dict[target] = gene_expr
            output_name = f'{target}_{n_propagation}n_{round(gene_expr, 2)}x'
            
        elif isinstance(target, list) and isinstance(gene_expr, list):
            assert len(target) == len(gene_expr)
            payload_dict = {t: g for t, g in zip(target, gene_expr)}
            output_name = '_'.join([f'{t}_{n_propagation}n_{round(g, 2)}x' for t, g in zip(target, gene_expr)])
        else:
            raise ValueError(f'Invalid target info')
        
        self.current_target = output_name
        
        obs = self.obs_names
        
        gene_mtx = self.adata.to_df(layer='imputed_count').loc[obs]
        self.payload_dict = payload_dict

        if isinstance(gene_mtx, pd.DataFrame):
            gene_mtx = gene_mtx.values
            
        simulation_input = gene_mtx.copy()

        for target, gene_expr in self.payload_dict.items():
            assert gene_expr >= 0
            assert target in self.adata.var_names
            target_index = self.gene2index[target]  

            if cells is None:
                simulation_input[:, target_index] = gene_expr   
            else:
                # cells is a list of cell indices
                simulation_input[cells, target_index] = gene_expr
        
        delta_input = simulation_input - gene_mtx
        delta_simulated = delta_input.copy() 

        if self.beta_dict is None:
            self.beta_dict = self._get_spatial_betas_dict(obs_names=self.obs_names) 

        # get LR specific filtered gex contributions
        cell_thresholds = self.adata.uns.get('cell_thresholds').loc[obs]
        if cell_thresholds is not None:
            cell_thresholds = cell_thresholds.reindex(              
                index=obs, columns=self.adata.var_names, fill_value=1)
            self.adata.uns['cell_thresholds'] = cell_thresholds
        else:
            print('warning: cell_thresholds not found in adata.uns')

        rw_ligands_0 = self.adata.uns.get('received_ligands')
        rw_tfligands_0 = self.adata.uns.get('received_ligands_tfl')
        
        if rw_ligands_0 is None or rw_tfligands_0 is None:
            rw_ligands_0 = self._compute_weighted_ligands(
                gene_mtx, cell_thresholds, genes=self.ligands)
            rw_tfligands_0 = self._compute_weighted_ligands(
                gene_mtx, cell_thresholds=None, genes=self.tfl_ligands)
            self.adata.uns['received_ligands'] = rw_ligands_0
            self.adata.uns['received_ligands_tfl'] = rw_tfligands_0
        
        # Make sure that the received ligands are indexed to the same cells as in the adata
        rw_ligands_0 = self.adata.uns['received_ligands'].reindex(
            index=obs, 
            columns=self.adata.var_names, 
            fill_value=0
        )
        rw_tfligands_0 = self.adata.uns['received_ligands_tfl'].reindex(
            index=obs, 
            columns=self.adata.var_names, 
            fill_value=0
        )
        
        all_ligands = list(set(self.ligands) | set(self.tfl_ligands))
        ligands_0 = self.adata.to_df(layer='imputed_count')[all_ligands].reindex(
            index=self.obs_names, 
            columns=self.adata.var_names, 
            fill_value=0
        )

        # copy the original values
        gene_mtx_1 = gene_mtx.copy()
        rw_ligands_1 = rw_ligands_0.copy()
        rw_tfligands_1 = rw_tfligands_0.copy()

        # get the max weighted ligand expression (could be zeroed out in rw_ligands_0)
        rw_ligands_0 = pd.concat(
            [rw_ligands_0, rw_tfligands_0], axis=1
        ).groupby(level=0, axis=1).max().reindex(
            index=obs, 
            columns=self.adata.var_names, 
            fill_value=0
        )

        self.iter = 0
        self.max_iter = n_propagation
        min_ = gene_mtx.min(axis=0)
        max_ = gene_mtx.max(axis=0)
        
        ## refer: src/celloracle/trajectory/oracle_GRN.py

        for n in range(n_propagation):
            self.iter+=1
            self.update_status(
                f'{target} -> {gene_expr} - {n+1}/{n_propagation}', 
                color='black_on_salmon')

            # weight betas by the gene expression from the previous iteration
            splashed_beta_dict = self._get_wbetas_dict(
                self.beta_dict, rw_ligands_1, rw_tfligands_1, gene_mtx_1, cell_thresholds)
            
            # get updated gene expressions
            gene_mtx_1 = gene_mtx + delta_simulated
            w_ligands_1 = self._compute_weighted_ligands(
                gene_mtx_1, cell_thresholds, genes=self.ligands)
            w_tfligands_1 = self._compute_weighted_ligands(
                gene_mtx_1, cell_thresholds=None, genes=self.tfl_ligands)

            # update deltas to reflect change in received ligands
            # we consider dy/dwL: we replace delta l with delta wL in delta_simulated
            rw_ligands_1 = pd.concat(
                [w_ligands_1, w_tfligands_1], axis=1
            ).groupby(level=0, axis=1).max().reindex(      # w_ligands <= w_tfligands because of cell_thresholds
                index=self.obs_names, 
                columns=self.adata.var_names, 
                fill_value=0
            )

            delta_rw_ligands = rw_ligands_1.values - rw_ligands_0.values

            # get the change in ligand expression within the gene_df that should be replaced with rw_ligand
            gene_df_1 = pd.DataFrame(
                gene_mtx_1,
                columns=self.adata.var_names,
                index=obs
            )

            ligands_1 = pd.concat(
                [gene_df_1[self.ligands], gene_df_1[self.tfl_ligands]], axis=1
            ).groupby(level=0, axis=1).max().reindex(
                index=obs, 
                columns=self.adata.var_names, 
                fill_value=0
            )

            delta_ligands = ligands_1.values - ligands_0.values
            
            # the model sees delta wL, not delta L
            # delta_simulated contains delta L, so remove and replace with wL
            delta_simulated = delta_simulated + delta_rw_ligands - delta_ligands
            _simulated = self._perturb_all_cells(delta_simulated, splashed_beta_dict)
            delta_simulated = np.array(_simulated)
            
            # ensure values in delta_simulated match our desired KO / input
            delta_simulated = np.where(delta_input != 0, delta_input, delta_simulated)

            # Don't allow simulated to exceed observed values
            gem_tmp = gene_mtx + delta_simulated
            gem_tmp = pd.DataFrame(gem_tmp).clip(lower=min_, upper=max_, axis=1).values

            delta_simulated = gem_tmp - gene_mtx # update delta_simulated in case of negative values
            
            if delta_dir:
                os.makedirs(delta_dir, exist_ok=True)
                np.save(
                    f'{delta_dir}/{target}_{n}n_{gene_expr}x.npy', 
                    gene_mtx + delta_simulated
                )
            
            del splashed_beta_dict
            gc.collect()

        gem_simulated = gene_mtx + delta_simulated
        assert gem_simulated.shape == gene_mtx.shape

        for target_name, target_gene_expr in self.payload_dict.items():
            target_index = self.gene2index[target_name]  

            if cells is None:
                gem_simulated[:, target_index] = target_gene_expr   
            else:
                gem_simulated[cells, target_index] = target_gene_expr

            self.update_status(
                f'{target_name} -> {target_gene_expr} - {n_propagation}/{n_propagation} - Done')
        
        if save_layer:
            self.adata.layers[output_name] = gem_simulated
            
        gex_out = pd.DataFrame(gem_simulated, index=obs, columns=self.adata.var_names)
        gex_out.index.name = output_name
            
        return gex_out
    
    @staticmethod
    def get_ko_data(perturb_dir, adata):
        files = [i.split('/')[-1].split('_')[0] for i in glob.glob(
            f'{perturb_dir}/*.parquet')]
        
        ko_data = []
        
        pbar = enlighten.get_manager().counter(
            total=len(files), 
            desc='Getting KO data', 
            unit='genes',
            color='orange',
            autorefresh=True,
        )

        for kotarget in files:
            pbar.desc = f'Getting KO data - {kotarget}'
            pbar.refresh()
            data = pd.read_parquet(f'{perturb_dir}/{kotarget}_4n_0x.parquet')
            data = data.loc[adata.obs_names] - adata.to_df(layer='imputed_count')
            data = data.join(adata.obs.cell_type).groupby('cell_type').mean().abs().mean(axis=1)

            ds = {}
            for k, v in data.sort_values(ascending=False).to_dict().items():
                ds[k] = v

            data = pd.DataFrame.from_dict(ds, orient='index')
            data.columns = [kotarget]
            ko_data.append(data)
            pbar.update()
            
        pbar.close()
        
        return pd.concat(ko_data, axis=1)
    
    def perturb_batch(
        self, 
        target_genes, 
        save_to=None, 
        n_propagation=4, 
        gene_expr=0, 
        cells=None):
        
        self.update_status(f'Batch Perturbation mode: {len(target_genes)} genes')

        progress_bar = self.manager.counter(
            total=len(target_genes), 
            desc=f'Batch Perturbations', 
            unit='genes',
            color='orange',
            autorefresh=True,
        )

        os.makedirs(save_to, exist_ok=True)
        
        for target in target_genes:
            progress_bar.desc = f'Batch Perturbation - {target}'
            progress_bar.refresh()
            
            gex_out =self.perturb(
                target=target, 
                n_propagation=n_propagation, 
                save_layer=False,
                gene_expr=gene_expr, 
                cells=cells, 
            )
                     
            progress_bar.update()

            if save_to is not None:
                file_name = f'{target}_{n_propagation}n_{gene_expr}x'
                gex_out.to_parquet(f'{save_to}/{file_name}.parquet')
                
        self.update_status('Batch Perturbation: Done')
        progress_bar.close()
        
    @property
    def possible_targets(self):
        return list(set.union(
            self.beta_dict.receptors_set, 
            self.beta_dict.ligands_set, 
            self.beta_dict.tfs_set
        ))
        
    def genome_screen(
        self, save_to, n_propagation=4, priority_genes=None, mode='knockout', cells=None):
        """
        Perform a genome-wide knockout or overexpression of the target genes
        """
        
        assert mode in ['knockout', 'overexpress']
        
        if priority_genes is not None:
            priority_genes = list(np.intersect1d(priority_genes, self.possible_targets))
        
        screen_queue = OracleQueue(
            save_to, 
            all_genes=self.possible_targets,
            priority_genes=priority_genes,
            lock_timeout=3600
        )
        
        _manager = enlighten.get_manager()
        
        
        gene_bar = _manager.counter(
            total=len(screen_queue.all_genes), 
            desc=f'... initializing ...', 
            unit='genes',
            color='orange',
            autorefresh=True,
        )
        
        screen_queue.kill_old_locks()
        
        max_expr = self.adata.to_df(layer='imputed_count').max().to_dict()
        
        while not screen_queue.is_empty:
            target = next(screen_queue)
            
            gene_bar.count = len(screen_queue.all_genes) - len(screen_queue.remaining_genes)
            gene_bar.desc = f'🕵️️  {screen_queue.agents+1} agents'
            gene_bar.refresh()
            
            if os.path.exists(f'{screen_queue.model_dir}/{target}.lock'):
                print(f'Found duplicate lock for {target} - skipping')
                continue

            screen_queue.create_lock(target)
            
            gex_out = self.perturb(
                target=target, 
                n_propagation=n_propagation, 
                gene_expr=0 if mode == 'knockout' else max_expr[target], 
                cells=cells, 
                delta_dir=None
            )
            
            screen_queue.delete_lock(target)
            if screen_queue.last_refresh_age() > screen_queue.lock_timeout:
                screen_queue.kill_old_locks()
                screen_queue.last_refresh_on = datetime.datetime.now()
        
            gene_bar.update()

            # suffix = '0x' if mode == 'knockout' else f'{round(max_expr[target], 2)}x'
            suffix = '0x' if mode == 'knockout' else 'maxx'
            file_name = f'{target}_{n_propagation}n_{suffix}'
            gex_out.to_parquet(
                f'{save_to}/{file_name}.parquet')
                

    def perturb_track(
        self, 
        target, 
        n_propagation=4, 
        gene_expr=0, 
        cells=None, 
        save_layer=False,
        delta_dir=None,
        ):

        
        payload_dict = {}
        output_name = None
        
        if isinstance(target, str):
            assert isinstance(gene_expr, (int, float))
            assert target in self.adata.var_names
            payload_dict[target] = gene_expr
            output_name = f'{target}_{n_propagation}n_{round(gene_expr, 2)}x'
            
        elif isinstance(target, list) and isinstance(gene_expr, list):
            assert len(target) == len(gene_expr)
            payload_dict = {t: g for t, g in zip(target, gene_expr)}
            output_name = '_'.join([f'{t}_{n_propagation}n_{round(g, 2)}x' for t, g in zip(target, gene_expr)])
        else:
            raise ValueError(f'Invalid target info')
        
        self.current_target = output_name
        obs = self.obs_names
        gene_mtx = self.adata.to_df(layer='imputed_count').loc[obs]
        self.payload_dict = payload_dict

        gradients = {}

        if isinstance(gene_mtx, pd.DataFrame):
            gene_mtx = gene_mtx.values
            
        simulation_input = gene_mtx.copy()

        for target, gene_expr in self.payload_dict.items():
            assert gene_expr >= 0
            assert target in self.adata.var_names
            target_index = self.gene2index[target]  

            if cells is None:
                simulation_input[:, target_index] = gene_expr   
            else:
                # cells is a list of cell indices
                simulation_input[cells, target_index] = gene_expr
        
        delta_input = simulation_input - gene_mtx
        delta_simulated = delta_input.copy() 

        if self.beta_dict is None:
            self.beta_dict = self._get_spatial_betas_dict(obs_names=self.obs_names) 

        # get LR specific filtered gex contributions
        cell_thresholds = self.adata.uns.get('cell_thresholds').loc[obs]
        if cell_thresholds is not None:
            cell_thresholds = cell_thresholds.reindex(              
                index=obs, columns=self.adata.var_names, fill_value=1)
            self.adata.uns['cell_thresholds'] = cell_thresholds
        else:
            print('warning: cell_thresholds not found in adata.uns')

        rw_ligands_0 = self.adata.uns.get('received_ligands')
        rw_tfligands_0 = self.adata.uns.get('received_ligands_tfl')
        
        if rw_ligands_0 is None or rw_tfligands_0 is None:
            rw_ligands_0 = self._compute_weighted_ligands(
                gene_mtx, cell_thresholds, genes=self.ligands)
            rw_tfligands_0 = self._compute_weighted_ligands(
                gene_mtx, cell_thresholds=None, genes=self.tfl_ligands)
            self.adata.uns['received_ligands'] = rw_ligands_0
            self.adata.uns['received_ligands_tfl'] = rw_tfligands_0
        
        # Make sure that the received ligands are indexed to the same cells as in the adata
        rw_ligands_0 = self.adata.uns['received_ligands'].reindex(
            index=obs, 
            columns=self.adata.var_names, 
            fill_value=0
        )
        rw_tfligands_0 = self.adata.uns['received_ligands_tfl'].reindex(
            index=obs, 
            columns=self.adata.var_names, 
            fill_value=0
        )

        all_ligands = list(set(self.ligands) | set(self.tfl_ligands))
        ligands_0 = self.adata.to_df(layer='imputed_count')[all_ligands].reindex(
            index=self.obs_names, 
            columns=self.adata.var_names, 
            fill_value=0
        )

        # copy the original values
        gene_mtx_1 = gene_mtx.copy()
        rw_ligands_1 = rw_ligands_0.copy()
        rw_tfligands_1 = rw_tfligands_0.copy()

        # get the max weighted ligand expression (could be zeroed out in rw_ligands_0)
        rw_ligands_0 = pd.concat(
            [rw_ligands_0, rw_tfligands_0], axis=1
        ).groupby(level=0, axis=1).max().reindex(
            index=obs, 
            columns=self.adata.var_names, 
            fill_value=0
        )

        self.iter = 0
        self.max_iter = n_propagation
        min_ = gene_mtx.min(axis=0)
        max_ = gene_mtx.max(axis=0)
        
        ## refer: src/celloracle/trajectory/oracle_GRN.py

        for n in range(n_propagation):
            self.iter+=1
            self.update_status(
                f'{target} -> {gene_expr} - {n+1}/{n_propagation}', 
                color='black_on_salmon')

            # weight betas by the gene expression from the previous iteration
            splashed_beta_dict = self._get_wbetas_dict(
                self.beta_dict, rw_ligands_1, rw_tfligands_1, gene_mtx_1, cell_thresholds)
            
            # get updated gene expressions
            gene_mtx_1 = gene_mtx + delta_simulated
            w_ligands_1 = self._compute_weighted_ligands(
                gene_mtx_1, cell_thresholds, genes=self.ligands)
            w_tfligands_1 = self._compute_weighted_ligands(
                gene_mtx_1, cell_thresholds=None, genes=self.tfl_ligands)

            # update deltas to reflect change in received ligands
            # we consider dy/dwL: we replace delta l with delta wL in delta_simulated
            rw_ligands_1 = pd.concat(
                [w_ligands_1, w_tfligands_1], axis=1
            ).groupby(level=0, axis=1).max().reindex(      # w_ligands <= w_tfligands because of cell_thresholds
                index=self.obs_names, 
                columns=self.adata.var_names, 
                fill_value=0
            )

            delta_rw_ligands = rw_ligands_1.values - rw_ligands_0.values

            # get the change in ligand expression within the gene_df that should be replaced with rw_ligand
            gene_df_1 = pd.DataFrame(
                gene_mtx_1,
                columns=self.adata.var_names,
                index=obs
            )

            ligands_1 = pd.concat(
                [gene_df_1[self.ligands], gene_df_1[self.tfl_ligands]], axis=1
            ).groupby(level=0, axis=1).max().reindex(
                index=obs, 
                columns=self.adata.var_names, 
                fill_value=0
            )

            delta_ligands = ligands_1.values - ligands_0.values
            
            # the model sees delta wL, not delta L
            # delta_simulated contains delta L, so remove and replace with wL
            delta_simulated = delta_simulated + delta_rw_ligands - delta_ligands
            if n == n_propagation - 1:
                _simulated, gradients = self._perturb_all_cells_track(delta_simulated, splashed_beta_dict, gradients)
            else:
                _simulated = self._perturb_all_cells(delta_simulated, splashed_beta_dict)

            delta_simulated = np.array(_simulated)
            
            # ensure values in delta_simulated match our desired KO / input
            delta_simulated = np.where(delta_input != 0, delta_input, delta_simulated)

            # Don't allow simulated to exceed observed values
            gem_tmp = gene_mtx + delta_simulated
            gem_tmp = pd.DataFrame(gem_tmp).clip(lower=min_, upper=max_, axis=1).values

            delta_simulated = gem_tmp - gene_mtx # update delta_simulated in case of negative values
            
            if delta_dir:
                os.makedirs(delta_dir, exist_ok=True)
                np.save(
                    f'{delta_dir}/{target}_{n}n_{gene_expr}x.npy', 
                    gene_mtx + delta_simulated
                )
            
            del splashed_beta_dict
            gc.collect()

        gem_simulated = gene_mtx + delta_simulated
        assert gem_simulated.shape == gene_mtx.shape

        for target_name, target_gene_expr in self.payload_dict.items():
            target_index = self.gene2index[target_name]  

            if cells is None:
                gem_simulated[:, target_index] = target_gene_expr   
            else:
                gem_simulated[cells, target_index] = target_gene_expr

            self.update_status(
                f'{target_name} -> {target_gene_expr} - {n_propagation}/{n_propagation} - Done')
        
        if save_layer:
            self.adata.layers[output_name] = gem_simulated
            
        gex_out = pd.DataFrame(gem_simulated, index=obs, columns=self.adata.var_names)
        gex_out.index.name = output_name
            
        return gex_out, gradients

    def _perturb_all_cells_track(self, gex_delta, betas_dict, gradients):
        n_obs, n_genes = gex_delta.shape
        result = np.zeros((n_obs, n_genes))
        n_vars = len(self.adata.var_names)

        for i, gene in enumerate(self.adata.var_names):
            self.update_status(
                f'[{self.iter}/{self.max_iter}] | Perturbing 🧬️🐝️ {i+1}/{n_vars} ', 
                color='black_on_cyan'
            )
            
            _beta_out = betas_dict.get(gene, None)

            if _beta_out is not None:
                mod_idx = self.beta_dict.data[gene].modulator_gene_indices
                grad = _beta_out * gex_delta[:, mod_idx]
                gradients[gene] = grad
                result[:, i] = np.sum(grad.values, axis=1)
                
        assert not np.isnan(result).any(), "NaN values found in delta_simulated"
        return result, gradients
