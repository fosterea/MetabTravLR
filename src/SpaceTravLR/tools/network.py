# import celloracle as co
import numpy as np
import pandas as pd
import pickle
import os
import json 
import torch
import networkx as nx 

def get_human_housekeeping_genes():
    return pd.read_csv('https://housekeeping.unicamp.br/Housekeeping_GenesHuman.csv', sep=';')

def get_mouse_housekeeping_genes():
    return pd.read_csv('https://housekeeping.unicamp.br/Housekeeping_GenesMouse.csv', sep=';')


def get_cellchat_db(species):
    
    # import commot as ct
    # df_ligrec = ct.pp.ligand_receptor_database(
    #         database='CellChat', 
    #         species=species, 
    #         signaling_type=None
    #     )
        
    # df_ligrec.columns = ['ligand', 'receptor', 'pathway', 'signaling']  
    # return df_ligrec

    # data_path = os.path.join(
    #     os.path.dirname(__file__), '..', '..', '..', 'data', f'cellchat_{species}.csv')

    data_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'SpaceTravLR_data', f'cellchat_{species}.csv')
    return pd.read_csv(data_path)

def expand_paired_interactions(df):
    expanded_rows = []
    for _, row in df.iterrows():
        ligands = row['ligand'].split('_')
        receptors = row['receptor'].split('_')
        
        for ligand in ligands:
            for receptor in receptors:
                new_row = row.copy()
                new_row['ligand'] = ligand
                new_row['receptor'] = receptor
                expanded_rows.append(new_row)
    
    df = pd.DataFrame(expanded_rows)
    
    return df

def encode_labels(labels, reverse_dict=False):
    unique_labels = sorted(list(set(labels)))
    if reverse_dict:
        return {label: i for i, label in enumerate(unique_labels)}
    return {i: label for i, label in enumerate(unique_labels)}



class GeneRegulatoryNetwork:
    '''
    A completely unused class, but used for testing
    '''
    def __init__(self, organism='mouse'):
        if organism == 'mouse':
            # self.data = co.data.load_mouse_scATAC_atlas_base_GRN()
            
            # data_path = os.path.join(
            #     os.path.dirname(__file__), '..', '..', '..', 'data', 'mm9_mouse_atac_atlas_data_TSS.parquet')
            data_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'SpaceTravLR_data', 'mouse_base_grn.parquet')
            self.data = pd.read_parquet(data_path)
    
    def get_regulators(self, adata, target_gene):
        base_GRN = self.data
        
        df = base_GRN[base_GRN.gene_short_name==target_gene][
            np.intersect1d(adata.var_names, base_GRN[base_GRN.gene_short_name==target_gene].columns)].sum()
        df = df[df!=0]
        
        return df.index.tolist()
            

class CellOracleLinks:
    
    def __init__(self):
        pass

    def get_regulators(self, adata, target_gene, alpha=0.05):
        regulators_with_pvalues = self.get_regulators_with_pvalues(adata, target_gene, alpha)
        grouped_regulators = regulators_with_pvalues.groupby('source').mean()
        filtered_regulators = grouped_regulators[grouped_regulators.index.isin(adata.var_names)]

        return filtered_regulators.index.tolist()
    
    def get_targets(self, adata, tf, alpha=0.05):
        targets_with_pvalues = self.get_targets_with_pvalues(adata, tf, alpha)
        grouped_targets = targets_with_pvalues.groupby('target').mean()
        filtered_targets = grouped_targets[grouped_targets.index.isin(adata.var_names)]

        return filtered_targets.index.tolist()

    def get_regulators_with_pvalues(self, adata, target_gene, alpha=0.05):
        assert target_gene in adata.var_names, f'{target_gene} not in adata.var_names'
        co_links = pd.concat(
            [link_data.query(f'target == "{target_gene}" and p < {alpha}')[['source', 'coef_mean']] 
                for link_data in self.links.values()], axis=0).reset_index(drop=True)
        return co_links.query(f'source.isin({str(list(adata.var_names))})').reset_index(drop=True)
    
    def get_targets_with_pvalues(self, adata, tf, alpha=0.05):
        assert tf in adata.var_names, f'{tf} not in adata.var_names'
        co_links = pd.concat(
            [link_data.query(f'source == "{tf}" and p < {alpha}')[['target', 'coef_mean']] 
                for link_data in self.links.values()], axis=0).reset_index(drop=True)
        return co_links.query(f'target.isin({str(list(adata.var_names))})').reset_index(drop=True)
    
    @staticmethod
    def get_training_genes(co_links, gene_kos, n_propagation=3):
        grn = nx.DiGraph()
        edges = []

        for cluster, df in co_links.items():
            cluster_edges = [(u, v) for u, v in zip(df['source'], df['target'])]
            edges.extend(cluster_edges)
        
        grn.add_edges_from(edges)
        train_genes = []
        for ko in gene_kos:
            genes = [node for node, distance in nx.single_source_shortest_path_length(
                                                grn, ko, cutoff=n_propagation).items()]
            train_genes.extend(genes)
        
        return np.unique(train_genes)


class RegulatoryFactory(CellOracleLinks):
    def __init__(self, colinks_path=None, links=None, annot='cell_type_int'):
        self.colinks_path = colinks_path
        self.annot = annot

        if colinks_path is not None:
            with open(self.colinks_path, 'rb') as f:
                self.links = pickle.load(f)

        elif links is not None:
            self.links = links

        self.cluster_labels = encode_labels(self.links.keys())
        
 
    def get_cluster_regulators(self, adata, target_gene, alpha=0.05):
        adata_clusters = np.unique(adata.obs[self.annot])
        regulator_dict = {}
        all_regulators = set()

        for label in adata_clusters:
            grn_df = self.links[label]

            grn_df = grn_df[(grn_df.target == target_gene) & (grn_df.p <= alpha)]
            tfs = list(grn_df.source)
            
            regulator_dict[label] = tfs
            all_regulators.update(tfs)

        all_regulators = all_regulators & set(adata.to_df().columns) # only use genes also in adata
        all_regulators = sorted(list(all_regulators))
        regulator_masks = {}

        for label, tfs in regulator_dict.items():
            indices = [all_regulators.index(tf)+1 for tf in tfs if tf in all_regulators]
            
            mask = torch.zeros(len(all_regulators) + 1)     # prepend 1 for beta0
            mask[[0] + indices] = 1 
            regulator_masks[label] = mask

        self.regulator_dict = regulator_masks

        return all_regulators
    
class GeneralRegulatoryNetwork(CellOracleLinks):
    def __init__(self, colinks_path, annot):
        '''
        A general class for loading CellOracle GRNs
        Assumes lexicographical order for cluster labels
        '''

        assert colinks_path.endswith('.pkl'), 'colinks_path should be a pickle file'
        with open(colinks_path, 'rb') as f:
            self.links = pickle.load(f)
        
        self.annot = annot
    
    def get_cluster_regulators(self, adata, target_gene, alpha=0.05):

        adata_clusters = np.unique(adata.obs[self.annot])
        regulator_dict = {}
        all_regulators = set()

        for cluster, label in enumerate(adata_clusters):
            grn_df = self.links[cluster]

            grn_df = grn_df[(grn_df.target == target_gene) & (grn_df.p <= alpha)]
            tfs = list(grn_df.source)
            
            regulator_dict[label] = tfs
            all_regulators.update(tfs)

        all_regulators = all_regulators & set(adata.to_df().columns) # only use genes also in adata
        all_regulators = sorted(list(all_regulators))
        regulator_masks = {}

        for label, tfs in regulator_dict.items():
            indices = [all_regulators.index(tf)+1 for tf in tfs if tf in all_regulators]
            
            mask = torch.zeros(len(all_regulators) + 1)     # prepend 1 for beta0
            mask[[0] + indices] = 1 
            regulator_masks[label] = mask

        self.regulator_dict = regulator_masks

        return all_regulators


        
class SurveyRegulatoryNetwork(CellOracleLinks):
    def __init__(self):
        self.base_pth = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'data')

        with open(self.base_pth+'/survey/celloracle_links_spleen.pkl', 'rb') as f:
            self.links = pickle.load(f)

        self.cluster_labels = {
            '8': 'T',
            '4': 'Neutrophil',
            '5': 'Plasma_Cell',
            '0': 'B',
            '2': 'Macrophage',
            '3': 'NK',
            '6': 'Platelet',
            '7': 'RBC',
            '1': 'DC'
        }

        self.annot = 'cluster'



    def get_cluster_regulators(self, adata, target_gene, alpha=0.05):
        adata_clusters = np.unique(adata.obs[self.annot])
        regulator_dict = {}
        all_regulators = set()

        for label in adata_clusters:
            cluster = self.cluster_labels[str(label)]
            grn_df = self.links[cluster]

            grn_df = grn_df[(grn_df.target == target_gene) & (grn_df.p <= alpha)]
            tfs = list(grn_df.source)
            
            regulator_dict[label] = tfs
            all_regulators.update(tfs)

        all_regulators = all_regulators & set(adata.to_df().columns) # only use genes also in adata
        all_regulators = sorted(list(all_regulators))
        regulator_masks = {}

        for label, tfs in regulator_dict.items():
            indices = [all_regulators.index(tf)+1 for tf in tfs if tf in all_regulators]
            
            mask = torch.zeros(len(all_regulators) + 1)     # prepend 1 for beta0
            mask[[0] + indices] = 1 
            regulator_masks[label] = mask

        self.regulator_dict = regulator_masks

        return all_regulators
    


class DayThreeRegulatoryNetwork(CellOracleLinks):
    """
    CellOracle infered GRN 
    These are dataset specific and come with estimated betas and p-values
    """

    def __init__(self):

        self.base_pth = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'data')

        with open(self.base_pth+'/slideseq/celloracle_links_day3_1.pkl', 'rb') as f:
            self.links = pickle.load(f)

        self.annot = 'rctd_cluster'

        with open(os.path.join(self.base_pth, 'celltype_assign.json'), 'r') as f:
            self.cluster_labels = json.load(f)


    
    def get_cluster_regulators(self, adata, target_gene, alpha=0.05):
        adata_clusters = np.unique(adata.obs[self.annot])
        regulator_dict = {}
        all_regulators = set()

        for label in adata_clusters:
            # cluster = self.cluster_labels[str(label)]
            cluster = label
            grn_df = self.links[cluster]

            grn_df = grn_df[(grn_df.target == target_gene) & (grn_df.p <= alpha)]
            tfs = list(grn_df.source)
            
            regulator_dict[label] = tfs
            all_regulators.update(tfs)

        all_regulators = all_regulators & set(adata.to_df().columns) # only use genes also in adata
        all_regulators = sorted(list(all_regulators))
        regulator_masks = {}

        for label, tfs in regulator_dict.items():
            indices = [all_regulators.index(tf)+1 for tf in tfs if tf in all_regulators]
            
            mask = torch.zeros(len(all_regulators) + 1)     # prepend 1 for beta0
            mask[[0] + indices] = 1 
            regulator_masks[label] = mask

        self.regulator_dict = regulator_masks

        return all_regulators
    

class MouseKidneyRegulatoryNetwork(CellOracleLinks):
    def __init__(self):

        self.base_pth = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'data')

        with open(self.base_pth+'/kidney/celloracle_links.pkl', 'rb') as f:
            self.links = pickle.load(f)

        self.annot = 'cluster_cat'

        with open(os.path.join(self.base_pth, 'kidney/celltype_assign.json'), 'r') as f:
            self.cluster_labels = json.load(f)

    
    def get_cluster_regulators(self, adata, target_gene, alpha=0.05):
        adata_clusters = np.unique(adata.obs[self.annot])
        regulator_dict = {}
        all_regulators = set()

        for label in adata_clusters:
            cluster = self.cluster_labels[str(label)]
            grn_df = self.links[cluster]

            grn_df = grn_df[(grn_df.target == target_gene) & (grn_df.p <= alpha)]
            tfs = list(grn_df.source)
            
            regulator_dict[label] = tfs
            all_regulators.update(tfs)

        all_regulators = all_regulators & set(adata.to_df().columns) # only use genes also in adata
        all_regulators = sorted(list(all_regulators))
        regulator_masks = {}

        for label, tfs in regulator_dict.items():
            indices = [all_regulators.index(tf)+1 for tf in tfs if tf in all_regulators]
            
            mask = torch.zeros(len(all_regulators) + 1)     # prepend 1 for beta0
            mask[[0] + indices] = 1 
            regulator_masks[label] = mask

        self.regulator_dict = regulator_masks

        return all_regulators
    

class MouseSpleenRegulatoryNetwork(CellOracleLinks):
    def __init__(self):

        self.base_pth = '/ix/djishnu/alw399/SpaceOracle/data'

        with open(self.base_pth+'/spleen/celloracle_links.pkl', 'rb') as f:
            self.links = pickle.load(f)

        self.annot = 'clusters'

        self.cluster_labels = {
            "0": "B cell",
            "1": "CD4 T cell",
            "2": "CD8 T cell",
            "3": "Memory T cell",
            "4": "Myeloid"
        }
    
    def get_cluster_regulators(self, adata, target_gene, alpha=0.05):
        adata_clusters = np.unique(adata.obs[self.annot])
        regulator_dict = {}
        all_regulators = set()

        for label in adata_clusters:
            cluster = self.cluster_labels[str(label)]
            # cluster = str(label)
            grn_df = self.links[cluster]

            grn_df = grn_df[(grn_df.target == target_gene) & (grn_df.p <= alpha)]
            tfs = list(grn_df.source)
            
            regulator_dict[label] = tfs
            all_regulators.update(tfs)

        all_regulators = all_regulators & set(adata.to_df().columns) # only use genes also in adata
        all_regulators = sorted(list(all_regulators))
        regulator_masks = {}

        for label, tfs in regulator_dict.items():
            indices = [all_regulators.index(tf)+1 for tf in tfs if tf in all_regulators]
            
            mask = torch.zeros(len(all_regulators) + 1)     # prepend 1 for beta0
            mask[[0] + indices] = 1 
            regulator_masks[label] = mask

        self.regulator_dict = regulator_masks

        return all_regulators



class HumanMelanomaRegulatoryNetwork(CellOracleLinks):
    def __init__(self):

        self.base_pth = '/ix/djishnu/alw399/SpaceOracle/data'

        with open(self.base_pth+'/melanoma/celloracle_links.pkl', 'rb') as f:
            self.links = pickle.load(f)

        self.annot = 'cluster_cat'

        self.cluster_labels = {
            '2': 'CD8_T',
            '10': 'tumour_1',
            '3': 'T_reg',
            '1': 'CD4_T',
            '7': 'mono-mac',
            '9': 'plasma',
            '8': 'pDC',
            '6': 'mDC',
            '5': 'fibroblast',
            '11': 'tumour_2',
            '0': 'B_cell',
            '4': 'endothelial'
        }
    
    def get_cluster_regulators(self, adata, target_gene, alpha=0.05):
        adata_clusters = np.unique(adata.obs[self.annot])
        regulator_dict = {}
        all_regulators = set()

        for label in adata_clusters:
            cluster = self.cluster_labels[str(label)]
            # cluster = str(label)
            grn_df = self.links[cluster]

            grn_df = grn_df[(grn_df.target == target_gene) & (grn_df.p <= alpha)]
            tfs = list(grn_df.source)
            
            regulator_dict[label] = tfs
            all_regulators.update(tfs)

        all_regulators = all_regulators & set(adata.to_df().columns) # only use genes also in adata
        all_regulators = sorted(list(all_regulators))
        regulator_masks = {}

        for label, tfs in regulator_dict.items():
            indices = [all_regulators.index(tf)+1 for tf in tfs if tf in all_regulators]
            
            mask = torch.zeros(len(all_regulators) + 1)     # prepend 1 for beta0
            mask[[0] + indices] = 1 
            regulator_masks[label] = mask

        self.regulator_dict = regulator_masks

        return all_regulators
    
    


class HumanTonsilNetwork(CellOracleLinks):
    def __init__(self):

        self.base_pth = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'data')

        with open(self.base_pth+'/BaseGRNs/tonsil_celloracle.pkl', 'rb') as f:
            self.links = pickle.load(f)

        self.annot = 'cluster'

        self.cluster_labels = {
            0: 'Plasma Cells',
            1: 'Cycling B Cells',
            2: 'Follicular Dendritic Cells ',
            3: 'Dark Zone B Cells',
            4: 'IFN B Cells',
            5: 'T Cells',
            6: 'Light Zone B Cells',
            7: 'Memory B Cells',
            8: 'Naive B Cells',
            9: 'GC-Tfh'
        }


    def get_cluster_regulators(self, adata, target_gene, alpha=0.05):
        adata_clusters = np.unique(adata.obs[self.annot])
        regulator_dict = {}
        all_regulators = set()

        for label in adata_clusters:
            cluster = self.cluster_labels[int(label)]
            grn_df = self.links[cluster]

            grn_df = grn_df[(grn_df.target == target_gene) & (grn_df.p <= alpha)]
            tfs = list(grn_df.source)
            
            regulator_dict[label] = tfs
            all_regulators.update(tfs)

        all_regulators = all_regulators & set(adata.to_df().columns) # only use genes also in adata
        all_regulators = sorted(list(all_regulators))
        regulator_masks = {}

        for label, tfs in regulator_dict.items():
            indices = [all_regulators.index(tf)+1 for tf in tfs if tf in all_regulators]
            
            mask = torch.zeros(len(all_regulators) + 1)     # prepend 1 for beta0
            mask[[0] + indices] = 1 
            regulator_masks[label] = mask

        self.regulator_dict = regulator_masks

        return all_regulators
    

class HumanTonsilRegulatoryNetwork(CellOracleLinks):
    def __init__(self):

        self.base_pth = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'data')

        with open(self.base_pth+'/slidetags/tonsil_colinks.pkl', 'rb') as f:
            self.links = pickle.load(f)

        self.annot = 'cell_type_int'
        
        self.cluster_labels = {0: 'B_germinal_center',
            1: 'B_memory', 
            2: 'B_naive',
            3: 'FDC',
            4: 'NK',
            5: 'T_CD4',
            6: 'T_CD8',
            7: 'T_double_neg',
            8: 'T_follicular_helper',
            9: 'mDC',
            10: 'myeloid',
            11: 'pDC',
            12: 'plasma'
        }

    def get_cluster_regulators(self, adata, target_gene, alpha=0.05):
        adata_clusters = np.unique(adata.obs[self.annot])
        regulator_dict = {}
        all_regulators = set()

        for label in adata_clusters:
            cluster = self.cluster_labels[label]
            # cluster = str(label)
            grn_df = self.links[cluster]

            grn_df = grn_df[(grn_df.target == target_gene) & (grn_df.p <= alpha)]
            tfs = list(grn_df.source)
            
            regulator_dict[label] = tfs
            all_regulators.update(tfs)

        all_regulators = all_regulators & set(adata.to_df().columns) # only use genes also in adata
        all_regulators = sorted(list(all_regulators))
        regulator_masks = {}

        for label, tfs in regulator_dict.items():
            indices = [all_regulators.index(tf)+1 for tf in tfs if tf in all_regulators]
            
            mask = torch.zeros(len(all_regulators) + 1)     # prepend 1 for beta0
            mask[[0] + indices] = 1 
            regulator_masks[label] = mask

        self.regulator_dict = regulator_masks

        return all_regulators