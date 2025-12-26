import unittest
import numpy as np
import pandas as pd 
import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import tempfile
import shutil
import pickle
from unittest.mock import patch, MagicMock, Mock
import scanpy as sc
import anndata as ad
from scipy import sparse
from SpaceTravLR.spaceship import SpaceShip, Status, catch_and_retry, catch_errors

def quick_normalize(adata):
    adata.layers['raw_count'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers['normalized_count'] = adata.X.copy()
    return adata


def create_test_adata(n_cells=100, n_genes=50, species='human'):
    np.random.seed(42)
    X = np.random.rand(n_cells, n_genes)
    
    if species == 'mouse':
        gene_names = [f'Gene{i}' for i in range(n_genes)]
    else:
        gene_names = [f'GENE{i}' for i in range(n_genes)]
    
    adata = ad.AnnData(X=X)
    adata.var_names = gene_names
    adata.obs_names = [f'cell_{i}' for i in range(n_cells)]
    
    cell_types = np.random.choice(['TypeA', 'TypeB', 'TypeC'], size=n_cells)
    adata.obs['cell_type'] = cell_types
    
    spatial_coords = np.random.rand(n_cells, 2) * 1000
    adata.obsm['spatial'] = spatial_coords
    
    adata.layers['raw_count'] = X.copy()
    adata.layers['normalized_count'] = X.copy()
    adata.layers['imputed_count'] = X.copy()

    
    return adata

def create_test_tfls(gene_names, n_ligands=10, n_tfs=10):
    cols = np.random.choice(gene_names, size=n_ligands, replace=False)
    tfs = np.random.choice(gene_names, size=n_tfs, replace=False)
    df_vals = np.random.rand(n_ligands, n_tfs)
    df = pd.DataFrame(df_vals, index=cols, columns=tfs)
    return df

class TestSpaceShip(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        ship = SpaceShip()
        assert ship.name == 'AlienTissue'
        assert ship.status == Status.BORN
        del ship
        
        ship = SpaceShip(name='TestShip')
        assert ship.name == 'TestShip'
        assert ship.status == Status.BORN
        del ship
    
    def test_process_adata_basic(self):
        os.makedirs('output/input_data', exist_ok=True)
        
        adata = create_test_adata(species='human')
        ship = SpaceShip()
        
        with patch('SpaceTravLR.tools.utils.scale_adata') as mock_scale, \
             patch('SpaceTravLR.tools.utils.is_mouse_data', return_value=False), \
             patch('SpaceTravLR.tools.network.encode_labels') as mock_encode, \
             patch('SpaceTravLR.oracles.BaseTravLR.impute_clusterwise') as mock_impute:
            
            mock_encode.return_value = {'TypeA': 0, 'TypeB': 1, 'TypeC': 2}
            mock_scale.return_value = adata
            
            ship.process_adata_(adata, annot='cell_type')
            
            assert ship.annot == 'cell_type'
            assert ship.species == 'human'
            assert os.path.exists('output/input_data/_adata.h5ad')
            assert 'cell_type_int' in ship.adata.obs.columns
    
    def test_process_adata_missing_spatial(self):
        adata = create_test_adata()
        del adata.obsm['spatial']
        ship = SpaceShip()
        
        with self.assertRaises(AssertionError):
            ship.process_adata_(adata)
    
    def test_process_adata_missing_annot(self):
        adata = create_test_adata()
        ship = SpaceShip()
        
        with self.assertRaises(AssertionError):
            ship.process_adata_(adata, annot='nonexistent')
    
    
    def test_load_base_cell_thresholds(self):
        adata = create_test_adata()
        ship = SpaceShip()
        ship.adata = adata
        ship.species = 'human'
        
        mock_df = pd.DataFrame({
            'ligand': ['Lig1', 'Lig2'],
            'receptor': ['Rec1', 'Rec2']
        })
        
        mock_expanded = pd.DataFrame({
            'ligand': ['Lig1', 'Lig2'],
            'receptor': ['Rec1', 'Rec2']
        })
        
        with patch('SpaceTravLR.spaceship.get_cellchat_db', return_value=mock_df), \
             patch('SpaceTravLR.spaceship.expand_paired_interactions', return_value=mock_expanded):
            
            result = ship.load_base_cell_thresholds()
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(adata.obs_names)
            assert set(result.columns) == {'Lig1', 'Lig2', 'Rec1', 'Rec2'}
    
    def test_load_base_GRN_mouse(self):
        mock_df = pd.DataFrame({
            'gene_short_name': ['Gene1', 'Gene2'],
            'TF1': [1, 0],
            'peak_id': ['peak1', 'peak2']
        })
        
        with patch('pandas.read_parquet', return_value=mock_df):
            result = SpaceShip.load_base_GRN('mouse')
            
            assert isinstance(result, pd.DataFrame)
    
    def test_load_base_GRN_invalid_species(self):
        with self.assertRaises(AssertionError):
            SpaceShip.load_base_GRN('invalid')
    
    def test_run_celloracle_(self):
        os.makedirs('output/input_data', exist_ok=True)
        
        adata = create_test_adata()
        adata.layers['raw_count'] = adata.X.copy()
        adata.obsm['X_umap'] = np.random.rand(len(adata), 2)
        
        ship = SpaceShip()
        ship.adata = adata
        ship.annot = 'cell_type'
        ship.species = 'human'
        
        mock_oracle = MagicMock()
        mock_links = MagicMock()
        mock_links.links_dict = {'TypeA': pd.DataFrame(), 'TypeB': pd.DataFrame()}
        mock_oracle.get_links.return_value = mock_links
        
        mock_co = MagicMock()
        mock_co.Oracle.return_value = mock_oracle
        mock_co.data.load_human_promoter_base_GRN.return_value = pd.DataFrame()
        
        with patch('SpaceTravLR.spaceship.sys.path'), \
             patch.dict('sys.modules', {'celloracle_tmp': mock_co}):
            ship.run_celloracle_(alpha=5)
            
            self.assertTrue(hasattr(ship, 'links'))
            self.assertTrue(os.path.exists('output/input_data/celloracle_links.pkl'))
    
    def test_run_celloracle_fallback(self):
        adata = create_test_adata()
        adata.layers['raw_count'] = adata.X.copy()
        adata.obsm['X_umap'] = np.random.rand(len(adata), 2)
        
        ship = SpaceShip()
        ship.adata = adata
        ship.annot = 'cell_type'
        ship.species = 'mouse'
        
        mock_oracle = MagicMock()
        mock_links = MagicMock()
        mock_links.links_dict = {'TypeA': pd.DataFrame()}
        mock_oracle.get_links.return_value = mock_links
        
        mock_co = MagicMock()
        mock_co.Oracle.return_value = mock_oracle
        mock_co.data.load_mouse_promoter_base_GRN.return_value = pd.DataFrame()
        
        with patch('SpaceTravLR.spaceship.sys.path'), \
             patch.dict('sys.modules', {'celloracle': None}), \
             patch('importlib.import_module', return_value=mock_co):
            try:
                ship.run_celloracle_()
            except (ImportError, AttributeError):
                pass

    def test_setup_new(self):
        adata = create_test_adata()
        ship = SpaceShip()
        
        with patch.object(ship, 'process_adata_') as mock_process, \
             patch.object(ship, 'run_celloracle_') as mock_celloracle, \
             patch.object(ship, 'run_commot_') as mock_commot, \
             patch.object(ship, 'get_nichenet_links_') as mock_nichenet:
            
            result = ship.setup_(adata, overwrite=False)
            
            assert ship.status == Status.BORED
            assert result == ship
            mock_process.assert_called_once()
            mock_celloracle.assert_called_once()
            mock_commot.assert_called_once()
            mock_nichenet.assert_called_once()
    
    def test_setup_existing_no_overwrite(self):
        adata = create_test_adata()
        ship = SpaceShip()
        
        os.makedirs('output', exist_ok=True)
        
        with patch.object(ship, 'process_adata_'):
            result = ship.setup_(adata, overwrite=False)
            
            assert ship.status == Status.FUBAR
            assert result is None
    
    def test_setup_existing_overwrite(self):
        adata = create_test_adata()
        ship = SpaceShip()
        
        os.makedirs('output', exist_ok=True)
        
        with patch.object(ship, 'process_adata_') as mock_process, \
             patch.object(ship, 'run_celloracle_') as mock_celloracle, \
             patch.object(ship, 'run_commot_') as mock_commot, \
             patch.object(ship, 'get_nichenet_links_') as mock_nichenet:
            
            result = ship.setup_(adata, overwrite=True)
            
            assert ship.status == Status.BORED
            mock_process.assert_called_once()
            mock_nichenet.assert_called_once()
    
    def test_spawn_worker(self):
        os.makedirs('output/logs', exist_ok=True)
        
        ship = SpaceShip(name='TestShip')
        
        mock_slurm = MagicMock()
        
        with patch('SpaceTravLR.spaceship.Slurm', return_value=mock_slurm), \
             patch('time.strftime', return_value='20240101_120000'):
            
            ship.spawn_worker(
                partition='test',
                clusters='cpu',
                gres='gpu:1',
                job_name='TestJob',
                lifespan=1,
                python_path='python3'
            )
            
            mock_slurm.sbatch.assert_called_once_with('python3 launch.py')
    
    def test_run_spacetravlr(self):
        os.makedirs('output/input_data', exist_ok=True)
        os.makedirs('output/betadata', exist_ok=True)
        
        ship = SpaceShip()
        
        adata = create_test_adata()
        adata.write_h5ad('output/input_data/_adata.h5ad')
        
        mock_links = {'TypeA': pd.DataFrame(), 'TypeB': pd.DataFrame()}
        with open('output/input_data/celloracle_links.pkl', 'wb') as f:
            pickle.dump(mock_links, f)

        mock_nichenet_links = create_test_tfls(adata.var_names, n_ligands=3, n_tfs=10)
        mock_nichenet_links.to_parquet('output/input_data/tflinks.parquet')
        
        mock_tflinks = pd.DataFrame({'ligand': ['Lig1'], 'target': ['Gene1'], 'weight': [0.5]})
        
        mock_space_travlr = MagicMock()
        mock_regulatory_factory = MagicMock()
        
        with patch('SpaceTravLR.oracles.SpaceTravLR', return_value=mock_space_travlr), \
             patch('SpaceTravLR.tools.network.RegulatoryFactory', return_value=mock_regulatory_factory):            
            ship.run_spacetravlr(
                max_epochs=10,
                learning_rate=1e-3,
                spatial_dim=32,
                batch_size=256,
                radius=200,
                contact_distance=50
            )
            
            mock_space_travlr.run.assert_called_once()
    
    def test_fit_alias(self):
        ship = SpaceShip()
        
        with patch.object(ship, 'run_spacetravlr') as mock_run:
            ship.fit(max_epochs=10)
            mock_run.assert_called_once_with(max_epochs=10)
    
    def test_is_everything_ok_success(self):
        ship = SpaceShip()
        
        os.makedirs('output/input_data', exist_ok=True)
        os.makedirs('output/betadata', exist_ok=True)
        os.makedirs('output/logs', exist_ok=True)
        
        adata = create_test_adata()
        adata.layers['normalized_count'] = adata.X.copy()
        adata.layers['imputed_count'] = adata.X.copy()
        adata.obsm['X_umap'] = np.random.rand(len(adata), 2)
        adata.obs['cell_type_int'] = np.random.randint(0, 3, len(adata))
        adata.write_h5ad('output/input_data/_adata.h5ad')
        
        with open('output/input_data/celloracle_links.pkl', 'wb') as f:
            pickle.dump({}, f)
        
        with open('launch.py', 'w') as f:
            f.write('# launch script\n')
        
        result = ship.is_everything_ok()
        assert result is True
    
    def test_is_everything_ok_missing_adata(self):
        ship = SpaceShip()
        
        os.makedirs('output/input_data', exist_ok=True)
        
        with self.assertRaisesRegex(AssertionError, "AnnData file not found"):
            ship.is_everything_ok()
    
    def test_is_everything_ok_missing_layers(self):
        ship = SpaceShip()
        
        os.makedirs('output/input_data', exist_ok=True)
        
        adata = create_test_adata()
        del adata.layers['imputed_count']
        adata.write_h5ad('output/input_data/_adata.h5ad')
        
        with open('output/input_data/celloracle_links.pkl', 'wb') as f:
            pickle.dump({}, f)
        
        with self.assertRaisesRegex(AssertionError, "Imputed count layer not found"):
            ship.is_everything_ok()
    
    def test_catch_and_retry_decorator(self):
        call_count = [0]
        
        @catch_and_retry(retry=1)
        def failing_function():
            call_count[0] += 1
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            failing_function()
        
        self.assertEqual(call_count[0], 1)
    
    def test_catch_errors_alias(self):
        self.assertTrue(callable(catch_errors))
        
        call_count = [0]
        
        @catch_errors
        def test_function():
            call_count[0] += 1
            return "success"
        
        result = test_function()
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 1)


if __name__ == '__main__':
    unittest.main()

