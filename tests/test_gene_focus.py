"""
Tests for the optional gene-focus feature (CU-3, D5):

    SpaceShip(genes=...) / oracles.SpaceTravLR(genes=...)

restricts *which target genes get a per-gene model trained* without touching
which genes are available as TF/ligand/receptor predictors (adata.var is never
subset).
"""
import os
import glob
import tempfile
import shutil

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from unittest.mock import patch, MagicMock

from SpaceTravLR.oracles import SpaceTravLR


class MockRegulatoryFactory:
    """Minimal stand-in for a CellOracle-backed RegulatoryFactory.

    We must never construct oracles.SpaceTravLR with the default GRN
    (DayThreeRegulatoryNetwork) in tests: it loads a pickle that does not
    exist in this checkout.
    """

    def __init__(self, links=None, annot='cell_type_int'):
        self.annot = annot
        self.links = links if links is not None else {
            0: pd.DataFrame({
                'source': ['TF1', 'TF2'],
                'target': ['CD74', 'CD74'],
                'coef_mean': [0.5, 0.3],
                'p': [0.01, 0.02],
            })
        }

    def get_regulators(self, adata, target_gene, alpha=0.05):
        regs = self.get_regulators_with_pvalues(adata, target_gene, alpha)
        if regs.empty:
            return []
        grouped = regs.groupby('source').mean(numeric_only=True)
        filtered = grouped[grouped.index.isin(adata.var_names)]
        return filtered.index.tolist()

    def get_regulators_with_pvalues(self, adata, target_gene, alpha=0.05):
        parts = [
            link_data.query(f'target == "{target_gene}" and p < {alpha}')[['source', 'coef_mean']]
            for link_data in self.links.values()
        ]
        co_links = pd.concat(parts, axis=0).reset_index(drop=True)
        if co_links.empty:
            return pd.DataFrame(columns=['source', 'coef_mean'])
        return co_links[co_links.source.isin(adata.var_names)].reset_index(drop=True)


def make_synthetic_adata(genes, n_cells=20, seed=0):
    """A tiny, fast, fully-synthetic AnnData used for construction-only tests
    (no training is triggered by these tests, so the exact values don't matter)."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_cells, len(genes))).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = genes
    adata.obs_names = [f'cell_{i}' for i in range(n_cells)]
    adata.obs['cell_type_int'] = 0
    adata.obsm['spatial'] = rng.random((n_cells, 2)).astype(np.float32) * 100
    adata.layers['imputed_count'] = X.copy()
    adata.layers['normalized_count'] = X.copy()
    return adata


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


ALL_GENES = ['CD74', 'TF1', 'TF2', 'LigA', 'RecA']


# ---------------------------------------------------------------------------
# Unit tests: oracles.SpaceTravLR(genes=...) queue restriction
# ---------------------------------------------------------------------------

def test_genes_none_trains_all(temp_dir):
    """Default behavior (genes=None) must be unchanged: all var_names are queued."""
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=None)

    assert st.queue.all_genes == list(adata.var_names)
    assert st.genes == list(adata.var_names)
    assert set(st.queue.remaining_genes) == set(adata.var_names)
    # adata.var (all predictors) must remain fully available, untouched
    assert list(st.adata.var_names) == ALL_GENES


def test_genes_subset_restricts_queue(temp_dir):
    """A valid subset trains only those genes; predictors (adata.var) stay full."""
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()
    subset = ['CD74', 'LigA']

    st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=subset)

    assert st.queue.all_genes == subset
    assert st.genes == subset
    assert set(st.queue.remaining_genes) <= set(subset)
    assert set(st.queue.remaining_genes) == set(subset)
    # Regulators/ligands/receptors must remain fully available as predictors
    assert list(st.adata.var_names) == ALL_GENES
    assert st.adata.shape[1] == len(ALL_GENES)


def test_genes_duplicates_are_deduped(temp_dir):
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74', 'LigA', 'CD74', 'LigA'])

    assert st.queue.all_genes == ['CD74', 'LigA']
    assert st.genes == ['CD74', 'LigA']


def test_genes_preserve_order(temp_dir):
    """Dedup must preserve first-seen order (not sort / not set-order)."""
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['RecA', 'CD74', 'RecA', 'TF1'])

    assert st.queue.all_genes == ['RecA', 'CD74', 'TF1']


def test_genes_missing_are_dropped_with_warning(temp_dir, capsys):
    """Genes absent from adata are DROPPED (not errored) with a printed warning naming
    them; the present genes are still trained, in order."""
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn,
                     genes=['CD74', 'NotAGene', 'LigA', 'AlsoBogus'])

    assert st.queue.all_genes == ['CD74', 'LigA']   # present kept, order preserved
    assert st.genes == ['CD74', 'LigA']
    assert list(st.adata.var_names) == ALL_GENES     # predictors untouched
    out = capsys.readouterr().out
    assert 'NotAGene' in out and 'AlsoBogus' in out
    assert 'drop' in out.lower()


def test_genes_all_missing_raises(temp_dir):
    """If NONE of the focus genes are in the data, that's a real error (not silent no-op)."""
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    with pytest.raises(ValueError, match='none of the requested focus genes'):
        SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['Bogus1', 'Bogus2'])


def test_genes_empty_list_raises(temp_dir):
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    with pytest.raises(ValueError):
        SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=[])


def test_genes_bare_string_raises_typeerror(temp_dir):
    """A bare string must be rejected up front, not silently iterated into
    single characters ('CD74' -> ['C','D','7','4'])."""
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    with pytest.raises(TypeError, match='not a string'):
        SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes='CD74')


# ---------------------------------------------------------------------------
# Multi-gene coverage (attempt-level): run() attempts EXACTLY the focus genes
# ---------------------------------------------------------------------------

def test_multi_gene_focus_attempts_exactly_focus_set(temp_dir):
    """A 2+-gene focus list must cause run() to attempt training for exactly
    those genes and write a betadata file for each -- and never touch any
    non-focus gene in adata.var_names.

    We patch SpatialCellularProgramsEstimator at the oracles level with a
    lightweight fake that (a) records every target_gene run() attempts and
    (b) emits trivial non-zero betadata, so the queue/attempt/write plumbing is
    exercised in full without the real estimator's cost or the pre-existing
    `activation`-kwarg crash.
    """
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()
    focus = ['CD74', 'LigA', 'RecA']

    attempted = []

    class FakeEstimator:
        def __init__(self, adata, target_gene, **kwargs):
            attempted.append(target_gene)
            self._obs_names = adata.obs_names
            # non-empty so run() does not short-circuit to add_orphan
            self.regulators = ['TF1']
            self.test_mode = None

        def fit(self, *args, **kwargs):
            return None

        @property
        def betadata(self):
            return pd.DataFrame(
                {'beta0': 1.0, 'beta_TF1': 1.0},
                index=self._obs_names,
            )

    with patch('SpaceTravLR.oracles.SpatialCellularProgramsEstimator', FakeEstimator):
        st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=focus, max_epochs=1)
        st.run()

    # every attempted gene must be in the focus set, and every focus gene attempted
    assert set(attempted) == set(focus)
    # no gene attempted more than once
    assert len(attempted) == len(focus)

    completed = sorted(
        os.path.basename(p).replace('_betadata.parquet', '')
        for p in glob.glob(f'{temp_dir}/*_betadata.parquet')
    )
    assert completed == sorted(focus)

    # no non-focus gene was attempted or written
    non_focus = set(ALL_GENES) - set(focus)
    assert non_focus.isdisjoint(set(attempted))
    for gene in non_focus:
        assert not os.path.exists(f'{temp_dir}/{gene}_betadata.parquet')
        assert not os.path.exists(f'{temp_dir}/{gene}.orphan')


# ---------------------------------------------------------------------------
# End-to-end (real, tiny data): only the focused gene gets a betadata file
# ---------------------------------------------------------------------------

def test_gene_focus_end_to_end_trains_only_target_gene(temp_dir):
    """
    Runs the real `oracles.SpaceTravLR.run()` training loop (not mocked) on a
    600-cell (test data allows only ~400 after filtering) real subset of
    data/snrna_germinal_center.h5ad, restricted via genes=['CD74'], and checks
    that ONLY CD74_betadata.parquet is produced in save_dir -- no other gene.

    This also serves as a regression test for the `activation`-kwarg fix: the
    tiny/low-signal fit lands in the group-lasso r2 < 0.15 fallback branch, which
    previously crashed by passing an unsupported `activation` kwarg into
    CellularNicheNetwork. It now runs unpatched.
    """
    import time

    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(test_dir, '..', 'data', 'snrna_germinal_center.h5ad'))
    if not os.path.exists(data_path):
        pytest.skip('real test data not available locally')

    adata = ad.read_h5ad(data_path)

    keep_genes = ['CD74', 'NFKB1', 'STAT3', 'FOXP3', 'IL2RA', 'CCL5', 'BMP2', 'BMPR1A', 'FOS']
    adata = adata[:, adata.var_names.isin(keep_genes)].copy()
    adata = adata[adata.obs['cell_type_int'].isin([0, 2])].copy()
    adata = adata[:400, :].copy()
    # remap to contiguous 0..k-1 cluster ints (spatial map channel dim depends on this)
    codes, _ = pd.factorize(adata.obs['cell_type_int'])
    adata.obs['cell_type_int'] = pd.Categorical(codes)

    adata.layers['imputed_count'] = adata.X.copy()
    adata.layers['normalized_count'] = adata.layers['imputed_count'].copy()

    grn = MockRegulatoryFactory(links={
        0: pd.DataFrame({
            'source': ['NFKB1', 'STAT3'],
            'target': ['CD74', 'CD74'],
            'coef_mean': [0.5, 0.3],
            'p': [0.01, 0.02],
        })
    })

    # tiny NicheNet-style ligand-target matrix: TFs (index) x ligands (columns)
    tflinks = pd.DataFrame([[0.5], [0.5]], index=['NFKB1', 'STAT3'], columns=['BMP2'])

    t0 = time.time()
    st = SpaceTravLR(
        adata=adata,
        save_dir=temp_dir,
        grn=grn,
        genes=['CD74'],
        max_epochs=2,
        batch_size=64,
        tflinks=tflinks,
        annot='cell_type_int',
    )
    st.run()
    elapsed = time.time() - t0
    print(f'\n[test_gene_focus_end_to_end] run() took {elapsed:.2f}s')

    betadata_files = sorted(os.path.basename(p) for p in glob.glob(f'{temp_dir}/*_betadata.parquet'))
    orphan_files = sorted(os.path.basename(p) for p in glob.glob(f'{temp_dir}/*.orphan'))

    # exactly one gene was ever considered, so exactly one outcome file (parquet or orphan)
    assert betadata_files + orphan_files != []
    assert all(f.startswith('CD74') for f in betadata_files + orphan_files)
    assert len(betadata_files) + len(orphan_files) == 1
    # no other gene from adata.var_names was ever attempted
    for gene in adata.var_names:
        if gene == 'CD74':
            continue
        assert f'{gene}_betadata.parquet' not in betadata_files
        assert f'{gene}.orphan' not in orphan_files
