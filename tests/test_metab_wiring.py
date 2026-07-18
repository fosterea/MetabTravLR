"""
Tests for CU-2: thread `metab_pairs` through the trainer (`oracles.SpaceTravLR`)
and orchestrator (`SpaceShip.run_spacetravlr`), and relax the no-TF orphan skip
so a gene with metabolite modulators but no TF regulators still trains.

CU-1 (already committed) gave `SpatialCellularProgramsEstimator` a `metab_pairs`
ctor arg and exposed `.metab_pairs` (list of `export@import` strings, `[]` when
none). CU-2 is pure plumbing: no new science, so these tests are Tier 0
(construction / kwarg-capture / file-presence), reusing the `FakeEstimator` +
`patch('SpaceTravLR.oracles.SpatialCellularProgramsEstimator', ...)` pattern
from `tests/test_gene_focus.py`.
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
from SpaceTravLR.spaceship import SpaceShip


class MockRegulatoryFactory:
    """Minimal stand-in for a CellOracle-backed RegulatoryFactory (copied from
    tests/test_gene_focus.py). We must never construct oracles.SpaceTravLR with
    the default GRN (DayThreeRegulatoryNetwork) in tests: it loads a pickle that
    does not exist in this checkout."""

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
    (no real training is triggered, so the exact values don't matter)."""
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


def _make_fake_estimator_class(attempted, regulators_by_gene, metab_pairs_seen, betadata_override=None):
    """Builds a FakeEstimator class (closure) that records:
    - every target_gene it's constructed for (`attempted`)
    - the `metab_pairs` kwarg it was constructed with, per gene (`metab_pairs_seen`)
    and reports `.regulators` per `regulators_by_gene[target_gene]` (default ['TF1']).

    `betadata_override`: optional `{gene: DataFrame}` to hand back a specific
    betadata frame for a gene (e.g. to simulate the group-lasso R^2<0.15
    zeroed-anchor fallback, where every beta including beta0 is 0), instead of
    the default trivially-nonzero frame.
    """
    betadata_override = betadata_override or {}

    class FakeEstimator:
        def __init__(self, adata, target_gene, **kwargs):
            attempted.append(target_gene)
            metab_pairs_seen[target_gene] = kwargs.get('metab_pairs', '<MISSING>')
            self._obs_names = adata.obs_names
            self._target_gene = target_gene
            self.regulators = regulators_by_gene.get(target_gene, ['TF1'])
            # CU-1 contract: .metab_pairs is [] when none supplied, else the list
            # of 'export@import' strings (the estimator formats the tuples it's
            # given into '<export>@<import>' column-name strings).
            supplied = kwargs.get('metab_pairs')
            self.metab_pairs = [f'{e}@{i}' for e, i in supplied] if supplied else []
            self.modulators = list(self.regulators) + list(self.metab_pairs)
            self.test_mode = None

        def fit(self, *args, **kwargs):
            return None

        @property
        def betadata(self):
            if self._target_gene in betadata_override:
                return betadata_override[self._target_gene]
            data = {'beta0': 1.0}
            for r in self.regulators:
                data[f'beta_{r}'] = 1.0
            for mp in self.metab_pairs:
                data[f'beta_{mp}'] = 1.0
            return pd.DataFrame(data, index=self._obs_names)

    return FakeEstimator


# ---------------------------------------------------------------------------
# 1. Threading through oracles: SpaceTravLR(metab_pairs=...) reaches the
#    estimator ctor unchanged.
# ---------------------------------------------------------------------------

def test_metab_pairs_threaded_to_estimator_ctor(temp_dir):
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()
    supplied = [('A', 'B'), ('B', 'A')]

    attempted = []
    metab_pairs_seen = {}
    FakeEstimator = _make_fake_estimator_class(attempted, {}, metab_pairs_seen)

    with patch('SpaceTravLR.oracles.SpatialCellularProgramsEstimator', FakeEstimator):
        st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'],
                          max_epochs=1, metab_pairs=supplied)
        assert st.metab_pairs == supplied
        st.run()

    assert attempted == ['CD74']
    assert metab_pairs_seen['CD74'] == supplied


# ---------------------------------------------------------------------------
# 2. Orphan relax: metab_pairs present, regulators empty -> gene trains.
# ---------------------------------------------------------------------------

def test_orphan_relaxed_when_metab_pairs_present(temp_dir):
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    attempted = []
    metab_pairs_seen = {}
    # CD74 reports NO TF regulators, but the FakeEstimator will still expose a
    # non-empty .metab_pairs because metab_pairs=[('A','B')] is supplied.
    FakeEstimator = _make_fake_estimator_class(attempted, {'CD74': []}, metab_pairs_seen)

    with patch('SpaceTravLR.oracles.SpatialCellularProgramsEstimator', FakeEstimator):
        st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'],
                          max_epochs=1, metab_pairs=[('A', 'B')])
        st.run()

    assert attempted == ['CD74']  # fit was reached (attempted only records ctor calls,
    # but combined with the assertions below we confirm fit()/write happened, not orphan)
    assert not os.path.exists(f'{temp_dir}/CD74.orphan')
    assert os.path.exists(f'{temp_dir}/CD74_betadata.parquet')

    betadata = pd.read_parquet(f'{temp_dir}/CD74_betadata.parquet')
    assert 'beta_A@B' in betadata.columns


# ---------------------------------------------------------------------------
# 3. Orphan PRESERVED: no metab_pairs, no TF regulators -> gene IS orphaned.
#    Pins the default-preserving gate.
# ---------------------------------------------------------------------------

def test_orphan_preserved_when_no_metab_and_no_regulators(temp_dir):
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    attempted = []
    metab_pairs_seen = {}
    fit_calls = []

    class FakeEstimatorNoFit(_make_fake_estimator_class(attempted, {'CD74': []}, metab_pairs_seen)):
        def fit(self, *args, **kwargs):
            fit_calls.append(True)
            return None

    with patch('SpaceTravLR.oracles.SpatialCellularProgramsEstimator', FakeEstimatorNoFit):
        st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'],
                          max_epochs=1, metab_pairs=None)
        st.run()

    assert os.path.exists(f'{temp_dir}/CD74.orphan')
    assert not os.path.exists(f'{temp_dir}/CD74_betadata.parquet')
    assert fit_calls == []  # fit() must NOT have been called


# ---------------------------------------------------------------------------
# 4. Default unchanged: metab_pairs omitted -> self.metab_pairs is None, and
#    the estimator is constructed with metab_pairs=None.
# ---------------------------------------------------------------------------

def test_default_metab_pairs_is_none(temp_dir):
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    attempted = []
    metab_pairs_seen = {}
    FakeEstimator = _make_fake_estimator_class(attempted, {}, metab_pairs_seen)

    with patch('SpaceTravLR.oracles.SpatialCellularProgramsEstimator', FakeEstimator):
        st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'], max_epochs=1)
        assert st.metab_pairs is None
        st.run()

    assert metab_pairs_seen['CD74'] is None


# ---------------------------------------------------------------------------
# 5. SpaceShip pass-through: run_spacetravlr(metab_pairs=...) forwards the
#    kwarg into the SpaceTravLR constructor. We patch the disk reads
#    (sc.read_h5ad / pd.read_parquet / pickle.load) so no real setup artifacts
#    are needed, and patch SpaceTravLR.oracles.SpaceTravLR (the name
#    run_spacetravlr imports locally) with a kwarg-capturing fake whose .run()
#    is a no-op.
# ---------------------------------------------------------------------------

def test_spaceship_run_spacetravlr_forwards_metab_pairs(temp_dir):
    # Minimal on-disk placeholders so `open(...)` succeeds; their *content* is
    # never parsed because sc.read_h5ad/pd.read_parquet/pickle.load are mocked.
    os.makedirs(f'{temp_dir}/input_data', exist_ok=True)
    os.makedirs(f'{temp_dir}/betadata', exist_ok=True)
    for fname in ('_adata.h5ad', 'tflinks.parquet', 'celloracle_links.pkl'):
        open(f'{temp_dir}/input_data/{fname}', 'wb').close()

    fake_links = {0: pd.DataFrame({'source': [], 'target': [], 'coef_mean': [], 'p': []})}
    captured_kwargs = {}

    class FakeSpaceTravLR:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def run(self):
            return None

    ship = SpaceShip(name='test', outdir=temp_dir, genes=['CD74'])
    supplied = [('A', 'B'), ('B', 'A')]

    with patch('SpaceTravLR.spaceship.sc.read_h5ad', return_value=make_synthetic_adata(ALL_GENES)), \
         patch('SpaceTravLR.spaceship.pd.read_parquet', return_value=pd.DataFrame()), \
         patch('SpaceTravLR.spaceship.pickle.load', return_value=fake_links), \
         patch('SpaceTravLR.oracles.SpaceTravLR', FakeSpaceTravLR):
        ship.run_spacetravlr(metab_pairs=supplied)

    assert captured_kwargs.get('metab_pairs') == supplied
    assert captured_kwargs.get('genes') == ['CD74']


def test_spaceship_run_spacetravlr_default_metab_pairs_is_none(temp_dir):
    os.makedirs(f'{temp_dir}/input_data', exist_ok=True)
    os.makedirs(f'{temp_dir}/betadata', exist_ok=True)
    for fname in ('_adata.h5ad', 'tflinks.parquet', 'celloracle_links.pkl'):
        open(f'{temp_dir}/input_data/{fname}', 'wb').close()

    fake_links = {0: pd.DataFrame({'source': [], 'target': [], 'coef_mean': [], 'p': []})}
    captured_kwargs = {}

    class FakeSpaceTravLR:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def run(self):
            return None

    ship = SpaceShip(name='test', outdir=temp_dir, genes=['CD74'])

    with patch('SpaceTravLR.spaceship.sc.read_h5ad', return_value=make_synthetic_adata(ALL_GENES)), \
         patch('SpaceTravLR.spaceship.pd.read_parquet', return_value=pd.DataFrame()), \
         patch('SpaceTravLR.spaceship.pickle.load', return_value=fake_links), \
         patch('SpaceTravLR.oracles.SpaceTravLR', FakeSpaceTravLR):
        ship.run_spacetravlr()

    assert captured_kwargs.get('metab_pairs') is None


# ---------------------------------------------------------------------------
# 6. Second orphan gate (review finding, medium): the write/orphan decision at
#    the bottom of run() must be made on the POST-filter betadata frame, not
#    the raw column count. Before CU-2, a gene only ever reached this point
#    with regulators != [], so the raw column count (> 1) and the post-filter
#    column count (> 0, since beta0 basically always survives) coincided. CU-2
#    makes a TF-less, metab-only gene reachable here too, and such a gene CAN
#    land in the group-lasso R^2<0.15 zeroed-anchor fallback (every beta,
#    incl. beta0, == 0) -- the raw-count check would then write a degenerate
#    0-column parquet instead of orphaning.
# ---------------------------------------------------------------------------

def test_second_gate_writes_identical_filtered_betadata_when_nonzero(temp_dir):
    """Behavior-preservation half of the fix: whenever >=1 beta is nonzero,
    the written parquet must be EXACTLY the same filtered frame as before
    (`betadata.loc[:, (betadata != 0).any(axis=0)]`), independent of whether
    the raw frame also contains all-zero columns to drop."""
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()
    n = adata.n_obs

    raw_betadata = pd.DataFrame({
        'beta0': np.zeros(n),          # zeroed out (poor-fit-like)
        'beta_TF1': np.zeros(n),       # zeroed out
        'beta_A@B': np.full(n, 0.5),   # the one real, nonzero beta
    }, index=adata.obs_names)

    attempted, metab_pairs_seen = [], {}
    FakeEstimator = _make_fake_estimator_class(
        attempted, {'CD74': []}, metab_pairs_seen,
        betadata_override={'CD74': raw_betadata},
    )

    with patch('SpaceTravLR.oracles.SpatialCellularProgramsEstimator', FakeEstimator):
        st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'],
                          max_epochs=1, metab_pairs=[('A', 'B')])
        st.run()

    assert not os.path.exists(f'{temp_dir}/CD74.orphan')
    assert os.path.exists(f'{temp_dir}/CD74_betadata.parquet')

    written = pd.read_parquet(f'{temp_dir}/CD74_betadata.parquet')
    expected = raw_betadata.loc[:, (raw_betadata != 0).any(axis=0)]
    pd.testing.assert_frame_equal(written, expected, check_dtype=False)
    assert list(written.columns) == ['beta_A@B']  # the all-zero columns were dropped, as before


def test_second_gate_orphans_when_all_betas_zero(temp_dir):
    """The bug this fixes: a metab-only (no-TF) gene whose fit lands entirely
    in the zeroed-anchor fallback (beta0 included) must now be ORPHANED, not
    written as a degenerate 0-column parquet. Only reachable post-CU-2 because
    pre-CU-2 a no-TF gene never got past the first orphan gate to reach fit()
    at all."""
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()
    n = adata.n_obs

    raw_betadata = pd.DataFrame({
        'beta0': np.zeros(n),
        'beta_A@B': np.zeros(n),
    }, index=adata.obs_names)

    attempted, metab_pairs_seen = [], {}
    FakeEstimator = _make_fake_estimator_class(
        attempted, {'CD74': []}, metab_pairs_seen,
        betadata_override={'CD74': raw_betadata},
    )

    with patch('SpaceTravLR.oracles.SpatialCellularProgramsEstimator', FakeEstimator):
        st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'],
                          max_epochs=1, metab_pairs=[('A', 'B')])
        st.run()

    assert os.path.exists(f'{temp_dir}/CD74.orphan')
    assert not os.path.exists(f'{temp_dir}/CD74_betadata.parquet')


# ---------------------------------------------------------------------------
# 7. Fail-fast validation (review finding, low): a malformed metab_pairs must
#    raise at SpaceTravLR construction, not silently propagate to the first
#    gene's estimator construction (matters for unattended Savio runs).
# ---------------------------------------------------------------------------

def test_metab_pairs_bad_type_raises_at_construction(temp_dir):
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    with pytest.raises(ValueError, match='metab_pairs'):
        SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'],
                    metab_pairs='not-a-list')


def test_metab_pairs_bad_element_shape_raises_at_construction(temp_dir):
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    with pytest.raises(ValueError, match='metab_pairs'):
        SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'],
                    metab_pairs=[('A', 'B', 'C')])  # 3-tuple, not (export, import)


def test_metab_pairs_non_string_genes_raises_at_construction(temp_dir):
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    with pytest.raises(ValueError, match='metab_pairs'):
        SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'],
                    metab_pairs=[(1, 2)])


def test_metab_pairs_empty_list_is_valid(temp_dir):
    """An empty list is a legitimate 'no metabolites' spelling (same as
    None), not a malformed input -- must NOT raise."""
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'], metab_pairs=[])
    assert st.metab_pairs == []


# ---------------------------------------------------------------------------
# 8. Tier-1, real estimator (review finding, medium): drive a REAL
#    SpatialCellularProgramsEstimator (not a fake) through the actual
#    oracles.SpaceTravLR.run() loop for a TF-less target gene whose only
#    modulator is a metabolite pair, with real spatial signal so it clears
#    the group-lasso R^2>=0.15 threshold (not the zeroed-anchor fallback).
#    Exercises the relaxed first gate AND the fixed second gate together on
#    real training output.
# ---------------------------------------------------------------------------

def test_real_estimator_trains_no_tf_metab_only_gene_via_run(temp_dir):
    import torch
    from SpaceTravLR.models.parallel_estimators import SpatialCellularProgramsEstimator

    # Seed the GLOBAL RNGs (not just a local Generator): model weight init,
    # DataLoader shuffling, and RotatedTensorDataset's rotation all draw from
    # these, so this must be pinned for the test to be non-flaky (see the
    # rationale in tests/test_get_betas_batching.py::_trained_estimator).
    torch.manual_seed(0)
    np.random.seed(0)
    rng = np.random.default_rng(0)

    N = 800
    genes = ['A', 'B']
    target = 'T'
    names = genes + [target]
    X = rng.random((N, len(names))).astype(np.float32)

    # Hand-craft the target as a clean function of the metabolite flux
    # (received(A, diffused) x raw(B)) so the real CNN fit clears R^2>=0.15
    # in at least the (single) cluster, instead of hitting the fallback.
    received_A = rng.uniform(0.5, 2.5, N).astype(np.float32)
    raw_B = rng.uniform(0.5, 2.5, N).astype(np.float32)
    flux = received_A * raw_B
    flux_z = (flux - flux.mean()) / flux.std()
    target_vals = (flux_z + 0.05 * rng.standard_normal(N)).astype(np.float32)

    X[:, names.index('B')] = raw_B
    X[:, names.index(target)] = target_vals

    a = ad.AnnData(X=X)
    a.var_names = names
    a.obs_names = [f'c{i}' for i in range(N)]
    a.obs['cell_type_int'] = 0  # single cluster -- simple, fast, deterministic
    a.obsm['spatial'] = rng.uniform(0, 500, size=(N, 2)).astype(np.float32)
    a.layers['imputed_count'] = X.copy()
    a.layers['normalized_count'] = X.copy()
    # Pre-seed the diffused export-gene cache: this is the same trick
    # tests/test_metab_group.py uses (`_build_metab_estimator`) to skip the
    # real O(N^2) diffusion and hand the estimator a known-good flux term.
    a.uns['received_ligands_tfl'] = pd.DataFrame(
        {'A': received_A, 'B': raw_B}, index=a.obs_names
    )

    class NoTFGRN:
        """Reports zero TF regulators for every gene, so the target's only
        modulator is the metabolite pair."""
        def get_regulators(self, adata, target_gene, alpha=0.05):
            return []

    captured = []

    class CapturingEstimator(SpatialCellularProgramsEstimator):
        """The REAL estimator, thinly wrapped to stash `self` after a real
        fit() so the test can inspect `.scores` -- no logic is overridden."""
        def fit(self, *args, **kwargs):
            result = super().fit(*args, **kwargs)
            captured.append(self)
            return result

    with patch('SpaceTravLR.oracles.SpatialCellularProgramsEstimator', CapturingEstimator):
        st = SpaceTravLR(
            adata=a, save_dir=temp_dir, grn=NoTFGRN(), genes=[target],
            max_epochs=25, batch_size=256,
            tflinks=pd.DataFrame(),  # avoid a real NicheNet network download
            metab_pairs=[('A', 'B')],
        )
        st.run()

    assert len(captured) == 1
    est = captured[0]
    assert est.regulators == []
    assert est.metab_pairs == ['A@B']
    assert any(v >= 0.15 for v in est.scores.values()), (
        f"fixture didn't train a real CNN (all clusters hit the zeroed-anchor "
        f"fallback): scores={est.scores}"
    )

    assert not os.path.exists(f'{temp_dir}/{target}.orphan')
    bd_path = f'{temp_dir}/{target}_betadata.parquet'
    assert os.path.exists(bd_path)

    betadata = pd.read_parquet(bd_path)
    assert 'beta_A@B' in betadata.columns
    assert np.isfinite(betadata['beta_A@B'].values.astype(float)).all()
