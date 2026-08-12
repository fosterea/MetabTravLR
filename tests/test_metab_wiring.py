"""
Tests for threading `metabolites` through the trainer (`oracles.SpaceTravLR`) and
orchestrator (`SpaceShip.run_spacetravlr`), plus the no-TF orphan-skip relaxation
so a gene with metabolite modulators but no TF regulators still trains.

`SpatialCellularProgramsEstimator` takes a `metabolites` ctor arg
(`dict {name: [(export, import), ...]}`) and exposes `.metab_pairs` (list of
`metab@<name>` column-name strings, `[]` when none). This plumbing carries no new
science, so these are Tier 0 (construction / kwarg-capture / file-presence),
reusing the `FakeEstimator` + `patch(...)` pattern from `tests/test_gene_focus.py`,
plus one Tier-1 real-estimator end-to-end.
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


def _make_fake_estimator_class(attempted, regulators_by_gene, metabolites_seen, betadata_override=None):
    """Builds a FakeEstimator class (closure) that records:
    - every target_gene it's constructed for (`attempted`)
    - the `metabolites` kwarg it was constructed with, per gene (`metabolites_seen`)
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
            metabolites_seen[target_gene] = kwargs.get('metabolites', '<MISSING>')
            self._obs_names = adata.obs_names
            self._target_gene = target_gene
            self.regulators = regulators_by_gene.get(target_gene, ['TF1'])
            # Contract: .metab_pairs is [] when none supplied, else one
            # 'metab@<name>' column-name string per metabolite (the estimator
            # sums each metabolite's pairs into a single such column).
            supplied = kwargs.get('metabolites')
            self.metab_pairs = [f'metab@{name}' for name in supplied] if supplied else []
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
# 1. Threading through oracles: SpaceTravLR(metabolites=...) reaches the
#    estimator ctor unchanged.
# ---------------------------------------------------------------------------

def test_metabolites_threaded_to_estimator_ctor(temp_dir):
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()
    supplied = {'M1': [('A', 'B'), ('B', 'A')], 'M2': [('LigA', 'RecA')]}

    attempted = []
    metabolites_seen = {}
    FakeEstimator = _make_fake_estimator_class(attempted, {}, metabolites_seen)

    with patch('SpaceTravLR.oracles.SpatialCellularProgramsEstimator', FakeEstimator):
        st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'],
                          max_epochs=1, metabolites=supplied)
        assert st.metabolites == supplied
        st.run()

    assert attempted == ['CD74']
    assert metabolites_seen['CD74'] == supplied


# ---------------------------------------------------------------------------
# 2. Orphan relax: metabolites present, regulators empty -> gene trains.
# ---------------------------------------------------------------------------

def test_orphan_relaxed_when_metabolites_present(temp_dir):
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    attempted = []
    metabolites_seen = {}
    # CD74 reports NO TF regulators, but the FakeEstimator will still expose a
    # non-empty .metab_pairs because metabolites={'M': ...} is supplied.
    FakeEstimator = _make_fake_estimator_class(attempted, {'CD74': []}, metabolites_seen)

    with patch('SpaceTravLR.oracles.SpatialCellularProgramsEstimator', FakeEstimator):
        st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'],
                          max_epochs=1, metabolites={'M': [('A', 'B')]})
        st.run()

    assert attempted == ['CD74']
    assert not os.path.exists(f'{temp_dir}/CD74.orphan')
    assert os.path.exists(f'{temp_dir}/CD74_betadata.parquet')

    betadata = pd.read_parquet(f'{temp_dir}/CD74_betadata.parquet')
    assert 'beta_metab@M' in betadata.columns


# ---------------------------------------------------------------------------
# 3. Orphan PRESERVED: no metabolites, no TF regulators -> gene IS orphaned.
#    Pins the default-preserving gate.
# ---------------------------------------------------------------------------

def test_orphan_preserved_when_no_metab_and_no_regulators(temp_dir):
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    attempted = []
    metabolites_seen = {}
    fit_calls = []

    class FakeEstimatorNoFit(_make_fake_estimator_class(attempted, {'CD74': []}, metabolites_seen)):
        def fit(self, *args, **kwargs):
            fit_calls.append(True)
            return None

    with patch('SpaceTravLR.oracles.SpatialCellularProgramsEstimator', FakeEstimatorNoFit):
        st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'],
                          max_epochs=1, metabolites=None)
        st.run()

    assert os.path.exists(f'{temp_dir}/CD74.orphan')
    assert not os.path.exists(f'{temp_dir}/CD74_betadata.parquet')
    assert fit_calls == []  # fit() must NOT have been called


# ---------------------------------------------------------------------------
# 4. Default unchanged: metabolites omitted -> self.metabolites is None, and
#    the estimator is constructed with metabolites=None.
# ---------------------------------------------------------------------------

def test_default_metabolites_is_none(temp_dir):
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    attempted = []
    metabolites_seen = {}
    FakeEstimator = _make_fake_estimator_class(attempted, {}, metabolites_seen)

    with patch('SpaceTravLR.oracles.SpatialCellularProgramsEstimator', FakeEstimator):
        st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'], max_epochs=1)
        assert st.metabolites is None
        st.run()

    assert metabolites_seen['CD74'] is None


# ---------------------------------------------------------------------------
# 5. SpaceShip pass-through: run_spacetravlr(metabolites=...) forwards the
#    kwarg into the SpaceTravLR constructor.
# ---------------------------------------------------------------------------

def test_spaceship_run_spacetravlr_forwards_metabolites(temp_dir):
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
    supplied = {'M1': [('A', 'B'), ('B', 'A')]}

    with patch('SpaceTravLR.spaceship.sc.read_h5ad', return_value=make_synthetic_adata(ALL_GENES)), \
         patch('SpaceTravLR.spaceship.pd.read_parquet', return_value=pd.DataFrame()), \
         patch('SpaceTravLR.spaceship.pickle.load', return_value=fake_links), \
         patch('SpaceTravLR.oracles.SpaceTravLR', FakeSpaceTravLR):
        ship.run_spacetravlr(metabolites=supplied)

    assert captured_kwargs.get('metabolites') == supplied
    assert captured_kwargs.get('genes') == ['CD74']


def test_spaceship_run_spacetravlr_default_metabolites_is_none(temp_dir):
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

    assert captured_kwargs.get('metabolites') is None


# ---------------------------------------------------------------------------
# 6. Second orphan gate: the write/orphan decision at the bottom of run() must
#    be made on the POST-filter betadata frame, not the raw column count. A
#    TF-less, metab-only gene can land in the group-lasso R^2<0.15 zeroed-anchor
#    fallback (every beta, incl. beta0, == 0) -- the raw-count check would then
#    write a degenerate 0-column parquet instead of orphaning.
# ---------------------------------------------------------------------------

def test_second_gate_writes_identical_filtered_betadata_when_nonzero(temp_dir):
    """Behavior-preservation half: whenever >=1 beta is nonzero, the written
    parquet must be EXACTLY the same filtered frame as before
    (`betadata.loc[:, (betadata != 0).any(axis=0)]`)."""
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()
    n = adata.n_obs

    raw_betadata = pd.DataFrame({
        'beta0': np.zeros(n),                # zeroed out (poor-fit-like)
        'beta_TF1': np.zeros(n),             # zeroed out
        'beta_metab@M': np.full(n, 0.5),     # the one real, nonzero beta
    }, index=adata.obs_names)

    attempted, metabolites_seen = [], {}
    FakeEstimator = _make_fake_estimator_class(
        attempted, {'CD74': []}, metabolites_seen,
        betadata_override={'CD74': raw_betadata},
    )

    with patch('SpaceTravLR.oracles.SpatialCellularProgramsEstimator', FakeEstimator):
        st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'],
                          max_epochs=1, metabolites={'M': [('A', 'B')]})
        st.run()

    assert not os.path.exists(f'{temp_dir}/CD74.orphan')
    assert os.path.exists(f'{temp_dir}/CD74_betadata.parquet')

    written = pd.read_parquet(f'{temp_dir}/CD74_betadata.parquet')
    expected = raw_betadata.loc[:, (raw_betadata != 0).any(axis=0)]
    pd.testing.assert_frame_equal(written, expected, check_dtype=False)
    assert list(written.columns) == ['beta_metab@M']  # the all-zero columns were dropped


def test_second_gate_orphans_when_all_betas_zero(temp_dir):
    """A metab-only (no-TF) gene whose fit lands entirely in the zeroed-anchor
    fallback (beta0 included) must be ORPHANED, not written as a degenerate
    0-column parquet."""
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()
    n = adata.n_obs

    raw_betadata = pd.DataFrame({
        'beta0': np.zeros(n),
        'beta_metab@M': np.zeros(n),
    }, index=adata.obs_names)

    attempted, metabolites_seen = [], {}
    FakeEstimator = _make_fake_estimator_class(
        attempted, {'CD74': []}, metabolites_seen,
        betadata_override={'CD74': raw_betadata},
    )

    with patch('SpaceTravLR.oracles.SpatialCellularProgramsEstimator', FakeEstimator):
        st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'],
                          max_epochs=1, metabolites={'M': [('A', 'B')]})
        st.run()

    assert os.path.exists(f'{temp_dir}/CD74.orphan')
    assert not os.path.exists(f'{temp_dir}/CD74_betadata.parquet')


# ---------------------------------------------------------------------------
# 7. Fail-fast validation: a malformed metabolites must raise at SpaceTravLR
#    construction, not silently propagate to the first gene's estimator.
# ---------------------------------------------------------------------------

def test_metabolites_bad_type_raises_at_construction(temp_dir):
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    with pytest.raises(ValueError, match='metabolites'):
        SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'],
                    metabolites=[('A', 'B')])  # a list, not a dict


def test_metabolites_bad_element_shape_raises_at_construction(temp_dir):
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    with pytest.raises(ValueError, match='metabolites'):
        SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'],
                    metabolites={'M': [('A', 'B', 'C')]})  # 3-tuple, not (export, import)


def test_metabolites_non_string_genes_raises_at_construction(temp_dir):
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    with pytest.raises(ValueError, match='metabolites'):
        SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'],
                    metabolites={'M': [(1, 2)]})


def test_metabolites_empty_dict_is_valid(temp_dir):
    """An empty dict is a legitimate 'no metabolites' spelling (same as None),
    not a malformed input -- must NOT raise."""
    adata = make_synthetic_adata(ALL_GENES)
    grn = MockRegulatoryFactory()

    st = SpaceTravLR(adata=adata, save_dir=temp_dir, grn=grn, genes=['CD74'], metabolites={})
    assert st.metabolites == {}


# ---------------------------------------------------------------------------
# 8. Tier-1, real estimator: drive a REAL SpatialCellularProgramsEstimator
#    through the actual oracles.SpaceTravLR.run() loop for a TF-less target
#    gene whose only modulator is a metabolite, with real spatial signal so it
#    clears the group-lasso R^2>=0.15 threshold (not the zeroed-anchor
#    fallback). Exercises the relaxed first gate AND the fixed second gate.
# ---------------------------------------------------------------------------

def test_real_estimator_trains_no_tf_metab_only_gene_via_run(temp_dir):
    import torch
    from SpaceTravLR.models.parallel_estimators import SpatialCellularProgramsEstimator

    torch.manual_seed(0)
    np.random.seed(0)
    rng = np.random.default_rng(0)

    N = 800
    genes = ['A', 'B']
    target = 'T'
    names = genes + [target]
    X = rng.random((N, len(names))).astype(np.float32)

    # Hand-craft the target as a clean function of the metabolite flux
    # (received(A, diffused) x raw(B)) so the real CNN fit clears R^2>=0.15.
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
    a.uns['received_ligands_tfl'] = pd.DataFrame(
        {'A': received_A, 'B': raw_B}, index=a.obs_names
    )

    class NoTFGRN:
        """Reports zero TF regulators for every gene."""
        def get_regulators(self, adata, target_gene, alpha=0.05):
            return []

    captured = []

    class CapturingEstimator(SpatialCellularProgramsEstimator):
        def fit(self, *args, **kwargs):
            result = super().fit(*args, **kwargs)
            captured.append(self)
            return result

    with patch('SpaceTravLR.oracles.SpatialCellularProgramsEstimator', CapturingEstimator):
        st = SpaceTravLR(
            adata=a, save_dir=temp_dir, grn=NoTFGRN(), genes=[target],
            max_epochs=25, batch_size=256,
            tflinks=pd.DataFrame(),  # avoid a real NicheNet network download
            metabolites={'MET': [('A', 'B')]},
        )
        st.run()

    assert len(captured) == 1
    est = captured[0]
    assert est.regulators == []
    assert est.metab_pairs == ['metab@MET']
    assert any(v >= 0.15 for v in est.scores.values()), (
        f"fixture didn't train a real CNN (all clusters hit the zeroed-anchor "
        f"fallback): scores={est.scores}"
    )

    assert not os.path.exists(f'{temp_dir}/{target}.orphan')
    bd_path = f'{temp_dir}/{target}_betadata.parquet'
    assert os.path.exists(bd_path)

    betadata = pd.read_parquet(bd_path)
    assert 'beta_metab@MET' in betadata.columns
    assert np.isfinite(betadata['beta_metab@MET'].values.astype(float)).all()
