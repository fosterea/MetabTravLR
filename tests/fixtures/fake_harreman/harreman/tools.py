"""Fake ``harreman.tools`` submodule for local tests.

Everything below is VENDORED VERBATIM from a real, locally-cloned copy of harreman
(``/Users/fosterangus/Documents/Code/Single Cell/Harreman/src/harreman/`` -- not part of
this repo, Savio-only in production) -- copied exactly, not reimplemented, so the local
equivalence test (``tests/test_cell_communication_lowmem.py``) exercises real numerics:

  - ``compute_interacting_cell_scores`` (``tools/cell_communication.py:1473``),
    ``compute_metabolite_cs`` (``:2278``), ``compute_p_int_cell_results_no_ct`` (``:2720``)
    -- confirmed byte-identical (mod trailing newline) to our read-only reference copy
    ``DataForClaude/cell_communication.py``, so these were vendored from that reference.
  - ``counts_from_anndata`` (``preprocessing/anndata.py:9``).
  - ``make_weights_non_redundant`` (``tools/knn.py:334``).
  - ``standardize_counts`` (``hotspot/local_autocorrelation.py:449``), which calls
    ``center_counts_torch`` (same file, ``:376``), which for ``model='danb'`` calls
    ``models.danb_model_torch`` (``hotspot/models.py:39``, pure torch, no numba).
    We vendor ONLY the ``danb`` branch's model function (``_StubModels`` below) since our
    tests only ever pass ``model='danb'`` (matching what `harreman_funcs.py` actually
    uses) -- the other three ``center_counts_torch`` branches (bernoulli/normal/none) are
    kept verbatim for structural fidelity but are unreachable dead code here (out of scope
    for CU-A; would need `models.py`'s other ~450 lines, most numba-jitted numpy code we
    don't need).

CU-B (2026-07-20) added the cell-type-INDEPENDENT aggregate function and its transitive
deps, needed by ``compute_cell_communication_lowmem`` in ``cell_communication_lowmem.py``:
``compute_cell_communication`` (``:464``), ``run_cell_communication_analysis`` (``:569``),
``flatten`` (``:1359``), ``compute_p_results`` (``:2539``),
``get_cell_communication_results`` (``:3045``), ``normalize_values`` (``:3172``),
``compute_max_cs`` (``:3211``), ``compute_max_cs_gp`` (``:3237``) -- all confirmed
byte-identical (via AST comparison) to ``DataForClaude/cell_communication.py``. None of
these touch pydata ``sparse`` or numba (`@jit`/`njit`) -- verified by grepping each
function body -- so the cell-type-independent path is torch/scipy/pandas/statsmodels-only,
consistent with what CU-B needed.

CU-C (2026-07-20) added the cell-type-AWARE twin needed by
``compute_ct_cell_communication_lowmem``: ``compute_ct_cell_communication`` (``:879``),
``run_ct_cell_communication_analysis`` (``:1004``), ``standardize_ct_counts`` (``:1346``),
``create_weights_ct_pairs`` (``:1367``), ``compute_metabolite_cs_ct`` (``:2321``),
``compute_ct_p_results`` (``:2517``), ``get_ct_cell_communication_results`` (``:2815``),
``normalize_ct_values`` (``:3134``), ``center_ct_counts_torch`` (``:3254``) -- all copied
verbatim from ``DataForClaude/cell_communication.py`` (line numbers refer to that file).
``center_ct_counts_torch`` calls ``models.apply_model_per_cell_type`` (real clone:
``harreman/hotspot/models.py:459``, also vendored below into ``_StubModels``), which for
``model='danb'`` slices cells by cell type and calls the SAME ``danb_model_torch`` already
vendored for CU-A/B -- confirmed by inspection that ``danb_model_torch``'s per-gene
reductions (``counts.sum(dim=1)``, ``.mean(dim=1)``) only ever touch that gene's own row, so
slicing by cell type does not break the CU-A/B row-independence finding. None of the CU-C
functions touch pydata ``sparse`` or numba -- verified by grepping each body (``flatten``,
``compute_max_cs``, ``compute_max_cs_gp`` are reused from the CU-B block above, unchanged).

All names must live in this one module (not spread across files) because the low-mem
drop-ins do ``inspect.getmodule(harreman.tools.compute_interacting_cell_scores)`` and
then look up each helper as an attribute of *that* module.
"""
from __future__ import annotations

import itertools
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import issparse
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from typing import Callable, Literal, Optional, Union
from anndata import AnnData


# ============================================================================
# VENDORED VERBATIM from harreman/preprocessing/anndata.py:9
# ============================================================================

def counts_from_anndata(adata, layer_key=None, dense=False):
    # 1. Extract matrix
    if layer_key is None:
        counts = adata.X
    elif layer_key == "use_raw":
        counts = adata.raw.X
    else:
        counts = adata.layers[layer_key]

    # 2. Transpose efficiently
    if issparse(counts):
        counts = counts.transpose().tocsr(copy=False)  # keep CSR format for efficient row slicing
    else:
        counts = counts.T  # transpose numpy array directly

    # 3. Convert to dense if requested
    if dense:
        if issparse(counts):
            counts = counts.toarray()
        else:
            counts = np.asarray(counts)

    return counts


# ============================================================================
# VENDORED VERBATIM from harreman/tools/knn.py:334
# ============================================================================

def make_weights_non_redundant(weights):

    w_no_redundant = weights.copy()

    rows, cols = w_no_redundant.nonzero()
    upper_diag_mask = rows < cols
    upper_rows, upper_cols = rows[upper_diag_mask], cols[upper_diag_mask]

    w_no_redundant[upper_rows, upper_cols] += w_no_redundant[upper_cols, upper_rows]
    w_no_redundant[upper_cols, upper_rows] = 0
    w_no_redundant.eliminate_zeros()

    return w_no_redundant


# ============================================================================
# VENDORED VERBATIM from harreman/hotspot/models.py:39 (danb branch only -- see
# module docstring for why the other three model variants are not vendored).
# ============================================================================

def danb_model_torch(counts: torch.Tensor, umi_counts: torch.Tensor, eps: float = 1e-10):
    """
    Vectorized DANB model computation in PyTorch for a batch of genes.

    Args:
        counts: Tensor of shape [genes, cells], gene expression counts.
        umi_counts: Tensor of shape [cells], total UMI per cell.
        eps: Small constant to avoid division by zero or log(0).

    Returns:
        mu: Mean per gene per cell [genes, cells]
        var: Variance per gene per cell [genes, cells]
        x2: Second moment per gene per cell [genes, cells]
    """
    tj = counts.sum(dim=1, keepdim=True)              # [genes, 1]
    total = umi_counts.sum()                          # scalar
    N = counts.shape[1]                               # number of cells

    mu = tj * umi_counts / total                      # [genes, cells]
    diff = counts - mu                                # [genes, cells]

    # Unbiased sample variance (N / (N - 1))
    var_gene = (diff ** 2).mean(dim=1) * N / (N - 1)  # [genes]

    numerator = ((tj ** 2) / total).squeeze() * (umi_counts ** 2).sum() / total  # [genes]
    denominator = (N - 1) * var_gene - tj.squeeze()                               # [genes]
    size = numerator / (denominator + eps)                                       # [genes]

    # Clamp size for numerical stability
    size = torch.where(size < 0, torch.tensor(1e9, device=size.device), size)
    size = torch.clamp(size, min=eps)

    size = size.unsqueeze(1)                      # [genes, 1] for broadcasting
    var = mu * (1 + mu / size)                    # [genes, cells]
    x2 = var + mu**2                              # [genes, cells]

    return mu, var, x2


def apply_model_per_cell_type(
    model_fn: Callable,
    counts: torch.Tensor,
    umi_counts: torch.Tensor,
    cell_types: Union[list, torch.Tensor],
    **kwargs
):
    """
    Applies a model function to each cell type separately.

    Args:
        model_fn: function of form (counts, umi_counts, **kwargs) -> (mu, var, x2)
        counts: [genes, cells] tensor
        umi_counts: [cells] tensor
        cell_types: list or tensor of cell type labels, length = cells
        kwargs: other model-specific arguments

    Returns:
        mu, var, x2: [genes, cells] tensors, concatenated across all cell types
    """
    device = counts.device

    unique_types = cell_types.unique()
    genes, cells = counts.shape

    mu_all = torch.empty((genes, cells), dtype=torch.float64, device=device)
    var_all = torch.empty((genes, cells), dtype=torch.float64, device=device)
    x2_all = torch.empty((genes, cells), dtype=torch.float64, device=device)

    cell_index = np.arange(cells)

    for ct in unique_types:

        idx_array = cell_index[cell_types.values == ct]
        idx = torch.tensor(idx_array, device=device)

        counts_ct = counts[:, idx]
        umi_ct = umi_counts[idx]

        mu, var, x2 = model_fn(counts_ct, umi_ct, **kwargs)

        mu_all[:, idx] = mu
        var_all[:, idx] = var
        x2_all[:, idx] = x2

    return mu_all, var_all, x2_all


class _StubModels:
    """Namespace mimicking harreman's ``hotspot.models`` module, referenced by
    ``center_counts_torch`` below as ``models.<variant>_model_torch``. Only ``danb`` is
    vendored (see module docstring); the other three attributes are intentionally absent
    -- ``center_counts_torch``'s other branches would raise ``AttributeError`` if ever
    reached, but our tests only pass ``model='danb'`` so they never are.
    """
    danb_model_torch = staticmethod(danb_model_torch)
    apply_model_per_cell_type = staticmethod(apply_model_per_cell_type)


models = _StubModels()


# ============================================================================
# VENDORED VERBATIM from harreman/hotspot/local_autocorrelation.py:376 and :449
# ============================================================================

def center_counts_torch(counts, num_umi, model):
    """
    counts: Tensor [genes, cells]
    num_umi: Tensor [cells]
    model: 'bernoulli', 'danb', 'normal', or 'none'

    Returns:
        Centered counts: Tensor [genes, cells]
    """
    # Binarize if using Bernoulli
    if model == 'bernoulli':
        counts = (counts > 0).double()
        mu, var, _ = models.bernoulli_model_torch(counts, num_umi)
    elif model == 'danb':
        mu, var, _ = models.danb_model_torch(counts, num_umi)
    elif model == 'normal':
        mu, var, _ = models.normal_model_torch(counts, num_umi)
    elif model == 'none':
        mu, var, _ = models.none_model_torch(counts, num_umi)
    else:
        raise ValueError(f"Unsupported model type: {model}")

    # Avoid division by zero
    std = torch.sqrt(var)
    std[std == 0] = 1.0

    centered = (counts - mu) / std
    centered[centered == 0] = 0  # Optional: to match old behavior

    return centered


def standardize_counts(adata, counts, model, num_umi, sample_specific):

    if sample_specific:
        sample_key = adata.uns['sample_key']
        for sample in adata.obs[sample_key].unique():
            subset = np.where(adata.obs[sample_key] == sample)[0]
            counts[:, subset] = center_counts_torch(counts[:, subset], num_umi[subset], model)
    else:
        counts = center_counts_torch(counts, num_umi, model)

    return counts


# ============================================================================
# VENDORED VERBATIM from DataForClaude/cell_communication.py -- do not hand-edit the
# logic below; if harreman's internals change, re-copy from the reference file.
# ============================================================================

# --- DataForClaude/cell_communication.py:2278 --------------------------------------
def compute_metabolite_cs(
    cs_gp: torch.Tensor,
    gene_pair_dict: dict,
    interacting_cell_scores: bool = False
) -> torch.Tensor:
    """
    Computes metabolite-level communication scores from gene-pair scores.

    Parameters
    ----------
    cs_gp : torch.Tensor
        - If interacting_cell_scores is False: shape (gene_pairs,)
        - If interacting_cell_scores is True: shape (cells, gene_pairs)
    gene_pair_dict : dict
        Maps metabolite names to a list of indices (ints) referring to gene-pairs.
    interacting_cell_scores : bool, optional
        Whether cs_gp contains per-cell scores.

    Returns
    -------
    cs_m : torch.Tensor
        - If interacting_cell_scores is False: shape (num_metabolites,)
        - If interacting_cell_scores is True: shape (cells, num_metabolites)
    """
    device = cs_gp.device
    scores = []

    for indices in gene_pair_dict.values():
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        if interacting_cell_scores:
            summed = cs_gp[:, idx_tensor].sum(dim=1)  # shape: (cells,)
        else:
            summed = cs_gp[idx_tensor].sum()          # scalar
        scores.append(summed)

    if interacting_cell_scores:
        cs_m = torch.stack(scores, dim=1)  # shape: (cells, metabolites)
    else:
        cs_m = torch.stack(scores)        # shape: (metabolites,)

    return cs_m


# --- DataForClaude/cell_communication.py:2720 --------------------------------------
def compute_p_int_cell_results_no_ct(C_gp, C_m, gene_pairs_ind, Wtot2, eg2s_gp, gene_pair_dict):

    device = Wtot2.device
    n_gp = len(gene_pairs_ind)

    # Convert indices
    same_gene_mask = torch.tensor([
        (isinstance(g1, int) and isinstance(g2, int) and g1 == g2) or
        (isinstance(g1, list) and isinstance(g2, list) and sorted(g1) == sorted(g2))
        for g1, g2 in gene_pairs_ind
    ], device=device)

    # Unpack second moments
    EG2_a = eg2s_gp[0].clone()
    EG2_b = eg2s_gp[1].clone()
    EG2_a[:, same_gene_mask] = Wtot2
    EG2_b[:, same_gene_mask] = Wtot2

    stdG_a = torch.sqrt(EG2_a)
    stdG_b = torch.sqrt(EG2_b)
    stdG_a[stdG_a == 0] = 1
    stdG_b[stdG_b == 0] = 1

    # Compute gene-pair Z-scores
    if isinstance(C_gp, tuple):
        C_gp_0, C_gp_1 = C_gp
        z_0 = C_gp_0 / stdG_a
        z_1 = C_gp_1 / stdG_b
        mask = torch.abs(z_0) < torch.abs(z_1)
        Z_gp = torch.where(mask, z_0, z_1)
        EG2_gp = torch.where(mask, EG2_a, EG2_b)
    else:
        C_gp = C_gp
        z_a = C_gp / stdG_a
        z_b = C_gp / stdG_b
        mask = torch.abs(z_a) < torch.abs(z_b)
        Z_gp = torch.where(mask, z_a, z_b)
        EG2_gp = torch.where(mask, EG2_a, EG2_b)

    # Compute metabolite-level expected variance
    EG2_m = compute_metabolite_cs(EG2_gp, gene_pair_dict, interacting_cell_scores=True)
    if not isinstance(EG2_m, torch.Tensor):
        EG2_m = torch.tensor(EG2_m, device=device, dtype=torch.float64)

    stdG_m = torch.sqrt(EG2_m)
    stdG_m[stdG_m == 0] = 1

    # Compute metabolite Z-scores
    if isinstance(C_m, tuple):
        C_m_0, C_m_1 = C_m
        z_0 = C_m_0 / stdG_m
        z_1 = C_m_1 / stdG_m
        Z_m = torch.where(torch.abs(z_0) < torch.abs(z_1), z_0, z_1)
    else:
        Z_m = C_m / stdG_m

    return (Z_gp, Z_m)


# --- DataForClaude/cell_communication.py:1473 --------------------------------------
def compute_interacting_cell_scores(
    adata: Union[str, AnnData],
    center_counts_for_np_test: Optional[bool] = False,
    test: Optional[Union[Literal["parametric"], Literal["non-parametric"], Literal["both"]]] = "both",
    restrict_significance: Optional[Union[Literal["gene pairs"], Literal["metabolites"], Literal["both"]]] = "both",
    compute_significance: Optional[Union[Literal["parametric"], Literal["non-parametric"], Literal["both"]]] = "both",
    M: Optional[int] = 1000,
    seed: Optional[int] = 42,
    check_analytic_null: Optional[bool] = False,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose: Optional[bool] = False,
):
    """
    Compute interacting cell scores for gene pairs and metabolites.

    Parameters
    ----------
    adata : AnnData or str
        AnnData object containing:
        - ``uns['model']`` and ``uns['mean']`` (expression normalization model)
        - ``uns['gene_pairs']``, ``uns['gene_pairs_per_metabolite']``
        - ``obsp['weights']``: sparse spatial weight matrix
        - ``uns['ccc_results']`` (for significance filtering)
    center_counts_for_np_test : bool, optional
        If True, center/normalize counts prior to the non-parametric test.
    test : {"parametric", "non-parametric", "both"}
        Which interacting cell score tests to compute.
    restrict_significance : {"gene pairs", "metabolites", "both"}
        Use only significant gene pairs/metabolites from CCC results.
    compute_significance : {"parametric", "non-parametric", "both"}
        Whether to compute significance (p-values, FDR) in each test.
    M : int, default 1000
        Number of permutations for the non-parametric test.
    seed : int, default 42
        Random seed for permutation reproducibility.
    check_analytic_null : bool, default False
        If True, evaluate the analytic null distribution during permutations.
    device : torch.device
        CPU or GPU device for tensor operations.
    verbose : bool, default False
        Print status updates.

    Returns
    -------
    None
        Results are stored in ``adata.uns['interacting_cell_results']``.
    """

    import time

    start = time.time()
    if verbose:
        print("Computing gene pair and metabolite scores...")

    adata.uns['interacting_cell_results'] = {}

    model = adata.uns["model"]
    mean = adata.uns["mean"]

    if test not in ['both', 'parametric', 'non-parametric']:
        raise ValueError('The "test" variable should be one of ["both", "parametric", "non-parametric"].')

    if restrict_significance is not None and restrict_significance not in ['both', 'gene pairs', 'metabolites']:
        raise ValueError('The "restrict_significance" variable should be one of ["both", "gene pairs", "metabolites"].')

    if compute_significance is not None and compute_significance not in ['both', 'parametric', 'non-parametric']:
        raise ValueError('The "compute_significance" variable should be one of ["both", "parametric", "non-parametric"].')

    sample_specific = 'sample_key' in adata.uns

    layer_key_p_test = adata.uns.get("layer_key_p_test", None)
    layer_key_np_test = adata.uns.get("layer_key_np_test", None)
    use_raw = (layer_key_p_test == "use_raw") and (layer_key_np_test == "use_raw")

    gene_pairs = adata.uns.get('gene_pairs', None)
    gene_pairs_per_metabolite = adata.uns['gene_pairs_per_metabolite']

    def to_tuple(x):
        # Recursively convert lists to tuples
        if isinstance(x, list):
            return tuple(to_tuple(i) for i in x)
        return x

    metabolite_gene_pair_df = pd.DataFrame.from_dict(gene_pairs_per_metabolite, orient="index").reset_index()
    metabolite_gene_pair_df = metabolite_gene_pair_df.rename(columns={"index": "metabolite"})
    metabolite_gene_pair_df['gene_pair'] = metabolite_gene_pair_df['gene_pair'].apply(
        lambda arr: [(to_tuple(gp[0]), to_tuple(gp[1])) for gp in arr]
    )
    metabolite_gene_pair_df['gene_type'] = metabolite_gene_pair_df['gene_type'].apply(
        lambda arr: [(to_tuple(gt[0]), to_tuple(gt[1])) for gt in arr]
    )
    metabolite_gene_pair_df = pd.concat([
        metabolite_gene_pair_df['metabolite'],
        metabolite_gene_pair_df.explode('gene_pair')['gene_pair'],
        metabolite_gene_pair_df.explode('gene_type')['gene_type'],
    ], axis=1).reset_index(drop=True)

    if 'LR_database' in adata.uns:
        LR_database = adata.uns['LR_database']
        df_merged = pd.merge(metabolite_gene_pair_df, LR_database, left_on='metabolite', right_on='interaction_name', how='left')
        LR_df = df_merged.dropna(subset=['pathway_name'])
        metabolite_gene_pair_df['metabolite'][metabolite_gene_pair_df.metabolite.isin(LR_df.metabolite)] = LR_df['pathway_name']

    if restrict_significance in ['both', 'gene pairs']:
        cell_com_gp_df = adata.uns['ccc_results']['cell_com_df_gp_sig'].copy()
        cell_com_gp_df[['Gene 1', 'Gene 2']] = cell_com_gp_df[['Gene 1', 'Gene 2']].applymap(
            lambda x: tuple(x) if isinstance(x, list) else x)

        gene_pairs_set = set([tuple(x) for x in cell_com_gp_df[['Gene 1', 'Gene 2']].values])
        metabolite_gene_pair_df = metabolite_gene_pair_df[metabolite_gene_pair_df['gene_pair'].isin(gene_pairs_set)]

    if restrict_significance in ['both', 'metabolites']:
        cell_com_m_df = adata.uns['ccc_results']['cell_com_df_m_sig'].copy()
        metabolite_set = set(cell_com_m_df['Metabolite'].values)
        metabolite_gene_pair_df = metabolite_gene_pair_df[metabolite_gene_pair_df['metabolite'].isin(metabolite_set)]

    genes = adata.uns["genes"]
    gene_pairs_sig = []
    if gene_pairs:
        for g1, g2 in gene_pairs:
            g1 = tuple(g1) if isinstance(g1, list) else g1
            g2 = tuple(g2) if isinstance(g2, list) else g2
            if not metabolite_gene_pair_df[metabolite_gene_pair_df['gene_pair'] == (g1, g2)].empty:
                gene_pairs_sig.append((g1, g2))

    adata.uns["gene_pairs_sig"] = gene_pairs_sig

    gene_pairs_sig_ind = []
    for g1, g2 in gene_pairs_sig:
        idx1 = tuple([genes.index(g) for g in g1]) if isinstance(g1, tuple) else genes.index(g1)
        idx2 = tuple([genes.index(g) for g in g2]) if isinstance(g2, tuple) else genes.index(g2)
        gene_pairs_sig_ind.append((idx1, idx2))

    adata.uns["gene_pairs_sig_ind"] = gene_pairs_sig_ind

    if 'barcode_key' in adata.uns:
        barcode_key = adata.uns['barcode_key']
        cells = pd.Series(adata.obs[barcode_key].tolist())
    else:
        cells = adata.obs_names if not use_raw else adata.raw.obs_names

    # Compute weights
    weights = make_weights_non_redundant(adata.obsp["weights"]).tocoo()
    weights = torch.sparse_coo_tensor(
        torch.tensor(np.vstack((weights.row, weights.col)), dtype=torch.long, device=device),
        torch.tensor(weights.data, dtype=torch.float64, device=device),
        torch.Size(weights.shape),
        device=device)

    gene_pair_dict = {}
    for metabolite, group in metabolite_gene_pair_df.groupby("metabolite"):
        idxs = group["gene_pair"].apply(lambda gp: gene_pairs_sig.index(gp) if gp in gene_pairs_sig else None).dropna().tolist()
        idxs = [int(ind) for ind in idxs if ind is not None]
        if idxs:
            gene_pair_dict[metabolite] = idxs
    metabolites = list(gene_pair_dict.keys())

    adata.uns['metabolites'] = metabolites

    gene_pairs_sig_names = [
        "_".join("_".join(g) if isinstance(g, tuple) else g for g in gp)
        for gp in gene_pairs_sig
    ]

    adata.uns['gene_pairs_sig_names'] = gene_pairs_sig_names

    if test in ['parametric', 'both']:

        if verbose:
            print("Running the parametric test...")

        adata.uns['interacting_cell_results']['p'] = {
            'gp': {},
            'm': {}
        }

        Wtot2 = torch.tensor((weights.data ** 2).sum(), device=device)

        # Load counts
        counts = counts_from_anndata(adata[cells, genes], layer_key_p_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        num_umi = counts.sum(dim=0)

        # Prepare counts_1 and counts_2
        counts_1 = []
        counts_2 = []
        for (idx1, idx2) in gene_pairs_sig_ind:
            if isinstance(idx1, tuple):
                c1 = counts[idx1, :].mean(dim=0) if mean == 'algebraic' else torch.exp(torch.log(counts[idx1, :] + 1e-8).mean(dim=0))
            else:
                c1 = counts[idx1, :]
            if isinstance(idx2, tuple):
                c2 = counts[idx2, :].mean(dim=0) if mean == 'algebraic' else torch.exp(torch.log(counts[idx2, :] + 1e-8).mean(dim=0))
            else:
                c2 = counts[idx2, :]
            counts_1.append(c1)
            counts_2.append(c2)

        counts_1 = torch.stack(counts_1)
        counts_2 = torch.stack(counts_2)

        counts_1 = standardize_counts(adata, counts_1, model, num_umi, sample_specific)
        counts_2 = standardize_counts(adata, counts_2, model, num_umi, sample_specific)

        # Compute CCC scores
        WX2t = torch.sparse.mm(weights, counts_2.T)
        WtX2t = torch.sparse.mm(weights.transpose(0, 1), counts_2.T)
        cs_gp = (counts_1.T * WX2t) + (counts_1.T * WtX2t)
        same_gene_mask = torch.tensor([g1 == g2 for g1, g2 in gene_pairs_sig], device=device)
        cs_gp[:, same_gene_mask] = cs_gp[:, same_gene_mask] / 2
        adata.uns['interacting_cell_results']['p']['gp']['cs'] = cs_gp.detach().cpu().numpy()

        # Compute metabolite-level scores
        cs_m = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=True)
        adata.uns['interacting_cell_results']['p']['m']['cs'] = cs_m.detach().cpu().numpy()

        if compute_significance in ['parametric', 'both']:
            # Compute second moments
            WX1t = torch.sparse.mm(weights, counts_1.T)
            WtX1t = torch.sparse.mm(weights.transpose(0, 1), counts_1.T)
            eg2_a = (WX1t + WtX1t).pow(2)
            eg2_b = (WX2t + WtX2t).pow(2)
            eg2s_gp = (eg2_a, eg2_b)

            Z_gp, Z_m = compute_p_int_cell_results_no_ct(cs_gp, cs_m, gene_pairs_sig_ind, Wtot2, eg2s_gp, gene_pair_dict)

            Z_gp_np = Z_gp.detach().cpu().numpy()
            Z_m_np = Z_m.detach().cpu().numpy()
            # Compute p-values and FDRs
            Z_pvals_gp = norm.sf(Z_gp_np)
            Z_pvals_m = norm.sf(Z_m_np)
            FDR_gp = multipletests(Z_pvals_gp.flatten(), method="fdr_bh")[1].reshape(Z_pvals_gp.shape)
            FDR_m = multipletests(Z_pvals_m.flatten(), method="fdr_bh")[1].reshape(Z_pvals_m.shape)

            adata.uns['interacting_cell_results']['p']['gp']['Z'] = Z_gp_np
            adata.uns['interacting_cell_results']['p']['gp']['Z_pval'] = Z_pvals_gp
            adata.uns['interacting_cell_results']['p']['gp']['Z_FDR'] = FDR_gp
            adata.uns['interacting_cell_results']['p']['m']['Z'] = Z_m_np
            adata.uns['interacting_cell_results']['p']['m']['Z_pval'] = Z_pvals_m
            adata.uns['interacting_cell_results']['p']['m']['Z_FDR'] = FDR_m

            # P-value
            mask_gp = adata.uns['interacting_cell_results']['p']['gp']['Z_pval'] < 0.05
            mask_m = adata.uns['interacting_cell_results']['p']['m']['Z_pval'] < 0.05

            cs_gp_sig = adata.uns['interacting_cell_results']['p']['gp']['cs'].copy()
            cs_m_sig = adata.uns['interacting_cell_results']['p']['m']['cs'].copy()

            cs_gp_sig[~mask_gp] = np.nan
            cs_m_sig[~mask_m] = np.nan
            adata.uns['interacting_cell_results']['p']['gp']['cs_sig_pval'] = cs_gp_sig
            adata.uns['interacting_cell_results']['p']['m']['cs_sig_pval'] = cs_m_sig

            # FDR
            mask_gp = adata.uns['interacting_cell_results']['p']['gp']['Z_FDR'] < 0.05
            mask_m = adata.uns['interacting_cell_results']['p']['m']['Z_FDR'] < 0.05

            cs_gp_sig = adata.uns['interacting_cell_results']['p']['gp']['cs'].copy()
            cs_m_sig = adata.uns['interacting_cell_results']['p']['m']['cs'].copy()

            cs_gp_sig[~mask_gp] = np.nan
            cs_m_sig[~mask_m] = np.nan
            adata.uns['interacting_cell_results']['p']['gp']['cs_sig_FDR'] = cs_gp_sig
            adata.uns['interacting_cell_results']['p']['m']['cs_sig_FDR'] = cs_m_sig

        if verbose:
            print("Parametric test finished.")


    if test in ["non-parametric", "both"]:

        if verbose:
            print("Running the non-parametric test...")

        adata.uns['interacting_cell_results']['np'] = {
            'gp': {},
            'm': {}
        }

        # Load counts
        counts = counts_from_anndata(adata[cells, genes], layer_key_np_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)

        # Prepare counts_1 and counts_2
        counts_1 = []
        counts_2 = []
        for (idx1, idx2) in gene_pairs_sig_ind:
            if isinstance(idx1, tuple):
                c1 = counts[idx1, :].mean(dim=0) if mean == 'algebraic' else torch.exp(torch.log(counts[idx1, :] + 1e-8).mean(dim=0))
            else:
                c1 = counts[idx1, :]
            if isinstance(idx2, tuple):
                c2 = counts[idx2, :].mean(dim=0) if mean == 'algebraic' else torch.exp(torch.log(counts[idx2, :] + 1e-8).mean(dim=0))
            else:
                c2 = counts[idx2, :]
            counts_1.append(c1)
            counts_2.append(c2)

        counts_1 = torch.stack(counts_1)
        counts_2 = torch.stack(counts_2)

        if center_counts_for_np_test:
            num_umi = counts.sum(dim=0)
            counts_1 = standardize_counts(adata, counts_1, model, num_umi, sample_specific)
            counts_2 = standardize_counts(adata, counts_2, model, num_umi, sample_specific)

        n_cells = counts_1.shape[1]
        same_gene_mask = torch.tensor([g1 == g2 for g1, g2 in gene_pairs_sig], device=device)

        if center_counts_for_np_test and test == 'both':
            adata.uns['interacting_cell_results']['np']['gp']['cs'] = np.array(adata.uns['interacting_cell_results']['p']['gp']['cs'])
            adata.uns['interacting_cell_results']['np']['m']['cs'] = np.array(adata.uns['interacting_cell_results']['p']['m']['cs'])
        else:
            WX2t = torch.sparse.mm(weights, counts_2.T)
            WtX2t = torch.sparse.mm(weights.transpose(0, 1), counts_2.T)
            cs_gp = (counts_1.T * WX2t) + (counts_1.T * WtX2t)
            cs_gp[:, same_gene_mask] = cs_gp[:, same_gene_mask] / 2
            adata.uns['interacting_cell_results']['np']['gp']['cs'] = cs_gp.detach().cpu().numpy()
            cs_m = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=True)
            adata.uns['interacting_cell_results']['np']['m']['cs'] = cs_m.detach().cpu().numpy()

        if compute_significance in ['non-parametric', 'both']:
            perm_cs_gp_a = torch.zeros((n_cells, counts_1.shape[0], M), dtype=torch.float64, device=device)
            perm_cs_gp_b = torch.zeros_like(perm_cs_gp_a)
            perm_cs_m_a = torch.zeros((n_cells, len(gene_pair_dict), M), dtype=torch.float64, device=device)
            perm_cs_m_b = torch.zeros_like(perm_cs_m_a)

            if check_analytic_null:
                gp_zs_perm_array = torch.zeros_like(perm_cs_gp_a)
                gp_pvals_perm_array = torch.zeros_like(perm_cs_gp_a)
                m_zs_perm_array = torch.zeros_like(perm_cs_m_a)
                m_pvals_perm_array = torch.zeros_like(perm_cs_m_a)

            torch.manual_seed(seed)
            for i in tqdm(range(M), desc="Permutation test"):
                idx = torch.randperm(n_cells, device=device)

                c1_perm_a = counts_1.clone()
                c2_perm_a = counts_2[:, idx]
                c1_perm_a[same_gene_mask] = counts_1[same_gene_mask, :][:, idx]

                WX2t_a = torch.sparse.mm(weights, c2_perm_a.T)
                WtX2t_a = torch.sparse.mm(weights.transpose(0, 1), c2_perm_a.T)
                cs_a = (c1_perm_a.T * WX2t_a) + (c1_perm_a.T * WtX2t_a)
                cs_a[:, same_gene_mask] = cs_a[:, same_gene_mask] / 2
                perm_cs_gp_a[:, :, i] = cs_a

                cs_m_a = compute_metabolite_cs(cs_a, gene_pair_dict, interacting_cell_scores=True)
                perm_cs_m_a[:, :, i] = cs_m_a

                c2_perm_b = counts_2.clone()
                c1_perm_b = counts_1[:, idx]
                c2_perm_b[same_gene_mask] = counts_2[same_gene_mask, :][:, idx]

                WX2t_b = torch.sparse.mm(weights, c2_perm_b.T)
                WtX2t_b = torch.sparse.mm(weights.transpose(0, 1), c2_perm_b.T)
                cs_b = (c1_perm_b.T * WX2t_b) + (c1_perm_b.T * WtX2t_b)
                cs_b[:, same_gene_mask] = cs_b[:, same_gene_mask] / 2
                perm_cs_gp_b[:, :, i] = cs_b

                cs_m_b = compute_metabolite_cs(cs_b, gene_pair_dict, interacting_cell_scores=True)
                perm_cs_m_b[:, :, i] = cs_m_b

                if check_analytic_null:
                    Z_gp_perm, Z_m_perm = compute_p_results((cs_a, cs_b), (cs_m_a, cs_m_b), gene_pairs_ind, Wtot2, eg2s_gp, gene_pair_dict)
                    gp_zs_perm_array[:, :, i] = Z_gp_perm
                    gp_pvals_perm_array[:, :, i] = torch.tensor(norm.sf(Z_gp_perm.cpu().numpy()), device=device)
                    m_zs_perm_array[:, :, i] = Z_m_perm
                    m_pvals_perm_array[:, :, i] = torch.tensor(norm.sf(Z_m_perm.cpu().numpy()), device=device)

            adata.uns['interacting_cell_results']['np']['gp']['perm_cs_a'] = perm_cs_gp_a.detach().cpu().numpy()
            adata.uns['interacting_cell_results']['np']['gp']['perm_cs_b'] = perm_cs_gp_b.detach().cpu().numpy()
            adata.uns['interacting_cell_results']['np']['m']['perm_cs_a'] = perm_cs_m_a.detach().cpu().numpy()
            adata.uns['interacting_cell_results']['np']['m']['perm_cs_b'] = perm_cs_m_b.detach().cpu().numpy()

            x_gp_a = (perm_cs_gp_a > cs_gp[:, :, None]).sum(dim=2)
            x_gp_b = (perm_cs_gp_b > cs_gp[:, :, None]).sum(dim=2)
            x_m_a = (perm_cs_m_a > cs_m[:, :, None]).sum(dim=2)
            x_m_b = (perm_cs_m_b > cs_m[:, :, None]).sum(dim=2)

            pvals_gp_a = (x_gp_a + 1).float() / (M + 1)
            pvals_gp_b = (x_gp_b + 1).float() / (M + 1)
            pvals_m_a = (x_m_a + 1).float() / (M + 1)
            pvals_m_b = (x_m_b + 1).float() / (M + 1)

            pvals_gp = torch.where(pvals_gp_a > pvals_gp_b, pvals_gp_a, pvals_gp_b)
            pvals_m = torch.where(pvals_m_a > pvals_m_b, pvals_m_a, pvals_m_b)

            pvals_gp = pvals_gp.cpu().numpy()
            pvals_m = pvals_m.cpu().numpy()

            adata.uns['interacting_cell_results']['np']['gp']['pval'] = pvals_gp
            adata.uns['interacting_cell_results']['np']['gp']['FDR'] = multipletests(pvals_gp.flatten(), method="fdr_bh")[1].reshape(pvals_gp.shape)
            adata.uns['interacting_cell_results']['np']['m']['pval'] = pvals_m
            adata.uns['interacting_cell_results']['np']['m']['FDR'] = multipletests(pvals_m.flatten(), method="fdr_bh")[1].reshape(pvals_m.shape)

            if check_analytic_null:
                adata.uns['interacting_cell_results']['np']['analytic_null'] = {
                    'gp_zs_perm': gp_zs_perm_array.detach().cpu().numpy(),
                    'gp_pvals_perm': gp_pvals_perm_array.detach().cpu().numpy(),
                    'm_zs_perm': m_zs_perm_array.detach().cpu().numpy(),
                    'm_pvals_perm': m_pvals_perm_array.detach().cpu().numpy(),
                }

            # P-value
            mask_gp = adata.uns['interacting_cell_results']['np']['gp']['pval'] < 0.05
            mask_m = adata.uns['interacting_cell_results']['np']['m']['pval'] < 0.05

            cs_gp_sig = adata.uns['interacting_cell_results']['np']['gp']['cs'].copy()
            cs_m_sig = adata.uns['interacting_cell_results']['np']['m']['cs'].copy()

            cs_gp_sig[~mask_gp] = np.nan
            cs_m_sig[~mask_m] = np.nan
            adata.uns['interacting_cell_results']['np']['gp']['cs_sig_pval'] = cs_gp_sig
            adata.uns['interacting_cell_results']['np']['m']['cs_sig_pval'] = cs_m_sig

            # FDR
            mask_gp = adata.uns['interacting_cell_results']['np']['gp']['FDR'] < 0.05
            mask_m = adata.uns['interacting_cell_results']['np']['m']['FDR'] < 0.05

            cs_gp_sig = adata.uns['interacting_cell_results']['np']['gp']['cs'].copy()
            cs_m_sig = adata.uns['interacting_cell_results']['np']['m']['cs'].copy()

            cs_gp_sig[~mask_gp] = np.nan
            cs_m_sig[~mask_m] = np.nan
            adata.uns['interacting_cell_results']['np']['gp']['cs_sig_FDR'] = cs_gp_sig
            adata.uns['interacting_cell_results']['np']['m']['cs_sig_FDR'] = cs_m_sig

        if verbose:
            print("Non-parametric test finished.")

    if verbose:
        print("Finished computing gene pair and metabolite scores in %.3f seconds" % (time.time() - start))

    return


# ============================================================================
# VENDORED VERBATIM from DataForClaude/cell_communication.py -- CU-B (the two
# cell-type-INDEPENDENT aggregate functions + their transitive deps needed by
# `compute_cell_communication_lowmem` in cell_communication_lowmem.py). Same
# provenance/fidelity rule as the CU-A block above: copied exactly, not
# reimplemented; do not hand-edit the logic below -- if harreman's internals
# change, re-copy from the reference file.
# ============================================================================

# --- DataForClaude/cell_communication.py:1359 ----------------------------------------
def flatten(nested_list):
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item

# --- DataForClaude/cell_communication.py:2539 ----------------------------------------
def compute_p_results(C_gp, C_m, gene_pairs_ind, Wtot2, eg2s_gp, gene_pair_dict):
    
    device = Wtot2.device
    n_gp = len(gene_pairs_ind)

    # Convert indices
    same_gene_mask = torch.tensor([
        (isinstance(g1, int) and isinstance(g2, int) and g1 == g2) or
        (isinstance(g1, list) and isinstance(g2, list) and sorted(g1) == sorted(g2))
        for g1, g2 in gene_pairs_ind
    ], device=device)
    
    # Unpack second moments
    EG2_a = eg2s_gp[0].clone()
    EG2_b = eg2s_gp[1].clone()
    EG2_a[same_gene_mask] = Wtot2
    EG2_b[same_gene_mask] = Wtot2

    stdG_a = torch.sqrt(EG2_a)
    stdG_b = torch.sqrt(EG2_b)
    stdG_a[stdG_a == 0] = 1
    stdG_b[stdG_b == 0] = 1
    
    # Compute gene-pair Z-scores
    if isinstance(C_gp, tuple):
        C_gp_0, C_gp_1 = C_gp
        z_0 = C_gp_0 / stdG_a
        z_1 = C_gp_1 / stdG_b
        mask = torch.abs(z_0) < torch.abs(z_1)
        Z_gp = torch.where(mask, z_0, z_1)
        EG2_gp = torch.where(mask, EG2_a, EG2_b)
    else:
        C_gp = C_gp
        z_a = C_gp / stdG_a
        z_b = C_gp / stdG_b
        mask = torch.abs(z_a) < torch.abs(z_b)
        Z_gp = torch.where(mask, z_a, z_b)
        EG2_gp = torch.where(mask, EG2_a, EG2_b)

    # Compute metabolite-level expected variance
    EG2_m = compute_metabolite_cs(EG2_gp, gene_pair_dict, interacting_cell_scores=False)
    if not isinstance(EG2_m, torch.Tensor):
        EG2_m = torch.tensor(EG2_m, device=device, dtype=torch.float64)

    stdG_m = torch.sqrt(EG2_m)
    stdG_m[stdG_m == 0] = 1

    # Compute metabolite Z-scores
    if isinstance(C_m, tuple):
        C_m_0, C_m_1 = C_m
        z_0 = C_m_0 / stdG_m
        z_1 = C_m_1 / stdG_m
        Z_m = torch.where(torch.abs(z_0) < torch.abs(z_1), z_0, z_1)
    else:
        Z_m = C_m / stdG_m

    return Z_gp, Z_m


# --- DataForClaude/cell_communication.py:464 ----------------------------------------
def compute_cell_communication(
    adata: AnnData,
    layer_key_p_test: Optional[Union[Literal["use_raw"], str]] = None,
    layer_key_np_test: Optional[Union[Literal["use_raw"], str]] = None,
    model: str = None,
    center_counts_for_np_test: Optional[bool] = False,
    subset_gene_pairs: Optional[str] = None,
    M: Optional[int] = 1000,
    seed: Optional[int] = 42,
    test: Optional[Union[Literal["parametric"], Literal["non-parametric"], Literal["both"]]] = "both",
    mean: Optional[Union[Literal["algebraic"], Literal["geometric"]]] = 'algebraic',
    check_analytic_null: Optional[bool] = False,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose: Optional[bool] = False,
):
    """
    Computes spatially-informed cell-type-agnostic cell-cell communication (CCC) scores and 
    significance across all gene pairs using both parametric and non-parametric statistical tests.

    Parameters
    ----------
    adata : AnnData
        Annotated data object. Required fields include:
            - `uns["gene_pairs"]`: list of gene pairs to evaluate.
            - `uns["gene_pairs_per_metabolite"]`: dictionary mapping metabolites to gene pairs.
            - `obsp["weights"]`: sparse matrix encoding spatial cell-cell proximity.
            - (Optional) `uns["LR_database"]`: interaction metadata for pathway annotation.
            - (Optional) `uns["sample_key"]`: if modeling includes sample-specific factors.
    layer_key_p_test : str or "use_raw", optional
        Data layer to use for the parametric test. If `"use_raw"`, uses `adata.raw`.
    layer_key_np_test : str or "use_raw", optional
        Data layer to use for the non-parametric test. If `"use_raw"`, uses `adata.raw`.
    model : str, optional
        Normalization model to use for centering gene expression. Options include "none", "normal", "bernoulli", or "danb".
    center_counts_for_np_test : bool, optional (default: False)
        Whether to center expression counts using the specified model before non-parametric testing.
    subset_gene_pairs : list, optional
        If provided, restricts the analysis to this subset of gene pairs.
    M : int, optional (default: 1000)
        Number of permutations to use if `permutation_test` is True.
    seed : int, optional (default: 42)
        Random seed for permutation reproducibility.
    test : {'parametric', 'non-parametric', 'both'}, optional (default: 'both')
        Specifies which statistical test(s) to run.
    mean : {'algebraic', 'geometric'}, optional (default: 'algebraic')
        Averaging method for multi-gene interactions.
    check_analytic_null : bool, optional (default: False)
        Whether to evaluate Z-scores under an analytic null distribution using permutation Z-scores.
    device : torch.device, optional
        PyTorch device to run computations on. Defaults to CUDA if available.
    verbose : bool, optional (default: False)
        Whether to print progress and status messages.

    Returns
    -------
    None
        Results are stored in the following `adata.uns` fields:
            - `uns["ccc_results"]["p"]`: Parametric test results (gene pair and metabolite scores, Z, p-values, FDR).
            - `uns["ccc_results"]["np"]`: Non-parametric test results (communication scores, empirical p-values, FDR).
            - `uns["lc_zs"]`: Symmetric matrix of ligand-receptor Z-scores across genes.
            - `uns["gene_pair_dict"]`: Dictionary mapping metabolites to index positions of gene pairs.
            - `uns["D"]`: Vector of total node degrees per cell (spatial connectivity).
            - `uns["genes"]`: Ordered list of involved genes.
            - `uns["gene_pairs_ind"]`: Index-referenced version of `uns["gene_pairs"]`.
    """
    
    start = time.time()
    if verbose:
        print("Starting cell-cell communication analysis...")

    adata.uns['ccc_results'] = {}

    if test not in ['both', 'parametric', 'non-parametric']:
        raise ValueError('The "test" variable should be one of ["both", "parametric", "non-parametric"].')
    
    if mean not in ['algebraic', 'geometric']:
        raise ValueError('The "mean" variable should be one of ["algebraic", "geometric"].')
    
    adata.uns['layer_key_p_test'] = layer_key_p_test
    adata.uns['layer_key_np_test'] = layer_key_np_test
    adata.uns['model'] = model
    adata.uns['center_counts_for_np_test'] = center_counts_for_np_test
    adata.uns['mean'] = mean

    run_cell_communication_analysis(adata, layer_key_p_test, layer_key_np_test, model, center_counts_for_np_test, subset_gene_pairs, M, seed, test, mean, check_analytic_null, device, verbose)

    if verbose:
        print("Obtaining communication results...")
    get_cell_communication_results(
        adata,
        adata.uns["genes"],
        layer_key_p_test,
        layer_key_np_test,
        model,
        adata.uns["D"],
        test,
        device,
    )

    if verbose:
        print("Finished computing cell-cell communication analysis in %.3f seconds" %(time.time()-start))

    return

# --- DataForClaude/cell_communication.py:569 ----------------------------------------
def run_cell_communication_analysis(
    adata,
    layer_key_p_test,
    layer_key_np_test,
    model,
    center_counts_for_np_test,
    subset_gene_pairs,
    M,
    seed,
    test,
    mean,
    check_analytic_null,
    device,
    verbose,
):
    
    use_raw = (layer_key_p_test == "use_raw") & (layer_key_np_test == "use_raw")

    cells = adata.raw.obs.index.values.astype(str) if use_raw else adata.obs_names.values.astype(str)
    
    sample_specific = 'sample_key' in adata.uns

    gene_pairs = adata.uns["gene_pairs"] if subset_gene_pairs is None else subset_gene_pairs
    genes = list(np.unique(list(flatten(adata.uns["gene_pairs"]))))
    adata.uns["genes"] = genes
    adata.uns["cells"] = cells
    
    # Map gene_pairs to index
    gene_pairs_ind = []
    for pair in gene_pairs:
        idx1 = [genes.index(g) for g in pair[0]] if isinstance(pair[0], list) else genes.index(pair[0])
        idx2 = [genes.index(g) for g in pair[1]] if isinstance(pair[1], list) else genes.index(pair[1])
        gene_pairs_ind.append((idx1, idx2))
    adata.uns["gene_pairs_ind"] = gene_pairs_ind
    
    # Compute weights
    weights = make_weights_non_redundant(adata.obsp["weights"]).tocoo()
    weights = torch.sparse_coo_tensor(
        torch.tensor(np.vstack((weights.row, weights.col)), dtype=torch.long, device=device),
        torch.tensor(weights.data, dtype=torch.float64, device=device),
        torch.Size(weights.shape), 
        device=device)
    
    # Compute node degree
    row_degrees = torch.sparse.sum(weights, dim=1).to_dense()
    col_degrees = torch.sparse.sum(weights, dim=0).to_dense()
    D = row_degrees + col_degrees

    adata.uns["D"] = D.cpu().numpy()
    
    gene_pairs_per_metabolite = adata.uns['gene_pairs_per_metabolite']

    metabolite_gene_pair_df = pd.DataFrame.from_dict(gene_pairs_per_metabolite, orient="index").reset_index()
    metabolite_gene_pair_df = metabolite_gene_pair_df.rename(columns={"index": "metabolite"})

    metabolite_gene_pair_df['gene_pair'] = metabolite_gene_pair_df['gene_pair'].apply(lambda arr: [(sub_array[0], sub_array[1]) for sub_array in arr])
    metabolite_gene_pair_df['gene_type'] = metabolite_gene_pair_df['gene_type'].apply(lambda arr: [(sub_array[0], sub_array[1]) for sub_array in arr])

    metabolite_gene_pair_df = pd.concat(
        [
            metabolite_gene_pair_df['metabolite'],
            metabolite_gene_pair_df.explode('gene_pair')['gene_pair'],
            metabolite_gene_pair_df.explode('gene_type')['gene_type'],
        ],
        axis=1,
    )
    metabolite_gene_pair_df = metabolite_gene_pair_df.reset_index(drop=True)
    
    if 'LR_database' in adata.uns.keys():
        LR_database = adata.uns['LR_database']
        df_merged = pd.merge(metabolite_gene_pair_df, LR_database, left_on='metabolite', right_on='interaction_name', how='left')
        LR_df = df_merged.dropna(subset=['pathway_name'])
        metabolite_gene_pair_df['metabolite'][metabolite_gene_pair_df.metabolite.isin(LR_df.metabolite)] = LR_df['pathway_name']
    
    gene_pair_dict = {}
    for metabolite, group in metabolite_gene_pair_df.groupby("metabolite"):
        idxs = group["gene_pair"].apply(lambda gp: gene_pairs.index(gp) if gp in gene_pairs else None).dropna().tolist()
        idxs = [int(ind) for ind in idxs if ind is not None]
        if idxs:
            gene_pair_dict[metabolite] = idxs
    
    adata.uns["gene_pair_dict"] = gene_pair_dict

    if test in ['parametric', 'both']:
        
        if verbose:
            print("Running the parametric test...")
        
        adata.uns['ccc_results']['p'] = {
            'gp': {},
            'm': {}
        }

        Wtot2 = torch.tensor((weights.data ** 2).sum(), device=device)

        # Load counts
        counts = counts_from_anndata(adata[cells, genes], layer_key_p_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        num_umi = counts.sum(dim=0)
        
        # Prepare counts_1 and counts_2
        counts_1 = []
        counts_2 = []
        for (idx1, idx2) in gene_pairs_ind:
            if isinstance(idx1, list):
                c1 = counts[idx1, :].mean(dim=0) if mean == 'algebraic' else torch.exp(torch.log(counts[idx1, :] + 1e-8).mean(dim=0))
            else:
                c1 = counts[idx1, :]
            if isinstance(idx2, list):
                c2 = counts[idx2, :].mean(dim=0) if mean == 'algebraic' else torch.exp(torch.log(counts[idx2, :] + 1e-8).mean(dim=0))
            else:
                c2 = counts[idx2, :]
            counts_1.append(c1)
            counts_2.append(c2)

        counts_1 = torch.stack(counts_1)
        counts_2 = torch.stack(counts_2)

        # Standardize counts
        counts_1 = standardize_counts(adata, counts_1, model, num_umi, sample_specific)
        counts_2 = standardize_counts(adata, counts_2, model, num_umi, sample_specific)
        
        # Compute CCC scores
        WX2t = torch.sparse.mm(weights, counts_2.T)
        WtX2t = torch.sparse.mm(weights.transpose(0, 1), counts_2.T)
        cs_gp = (counts_1.T * WX2t).sum(0) + (counts_1.T * WtX2t).sum(0)
        same_gene_mask = torch.tensor([g1 == g2 for g1, g2 in gene_pairs], device=device)
        cs_gp[same_gene_mask] = cs_gp[same_gene_mask] / 2
        adata.uns['ccc_results']['p']['gp']['cs'] = cs_gp.detach().cpu().numpy()
        
        # Compute metabolite-level scores
        cs_m = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=False)
        adata.uns['ccc_results']['p']['m']['cs'] = cs_m.detach().cpu().numpy()
                
        # Compute second moments
        WX1t = torch.sparse.mm(weights, counts_1.T)
        WtX1t = torch.sparse.mm(weights.transpose(0, 1), counts_1.T)
        eg2_a = (WX1t + WtX1t).pow(2).sum(dim=0)
        eg2_b = (WX2t + WtX2t).pow(2).sum(dim=0)
        eg2s_gp = (eg2_a, eg2_b)

        # Z-score computation
        Z_gp, Z_m = compute_p_results(cs_gp, cs_m, gene_pairs_ind, Wtot2, eg2s_gp, gene_pair_dict)
        # Convert tensors to numpy for statsmodels and pandas
        Z_gp_np = Z_gp.detach().cpu().numpy()
        Z_m_np = Z_m.detach().cpu().numpy()
        # Compute p-values and FDRs
        Z_pvals_gp = norm.sf(Z_gp_np)
        Z_pvals_m = norm.sf(Z_m_np)
        FDR_gp = multipletests(Z_pvals_gp, method="fdr_bh")[1]
        FDR_m = multipletests(Z_pvals_m, method="fdr_bh")[1]
        
        # Store in AnnData
        adata.uns['ccc_results']['p']['gp']['Z'] = Z_gp_np
        adata.uns['ccc_results']['p']['gp']['Z_pval'] = Z_pvals_gp
        adata.uns['ccc_results']['p']['gp']['Z_FDR'] = FDR_gp
        adata.uns['ccc_results']['p']['m']['Z'] = Z_m_np
        adata.uns['ccc_results']['p']['m']['Z_pval'] = Z_pvals_m
        adata.uns['ccc_results']['p']['m']['Z_FDR'] = FDR_m
        
        # Symmetric LC Z-score matrix
        genes_ = [tuple(i) if isinstance(i, list) else i for i in pd.Series([g for pair in gene_pairs for g in pair]).drop_duplicates()]
        gene_pairs_ = [(tuple(a) if isinstance(a, list) else a, tuple(b) if isinstance(b, list) else b) for a, b in gene_pairs]
        lc_zs = pd.DataFrame(np.zeros((len(genes_), len(genes_))), index=genes_, columns=genes_)
        for i, (g1, g2) in enumerate(gene_pairs_):
            lc_zs.iloc[genes_.index(g1), genes_.index(g2)] = Z_gp_np[i]
        # Force diagonal to 0 and symmetrize
        np.fill_diagonal(lc_zs.values, 0)
        adata.uns['lc_zs'] = (lc_zs + lc_zs.T) / 2
        
        if verbose:
            print("Parametric test finished.")

    if test in ["non-parametric", "both"]:
        
        if verbose:
            print("Running the non-parametric test...")
        
        adata.uns['ccc_results']['np'] = {
            'gp': {},
            'm': {}
        }
        
        # Load counts
        counts = counts_from_anndata(adata[cells, genes], layer_key_np_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        
        # Prepare counts_1 and counts_2
        counts_1 = []
        counts_2 = []
        for (idx1, idx2) in gene_pairs_ind:
            if isinstance(idx1, list):
                c1 = counts[idx1, :].mean(dim=0) if mean == 'algebraic' else torch.exp(torch.log(counts[idx1, :] + 1e-8).mean(dim=0))
            else:
                c1 = counts[idx1, :]
            if isinstance(idx2, list):
                c2 = counts[idx2, :].mean(dim=0) if mean == 'algebraic' else torch.exp(torch.log(counts[idx2, :] + 1e-8).mean(dim=0))
            else:
                c2 = counts[idx2, :]
            counts_1.append(c1)
            counts_2.append(c2)

        counts_1 = torch.stack(counts_1)
        counts_2 = torch.stack(counts_2)
        
        if center_counts_for_np_test:
            num_umi = counts.sum(dim=0)
            counts_1 = standardize_counts(adata, counts_1, model, num_umi, sample_specific)
            counts_2 = standardize_counts(adata, counts_2, model, num_umi, sample_specific)
        
        n_cells = counts_1.shape[1]
        same_gene_mask = torch.tensor([g1 == g2 for g1, g2 in gene_pairs], device=device)
        
        if center_counts_for_np_test and test == 'both':
            adata.uns['ccc_results']['np']['gp']['cs'] = np.array(adata.uns['ccc_results']['p']['gp']['cs'])
            adata.uns['ccc_results']['np']['m']['cs'] = np.array(adata.uns['ccc_results']['p']['m']['cs'])
        else:
            WX2t = torch.sparse.mm(weights, counts_2.T)
            WtX2t = torch.sparse.mm(weights.transpose(0, 1), counts_2.T)
            cs_gp = (counts_1.T * WX2t).sum(0) + (counts_1.T * WtX2t).sum(0)
            cs_gp[same_gene_mask] = cs_gp[same_gene_mask] / 2
            adata.uns['ccc_results']['np']['gp']['cs'] = cs_gp.detach().cpu().numpy()
            cs_m = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=False)
            adata.uns['ccc_results']['np']['m']['cs'] = cs_m.detach().cpu().numpy()
        
        perm_cs_gp_a = torch.zeros((counts_1.shape[0], M), dtype=torch.float64, device=device)
        perm_cs_gp_b = torch.zeros_like(perm_cs_gp_a)
        perm_cs_m_a = torch.zeros((len(gene_pair_dict), M), dtype=torch.float64, device=device)
        perm_cs_m_b = torch.zeros_like(perm_cs_m_a)
        
        if check_analytic_null:
            gp_zs_perm_array = torch.zeros_like(perm_cs_gp_a)
            gp_pvals_perm_array = torch.zeros_like(perm_cs_gp_a)
            m_zs_perm_array = torch.zeros_like(perm_cs_m_a)
            m_pvals_perm_array = torch.zeros_like(perm_cs_m_a)
        
        torch.manual_seed(seed)
        for i in tqdm(range(M), desc="Permutation test"):
            idx = torch.randperm(n_cells, device=device)
            
            c1_perm_a = counts_1.clone()
            c2_perm_a = counts_2[:, idx]
            c1_perm_a[same_gene_mask] = counts_1[same_gene_mask, :][:, idx]

            WX2t_a = torch.sparse.mm(weights, c2_perm_a.T)
            WtX2t_a = torch.sparse.mm(weights.transpose(0, 1), c2_perm_a.T)
            cs_a = (c1_perm_a.T * WX2t_a).sum(0) + (c1_perm_a.T * WtX2t_a).sum(0)
            cs_a[same_gene_mask] = cs_a[same_gene_mask] / 2
            perm_cs_gp_a[:, i] = cs_a

            cs_m_a = compute_metabolite_cs(cs_a, gene_pair_dict, interacting_cell_scores=False)
            perm_cs_m_a[:, i] = cs_m_a

            c2_perm_b = counts_2.clone()
            c1_perm_b = counts_1[:, idx]
            c2_perm_b[same_gene_mask] = counts_2[same_gene_mask, :][:, idx]

            WX2t_b = torch.sparse.mm(weights, c2_perm_b.T)
            WtX2t_b = torch.sparse.mm(weights.transpose(0, 1), c2_perm_b.T)
            cs_b = (c1_perm_b.T * WX2t_b).sum(0) + (c1_perm_b.T * WtX2t_b).sum(0)
            cs_b[same_gene_mask] = cs_b[same_gene_mask] / 2
            perm_cs_gp_b[:, i] = cs_b

            cs_m_b = compute_metabolite_cs(cs_b, gene_pair_dict, interacting_cell_scores=False)
            perm_cs_m_b[:, i] = cs_m_b

            if check_analytic_null:
                Z_gp_perm, Z_m_perm = compute_p_results((cs_a, cs_b), (cs_m_a, cs_m_b), gene_pairs_ind, Wtot2, eg2s_gp, gene_pair_dict)
                gp_zs_perm_array[:, i] = Z_gp_perm
                gp_pvals_perm_array[:, i] = torch.tensor(norm.sf(Z_gp_perm.cpu().numpy()), device=device)
                m_zs_perm_array[:, i] = Z_m_perm
                m_pvals_perm_array[:, i] = torch.tensor(norm.sf(Z_m_perm.cpu().numpy()), device=device)

        adata.uns['ccc_results']['np']['gp']['perm_cs_a'] = perm_cs_gp_a.detach().cpu().numpy()
        adata.uns['ccc_results']['np']['gp']['perm_cs_b'] = perm_cs_gp_b.detach().cpu().numpy()
        adata.uns['ccc_results']['np']['m']['perm_cs_a'] = perm_cs_m_a.detach().cpu().numpy()
        adata.uns['ccc_results']['np']['m']['perm_cs_b'] = perm_cs_m_b.detach().cpu().numpy()
        
        x_gp_a = (perm_cs_gp_a > cs_gp[:, None]).sum(dim=1)
        x_gp_b = (perm_cs_gp_b > cs_gp[:, None]).sum(dim=1)
        x_m_a = (perm_cs_m_a > cs_m[:, None]).sum(dim=1)
        x_m_b = (perm_cs_m_b > cs_m[:, None]).sum(dim=1)

        pvals_gp_a = (x_gp_a + 1).float() / (M + 1)
        pvals_gp_b = (x_gp_b + 1).float() / (M + 1)
        pvals_m_a = (x_m_a + 1).float() / (M + 1)
        pvals_m_b = (x_m_b + 1).float() / (M + 1)

        pvals_gp = torch.where(pvals_gp_a > pvals_gp_b, pvals_gp_a, pvals_gp_b)
        pvals_m = torch.where(pvals_m_a > pvals_m_b, pvals_m_a, pvals_m_b)
        
        adata.uns['ccc_results']['np']['gp']['pval'] = pvals_gp.cpu().numpy()
        adata.uns['ccc_results']['np']['gp']['FDR'] = multipletests(pvals_gp.cpu().numpy(), method="fdr_bh")[1]
        adata.uns['ccc_results']['np']['m']['pval'] = pvals_m.cpu().numpy()
        adata.uns['ccc_results']['np']['m']['FDR'] = multipletests(pvals_m.cpu().numpy(), method="fdr_bh")[1]

        if check_analytic_null:
            adata.uns['ccc_results']['np']['analytic_null'] = {
                'gp_zs_perm': gp_zs_perm_array.detach().cpu().numpy(),
                'gp_pvals_perm': gp_pvals_perm_array.detach().cpu().numpy(),
                'm_zs_perm': m_zs_perm_array.detach().cpu().numpy(),
                'm_pvals_perm': m_pvals_perm_array.detach().cpu().numpy(),
            }
    
    if verbose:
        print("Non-parametric test finished.")
    
    return


# --- DataForClaude/cell_communication.py:3045 ----------------------------------------
def get_cell_communication_results(
    adata, 
    genes,
    layer_key_p_test,
    layer_key_np_test,
    model,
    D,
    test,
    device,
):
    
    gene_pairs = adata.uns['gene_pairs']
    gene_pairs_ind = adata.uns['gene_pairs_ind']
    gene_pair_dict = adata.uns["gene_pair_dict"]

    sample_specific = 'sample_key' in adata.uns
    
    if isinstance(D, np.ndarray):
        D = torch.tensor(D, dtype=torch.float64, device=device)
    
    # Initialize dataframes
    cell_com_df_gp = pd.DataFrame(gene_pairs, columns=["Gene 1", "Gene 2"])
    cell_com_df_m = pd.DataFrame({"Metabolite": list(gene_pair_dict.keys())})

    if test in ["parametric", "both"]:
        suffix = "p"
        # Gene pair
        c_values = adata.uns['ccc_results'][suffix]['gp']['cs']
        z_values = adata.uns['ccc_results'][suffix]['gp']['Z']
        p_values = adata.uns['ccc_results'][suffix]['gp']['Z_pval']
        fdr_values = adata.uns['ccc_results'][suffix]['gp']['Z_FDR']
        cell_com_df_gp[f'C_{suffix}'] = c_values
        cell_com_df_gp['Z'] = z_values
        cell_com_df_gp['Z_pval'] = p_values
        cell_com_df_gp['Z_FDR'] = fdr_values

        counts = counts_from_anndata(adata[:, genes], layer_key_p_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        num_umi = counts.sum(dim=0)
        counts_std = standardize_counts(adata, counts, model, num_umi, sample_specific)
        
        c_values_norm = normalize_values(counts_std, gene_pairs_ind, c_values, D)
        adata.uns['ccc_results'][suffix]['gp']['cs_norm'] = c_values_norm.cpu().numpy()
        cell_com_df_gp[f'C_norm_{suffix}'] = c_values_norm.cpu().numpy()

        # Metabolite
        c_values = adata.uns['ccc_results'][suffix]['m']['cs']
        z_values = adata.uns['ccc_results'][suffix]['m']['Z']
        p_values = adata.uns['ccc_results'][suffix]['m']['Z_pval']
        fdr_values = adata.uns['ccc_results'][suffix]['m']['Z_FDR']
        cell_com_df_m[f'C_{suffix}'] = c_values
        cell_com_df_m['Z'] = z_values
        cell_com_df_m['Z_pval'] = p_values
        cell_com_df_m['Z_FDR'] = fdr_values

    if test in ["non-parametric", "both"]:
        suffix = "np"
        # Gene pair
        c_values = adata.uns['ccc_results'][suffix]['gp']['cs']
        p_values = adata.uns['ccc_results'][suffix]['gp']['pval']
        fdr_values = adata.uns['ccc_results'][suffix]['gp']['FDR']
        cell_com_df_gp[f'C_{suffix}'] = c_values
        cell_com_df_gp[f'pval_{suffix}'] = p_values
        cell_com_df_gp[f'FDR_{suffix}'] = fdr_values

        counts = counts_from_anndata(adata[:, genes], layer_key_np_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        if adata.uns.get('center_counts_for_np_test', False):
            num_umi = counts.sum(dim=0)
            counts = standardize_counts(adata, counts, model, num_umi, sample_specific)
        
        c_values_norm = normalize_values(counts, gene_pairs_ind, c_values, D)
        adata.uns['ccc_results'][suffix]['gp']['cs_norm'] = c_values_norm.cpu().numpy()
        cell_com_df_gp[f'C_norm_{suffix}'] = c_values_norm.cpu().numpy()
        
        # Metabolite
        c_values = adata.uns['ccc_results'][suffix]['m']['cs']
        p_values = adata.uns['ccc_results'][suffix]['m']['pval']
        fdr_values = adata.uns['ccc_results'][suffix]['m']['FDR']
        cell_com_df_m[f'C_{suffix}'] = c_values
        cell_com_df_m[f'pval_{suffix}'] = p_values
        cell_com_df_m[f'FDR_{suffix}'] = fdr_values

    adata.uns['ccc_results']['cell_com_df_gp'] = cell_com_df_gp
    adata.uns['ccc_results']['cell_com_df_m'] = cell_com_df_m
    
    return

# --- DataForClaude/cell_communication.py:3172 ----------------------------------------
def normalize_values(counts, gene_pairs_ind, lcs, D):
    """
    Normalize communication scores (lcs) using maximum possible score estimates.
    """
    lc_maxs = compute_max_cs(D, counts, gene_pairs_ind)
    lc_maxs = torch.where(lc_maxs == 0, torch.tensor(1.0, device=lc_maxs.device), lc_maxs)
    if isinstance(lcs, np.ndarray):
        lcs = torch.tensor(lcs, dtype=lc_maxs.dtype, device=lc_maxs.device)
    c_values_norm = lcs / lc_maxs
    c_values_norm = torch.where(torch.isinf(c_values_norm), torch.tensor(1.0, device=c_values_norm.device), c_values_norm)
    return c_values_norm


# --- DataForClaude/cell_communication.py:3211 ----------------------------------------
def compute_max_cs(node_degrees, counts, gene_pairs_ind):
    """
    Compute max communication scores per gene pair.
    """
    result = torch.empty(len(gene_pairs_ind), dtype=counts.dtype, device=counts.device)
    
    for i, (g1, _) in enumerate(gene_pairs_ind):
        if isinstance(g1, list):
            vals = counts[g1].mean(dim=0)
        else:
            vals = counts[g1]
        result[i] = compute_max_cs_gp(vals, node_degrees)
    
    return result


# --- DataForClaude/cell_communication.py:3237 ----------------------------------------
def compute_max_cs_gp(vals, node_degrees):
    """
    Compute max communication score for a single gene (vector).
    """
    return 0.5 * torch.sum(node_degrees * vals ** 2)



# ============================================================================
# VENDORED VERBATIM from DataForClaude/cell_communication.py -- CU-C (the
# cell-type-AWARE aggregate function + its transitive deps needed by
# `compute_ct_cell_communication_lowmem` in cell_communication_lowmem.py). Same
# provenance/fidelity rule as the CU-A/CU-B blocks above: copied exactly, not
# reimplemented; do not hand-edit the logic below -- if harreman's internals
# change, re-copy from the reference file.
# ============================================================================

# --- DataForClaude/cell_communication.py:879 -----------------------------------------
def compute_ct_cell_communication(
    adata: AnnData,
    layer_key_p_test: Optional[Union[Literal["use_raw"], str]] = None,
    layer_key_np_test: Optional[Union[Literal["use_raw"], str]] = None,
    model: str = None,
    cell_type_key: Optional[str] = None,
    center_counts_for_np_test: Optional[bool] = False,
    subset_gene_pairs: Optional[list] = None,
    subset_metabolites: Optional[list] = None,
    fix_gp: Optional[bool] = False,
    M: Optional[int] = 1000,
    seed: Optional[int] = 42,
    test: Optional[Union[Literal["parametric"], Literal["non-parametric"], Literal["both"]]] = "both",
    mean: Optional[Union[Literal["algebraic"], Literal["geometric"]]] = 'algebraic',
    check_analytic_null: Optional[bool] = False,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose: Optional[bool] = False,
):
    """
    Computes cell type-aware cell-cell communication (CCC) scores by stratifying communication 
    by interacting cell type pairs. Supports parametric and non-parametric statistical inference.

    Parameters
    ----------
    adata : AnnData
        Annotated data object. Required fields include:
            - `uns["gene_pairs"]`: gene pairs involved in communication.
            - `uns["gene_pairs_per_metabolite"]`: maps metabolites to gene pairs.
            - `uns["gene_pairs_per_ct_pair"]`: gene pairs per cell type pair.
            - `obsp["weights"]`: sparse cell-cell proximity matrix.
            - `obs[cell_type_key]`: categorical cell type annotations.
            - `uns["cell_type_pairs"]`: list of interacting cell type pairs.
            - (Optional) `uns["LR_database"]`: for metabolite/pathway annotation.
    layer_key_p_test : str or "use_raw", optional
        Data layer to use for parametric test.
    layer_key_np_test : str or "use_raw", optional
        Data layer to use for non-parametric test.
    model : str, optional
        Normalization model to use for centering gene expression. Options include "none", "normal", "bernoulli", or "danb".
    cell_type_key : str, optional
        Key in `adata.obs` corresponding to cell type annotations. Required if not stored in `uns`.
    center_counts_for_np_test : bool, optional (default: False)
        Whether to center expression counts using the specified model before non-parametric testing.
    subset_gene_pairs : list, optional
        Subset of gene pairs to consider. If None, uses all pairs.
    subset_metabolites : list, optional
        Subset of metabolites to include in the analysis.
    fix_gp : bool, optional (default: False)
        If True, keeps gene pair identity fixed during permutation testing, randomizing cell types only.
    M : int, optional (default: 1000)
        Number of permutations to use if `permutation_test` is True.
    seed : int, optional (default: 42)
        Random seed for permutation reproducibility.
    test : {'parametric', 'non-parametric', 'both'}, optional (default: 'both')
        Specifies which statistical test(s) to run.
    mean : {'algebraic', 'geometric'}, optional (default: 'algebraic')
        Averaging method for multi-gene modules.
    check_analytic_null : bool, optional (default: False)
        Whether to compute Z-scores and p-values under the null distribution for the permutation test.
    device : torch.device, optional
        PyTorch device to run computations on. Defaults to CUDA if available.
    verbose : bool, optional (default: False)
        Whether to print progress and status messages.

    Returns
    -------
    None
        Results are stored in the following `adata.uns` fields:
            - `ct_ccc_results["p"]`: parametric test results (scores, Z, p-values, FDRs) per gene pair and metabolite per cell type pair.
            - `ct_ccc_results["np"]`: non-parametric test results (communication scores, empirical p-values, FDRs).
            - `gene_pair_dict`: dictionary mapping metabolites to relevant gene pairs.
            - `gene_pairs_ind`, `gene_pairs_ind_per_ct_pair`: index-referenced gene pair representations.
            - `D`: spatial node degree for each cell per cell type pair.
            - `cells`, `genes`: ordered list of cells and genes used in analysis.
            - (optional) `ct_ccc_results["np"]["analytic_null"]`: null distributions from permutation test Z-scores and p-values.
    """
    
    start = time.time()
    if verbose:
        print("Starting cell type-aware cell-cell communication analysis...")

    adata.uns['ct_ccc_results'] = {}

    if test not in ['both', 'parametric', 'non-parametric']:
        raise ValueError('The "test" variable should be one of ["both", "parametric", "non-parametric"].')
    
    if mean not in ['algebraic', 'geometric']:
        raise ValueError('The "mean" variable should be one of ["algebraic", "geometric"].')
    
    if 'cell_type_key' in adata.uns and cell_type_key is None:
        cell_type_key = adata.uns['cell_type_key']
    elif 'cell_type_key' not in adata.uns and cell_type_key is None:
        raise ValueError('Please provide the "cell_type_key" argument.')
    
    adata.uns['layer_key_p_test'] = layer_key_p_test
    adata.uns['layer_key_np_test'] = layer_key_np_test
    adata.uns['model'] = model
    adata.uns['cell_type_key'] = cell_type_key
    adata.uns['center_counts_for_np_test'] = center_counts_for_np_test
    adata.uns['mean'] = mean

    run_ct_cell_communication_analysis(adata, layer_key_p_test, layer_key_np_test, model, cell_type_key, center_counts_for_np_test, subset_gene_pairs, subset_metabolites, fix_gp, M, seed, test, mean, check_analytic_null, device, verbose)

    if verbose:
        print("Obtaining cell type-aware communication results...")
    get_ct_cell_communication_results(
        adata,
        adata.uns["genes"],
        adata.uns["cells"],
        layer_key_p_test,
        layer_key_np_test,
        model,
        adata.obs[cell_type_key],
        adata.uns["cell_type_pairs"],
        adata.uns["D"],
        test,
        device,
    )

    if verbose:
        print("Finished computing cell type-aware cell-cell communication analysis in %.3f seconds" %(time.time()-start))

    return


# --- DataForClaude/cell_communication.py:1004 ----------------------------------------
def run_ct_cell_communication_analysis(
    adata,
    layer_key_p_test,
    layer_key_np_test,
    model,
    cell_type_key,
    center_counts_for_np_test,
    subset_gene_pairs,
    subset_metabolites,
    fix_gp,
    M,
    seed,
    test,
    mean,
    check_analytic_null,
    device,
    verbose,
):
    
    use_raw = (layer_key_p_test == "use_raw") & (layer_key_np_test == "use_raw")
    obs = adata.raw.obs if use_raw else adata.obs
    cells = adata.raw.obs.index.values.astype(str) if use_raw else adata.obs_names.values.astype(str)
    
    sample_specific = 'sample_key' in adata.uns
    
    fix_ct = True if adata.uns['fix_ct'] else False
    
    gene_pairs = adata.uns["gene_pairs"] if subset_gene_pairs is None else subset_gene_pairs
    genes = list(np.unique(list(flatten(adata.uns["gene_pairs"]))))
    adata.uns["genes"] = genes
        
    cell_types = obs[cell_type_key]
    cell_type_pairs = adata.uns.get("cell_type_pairs")
    gene_pairs_per_ct_pair = adata.uns.get("gene_pairs_per_ct_pair", {})
    
    weights = adata.obsp["weights"]
    
    used_ct_pairs = list(set(ct for cell_type_pair in cell_type_pairs for ct in cell_type_pair))
    all_cell_types = set(cell_types.unique())
    used_ct_pairs_set = set(used_ct_pairs)
    if used_ct_pairs_set < all_cell_types:
        keep_mask = cell_types[cells].isin(used_ct_pairs).values
        keep_indices = np.where(keep_mask)[0]
        weights = weights[keep_indices][:, keep_indices]
        cells = cells[keep_indices]
        cell_types = cell_types.loc[cells]
        
    adata.uns["cells"] = cells
    
    weights_ct_pairs = create_weights_ct_pairs(weights.tocoo(), cell_types, cell_type_pairs, device)
    
    row_degrees = torch.sparse.sum(weights_ct_pairs, dim=2).to_dense()
    col_degrees = torch.sparse.sum(weights_ct_pairs, dim=1).to_dense()
    D = row_degrees + col_degrees
    if used_ct_pairs_set < all_cell_types:
        D_full = torch.zeros((len(cell_type_pairs), adata.shape[0]), device=weights_ct_pairs.device, dtype=weights_ct_pairs.dtype)
        D_full[:, keep_indices] = D
        adata.uns["D"] = D_full.cpu().numpy()
    else:
        adata.uns["D"] = D.cpu().numpy()
        
    # Map gene_pairs to index
    gene_pairs_ind = []
    for pair in gene_pairs:
        idx1 = [genes.index(g) for g in pair[0]] if isinstance(pair[0], list) else genes.index(pair[0])
        idx2 = [genes.index(g) for g in pair[1]] if isinstance(pair[1], list) else genes.index(pair[1])
        gene_pairs_ind.append((idx1, idx2))
    adata.uns["gene_pairs_ind"] = gene_pairs_ind

    # Cell-type pair-specific indices
    gene_pairs_ind_per_ct_pair = defaultdict(list)
    gene_pairs_per_ct_pair_ind = defaultdict(list)
    for ct_pair, gpairs in gene_pairs_per_ct_pair.items():
        for pair in gpairs:
            if pair not in gene_pairs:
                continue
            idx = gene_pairs.index(pair)
            gene_pairs_ind_per_ct_pair[ct_pair].append(gene_pairs_ind[idx])
            gene_pairs_per_ct_pair_ind[ct_pair].append(idx)

    adata.uns["gene_pairs_ind_per_ct_pair"] = dict(gene_pairs_ind_per_ct_pair)
    adata.uns["gene_pairs_per_ct_pair_ind"] = dict(gene_pairs_per_ct_pair_ind)
    
    def make_hashable(pair):
        return tuple(tuple(x) if isinstance(x, list) else x for x in pair)

    gene_pairs_ind_set = {make_hashable(pair) for pair in gene_pairs_ind}
    ct_specific_gene_pairs = [
        i for i, pairs in enumerate(gene_pairs_ind_per_ct_pair.values())
        if {make_hashable(pair) for pair in pairs} < gene_pairs_ind_set
    ]
    
    # Metabolite-gene pair preparation
    gp_metab = adata.uns['gene_pairs_per_metabolite']
    metabolite_gene_pair_df = (
        pd.DataFrame.from_dict(gp_metab, orient="index")
        .rename_axis("metabolite")
        .explode(["gene_pair", "gene_type"])
        .reset_index()
    )

    if 'LR_database' in adata.uns:
        merged = metabolite_gene_pair_df.merge(
            adata.uns['LR_database'], left_on='metabolite', right_on='interaction_name', how='left'
        )
        LR_df = merged.dropna(subset=['pathway_name'])
        metabolite_gene_pair_df.loc[
            metabolite_gene_pair_df.metabolite.isin(LR_df.metabolite), 'metabolite'] = LR_df['pathway_name'].values

    if subset_metabolites:
        metabolite_gene_pair_df = metabolite_gene_pair_df[metabolite_gene_pair_df.metabolite.isin(subset_metabolites)]
    
    gene_pair_dict = {}
    for metabolite, group in metabolite_gene_pair_df.groupby("metabolite"):
        idxs = group["gene_pair"].apply(lambda gp: gene_pairs.index(gp) if gp in gene_pairs else None).dropna().tolist()
        idxs = [int(ind) for ind in idxs if ind is not None]
        if idxs:
            gene_pair_dict[metabolite] = idxs
    adata.uns['gene_pair_dict'] = gene_pair_dict

    if test in ['parametric', 'both']:

        if verbose:
            print("Running the parametric test...")
        
        adata.uns['ct_ccc_results']['p'] = {
            'gp': {},
            'm': {}
        }
        
        weights_sq_data = weights_ct_pairs.values() ** 2
        weights_sq = torch.sparse_coo_tensor(
            weights_ct_pairs.indices(),
            weights_sq_data,
            weights_ct_pairs.shape,
            device=weights_ct_pairs.device
        )
        Wtot2 = torch.sparse.sum(weights_sq, dim=(1, 2)).to_dense()

        # Load counts
        counts = counts_from_anndata(adata[cells, genes], layer_key_p_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        num_umi = counts.sum(dim=0)
        
        # Prepare counts_1 and counts_2
        counts_1 = []
        counts_2 = []
        for (idx1, idx2) in gene_pairs_ind:
            if isinstance(idx1, list):
                c1 = counts[idx1, :].mean(dim=0) if mean == 'algebraic' else torch.exp(torch.log(counts[idx1, :] + 1e-8).mean(dim=0))
            else:
                c1 = counts[idx1, :]
            if isinstance(idx2, list):
                c2 = counts[idx2, :].mean(dim=0) if mean == 'algebraic' else torch.exp(torch.log(counts[idx2, :] + 1e-8).mean(dim=0))
            else:
                c2 = counts[idx2, :]
            counts_1.append(c1)
            counts_2.append(c2)

        counts_1 = torch.stack(counts_1)
        counts_2 = torch.stack(counts_2)
        
        counts_1 = standardize_ct_counts(adata, counts_1, model, num_umi, sample_specific, cell_types)
        counts_2 = standardize_ct_counts(adata, counts_2, model, num_umi, sample_specific, cell_types)
        
        # Compute CCC scores        
        cs_gp = torch.zeros((len(cell_type_pairs), counts_1.shape[0]), device=counts_1.device)
        for ct_pair in range(len(cell_type_pairs)):
            W = weights_ct_pairs[ct_pair].coalesce()
            WX2t = torch.sparse.mm(W, counts_2.T)
            cs_gp[ct_pair] = (counts_1.T * WX2t).sum(0)
        adata.uns['ct_ccc_results']['p']['gp']['cs'] = cs_gp.detach().cpu().numpy()
        
        cs_m = compute_metabolite_cs_ct(cs_gp, cell_type_key, gene_pair_dict, gene_pairs_per_ct_pair_ind, ct_specific_gene_pairs, interacting_cell_scores=False)
        adata.uns['ct_ccc_results']['p']['m']['cs'] = cs_m.detach().cpu().numpy()
        
        EG2_gp = torch.zeros_like(cs_gp) if fix_ct or fix_gp else Wtot2
        if fix_ct:
            for ct_pair in range(len(cell_type_pairs)):
                W = weights_ct_pairs[ct_pair].coalesce()
                W_sq_data = W.values() ** 2
                W_sq = torch.sparse_coo_tensor(W.indices(), W_sq_data, W.shape, device=W.device)
                X1_sq = counts_1 ** 2
                EG2_gp[ct_pair] = torch.sparse.mm(W_sq, X1_sq.T).sum(0)
        elif fix_gp:
            for ct_pair in range(len(cell_type_pairs)):
                W = weights_ct_pairs[ct_pair].coalesce()
                W_sq_data = W.values() ** 2
                W_sq = torch.sparse_coo_tensor(W.indices(), W_sq_data, W.shape, device=W.device)
                X1_sq = counts_1 ** 2
                X2_sq = counts_2 ** 2
                EG2_gp[ct_pair] = (X1_sq.T * torch.sparse.mm(W_sq, X2_sq.T)).sum(0)

        Z_gp, Z_m = compute_ct_p_results(cs_gp, cs_m, gene_pairs_per_ct_pair_ind, ct_specific_gene_pairs, EG2_gp, cell_type_key, gene_pair_dict)

        # Convert tensors to numpy for statsmodels and pandas
        Z_gp_np = Z_gp.detach().cpu().numpy()
        Z_m_np = Z_m.detach().cpu().numpy()
        # Compute p-values and FDRs
        Z_pvals_gp = norm.sf(Z_gp_np)
        Z_pvals_m = norm.sf(Z_m_np)
        FDR_gp = multipletests(Z_pvals_gp.flatten(), method="fdr_bh")[1].reshape(Z_pvals_gp.shape)
        FDR_m = multipletests(Z_pvals_m.flatten(), method="fdr_bh")[1].reshape(Z_pvals_m.shape)
        
        # Store in AnnData
        adata.uns['ct_ccc_results']['p']['gp']['Z'] = Z_gp_np
        adata.uns['ct_ccc_results']['p']['gp']['Z_pval'] = Z_pvals_gp
        adata.uns['ct_ccc_results']['p']['gp']['Z_FDR'] = FDR_gp
        adata.uns['ct_ccc_results']['p']['m']['Z'] = Z_m_np
        adata.uns['ct_ccc_results']['p']['m']['Z_pval'] = Z_pvals_m
        adata.uns['ct_ccc_results']['p']['m']['Z_FDR'] = FDR_m

    if test in ["non-parametric", "both"]:

        if verbose:
            print("Running the non-parametric test...")
        
        adata.uns['ct_ccc_results']['np'] = {
            'gp': {},
            'm': {}
        }

        # Load counts
        counts = counts_from_anndata(adata[cells, genes], layer_key_np_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)

        # Prepare counts_1 and counts_2
        counts_1 = []
        counts_2 = []
        for (idx1, idx2) in gene_pairs_ind:
            if isinstance(idx1, list):
                c1 = counts[idx1, :].mean(dim=0) if mean == 'algebraic' else torch.exp(torch.log(counts[idx1, :] + 1e-8).mean(dim=0))
            else:
                c1 = counts[idx1, :]
            if isinstance(idx2, list):
                c2 = counts[idx2, :].mean(dim=0) if mean == 'algebraic' else torch.exp(torch.log(counts[idx2, :] + 1e-8).mean(dim=0))
            else:
                c2 = counts[idx2, :]
            counts_1.append(c1)
            counts_2.append(c2)

        counts_1 = torch.stack(counts_1)
        counts_2 = torch.stack(counts_2)
        
        if center_counts_for_np_test:
            num_umi = counts.sum(dim=0)
            counts_1 = standardize_ct_counts(adata, counts_1, model, num_umi, sample_specific, cell_types)
            counts_2 = standardize_ct_counts(adata, counts_2, model, num_umi, sample_specific, cell_types)
        
        if center_counts_for_np_test and test == 'both':
            adata.uns['ct_ccc_results']['np']['gp']['cs'] = np.array(adata.uns['ct_ccc_results']['p']['gp']['cs'])
            adata.uns['ct_ccc_results']['np']['m']['cs'] = np.array(adata.uns['ct_ccc_results']['p']['m']['cs'])
        else:            
            cs_gp = torch.zeros((len(cell_type_pairs), counts_1.shape[0]), device=counts_1.device)
            for ct_pair in range(len(cell_type_pairs)):
                W = weights_ct_pairs[ct_pair].coalesce()
                WX2t = torch.sparse.mm(W, counts_2.T)
                cs_gp[ct_pair] = (counts_1.T * WX2t).sum(0)
            adata.uns['ct_ccc_results']['np']['gp']['cs'] = cs_gp.detach().cpu().numpy()
            cs_m = compute_metabolite_cs_ct(cs_gp, cell_type_key, gene_pair_dict, gene_pairs_per_ct_pair_ind, ct_specific_gene_pairs, interacting_cell_scores=False)
            adata.uns['ct_ccc_results']['np']['m']['cs'] = cs_m.detach().cpu().numpy()
        
        perm_cs_gp = torch.zeros((len(cell_type_pairs), counts_1.shape[0], M), dtype=torch.float64, device=device)
        perm_cs_m = torch.zeros((len(cell_type_pairs), len(gene_pair_dict), M), dtype=torch.float64, device=device)
        
        if check_analytic_null:
            gp_zs_perm_array = torch.zeros_like(perm_cs_gp)
            gp_pvals_perm_array = torch.zeros_like(perm_cs_gp)
            m_zs_perm_array = torch.zeros_like(perm_cs_m)
            m_pvals_perm_array = torch.zeros_like(perm_cs_m)
        
        if fix_gp:
            c1_perm = counts_1
            c2_perm = counts_2
        
        torch.manual_seed(seed)
        for i in tqdm(range(M), desc="Permutation test"):
            
            if fix_gp:
                indices = torch.randperm(len(cell_types)).numpy()
                shuffled_cell_types = cell_types.iloc[indices].reset_index(drop=True)
                weights_ct_pairs = create_weights_ct_pairs(weights.tocoo(), shuffled_cell_types, cell_type_pairs, device)
            else:
                cell_type_labels = torch.tensor(cell_types.astype('category').cat.codes.values, device=counts_1.device)
                idx = torch.empty_like(cell_type_labels, dtype=torch.int64)

                for ct in torch.unique(cell_type_labels):
                    ct_mask = cell_type_labels == ct
                    ct_indices = torch.nonzero(ct_mask, as_tuple=True)[0]
                    permuted_indices = ct_indices[torch.randperm(len(ct_indices))]
                    idx[ct_indices] = permuted_indices

                c1_perm = counts_1 if fix_ct else counts_1[:, idx.long()]
                c2_perm = counts_2[:, idx.long()]
            
            cs_gp = torch.zeros((len(cell_type_pairs), c1_perm.shape[0]), device=c1_perm.device)
            for ct_pair in range(len(cell_type_pairs)):
                W = weights_ct_pairs[ct_pair].coalesce()
                WX2t = torch.sparse.mm(W, c2_perm.T)
                cs_gp[ct_pair] = (c1_perm.T * WX2t).sum(0)
            perm_cs_gp[:, :, i] = cs_gp
            
            cs_m = compute_metabolite_cs_ct(cs_gp, cell_type_key, gene_pair_dict, gene_pairs_per_ct_pair_ind, ct_specific_gene_pairs, interacting_cell_scores=False)
            perm_cs_m[:, :, i] = cs_m
            
            if check_analytic_null:
                Z_gp_perm, Z_m_perm = compute_ct_p_results(cs_gp, cs_m, gene_pairs_per_ct_pair_ind, ct_specific_gene_pairs, EG2_gp, cell_type_key, gene_pair_dict)
                gp_zs_perm_array[:, :, i] = Z_gp_perm
                gp_pvals_perm_array[:, :, i] = torch.tensor(norm.sf(Z_gp_perm.cpu().numpy()), device=device)
                m_zs_perm_array[:, :, i] = Z_m_perm
                m_pvals_perm_array[:, :, i] = torch.tensor(norm.sf(Z_m_perm.cpu().numpy()), device=device)
            
        adata.uns['ct_ccc_results']['np']['gp']['perm_cs'] = perm_cs_gp.detach().cpu().numpy()
        adata.uns['ct_ccc_results']['np']['m']['perm_cs'] = perm_cs_m.detach().cpu().numpy()
        
        x_gp = np.sum(adata.uns['ct_ccc_results']['np']['gp']['perm_cs'] > adata.uns['ct_ccc_results']['np']['gp']['cs'][:, :, np.newaxis], axis=2)
        x_m = np.sum(adata.uns['ct_ccc_results']['np']['m']['perm_cs'] > adata.uns['ct_ccc_results']['np']['m']['cs'][:, :, np.newaxis], axis=2)
        
        pvals_gp = (x_gp + 1) / (M + 1)
        pvals_m = (x_m + 1) / (M + 1)
        
        adata.uns['ct_ccc_results']['np']['gp']['pval'] = pvals_gp
        adata.uns['ct_ccc_results']['np']['gp']['FDR'] = multipletests(pvals_gp.flatten(), method="fdr_bh")[1].reshape(pvals_gp.shape)
        adata.uns['ct_ccc_results']['np']['m']['pval'] = pvals_m
        adata.uns['ct_ccc_results']['np']['m']['FDR'] = multipletests(pvals_m.flatten(), method="fdr_bh")[1].reshape(pvals_m.shape)
        
        if check_analytic_null:
            adata.uns['ct_ccc_results']['np']['analytic_null'] = {
                'gp_zs_perm': gp_zs_perm_array.detach().cpu().numpy(),
                'gp_pvals_perm': gp_pvals_perm_array.detach().cpu().numpy(),
                'm_zs_perm': m_zs_perm_array.detach().cpu().numpy(),
                'm_pvals_perm': m_pvals_perm_array.detach().cpu().numpy(),
            }

    adata.uns["cell_types"] = cell_types.tolist() if cell_type_key else None
    
    if verbose:
        print("Non-parametric test finished.")
    
    return


# --- DataForClaude/cell_communication.py:1346 ----------------------------------------
def standardize_ct_counts(adata, counts, model, num_umi, sample_specific, cell_types):
    
    if sample_specific:
        sample_key = adata.uns['sample_key']
        for sample in adata.obs[sample_key].unique():
            subset = np.where(adata.obs[sample_key] == sample)[0]
            counts[:, subset] = center_ct_counts_torch(counts[:, subset], num_umi[subset], model, cell_types[subset])
    else:
        counts = center_ct_counts_torch(counts, num_umi, model, cell_types)
            
    return counts


# --- DataForClaude/cell_communication.py:1367 ----------------------------------------
def create_weights_ct_pairs(weights, cell_types, cell_type_pairs, device):
    
    indices = torch.tensor([weights.row, weights.col], dtype=torch.long, device=device)
    values = torch.tensor(weights.data, dtype=torch.float64, device=device)
    shape = weights.shape

    cell_type_cats = cell_types.astype("category")
    cell_type_codes = torch.tensor(cell_type_cats.cat.codes.values, dtype=torch.long, device=device)
    ct_name_to_code = {name: code for code, name in enumerate(cell_type_cats.cat.categories)}

    row_idx, col_idx = indices
    sender_types = cell_type_codes[row_idx]
    receiver_types = cell_type_codes[col_idx]
    
    weights_list = []
    coord_list = []

    for i, (ct1, ct2) in enumerate(cell_type_pairs):
        code1 = ct_name_to_code[ct1]
        code2 = ct_name_to_code[ct2]

        pair_mask = (sender_types == code1) & (receiver_types == code2)
        if pair_mask.sum() == 0:
            continue

        pair_values = values[pair_mask]
        pair_coords = torch.stack([
            torch.full((pair_values.shape[0],), i, dtype=torch.long, device=device),
            row_idx[pair_mask],
            col_idx[pair_mask]
        ], dim=0)

        weights_list.append(pair_values)
        coord_list.append(pair_coords)

    all_values = torch.cat(weights_list)
    all_coords = torch.cat(coord_list, dim=1)
    weights_ct_pairs = torch.sparse_coo_tensor(
        all_coords, all_values, (len(cell_type_pairs), shape[0], shape[1]), device=device
    )
    weights_ct_pairs = weights_ct_pairs.coalesce()
    
    return weights_ct_pairs


# --- DataForClaude/cell_communication.py:2321 ----------------------------------------
def compute_metabolite_cs_ct(cs_gp, cell_type_key, gene_pair_dict, gene_pairs_per_ct_pair_ind=None, ct_specific_gene_pairs=None, interacting_cell_scores=False):
    if cell_type_key and ct_specific_gene_pairs:
        for i, ct_pair in enumerate(gene_pairs_per_ct_pair_ind.keys()):
            if i not in ct_specific_gene_pairs:
                continue
            mask_dim = 2 if interacting_cell_scores else 1
            mask = np.ones(cs_gp.shape[mask_dim], dtype=bool)
            mask[gene_pairs_per_ct_pair_ind[ct_pair]] = False
            if interacting_cell_scores:
                cs_gp[i, :, mask] = 0
            else:
                cs_gp[i, mask] = 0
    
    device = cs_gp.device
    scores = []

    for indices in gene_pair_dict.values():
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        if interacting_cell_scores:
            summed = cs_gp[:, :, idx_tensor].sum(dim=2)  # shape: (cells,)
        else:
            summed = cs_gp[:, idx_tensor].sum(dim=1)  # scalar
        scores.append(summed)

    if interacting_cell_scores:
        cs_m = torch.stack(scores, dim=2)  # shape: (cells, metabolites)
    else:
        cs_m = torch.stack(scores, dim=1) 
    
    return cs_m


# --- DataForClaude/cell_communication.py:2517 ----------------------------------------
def compute_ct_p_results(C_gp, C_m, gene_pairs_per_ct_pair_ind, ct_specific_gene_pairs, EG2_gp, cell_type_key, gene_pair_dict):

    EG2_gp = EG2_gp.unsqueeze(1).expand(-1, C_gp.shape[1]) if len(EG2_gp.shape) == 1 else EG2_gp
    
    stdG = torch.sqrt(EG2_gp)
    stdG[stdG == 0] = 1
    
    Z_gp = C_gp / stdG

    EG2_m = compute_metabolite_cs_ct(EG2_gp, cell_type_key, gene_pair_dict, gene_pairs_per_ct_pair_ind, ct_specific_gene_pairs, interacting_cell_scores=False)
    if not isinstance(EG2_m, torch.Tensor):
        device = EG2_gp.device
        EG2_m = torch.tensor(EG2_m, device=device, dtype=torch.float64)

    stdG_m = torch.sqrt(EG2_m)
    stdG_m[stdG_m == 0] = 1

    Z_m = C_m / stdG_m

    return Z_gp, Z_m


# --- DataForClaude/cell_communication.py:2815 ----------------------------------------
def get_ct_cell_communication_results(
    adata, 
    genes,
    cells,
    layer_key_p_test,
    layer_key_np_test,
    model, 
    cell_types, 
    cell_type_pairs,
    D,
    test,
    device,
):
    
    gene_pairs_ind_per_ct_pair = adata.uns['gene_pairs_ind_per_ct_pair']
    gene_pair_dict = adata.uns["gene_pair_dict"]
    genes = adata.uns["genes"]
    
    sample_specific = 'sample_key' in adata.uns
    
    if isinstance(D, np.ndarray):
        D = torch.tensor(D, dtype=torch.float64, device=device)
    
    def idx_to_gene(idx):
        return [genes[i] for i in idx] if isinstance(idx, list) else genes[idx]

    records = [
        {
            "Cell Type 1": ct1,
            "Cell Type 2": ct2,
            "Gene 1": idx_to_gene(gp[0]),
            "Gene 2": idx_to_gene(gp[1]),
        }
        for (ct1, ct2), gp_list in gene_pairs_ind_per_ct_pair.items()
        for gp in gp_list
    ]
    cell_com_df_gp = pd.DataFrame.from_records(records)

    # Generate metabolite interaction table
    ct_pairs = list(gene_pairs_ind_per_ct_pair.keys())
    metabolites = list(gene_pair_dict.keys())
    cell_com_df_m = pd.DataFrame([
        {"Cell Type 1": ct1, "Cell Type 2": ct2, "metabolite": m}
        for (ct1, ct2), m in itertools.product(ct_pairs, metabolites)
    ])

    if test in ["parametric", "both"]:
        suffix = "p"
        # Gene pair
        c_values = adata.uns['ct_ccc_results'][suffix]['gp']['cs']
        z_values = adata.uns['ct_ccc_results'][suffix]['gp']['Z']
        p_values = adata.uns['ct_ccc_results'][suffix]['gp']['Z_pval']
        fdr_values = adata.uns['ct_ccc_results'][suffix]['gp']['Z_FDR']
        cell_com_df_gp[f'C_{suffix}'] = c_values.flatten()
        cell_com_df_gp['Z'] = z_values.flatten()
        cell_com_df_gp['Z_pval'] = p_values.flatten()
        cell_com_df_gp['Z_FDR'] = fdr_values.flatten()
        
        counts = counts_from_anndata(adata[:, genes], layer_key_p_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        num_umi = counts.sum(dim=0)
        counts_std = standardize_ct_counts(adata, counts, model, num_umi, sample_specific, cell_types)

        c_values_norm = normalize_ct_values(counts_std, cell_types, cell_type_pairs, gene_pairs_ind_per_ct_pair, c_values, D)
        adata.uns['ct_ccc_results'][suffix]['gp']['cs_norm'] = c_values_norm.cpu().numpy()
        cell_com_df_gp[f'C_norm_{suffix}'] = c_values_norm.cpu().numpy().flatten()
        
        # Metabolite
        c_values = adata.uns['ct_ccc_results'][suffix]['m']['cs']
        z_values = adata.uns['ct_ccc_results'][suffix]['m']['Z']
        p_values = adata.uns['ct_ccc_results'][suffix]['m']['Z_pval']
        fdr_values = adata.uns['ct_ccc_results'][suffix]['m']['Z_FDR']
        cell_com_df_m[f'C_{suffix}'] = c_values.flatten()
        cell_com_df_m['Z'] = z_values.flatten()
        cell_com_df_m['Z_pval'] = p_values.flatten()
        cell_com_df_m['Z_FDR'] = fdr_values.flatten()

    if test in ["non-parametric", "both"]:
        suffix = "np"
        # Gene pair
        c_values = adata.uns['ct_ccc_results'][suffix]['gp']['cs']
        p_values = adata.uns['ct_ccc_results'][suffix]['gp']['pval']
        fdr_values = adata.uns['ct_ccc_results'][suffix]['gp']['FDR']
        cell_com_df_gp[f'C_{suffix}'] = c_values.flatten()
        cell_com_df_gp[f'pval_{suffix}'] = p_values.flatten()
        cell_com_df_gp[f'FDR_{suffix}'] = fdr_values.flatten()
        
        counts = counts_from_anndata(adata[:, genes], layer_key_np_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        if adata.uns.get('center_counts_for_np_test', False):
            num_umi = counts.sum(dim=0)
            counts = standardize_ct_counts(adata, counts, model, num_umi, sample_specific, cell_types)
        
        c_values_norm = normalize_ct_values(counts, cell_types, cell_type_pairs, gene_pairs_ind_per_ct_pair, c_values, D)
        adata.uns['ct_ccc_results'][suffix]['gp']['cs_norm'] = c_values_norm.cpu().numpy()
        cell_com_df_gp[f'C_norm_{suffix}'] = c_values_norm.cpu().numpy().flatten()

        # Metabolite
        c_values = adata.uns['ct_ccc_results'][suffix]['m']['cs']
        p_values = adata.uns['ct_ccc_results'][suffix]['m']['pval']
        fdr_values = adata.uns['ct_ccc_results'][suffix]['m']['FDR']
        cell_com_df_m[f'C_{suffix}'] = c_values.flatten()
        cell_com_df_m[f'pval_{suffix}'] = p_values.flatten()
        cell_com_df_m[f'FDR_{suffix}'] = fdr_values.flatten()

    adata.uns['ct_ccc_results']['cell_com_df_gp'] = cell_com_df_gp
    adata.uns['ct_ccc_results']['cell_com_df_m'] = cell_com_df_m
    
    return


# --- DataForClaude/cell_communication.py:3134 ----------------------------------------
def normalize_ct_values(
    counts,
    cell_types,
    cell_type_pairs,
    gene_pairs_per_ct_pair_ind,
    lcs,
    D,
):
    
    if isinstance(cell_types, pd.Series):
        cell_types = cell_types.values
    
    if isinstance(lcs, np.ndarray):
        lcs = torch.tensor(lcs, dtype=counts.dtype, device=counts.device)

    c_values_norm = torch.empty_like(lcs, dtype=counts.dtype, device=counts.device)

    for i, ct_pair in enumerate(cell_type_pairs):
        ct_t, _ = ct_pair

        ct_mask = (cell_types == ct_t)
        if isinstance(ct_mask, np.ndarray):
            ct_mask = torch.tensor(ct_mask, device=counts.device)

        counts_ct = counts[:, ct_mask]
        D_ct = D[i][ct_mask]
        gene_pairs_ind = gene_pairs_per_ct_pair_ind[ct_pair]

        lc_maxs = compute_max_cs(D_ct, counts_ct, gene_pairs_ind)
        lc_maxs = torch.where(lc_maxs == 0, torch.tensor(1.0, device=counts.device), lc_maxs)

        c_values = lcs[i] if lcs.ndim == 2 else lcs[i:i+1]  # allow 1D or 2D lcs
        c_values_norm[i] = c_values / lc_maxs
        c_values_norm[i] = torch.where(torch.isinf(c_values_norm[i]), torch.tensor(1.0, device=counts.device), c_values_norm[i])
    
    return c_values_norm


# --- DataForClaude/cell_communication.py:3254 ----------------------------------------
def center_ct_counts_torch(counts, num_umi, model, cell_types):
    """
    counts: Tensor [genes, cells]
    num_umi: Tensor [cells]
    model: 'bernoulli', 'danb', 'normal', or 'none'
    
    Returns:
        Centered counts within cell types: Tensor [genes, cells]
    """
    # Binarize if using Bernoulli
    if model == 'bernoulli':
        counts = (counts > 0).double()
        mu, var, _ = models.apply_model_per_cell_type(
            models.bernoulli_model_torch, counts, num_umi, cell_types
        )
    elif model == 'danb':
        mu, var, _ = models.apply_model_per_cell_type(
            models.danb_model_torch, counts, num_umi, cell_types
        )
    elif model == 'normal':
        mu, var, _ = models.apply_model_per_cell_type(
            models.normal_model_torch, counts, num_umi, cell_types
        )
    elif model == 'none':
        mu, var, _ = models.apply_model_per_cell_type(
            models.none_model_torch, counts, num_umi, cell_types
        )
    else:
        raise ValueError(f"Unsupported model type: {model}")
    
    # Avoid division by zero
    std = torch.sqrt(var)
    std[std == 0] = 1.0

    centered = (counts - mu) / std
    centered[centered == 0] = 0  # Optional: to match old behavior
    
    return centered

