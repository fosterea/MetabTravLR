"""Build the LinearRegression factor store WITHOUT importing `SpaceTravLR` (the heavy
package: torch, group_lasso, numba, celloracle_tmp, ...) at all, directly or
transitively. Only numpy / pandas / scipy / scikit-learn (all already used elsewhere in
this repo) -- no new dependencies.

This module produces EXACTLY the same store format `build_x.get_gene_factors` /
`least_squares.fit_gene_betas` already consume (`obsm['x_factors']` +
`uns['x_factors_cols']` + `uns['x_factor_map']` + `uns['x_genes']`, plus
`obsm['x_metab']` + `uns['x_metab_modulators']`), so every downstream function in
`build_x.py`/`least_squares.py` works unchanged on a store built here.

Why this exists: `build_x.build_factor_block` builds each gene's modulators by
instantiating the REAL `SpatialCellularProgramsEstimator` (`src/SpaceTravLR/models/
parallel_estimators.py`), which imports `torch`/`group_lasso`/`numba`/CellOracle at
module level. On a CPU-only box (or any environment where that heavy stack isn't
installed/desired), that import is a hard blocker. This module re-derives the same
*values* from first principles -- CellChat L-R + a direct read of the base GRN +
NicheNet L-TF -- using small helpers LIFTED (copied, with their source cited) from the
real package, so the two stores agree without ever importing it.

=== Decisions recorded here (see DataForClaude/documentation/04_decisions_and_state.md,
2026-10-18 entry) ===
  - No regression pruning: every base-GRN candidate TF survives (the real pipeline would
    normally prune TFs via a per-cluster Bayesian-ridge fit over `celloracle_links.pkl`;
    we read the prior base GRN directly instead, since fitting that regression requires
    the heavy stack we're deliberately avoiding here).
  - Feedback self-exclusion is simplified to DIRECT self-exclusion only (TF==target,
    lig==target, rec==target, tf==target for L-TF) -- CellOracle's regression-based
    "does the ligand's own regulator set include the target/TF" feedback check
    (`parallel_estimators.py` lines 576-581) is dropped.
  - We do NOT claim bit-for-bit parity with the real estimator's numbers (different GRN
    source, no feedback exclusion) -- only the same STORE FORMAT and the same general
    L-R/L-TF/metabolite math.

=== Public API ===
  - `build_linear_factor_store(adata, focus_genes, *, annot_col, ...)` -- the core
    builder (TF + L-R + L-TF groups). Mutates `adata` in place; returns it.
  - `add_metabolites(adata, metabolites, *, ...)` -- the standalone metabolite group
    (same `metab@<name>` summed-column convention as D11), mergeable, idempotent by name.
  - `diffuse(xy, lig_df, radius, scale_factor, eps=0.001, chunk_size=8000)` -- the sparse
    spatial-diffusion primitive (radius-cutoff Gaussian kernel), usable standalone.
  - `metabolite_interactions(received_export_df, import_gex_df, metabolites)`,
    `is_mouse_data(adata)`, `get_cellchat_db(species, data_dir=None)`,
    `expand_paired_interactions(df)`, `compute_scaled_xy(adata, cell_size=15)` -- the
    lifted primitives, exposed in case a caller wants them directly.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors

# Make the repo root importable regardless of CWD, mirroring build_x.py's own
# resolution -- pure pathlib, no package import, so this is safe under the
# no-SpaceTravLR-import constraint.
_root = next(
    (p for p in Path(__file__).resolve().parents
     if (p / ".git").exists() or (p / "setup.py").exists()),
    Path(__file__).resolve().parents[2],
)
for _p in (str(_root), str(_root / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# `build_x` imports no torch/SpaceTravLR at module level (only sys/warnings/pathlib/
# numpy/pandas) -- safe to import from. Reusing its LAYER/ensure_lognorm_layer keeps
# this store on the exact same y-layer convention as the rest of LinearRegression/.
from metab_processing.LinearRegression.build_x import LAYER, ensure_lognorm_layer  # noqa: E402

# Default location of the shipped reference data (CellChat DB + base GRN), mirroring
# `SpaceTravLR/tools/network.py::get_cellchat_db`'s own path resolution
# (`<repo_root>/src/SpaceTravLR_data/`). Overridable per-call via `data_dir=`.
_DEFAULT_DATA_DIR = _root / "src" / "SpaceTravLR_data"

NICHENET_URL_TMPL = "https://zenodo.org/records/17594271/files/ligand_target_{species}.parquet"


def _resolve_data_dir(data_dir=None):
    return Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR


# ---------------------------------------------------------------------------------
# Lifted primitives (copied faithfully; source cited per function). None of these
# import SpaceTravLR or torch.
# ---------------------------------------------------------------------------------


def _sparse_gaussian_kernel_2d(xy, radius, eps=0.001, query_xy=None, tree=None):
    """Lifted VERBATIM from `src/SpaceTravLR/models/parallel_estimators.py:119-159`
    (`_sparse_gaussian_kernel_2d`). Sparse Gaussian kernel matching
    `tools/utils.py::gaussian_kernel_2d`'s semantics: sigma = radius/sqrt(-2*ln(eps)),
    hard cutoff at `radius` (pairs beyond `radius` get weight 0, not just a small one).
    Self-pairs (dist == 0) always get weight 1.
    """
    if not (radius > 0):
        raise ValueError(f"radius must be positive, got {radius}")
    if query_xy is None:
        query_xy = xy
    if tree is None:
        tree = cKDTree(xy)
    n_rows = query_xy.shape[0]
    n_cols = xy.shape[0]
    sigma = radius / np.sqrt(-2.0 * np.log(eps))
    query_tree = cKDTree(query_xy)
    sdm = query_tree.sparse_distance_matrix(tree, max_distance=radius, output_type="coo_matrix")
    weights = np.exp(-(sdm.data ** 2) / (2.0 * sigma * sigma))
    return coo_matrix((weights, (sdm.row, sdm.col)), shape=(n_rows, n_cols)).tocsr()


def compute_radius_weights_fast_sparse(xy, lig_df, radius, scale_factor, eps=0.001, chunk_size=8000):
    """Lifted VERBATIM from `src/SpaceTravLR/models/parallel_estimators.py:162-199`
    (`compute_radius_weights_fast_sparse`). Row-chunked sparse (KDTree radius-neighbors)
    spatial diffusion:

        received[i, j] = (scale_factor / N) * sum_k Knarrow(i, k) * lig[k, j]

    where `Knarrow` is `_sparse_gaussian_kernel_2d` and N = len(lig_df) (total cell
    count, not neighbor count). Chunking only bounds memory; the result is identical
    (not merely close) regardless of `chunk_size`.
    """
    n = len(lig_df)
    xy = np.ascontiguousarray(xy, dtype=np.float64)
    lig_values = np.ascontiguousarray(lig_df.values, dtype=np.float64)
    n_lig = lig_values.shape[1]
    result = np.empty((n, n_lig), dtype=np.float64)

    if n == 0:
        return pd.DataFrame(result, index=lig_df.index, columns=lig_df.columns)

    effective_chunk = n if (chunk_size is None or chunk_size <= 0) else chunk_size
    tree = cKDTree(xy)
    for start in range(0, n, effective_chunk):
        end = min(start + effective_chunk, n)
        W_chunk = _sparse_gaussian_kernel_2d(
            xy, radius, eps=eps, query_xy=xy[start:end], tree=tree
        )
        weighted_sum = np.asarray(W_chunk @ lig_values)
        result[start:end] = (scale_factor / n) * weighted_sum

    return pd.DataFrame(result, index=lig_df.index, columns=lig_df.columns)


def diffuse(xy, lig_df, radius, scale_factor, eps=0.001, chunk_size=8000):
    """Small public wrapper around `compute_radius_weights_fast_sparse` (mirrors
    `parallel_estimators.compute_radius_weights_fast`'s role as the public entry point)."""
    return compute_radius_weights_fast_sparse(xy, lig_df, radius, scale_factor, eps=eps, chunk_size=chunk_size)


def compute_scaled_xy(adata, cell_size=15):
    """LOCAL (non-mutating) equivalent of `src/SpaceTravLR/tools/utils.py:33-43`
    (`scale_adata`): `scale = cell_size / median(nearest-neighbor distance)` via
    `sklearn.neighbors.NearestNeighbors(n_neighbors=2)` on `adata.obsm['spatial']`.

    Unlike the original (which mutates `adata.obsm['spatial'] *= scale` in place as a
    one-time setup step), this returns `spatial * scale` as a NEW array and leaves
    `adata` untouched -- the real pipeline scales once during `SpaceShip.process_adata_`
    (`spaceship.py:144`) before the estimator ever sees the data; this module has no
    such setup phase, so every caller that needs spatially-scaled coordinates (the
    diffusion kernel's `radius` is in these scaled units) must compute them explicitly.
    """
    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(spatial)
    distances, _ = nbrs.kneighbors(spatial)
    nn_distance = np.median(distances[:, 1])
    scale = cell_size / nn_distance
    return spatial * scale


def is_mouse_data(adata):
    """Lifted VERBATIM from `src/SpaceTravLR/tools/utils.py:279-310`. Gene-name casing
    heuristic (mouse: `Capitalized`, human: `ALLCAPS`), sampled over up to 100 genes."""
    gene_sample = np.random.choice(adata.var_names, size=min(100, len(adata.var_names)), replace=False)
    mouse_pattern_count = sum(
        1 for gene in gene_sample
        if gene[0].isupper() and all(not c.isupper() for c in gene[1:]) and len(gene) > 1
    )
    human_pattern_count = sum(
        1 for gene in gene_sample
        if all(c.isupper() or not c.isalpha() for c in gene) and any(c.isupper() for c in gene)
    )
    return mouse_pattern_count > human_pattern_count


def get_cellchat_db(species, data_dir=None):
    """Re-derived from `src/SpaceTravLR/tools/network.py:17-34` (`get_cellchat_db`):
    reads `cellchat_{species}.csv` from `data_dir` (default
    `<repo_root>/src/SpaceTravLR_data/`, the same directory the real function resolves
    to from its own location)."""
    data_dir = _resolve_data_dir(data_dir)
    return pd.read_csv(data_dir / f"cellchat_{species}.csv")


def expand_paired_interactions(df):
    """Lifted VERBATIM from `src/SpaceTravLR/tools/network.py:36-51`. Splits `_`-joined
    ligand/receptor complexes into one row per ligand x receptor."""
    expanded_rows = []
    for _, row in df.iterrows():
        ligands = row["ligand"].split("_")
        receptors = row["receptor"].split("_")
        for ligand in ligands:
            for receptor in receptors:
                new_row = row.copy()
                new_row["ligand"] = ligand
                new_row["receptor"] = receptor
                expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)


def metabolite_interactions(received_export_df, import_gex_df, metabolites):
    """Re-derived from `src/SpaceTravLR/models/parallel_estimators.py:891-910`
    (`SpatialCellularProgramsEstimator.metabolite_interactions`, there a staticmethod).
    One summed column per metabolite: for `name` with pairs `[(e1,i1),(e2,i2),...]`,
    column `metab@<name>` = sum_k received(e_k) * import(i_k). Both orientations of a
    heterotypic pair are just two list entries, summed here (direction dropped).
    """
    assert isinstance(received_export_df, pd.DataFrame)
    assert isinstance(import_gex_df, pd.DataFrame)
    assert received_export_df.index.equals(import_gex_df.index)
    out = {}
    for name, pairs in metabolites.items():
        acc = np.zeros(len(import_gex_df), dtype=float)
        for e, i in pairs:
            acc = acc + received_export_df[e].values * import_gex_df[i].values
        out[f"metab@{name}"] = acc
    return pd.DataFrame(out, index=import_gex_df.index)


def _load_tflinks(tflinks, species):
    """`tflinks` is a DataFrame (used as-is), a path (read via `pd.read_parquet`), or
    `None` (download NicheNet's `ligand_target_{species}.parquet` from Zenodo -- mirrors
    `parallel_estimators.py:541-552`'s fallback URL)."""
    if tflinks is None:
        return pd.read_parquet(NICHENET_URL_TMPL.format(species=species))
    if isinstance(tflinks, (str, Path)):
        return pd.read_parquet(tflinks)
    return tflinks


# ---------------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------------


def build_linear_factor_store(adata, focus_genes, *, annot_col, radius=300, contact_distance=50,
                              scale_factor=1, receptor_thresh=0.01, tf_ligand_cutoff=0.01,
                              eps=1e-3, species=None, tflinks=None, data_dir=None):
    """Build the TF + L-R + L-TF factor store (no metabolites -- see `add_metabolites`),
    on `LAYER` ('normalized_count'), WITHOUT importing SpaceTravLR.

    Mutates `adata` in place: sets `obsm['x_factors']` (cells x n_unique_cols, float32),
    `uns['x_factors_cols']` (column order), `uns['x_factor_map']` (`{gene: [col names]}`,
    each gene's own order: TF cols, then L-R cols, then L-TF cols), and `uns['x_genes']`
    (genes actually stored, in `focus_genes` order). Returns `adata`.

    A `focus_genes` entry missing from `adata.var_names` is dropped with a warning; a
    gene left with zero columns (either because it had none to start, or because all of
    its columns were dropped as all-zero -- step 7 below) is skipped from
    `x_factor_map`/`x_genes` with a warning, matching `build_x.build_factor_block`'s
    "no modulators" behavior.
    """
    ensure_lognorm_layer(adata)
    if species is None:
        species = "mouse" if is_mouse_data(adata) else "human"
    xy = compute_scaled_xy(adata)
    var_names = set(adata.var_names)

    for gene in focus_genes:
        if gene not in var_names:
            warnings.warn(f"build_linear_factor_store: {gene!r} not in adata.var_names; skipping.")
    genes = [g for g in focus_genes if g in var_names]

    # --- Step 2: GRN edges (NO regression -- include every base-GRN candidate TF). ---
    # No regression pruning yet (all base-GRN candidate TFs included); feedback
    # self-exclusion simplified to direct self only -- see decision log 2026-10-18.
    data_dir_p = _resolve_data_dir(data_dir)
    base_grn = pd.read_parquet(data_dir_p / f"{species}_base_grn.parquet")
    tf_cols_all = [c for c in base_grn.columns if c not in ("peak_id", "gene_short_name")]
    # Restrict to focus genes before the groupby: identical per-gene result, far cheaper
    # than grouping the full ~37k-row base GRN (gene-focus philosophy, 04 decision log).
    base_grn = base_grn[base_grn["gene_short_name"].isin(genes)]
    grn_max = base_grn.groupby("gene_short_name")[tf_cols_all].max()

    gene_tfs = {}
    for g in genes:
        if g not in grn_max.index:
            gene_tfs[g] = []
            continue
        row = grn_max.loc[g]
        tfs = sorted(
            tf for tf in tf_cols_all
            if row[tf] > 0 and tf in var_names and tf != g
        )
        gene_tfs[g] = tfs

    # --- Step 3: L-R pairs (global, gene-independent set). ---
    cellchat = get_cellchat_db(species, data_dir=data_dir_p)
    lr = expand_paired_interactions(cellchat)
    lr = lr[lr["ligand"].isin(var_names) & lr["receptor"].isin(var_names)].reset_index(drop=True)

    # Resolve each ligand's diffusion radius from THIS var_names-filtered table -- BEFORE the
    # receptor_thresh filter and pair dedup -- mirroring init_received_ligands, which builds
    # its own fresh lr (var_names only) and dedups ligands keep-first (CSV order). Sourcing it
    # from the post-filter `lr` would pick the wrong radius for a ligand whose first-order pair
    # fails receptor_thresh but whose surviving pair is at the other signaling radius.
    lig_radius = (
        lr.assign(radius=np.where(lr["signaling"] == "Secreted Signaling", radius, contact_distance))
          .drop_duplicates(subset="ligand", keep="first")
          .set_index("ligand")["radius"]
    )

    raw_receptors = np.unique(lr["receptor"]) if len(lr) else np.array([], dtype=object)
    if len(raw_receptors):
        receptor_expr = adata[:, raw_receptors].to_df(LAYER)
        receptor_levels = receptor_expr.join(adata.obs[annot_col]).groupby(
            annot_col, observed=False).mean().max(0)
        ok_receptors = set(receptor_levels.index[receptor_levels > receptor_thresh])
    else:
        ok_receptors = set()
    lr = lr[lr["receptor"].isin(ok_receptors)]

    lr["pairs"] = lr["ligand"].values + "$" + lr["receptor"].values
    lr = lr.drop_duplicates(subset="pairs", keep="first").reset_index(drop=True)
    lr_pairs_list = list(zip(lr["ligand"], lr["receptor"]))

    # --- Step 4: L-TF ligands per TF (NicheNet), restricted to the L-R ligand set and
    # tf_ligand_cutoff, kept as a RANKED list so each gene's top-5 can exclude that gene from
    # the ligand pool BEFORE ranking -- matching init_ligands_and_receptors (it removes the
    # target gene from the pool, then nlargest(5), so a self-ligand target backfills the next
    # ligand). Columns are materialized only for (lig,tf) actually referenced by some gene. ---
    nichenet_lt = _load_tflinks(tflinks, species)
    lr_ligand_set = set(lr["ligand"].unique())
    all_regulator_tfs = sorted({tf for tfs in gene_tfs.values() for tf in tfs})

    tf_ranked_ligs = {}
    for tf in all_regulator_tfs:
        if tf not in nichenet_lt.index:
            continue
        row = nichenet_lt.loc[tf]
        row = row[row.index.isin(lr_ligand_set)]
        row = row[row > tf_ligand_cutoff].sort_values(ascending=False)
        if len(row):
            tf_ranked_ligs[tf] = list(row.index)

    # --- Step 5: per-gene column lists (direct self-exclusion: no column references g). ---
    x_factor_map = {}
    for g in genes:
        tfs_g = gene_tfs.get(g, [])
        tf_cols = list(tfs_g)  # already target-excluded by construction (tf != g above)
        lr_cols = [f"{lig}${rec}" for lig, rec in lr_pairs_list if lig != g and rec != g]
        ltf_cols = []
        for tf in tfs_g:
            top5 = [lig for lig in tf_ranked_ligs.get(tf, []) if lig != g][:5]
            ltf_cols += [f"{lig}#{tf}" for lig in top5]
        x_factor_map[g] = tf_cols + lr_cols + ltf_cols

    # --- Step 6: diffuse exactly the referenced ligands (one kernel call per distinct radius),
    # then materialize each referenced unique column once (values are target-gene-independent,
    # so shared across genes), in a deterministic column order. ---
    referenced = {c for cols in x_factor_map.values() for c in cols}
    needed_ligs = sorted({c.split("$")[0] for c in referenced if "$" in c}
                         | {c.split("#")[0] for c in referenced if "#" in c})
    needed_genes = sorted(set(all_regulator_tfs) | set(lr["receptor"]) | set(needed_ligs))
    expr_df = adata[:, needed_genes].to_df(LAYER) if needed_genes else pd.DataFrame(index=adata.obs_names)

    lig_radius_needed = lig_radius.reindex(needed_ligs).dropna()
    received_frames = []
    for rad_value in sorted(lig_radius_needed.unique()):
        ligs = lig_radius_needed.index[lig_radius_needed == rad_value].tolist()
        received_frames.append(diffuse(xy, expr_df[ligs], rad_value, scale_factor, eps=eps))
    received = pd.concat(received_frames, axis=1) if received_frames else pd.DataFrame(index=adata.obs_names)

    # Deterministic candidate order: TF columns, then L-R pairs, then L-TF (top-6 per TF --
    # excluding one gene shifts at most the 6th ligand into a gene's top-5). Filter to the
    # referenced set so no orphan column is stored.
    ordered_candidates = list(all_regulator_tfs)
    ordered_candidates += [f"{lig}${rec}" for lig, rec in lr_pairs_list]
    ordered_candidates += [f"{lig}#{tf}" for tf in sorted(tf_ranked_ligs)
                           for lig in tf_ranked_ligs[tf][:6]]

    union_cols = {}
    for col in ordered_candidates:
        if col in union_cols or col not in referenced:
            continue
        if "$" in col:
            lig, rec = col.split("$")
            union_cols[col] = (received[lig].to_numpy() * expr_df[rec].to_numpy()).astype(np.float32, copy=True)
        elif "#" in col:
            lig, tf = col.split("#")
            union_cols[col] = (received[lig].to_numpy() * expr_df[tf].to_numpy()).astype(np.float32, copy=True)
        else:
            union_cols[col] = expr_df[col].to_numpy().astype(np.float32, copy=True)

    # --- Step 7: drop all-zero columns (values are non-negative) from union_cols AND
    # from every gene's map; a gene left with zero columns is skipped (warn), mirroring
    # build_x.build_factor_block's "no modulators" behavior. ---
    zero_cols = {c for c, v in union_cols.items() if float(np.max(v)) == 0.0}
    for c in zero_cols:
        del union_cols[c]
    for g in list(x_factor_map.keys()):
        kept_cols = [c for c in x_factor_map[g] if c not in zero_cols]
        if kept_cols:
            x_factor_map[g] = kept_cols
        else:
            warnings.warn(f"build_linear_factor_store: {g!r} has no modulators; skipping.")
            del x_factor_map[g]

    # --- Step 8: preallocated float32 block (mirrors build_x.build_factor_block). ---
    cols = list(union_cols.keys())
    if cols:
        block = np.empty((adata.n_obs, len(cols)), dtype=np.float32)
        for j, c in enumerate(cols):
            block[:, j] = union_cols[c]
    else:
        block = np.empty((adata.n_obs, 0), dtype=np.float32)

    adata.obsm["x_factors"] = block
    adata.uns["x_factors_cols"] = cols
    adata.uns["x_factor_map"] = x_factor_map
    adata.uns["x_genes"] = list(x_factor_map.keys())
    return adata


# ---------------------------------------------------------------------------------
# Standalone metabolite group
# ---------------------------------------------------------------------------------


def _merge_metab(adata, x_new):
    """Idempotent-by-name merge into `obsm['x_metab']` / `uns['x_metab_modulators']`,
    mirroring `build_x.add_metabolites`'s merge pattern (reimplemented here so this
    module never calls that function, which lazily imports SpaceTravLR/torch via
    `metab_processing.SpaceTravLR.beta_analysis.compute_metab_x`)."""
    if "x_metab" not in adata.obsm:
        adata.obsm["x_metab"] = x_new.to_numpy()
        adata.uns["x_metab_modulators"] = list(x_new.columns)
        return adata

    existing_cols = list(adata.uns["x_metab_modulators"])
    existing = pd.DataFrame(adata.obsm["x_metab"], columns=existing_cols, index=adata.obs_names)
    new_cols = [c for c in x_new.columns if c not in existing_cols]
    merged = pd.concat([existing, x_new[new_cols]], axis=1)

    adata.obsm["x_metab"] = merged.to_numpy()
    adata.uns["x_metab_modulators"] = list(merged.columns)
    return adata


def add_metabolites(adata, metabolites, *, radius=300, contact_distance=50, scale_factor=1,
                    eps=1e-3, data_dir=None):
    """Standalone metabolite group (no SpaceTravLR import): for `metabolites =
    {name: [(export, import), ...]}`, diffuse the (deduped, var-filtered) export genes
    at the SECRETED `radius` (`contact_distance` is accepted for signature parity with
    the core builder but unused -- metabolites always diffuse at `radius`, matching
    `beta_analysis.compute_metab_x`), then `metab_x = metabolite_interactions(
    received_exports, normalized_count[imports], kept)`.

    Mirrors `metab_processing.SpaceTravLR.beta_analysis.compute_metab_x`'s var-filtering
    + dedup: a pair with either gene missing from `adata.var_names` is dropped; exact
    duplicate pairs within one metabolite are deduped; a metabolite left with no valid
    pairs is dropped entirely. Unlike that function (and unlike the real estimator's
    per-target-gene metabolite group), there is no target gene here, so no
    self-exclusion is applied -- the computed value is purely a function of expression
    + diffusion, as `compute_metab_x`'s own docstring notes.

    Merges into `obsm['x_metab']` + `uns['x_metab_modulators']`, idempotent by name
    (an already-stored metabolite column is kept, not recomputed/overwritten) -- see
    `_merge_metab`. Mutates `adata` in place; returns it.
    """
    ensure_lognorm_layer(adata)
    var_names = set(adata.var_names)

    kept = {}
    diffusion_pairs, seen_diff = [], set()
    for name, pairs in metabolites.items():
        keep, seen = [], set()
        for e, i in pairs:
            if e not in var_names or i not in var_names:
                continue
            if (e, i) not in seen_diff:
                seen_diff.add((e, i))
                diffusion_pairs.append((e, i))
            if (e, i) in seen:
                continue
            seen.add((e, i))
            keep.append((e, i))
        if keep:
            kept[name] = keep

    if not kept:
        warnings.warn("add_metabolites: no metabolite had any pair with both genes in "
                      "adata.var_names; nothing added.")
        x_new = pd.DataFrame(index=adata.obs_names)
        return _merge_metab(adata, x_new)

    exports = sorted({e for pairs in kept.values() for e, _ in pairs})
    imports = sorted({i for pairs in kept.values() for _, i in pairs})

    xy = compute_scaled_xy(adata)
    export_expr = adata[:, exports].to_df(LAYER)
    received = diffuse(xy, export_expr, radius, scale_factor, eps=eps)

    import_expr = adata[:, imports].to_df(LAYER)
    x_new = metabolite_interactions(received, import_expr, kept)

    return _merge_metab(adata, x_new)
