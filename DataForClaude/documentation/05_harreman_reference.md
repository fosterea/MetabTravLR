# 05 — harreman reference (what it does, from the source)

Durable notes on **harreman**, the package Foster runs to score metabolite crosstalk between
cells. Written by reading harreman's `tools` source + the `HarremanRunner` wrapper
(`metab_processing/harreman_funcs.py`) and verifying against the `easy_download` example.
**Read this before touching anything that consumes harreman output.** It is the source of
truth for the arrow semantics, the output tables, and the known GPU-OOM in the per-cell
("neighborhood") analysis.

---

## 1. What harreman is

harreman infers **spatial cell–cell communication (CCC)** from spatial transcriptomics
(here Xenium). It has two databases: **`LR`** (ligand–receptor) and **`transporter`**
(metabolite importers/exporters). We use **`transporter` only** — the question is which
**metabolites** are exchanged between neighboring cells, read out through **transporter gene
pairs** (SLC*/ABC*/AQP*/ATP* families).

Core idea: a metabolite is "communicated" between two spatially-adjacent cells if one cell
expresses a transporter for it and its neighbor expresses a (possibly the same) transporter
for it. harreman scores this as a spatially-weighted co-expression of the metabolite's
transporter gene pairs, and tests significance by permutation (non-parametric) and/or an
analytic Z-score (parametric).

The method is a spatial-CCC cousin of **Hotspot** (harreman vendors `..hotspot` models);
the statistical models `danb` / `bernoulli` / `normal` / `none` used to standardize counts
come from there.

---

## 2. The pipeline (functions in call order)

Wrapper = `HarremanRunner` in `metab_processing/harreman_funcs.py`. It calls these harreman
`tl`/`pp` functions:

| Step | Function | What it does |
|---|---|---|
| DB | `pp.extract_interaction_db(adata, database='transporter', extracellular_only=True)` | Loads the transporter interaction DB into `adata.varm['database']` (genes × metabolites, values ∈ {−1 IMP, +1 EXP, 2 IMP-EXP}) and `adata.uns['database']='transporter'`. |
| Filter | `tl.apply_gene_filtering(model='danb', ...)` | Optional gene filters: feature-elimination (sparsity), spatial-autocorrelation, per-cell-type expression, per-cell-type DE. Foster runs with `autocorrelation_filt=False` (i.e. keep all DB genes). Writes `uns['filtered_genes']`, `uns['filtered_genes_ct']`. |
| Graph | `tl.compute_knn_graph(compute_neighbors_on_key='spatial', n_neighbors=5, weighted_graph=False)` | Spatial KNN → `adata.obsp['weights']`. `weights[i,j] > 0` means **j is a spatial neighbor of i** (row = focal cell i, col = neighbor j). Not symmetric in general. |
| Pairs | `tl.compute_gene_pairs(ct_specific=?, cell_type_key=?)` | Builds the metabolite→gene-pair map and, when `ct_specific`, the cell-type-pair enumeration. **See §3 — this is where the "arrow" is decided.** |
| CCC (agnostic) | `tl.compute_cell_communication(model='danb', M=1000, test='both', layer_key_p_test='counts', layer_key_np_test='log_norm')` | Cell-type-**independent** scores + significance → `uns['ccc_results']`. |
| CCC (aware) | `tl.compute_ct_cell_communication(..., cell_type_key, subset_gene_pairs=<sig from agnostic>, fix_gp=False)` | Cell-type-**aware** scores stratified by cell-type pair → `uns['ct_ccc_results']`. |
| Select | `tl.select_significant_interactions(test='non-parametric', ct_aware=?, threshold=0.05)` | Adds `selected` bool + `_sig` subset tables. **NP significance = `FDR_np < thr AND C_np > 0`.** |

`HarremanRunner.run_harreman(cell_type_col)` runs agnostic → aware(for that annotation column
= "tier") → saves CSVs. Each **Tier** is just one `cell_type_col` (a different `adata.obs`
annotation granularity); harreman has no notion of "tiers" — that is Foster's naming.

Layers: parametric test uses raw **`counts`**; non-parametric uses **`log_norm`**
(normalize_total 1e4 + log1p). We care about **non-parametric FDR (`FDR_np`)**.

---

## 3. THE ARROW: `Cell Type 1 → Cell Type 2` is NOT a direction

**This is the single most important gotcha.** For a cell-type pair `(CT1, CT2)` and gene
pair `(Gene 1, Gene 2)`, the cell-type-aware score is (from
`run_ct_cell_communication_analysis` + `create_weights_ct_pairs`):

```
score(CT1, CT2) = Σ_{i ∈ CT1, j ∈ CT2}  Gene1_expr(i) · W(i, j) · Gene2_expr(j)
```

- `W(i,j)` = spatial weight, **CT1 = row/focal cell i, CT2 = col/neighbor cell j**. `Gene 1`
  is read in CT1, `Gene 2` in CT2. (`compute_gene_pairs` keeps a `(var1,var2)`×`(ct1,ct2)`
  combo iff `var1 ∈ filtered_genes_ct[ct1]` and `var2 ∈ filtered_genes_ct[ct2]`.)
- Uses **only `W`** (not `W + Wᵀ`) — contrast the cell-type-**independent** path, which
  **symmetrizes** (`(counts_1·WX2) + (counts_1·WtX2)`) and is explicitly undirected.

Despite that anchoring, the ordering **cannot be read as flux direction**, for three
independently-sufficient reasons:

1. **Only the sorted ordering is ever computed.** `compute_gene_pairs` builds ct pairs with
   `itertools.combinations_with_replacement(sorted_cell_types, 2)` when `fix_ct` is unset
   (Foster's runs). Each *unordered* pair appears **once**, in sorted-label order; the reverse
   is never tested. **Verified**: the `(CT1,CT2)` set in every tier CSV == `combinations_with_
   replacement` of the sorted labels. Because `"other"` sorts last (lowercase), `(other → T)`
   simply never exists — it was *not tested*, not *found insignificant*.
2. **Genes carry no direction.** Every transporter gene is typed **`IMP-EXP`** (bidirectional)
   — verified 2042/2042 type slots. No pure IMP/EXP anywhere.
3. **Both gene orders are folded in.** For transporters `compute_gene_pairs` unions
   `combinations_with_replacement` **and** `permutations`, so within a ct pair both `(a,b)` and
   `(b,a)` gene orders contribute to the metabolite score.

**Correct reading:** `score(CT1,CT2)` is an **undirected** measure of how strongly a
metabolite's transporters are **spatially co-expressed across the CT1–CT2 neighbor interface**
(or *within* a type, on the diagonal `CT==CT`). You may say *"metabolite M is exchanged at the
Effector-CD8 ↔ other interface"*; you may **not** say *"M flows into/out of T cells."*

To get any directionality you'd have to rerun with **`fix_ct='all'`** (→ `itertools.product`,
both orderings) and compare `score(X,Y)` vs `score(Y,X)` — but with `IMP-EXP` genes even that
asymmetry is spatial-anchoring, not metabolite flux. Not recommended as "direction".

`metab_processing/harreman_summary.py` encodes all of the above: undirected `A–B` interfaces
and `A (self)` diagonals; `tcell_involvement ∈ {within_Tcell, within_other, Tcell_interface,
non_Tcell_interface, cell_type_independent}`.

---

## 4. Output data model

### 4a. `harreman_network.json` (from `save_harreman_network`, DB baseline)
- `num_transporter_genes` (100), `num_metab` (160), `num_gp` (416).
- `metabolite_pair_counts`: `{metabolite: n_gene_pairs}` (1–91).
- `gp`: list of `[geneA, geneB]` pairs. `gp_per_metabolite`: `{metabolite: {gene_pair:[[g1,g2],…],
  gene_type:[["IMP-EXP","IMP-EXP"],…]}}`.
- **Gene pair ↔ metabolite is many-to-many** (139/416 pairs serve >1 metabolite; e.g.
  `SLC2A3–SLC2A4` → glucose, DHA, dehydroascorbate). So a significant *gene pair* does not
  pin a single metabolite. Typo key: `transporter_gense`.

### 4b. Result tables (per `select_significant_interactions`)
Two grains × two tests. Grain: **`gp`** (gene-pair) and **`m`** (metabolite, = sum of its gene
pairs' scores via `compute_metabolite_cs`). Test columns: parametric `C_p, Z, Z_pval, Z_FDR`;
non-parametric `C_np, pval_np, FDR_np`. `selected` bool + `_sig` filtered copy added by
`select_significant_interactions`.

| Table (in `uns`) | Written CSV (by `save_harreman_outputs`) | Rows |
|---|---|---|
| `ccc_results['cell_com_df_m']` | `[ccc_results][cell_com_df_m].csv` | metabolite (no cell types) |
| `ccc_results['cell_com_df_gp_sig']` | `[ccc_results][cell_com_df_gp_sig].csv` | gene pair (no cell types) |
| `ct_ccc_results['cell_com_df_m']` | `Tier*/[ct_ccc_results][cell_com_df_m].csv` | (CT1, CT2, metabolite) |
| `ct_ccc_results['cell_com_df_gp_sig']` | `Tier*/[ct_ccc_results][cell_com_df_gp_sig].csv` | (CT1, CT2, Gene1, Gene2) |

The ct `_gp_sig` file **is** a real gene-pair table (an earlier version of the wrapper
duplicated `_m` there; current code fixed it — verified). `_m_sig` = `selected==True` subset.
NP `selected` requires `FDR_np < thr AND C_np > 0` (positive communication only).

### 4c. Our summary (`harreman_summary.py`)
`summarize_harreman_folder(path, background_label='other', sample_id=None) → (master, genepairs)`.
- **master** (1 row/metabolite): `n_gene_pairs`, `transporter_genes`, `global_significant`,
  `global_FDR_np`, `n_sig_gene_pairs_global`, and per tier `_n_sig_pairs`, `_interactions`
  (all undirected, compact), `_within_Tcell`, `_Tcell_interfaces`, `_tcell_involved`.
- **genepairs** (1 row/significant `tier × ct-pair × gene pair`): `tcell_involvement`,
  `metabolites` (many-to-many join), `FDR_np, pval_np, C_np`. `tier='global'` rows = the
  cell-type-independent gp table.
- Built for one folder; a later multi-sample manager concats with `sample_id`.

---

## 5. Per-cell "neighborhood" analysis + the GPU-OOM ⚠️

Beyond the aggregate CCC scores, harreman can compute **per-cell interacting-cell scores** —
for **each individual cell**, its contribution to each gene-pair / metabolite interaction
(i.e. *which cells* are doing the talking, not just whether the interaction exists). These are
**not** run by the current `HarremanRunner`; they are separate `tl` functions:

- `compute_interacting_cell_scores(...)` — cell-type-**independent**, per cell. Stores into
  `uns['interacting_cell_results']` and (implicitly) per-cell score matrices.
- `compute_ct_interacting_cell_scores(...)` — cell-type-**aware**, per cell per ct-pair.
  Stores per-cell dataframes in `adata.obsm['ct_interacting_cell_results_{p,np}_{gp,m}_cs_df']`
  with columns like `"CT1 - CT2: metabolite"`.
- Downstream: `compute_interaction_module_correlation(...)` correlates those per-cell scores
  against `obsm['module_scores']` (Hotspot gene modules) → which metabolite interactions
  track which expression modules.

**The exact call Foster is making** (`run_harreman.ipynb`, cell 9) is the non-ct one:
```python
harreman.tools.compute_interacting_cell_scores(
    adata, center_counts_for_np_test=False, test='both',
    restrict_significance='both', compute_significance='both',
    M=1000, seed=42, check_analytic_null=False, verbose=True)
```
`compute_significance='both'` + `M=1000` is precisely what triggers the giant permutation-null
allocation below; `test='both'` runs the parametric *and* non-parametric passes. (He also
inspects `uns['ct_interacting_cell_results']` in cell 12 — the ct twin has the same problem.)

**Why it OOMs on large data (100k–1M cells).** These functions allocate **dense tensors with
a per-cell axis**, on GPU:

- `compute_interacting_cell_scores` (non-parametric permutation) allocates
  `perm_cs_gp_a`, `perm_cs_gp_b` of shape **`(n_cells, n_gene_pairs_sig, M)`** and
  `perm_cs_m_a`, `perm_cs_m_b` of shape **`(n_cells, n_metabolites, M)`** — i.e. it stores the
  *entire permutation null for every cell*. At 100k cells × 150 pairs × M=1000 × 8 bytes ≈
  **120 GB per array** (×4). This is the crash.
- `compute_ct_interacting_cell_scores` allocates `cs_gp` of shape
  **`(n_ct_pairs, n_cells, n_gene_pairs_sig)`** (dense) plus a metabolite twin — e.g. 15 × 100k
  × 150 × 8 ≈ **180 GB** — and OOMs at the *score* step (it doesn't even implement the
  permutation test).

**Fix directions (noted, not implemented):**
1. **Don't materialize the permutation null.** The aggregate tests already do the right thing:
   they accumulate an exceedance count `x = (perm_cs > cs).sum(dim=M)` — you can accumulate
   that counter **incrementally per permutation** and never store the `(…, M)` axis. Porting
   that pattern to the per-cell path removes the `M` dimension entirely.
2. **Chunk the cell (and/or gene-pair) axis** and move results to CPU / write to disk per
   chunk; the ct loop already iterates ct-pairs, so chunk cells inside it.
3. **Lower `M`** or use the analytic Z-score (parametric) for the per-cell scores.
4. Keep dense per-cell matrices as **sparse**/`float32` and off-GPU where possible.

If we ever need per-cell metabolite scores at Xenium scale, option (1) is the surgical fix.
(Also relevant to the tractability analysis in `02_metab_integration_notes.md`.)

**Implemented (2026-07-16):** `metab_processing/interacting_cell_scores_lowmem.py` is a
drop-in `compute_interacting_cell_scores_lowmem` applying fix (1) — it accumulates the
exceedance counters per permutation instead of storing the `(cells, pairs, M)` arrays, and
no longer writes the raw `perm_cs_*` to `uns`. It reuses harreman's own helpers (imported
from the installed package at runtime), so numerics match the stock function for a given
seed. `run_harreman.ipynb` cell 9 now calls it (stock call kept commented). **Untested
locally** (harreman is Savio-only); validate on Savio — the file's footer has a small
diff-against-stock sanity check. Confirmed OOM site: `harreman/tools/cell_communication.py`
line ~1793 (`perm_cs_gp_a = torch.zeros((n_cells, n_gene_pairs, M), ...)`); the example run
was 112,551 cells × 136 sig pairs × M=1000 → ~114 GiB *per array* (×4). Only the ct twin
(`compute_ct_interacting_cell_scores`) is still un-patched — it needs cell-axis chunking as
well since its `cs_gp` is dense `(n_ct_pairs, n_cells, n_gene_pairs)`.

---

## 6. Quick "how do I…" index
- *Is metabolite M exchanged at all?* → `[ccc_results][cell_com_df_m].csv`, `selected`/`FDR_np`.
- *Between which cell types (this annotation)?* → `Tier*/[ct_ccc_results][cell_com_df_m].csv`,
  read pairs as **undirected** interfaces (§3).
- *Which transporter genes carry it there?* → `Tier*/[ct_ccc_results][cell_com_df_gp_sig].csv`
  (+ `gp_per_metabolite` in the network JSON for the metabolite↔pair map).
- *One tidy view of all of the above* → `metab_processing/harreman_summary.py`.
- *Per-cell "who is talking"* → the interacting-cell-score funcs (§5) — **OOM risk**.
