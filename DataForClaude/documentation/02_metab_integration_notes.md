# Metabolite integration + tractability notes

> Planning scratchpad for the MetabTravLR adaptation. Captures (1) the **harreman input
> data contract**, (2) how metabolites **map onto SpaceTravLR modulators**, (3) a
> **computational-tractability analysis** for 100k–1M-cell Xenium data, and (4) the **open
> design decisions** to settle before we write any code.
>
> Status: analysis + options only. **No code changes proposed as decided.** When we do
> change code, changes to the existing package must be **minimal and surgical**; our
> data-specific logic lives in separate scripts (e.g. `metab_processing/`).

---

## 1. Harreman input data contract

**What harreman is:** a metabolic cell–cell-communication tool (the metabolite analog of
COMMOT). It uses a **transporter** database to build **gene pairs** per metabolite, then
permutation-tests which pairs/metabolites show significant spatial crosstalk. It runs on a
cheap **KNN graph** (`n_neighbors=5`), not a dense spatial kernel. Driven by
`metab_processing/Harreman/harreman_funcs.py::HarremanRunner`.

**On-disk layout** (example in `easy_download/harreman_outputs/`; full data on Savio under
`<data>/easy_download/harreman_outputs/`):

```
easy_download/harreman_outputs/
├── harreman_network.json                         # metabolite ↔ gene-pair DATABASE (baseline)
├── [ccc_results][cell_com_df_gp_sig].csv         # cell-type-INDEPENDENT, gene-pair level
├── Tier1/                                         # cell-type-AWARE, per annotation column "Tier1"
│   ├── [ct_ccc_results][cell_com_df_m].csv        #   metabolite level, per (CellType1→CellType2)
│   └── [ct_ccc_results][cell_com_df_gp_sig].csv   #   gene-pair level, per (CellType1→CellType2) — real (see note)
└── Tier2/  (same as Tier1 for the finer annotation)
```

### `harreman_network.json` — the metabolite → gene-pair map (our key lookup)
Keys: `num_transporter_genes` (100), `num_metab` (160), `num_gp` (416),
`metabolite_pair_counts`, `transporter_gense` [sic], `gp`, `gp_per_metabolite`.
- `gp`: list of `[geneA, geneB]` transporter pairs, e.g. `["SLC2A1","SLC2A9"]`.
- `gp_per_metabolite`: `{ metabolite_name: { "gene_pair": [[g1,g2],…],
  "gene_type": [["IMP-EXP","IMP-EXP"],…] } }`.
- Transporter genes are `SLC*`, `ABC*`, `AQP*`, `ATP*` families.
- ⚠️ **Direction:** every gene is typed **`IMP-EXP`** (transporters are bidirectional), and
  **many pairs are homotypic** (`SLC2A2–SLC2A2`). This is *not* a directional
  ligand→receptor edge. See §2 for how we handle orientation.

### Significant-interaction tables
- **Cell-type-independent** `[ccc_results][cell_com_df_gp_sig].csv` — one row per gene pair:
  `Gene 1, Gene 2, C_p, Z, Z_pval, Z_FDR, C_np, pval_np, FDR_np, …, selected`. `selected`
  = passed FDR. (~136 significant pairs in the example.)
- **Cell-type-aware** `Tier{N}/[ct_ccc_results][cell_com_df_m].csv` — one row per
  `(Cell Type 1, Cell Type 2, metabolite)`: `…, C_p, Z, Z_pval, Z_FDR, C_np, pval_np,
  FDR_np, selected`. This is the **directional (sender→receiver) metabolite-level**
  significance, and is what "**pick Tier1 → select all significant metabolites**" reads:
  filter `selected == True` (or by `FDR_np`).

**Selecting metabolites (Foster's intended entry point):**
1. Choose a tier (`Tier1` coarse / `Tier2` fine — these are `adata.obs` cell-type columns
   passed as `cell_type_col` to `run_harreman`).
2. Read that tier's `_m` CSV, keep `selected == True` → list of `(sender, receiver,
   metabolite)`.
3. Expand each metabolite → its transporter `gene_pair`s via `gp_per_metabolite`.
4. Those gene pairs become the candidate metabolite edges for SpaceTravLR (§2).

### Bugs / rough edges spotted (note, don't fix yet)
- ~~`save_harreman_outputs` writes `cell_com_df_m` to both `_m` and `_gp_sig`~~ — **fixed in the
  current runner** (`save_harreman_outputs` writes `cell_com_df_gp_sig` distinctly, cols:
  `Cell Type 1, Cell Type 2, Gene 1, Gene 2, …, selected`). Verified 2026-07-16 against the
  `easy_download` example; `metab_processing/Harreman/harreman_summary.py` relies on this real ct-level
  gene-pair table.
- JSON key typo `transporter_gense`.
- ⚠️ **`Cell Type 1 → Cell Type 2` is NOT a direction — RESOLVED (2026-07-16).** The
  ordering is just the **sorted label order**: `compute_gene_pairs` enumerates ct pairs with
  `itertools.combinations_with_replacement` (when `fix_ct` unset), so each unordered pair is
  computed once and the reverse is never tested (verified: the ct-pair set in every tier CSV
  equals `combinations_with_replacement` of the sorted labels; `other` sorts last so
  `(other → T)` never exists). All transporter genes are typed `IMP-EXP` (2042/2042) and both
  gene orders are folded into each pair, so the score is an **undirected** spatial
  co-expression across the CT1–CT2 interface. My earlier "T cells only export" reading was an
  artifact. `harreman_summary.py` now reports undirected `A–B` interfaces / `A (self)`.
  Full trace + score formula in [`05_harreman_reference.md`](05_harreman_reference.md).

---

## 2. Mapping metabolites → SpaceTravLR modulators

SpaceTravLR's signaling modulator is an **L–R pair**: `received_ligand(export, diffused) ×
receptor(local)`, with a learned spatial coefficient β. A metabolite edge fits this shape:

- **export/source gene** → treated as the "ligand": its expression is **Gaussian-distance-
  weighted from neighbors** (the metabolite a cell receives from its surroundings).
- **import/sink gene** → treated as the "receptor": **local** expression (surface uptake).
- product → per-cell metabolite-flux proxy; β on it = "how strongly this metabolite edge
  modulates the target gene, here."

**Insertion approach — DECIDED (D6): metabolites are their OWN new modulator group**, not
folded into the existing L–R `extra_lr` group. We reuse the L–R *computation* (received
ligand × import, radius-diffused) but keep metabolites as a **separate group-lasso group
(#5)** with a **distinct betadata separator** (e.g. `@` → `beta_<export>@<import>`), because
`beta.py::BetaFrame` classifies columns purely by separator (`$`=L–R, `#`=L–TF, none=TF) and
we want metabolite β's independently identifiable/readable.
- The existing `extra_lr` param (`init_ligands_and_receptors`, `init_received_ligands`) is the
  closest reference implementation but puts pairs in group #2 with a `$` separator — so we
  add a **parallel** metabolite path rather than reuse it verbatim.
- Full surgical map (exact functions/lines: modulator assembly, `groups` array in `fit`,
  betadata column naming, reading back) is in `01_pipeline_deep_dive.md`.
⚠️ Nothing metabolite-specific is threaded through `SpaceShip` / `oracles.SpaceTravLR.run()`
yet (§4).

**Directionality (`IMP-EXP`) — the one real modeling question:**
- L–R is directional; transporter pairs are symmetric. Options:
  - (a) **Both orientations**: add `(g1→g2)` and `(g2→g1)`; let group-lasso keep whichever
    matters. Doubles metabolite columns (still small).
  - (b) **Orient by harreman's cell-type direction**: the `_m` table gives sender→receiver;
    use that to pick export vs import. More principled, more bookkeeping.
  - (c) **Homotypic pairs** (`g–g`): received(g) × local(g) — a single diffusible-autocrine
    term; fine as-is (the code only forbids pairs where ligand/receptor == the *target*
    gene, not ligand==receptor).
- ✅ **DECIDED (D3, 2026-07-10): start with (a) both orientations** — add `(g1→g2)` and
  `(g2→g1)` and let group-lasso prune. Revisit (b) harreman-directed orientation later.

**Scale of what we're adding:** 160 metabolites / 416 gene pairs / 100 transporter genes —
tiny next to CellChat's L–R set. Adding them as candidate modulators is cheap; group-lasso
zeros out the irrelevant ones so the learned β's are the "actually has an effect" filter
Foster described (harreman = biological prior on what *could* flux; SpaceTravLR β = what
*measurably modulates* the target genes, spatially).

---

## 3. Tractability & computational complexity (100k–1M cells, Xenium ~5k-gene panel)

### The headline blockers: two **dense O(N²)** ops on the training path
Both live in `models/parallel_estimators.py`, run over **all N cells at once**, with **no
spatial sparsity**:

| Op | Location | Cost | @100k cells | @1M cells |
|---|---|---|---|---|
| Received-ligand Gaussian kernel | `_gaussian_kernel_2d_batch` (L65) → `received_ligands` (L106) | materializes **N×N** weight matrix | ~80 GB (f64) | ~8 TB |
| Spatial neighborhood features | `create_spatial_features` → `cdist(coords,coords)` (L244) | **N×N** distance matrix | ~80 GB | ~8 TB |

- Each is computed **once and cached** (received ligands in `adata.uns`; spatial features in
  `adata.obsm`), so it's not per-gene — but even **once** it OOMs at ≥100k cells. **These are
  hard blockers for Foster's data as-is.**
- Note the "fast" numba kernel has **no radius cutoff** (unlike the per-cell
  `gaussian_kernel_2d`, which zeroes weights beyond `radius`). Since real neighborhoods are
  local (radius ≪ tissue), the true weight matrix is **sparse** — this is exactly the
  structure to exploit.

**Mitigations, least→most invasive** (for the plan; not implementing now):
1. **Row-blocking (least invasive, exact):** compute received ligands / neighbor counts in
   chunks of rows, never materializing full N×N. Same math, O(N²) time but O(N·chunk)
   memory. Localized edit to two helper functions.
2. **Sparse radius-neighbors (recommended, ~exact):** replace the dense kernel with a
   `cKDTree.query_ball_point` / `radius_neighbors` at `radius` (where the Gaussian ≈ 0) →
   sparse weights → sparse matmul. O(N·k), k = avg neighbors in radius. `create_spatial_features`
   becomes a KDTree radius-count. **The repo already uses `cKDTree`/`NearestNeighbors`
   elsewhere** (`virtual_tissue.py`, `tools/analysis.py`, `tools/utils.py`), so the pattern
   is in-house — but this does edit core functions, so weigh against the "minimal changes"
   rule. It may be unavoidable at 1M cells.
3. **Spatial tiling:** partition the tissue into overlapping tiles and process independently
   (harreman-style locality). Most invasive; only if 1–2 aren't enough.

### Per-gene training cost (the other axis)
- Per gene: for each cluster (cell type) → group-lasso fit (`N_cluster × M_modulators`) +
  CNN train (`N_cluster × epochs`). Paper benchmark: **~60 s/gene @ 10k cells, 100 epochs,
  A100**. Roughly **linear in N_cells** and in **#clusters** (Tier2 finer ⇒ more clusters ⇒
  more per-gene work; also more spatial-map channels).
- Training **all ~5000 panel genes** at, say, 100k cells would be ~hundreds of GPU-hours —
  only feasible via the SLURM worker fan-out (`OracleQueue`). But we don't need all genes:

### ⭐ Restrict training to genes-of-interest (biggest, cheapest win)
Because we **read β directly (no multi-hop propagation)**, we only need
`betadata/{G}_betadata.parquet` for the **target genes G in our gene sets S** — typically
dozens–hundreds, not 5000. Each G's model still uses metabolite/TF/LR predictors (those only
need to be *expression columns*, which they are). → potential **10–100× compute reduction**,
and it composes with the O(N²) fixes above.

- **Near-zero-edit mechanism:** `SpaceTravLR.run()` iterates `self.queue`; the queue's gene
  list is `OracleQueue(save_dir, all_genes=…)`. We can seed/override it with our subset
  (e.g. set `space_travlr.queue = OracleQueue(base_dir, all_genes=target_genes)` from a
  script — **no package edit**), or add an optional `genes=` param (one-line surgical edit).
- ⚠️ **Orphan gotcha:** `run()` skips any gene whose **TF** `regulators == []`
  (`oracles.py:475`) — even if it has L–R/metabolite modulators. So a target gene with no
  CellOracle TF links gets dropped. If we want metabolite-only models for such genes we must
  relax that check (surgical) or ensure TFs are present. Ties into §4.2.

### Setup-phase costs (one-time, on Savio)
- `run_celloracle_`: per-cluster Bayesian-ridge GRN over the panel — scales with genes ×
  clusters × cells; CellOracle handles it but it's non-trivial at scale.
- COMMOT (`run_commot_`) is **optional** and expensive (optimal transport); the paper itself
  subsampled to ~20k cells for it. For metabolites we get our prior from **harreman instead**,
  so we likely **skip COMMOT** and pass metabolite edges directly. (One reason to make the
  built-in L–R/COMMOT path optional — §4.1.)

---

## 4. Open design decisions (settle these before coding)

1. ~~**Keep vs. drop the built-in TF and L–R modulators?**~~ ✅ DECIDED (D7): **keep, but
   optional** (default keep) — metabolite β estimated *controlling for* known regulation.
   Existing toggles: `use_ligands=True/False`, `grn` choice. Metabolites added as a new group
   (D6) on top.
2. **Genes to train:** just the gene sets S? (Recommended, §3.) And how to handle the
   **orphan/no-TF** case for target genes lacking GRN links.
3. ~~**Metabolite orientation:** both-orientations (a) vs harreman-directed (b).~~ ✅ DECIDED
   D3: both orientations.
4. **Gene-set input format:** JSON or `dict{label: [genes]}`. First pass: labels
   `positive` (T-cell activity) / `negative` (exhaustion).
5. **Signed aggregation across a gene set:** proposal —
   `score(metabolite) = mean_{g∈positive} β̄(metab,g) − mean_{g∈exhaustion} β̄(metab,g)`
   (β̄ = β aggregated over cells / cell types / a spatial region). A metabolite that ↑activity
   and ↓exhaustion scores high. Where β varies in space, we can also map the score. Confirm
   whether the "negative sign" combines the two labels this way.
6. **Tier selection:** which harreman tier drives edge selection, and whether to use its
   sender/receiver cell types to restrict which cells carry the edge.

---

## 5. Candidate integration surface (where minimal edits would land — for later)

Kept here so the eventual plan is concrete. **Not** proposing to edit yet.
- **Our new code (separate scripts, e.g. `metab_processing/`):** load harreman outputs →
  select significant metabolites for a tier → build `extra_lr` list + gene-set dict →
  driver that trains only S and reads `beta_<export>$<import>` from betadata → signed
  aggregation / plots.
- **Minimal package touches (only if needed):** (i) thread `extra_lr` from `SpaceShip`/
  `oracles.SpaceTravLR` into the estimator; (ii) optional `genes=` subset on the trainer/
  queue; (iii) relax the no-TF orphan skip when metabolite/LR modulators exist; (iv) the
  O(N²)→sparse kernel swap if Foster's cell counts demand it. Reuse existing β-reading
  helpers (`beta.BetaFrame`, `plotting/niche.get_modulator_betas`) rather than adding code.

---

## 6. Cross-refs
- Architecture & two-phase model: `00_overview.md`.
- Paper text: `paper_fulltext.txt`.
- Harreman driver: `../../metab_processing/Harreman/harreman_funcs.py`, `run_harreman.ipynb`.
