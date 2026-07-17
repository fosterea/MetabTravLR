# SpaceTravLR: data flow, where the costs are, and the efficiency work

> A memory/compute-lens view of how a training run flows, so we can reason about scaling.
> Answers Foster's questions: (a) how info flows, (b) what we've sped up + what's left,
> (c) what "only metabolites" would save, (d) how the CNN-over-a-grid scales.
>
> Concrete reference run (the one that crashed): **Xenium melanoma, 112,551 cells × 5,006
> genes**, `leiden_scVI_res_0.5` cell types, `fit()` in-notebook on a **10.5 GB GTX 1080 Ti**.

## Notation (the things costs scale with)
- **N** = cells (112k here). **C** = cell-type clusters (~11–15). **G** = genes *trained*
  (focus list, 8; or all ~5006). **M** = modulators per gene (TF + L–R + L–TF; ~1121 for IL10).
- **L** = candidate ligand genes (CellChat, hundreds–~1000). **nbr** = avg neighbors within
  `radius` (~1000 at Xenium density, radius=300). **d** = spatial grid side (64). **E** = epochs.

---

## 1. Information flow (with cost annotations)

```
INPUT: AnnData  X:(N×genes) counts, obsm['spatial']:(N×2), obs['cell_type']
  │
  ▼  SpaceShip.setup_()                                              [ONE-TIME]
  ├─ process_adata_: normalize/log1p, MAGIC impute (per cluster)     time ~O(N·k) (Foster: OK@100k)
  │                  → layer 'imputed_count'; write _adata.h5ad      mem  ~O(N·genes)
  ├─ run_celloracle_: per-cluster bagging-ridge GRN over FOCUS       time O(C·G_focus·bag·ridge)
  │                   targets → celloracle_links.pkl                 (restricted to focus genes ✓)
  └─ get_nichenet_links_: download tflinks.parquet                   one download
  │
  ▼  run_spacetravlr()/fit():  per gene g in the focus queue (G genes)
  │
  ├─ init_data()  [heavy structures are BUILT ON FIRST GENE, then CACHED in adata.uns/obsm]
  │   ├─ init_received_ligands → received_ligands over L ligands     ⚠ time O(N·nbr·L),
  │   │      sparse KDTree Gaussian, row-chunked                        mem O(chunk·nbr + N·L)
  │   │      → adata.uns['received_ligands']                            (was 57 GB; now ~1.5 GB) ✅
  │   ├─ xyc2spatial_fast → spatial maps tensor                       ⚠⚠ mem O(N·C·d²)=20 GB@100k
  │   │      → adata.obsm['spatial_maps']  (N × C × 64 × 64 float32)      59 GB@300k  ← BIGGEST LEFT
  │   ├─ create_spatial_features → KDTree radius counts               time O(N·nbr), mem O(N·C) ✅
  │   └─ per-gene: init_ligands_and_receptors (CellChat+NicheNet)     time O(L+M)
  │         build train_df (N × M design matrix)                     mem O(N·M)
  │
  ├─ fit(): per cluster c (C of them)
  │   ├─ GroupLasso on (N_c × M) → anchor betas                      time O(N_c·M·iters), mem O(N_c·M)
  │   └─ CNN train E epochs, DataLoader batch=512                    time O(N_c·E·conv(d²))
  │         (SPG 64×64 → CNN → MLP → β);  GPU mem O(batch·16·d²)      (batched ✓)
  │
  ├─ get_betas(): per cluster, run CNN forward on all N_c cells      ⚠ GPU mem O(N_c·16·d²)
  │         → per-cell betas                                            was UNBATCHED → OOM ❌→✅(batching)
  │
  └─ write betadata/{g}_betadata.parquet  (N × M)                    disk O(N·M) (~1 GB/gene @ M=1121)
```

**Two axes of cost.** Everything scales along one of two axes:
- **"Tall" (cells, N):** spatial-maps tensor, received-ligand precompute, CNN train/infer,
  MAGIC. This is the **scaling wall** — it grows with your Xenium cell count.
- **"Wide" (modulators/ligands, M/L):** GroupLasso, the design matrix, betadata size, the
  final MLP width. This is what "only metabolites" shrinks (§4) — but it is *not* the wall.

---

## 2. The memory hot-spots (why it crashed, in order)

| Structure | Size | @100k | @300k | Status |
|---|---|---|---|---|
| received-ligand matrix | O(N·nbr) | ~1.5 GB | ~4 GB | ✅ fixed (narrow kernel + chunk, `793c096`) — was **57 GB** |
| **get_betas CNN forward** (per cluster, GPU) | O(N_c·16·d²) | **>9 GB → OOM** | worse | ⏳ **being batched now** (the actual crash) |
| **spatial-maps tensor** (CPU) | O(N·C·d²) | **~20 GB** | ~59 GB | ⚠ **next ceiling — not yet fixed** (CU-5c) |
| betadata parquet (per gene) | O(N·M) | ~1 GB | ~3 GB | ok; smaller if fewer modulators (§4) |
| create_spatial_features | O(N·nbr) | ~1.4 GB | ~4 GB | ✅ KDTree (`691ab41`) — was 80 GB |

The crash you hit was the **get_betas GPU OOM** (`Tried to allocate 9.28 GiB` on a 10.5 GB
card): after IL10 trained, `get_betas` ran the CNN on a whole cluster's cells at once. Fixed by
batching (in eval mode — see §5). The **next** thing you'll hit at ~200–300k cells is the
20–60 GB **spatial-maps tensor** held in RAM for all cells; that's CU-5c (§3).

---

## 3. Efficiency improvements — made vs. still to make

### Made (all committed, behavior-preserving unless noted)
| # | Change | Effect | Commit |
|---|---|---|---|
| 1 | **Gene focus** — train only target genes | G: 5006 → |focus| (10–100× fewer gene-models) | `2e523aa` |
| 2 | **CellOracle GRN restricted to focus genes** | setup GRN O(all genes) → O(focus) | `1c7c40f` |
| 3 | **activation-kwarg crash fix** | unblocks real training (poor-fit branch) | `2699947` |
| 4 | **create_spatial_features → KDTree** (bit-exact) | 80 GB → 1.4 GB @100k | `691ab41` |
| 5 | **received-ligand kernel: narrow + hard cutoff + row-chunk** | 57 GB → ~1.5 GB @100k; also a paper-correctness fix (hard cutoff at `radius`) | `793c096` |
| 6 | **get_betas batched (eval mode)** | fixes GPU OOM; ~1e-7 β change (correct) | ⏳ in progress |

### Still to make (ranked by impact for your data)
1. **CU-5c — stream the spatial-maps tensor** (the 20 GB@100k / 59 GB@300k ceiling). Instead
   of materializing `adata.obsm['spatial_maps']` (N × C × 64²) up front, generate each cell's
   64×64 map **on the fly inside the DataLoader** (per batch), and drop the cached tensor.
   Moderately invasive (touches `xyc2spatial`, `RotatedTensorDataset`, `init_data`); must be
   behavior-preserving. **This is what unblocks 300k–1M.** *(Needed once you go past ~150k on a
   64 GB node.)*
2. **received-ligand: restrict L to the ligands actually used** (only ligands that appear in
   some gene's modulators, or only metabolite exporters if we go metabolite-only) — cuts the
   O(N·nbr·L) precompute. Cheap; big if L drops 10×.
3. **betadata dtype/format** — write float32 (or float16) parquet; only nonzero-column filtering
   is done today. ~2–4× smaller/faster I/O.
4. **1M-only:** the received-ligand cutoff is now `radius` (tractable), but at 1M the KDTree
   query itself is the cost — the row-chunking already bounds memory; if time matters, cap
   neighbors or coarsen. Lower priority than CU-5c.
5. **MAGIC** — Foster reports tractable at 100k; revisit only near 1M.

---

## 4. Q: What does "only metabolites" (drop TF / L–R / L–TF) buy?

Training **only** on metabolite modulators shrinks the **wide** axis (M, L), not the tall one:

**Saves (roughly proportional to the shrink):**
- **GroupLasso + design matrix:** M drops ~1121 → dozens (10–50×) → the lasso and the
  `train_df` (N × M) shrink ~proportionally. Real per-gene speedup + less RAM.
- **received-ligand precompute:** L drops ~1000 → ~100 transporter exporters (~10×) → the
  O(N·nbr·L) term and the received-ligand memory drop ~10×.
- **Per-gene modulator assembly:** `init_ligands_and_receptors` (CellChat + NicheNet) is
  skipped entirely → faster per-gene setup.
- **betadata size:** N × M → ~10–50× smaller parquet files (nice for the coefficient-reading
  workflow — fewer columns to load and aggregate).

**Does NOT save (the tall axis / the scaling wall):**
- **spatial-maps tensor** (N·C·d²) — independent of M. Still 20 GB@100k.
- **CNN train + get_betas** — the conv runs on the per-cell 64×64 grid; cost is independent of
  M (only the final MLP output width M+1 shrinks marginally). So the GPU cost and the get_betas
  OOM are ~unchanged by going metabolite-only.
- MAGIC, create_spatial_features — cell-driven, unchanged.

**Bottom line:** metabolite-only is a solid **constant-factor** win (maybe ~2–5× on total
training time, and much smaller outputs), and it's exactly what our project wants scientifically
(D6/D7). But it does **not** change the asymptotic scaling — the N-driven spatial-maps + CNN
costs remain, so you still need CU-5c for very large N. (It also composes cleanly with gene
focus: few genes × few modulators.)

---

## 5. Q: The CNN-over-a-grid — how does it scale with sample size?

The β's come from a **Spatial Proximity Grid (SPG) → CNN → MLP → β** per cell (paper Methods,
"Spatially Weighted Regression": the 2D map of cell locations is split into "uniform grids with
a predefined resolution (64×64 by default)"). Key point: **the grid is a fixed d×d = 64×64
representation of each cell's neighborhood, and d is a hyperparameter independent of N.**

Consequences:
- **Per-cell CNN cost is constant** (fixed 64×64 conv, regardless of how many cells exist).
- **Total CNN cost is O(N · E · const) — linear in N.** Asymptotically fine; there is no
  quadratic blowup in the CNN itself.
- The problems are **engineering, not complexity:** (a) materializing all N grids at once
  (memory, CU-5c) and (b) the unbatched get_betas forward (the OOM we're fixing). Both are
  solved by batching/streaming, not by changing the model.
- **Should the grid grow with N? No.** More cells means more per-cell maps, not finer maps. The
  neighborhood radius and grid resolution are modeling hyperparameters set independently of N.
  (Caveat: at extreme density a 64×64 grid could under-resolve a very crowded neighborhood —
  that's a resolution/modeling choice, not a scaling requirement; the paper fixes it at 64.)
- The **cell-type info** you asked about enters two ways, both cheap and N-linear: (i) the SPG
  has one channel per cell type (the C in N·C·d² — this is why more cell types cost more
  *memory*, linearly), and (ii) `spatial_features` (per-cell neighbor counts per type) feed a
  small MLP added to the CNN output. Neither is a scaling wall; C just multiplies the
  spatial-maps memory linearly (another reason CU-5c matters when C is large).

So: the CNN is appropriately **linear in cells**; the work to scale is memory-streaming
(CU-5c) + batching (done), not a rethink of the model.

---

## 6. Learnings to keep (from the 112k melanoma run, 2026-07-17)
- Real reference dataset: **112,551 cells × 5,006 genes**, ~11–15 leiden clusters, on a **10.5 GB
  GPU** (GTX 1080 Ti). `fit()` was run **in the notebook kernel** (not via `spawn_worker`), so it
  used that node's GPU/RAM.
- Setup ("a while to start training") = MAGIC + CellOracle + the one-time received-ligand /
  spatial-maps builds on the first gene.
- **The crash was a GPU OOM in `get_betas`** (unbatched CNN forward per cluster), NOT in the
  received-ligand path (the `793c096` fix held). Fixed by batching in eval mode.
- **`get_betas` used train-mode BatchNorm** (full-cluster stats), inconsistent with `predict()`
  (eval). Eval-vs-train β difference measured at **max abs 1.19e-7 / mean 5.6e-10** — negligible;
  eval mode is correct and batch-independent.
- The **spatial-maps tensor is the next ceiling** (~20 GB@100k) — CU-5c.
