# 07 — Plan: per-cell "neighborhood" analysis OOMs at ≥600k cells (fix design)

> **Status:** ✅ **IMPLEMENTED 2026-07-21 as CU-E — Option B** (dev/review loop, Opus reviewer).
> Diagnosed/written 2026-07-20; carried out the next day. The two-pass gene-pair + metabolite
> chunking now lives in `compute_interacting_cell_scores_lowmem`'s `np` branch; local CPU
> equivalence vs vendored stock is bit-for-bit across all chunk sizes (see §10 below).
> **GPU/Savio gate still pending** (§6 CUDA caveat) — run `validate_lowmem_savio.py` at ≥600k
> cells before a production run.
> **Prereq reading:** `05_harreman_reference.md` §5 + §5c (the whole harreman OOM story), and
> the three drop-ins in `metab_processing/Harreman/cell_communication_lowmem.py`. This plan is
> the **next** memory bottleneck after the CU-A–D aggregate work — a *different* function.
>
> **Why Option B (not A):** Foster chose B — our wall is **GPU** memory, not RAM, and B is
> simpler (preserves the exact `uns` contract; no `summarize_nbhd_scores` refactor). B bounds
> GPU to `(n_cells, chunk)` while still storing the full `(n_cells, n_gp)`/`(n_cells, n_m)`
> matrices on **CPU** — accepted, since CPU RAM is not the constraint here.

---

## 1. Symptom
`HarremanRunner.run_harreman` (via `nbhd_scores.compute_nbhd_scores`) **OOMs on the GPU for
large datasets (≈600k cells and up)**. The aggregate CCC fix (CU-B/C) and the M-axis removal
(CU-A) are both in place and working — this is the *per-cell* neighborhood scores hitting a
**different** wall.

## 2. Which function, exactly
`nbhd_scores.compute_nbhd_scores(adata, M, seed)` calls
`compute_interacting_cell_scores_lowmem(adata, test='non-parametric',
restrict_significance='both', compute_significance='non-parametric', M=M, seed=seed)`
(`metab_processing/Harreman/nbhd_scores.py:16-27`). So **only the non-parametric (`np`) path
runs** — the parametric path is never entered here. The fix only needs to handle the `np` path.

The results are read back by `nbhd_scores.summarize_nbhd_scores` from
`adata.uns['interacting_cell_results']['np'][grain]` (`grain ∈ {'gp','m'}`), using the three
arrays `cs`, `pval`, `FDR` — each of shape **`(n_cells, n_interactions)`** — to compute a
per-(cell-type, interaction) summary (`sig = (FDR<alpha) & (cs>0)`, then groupby cell type).

## 3. Root cause (memory profile)
CU-A removed the catastrophic `(n_cells, n_pairs, M)` permutation-null arrays. But the
`np`-path **still holds several dense `(n_cells × n_gene_pairs)` float64 tensors on GPU**, and
those alone OOM at Xenium scale. In `cell_communication_lowmem.py`
(`compute_interacting_cell_scores_lowmem`, `np` branch):

| Array | Line | Shape | Lifetime |
|---|---|---|---|
| `cs_gp` (observed) | ~522 | `(n_cells, n_gp)` | persistent |
| `x_gp_a`, `x_gp_b` (exceedance accumulators) | 635–636 | `(n_cells, n_gp)` each | persistent across M |
| `cs_m`, `x_m_a`, `x_m_b` | ~529, 637–638 | `(n_cells, n_m)` each | persistent |
| `counts_1`, `counts_2` | ~460 | `(n_gp, n_cells)` each | persistent |
| `c2_perm_a/b`, `WX2t_a/b`, `WtX2t_a/b`, `cs_a`, `cs_b`, `cs_m_a/b` | 646–666 | `(n_cells, n_gp)` / `(n_gp, n_cells)` / `(n_cells, n_m)` | transient, **per permutation** |

At **600k cells × ~136 sig pairs × 8 bytes ≈ 0.65 GB per `(n_cells, n_gp)` array**. There are
~5 persistent + ~4 live transients of that size (plus the smaller metabolite twins and
`counts`), so peak ≈ **6–9 GB**, which OOMs the observed **10.57 GiB** Savio GPU (fragmentation
makes it worse). At 1M cells it is ~11–15 GB — hopeless without chunking. **Note: lowering `M`
does NOT help** — the lowmem arrays are already M-independent (that was CU-A's whole point).

## 4. Why this is harder than the aggregate (CU-B/C) fix
CU-B/C chunked the **gene-pair axis** and it was clean because the aggregate score reduces over
cells (`.sum(0)` → `(n_gp,)`), so a gene-pair block's result is a tiny vector. **Here the
per-cell axis IS the output** (`(n_cells, n_gp)`), and two things couple across a naive
gene-pair chunk:

1. **Metabolite scores sum a metabolite's gene pairs** (`cs_m[:, metab] = Σ_pairs cs_gp[:, pair]`),
   and the pair↔metabolite map is **many-to-many** (05 §4a: 139/416 pairs serve >1 metabolite).
   The metabolite permutation exceedance `x_m += (cs_m_a > cs_m)` needs the **full** per-cell
   `cs_m_a` for each permutation — i.e. all of a metabolite's pairs summed — so you cannot finish
   a metabolite inside one arbitrary gene-pair block.
2. **FDR is BH over the full flattened `(n_cells × n_interactions)` matrix**
   (`multipletests(pvals.flatten(), 'fdr_bh')`), so per-element FDR needs **all** p-values before
   thresholding — you can't BH a chunk in isolation.

## 5. Proposed fix

### Option A (recommended for scale) — stream to the per-cell-type summary, never store the full matrices
The only consumer at scale is `nbhd_scores.summarize_nbhd_scores`, which reduces the
`(n_cells, n_interactions)` matrices to a **per-(cell-type, interaction)** table. Compute that
summary *incrementally* and never materialize/store the full per-cell arrays:

- **Gene-pair pass** — chunk the gene-pair axis (outer chunk, inner M perm, `torch.manual_seed(seed)`
  re-issued per chunk to replay the identical `idx` sequence — the exact CU-B/C RNG trick). For each
  gene-pair chunk: compute observed `cs[:, chunk]` and the exceedance `x_gp[:, chunk]`; derive
  `pval[:, chunk]`; **move that chunk to CPU**. GPU only ever holds `(n_cells, chunk)`.
- **BH**: accumulate the per-chunk `pval` columns into one CPU `pval` matrix `(n_cells, n_gp)`
  (float32 is fine here), then run BH **once** over the full flattened array → `FDR`. (Holding a
  single CPU `pval` matrix is ~0.65 GB at 600k — acceptable in RAM; the GPU never holds it.)
- **Summary**: fold each chunk (or the final CPU matrices) directly into the per-cell-type
  accumulators `summarize_nbhd_scores` needs (`n_sig`, `Σcs`, `Σcs_sig`, `Σ-log10 pval`, counts)
  by cell type — never storing the `(n_cells, n_gp)` matrices in `uns`.
- **Metabolite pass** — chunk the **metabolite** axis (outer chunk, inner M perm, re-seed per chunk).
  For each metabolite chunk, gather the union of its gene pairs, recompute those pairs' per-perm
  `cs_a`, sum per metabolite → `cs_m_a[:, metab_chunk]`, accumulate `x_m`, derive `pval`/`FDR`,
  fold into the metabolite summary. Bounded to `(n_cells, metab_chunk)` + `(n_cells, |pairs in chunk|)`.
  (Yes, this recomputes gene-pair `cs_a` — ~2× the permutation matmuls total. Acceptable; stock OOMs
  outright.)

**Pros:** bounds BOTH GPU (to `(n_cells, chunk)`) and CPU RAM (to ~1 pval matrix, not 5+ full
matrices). Cleanest at Xenium scale. **Con:** changes what's stored in `uns` — but the **final
`nbhd_scores` CSVs are identical**. Would need a `summarize`-side tweak to consume the streamed
accumulators (keep the current path for small data; add a streaming path for large).

### Option B (preserve the exact `uns` contract) — two-pass gene-pair + metabolite chunking, still store full matrices
Same two-pass chunking as A, but assemble and store the full `(n_cells, n_gp)` / `(n_cells, n_m)`
`cs`/`pval`/`FDR`/`cs_sig_*` arrays in `uns` (on CPU) exactly as today. Bounds **GPU** memory but
**not** CPU RAM (still ~several GB of stored per-cell matrices at 600k, ~5+ GB at 1M). Choose this
if something other than `summarize_nbhd_scores` needs the raw per-cell matrices. Bit-for-bit
identical to the current function.

### Option C (quick partial win, if A/B are too big a lift now)
Chunk **only the per-permutation transients** (`WX2t_a`, `cs_a`, …) over gene-pair sub-blocks
inside the existing perm loop, accumulating `cs_m_a` per permutation across sub-blocks. This bounds
the transient `(n_cells, n_gp)` arrays but leaves the **persistent** `cs_gp`, `x_gp_a`, `x_gp_b`
`(n_cells, n_gp)` — so it only cuts ~40% of peak (roughly `600k → ~1M` headroom, not a real fix).
Bit-identical, small diff. A stopgap, not the destination.

**Recommendation:** **Option A.** It's the only one that also solves the CPU-RAM growth (which
Option B just defers to ~1M cells), and per-cell matrices for 1M×150 aren't otherwise consumed.

### Adaptive chunk size
Reuse the CU-B/C convention: `gene_pair_chunk_size` / `metabolite_chunk_size` params defaulting to
`max(1, ELEMENT_BUDGET // n_cells)` (~50M-element budget). Thread through `nbhd_scores` →
`HarremanRunner`.

## 6. Bit-identity requirements (the bar, per Foster)
- **Non-parametric `cs`/`pval`/`FDR` must be bit-for-bit identical** to the current lowmem
  (== stock) on CPU, for the same seed, across all chunk sizes. This is the production-gating path.
- The **re-seed-per-chunk RNG replay** must reproduce the exact `idx_0..idx_{M-1}` sequence
  (verified pattern from CU-B/C — only `torch.randperm` consumes RNG in the loop).
- The **metabolite recompute** (Option A/B pass 2) must yield `cs_m` identical to the single-pass
  `compute_metabolite_cs(cs_a, gene_pair_dict, interacting_cell_scores=True)`.
- **BH once over the full assembled p-value matrix** — never per-chunk.
- **GPU caveat (05 §5c):** on CUDA, `.sum` reductions can reorder by width, so `cs`/`perm` may pick
  up ULP drift and, at measure-zero, flip one integer exceedance vs an unchunked GPU run. Validate
  on Savio; the non-parametric path stayed exact in the CU-D Savio run, but re-check here.

## 7. Testing plan
- Extend the fake-harreman suite (`tests/test_cell_communication_lowmem.py`): assert the **new
  chunked** per-cell `np` path == the **current** `compute_interacting_cell_scores_lowmem`
  (== vendored stock) **bit-for-bit** on `cs`/`pval`/`FDR`/`cs_sig_*`, across
  `gene_pair_chunk_size ∈ {1, 2, n_gp, None}` and `metabolite_chunk_size ∈ {1, 2, n_m, None}`.
- **Metabolite-spanning-chunks** case: a metabolite whose gene pairs land in different chunks, and a
  gene pair shared by ≥2 metabolites (many-to-many), must still match.
- For **Option A**: assert the streamed `summarize_nbhd_scores` output DataFrame is identical to the
  current store-then-summarize path on a small fixture.
- Memory-shape guard: assert no `(n_cells, n_gp)` tensor with `n_gp > chunk` is allocated on device
  (e.g. spy on `torch.zeros`/`sparse.mm` widths, as CU-B/C did).
- Savio real-data gate: extend `validate_lowmem_savio.py` to exercise the chunked per-cell function
  at ≥600k cells and confirm it no longer OOMs and matches (non-parametric exact).

## 8. Effort / risk
This is the **hardest** unit in the harreman memory work — per-cell output + many-to-many metabolite
coupling + BH-over-full-matrix + (Option A) a `summarize` refactor. Use the dev/review loop with an
**Opus reviewer** (numeric + structural). Budget more than CU-B/C. Land Option C first only if a
production run is blocked and A/B can't be finished in time.

## 9. File/line anchors for the implementer
- `metab_processing/Harreman/cell_communication_lowmem.py` — `compute_interacting_cell_scores_lowmem`
  `np` branch (~505–680); the offending allocations at lines ~522, 635–638, 646–666.
- `metab_processing/Harreman/nbhd_scores.py` — `compute_nbhd_scores` (16–27, the call);
  `summarize_nbhd_scores` (29–63, the consumer to stream into for Option A).
- CU-B/C chunking + re-seed-per-chunk reference implementation: same file,
  `_run_cell_communication_analysis_lowmem` (~714+) and `_run_ct_cell_communication_analysis_lowmem`
  (~1308+).
- Stock reference: `DataForClaude/cell_communication.py` `compute_interacting_cell_scores`
  (1473–1905); fake-harreman vendored copy in `tests/fixtures/fake_harreman/harreman/tools.py`.

## 10. What was actually implemented (CU-E, 2026-07-21)
**Design shipped = Option B, two passes**, in `compute_interacting_cell_scores_lowmem`'s `np`
branch (old block commented `# OLD:`, new block `# NEW (CU-E, Option B ...)`, git-record style):
- **Pass 1 (gene-pair chunks):** outer loop over `gene_pair_chunk_size`-wide slices, inner M
  perms, `torch.manual_seed(seed)` re-issued per chunk (the proven CU-B/C RNG replay). GPU holds
  only `(n_cells, chunk)`; observed `cs` + exceedance counters `x_gp_a/b` assembled into full
  **CPU** `(n_cells, n_gp)` arrays.
- **Pass 2 (metabolite chunks):** outer loop over `metabolite_chunk_size` metabolites; per chunk
  gather the **union** of their gene pairs (many-to-many), recompute that union's per-perm gene-pair
  scores, reduce to per-metabolite via a **remapped `sub_dict`**. ~2× the permutation matmuls
  (sanctioned). Observed `cs_m` and its exceedance threshold are both computed **on `device`**
  (see the review-caught bug below). BH run **once** over each full flattened CPU p-value matrix.
- **Params:** `gene_pair_chunk_size` + `metabolite_chunk_size` on the drop-in (adaptive default
  `max(1, 50M // n_cells)`), threaded through `nbhd_scores.compute_nbhd_scores` **only** —
  `HarremanRunner`/`harreman_funcs.py` deliberately untouched (Foster's call on param surface).

**The one real bug the loop caught (Opus review) — a device seam, invisible to CPU tests.** The
first dev pass computed the observed/stored `cs_m` with a **CPU-side** `compute_metabolite_cs`,
then used that CPU value as the exceedance threshold against **GPU**-computed permutation `cs_m`.
`compute_metabolite_cs` does `cs_gp[:, idx].sum(dim=1)`; for a metabolite with **≥3 gene pairs**
(real ones reach 91), a CUDA tree/FMA reduction can differ from the CPU sum by 1 ULP → stored `cs`
no longer bit-exact vs stock **and** an `x_m` exceedance can flip when a perm score sits within
1 ULP of the threshold. Fixed by computing the observed `cs_m` **on `device`** (recompute the
union's observed gene-pair `cs` on GPU, reduce with `compute_metabolite_cs` on GPU) for both the
stored value and the threshold — no CPU↔GPU reduction boundary anywhere on the metabolite path.
`use_p_shortcut` (center=True, both) now **copies** the parametric `cs_m` like stock, not recompute.
**Lesson for the next CUDA-numerics unit:** bit-identity requires not just the same *math* but the
same *device+reduction*; a CPU-only test suite cannot see a CPU-vs-GPU reduction-order divergence.

**Second device-parity bug — caught by the real Savio GPU run, not the review (2026-07-21).** The
first Savio validation showed per-cell `cs` **exactly** bit-identical (0.0, both grains) but
`pval`/`FDR` off by **exactly `5.96e-08 = 2⁻²⁴`** (0.5 float32-ULP at magnitude ~1.0), both grains;
the two aggregates were exact. Fingerprint analysis (not a guess): a 1-count exceedance flip would
move a p-value by `1/(M+1)` (~5e-3 at M=200), *five orders larger* than observed — so the integer
counts `x` were identical, isolating the cause to the float32 **division** `(x+1).float()/(M+1)`.
The chunked code accumulated `x` to CPU and did the cast+divide **on CPU**, under an explicit (wrong)
comment that "elementwise cast/divide is device-independent." It is not: **CUDA and x86 float32
division can disagree by up to 1 ULP** for values in `[0.5,1.0)` (both IEEE-754-correct; PyTorch folds
`/201` differently per backend). `+`, `.to(float32)`, and `torch.where` *are* device-invariant — only
`/` is not. Fix: do the p-value divide **on `device`, per chunk** (elementwise → chunk-width GPU
division is bit-identical to stock's full-width GPU division; only *which device rounds* matters),
then move the finished p-value chunk to CPU; BH stays once on CPU (stock does BH on CPU too). Note the
**mis-diagnosis worth remembering**: the `2⁻²⁴` fingerprint first read as a float64-vs-float32 *dtype*
mismatch (was our vendored reference stale vs installed harreman?) — checking installed harreman
(v0.1.4) showed its per-cell pval uses `.float()` too, *ruling that out* and pointing to the
CPU-vs-GPU division. **Broadened lesson:** "same device" applies to float32 **division**, not just
reductions — and confirm a numeric fingerprint against the *installed* package before assuming a
reference is stale.

**Testing (Tier-0, `spacetravlr_env`, no real harreman — fake fixture):** `tests/test_cell_
communication_lowmem.py` now sweeps `gene_pair_chunk_size ∈ {1,2,n_gp,None}` × `metabolite_chunk_
size ∈ {1,2,n_m,None}` vs **true stock**, incl. a metabolite spanning gene-pair chunks, a pair
shared by ≥2 metabolites, a ≥3-pair metabolite, and a `center=True` (shortcut + standardize) sweep;
plus a `sparse.mm`-width memory-shape guard with a positive control. Full comm suite: **60 passed /
636 subtests**. Ad hoc dev sweep: 23,040 checks vs stock, 0 mismatches (all CPU). **GPU exactness
can't be proven by the CPU suite** — the `validate_lowmem_savio.py` gate (section [3/3], now forces
small nbhd chunks via `--nbhd-gp-chunk-size`/`--nbhd-m-chunk-size`) is the real proof.

**Savio validation status (2026-07-21):** first GPU run (8k cells, M=200) — aggregates non-parametric
EXACT; per-cell `cs` EXACT but `pval`/`FDR` off by 0.5 float32-ULP → the CPU-vs-GPU **division** bug
above, now **fixed in code**. Local suite still 60-green (the fix is a no-op on CPU). **Re-run the
Savio gate to confirm the per-cell `pval`/`FDR` are now EXACT**, then a ≥600k-cell run for the OOM
proof (stock OOMs, chunked drop-in survives). Until that re-run lands, GPU exactness of the per-cell
path is *expected but not yet demonstrated*.

## 11. CU-F — Pass 2 OOM'd at Xenium scale; union-footprint sizing + auto OOM fallback (2026-07-21)
After the CU-E division fix, a real `run_harr_all.py` run (Human_Breast, ~Xenium scale) OOM'd — but
in **Pass 2 (metabolite chunks)**, not Pass 1. Root cause: Pass 2 sized its chunk by metabolite
**count** (`50M // n_cells` metabolites), but its true GPU footprint is `(n_cells, |union of that
chunk's gene pairs|)`, and pair↔metabolite is **many-to-many** (139/416 pairs serve >1 metabolite,
§4a) — so a chunk of ~60 metabolites referenced nearly the **full `n_gp`**, silently rebuilding the
`(n_cells, n_gp)`-scale tensors Pass 1 chunks to avoid. Pass 2 was effectively unbounded. (Foster runs
all datasets through `run_harr_all.py` with **no per-dataset hand-tuning**, so the automatic default
had to bound this itself.)

**Fix (bit-identity untouched — only chunk boundaries move):**
- **`_greedy_metabolite_chunks`** groups consecutive metabolites so their running gene-pair **union**
  ≤ `max_pairs_per_chunk = max(1, element_budget // n_cells)` — the SAME per-chunk pair budget Pass 1
  uses, so Pass 2 is now memory-symmetric with Pass 1 (if Pass 1 fits, Pass 2 fits). A lone metabolite
  whose own union exceeds the budget still gets its own chunk (its pairs can't be split; worst case
  ~91 pairs → `(n_cells, 91)`, fine). Bit-identical because each metabolite's `cs_m` sums only its own
  pairs via `sub_dict`, independent of chunk composition.
- **New `element_budget` param** (default `50_000_000`) replaces the two hardcoded literals and drives
  both passes. `metabolite_chunk_size` re-semantic'd: now an optional *hard cap* on metabolites/chunk
  **on top of** the union budget (`None` = fully automatic).
- **Both Pass-2 axes bounded** (review hardening): besides the gene-pair *union* (the `(n_cells,|union|)`
  tensors), the metabolite-*count* axis (`x_m_*`/`cs_m_*` = `(n_cells,|metabs in chunk|)`) is also
  capped — when `metabolite_chunk_size=None` an implicit `hard_cap = max_pairs_per_chunk` is applied, so
  a heavy-sharing dataset (many metabolites, tiny shared pair set — union stays small but metab count
  grows) can't reach `(n_cells, n_m)`. Both caps derive from `max_pairs_per_chunk`, so the OOM-halving
  retry shrinks both. Explicit `gene_pair_chunk_size`/`metabolite_chunk_size` overrides are now
  **clamped** to `max_pairs_per_chunk` (recomputed per retry) so a pinned chunk still benefits from the
  halving; `element_budget=None` coalesces to the default; `max_oom_retries<=0` still runs exactly once.
- **Inter-pass cleanup:** null Pass-1 last-chunk GPU locals + `torch.cuda.empty_cache()` so Pass 2
  doesn't start on a nearly-full card (the OOM showed 9 GiB still resident at Pass 2 entry).
- **Automatic OOM fallback (the "no hand-tuning across datasets" piece):** the np body is a closure
  `_run_non_parametric_pass(cur_element_budget)` that fully re-inits `uns['…']['np']`; a retry loop
  catches a CUDA OOM, `empty_cache()`s, **halves `element_budget`**, and retries up to `max_oom_retries`
  (4) before re-raising. Non-OOM exceptions propagate. So a too-big dataset auto-shrinks instead of
  failing the batch.
- **Knobs threaded onto `HarremanRunner`** (`gene_pair_chunk_size`/`metabolite_chunk_size`/`element_budget`,
  all default `None` → automatic) → `compute_nbhd_scores` → drop-in — a manual escape hatch that
  Foster shouldn't need. (This closed a pre-existing gap: `HarremanRunner` never forwarded chunk sizes
  to the nbhd path at all.)

**Testing:** target comm-test file **31 passed** (from 24), full comm+wiring+agg+ct **77 passed / 649
subtests**; full repo suite **241 passed** (only the known-unrelated `test_spawn_worker` red). New:
tiny-`element_budget` bit-identity sweeps vs **true stock** forcing multi-metabolite greedy chunks with
spanning/shared pairs AND a `make_heavy_sharing_test_adata` (10 metabolites, 1 shared pair — proves the
metabolite-count-axis bound); a deterministic `_greedy_metabolite_chunks` unit-test class; `sparse.mm`-
width + metabolite-count memory guards (non-vacuous, positive controls); and OOM-fallback tests
(Pass-1 AND Pass-2-surfacing OOM, multi-halving-then-success, `max_oom_retries∈{0,1}` still populates,
pinned-`gene_pair_chunk_size` clamp, non-OOM propagates, exhaustion re-raises). Ad hoc sweep: **25,600
checks vs stock, 0 mismatches** (Opus review added 480 more + 2000 greedy-fuzz cases, also clean).
**Opus-reviewed: no blockers/majors.** **Savio re-run still pending** — the same gate confirms both the
CU-E division fix (per-cell `pval`/`FDR` EXACT) and that CU-F no longer OOMs at ≥600k cells.
