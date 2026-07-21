"""Memory-safe drop-ins for three harreman `cell_communication.py` functions.

This module holds hand-maintained low-memory replacements for harreman's
`cell_communication.py` functions:

  1. `compute_interacting_cell_scores_lowmem` (CU-A) -- the per-cell / non-ct function.
  2. `compute_cell_communication_lowmem` (CU-B) -- the cell-type-INDEPENDENT aggregate
     function (no per-cell axis in its outputs, but still OOMs via dense
     `(n_cells, n_gene_pairs)` intermediates at ~1M cells).
  3. `compute_ct_cell_communication_lowmem` (CU-C) -- the cell-type-AWARE twin of CU-B
     (stratifies scores by cell-type pair; same OOM mechanism, one extra ct-pair axis).

All three are formatted as an ANNOTATED DIFF against the stock source
(`DataForClaude/cell_communication.py`) so a harreman maintainer can see exactly what to
change: each function body follows the stock structure closely, and the block that
actually differs is shown as commented-out `# STOCK:` lines immediately followed by the
replacement. Everywhere else the code is behavior-equivalent to stock, with small local
refactors (`_prep_counts_1_2`, `_write_sig_masks` helpers, the "[lowmem]" log prefix).

------------------------------------------------------------------------------------
CU-A: `compute_interacting_cell_scores_lowmem`
------------------------------------------------------------------------------------
WHY: the stock non-parametric (permutation) path allocates, on GPU, the *entire*
permutation null with a per-cell axis:

    perm_cs_gp_a, perm_cs_gp_b : (n_cells, n_gene_pairs, M)
    perm_cs_m_a,  perm_cs_m_b  : (n_cells, n_metabolites, M)

At ~1e5-1e6 cells x ~1e2 pairs x M=1000 this is 10s-100s of GB per array -> OOM.
See DataForClaude/documentation/05_harreman_reference.md sec.5.

WHAT CHANGED (diff summary -- see the `# STOCK:` block in the non-parametric section
for the literal lines being replaced):
  * The permutation loop no longer stores each permutation. It accumulates the
    exceedance counters incrementally:
        x = sum_over_perms( perm_cs > observed_cs )
    which is exactly what the stock code computed *after* the loop from the giant
    arrays. Peak memory drops from O(cells * pairs * M) to O(cells * pairs).
  * The raw ``perm_cs_a/perm_cs_b`` arrays are NO LONGER written to
    ``adata.uns[...]['np'][...]`` (they were only ever used to derive p-values, and
    storing them is itself huge). Everything downstream (``pval``, ``FDR``, ``cs``,
    ``cs_sig_pval``, ``cs_sig_FDR``) is unchanged and still written.
  * ``check_analytic_null=True`` is rejected for the non-parametric / both paths (it
    re-introduces the (cells,pairs,M) arrays and also hits a latent NameError in the
    stock code); with test='parametric' it is ignored, exactly as stock does. Foster
    calls it with False, so this is a no-op for current usage.
  * The parametric ('p') path is reproduced verbatim (it has no M axis / no OOM).

Numerics are identical to stock for the same seed: all heavy lifting still uses
harreman's own helper functions, imported from the installed package at runtime (below),
and the permutation p-values replicate stock's float32 cast of the exceedance count
(`(x + 1).float() / (M + 1)`) exactly -- so `cs`, `pval`, `FDR` and `cs_sig_*` are all
bit-for-bit equal to stock. See `tests/test_cell_communication_lowmem.py`.

------------------------------------------------------------------------------------
CU-E: per-cell (`np`) gene-pair + metabolite chunking (Option B, 2026-07-21)
------------------------------------------------------------------------------------
WHY: the CU-A fix above removed the `(cells, pairs, M)` permutation-null axis, but the
`np` branch still built the observed `cs`/exceedance-counter arrays as FULL, persistent
`(n_cells, n_gp)` / `(n_cells, n_m)` GPU tensors. At Xenium scale (>=~600k cells) those
alone OOM the GPU (see `05_harreman_reference.md` sec.5, diagnosed further in
`07_nbhd_percell_chunking_plan.md`). Unlike CU-B/C (whose aggregate score reduces over
the cell axis, `.sum(0)` -> a tiny `(n_gp,)` vector), here the **per-cell axis is the
output** -- `(n_cells, n_gp)` can't be reduced away, and a metabolite's gene pairs are
**many-to-many** (a pair can serve >1 metabolite) and can span multiple gene-pair chunks,
so a metabolite's exceedance count can't be finished from one arbitrary chunk alone. Per
Foster's decision, this implements **Option B** (`07_nbhd_percell_chunking_plan.md` sec.5B):
bound GPU memory to `(n_cells, chunk)` while still assembling and storing the exact same
full `(n_cells, n_gp)` / `(n_cells, n_m)` arrays in `uns` on CPU -- the `uns` contract is
unchanged; only GPU memory is bounded, not CPU RAM (intentional -- the wall is GPU).

WHAT CHANGED (see the `# OLD:` / `# NEW (CU-E...)` markers in the `np` branch): the
single-pass, full-array code is commented out wholesale (`# OLD:`) and replaced by a
**two-pass, chunked** implementation, mirroring the proven CU-B re-seed-per-chunk pattern:

  * **Pass 1 (gene-pair chunks).** Outer loop over gene-pair chunks (`gene_pair_chunk_size`,
    same adaptive-default convention as CU-B/C: `max(1, 50_000_000 // n_cells)`), inner
    loop over the `M` permutations, with `torch.manual_seed(seed)` re-issued at the START
    of each chunk. Only `torch.randperm` consumes RNG in the loop body, so re-seeding per
    chunk replays the *identical* `idx_0..idx_{M-1}` sequence every chunk would have drawn
    in one pass -- the same argument CU-B/C already proved. Each chunk's observed `cs` and
    exceedance counters (`x_gp_a`/`x_gp_b`, accumulated in float64 -- exact for integer
    counts) are computed as `(n_cells, chunk)` tensors and moved to CPU into pre-allocated
    full `(n_cells, n_gp)` numpy arrays; GPU never holds more than one chunk's worth.
  * **Pass 2 (metabolite chunks).** Outer loop over metabolite chunks
    (`metabolite_chunk_size`, same adaptive default), re-seeded per chunk identically to
    Pass 1. For each chunk, the UNION of gene-pair indices needed by that chunk's
    metabolites is gathered (handles many-to-many: a pair used by 2+ metabolites in the
    same chunk is only computed once), a remapped `sub_dict {metab: [positions within the
    union]}` is built, and that union's per-permutation gene-pair scores are RECOMPUTED
    (not read back from Pass 1 -- a metabolite's pairs may span >1 gene-pair chunk, so no
    single Pass-1 chunk has the full picture) via the identical arm-a/arm-b formula, then
    summed into per-metabolite scores with `compute_metabolite_cs(..., sub_dict, ...)`.
    This is a SANCTIONED ~2x recompute of gene-pair scores (plan sec.5) -- the alternative
    (holding a per-permutation `(n_cells, n_gp)` array to defer the metabolite sum, as
    CU-B does for its tiny `(n_gp, M)` case) is exactly the OOM this fix removes. GPU is
    bounded to `(n_cells, metabolite_chunk)` + `(n_cells, |union|)`.
  * **Observed `cs_m` is computed INSIDE Pass 2, per chunk, ON `device`** (fixed after
    review -- see below), structured exactly like Pass 1's observed `cs_gp_c`: build the
    union's observed gene-pair scores on `device` with NO permutation
    (`(counts_1u.T*WX2t_u)+(counts_1u.T*WtX2t_u)`, `same_gene_union` halving applied), then
    reduce with `compute_metabolite_cs(..., sub_dict, ...)` -- also on `device`. That same
    on-device `cs_m_c` tensor is used BOTH as the stored observed value (moved to CPU into
    the assembled `(n_cells, n_m)` array) AND as the exceedance threshold for the
    permutation arms immediately below it, so the comparison `cs_m_a_chunk > cs_m_c` always
    happens between two device-resident tensors produced by the identical reduction kernel.
    **Why this matters (bug found in review, fixed):** an earlier version of this pass
    computed the FULL `cs_m` ONCE via a CPU-side `compute_metabolite_cs` call (reusing the
    assembled CPU `cs_gp`) and used that CPU value as the threshold against the
    GPU-computed permutation scores. On CPU that is harmless (both sides reduce on CPU in
    this local/CI environment), but on a real CUDA `device` it reintroduces exactly the
    kind of device asymmetry CU-B's docs warn about: `.sum(dim=1)` over a metabolite's
    gene-pair columns can round differently between a CPU and a CUDA reduction kernel for
    >= 3 pairs (real metabolites go up to 91), which would (a) make the stored `cs_m` not
    bit-identical to stock on GPU, and (b) risk an `x_m` exceedance flip whenever a
    permutation score lands within 1 ULP of the threshold. Computing `cs_m_c` on `device`
    for both roles removes that seam entirely -- single-chunk now reduces to stock's GPU
    computation for the `m` grain exactly as it already did for the `gp` grain.
  * **`center_counts_for_np_test and test == "both"` (the "p"-shortcut) copies, for BOTH
    grains.** Stock's shortcut sets `np_gp['cs']`/`np_m['cs']` to a COPY of the already-
    computed parametric ('p') `cs`, not a re-derivation from the (possibly re-standardized)
    'np' counts. Pass 1 already reproduced this for `gp`; Pass 2 mirrors it for `m`
    (`cs_m_cpu[...] = np.asarray(adata.uns[...]['p']['m']['cs'])`), and when significance is
    also wanted, the per-chunk threshold `cs_m_c` is sliced from that COPIED array (moved to
    `device`), not recomputed from `counts_1u`/`counts_2u` -- matching stock's behavior in
    this combination bit-for-bit.
  * **Bit-identity argument (why this is provably exact, not merely close):** each output
    column of `sparse.mm(weights, X)` depends only on that column of `X` -- proven in the
    CU-B section above -- so slicing the gene-pair axis into ANY subset (a contiguous
    Pass-1 chunk, or the arbitrary Pass-2 union) reproduces bit-identical per-column values
    to a full-width computation, given the identical permutation `idx` (guaranteed by the
    re-seed). `compute_metabolite_cs` itself always gathers the relevant columns into a new
    tensor before `.sum(dim=1)` (`cs_gp[:, idx_tensor].sum(dim=1)`), so the reduction width
    for a given metabolite is the SAME (its own gene-pair count) whether that gather comes
    from a full `(n_cells, n_gp)` tensor or a chunk/union subset -- no reduction-order
    sensitivity is introduced (contrast CU-B's float64 `.sum(dim=0)`-over-the-cell-axis ULP
    caveat, which does not apply here since nothing sums over the cell axis in this branch),
    PROVIDED (as fixed above) both sides of every `>` comparison are produced by the same
    device's reduction kernel -- there is no longer any CPU/GPU seam in either grain.
  * **BH is run ONCE over each full flattened CPU p-value matrix** (`pval_gp`/`pval_m`),
    never per-chunk -- required for correct FDR (plan sec.6).
  * Single-chunk (`gene_pair_chunk_size >= n_gp` and `metabolite_chunk_size >= n_m`)
    executes the exact same code path as any other chunk count -- there is no separate
    "unchunked" special case, so it is bit-identical by construction, not by a fallback,
    for BOTH grains.

Tests: `tests/test_cell_communication_lowmem.py`'s chunked-equivalence test class sweeps
`gene_pair_chunk_size` x `metabolite_chunk_size` (incl. 1, 2, `n_gp`/`n_m`, and `None`)
against **stock** `harreman.tools.compute_interacting_cell_scores` directly (not just the
unchunked lowmem), asserting `cs`/`pval`/`FDR`/`cs_sig_pval`/`cs_sig_FDR` are bit-for-bit
identical (`np.testing.assert_array_equal`) in every configuration, using fixtures that
exercise a metabolite whose gene pairs land in different chunks and a gene pair shared by
>=2 metabolites (many-to-many). A dedicated sweep also covers
`center_counts_for_np_test=True, test='both'` (the shortcut-copy path for both grains,
plus the chunked `standardize_counts` calls), which the adaptive-default-only fixtures
never previously exercised.

------------------------------------------------------------------------------------
CU-B: `compute_cell_communication_lowmem`
------------------------------------------------------------------------------------
WHY: stock's `run_cell_communication_analysis` (the analysis body behind
`compute_cell_communication`) builds dense `(n_gene_pairs, n_cells)` `counts_1`/`counts_2`
stacks and dense `(n_cells, n_gene_pairs)` `WX2t`/`WtX2t = sparse.mm(weights, counts_2.T)`
intermediates for ALL gene pairs at once. At ~1M cells these OOM even though the *outputs*
of this function have no per-cell axis (`cs_gp` is `(n_gene_pairs,)`, `cs_m` is
`(n_metabolites,)`) -- the observed score is a SUM OVER THE CELL AXIS
(`cs_gp = (counts_1.T * WX2t).sum(0)`), so gene-pair columns can be sliced into chunks
without changing the result: each output column of `sparse.mm` is independent of the
others, and each gene pair's final scalar score sums over all cells exactly as before --
no summation reordering. See `DataForClaude/documentation/05_harreman_reference.md` sec.5
and the CU-B task spec for the full derivation.

WHAT CHANGED (diff summary -- see the `# STOCK:` blocks in `_run_cell_communication_
analysis_lowmem` for the literal lines being replaced):
  * `counts_1`/`counts_2` and the `WX2t`/`WtX2t`/`WX1t`/`WtX1t` intermediates are built
    PER GENE-PAIR CHUNK (`gene_pair_chunk_size`, default adaptive: `50_000_000 //
    n_cells`) instead of for all gene pairs at once. Each chunk's contribution to
    `cs_gp`/`eg2_a`/`eg2_b` (all reduced over the cell axis within the chunk) is written
    into the correct slice of a pre-allocated full-size `(n_gene_pairs,)` tensor. Peak
    memory drops from O(cells * pairs) to O(cells * chunk).
  * `num_umi` (used by `standardize_counts`, per-gene-pair-row DANB) is computed ONCE
    from the full counts matrix and stays global/unchunked -- it does not depend on which
    other gene pairs are processed alongside a given one.
  * The permutation null (`perm_cs_gp_a/b`, `perm_cs_m_a/b`) has NO per-cell axis in
    stock (`(n_gene_pairs, M)` / `(n_metabolites, M)`) so it is kept as-is (not the OOM
    site) -- but building it still requires the same per-chunk `WX2t`/`WtX2t` step, so
    the permutation loop is restructured as OUTER loop over gene-pair chunks, INNER loop
    over the M permutations, with `torch.manual_seed(seed)` re-issued at the START of
    each chunk so every chunk replays the identical `idx_0..idx_{M-1}` permutation
    sequence (only `torch.randperm` consumes RNG inside the loop, so re-seeding per chunk
    reproduces the exact index sequence stock would have drawn for all gene pairs at
    once). Metabolite permutation scores (`perm_cs_m_a/b`) are derived AFTER all chunks
    have filled the full `(n_gene_pairs, M)` `perm_cs_gp_a/b`, by calling
    `compute_metabolite_cs` on each permutation's full `(n_gene_pairs,)` column -- the
    exact same 1-D-vector call stock makes per iteration -- so the result is bit-for-bit
    identical, not merely equivalent in value (see module-level note in
    `_run_cell_communication_analysis_lowmem` for why summing the full column instead of
    a partial one per chunk is required for metabolites whose gene pairs span >1 chunk).
  * `check_analytic_null=True` is rejected (out of scope for this memory fix; the actual
    production caller, `harreman_funcs.py`, never sets it).
  * `get_cell_communication_results` (the small, non-chunked results/DataFrame builder)
    is called VERBATIM from harreman via the `_need(...)` shim -- it only ever touches
    the already-reduced `(n_gene_pairs,)`/`(n_metabolites,)` arrays, so it has no memory
    issue and needed no low-mem twin.

Numerics vs stock, measured (not assumed) across 29+ random seeds in
`tests/test_cell_communication_agg_lowmem.py`:
  * `gene_pair_chunk_size >= n_gene_pairs` (single chunk -- the adaptive default already
    does this on small data): EXACTLY bit-for-bit identical to stock for EVERY stored
    array, no exceptions found. This is expected: with one chunk, the code executes with
    the exact same tensor width/stride as stock's own reduction, i.e. it IS stock's
    computation, just routed through this function.
  * `gene_pair_chunk_size < n_gene_pairs` (real, multi-chunk runs): `cs` (both tests) and
    EVERY non-parametric ('np') output (`pval`, `FDR`, `perm_cs_a/b`) were bit-for-bit
    identical across all tested seeds/chunk sizes -- these are the values production
    (`harreman_funcs.py`) actually gates significance on (`select_significant_
    interactions(test='non-parametric', ...)`). The parametric ('p') `Z`/`Z_pval`/`Z_FDR`
    occasionally (not always) differ from stock by 1-4 ULPs of float64 (~1e-16 to
    2.2e-16 absolute, empirically). Root cause (confirmed by direct isolation, not
    guessed): PyTorch's `.sum(dim=0)`/`.pow(2).sum(dim=0)` reduction over the cell axis
    can select a different internal accumulation order depending on the tensor's WIDTH
    and memory STRIDE/contiguity, even when the per-column values feeding it are
    themselves bit-identical (verified: `sparse.mm`'s per-column outputs ARE always
    bit-exact regardless of chunking -- see the isolation script in the CU-B dev notes).
    This is a property of the reduction kernel, not of this code's logic, so it cannot be
    fixed by restructuring the chunk loop; it is the same reason two mathematically
    equivalent floating-point summations can differ in their last bit. `eg2_a`/`eg2_b`
    (inputs to the Z-score's `std`/division) are where this shows up; `cs_gp` itself
    happened to always match exactly in testing but is not proven immune in the same way
    permutation/`np` values are (no division/sqrt amplifying rounding). Net: treat
    multi-chunk `Z`/`Z_pval`/`Z_FDR` as "identical to float64 rounding" (~1e-14 or tighter
    in practice), not literally bit-for-bit; everything else stays exact. See the test
    file's `_ccc_results_equal` for exactly which keys use `assert_array_equal` vs a very
    tight `assert_allclose`.

CAVEAT: originally written without a local harreman install (harreman runs on Savio).
A fake `harreman` package now exists locally for tests (`tests/fixtures/fake_harreman/`,
CU-A/CU-B) so the equivalence tests run in CI/dev, but full validation on real Savio
data is still recommended before relying on this for a production run.

------------------------------------------------------------------------------------
CU-C: `compute_ct_cell_communication_lowmem`
------------------------------------------------------------------------------------
WHY: stock's `run_ct_cell_communication_analysis` (behind `compute_ct_cell_communication`)
builds dense `(n_gene_pairs, n_cells)` `counts_1`/`counts_2` for ALL gene pairs at once and,
per cell-type pair, a dense `(n_cells, n_gene_pairs)` `WX2t = sparse.mm(weights_ct_pairs[ct_pair],
counts_2.T)` -- the same OOM shape as CU-B, times `n_ct_pairs`. The observed score is
`cs_gp[ct_pair] = (counts_1.T * WX2t).sum(0)`, a SUM OVER THE CELL AXIS, so gene-pair columns
chunk exactly like CU-B (see `05_harreman_reference.md` sec.5 and the CU-B derivation above --
identical argument, just repeated per ct-pair). Unlike the cell-indep path, the ct score is
**not** symmetrized (`W` only, no `+ Wt`) and has no `same_gene_mask` halving -- stock's own
code has neither for the ct case (`05_harreman_reference.md` sec.3: only the sorted-order ct
pair is ever scored).

WHAT CHANGED (see the `# STOCK:` blocks in `_run_ct_cell_communication_analysis_lowmem`):
  * `weights_ct_pairs = create_weights_ct_pairs(...)` (a 3D torch sparse tensor) is built
    ONCE, same as stock -- it is sparse, not the OOM site. Its per-ct-pair 2D coalesced
    slices (`Ws[ct_pair] = weights_ct_pairs[ct_pair].coalesce()`) are ALSO hoisted out and
    built once (stock rebuilds this slice on every use -- inside every ct-pair loop, inside
    every permutation, inside every call); reusing the same coalesced sparse tensor object
    across chunks/permutations is a pure efficiency refactor with no numerics change,
    since `sparse.mm`'s result depends only on the sparse tensor's content.
  * `counts_1`/`counts_2` `(n_gene_pairs, n_cells)` are built PER GENE-PAIR CHUNK from the
    full `counts` `(n_genes, n_cells)`, exactly like CU-B. `num_umi` stays global (computed
    once from the full counts). `cs_gp` `(n_ct_pairs, n_gene_pairs)` and, when `fix_ct=True`,
    `EG2_gp` `(n_ct_pairs, n_gene_pairs)` are pre-allocated once and filled chunk-by-chunk,
    looping cell-type pairs inside each chunk (the ordering the task spec calls for: outer
    gene-pair chunk, inner ct-pair loop).
  * `standardize_ct_counts` -> `center_ct_counts_torch` -> `models.apply_model_per_cell_type`
    (DANB branch): verified (module docstring of the fake-harreman fixture, and directly
    against the real clone) that this is per-gene-ROW independent -- `apply_model_per_cell_type`
    slices CELLS by cell type and calls the same row-wise `danb_model_torch` CU-A/B already
    established is row-independent (`counts.sum(dim=1)`/`.mean(dim=1)` never touch another
    row). So per-chunk standardization of `counts_1c`/`counts_2c` is identical to standardizing
    the full stack at once. The non-parametric path only standardizes when
    `center_counts_for_np_test=True`; `harreman_funcs.py` passes `center_counts_for_np_test`
    unset (its default, `False`), so that branch is chunk-safe trivially (no standardization
    call at all) for the production caller, and chunk-safe by the same DANB argument otherwise.
  * **Permutation (np path), the tricky part:** stock draws ONE cell-type-STRATIFIED
    permutation `idx` per iteration `i` (`torch.randperm` once per unique cell type, in
    `torch.unique(cell_type_labels)` order) when `fix_gp=False` (what `harreman_funcs.py`
    always passes). This `idx` draw depends ONLY on `cell_types` -- never on gene pairs,
    counts, or which chunk is being processed -- so it is IDENTICAL for every chunk. The
    lowmem version restructures as OUTER loop over gene-pair chunks, `torch.manual_seed(seed)`
    re-issued at the START of each chunk, INNER loop over the `M` permutations: re-seeding
    replays the exact `idx_0..idx_{M-1}` sequence stock would draw once for all gene pairs
    (verified: the only RNG consumption inside stock's per-iteration body is the per-cell-type
    `torch.randperm` calls -- `sparse.mm`, `compute_metabolite_cs_ct`, and the `cell_type_labels`
    construction do not touch RNG). `fix_gp=True` (a different, non-stratified single
    `torch.randperm` over all cells, PLUS a full `weights_ct_pairs` rebuild per permutation
    from shuffled cell-type labels -- unrelated RNG shape, doesn't compose with the
    reuse-across-M chunk structure) is rejected up front, mirroring the `check_analytic_null`
    guard pattern from CU-A/B; `harreman_funcs.py` never sets it.
  * `perm_cs_gp` `(n_ct_pairs, n_gene_pairs, M)` and `perm_cs_m` `(n_ct_pairs, n_metabolites, M)`
    have NO per-cell axis in stock, so they are kept full-size (not the OOM site), same as
    CU-B's `perm_cs_gp_a/b`. Metabolite permutation scores (`perm_cs_m`) are derived AFTER all
    chunks have filled `perm_cs_gp`, by calling `compute_metabolite_cs_ct` on a **clone** of
    each permutation's full `(n_ct_pairs, n_gene_pairs)` column. The clone matters: unlike
    `compute_metabolite_cs` (CU-B, no ct axis), `compute_metabolite_cs_ct` MUTATES its input
    in place (zeroing gene-pair columns not relevant to a cell-type-pair whose tested gene-pair
    set is a strict subset of the full set -- `ct_specific_gene_pairs`). Stock is safe from this
    by construction: `perm_cs_gp[:, :, i] = cs_gp` is an index-assignment (a value copy into
    stock's pre-allocated array) that happens BEFORE stock's own throwaway `cs_gp` local is
    passed into `compute_metabolite_cs_ct` and mutated. Passing a bare slice
    (`perm_cs_gp[:, :, i]`, a VIEW under basic indexing) into the same function here would
    corrupt the stored `perm_cs` this code writes to `uns` -- `.clone()` reproduces stock's
    decoupling. The observed (non-permutation) `cs_gp`/`cs_m` follow stock's own order exactly
    (store `.cs` to `uns`, THEN call `compute_metabolite_cs_ct(cs_gp, ...)`), which on CPU
    means the stored `.cs` numpy array (a view sharing memory with the tensor, since
    `.detach().cpu().numpy()` is a no-op copy when already on CPU) reflects the SAME
    ct-specific in-place masking stock's own `.cs` would show for the same reason -- this is
    reproduced faithfully (not defended against), because the goal is bit-identical output,
    not "fixing" a stock quirk. The test fixture's `cell_type_pairs` all reference the FULL
    gene-pair list (not ct-specific), so `ct_specific_gene_pairs` is empty in every test here
    and this code path is exercised structurally (the masking `if` is reached and evaluates
    False) but not with an actual non-empty mask -- flagged for the reviewer.
  * The non-parametric p-value derivation is NUMPY, not torch, and stays in float64 with NO
    float32 cast (stock: `x_gp = np.sum(perm_cs > cs[..., None], axis=2); pvals = (x_gp + 1) /
    (M + 1)`) -- this differs from CU-A/B's torch/float32 p-value path and is reproduced
    verbatim; since the exact `(n_ct_pairs, n_gene_pairs, M)` `perm_cs` is reconstructed by
    chunking, this is bit-identical to stock, not merely close.
  * `check_analytic_null=True` is rejected (mirrors CU-A/B; `harreman_funcs.py` never sets it).
  * `get_ct_cell_communication_results` (the small, non-chunked results/DataFrame builder) is
    called VERBATIM from harreman via the `_need(...)` shim -- it only touches the already-
    reduced `(n_ct_pairs, n_gene_pairs)`/`(n_ct_pairs, n_metabolites)` arrays, no memory issue.

STOCK QUIRK DISCOVERED WHILE BUILDING THIS (important, not obvious from reading the source
casually): stock's `cs_gp = torch.zeros((len(cell_type_pairs), counts_1.shape[0]),
device=counts_1.device)` -- both the observed-score allocations (parametric and
non-parametric) AND the per-permutation local inside the `M` loop -- omits `dtype`, so it is
`torch.float32` (PyTorch's default dtype), NOT `float64` like every other tensor in this
pipeline (`counts`, `weights`, `Wtot2`, `perm_cs_gp`/`perm_cs_m` are all explicitly float64).
Assigning the float64 `(counts_1.T * WX2t).sum(0)` result into a float32 `cs_gp` row silently
rounds it to float32 precision (~1e-8 relative -- this is what a "clean" float64
reimplementation of this function would get WRONG, and did, on the first pass here: measured
divergence up to 5.9e-8 absolute against real stock before this was tracked down). The
consequences propagate: `EG2_gp = torch.zeros_like(cs_gp) if fix_ct or fix_gp else Wtot2` is
ALSO float32 when `fix_ct`/`fix_gp`; `compute_metabolite_cs_ct`'s per-metabolite gene-pair SUM
runs in whatever dtype `cs_gp` has, so it too executes in float32 arithmetic, not float64; and
the permutation loop's per-`i` `cs_m` (before being upcast into the float64 `perm_cs_m`) is
likewise a float32 sum. This code reproduces all of that verbatim (see the "STOCK QUIRK"
inline comments at each `cs_gp`/`EG2_gp` allocation and at the deferred `perm_cs_m` derivation)
rather than "fixing" it, because the goal is bit-identical output for a given seed, not higher
precision. It is chunk-safe: float32 rounding is a pure per-element operation and does not
depend on how many gene pairs are processed together, so splitting the gene-pair axis into
chunks changes nothing about which float32 value each entry rounds to.

Numerics vs stock: see `tests/test_cell_communication_ct_lowmem.py`, plus an ad hoc 30-seed x
4-chunk-size (120 runs) sweep during development -- EVERY key (`cs`, `pval`, `FDR`, `perm_cs`,
AND parametric `Z`/`Z_pval`/`Z_FDR`) was exactly bit-identical (`assert_array_equal`, max
observed diff 0.0) in that sweep, on this fixture/environment. The test file still uses a
tight `assert_allclose` (rtol=1e-11, atol=1e-13) for `Z`/`Z_pval`/`Z_FDR` as a safety margin
(mirroring the CU-B float64-reduction-order caveat, in case a different environment/tensor
width ever exercises it), but no ULP divergence was actually observed for CU-C in this
sweep -- likely because the float32 rounding of `cs_gp` (coarser than float64) makes the
downstream reduction-order sensitivity CU-B documented even less likely to surface here.
"""

from __future__ import annotations

import inspect
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

import harreman

# --- reuse harreman's own internals so the math is identical --------------------------
_MOD = inspect.getmodule(harreman.tools.compute_interacting_cell_scores)


def _need(name):
    fn = getattr(_MOD, name, None)
    if fn is None:
        raise ImportError(
            f"Could not find helper '{name}' in {_MOD.__name__}. harreman's internal "
            f"layout may have changed; update cell_communication_lowmem.py."
        )
    return fn


counts_from_anndata = _need("counts_from_anndata")
standardize_counts = _need("standardize_counts")
make_weights_non_redundant = _need("make_weights_non_redundant")
compute_metabolite_cs = _need("compute_metabolite_cs")
compute_p_int_cell_results_no_ct = _need("compute_p_int_cell_results_no_ct")

# CU-B additions (cell-type-independent aggregate function + its transitive deps).
flatten = _need("flatten")
compute_p_results = _need("compute_p_results")
get_cell_communication_results = _need("get_cell_communication_results")

# CU-C additions (cell-type-AWARE aggregate function + its transitive deps).
create_weights_ct_pairs = _need("create_weights_ct_pairs")
standardize_ct_counts = _need("standardize_ct_counts")
compute_metabolite_cs_ct = _need("compute_metabolite_cs_ct")
compute_ct_p_results = _need("compute_ct_p_results")
get_ct_cell_communication_results = _need("get_ct_cell_communication_results")


def compute_interacting_cell_scores_lowmem(
    adata,
    center_counts_for_np_test: bool = False,
    test: str = "both",
    restrict_significance: str = "both",
    compute_significance: str = "both",
    M: int = 1000,
    seed: int = 42,
    check_analytic_null: bool = False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose: bool = False,
    gene_pair_chunk_size: int = None,
    metabolite_chunk_size: int = None,
):
    """Memory-safe equivalent of ``harreman.tools.compute_interacting_cell_scores``.

    Same signature (plus ``gene_pair_chunk_size``/``metabolite_chunk_size``) and same
    ``adata.uns['interacting_cell_results']`` outputs, minus the raw permutation arrays.
    See module docstring for the (small) list of differences.

    Parameters
    ----------
    gene_pair_chunk_size : int, optional
        Number of gene pairs processed per chunk in the non-parametric ('np') branch's
        gene-pair pass (CU-E, Option B). Never materializes more than
        ``(n_cells, gene_pair_chunk_size)`` tensors on GPU. ``None`` (default) picks an
        adaptive size ``max(1, 50_000_000 // n_cells)``. A value ``>= n_gp`` reproduces
        the single-chunk (bit-identical) behavior.
    metabolite_chunk_size : int, optional
        Number of metabolites processed per chunk in the non-parametric branch's
        metabolite pass (CU-E). Never materializes more than
        ``(n_cells, metabolite_chunk_size)`` + ``(n_cells, |union of that chunk's gene
        pairs|)`` tensors on GPU. ``None`` (default) picks the same adaptive size as
        ``gene_pair_chunk_size``. A value ``>= n_m`` reproduces the single-chunk
        (bit-identical) behavior.
    """
    start = time.time()
    if verbose:
        print("[lowmem] Computing gene pair and metabolite scores...")

    # Only the non-parametric path materializes the (cells, pairs, M) tensors, so only
    # there is check_analytic_null unsupported; with test='parametric' stock ignores the
    # flag and we match that (it never touches the parametric branch).
    if check_analytic_null and test in ("non-parametric", "both"):
        raise NotImplementedError(
            "check_analytic_null=True is not supported in the low-memory version "
            "(it re-introduces the (cells, pairs, M) tensors this patch removes)."
        )

    adata.uns["interacting_cell_results"] = {}

    model = adata.uns["model"]
    mean = adata.uns["mean"]

    if test not in ["both", "parametric", "non-parametric"]:
        raise ValueError('The "test" variable should be one of ["both", "parametric", "non-parametric"].')
    if restrict_significance is not None and restrict_significance not in ["both", "gene pairs", "metabolites"]:
        raise ValueError('The "restrict_significance" variable should be one of ["both", "gene pairs", "metabolites"].')
    if compute_significance is not None and compute_significance not in ["both", "parametric", "non-parametric"]:
        raise ValueError('The "compute_significance" variable should be one of ["both", "parametric", "non-parametric"].')

    import pandas as pd  # local, matches harreman's usage

    sample_specific = "sample_key" in adata.uns

    layer_key_p_test = adata.uns.get("layer_key_p_test", None)
    layer_key_np_test = adata.uns.get("layer_key_np_test", None)
    use_raw = (layer_key_p_test == "use_raw") and (layer_key_np_test == "use_raw")

    gene_pairs = adata.uns.get("gene_pairs", None)
    gene_pairs_per_metabolite = adata.uns["gene_pairs_per_metabolite"]

    def to_tuple(x):
        if isinstance(x, list):
            return tuple(to_tuple(i) for i in x)
        return x

    metabolite_gene_pair_df = pd.DataFrame.from_dict(gene_pairs_per_metabolite, orient="index").reset_index()
    metabolite_gene_pair_df = metabolite_gene_pair_df.rename(columns={"index": "metabolite"})
    metabolite_gene_pair_df["gene_pair"] = metabolite_gene_pair_df["gene_pair"].apply(
        lambda arr: [(to_tuple(gp[0]), to_tuple(gp[1])) for gp in arr]
    )
    metabolite_gene_pair_df["gene_type"] = metabolite_gene_pair_df["gene_type"].apply(
        lambda arr: [(to_tuple(gt[0]), to_tuple(gt[1])) for gt in arr]
    )
    metabolite_gene_pair_df = pd.concat(
        [
            metabolite_gene_pair_df["metabolite"],
            metabolite_gene_pair_df.explode("gene_pair")["gene_pair"],
            metabolite_gene_pair_df.explode("gene_type")["gene_type"],
        ],
        axis=1,
    ).reset_index(drop=True)

    if "LR_database" in adata.uns:
        LR_database = adata.uns["LR_database"]
        df_merged = pd.merge(metabolite_gene_pair_df, LR_database, left_on="metabolite", right_on="interaction_name", how="left")
        LR_df = df_merged.dropna(subset=["pathway_name"])
        metabolite_gene_pair_df["metabolite"][metabolite_gene_pair_df.metabolite.isin(LR_df.metabolite)] = LR_df["pathway_name"]

    if restrict_significance in ["both", "gene pairs"]:
        cell_com_gp_df = adata.uns["ccc_results"]["cell_com_df_gp_sig"].copy()
        cell_com_gp_df[["Gene 1", "Gene 2"]] = cell_com_gp_df[["Gene 1", "Gene 2"]].applymap(
            lambda x: tuple(x) if isinstance(x, list) else x
        )
        gene_pairs_set = set([tuple(x) for x in cell_com_gp_df[["Gene 1", "Gene 2"]].values])
        metabolite_gene_pair_df = metabolite_gene_pair_df[metabolite_gene_pair_df["gene_pair"].isin(gene_pairs_set)]

    if restrict_significance in ["both", "metabolites"]:
        cell_com_m_df = adata.uns["ccc_results"]["cell_com_df_m_sig"].copy()
        metabolite_set = set(cell_com_m_df["Metabolite"].values)
        metabolite_gene_pair_df = metabolite_gene_pair_df[metabolite_gene_pair_df["metabolite"].isin(metabolite_set)]

    genes = adata.uns["genes"]
    gene_pairs_sig = []
    if gene_pairs:
        for g1, g2 in gene_pairs:
            g1 = tuple(g1) if isinstance(g1, list) else g1
            g2 = tuple(g2) if isinstance(g2, list) else g2
            if not metabolite_gene_pair_df[metabolite_gene_pair_df["gene_pair"] == (g1, g2)].empty:
                gene_pairs_sig.append((g1, g2))
    adata.uns["gene_pairs_sig"] = gene_pairs_sig

    gene_pairs_sig_ind = []
    for g1, g2 in gene_pairs_sig:
        idx1 = tuple([genes.index(g) for g in g1]) if isinstance(g1, tuple) else genes.index(g1)
        idx2 = tuple([genes.index(g) for g in g2]) if isinstance(g2, tuple) else genes.index(g2)
        gene_pairs_sig_ind.append((idx1, idx2))
    adata.uns["gene_pairs_sig_ind"] = gene_pairs_sig_ind

    if "barcode_key" in adata.uns:
        barcode_key = adata.uns["barcode_key"]
        cells = pd.Series(adata.obs[barcode_key].tolist())
    else:
        cells = adata.obs_names if not use_raw else adata.raw.obs_names

    weights = make_weights_non_redundant(adata.obsp["weights"]).tocoo()
    weights = torch.sparse_coo_tensor(
        torch.tensor(np.vstack((weights.row, weights.col)), dtype=torch.long, device=device),
        torch.tensor(weights.data, dtype=torch.float64, device=device),
        torch.Size(weights.shape),
        device=device,
    )

    gene_pair_dict = {}
    for metabolite, group in metabolite_gene_pair_df.groupby("metabolite"):
        idxs = group["gene_pair"].apply(lambda gp: gene_pairs_sig.index(gp) if gp in gene_pairs_sig else None).dropna().tolist()
        idxs = [int(ind) for ind in idxs if ind is not None]
        if idxs:
            gene_pair_dict[metabolite] = idxs
    metabolites = list(gene_pair_dict.keys())
    adata.uns["metabolites"] = metabolites

    gene_pairs_sig_names = [
        "_".join("_".join(g) if isinstance(g, tuple) else g for g in gp) for gp in gene_pairs_sig
    ]
    adata.uns["gene_pairs_sig_names"] = gene_pairs_sig_names

    # ================================ parametric ('p') ================================
    # (verbatim from harreman; no per-permutation / M axis, so no OOM here)
    if test in ["parametric", "both"]:
        if verbose:
            print("[lowmem] Running the parametric test...")
        adata.uns["interacting_cell_results"]["p"] = {"gp": {}, "m": {}}

        Wtot2 = torch.tensor((weights.data ** 2).sum(), device=device)  # verbatim: stock uses weights.data

        counts = counts_from_anndata(adata[cells, genes], layer_key_p_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        num_umi = counts.sum(dim=0)

        counts_1, counts_2 = _prep_counts_1_2(counts, gene_pairs_sig_ind, mean)
        counts_1 = standardize_counts(adata, counts_1, model, num_umi, sample_specific)
        counts_2 = standardize_counts(adata, counts_2, model, num_umi, sample_specific)

        WX2t = torch.sparse.mm(weights, counts_2.T)
        WtX2t = torch.sparse.mm(weights.transpose(0, 1), counts_2.T)
        cs_gp = (counts_1.T * WX2t) + (counts_1.T * WtX2t)
        same_gene_mask = torch.tensor([g1 == g2 for g1, g2 in gene_pairs_sig], device=device)
        cs_gp[:, same_gene_mask] = cs_gp[:, same_gene_mask] / 2
        adata.uns["interacting_cell_results"]["p"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()

        cs_m = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=True)
        adata.uns["interacting_cell_results"]["p"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        if compute_significance in ["parametric", "both"]:
            WX1t = torch.sparse.mm(weights, counts_1.T)
            WtX1t = torch.sparse.mm(weights.transpose(0, 1), counts_1.T)
            eg2_a = (WX1t + WtX1t).pow(2)
            eg2_b = (WX2t + WtX2t).pow(2)
            eg2s_gp = (eg2_a, eg2_b)

            Z_gp, Z_m = compute_p_int_cell_results_no_ct(cs_gp, cs_m, gene_pairs_sig_ind, Wtot2, eg2s_gp, gene_pair_dict)
            Z_gp_np = Z_gp.detach().cpu().numpy()
            Z_m_np = Z_m.detach().cpu().numpy()
            Z_pvals_gp = norm.sf(Z_gp_np)
            Z_pvals_m = norm.sf(Z_m_np)
            FDR_gp = multipletests(Z_pvals_gp.flatten(), method="fdr_bh")[1].reshape(Z_pvals_gp.shape)
            FDR_m = multipletests(Z_pvals_m.flatten(), method="fdr_bh")[1].reshape(Z_pvals_m.shape)

            p_gp = adata.uns["interacting_cell_results"]["p"]["gp"]
            p_m = adata.uns["interacting_cell_results"]["p"]["m"]
            p_gp["Z"], p_gp["Z_pval"], p_gp["Z_FDR"] = Z_gp_np, Z_pvals_gp, FDR_gp
            p_m["Z"], p_m["Z_pval"], p_m["Z_FDR"] = Z_m_np, Z_pvals_m, FDR_m

            _write_sig_masks(p_gp, p_m, Z_pvals_gp, Z_pvals_m, FDR_gp, FDR_m)

        if verbose:
            print("[lowmem] Parametric test finished.")

    # ============================ non-parametric ('np') ==============================
    # THE MEMORY FIX LIVES HERE (see also the module docstring's new CU-E section above).
    #
    # OLD (pre-Option-B, CU-A, 2026-07-20): the block below (commented out) computed the
    # observed cs_gp/cs_m and the permutation exceedance counters as FULL (n_cells, n_gp)
    # / (n_cells, n_m) tensors kept on GPU for the whole M-permutation loop. That removed
    # the (..., M) axis (CU-A's fix vs raw stock) but is still an O(n_cells * n_gp) /
    # O(n_cells * n_m) peak GPU allocation -- the "next bottleneck" documented in
    # 05_harreman_reference.md sec.5 and diagnosed in 07_nbhd_percell_chunking_plan.md.
    # Kept here, fully commented out, for reference / diffing against the prior version --
    # superseded by the CU-E two-pass chunked block that follows it.
    #
    # OLD: # ============================ non-parametric ('np') ==============================
    # OLD: # THE MEMORY FIX LIVES HERE.
    # OLD: if test in ["non-parametric", "both"]:
    # OLD: if verbose:
    # OLD: print("[lowmem] Running the non-parametric test...")
    # OLD: adata.uns["interacting_cell_results"]["np"] = {"gp": {}, "m": {}}
    #
    # OLD: counts = counts_from_anndata(adata[cells, genes], layer_key_np_test, dense=True)
    # OLD: counts = torch.tensor(counts, dtype=torch.float64, device=device)
    #
    # OLD: counts_1, counts_2 = _prep_counts_1_2(counts, gene_pairs_sig_ind, mean)
    #
    # OLD: if center_counts_for_np_test:
    # OLD: num_umi = counts.sum(dim=0)
    # OLD: counts_1 = standardize_counts(adata, counts_1, model, num_umi, sample_specific)
    # OLD: counts_2 = standardize_counts(adata, counts_2, model, num_umi, sample_specific)
    #
    # OLD: n_cells = counts_1.shape[1]
    # OLD: n_gp = counts_1.shape[0]
    # OLD: same_gene_mask = torch.tensor([g1 == g2 for g1, g2 in gene_pairs_sig], device=device)
    #
    # OLD: if center_counts_for_np_test and test == "both":
    # OLD: adata.uns["interacting_cell_results"]["np"]["gp"]["cs"] = np.array(adata.uns["interacting_cell_results"]["p"]["gp"]["cs"])
    # OLD: adata.uns["interacting_cell_results"]["np"]["m"]["cs"] = np.array(adata.uns["interacting_cell_results"]["p"]["m"]["cs"])
    # OLD: else:
    # OLD: WX2t = torch.sparse.mm(weights, counts_2.T)
    # OLD: WtX2t = torch.sparse.mm(weights.transpose(0, 1), counts_2.T)
    # OLD: cs_gp = (counts_1.T * WX2t) + (counts_1.T * WtX2t)
    # OLD: cs_gp[:, same_gene_mask] = cs_gp[:, same_gene_mask] / 2
    # OLD: adata.uns["interacting_cell_results"]["np"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()
    # OLD: cs_m = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=True)
    # OLD: adata.uns["interacting_cell_results"]["np"]["m"]["cs"] = cs_m.detach().cpu().numpy()
    #
    # OLD: if compute_significance in ["non-parametric", "both"]:
    # OLD: # observed scores as tensors (robust to either branch above)
    # OLD: cs_gp = torch.as_tensor(
    # OLD: np.asarray(adata.uns["interacting_cell_results"]["np"]["gp"]["cs"]), dtype=torch.float64, device=device
    # OLD: )
    # OLD: cs_m = torch.as_tensor(
    # OLD: np.asarray(adata.uns["interacting_cell_results"]["np"]["m"]["cs"]), dtype=torch.float64, device=device
    # OLD: )
    # OLD: n_m = cs_m.shape[1]
    #
    # OLD: # ---------------------------------------------------------------------------
    # OLD: # STOCK (harreman's compute_interacting_cell_scores, lines ~1792-1873): builds
    # OLD: # the *entire* permutation null as dense (n_cells, n_pairs, M) tensors, fills
    # OLD: # them one permutation at a time, then derives the exceedance counts (x_*) and
    # OLD: # p-values from those materialized arrays afterwards. This is the OOM site --
    # OLD: # kept verbatim below (commented) for reference / diffing against upstream.
    # OLD: #
    # OLD: # STOCK: perm_cs_gp_a = torch.zeros((n_cells, counts_1.shape[0], M), dtype=torch.float64, device=device)
    # OLD: # STOCK: perm_cs_gp_b = torch.zeros_like(perm_cs_gp_a)
    # OLD: # STOCK: perm_cs_m_a = torch.zeros((n_cells, len(gene_pair_dict), M), dtype=torch.float64, device=device)
    # OLD: # STOCK: perm_cs_m_b = torch.zeros_like(perm_cs_m_a)
    # OLD: # STOCK:
    # OLD: # STOCK: if check_analytic_null:
    # OLD: # STOCK:     gp_zs_perm_array = torch.zeros_like(perm_cs_gp_a)
    # OLD: # STOCK:     gp_pvals_perm_array = torch.zeros_like(perm_cs_gp_a)
    # OLD: # STOCK:     m_zs_perm_array = torch.zeros_like(perm_cs_m_a)
    # OLD: # STOCK:     m_pvals_perm_array = torch.zeros_like(perm_cs_m_a)
    # OLD: # STOCK:
    # OLD: # STOCK: torch.manual_seed(seed)
    # OLD: # STOCK: for i in tqdm(range(M), desc="Permutation test"):
    # OLD: # STOCK:     idx = torch.randperm(n_cells, device=device)
    # OLD: # STOCK:
    # OLD: # STOCK:     c1_perm_a = counts_1.clone()
    # OLD: # STOCK:     c2_perm_a = counts_2[:, idx]
    # OLD: # STOCK:     c1_perm_a[same_gene_mask] = counts_1[same_gene_mask, :][:, idx]
    # OLD: # STOCK:
    # OLD: # STOCK:     WX2t_a = torch.sparse.mm(weights, c2_perm_a.T)
    # OLD: # STOCK:     WtX2t_a = torch.sparse.mm(weights.transpose(0, 1), c2_perm_a.T)
    # OLD: # STOCK:     cs_a = (c1_perm_a.T * WX2t_a) + (c1_perm_a.T * WtX2t_a)
    # OLD: # STOCK:     cs_a[:, same_gene_mask] = cs_a[:, same_gene_mask] / 2
    # OLD: # STOCK:     perm_cs_gp_a[:, :, i] = cs_a
    # OLD: # STOCK:
    # OLD: # STOCK:     cs_m_a = compute_metabolite_cs(cs_a, gene_pair_dict, interacting_cell_scores=True)
    # OLD: # STOCK:     perm_cs_m_a[:, :, i] = cs_m_a
    # OLD: # STOCK:
    # OLD: # STOCK:     c2_perm_b = counts_2.clone()
    # OLD: # STOCK:     c1_perm_b = counts_1[:, idx]
    # OLD: # STOCK:     c2_perm_b[same_gene_mask] = counts_2[same_gene_mask, :][:, idx]
    # OLD: # STOCK:
    # OLD: # STOCK:     WX2t_b = torch.sparse.mm(weights, c2_perm_b.T)
    # OLD: # STOCK:     WtX2t_b = torch.sparse.mm(weights.transpose(0, 1), c2_perm_b.T)
    # OLD: # STOCK:     cs_b = (c1_perm_b.T * WX2t_b) + (c1_perm_b.T * WtX2t_b)
    # OLD: # STOCK:     cs_b[:, same_gene_mask] = cs_b[:, same_gene_mask] / 2
    # OLD: # STOCK:     perm_cs_gp_b[:, :, i] = cs_b
    # OLD: # STOCK:
    # OLD: # STOCK:     cs_m_b = compute_metabolite_cs(cs_b, gene_pair_dict, interacting_cell_scores=True)
    # OLD: # STOCK:     perm_cs_m_b[:, :, i] = cs_m_b
    # OLD: # STOCK:
    # OLD: # STOCK:     if check_analytic_null:
    # OLD: # STOCK:         Z_gp_perm, Z_m_perm = compute_p_results((cs_a, cs_b), (cs_m_a, cs_m_b), gene_pairs_ind, Wtot2, eg2s_gp, gene_pair_dict)
    # OLD: # STOCK:         gp_zs_perm_array[:, :, i] = Z_gp_perm
    # OLD: # STOCK:         gp_pvals_perm_array[:, :, i] = torch.tensor(norm.sf(Z_gp_perm.cpu().numpy()), device=device)
    # OLD: # STOCK:         m_zs_perm_array[:, :, i] = Z_m_perm
    # OLD: # STOCK:         m_pvals_perm_array[:, :, i] = torch.tensor(norm.sf(Z_m_perm.cpu().numpy()), device=device)
    # OLD: # STOCK:
    # OLD: # STOCK: adata.uns['interacting_cell_results']['np']['gp']['perm_cs_a'] = perm_cs_gp_a.detach().cpu().numpy()
    # OLD: # STOCK: adata.uns['interacting_cell_results']['np']['gp']['perm_cs_b'] = perm_cs_gp_b.detach().cpu().numpy()
    # OLD: # STOCK: adata.uns['interacting_cell_results']['np']['m']['perm_cs_a'] = perm_cs_m_a.detach().cpu().numpy()
    # OLD: # STOCK: adata.uns['interacting_cell_results']['np']['m']['perm_cs_b'] = perm_cs_m_b.detach().cpu().numpy()
    # OLD: # STOCK:
    # OLD: # STOCK: x_gp_a = (perm_cs_gp_a > cs_gp[:, :, None]).sum(dim=2)
    # OLD: # STOCK: x_gp_b = (perm_cs_gp_b > cs_gp[:, :, None]).sum(dim=2)
    # OLD: # STOCK: x_m_a = (perm_cs_m_a > cs_m[:, :, None]).sum(dim=2)
    # OLD: # STOCK: x_m_b = (perm_cs_m_b > cs_m[:, :, None]).sum(dim=2)
    # OLD: # STOCK:
    # OLD: # STOCK: pvals_gp_a = (x_gp_a + 1).float() / (M + 1)
    # OLD: # STOCK: pvals_gp_b = (x_gp_b + 1).float() / (M + 1)
    # OLD: # STOCK: pvals_m_a = (x_m_a + 1).float() / (M + 1)
    # OLD: # STOCK: pvals_m_b = (x_m_b + 1).float() / (M + 1)
    # OLD: # STOCK:
    # OLD: # STOCK: pvals_gp = torch.where(pvals_gp_a > pvals_gp_b, pvals_gp_a, pvals_gp_b)
    # OLD: # STOCK: pvals_m = torch.where(pvals_m_a > pvals_m_b, pvals_m_a, pvals_m_b)
    # OLD: # STOCK:
    # OLD: # STOCK: pvals_gp = pvals_gp.cpu().numpy()
    # OLD: # STOCK: pvals_m = pvals_m.cpu().numpy()
    # OLD: # STOCK:
    # OLD: # STOCK: if check_analytic_null:
    # OLD: # STOCK:     adata.uns['interacting_cell_results']['np']['analytic_null'] = {
    # OLD: # STOCK:         'gp_zs_perm': gp_zs_perm_array.detach().cpu().numpy(),
    # OLD: # STOCK:         'gp_pvals_perm': gp_pvals_perm_array.detach().cpu().numpy(),
    # OLD: # STOCK:         'm_zs_perm': m_zs_perm_array.detach().cpu().numpy(),
    # OLD: # STOCK:         'm_pvals_perm': m_pvals_perm_array.detach().cpu().numpy(),
    # OLD: # STOCK:     }
    # OLD: #
    # OLD: # LOWMEM REPLACEMENT: never materialize perm_cs_*; accumulate the exceedance
    # OLD: # counters (x_gp_a/b, x_m_a/b) incrementally inside the permutation loop
    # OLD: # instead. This is the O(n_cells*n_pairs*M) -> O(n_cells*n_pairs) memory fix
    # OLD: # (05_harreman_reference.md sec.5). check_analytic_null=True is rejected at
    # OLD: # the top of this function (it needs the removed arrays), so the
    # OLD: # check_analytic_null branches in the STOCK block above are permanently dead
    # OLD: # here and kept only for reference.
    # OLD: #
    # OLD: # NOTE (float precision): we accumulate the exceedance counter in float64
    # OLD: # (exact for integer counts <= M+1) but replicate stock's float32 cast at the
    # OLD: # divide below ("(x + 1).float() / (M + 1)"), so pval/FDR are bit-for-bit
    # OLD: # identical to stock for the same seed.
    # OLD: # ---------------------------------------------------------------------------
    # OLD: x_gp_a = torch.zeros((n_cells, n_gp), dtype=torch.float64, device=device)
    # OLD: x_gp_b = torch.zeros_like(x_gp_a)
    # OLD: x_m_a = torch.zeros((n_cells, n_m), dtype=torch.float64, device=device)
    # OLD: x_m_b = torch.zeros_like(x_m_a)
    #
    # OLD: torch.manual_seed(seed)
    # OLD: for _ in tqdm(range(M), desc="[lowmem] Permutation test", disable=not verbose):
    # OLD: idx = torch.randperm(n_cells, device=device)
    #
    # OLD: # permute the "receiver" (counts_2), keep "sender" (counts_1) — arm a
    # OLD: c1_perm_a = counts_1.clone()
    # OLD: c2_perm_a = counts_2[:, idx]
    # OLD: c1_perm_a[same_gene_mask] = counts_1[same_gene_mask, :][:, idx]
    # OLD: WX2t_a = torch.sparse.mm(weights, c2_perm_a.T)
    # OLD: WtX2t_a = torch.sparse.mm(weights.transpose(0, 1), c2_perm_a.T)
    # OLD: cs_a = (c1_perm_a.T * WX2t_a) + (c1_perm_a.T * WtX2t_a)
    # OLD: cs_a[:, same_gene_mask] = cs_a[:, same_gene_mask] / 2
    # OLD: cs_m_a = compute_metabolite_cs(cs_a, gene_pair_dict, interacting_cell_scores=True)
    # OLD: x_gp_a += (cs_a > cs_gp).to(torch.float64)
    # OLD: x_m_a += (cs_m_a > cs_m).to(torch.float64)
    #
    # OLD: # permute the "sender" (counts_1), keep "receiver" (counts_2) — arm b
    # OLD: c2_perm_b = counts_2.clone()
    # OLD: c1_perm_b = counts_1[:, idx]
    # OLD: c2_perm_b[same_gene_mask] = counts_2[same_gene_mask, :][:, idx]
    # OLD: WX2t_b = torch.sparse.mm(weights, c2_perm_b.T)
    # OLD: WtX2t_b = torch.sparse.mm(weights.transpose(0, 1), c2_perm_b.T)
    # OLD: cs_b = (c1_perm_b.T * WX2t_b) + (c1_perm_b.T * WtX2t_b)
    # OLD: cs_b[:, same_gene_mask] = cs_b[:, same_gene_mask] / 2
    # OLD: cs_m_b = compute_metabolite_cs(cs_b, gene_pair_dict, interacting_cell_scores=True)
    # OLD: x_gp_b += (cs_b > cs_gp).to(torch.float64)
    # OLD: x_m_b += (cs_m_b > cs_m).to(torch.float64)
    #
    # OLD: # replicate stock's float32 cast (`(x + 1).float() / (M + 1)`) so pval/FDR
    # OLD: # match stock bit-for-bit; x_* hold integer counts <= M+1, exact in float32.
    # OLD: pvals_gp_a = (x_gp_a + 1).to(torch.float32) / (M + 1)
    # OLD: pvals_gp_b = (x_gp_b + 1).to(torch.float32) / (M + 1)
    # OLD: pvals_m_a = (x_m_a + 1).to(torch.float32) / (M + 1)
    # OLD: pvals_m_b = (x_m_b + 1).to(torch.float32) / (M + 1)
    #
    # OLD: pvals_gp = torch.where(pvals_gp_a > pvals_gp_b, pvals_gp_a, pvals_gp_b).cpu().numpy()
    # OLD: pvals_m = torch.where(pvals_m_a > pvals_m_b, pvals_m_a, pvals_m_b).cpu().numpy()
    #
    # OLD: np_gp = adata.uns["interacting_cell_results"]["np"]["gp"]
    # OLD: np_m = adata.uns["interacting_cell_results"]["np"]["m"]
    # OLD: np_gp["pval"] = pvals_gp
    # OLD: np_gp["FDR"] = multipletests(pvals_gp.flatten(), method="fdr_bh")[1].reshape(pvals_gp.shape)
    # OLD: np_m["pval"] = pvals_m
    # OLD: np_m["FDR"] = multipletests(pvals_m.flatten(), method="fdr_bh")[1].reshape(pvals_m.shape)
    #
    # OLD: _write_sig_masks(np_gp, np_m, pvals_gp, pvals_m, np_gp["FDR"], np_m["FDR"])
    #
    # OLD: if verbose:
    # OLD: print("[lowmem] Non-parametric test finished.")
    #
    # NEW (CU-E, Option B two-pass chunking): see the module docstring's CU-E section for
    # the full derivation. Bounds GPU to (n_cells, chunk) while still assembling and
    # storing the exact same full (n_cells, n_gp) / (n_cells, n_m) arrays in `uns`.
    if test in ["non-parametric", "both"]:
        if verbose:
            print("[lowmem] Running the non-parametric test...")
        adata.uns["interacting_cell_results"]["np"] = {"gp": {}, "m": {}}

        counts = counts_from_anndata(adata[cells, genes], layer_key_np_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)

        n_cells = counts.shape[1]
        n_gp = len(gene_pairs_sig_ind)
        n_m = len(gene_pair_dict)
        same_gene_mask_full = torch.tensor([g1 == g2 for g1, g2 in gene_pairs_sig], device=device)

        num_umi = counts.sum(dim=0) if center_counts_for_np_test else None  # GLOBAL, unchunked

        use_p_shortcut = center_counts_for_np_test and test == "both"
        want_significance = compute_significance in ["non-parametric", "both"]

        # Adaptive chunk sizes: bound n_cells * chunk to ~50M elements (CU-B/C convention),
        # independently for the gene-pair axis and the metabolite axis.
        gp_chunk = max(1, 50_000_000 // max(n_cells, 1)) if gene_pair_chunk_size is None else max(1, int(gene_pair_chunk_size))
        m_chunk = max(1, 50_000_000 // max(n_cells, 1)) if metabolite_chunk_size is None else max(1, int(metabolite_chunk_size))

        # --------------------------- Pass 1: gene-pair chunks ------------------------------
        # Outer loop over gene-pair chunks, inner loop over the M permutations, with
        # `torch.manual_seed(seed)` re-issued at the START of each chunk -- the proven CU-B/C
        # RNG-replay trick: only `torch.randperm` consumes RNG in the loop body, so re-seeding
        # per chunk reproduces the exact idx_0..idx_{M-1} sequence stock draws once for all
        # gene pairs. GPU only ever holds (n_cells, chunk)-shaped tensors; the full
        # (n_cells, n_gp) `cs`/exceedance-counter arrays are assembled on CPU.
        cs_gp_cpu = np.empty((n_cells, n_gp), dtype=np.float64)
        if use_p_shortcut:
            # STOCK quirk, reproduced verbatim: when centering AND both tests ran, the 'np'
            # observed score is just a copy of the already-computed 'p' score (no recompute).
            cs_gp_cpu[...] = np.asarray(adata.uns["interacting_cell_results"]["p"]["gp"]["cs"])
        if want_significance:
            x_gp_a_cpu = np.zeros((n_cells, n_gp), dtype=np.float64)
            x_gp_b_cpu = np.zeros((n_cells, n_gp), dtype=np.float64)

        if (not use_p_shortcut) or want_significance:
            for i0 in range(0, n_gp, gp_chunk):
                sl = slice(i0, min(i0 + gp_chunk, n_gp))
                gp_ind_chunk = gene_pairs_sig_ind[sl]
                same_gene_chunk = same_gene_mask_full[sl]

                counts_1c, counts_2c = _prep_counts_1_2(counts, gp_ind_chunk, mean)
                if center_counts_for_np_test:
                    counts_1c = standardize_counts(adata, counts_1c, model, num_umi, sample_specific)
                    counts_2c = standardize_counts(adata, counts_2c, model, num_umi, sample_specific)

                if use_p_shortcut:
                    cs_gp_c = torch.as_tensor(cs_gp_cpu[:, sl], dtype=torch.float64, device=device)
                else:
                    WX2t_c = torch.sparse.mm(weights, counts_2c.T)
                    WtX2t_c = torch.sparse.mm(weights.transpose(0, 1), counts_2c.T)
                    cs_gp_c = (counts_1c.T * WX2t_c) + (counts_1c.T * WtX2t_c)
                    cs_gp_c[:, same_gene_chunk] = cs_gp_c[:, same_gene_chunk] / 2
                    cs_gp_cpu[:, sl] = cs_gp_c.detach().cpu().numpy()

                if want_significance:
                    x_gp_a_c = torch.zeros((n_cells, sl.stop - sl.start), dtype=torch.float64, device=device)
                    x_gp_b_c = torch.zeros_like(x_gp_a_c)

                    torch.manual_seed(seed)  # reseed EVERY chunk -> replays idx_0..idx_{M-1} identically
                    for _ in tqdm(range(M), desc="[lowmem] Permutation test (gene-pair chunk)", disable=not verbose):
                        idx = torch.randperm(n_cells, device=device)

                        # permute the "receiver" (counts_2), keep "sender" (counts_1) -- arm a
                        c1_perm_a = counts_1c.clone()
                        c2_perm_a = counts_2c[:, idx]
                        c1_perm_a[same_gene_chunk] = counts_1c[same_gene_chunk, :][:, idx]
                        WX2t_a = torch.sparse.mm(weights, c2_perm_a.T)
                        WtX2t_a = torch.sparse.mm(weights.transpose(0, 1), c2_perm_a.T)
                        cs_a = (c1_perm_a.T * WX2t_a) + (c1_perm_a.T * WtX2t_a)
                        cs_a[:, same_gene_chunk] = cs_a[:, same_gene_chunk] / 2
                        x_gp_a_c += (cs_a > cs_gp_c).to(torch.float64)

                        # permute the "sender" (counts_1), keep "receiver" (counts_2) -- arm b
                        c2_perm_b = counts_2c.clone()
                        c1_perm_b = counts_1c[:, idx]
                        c2_perm_b[same_gene_chunk] = counts_2c[same_gene_chunk, :][:, idx]
                        WX2t_b = torch.sparse.mm(weights, c2_perm_b.T)
                        WtX2t_b = torch.sparse.mm(weights.transpose(0, 1), c2_perm_b.T)
                        cs_b = (c1_perm_b.T * WX2t_b) + (c1_perm_b.T * WtX2t_b)
                        cs_b[:, same_gene_chunk] = cs_b[:, same_gene_chunk] / 2
                        x_gp_b_c += (cs_b > cs_gp_c).to(torch.float64)

                    x_gp_a_cpu[:, sl] = x_gp_a_c.detach().cpu().numpy()
                    x_gp_b_cpu[:, sl] = x_gp_b_c.detach().cpu().numpy()

        adata.uns["interacting_cell_results"]["np"]["gp"]["cs"] = cs_gp_cpu

        # --------------------------- Pass 2: metabolite chunks -----------------------------
        # Outer loop over metabolite chunks, re-seeded per chunk exactly like Pass 1, inner
        # loop over the M permutations. For each metabolite chunk, gather the UNION of gene
        # pairs across that chunk's metabolites (many-to-many: a gene pair may serve several
        # metabolites), recompute only that union's per-permutation gene-pair scores, and sum
        # them into per-metabolite scores via a remapped sub-dict. This recomputes gene-pair
        # scores a second time (~2x total sparse.mm cost) but bounds the GPU to
        # (n_cells, metabolite_chunk) + (n_cells, |union|) -- sanctioned by the plan (sec.5)
        # as the only way to finish a metabolite without ever holding the full (n_cells, n_gp)
        # tensor on GPU (a metabolite's gene pairs can span multiple gene-pair chunks).
        #
        # IMPORTANT (device parity, fixed after review): the OBSERVED cs_m is computed HERE,
        # per metabolite chunk, ON THE SAME DEVICE as the permutation scores it is compared
        # against (mirrors Pass 1's structure exactly: build the union's observed gene-pair
        # `cs` on `device` -- no permutation -- then reduce with `compute_metabolite_cs` on
        # `device`). A prior version of this fix computed the FULL cs_m once via a CPU-side
        # `compute_metabolite_cs` call and used that CPU value as the exceedance threshold
        # for the GPU-computed permutation scores below -- for GPU runs, `.sum(dim=1)` over
        # >= 3 gene pairs can round differently between a CPU and a CUDA reduction kernel
        # (measure-zero ULP drift), which both corrupts the stored `cs` (no longer bit-exact
        # vs stock) and can flip an `x_m` exceedance count when a permutation score sits
        # within 1 ULP of the threshold. Computing cs_m_c on `device` for both roles removes
        # that CPU/GPU reduction seam entirely -- see the module docstring's CU-E section.
        cs_m_cpu = np.empty((n_cells, n_m), dtype=np.float64)
        if use_p_shortcut:
            # STOCK quirk, reproduced verbatim (mirrors the `gp`-grain shortcut above): when
            # centering AND both tests ran, the 'np' observed metabolite score is just a copy
            # of the already-computed 'p' score (no recompute) -- NOT a re-derivation from the
            # (possibly re-standardized) 'np' counts.
            cs_m_cpu[...] = np.asarray(adata.uns["interacting_cell_results"]["p"]["m"]["cs"])
        if want_significance:
            x_m_a_cpu = np.zeros((n_cells, n_m), dtype=np.float64)
            x_m_b_cpu = np.zeros((n_cells, n_m), dtype=np.float64)

        if (not use_p_shortcut) or want_significance:
            for m0 in range(0, n_m, m_chunk):
                sl_m = slice(m0, min(m0 + m_chunk, n_m))
                metabs_chunk = metabolites[sl_m]

                union_indices = sorted(set().union(*(set(gene_pair_dict[m]) for m in metabs_chunk)))
                local_pos = {gp_idx: i for i, gp_idx in enumerate(union_indices)}
                sub_dict = {m: [local_pos[gp_idx] for gp_idx in gene_pair_dict[m]] for m in metabs_chunk}
                gp_ind_union = [gene_pairs_sig_ind[i] for i in union_indices]
                union_idx_t = torch.tensor(union_indices, device=device, dtype=torch.long)
                same_gene_union = same_gene_mask_full[union_idx_t]

                counts_1u, counts_2u = _prep_counts_1_2(counts, gp_ind_union, mean)
                if center_counts_for_np_test:
                    counts_1u = standardize_counts(adata, counts_1u, model, num_umi, sample_specific)
                    counts_2u = standardize_counts(adata, counts_2u, model, num_umi, sample_specific)

                if use_p_shortcut:
                    cs_m_c = torch.as_tensor(cs_m_cpu[:, sl_m], dtype=torch.float64, device=device)
                else:
                    # Observed union gene-pair scores, ON DEVICE, no permutation -- identical
                    # formula to Pass 1's observed `cs_gp_c` (and to stock's `cs_gp`), restricted
                    # to this chunk's union of gene pairs. Bit-identical per-column to stock's
                    # full-width `cs_gp` (sparse.mm columns are independent of one another --
                    # same argument as Pass 1 / CU-B).
                    WX2t_u = torch.sparse.mm(weights, counts_2u.T)
                    WtX2t_u = torch.sparse.mm(weights.transpose(0, 1), counts_2u.T)
                    cs_union = (counts_1u.T * WX2t_u) + (counts_1u.T * WtX2t_u)
                    cs_union[:, same_gene_union] = cs_union[:, same_gene_union] / 2
                    cs_m_c = compute_metabolite_cs(cs_union, sub_dict, interacting_cell_scores=True)
                    cs_m_cpu[:, sl_m] = cs_m_c.detach().cpu().numpy()

                if want_significance:
                    x_m_a_c = torch.zeros((n_cells, sl_m.stop - sl_m.start), dtype=torch.float64, device=device)
                    x_m_b_c = torch.zeros_like(x_m_a_c)

                    torch.manual_seed(seed)  # reseed EVERY chunk -> replays idx_0..idx_{M-1} identically
                    for _ in tqdm(range(M), desc="[lowmem] Permutation test (metabolite chunk)", disable=not verbose):
                        idx = torch.randperm(n_cells, device=device)

                        # arm a: permute the "receiver" (counts_2), keep "sender" (counts_1) --
                        # structurally identical to Pass 1's arm a, restricted to this chunk's union.
                        c1_perm_a = counts_1u.clone()
                        c2_perm_a = counts_2u[:, idx]
                        c1_perm_a[same_gene_union] = counts_1u[same_gene_union, :][:, idx]
                        WX2t_a = torch.sparse.mm(weights, c2_perm_a.T)
                        WtX2t_a = torch.sparse.mm(weights.transpose(0, 1), c2_perm_a.T)
                        cs_a_union = (c1_perm_a.T * WX2t_a) + (c1_perm_a.T * WtX2t_a)
                        cs_a_union[:, same_gene_union] = cs_a_union[:, same_gene_union] / 2
                        cs_m_a_chunk = compute_metabolite_cs(cs_a_union, sub_dict, interacting_cell_scores=True)
                        x_m_a_c += (cs_m_a_chunk > cs_m_c).to(torch.float64)

                        # arm b: permute the "sender" (counts_1), keep "receiver" (counts_2)
                        c2_perm_b = counts_2u.clone()
                        c1_perm_b = counts_1u[:, idx]
                        c2_perm_b[same_gene_union] = counts_2u[same_gene_union, :][:, idx]
                        WX2t_b = torch.sparse.mm(weights, c2_perm_b.T)
                        WtX2t_b = torch.sparse.mm(weights.transpose(0, 1), c2_perm_b.T)
                        cs_b_union = (c1_perm_b.T * WX2t_b) + (c1_perm_b.T * WtX2t_b)
                        cs_b_union[:, same_gene_union] = cs_b_union[:, same_gene_union] / 2
                        cs_m_b_chunk = compute_metabolite_cs(cs_b_union, sub_dict, interacting_cell_scores=True)
                        x_m_b_c += (cs_m_b_chunk > cs_m_c).to(torch.float64)

                    x_m_a_cpu[:, sl_m] = x_m_a_c.detach().cpu().numpy()
                    x_m_b_cpu[:, sl_m] = x_m_b_c.detach().cpu().numpy()

        adata.uns["interacting_cell_results"]["np"]["m"]["cs"] = cs_m_cpu

        if want_significance:
            # Replicate stock's float32 cast (`(x + 1).float() / (M + 1)`) so pval/FDR match
            # stock bit-for-bit; x_* hold integer counts <= M+1, exact in float64, cast once
            # here. This elementwise cast/divide is device-independent (no reduction), so
            # doing it on CPU (as here) vs GPU (as stock/CU-A did) makes no numeric difference.
            pvals_gp_a = (torch.as_tensor(x_gp_a_cpu, dtype=torch.float64) + 1).to(torch.float32) / (M + 1)
            pvals_gp_b = (torch.as_tensor(x_gp_b_cpu, dtype=torch.float64) + 1).to(torch.float32) / (M + 1)
            pvals_m_a = (torch.as_tensor(x_m_a_cpu, dtype=torch.float64) + 1).to(torch.float32) / (M + 1)
            pvals_m_b = (torch.as_tensor(x_m_b_cpu, dtype=torch.float64) + 1).to(torch.float32) / (M + 1)

            pvals_gp = torch.where(pvals_gp_a > pvals_gp_b, pvals_gp_a, pvals_gp_b).numpy()
            pvals_m = torch.where(pvals_m_a > pvals_m_b, pvals_m_a, pvals_m_b).numpy()

            # BH ONCE over each FULL flattened p-value matrix -- never per-chunk.
            np_gp = adata.uns["interacting_cell_results"]["np"]["gp"]
            np_m = adata.uns["interacting_cell_results"]["np"]["m"]
            np_gp["pval"] = pvals_gp
            np_gp["FDR"] = multipletests(pvals_gp.flatten(), method="fdr_bh")[1].reshape(pvals_gp.shape)
            np_m["pval"] = pvals_m
            np_m["FDR"] = multipletests(pvals_m.flatten(), method="fdr_bh")[1].reshape(pvals_m.shape)

            _write_sig_masks(np_gp, np_m, pvals_gp, pvals_m, np_gp["FDR"], np_m["FDR"])

        if verbose:
            print("[lowmem] Non-parametric test finished.")

    if verbose:
        print("[lowmem] Finished in %.3f seconds" % (time.time() - start))
    return


def _prep_counts_1_2(counts, gene_pairs_sig_ind, mean):
    """Build the (n_gene_pairs, n_cells) counts_1 / counts_2 stacks. Verbatim logic from
    harreman (handles heterodimer tuple/list indices)."""
    counts_1, counts_2 = [], []
    for (idx1, idx2) in gene_pairs_sig_ind:
        if isinstance(idx1, (tuple, list)):
            c1 = counts[list(idx1), :].mean(dim=0) if mean == "algebraic" else torch.exp(torch.log(counts[list(idx1), :] + 1e-8).mean(dim=0))
        else:
            c1 = counts[idx1, :]
        if isinstance(idx2, (tuple, list)):
            c2 = counts[list(idx2), :].mean(dim=0) if mean == "algebraic" else torch.exp(torch.log(counts[list(idx2), :] + 1e-8).mean(dim=0))
        else:
            c2 = counts[idx2, :]
        counts_1.append(c1)
        counts_2.append(c2)
    return torch.stack(counts_1), torch.stack(counts_2)


def _write_sig_masks(gp_dict, m_dict, pval_or_none_gp, pval_or_none_m, fdr_gp, fdr_m):
    """Write the cs_sig_pval / cs_sig_FDR masked arrays (verbatim behavior)."""
    # p-value masks (only when a p-value array is available)
    if pval_or_none_gp is not None:
        for key, arr, mask_src in (
            ("cs_sig_pval", "cs", pval_or_none_gp),
        ):
            m = mask_src < 0.05
            cs = gp_dict["cs"].copy()
            cs[~m] = np.nan
            gp_dict[key] = cs
        m = pval_or_none_m < 0.05
        cs = m_dict["cs"].copy()
        cs[~m] = np.nan
        m_dict["cs_sig_pval"] = cs
    # FDR masks
    m = fdr_gp < 0.05
    cs = gp_dict["cs"].copy()
    cs[~m] = np.nan
    gp_dict["cs_sig_FDR"] = cs
    m = fdr_m < 0.05
    cs = m_dict["cs"].copy()
    cs[~m] = np.nan
    m_dict["cs_sig_FDR"] = cs


# ============================================================================================
# CU-B: compute_cell_communication_lowmem (cell-type-INDEPENDENT aggregate scores)
# ============================================================================================

def compute_cell_communication_lowmem(
    adata,
    layer_key_p_test=None,
    layer_key_np_test=None,
    model: str = None,
    center_counts_for_np_test: bool = False,
    subset_gene_pairs=None,
    M: int = 1000,
    seed: int = 42,
    test: str = "both",
    mean: str = "algebraic",
    check_analytic_null: bool = False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose: bool = False,
    gene_pair_chunk_size: int = None,
):
    """Memory-safe equivalent of ``harreman.tools.compute_cell_communication``.

    Same signature (plus ``gene_pair_chunk_size``) and the same ``adata.uns`` outputs
    (``ccc_results``, ``lc_zs``, ``gene_pair_dict``, ``D``, ``genes``, ``gene_pairs_ind``,
    ``cells``) -- see module docstring for what differs internally (nothing observable:
    outputs are bit-for-bit identical to stock for the same seed / chunk choice).

    Parameters
    ----------
    gene_pair_chunk_size : int, optional
        Number of gene pairs processed per chunk (never materializes more than
        ``(gene_pair_chunk_size, n_cells)`` / ``(n_cells, gene_pair_chunk_size)`` tensors).
        ``None`` (default) picks an adaptive size ``max(1, 50_000_000 // n_cells)`` so
        ``n_cells * chunk`` stays roughly bounded. A value ``>= n_gene_pairs`` reproduces
        stock's single-pass behavior exactly (one chunk = the whole thing).
    """
    start = time.time()
    if verbose:
        print("[lowmem] Starting cell-cell communication analysis...")

    # check_analytic_null re-derives Z-scores *inside* the permutation loop (per chunk,
    # per permutation) using arrays (eg2s_gp) that stock only computes when test hits the
    # parametric branch -- real usage (harreman_funcs.py) never sets this, so it is out of
    # scope for this memory fix (mirrors the CU-A guard for the same flag).
    if check_analytic_null:
        raise NotImplementedError(
            "check_analytic_null=True is not supported in the low-memory version "
            "(harreman_funcs.py never sets it; supporting it would require deriving "
            "Z-scores inside the chunked permutation loop, out of scope for this "
            "memory fix)."
        )

    adata.uns["ccc_results"] = {}

    if test not in ["both", "parametric", "non-parametric"]:
        raise ValueError('The "test" variable should be one of ["both", "parametric", "non-parametric"].')
    if mean not in ["algebraic", "geometric"]:
        raise ValueError('The "mean" variable should be one of ["algebraic", "geometric"].')

    adata.uns["layer_key_p_test"] = layer_key_p_test
    adata.uns["layer_key_np_test"] = layer_key_np_test
    adata.uns["model"] = model
    adata.uns["center_counts_for_np_test"] = center_counts_for_np_test
    adata.uns["mean"] = mean

    _run_cell_communication_analysis_lowmem(
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
        device,
        verbose,
        gene_pair_chunk_size,
    )

    if verbose:
        print("[lowmem] Obtaining communication results...")
    # Small: operates on the already-reduced (n_gene_pairs,)/(n_metabolites,) arrays, so
    # no memory issue -- call harreman's own function verbatim (via the _need shim).
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
        print("[lowmem] Finished computing cell-cell communication analysis in %.3f seconds" % (time.time() - start))

    return


def _run_cell_communication_analysis_lowmem(
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
    device,
    verbose,
    gene_pair_chunk_size,
):
    """Chunked equivalent of harreman's ``run_cell_communication_analysis``. See the
    module docstring's CU-B section for the derivation of why gene-pair chunking is
    bit-identical to stock."""

    use_raw = (layer_key_p_test == "use_raw") & (layer_key_np_test == "use_raw")
    cells = adata.raw.obs.index.values.astype(str) if use_raw else adata.obs_names.values.astype(str)
    n_cells = len(cells)

    sample_specific = "sample_key" in adata.uns

    gene_pairs = adata.uns["gene_pairs"] if subset_gene_pairs is None else subset_gene_pairs
    # STOCK quirk, kept verbatim: `genes` is derived from the FULL adata.uns["gene_pairs"]
    # even when `subset_gene_pairs` is given -- not our bug to fix.
    genes = list(np.unique(list(flatten(adata.uns["gene_pairs"]))))
    adata.uns["genes"] = genes
    adata.uns["cells"] = cells

    gene_pairs_ind = []
    for pair in gene_pairs:
        idx1 = [genes.index(g) for g in pair[0]] if isinstance(pair[0], list) else genes.index(pair[0])
        idx2 = [genes.index(g) for g in pair[1]] if isinstance(pair[1], list) else genes.index(pair[1])
        gene_pairs_ind.append((idx1, idx2))
    adata.uns["gene_pairs_ind"] = gene_pairs_ind

    weights = make_weights_non_redundant(adata.obsp["weights"]).tocoo()
    weights = torch.sparse_coo_tensor(
        torch.tensor(np.vstack((weights.row, weights.col)), dtype=torch.long, device=device),
        torch.tensor(weights.data, dtype=torch.float64, device=device),
        torch.Size(weights.shape),
        device=device,
    )

    # Node degree (D): from the sparse weights directly, (n_cells,) -- cheap, no chunk.
    row_degrees = torch.sparse.sum(weights, dim=1).to_dense()
    col_degrees = torch.sparse.sum(weights, dim=0).to_dense()
    D = row_degrees + col_degrees
    adata.uns["D"] = D.cpu().numpy()

    gene_pairs_per_metabolite = adata.uns["gene_pairs_per_metabolite"]
    metabolite_gene_pair_df = pd.DataFrame.from_dict(gene_pairs_per_metabolite, orient="index").reset_index()
    metabolite_gene_pair_df = metabolite_gene_pair_df.rename(columns={"index": "metabolite"})
    metabolite_gene_pair_df["gene_pair"] = metabolite_gene_pair_df["gene_pair"].apply(
        lambda arr: [(sub_array[0], sub_array[1]) for sub_array in arr]
    )
    metabolite_gene_pair_df["gene_type"] = metabolite_gene_pair_df["gene_type"].apply(
        lambda arr: [(sub_array[0], sub_array[1]) for sub_array in arr]
    )
    metabolite_gene_pair_df = pd.concat(
        [
            metabolite_gene_pair_df["metabolite"],
            metabolite_gene_pair_df.explode("gene_pair")["gene_pair"],
            metabolite_gene_pair_df.explode("gene_type")["gene_type"],
        ],
        axis=1,
    ).reset_index(drop=True)

    if "LR_database" in adata.uns.keys():
        LR_database = adata.uns["LR_database"]
        df_merged = pd.merge(metabolite_gene_pair_df, LR_database, left_on="metabolite", right_on="interaction_name", how="left")
        LR_df = df_merged.dropna(subset=["pathway_name"])
        metabolite_gene_pair_df["metabolite"][metabolite_gene_pair_df.metabolite.isin(LR_df.metabolite)] = LR_df["pathway_name"]

    gene_pair_dict = {}
    for metabolite, group in metabolite_gene_pair_df.groupby("metabolite"):
        idxs = group["gene_pair"].apply(lambda gp: gene_pairs.index(gp) if gp in gene_pairs else None).dropna().tolist()
        idxs = [int(ind) for ind in idxs if ind is not None]
        if idxs:
            gene_pair_dict[metabolite] = idxs
    adata.uns["gene_pair_dict"] = gene_pair_dict

    n_gp = len(gene_pairs_ind)
    n_m = len(gene_pair_dict)
    same_gene_mask_full = torch.tensor([g1 == g2 for g1, g2 in gene_pairs], device=device)

    # Adaptive chunk size: bound n_cells * chunk to ~50M elements so the (chunk, n_cells)
    # / (n_cells, chunk) intermediates stay small regardless of n_cells.
    if gene_pair_chunk_size is None:
        chunk = max(1, 50_000_000 // max(n_cells, 1))
    else:
        chunk = max(1, int(gene_pair_chunk_size))

    # ================================ parametric ('p') ================================
    if test in ["parametric", "both"]:
        if verbose:
            print("[lowmem] Running the parametric test...")
        adata.uns["ccc_results"]["p"] = {"gp": {}, "m": {}}

        Wtot2 = torch.tensor((weights.data ** 2).sum(), device=device)  # verbatim: stock uses weights.data

        counts = counts_from_anndata(adata[cells, genes], layer_key_p_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        num_umi = counts.sum(dim=0)  # GLOBAL: does not depend on which gene pairs share a chunk

        cs_gp = torch.empty(n_gp, dtype=torch.float64, device=device)
        eg2_a = torch.empty(n_gp, dtype=torch.float64, device=device)
        eg2_b = torch.empty(n_gp, dtype=torch.float64, device=device)

        # -----------------------------------------------------------------------------
        # STOCK (harreman's run_cell_communication_analysis, ~lines 669-708): builds
        # counts_1/counts_2 as dense (n_gp, n_cells) stacks for ALL gene pairs at once,
        # then WX2t/WtX2t/WX1t/WtX1t as dense (n_cells, n_gp) tensors -- the OOM site at
        # Xenium/1M-cell scale. Kept verbatim below (commented) for reference / diffing
        # against upstream.
        #
        # STOCK: counts_1, counts_2 = [], []
        # STOCK: for (idx1, idx2) in gene_pairs_ind:
        # STOCK:     if isinstance(idx1, list):
        # STOCK:         c1 = counts[idx1, :].mean(dim=0) if mean == 'algebraic' else ...
        # STOCK:     else:
        # STOCK:         c1 = counts[idx1, :]
        # STOCK:     ... (same for idx2 -> c2) ...
        # STOCK:     counts_1.append(c1); counts_2.append(c2)
        # STOCK: counts_1 = torch.stack(counts_1); counts_2 = torch.stack(counts_2)
        # STOCK: counts_1 = standardize_counts(adata, counts_1, model, num_umi, sample_specific)
        # STOCK: counts_2 = standardize_counts(adata, counts_2, model, num_umi, sample_specific)
        # STOCK: WX2t = torch.sparse.mm(weights, counts_2.T)
        # STOCK: WtX2t = torch.sparse.mm(weights.transpose(0, 1), counts_2.T)
        # STOCK: cs_gp = (counts_1.T * WX2t).sum(0) + (counts_1.T * WtX2t).sum(0)
        # STOCK: same_gene_mask = torch.tensor([g1 == g2 for g1, g2 in gene_pairs], device=device)
        # STOCK: cs_gp[same_gene_mask] = cs_gp[same_gene_mask] / 2
        # STOCK: WX1t = torch.sparse.mm(weights, counts_1.T)
        # STOCK: WtX1t = torch.sparse.mm(weights.transpose(0, 1), counts_1.T)
        # STOCK: eg2_a = (WX1t + WtX1t).pow(2).sum(dim=0)
        # STOCK: eg2_b = (WX2t + WtX2t).pow(2).sum(dim=0)
        #
        # LOWMEM REPLACEMENT: cs_gp/eg2_a/eg2_b are each a SUM OVER THE CELL AXIS per gene
        # pair, and each output column of `sparse.mm` is independent of the others -- so
        # slicing the gene-pair axis into chunks and reducing each chunk separately
        # reproduces the exact same per-gene-pair value (module docstring). `num_umi`
        # stays global/unchunked (computed once, above).
        # -----------------------------------------------------------------------------
        for i0 in range(0, n_gp, chunk):
            sl = slice(i0, min(i0 + chunk, n_gp))
            gp_ind_chunk = gene_pairs_ind[sl]
            same_gene_chunk = same_gene_mask_full[sl]

            counts_1c, counts_2c = _prep_counts_1_2(counts, gp_ind_chunk, mean)
            counts_1c = standardize_counts(adata, counts_1c, model, num_umi, sample_specific)
            counts_2c = standardize_counts(adata, counts_2c, model, num_umi, sample_specific)

            WX2t_c = torch.sparse.mm(weights, counts_2c.T)
            WtX2t_c = torch.sparse.mm(weights.transpose(0, 1), counts_2c.T)
            cs_gp_c = (counts_1c.T * WX2t_c).sum(0) + (counts_1c.T * WtX2t_c).sum(0)
            cs_gp_c[same_gene_chunk] = cs_gp_c[same_gene_chunk] / 2
            cs_gp[sl] = cs_gp_c

            WX1t_c = torch.sparse.mm(weights, counts_1c.T)
            WtX1t_c = torch.sparse.mm(weights.transpose(0, 1), counts_1c.T)
            eg2_a[sl] = (WX1t_c + WtX1t_c).pow(2).sum(dim=0)
            eg2_b[sl] = (WX2t_c + WtX2t_c).pow(2).sum(dim=0)

        adata.uns["ccc_results"]["p"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()

        cs_m = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=False)
        adata.uns["ccc_results"]["p"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        eg2s_gp = (eg2_a, eg2_b)
        Z_gp, Z_m = compute_p_results(cs_gp, cs_m, gene_pairs_ind, Wtot2, eg2s_gp, gene_pair_dict)
        Z_gp_np = Z_gp.detach().cpu().numpy()
        Z_m_np = Z_m.detach().cpu().numpy()
        Z_pvals_gp = norm.sf(Z_gp_np)
        Z_pvals_m = norm.sf(Z_m_np)
        FDR_gp = multipletests(Z_pvals_gp, method="fdr_bh")[1]
        FDR_m = multipletests(Z_pvals_m, method="fdr_bh")[1]

        adata.uns["ccc_results"]["p"]["gp"]["Z"] = Z_gp_np
        adata.uns["ccc_results"]["p"]["gp"]["Z_pval"] = Z_pvals_gp
        adata.uns["ccc_results"]["p"]["gp"]["Z_FDR"] = FDR_gp
        adata.uns["ccc_results"]["p"]["m"]["Z"] = Z_m_np
        adata.uns["ccc_results"]["p"]["m"]["Z_pval"] = Z_pvals_m
        adata.uns["ccc_results"]["p"]["m"]["Z_FDR"] = FDR_m

        # Symmetric LC Z-score matrix -- small (n_unique_genes^2), verbatim from stock.
        genes_ = [tuple(i) if isinstance(i, list) else i for i in pd.Series([g for pair in gene_pairs for g in pair]).drop_duplicates()]
        gene_pairs_ = [(tuple(a) if isinstance(a, list) else a, tuple(b) if isinstance(b, list) else b) for a, b in gene_pairs]
        lc_zs = pd.DataFrame(np.zeros((len(genes_), len(genes_))), index=genes_, columns=genes_)
        for i, (g1, g2) in enumerate(gene_pairs_):
            lc_zs.iloc[genes_.index(g1), genes_.index(g2)] = Z_gp_np[i]
        np.fill_diagonal(lc_zs.values, 0)
        adata.uns["lc_zs"] = (lc_zs + lc_zs.T) / 2

        if verbose:
            print("[lowmem] Parametric test finished.")

    # ============================ non-parametric ('np') ==============================
    if test in ["non-parametric", "both"]:
        if verbose:
            print("[lowmem] Running the non-parametric test...")
        adata.uns["ccc_results"]["np"] = {"gp": {}, "m": {}}

        counts = counts_from_anndata(adata[cells, genes], layer_key_np_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)

        num_umi = counts.sum(dim=0) if center_counts_for_np_test else None  # GLOBAL

        # STOCK shortcut: when centering AND both tests ran, the 'np' observed score is
        # just a copy of the already-computed 'p' score (no recompute) -- kept verbatim.
        use_p_shortcut = center_counts_for_np_test and test == "both"

        cs_gp = None if use_p_shortcut else torch.empty(n_gp, dtype=torch.float64, device=device)

        # ---------------------------------------------------------------------------
        # STOCK (harreman's run_cell_communication_analysis, ~lines 794-871): allocates
        # perm_cs_gp_a/b as (n_gene_pairs, M) and perm_cs_m_a/b as (n_metabolites, M) --
        # already small (NO per-cell axis), so these are NOT the OOM site and are kept
        # as-is. The OOM site is the per-permutation WX2t/WtX2t computation, which stock
        # does for ALL gene pairs at once inside a single `for i in range(M)` loop. Kept
        # verbatim below (commented) for reference / diffing against upstream.
        #
        # STOCK: perm_cs_gp_a = torch.zeros((counts_1.shape[0], M), dtype=torch.float64, device=device)
        # STOCK: perm_cs_gp_b = torch.zeros_like(perm_cs_gp_a)
        # STOCK: perm_cs_m_a = torch.zeros((len(gene_pair_dict), M), dtype=torch.float64, device=device)
        # STOCK: perm_cs_m_b = torch.zeros_like(perm_cs_m_a)
        # STOCK: torch.manual_seed(seed)
        # STOCK: for i in tqdm(range(M), desc="Permutation test"):
        # STOCK:     idx = torch.randperm(n_cells, device=device)
        # STOCK:     c1_perm_a = counts_1.clone(); c2_perm_a = counts_2[:, idx]
        # STOCK:     c1_perm_a[same_gene_mask] = counts_1[same_gene_mask, :][:, idx]
        # STOCK:     WX2t_a = torch.sparse.mm(weights, c2_perm_a.T)
        # STOCK:     WtX2t_a = torch.sparse.mm(weights.transpose(0, 1), c2_perm_a.T)
        # STOCK:     cs_a = (c1_perm_a.T * WX2t_a).sum(0) + (c1_perm_a.T * WtX2t_a).sum(0)
        # STOCK:     cs_a[same_gene_mask] = cs_a[same_gene_mask] / 2
        # STOCK:     perm_cs_gp_a[:, i] = cs_a
        # STOCK:     cs_m_a = compute_metabolite_cs(cs_a, gene_pair_dict, interacting_cell_scores=False)
        # STOCK:     perm_cs_m_a[:, i] = cs_m_a
        # STOCK:     ... (mirror for arm b: permute counts_1 instead of counts_2) ...
        #
        # LOWMEM REPLACEMENT: restructure as OUTER loop over gene-pair chunks, INNER loop
        # over the M permutations. `torch.manual_seed(seed)` is re-issued at the START of
        # EACH chunk so every chunk replays the identical idx_0..idx_{M-1} sequence (only
        # `torch.randperm` consumes RNG in the loop body, so re-seeding reproduces it
        # exactly) -- this is what lets `counts_1`/`counts_2` stay chunked while still
        # matching stock's per-permutation-i, all-gene-pairs idx usage. Each chunk's
        # (chunk, M) slice of cs_a/cs_b is written into the correct rows of the full
        # (n_gp, M) perm_cs_gp_a/b, exactly reconstructing what stock builds in one pass.
        # Metabolite permutation scores (perm_cs_m_a/b) are deferred until AFTER all
        # chunks are filled (see below the loop) because a metabolite's gene pairs can
        # span more than one chunk -- summing a per-chunk-only partial score would not
        # match stock, which always sums the FULL (n_gp,) cs_a per permutation.
        # ---------------------------------------------------------------------------
        perm_cs_gp_a = torch.zeros((n_gp, M), dtype=torch.float64, device=device)
        perm_cs_gp_b = torch.zeros_like(perm_cs_gp_a)

        for i0 in range(0, n_gp, chunk):
            sl = slice(i0, min(i0 + chunk, n_gp))
            gp_ind_chunk = gene_pairs_ind[sl]
            same_gene_chunk = same_gene_mask_full[sl]

            counts_1c, counts_2c = _prep_counts_1_2(counts, gp_ind_chunk, mean)
            if center_counts_for_np_test:
                counts_1c = standardize_counts(adata, counts_1c, model, num_umi, sample_specific)
                counts_2c = standardize_counts(adata, counts_2c, model, num_umi, sample_specific)

            if not use_p_shortcut:
                WX2t_c = torch.sparse.mm(weights, counts_2c.T)
                WtX2t_c = torch.sparse.mm(weights.transpose(0, 1), counts_2c.T)
                cs_gp_c = (counts_1c.T * WX2t_c).sum(0) + (counts_1c.T * WtX2t_c).sum(0)
                cs_gp_c[same_gene_chunk] = cs_gp_c[same_gene_chunk] / 2
                cs_gp[sl] = cs_gp_c

            torch.manual_seed(seed)  # reseed EVERY chunk -> replays idx_0..idx_{M-1} identically
            for i in tqdm(range(M), desc="[lowmem] Permutation test (chunk)", disable=not verbose):
                idx = torch.randperm(n_cells, device=device)

                c1_perm_a = counts_1c.clone()
                c2_perm_a = counts_2c[:, idx]
                c1_perm_a[same_gene_chunk] = counts_1c[same_gene_chunk, :][:, idx]
                WX2t_a = torch.sparse.mm(weights, c2_perm_a.T)
                WtX2t_a = torch.sparse.mm(weights.transpose(0, 1), c2_perm_a.T)
                cs_a = (c1_perm_a.T * WX2t_a).sum(0) + (c1_perm_a.T * WtX2t_a).sum(0)
                cs_a[same_gene_chunk] = cs_a[same_gene_chunk] / 2
                perm_cs_gp_a[sl, i] = cs_a

                c2_perm_b = counts_2c.clone()
                c1_perm_b = counts_1c[:, idx]
                c2_perm_b[same_gene_chunk] = counts_2c[same_gene_chunk, :][:, idx]
                WX2t_b = torch.sparse.mm(weights, c2_perm_b.T)
                WtX2t_b = torch.sparse.mm(weights.transpose(0, 1), c2_perm_b.T)
                cs_b = (c1_perm_b.T * WX2t_b).sum(0) + (c1_perm_b.T * WtX2t_b).sum(0)
                cs_b[same_gene_chunk] = cs_b[same_gene_chunk] / 2
                perm_cs_gp_b[sl, i] = cs_b

        if use_p_shortcut:
            adata.uns["ccc_results"]["np"]["gp"]["cs"] = np.array(adata.uns["ccc_results"]["p"]["gp"]["cs"])
            adata.uns["ccc_results"]["np"]["m"]["cs"] = np.array(adata.uns["ccc_results"]["p"]["m"]["cs"])
            cs_gp = torch.as_tensor(adata.uns["ccc_results"]["np"]["gp"]["cs"], dtype=torch.float64, device=device)
            cs_m = torch.as_tensor(adata.uns["ccc_results"]["np"]["m"]["cs"], dtype=torch.float64, device=device)
        else:
            adata.uns["ccc_results"]["np"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()
            cs_m = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=False)
            adata.uns["ccc_results"]["np"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        # Metabolite permutation scores, derived AFTER chunking from the FULL (n_gp,)
        # column per permutation -- the exact same 1-D-vector call stock makes per
        # iteration, so this is bit-for-bit identical (see module docstring).
        perm_cs_m_a = torch.zeros((n_m, M), dtype=torch.float64, device=device)
        perm_cs_m_b = torch.zeros_like(perm_cs_m_a)
        for i in range(M):
            perm_cs_m_a[:, i] = compute_metabolite_cs(perm_cs_gp_a[:, i], gene_pair_dict, interacting_cell_scores=False)
            perm_cs_m_b[:, i] = compute_metabolite_cs(perm_cs_gp_b[:, i], gene_pair_dict, interacting_cell_scores=False)

        adata.uns["ccc_results"]["np"]["gp"]["perm_cs_a"] = perm_cs_gp_a.detach().cpu().numpy()
        adata.uns["ccc_results"]["np"]["gp"]["perm_cs_b"] = perm_cs_gp_b.detach().cpu().numpy()
        adata.uns["ccc_results"]["np"]["m"]["perm_cs_a"] = perm_cs_m_a.detach().cpu().numpy()
        adata.uns["ccc_results"]["np"]["m"]["perm_cs_b"] = perm_cs_m_b.detach().cpu().numpy()

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

        adata.uns["ccc_results"]["np"]["gp"]["pval"] = pvals_gp.cpu().numpy()
        adata.uns["ccc_results"]["np"]["gp"]["FDR"] = multipletests(pvals_gp.cpu().numpy(), method="fdr_bh")[1]
        adata.uns["ccc_results"]["np"]["m"]["pval"] = pvals_m.cpu().numpy()
        adata.uns["ccc_results"]["np"]["m"]["FDR"] = multipletests(pvals_m.cpu().numpy(), method="fdr_bh")[1]

        if verbose:
            print("[lowmem] Non-parametric test finished.")

    return


# ============================================================================================
# CU-C: compute_ct_cell_communication_lowmem (cell-type-AWARE aggregate scores)
# ============================================================================================

def compute_ct_cell_communication_lowmem(
    adata,
    layer_key_p_test=None,
    layer_key_np_test=None,
    model: str = None,
    cell_type_key: str = None,
    center_counts_for_np_test: bool = False,
    subset_gene_pairs=None,
    subset_metabolites=None,
    fix_gp: bool = False,
    M: int = 1000,
    seed: int = 42,
    test: str = "both",
    mean: str = "algebraic",
    check_analytic_null: bool = False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose: bool = False,
    gene_pair_chunk_size: int = None,
):
    """Memory-safe equivalent of ``harreman.tools.compute_ct_cell_communication``.

    Same signature (plus ``gene_pair_chunk_size``) and the same ``adata.uns['ct_ccc_results']``
    outputs (``D``, ``genes``, ``cells``, ``gene_pairs_ind``, ``gene_pairs_ind_per_ct_pair``,
    ``gene_pairs_per_ct_pair_ind``, ``gene_pair_dict``, ``cell_types``) -- see module docstring
    (CU-C section) for what differs internally.

    Parameters
    ----------
    gene_pair_chunk_size : int, optional
        Number of gene pairs processed per chunk. ``None`` (default) picks an adaptive size
        ``max(1, 50_000_000 // n_cells)``. A value ``>= n_gene_pairs`` reproduces stock's
        single-pass behavior exactly (one chunk = the whole thing).
    """
    start = time.time()
    if verbose:
        print("[lowmem] Starting cell type-aware cell-cell communication analysis...")

    # Both guards mirror the CU-A/B pattern: reject flag combinations that would
    # re-introduce the removed (cells, ...) intermediates or that don't compose with the
    # chunk-reuse-across-M restructuring below. Neither is ever set by harreman_funcs.py.
    if check_analytic_null:
        raise NotImplementedError(
            "check_analytic_null=True is not supported in the low-memory version "
            "(harreman_funcs.py never sets it; mirrors the CU-A/CU-B guard)."
        )
    if fix_gp:
        raise NotImplementedError(
            "fix_gp=True is not supported in the low-memory version -- it rebuilds "
            "weights_ct_pairs from a fresh (non-cell-type-stratified) per-cell permutation "
            "every iteration, a different RNG-consumption shape that does not compose with "
            "the chunk-reuse-across-M restructuring used for fix_gp=False. "
            "harreman_funcs.py always calls with fix_gp=False."
        )

    adata.uns["ct_ccc_results"] = {}

    if test not in ["both", "parametric", "non-parametric"]:
        raise ValueError('The "test" variable should be one of ["both", "parametric", "non-parametric"].')
    if mean not in ["algebraic", "geometric"]:
        raise ValueError('The "mean" variable should be one of ["algebraic", "geometric"].')

    if "cell_type_key" in adata.uns and cell_type_key is None:
        cell_type_key = adata.uns["cell_type_key"]
    elif "cell_type_key" not in adata.uns and cell_type_key is None:
        raise ValueError('Please provide the "cell_type_key" argument.')

    adata.uns["layer_key_p_test"] = layer_key_p_test
    adata.uns["layer_key_np_test"] = layer_key_np_test
    adata.uns["model"] = model
    adata.uns["cell_type_key"] = cell_type_key
    adata.uns["center_counts_for_np_test"] = center_counts_for_np_test
    adata.uns["mean"] = mean

    _run_ct_cell_communication_analysis_lowmem(
        adata,
        layer_key_p_test,
        layer_key_np_test,
        model,
        cell_type_key,
        center_counts_for_np_test,
        subset_gene_pairs,
        subset_metabolites,
        M,
        seed,
        test,
        mean,
        device,
        verbose,
        gene_pair_chunk_size,
    )

    if verbose:
        print("[lowmem] Obtaining cell type-aware communication results...")
    # Small: operates on the already-reduced (n_ct_pairs, n_gp)/(n_ct_pairs, n_m) arrays,
    # so no memory issue -- call harreman's own function verbatim (via the _need shim).
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
        print(
            "[lowmem] Finished computing cell type-aware cell-cell communication analysis "
            "in %.3f seconds" % (time.time() - start)
        )

    return


def _run_ct_cell_communication_analysis_lowmem(
    adata,
    layer_key_p_test,
    layer_key_np_test,
    model,
    cell_type_key,
    center_counts_for_np_test,
    subset_gene_pairs,
    subset_metabolites,
    M,
    seed,
    test,
    mean,
    device,
    verbose,
    gene_pair_chunk_size,
):
    """Chunked equivalent of harreman's ``run_ct_cell_communication_analysis``. See the
    module docstring's CU-C section for the derivation of why gene-pair chunking is
    bit-identical to stock. ``fix_gp`` is always False here (guarded in the caller)."""

    use_raw = (layer_key_p_test == "use_raw") & (layer_key_np_test == "use_raw")
    obs = adata.raw.obs if use_raw else adata.obs
    cells = adata.raw.obs.index.values.astype(str) if use_raw else adata.obs_names.values.astype(str)

    sample_specific = "sample_key" in adata.uns
    fix_ct = True if adata.uns["fix_ct"] else False

    gene_pairs = adata.uns["gene_pairs"] if subset_gene_pairs is None else subset_gene_pairs
    # STOCK quirk, kept verbatim (matches CU-B's identical note): `genes` derives from the
    # FULL adata.uns["gene_pairs"] even when `subset_gene_pairs` is given.
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
    n_cells = len(cells)
    n_ct_pairs = len(cell_type_pairs)

    # weights_ct_pairs (3D torch sparse tensor) built ONCE -- sparse, not the OOM site.
    weights_ct_pairs = create_weights_ct_pairs(weights.tocoo(), cell_types, cell_type_pairs, device)
    # Per-ct-pair coalesced 2D slices, ALSO built ONCE and reused across every chunk/
    # permutation below (stock rebuilds this slice+coalesce on every single use -- inside
    # every ct-pair loop, inside every permutation; hoisting it is a pure efficiency
    # refactor, no numerics change, since sparse.mm's result depends only on tensor content).
    Ws = [weights_ct_pairs[ct_pair].coalesce() for ct_pair in range(n_ct_pairs)]

    row_degrees = torch.sparse.sum(weights_ct_pairs, dim=2).to_dense()
    col_degrees = torch.sparse.sum(weights_ct_pairs, dim=1).to_dense()
    D = row_degrees + col_degrees
    if used_ct_pairs_set < all_cell_types:
        D_full = torch.zeros((n_ct_pairs, adata.shape[0]), device=weights_ct_pairs.device, dtype=weights_ct_pairs.dtype)
        D_full[:, keep_indices] = D
        adata.uns["D"] = D_full.cpu().numpy()
    else:
        adata.uns["D"] = D.cpu().numpy()

    gene_pairs_ind = []
    for pair in gene_pairs:
        idx1 = [genes.index(g) for g in pair[0]] if isinstance(pair[0], list) else genes.index(pair[0])
        idx2 = [genes.index(g) for g in pair[1]] if isinstance(pair[1], list) else genes.index(pair[1])
        gene_pairs_ind.append((idx1, idx2))
    adata.uns["gene_pairs_ind"] = gene_pairs_ind

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

    gp_metab = adata.uns["gene_pairs_per_metabolite"]
    metabolite_gene_pair_df = (
        pd.DataFrame.from_dict(gp_metab, orient="index")
        .rename_axis("metabolite")
        .explode(["gene_pair", "gene_type"])
        .reset_index()
    )
    if "LR_database" in adata.uns:
        merged = metabolite_gene_pair_df.merge(
            adata.uns["LR_database"], left_on="metabolite", right_on="interaction_name", how="left"
        )
        LR_df = merged.dropna(subset=["pathway_name"])
        metabolite_gene_pair_df.loc[
            metabolite_gene_pair_df.metabolite.isin(LR_df.metabolite), "metabolite"
        ] = LR_df["pathway_name"].values
    if subset_metabolites:
        metabolite_gene_pair_df = metabolite_gene_pair_df[metabolite_gene_pair_df.metabolite.isin(subset_metabolites)]

    gene_pair_dict = {}
    for metabolite, group in metabolite_gene_pair_df.groupby("metabolite"):
        idxs = group["gene_pair"].apply(lambda gp: gene_pairs.index(gp) if gp in gene_pairs else None).dropna().tolist()
        idxs = [int(ind) for ind in idxs if ind is not None]
        if idxs:
            gene_pair_dict[metabolite] = idxs
    adata.uns["gene_pair_dict"] = gene_pair_dict

    n_gp = len(gene_pairs_ind)
    n_m = len(gene_pair_dict)

    # Adaptive chunk size: bound n_cells * chunk to ~50M elements, same rule as CU-B.
    if gene_pair_chunk_size is None:
        chunk = max(1, 50_000_000 // max(n_cells, 1))
    else:
        chunk = max(1, int(gene_pair_chunk_size))

    # ================================ parametric ('p') ================================
    if test in ["parametric", "both"]:
        if verbose:
            print("[lowmem] Running the parametric test...")
        adata.uns["ct_ccc_results"]["p"] = {"gp": {}, "m": {}}

        weights_sq_data = weights_ct_pairs.values() ** 2
        weights_sq = torch.sparse_coo_tensor(
            weights_ct_pairs.indices(), weights_sq_data, weights_ct_pairs.shape, device=weights_ct_pairs.device
        )
        Wtot2 = torch.sparse.sum(weights_sq, dim=(1, 2)).to_dense()

        counts = counts_from_anndata(adata[cells, genes], layer_key_p_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        num_umi = counts.sum(dim=0)  # GLOBAL: does not depend on which gene pairs share a chunk

        # STOCK QUIRK (important -- verified empirically, see module docstring): stock
        # allocates `cs_gp = torch.zeros((n_ct_pairs, n_cells_ish), device=...)` with NO
        # dtype, so it is torch.float32 (PyTorch's default), NOT float64 like every other
        # tensor in this pipeline -- assigning the float64 `(counts_1.T * WX2t).sum(0)`
        # result into it silently rounds each entry to float32 precision (~1e-8 relative,
        # exactly what a naive float64 reimplementation of this function would miss).
        # Reproduced verbatim: this is a genuine stock numerical quirk, not something we
        # are entitled to "fix" if the goal is bit-identical output for a given seed.
        cs_gp = torch.zeros((n_ct_pairs, n_gp), device=device)  # float32, matches stock exactly
        EG2_gp = torch.zeros_like(cs_gp) if fix_ct else Wtot2  # float32 if fix_ct, else Wtot2 (float64)

        # -----------------------------------------------------------------------------
        # STOCK (harreman's run_ct_cell_communication_analysis, ~lines 1148-1195): builds
        # counts_1/counts_2 as dense (n_gp, n_cells) stacks for ALL gene pairs at once, then
        # per ct_pair a dense (n_cells, n_gp) WX2t = sparse.mm(weights_ct_pairs[ct_pair],
        # counts_2.T), cs_gp[ct_pair] = (counts_1.T * WX2t).sum(0) -- the OOM site (sum over
        # the cell axis => gene-pair columns chunk exactly like CU-B). Kept verbatim below
        # (commented) for reference / diffing against upstream.
        #
        # STOCK: counts_1, counts_2 = [], []  (built for ALL gene_pairs_ind, same _prep_counts_1_2 logic)
        # STOCK: counts_1 = standardize_ct_counts(adata, counts_1, model, num_umi, sample_specific, cell_types)
        # STOCK: counts_2 = standardize_ct_counts(adata, counts_2, model, num_umi, sample_specific, cell_types)
        # STOCK: cs_gp = torch.zeros((len(cell_type_pairs), counts_1.shape[0]), device=counts_1.device)
        # STOCK: for ct_pair in range(len(cell_type_pairs)):
        # STOCK:     W = weights_ct_pairs[ct_pair].coalesce()
        # STOCK:     WX2t = torch.sparse.mm(W, counts_2.T)
        # STOCK:     cs_gp[ct_pair] = (counts_1.T * WX2t).sum(0)
        # STOCK: EG2_gp = torch.zeros_like(cs_gp) if fix_ct or fix_gp else Wtot2
        # STOCK: if fix_ct:
        # STOCK:     for ct_pair in range(len(cell_type_pairs)):
        # STOCK:         W = weights_ct_pairs[ct_pair].coalesce()
        # STOCK:         W_sq = torch.sparse_coo_tensor(W.indices(), W.values() ** 2, W.shape, device=W.device)
        # STOCK:         X1_sq = counts_1 ** 2
        # STOCK:         EG2_gp[ct_pair] = torch.sparse.mm(W_sq, X1_sq.T).sum(0)
        # STOCK: elif fix_gp: ... (never reached here -- fix_gp is always False, guarded above)
        #
        # LOWMEM REPLACEMENT: cs_gp[ct_pair, :] / EG2_gp[ct_pair, :] are each a SUM OVER THE
        # CELL AXIS per (ct_pair, gene pair) -- same chunking argument as CU-B, applied per
        # ct_pair inside each chunk. No same_gene_mask halving here: the ct path is NOT
        # symmetrized (05_harreman_reference.md sec.3) -- stock's own code has no such mask
        # for the ct cs, verified by inspection of the block above.
        # -----------------------------------------------------------------------------
        for i0 in range(0, n_gp, chunk):
            sl = slice(i0, min(i0 + chunk, n_gp))
            gp_ind_chunk = gene_pairs_ind[sl]

            counts_1c, counts_2c = _prep_counts_1_2(counts, gp_ind_chunk, mean)
            counts_1c = standardize_ct_counts(adata, counts_1c, model, num_umi, sample_specific, cell_types)
            counts_2c = standardize_ct_counts(adata, counts_2c, model, num_umi, sample_specific, cell_types)

            for ct_pair in range(n_ct_pairs):
                WX2t_c = torch.sparse.mm(Ws[ct_pair], counts_2c.T)
                cs_gp[ct_pair, sl] = (counts_1c.T * WX2t_c).sum(0)

            if fix_ct:
                X1_sq_c = counts_1c ** 2
                for ct_pair in range(n_ct_pairs):
                    W = Ws[ct_pair]
                    W_sq = torch.sparse_coo_tensor(W.indices(), W.values() ** 2, W.shape, device=W.device)
                    EG2_gp[ct_pair, sl] = torch.sparse.mm(W_sq, X1_sq_c.T).sum(0)

        adata.uns["ct_ccc_results"]["p"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()
        cs_m = compute_metabolite_cs_ct(
            cs_gp, cell_type_key, gene_pair_dict, gene_pairs_per_ct_pair_ind, ct_specific_gene_pairs, interacting_cell_scores=False
        )
        adata.uns["ct_ccc_results"]["p"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        Z_gp, Z_m = compute_ct_p_results(cs_gp, cs_m, gene_pairs_per_ct_pair_ind, ct_specific_gene_pairs, EG2_gp, cell_type_key, gene_pair_dict)
        Z_gp_np = Z_gp.detach().cpu().numpy()
        Z_m_np = Z_m.detach().cpu().numpy()
        Z_pvals_gp = norm.sf(Z_gp_np)
        Z_pvals_m = norm.sf(Z_m_np)
        FDR_gp = multipletests(Z_pvals_gp.flatten(), method="fdr_bh")[1].reshape(Z_pvals_gp.shape)
        FDR_m = multipletests(Z_pvals_m.flatten(), method="fdr_bh")[1].reshape(Z_pvals_m.shape)

        adata.uns["ct_ccc_results"]["p"]["gp"]["Z"] = Z_gp_np
        adata.uns["ct_ccc_results"]["p"]["gp"]["Z_pval"] = Z_pvals_gp
        adata.uns["ct_ccc_results"]["p"]["gp"]["Z_FDR"] = FDR_gp
        adata.uns["ct_ccc_results"]["p"]["m"]["Z"] = Z_m_np
        adata.uns["ct_ccc_results"]["p"]["m"]["Z_pval"] = Z_pvals_m
        adata.uns["ct_ccc_results"]["p"]["m"]["Z_FDR"] = FDR_m

        if verbose:
            print("[lowmem] Parametric test finished.")

    # ============================ non-parametric ('np') ==============================
    if test in ["non-parametric", "both"]:
        if verbose:
            print("[lowmem] Running the non-parametric test...")
        adata.uns["ct_ccc_results"]["np"] = {"gp": {}, "m": {}}

        counts = counts_from_anndata(adata[cells, genes], layer_key_np_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        num_umi = counts.sum(dim=0) if center_counts_for_np_test else None  # GLOBAL

        # STOCK shortcut: when centering AND both tests ran, the 'np' observed score is
        # just a copy of the already-computed 'p' score (no recompute) -- kept verbatim.
        use_p_shortcut = center_counts_for_np_test and test == "both"

        if use_p_shortcut:
            adata.uns["ct_ccc_results"]["np"]["gp"]["cs"] = np.array(adata.uns["ct_ccc_results"]["p"]["gp"]["cs"])
            adata.uns["ct_ccc_results"]["np"]["m"]["cs"] = np.array(adata.uns["ct_ccc_results"]["p"]["m"]["cs"])
        else:
            # STOCK QUIRK: same float32-by-omitted-dtype allocation as the parametric
            # `cs_gp` above (stock: `cs_gp = torch.zeros((len(cell_type_pairs), ...),
            # device=...)`, no dtype) -- reproduced verbatim.
            cs_gp = torch.zeros((n_ct_pairs, n_gp), device=device)  # float32, matches stock
            for i0 in range(0, n_gp, chunk):
                sl = slice(i0, min(i0 + chunk, n_gp))
                gp_ind_chunk = gene_pairs_ind[sl]
                counts_1c, counts_2c = _prep_counts_1_2(counts, gp_ind_chunk, mean)
                if center_counts_for_np_test:
                    counts_1c = standardize_ct_counts(adata, counts_1c, model, num_umi, sample_specific, cell_types)
                    counts_2c = standardize_ct_counts(adata, counts_2c, model, num_umi, sample_specific, cell_types)
                for ct_pair in range(n_ct_pairs):
                    WX2t_c = torch.sparse.mm(Ws[ct_pair], counts_2c.T)
                    cs_gp[ct_pair, sl] = (counts_1c.T * WX2t_c).sum(0)

            adata.uns["ct_ccc_results"]["np"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()
            cs_m = compute_metabolite_cs_ct(
                cs_gp, cell_type_key, gene_pair_dict, gene_pairs_per_ct_pair_ind, ct_specific_gene_pairs, interacting_cell_scores=False
            )
            adata.uns["ct_ccc_results"]["np"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        # ---------------------------------------------------------------------------
        # STOCK (~lines 1266-1317): allocates perm_cs_gp (n_ct_pairs, n_gp, M) and
        # perm_cs_m (n_ct_pairs, n_m, M) -- NO per-cell axis, small, NOT the OOM site (kept
        # full-size here too, matching CU-B's treatment of its own small perm_cs_gp_a/b).
        # The OOM site is the per-permutation cs_gp computation, which stock does for ALL
        # gene pairs at once inside the M loop using dense (n_cells, n_gp) WX2t. Kept
        # verbatim below (commented) for reference / diffing against upstream.
        #
        # STOCK: perm_cs_gp = torch.zeros((len(cell_type_pairs), counts_1.shape[0], M), dtype=torch.float64, device=device)
        # STOCK: perm_cs_m = torch.zeros((len(cell_type_pairs), len(gene_pair_dict), M), dtype=torch.float64, device=device)
        # STOCK: torch.manual_seed(seed)
        # STOCK: for i in tqdm(range(M), desc="Permutation test"):
        # STOCK:     cell_type_labels = torch.tensor(cell_types.astype('category').cat.codes.values, device=counts_1.device)
        # STOCK:     idx = torch.empty_like(cell_type_labels, dtype=torch.int64)
        # STOCK:     for ct in torch.unique(cell_type_labels):
        # STOCK:         ct_mask = cell_type_labels == ct
        # STOCK:         ct_indices = torch.nonzero(ct_mask, as_tuple=True)[0]
        # STOCK:         permuted_indices = ct_indices[torch.randperm(len(ct_indices))]
        # STOCK:         idx[ct_indices] = permuted_indices
        # STOCK:     c1_perm = counts_1 if fix_ct else counts_1[:, idx.long()]
        # STOCK:     c2_perm = counts_2[:, idx.long()]
        # STOCK:     cs_gp = torch.zeros((len(cell_type_pairs), c1_perm.shape[0]), device=c1_perm.device)
        # STOCK:     for ct_pair in range(len(cell_type_pairs)):
        # STOCK:         W = weights_ct_pairs[ct_pair].coalesce()
        # STOCK:         WX2t = torch.sparse.mm(W, c2_perm.T)
        # STOCK:         cs_gp[ct_pair] = (c1_perm.T * WX2t).sum(0)
        # STOCK:     perm_cs_gp[:, :, i] = cs_gp
        # STOCK:     cs_m = compute_metabolite_cs_ct(cs_gp, cell_type_key, gene_pair_dict, gene_pairs_per_ct_pair_ind, ct_specific_gene_pairs, interacting_cell_scores=False)
        # STOCK:     perm_cs_m[:, :, i] = cs_m
        #
        # LOWMEM REPLACEMENT: outer loop over gene-pair chunks, `torch.manual_seed(seed)`
        # re-issued at the START of each chunk -- the stratified idx per permutation i is
        # drawn from `cell_types`/`torch.randperm` only (never from gene pairs/counts), so
        # it is identical for every chunk; re-seeding replays the exact idx_0..idx_{M-1}
        # sequence stock would draw once for all gene pairs. Each chunk's (n_ct_pairs, chunk)
        # slice is written directly into the correct columns of the full (n_ct_pairs, n_gp,
        # M) perm_cs_gp. Metabolite permutation scores (perm_cs_m) are derived AFTER all
        # chunks are filled, from a CLONE of each permutation's full (n_ct_pairs, n_gp)
        # column (compute_metabolite_cs_ct mutates its input in place for ct-specific gene
        # pairs -- cloning keeps the stored perm_cs_gp pristine, exactly as stock's own
        # per-iteration `perm_cs_gp[:, :, i] = cs_gp` index-assignment already guarantees a
        # value-copy before its own throwaway `cs_gp` gets mutated -- see module docstring).
        # ---------------------------------------------------------------------------
        perm_cs_gp = torch.zeros((n_ct_pairs, n_gp, M), dtype=torch.float64, device=device)

        for i0 in range(0, n_gp, chunk):
            sl = slice(i0, min(i0 + chunk, n_gp))
            gp_ind_chunk = gene_pairs_ind[sl]
            counts_1c, counts_2c = _prep_counts_1_2(counts, gp_ind_chunk, mean)
            # STOCK: the permutation loop reuses the SAME (conditionally standardized)
            # counts_1/counts_2 prepared just above the loop -- not a fresh raw copy. Must
            # be applied here too (missed in an earlier draft): without it, pval/FDR for
            # center_counts_for_np_test=True silently used un-standardized counts.
            if center_counts_for_np_test:
                counts_1c = standardize_ct_counts(adata, counts_1c, model, num_umi, sample_specific, cell_types)
                counts_2c = standardize_ct_counts(adata, counts_2c, model, num_umi, sample_specific, cell_types)

            torch.manual_seed(seed)  # reseed EVERY chunk -> replays idx_0..idx_{M-1} identically
            for i in tqdm(range(M), desc="[lowmem] Permutation test (chunk)", disable=not verbose):
                cell_type_labels = torch.tensor(cell_types.astype("category").cat.codes.values, device=counts_1c.device)
                idx = torch.empty_like(cell_type_labels, dtype=torch.int64)
                for ct in torch.unique(cell_type_labels):
                    ct_mask = cell_type_labels == ct
                    ct_indices = torch.nonzero(ct_mask, as_tuple=True)[0]
                    permuted_indices = ct_indices[torch.randperm(len(ct_indices))]
                    idx[ct_indices] = permuted_indices

                c1_perm = counts_1c if fix_ct else counts_1c[:, idx.long()]
                c2_perm = counts_2c[:, idx.long()]

                # STOCK QUIRK (same float32-by-omitted-dtype allocation as the observed
                # cs_gp above): stock's per-permutation local `cs_gp = torch.zeros((n_ct_
                # pairs, c1_perm.shape[0]), device=...)` is ALSO float32, and THAT is what
                # gets copied into perm_cs_gp (`perm_cs_gp[:, :, i] = cs_gp`) -- so each
                # permutation's stored value already carries the float32 rounding before
                # ever reaching the float64 perm_cs_gp storage. A per-chunk float32 buffer
                # here reproduces exactly that rounding (elementwise, so slicing the
                # gene-pair axis into chunks changes nothing about which value each entry
                # rounds to).
                cs_gp_chunk = torch.zeros((n_ct_pairs, sl.stop - sl.start), device=device)  # float32
                for ct_pair in range(n_ct_pairs):
                    WX2t = torch.sparse.mm(Ws[ct_pair], c2_perm.T)
                    cs_gp_chunk[ct_pair] = (c1_perm.T * WX2t).sum(0)
                perm_cs_gp[:, sl, i] = cs_gp_chunk

        adata.uns["ct_ccc_results"]["np"]["gp"]["perm_cs"] = perm_cs_gp.detach().cpu().numpy()

        # STOCK QUIRK continued: stock's per-permutation `compute_metabolite_cs_ct(cs_gp,
        # ...)` call sums gene-pair columns using cs_gp's actual (float32) dtype, so the
        # per-metabolite SUM is itself accumulated in float32 arithmetic, not float64 --
        # reproduced by casting DOWN to float32 (lossless: these values were only ever
        # exactly representable in float32 to begin with, see cs_gp_chunk above) before
        # calling compute_metabolite_cs_ct, then letting the float64 perm_cs_m assignment
        # upcast the float32 result, exactly mirroring stock's `perm_cs_m[:, :, i] = cs_m`.
        perm_cs_m = torch.zeros((n_ct_pairs, n_m, M), dtype=torch.float64, device=device)
        for i in range(M):
            # match the other STOCK-QUIRK sites: stock's cs_gp is default-dtype (float32
            # via dtype-omitted torch.zeros), so track default dtype rather than hardcode.
            cs_gp_i = perm_cs_gp[:, :, i].to(torch.get_default_dtype()).clone()
            perm_cs_m[:, :, i] = compute_metabolite_cs_ct(
                cs_gp_i, cell_type_key, gene_pair_dict, gene_pairs_per_ct_pair_ind, ct_specific_gene_pairs, interacting_cell_scores=False
            )
        adata.uns["ct_ccc_results"]["np"]["m"]["perm_cs"] = perm_cs_m.detach().cpu().numpy()

        # STOCK (~lines 1319-1328): pure NUMPY, float64, NO float32 cast (unlike CU-A/B's
        # torch/float32 p-value path) -- reproduced verbatim.
        x_gp = np.sum(
            adata.uns["ct_ccc_results"]["np"]["gp"]["perm_cs"] > adata.uns["ct_ccc_results"]["np"]["gp"]["cs"][:, :, np.newaxis], axis=2
        )
        x_m = np.sum(
            adata.uns["ct_ccc_results"]["np"]["m"]["perm_cs"] > adata.uns["ct_ccc_results"]["np"]["m"]["cs"][:, :, np.newaxis], axis=2
        )

        pvals_gp = (x_gp + 1) / (M + 1)
        pvals_m = (x_m + 1) / (M + 1)

        adata.uns["ct_ccc_results"]["np"]["gp"]["pval"] = pvals_gp
        adata.uns["ct_ccc_results"]["np"]["gp"]["FDR"] = multipletests(pvals_gp.flatten(), method="fdr_bh")[1].reshape(pvals_gp.shape)
        adata.uns["ct_ccc_results"]["np"]["m"]["pval"] = pvals_m
        adata.uns["ct_ccc_results"]["np"]["m"]["FDR"] = multipletests(pvals_m.flatten(), method="fdr_bh")[1].reshape(pvals_m.shape)

        if verbose:
            print("[lowmem] Non-parametric test finished.")

    adata.uns["cell_types"] = cell_types.tolist() if cell_type_key else None

    return


# --- sanity check (run on Savio, small subset) ----------------------------------------
# To trust this before a full run, compare against the stock function on a tiny adata:
#
#   import harreman, numpy as np
#   from cell_communication_lowmem import compute_interacting_cell_scores_lowmem
#   sub = adata[:2000].copy()               # small enough for the stock version
#   a = sub.copy(); b = sub.copy()
#   harreman.tools.compute_interacting_cell_scores(a, test='both', M=200, seed=0)
#   compute_interacting_cell_scores_lowmem(b, test='both', M=200, seed=0)
#   for t in ['p','np']:
#       for g in ['gp','m']:
#           np.testing.assert_allclose(a.uns['interacting_cell_results'][t][g]['cs'],
#                                      b.uns['interacting_cell_results'][t][g]['cs'])
#   # p-values use RNG; with the same seed/device they should match closely.
