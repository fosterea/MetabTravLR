# SpaceTravLR adaptation — working documentation

Docs we (Foster (Andrew's son) + Claude) are building to adapt SpaceTravLR into a tool that scores
**which metabolites (import/export gene pairs) modulate which target gene sets**, by
**analyzing the model's learned coefficients directly** (no perturbation prediction).

This folder is deliberately separate from the repo's own `docs/` (Sphinx site) — we don't
touch that. It's the **source of truth** for project context (replaces Claude's private
memory; the root `../../CLAUDE.md` points here).

## Contents
- `00_overview.md` — developer mental model of how SpaceTravLR works + where our adaptation
  plugs in. **Start here.**
- `02_metab_integration_notes.md` — planning scratchpad: **harreman input data contract**,
  metabolite→modulator mapping, **tractability/complexity analysis** for 100k–1M-cell
  Xenium, and the open design decisions. **Read after 00.**
- `04_decisions_and_state.md` — **living decision log + project state** (who/how we work,
  decisions made, open questions). Update this when a decision changes.
- `03_dev_process_and_testing.md` — the **dev/review agent loop** + **layered testing
  regimen** (spoof-logic + gene-focused real-data tests), and model choices.
- `01_pipeline_deep_dive.md` — exact call sites + line refs and the **surgical wiring map**
  (change units CU-1…CU-5) for adding metabolites as a new modulator group. Read when we
  start coding.
- `06_efficiency_and_dataflow.md` — **how a training run flows with a memory/compute lens**:
  where the big tensors are, what scales with cells vs modulators, the efficiency improvements
  made + still to make, and answers to "what would metabolite-only training save?" and "how does
  the CNN-over-a-grid scale?". Read to reason about large-dataset performance.
- `07_nbhd_percell_chunking_plan.md` — **DIAGNOSED-not-implemented plan** for the next harreman
  memory bottleneck: the **per-cell "neighborhood" analysis OOMs at ≥600k cells** (the aggregate
  CU-A–D fix doesn't cover it). Root cause + three fix options + testing, written for a future
  agent. Read alongside `05` §5.
- `05_harreman_reference.md` — **what harreman actually does**, read from its source: the
  pipeline, output tables, the **`CT1→CT2` "arrow" is NOT a direction** (undirected interface;
  sorted-label artifact), and the **per-cell "neighborhood" analysis + its GPU-OOM** and fix
  directions. Read before consuming harreman output or debugging the OOM.
- `paper_fulltext.txt` — plain-text extraction of `../Space_TravLR_Preprint.pdf` (bioRxiv
  2025.11.13.688264), for quick grep/reference without re-parsing the PDF. Note: equations
  came through as garbled Unicode; use the PDF for the exact math.
- `easy_download/harreman_outputs/` — example of Foster's metabolite (harreman) output data;
  the real/full data lives on the Savio cluster alongside `adata.h5ad`.

## Key facts to remember
- Repo was renamed **SpaceOracle → SpaceTravLR** (both names appear in code/paths).
- Two phases: **train** (`SpaceShip.setup_` → `run_spacetravlr` → per-gene betadata parquet)
  and **simulate** (`GeneFactory.perturb`). **We only care about the train-phase
  coefficients.**
- Metabolite = (export gene ≈ ligand, import gene ≈ receptor); reuses the L–R *computation*
  but is added as its **own new modulator group** with a distinct separator
  (`beta_<export>@<import>`), TF/LR kept optional (decisions D6/D7 in `04_...md`).
- Coefficients live in `betadata/{gene}_betadata.parquet`: `beta_<TF>`, `beta_<lig>$<rec>`
  (L–R), `beta_<lig>#<TF>` (L–TF), and (D11, 2026-08-11) **`beta_metab@<name>` — one summed
  column per metabolite** (was per-pair `beta_<export>@<import>`; see `04` D11).
