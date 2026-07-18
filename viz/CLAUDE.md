# CLAUDE.md — viz/ (web-dev mode)

This folder is a **self-contained single-page web app** for exploring cell–cell
metabolite-crosstalk graphs produced by the parent MetabTravLR / harreman work. It is a
**side project** with its own docs and its own rules, deliberately separate from the data
pipeline in `metab_processing/` and the research docs in `DataForClaude/documentation/`.

## Golden rules (web-dev mode)
1. **Stay in `viz/`.** You and any sub-agents may only create/edit files under `viz/`. The
   one sanctioned exception outside it is the pointer block in the repo-root `CLAUDE.md`.
   Never touch `metab_processing/`, `src/SpaceTravLR/`, `docs/` (Sphinx), notebooks, or the
   parent project's files unless Foster explicitly asks.
2. **Read the source data, never mutate it.** The app consumes `easy_download` outputs
   (produced by `metab_processing/`). The ingest adapter reads them; nothing here writes back.
3. **Minimal, justified dependencies.** The stack is fixed (see `docs/01_architecture.md`).
   Do not add a library without recording why in `docs/04_decisions_and_state.md`.
4. **Testing = the `playwright` MCP server only.** No other test tooling gets installed.
   Drive the running dev server (`npm run dev`, port 5173) with the playwright MCP tools.
5. **Document as you go.** This app may be picked up after a restart. Update
   `docs/04_decisions_and_state.md` whenever a decision or state changes, before finishing.
6. **Commit your own work only.** Commit `viz/**` (and the root-CLAUDE pointer). Never stage
   Foster's unrelated working changes elsewhere in the repo.

## Where things are
- **Start every session:** read `docs/README.md`, then `docs/04_decisions_and_state.md` for
  current state, then whatever file the task touches.
- App source: `src/`  ·  ingest adapter: `scripts/ingest.mjs`  ·  generated data: `public/data/`
- Docs index: `docs/README.md`. Style/conventions: `docs/02_style_and_conventions.md`.
- Dev/review sub-agent loop + how to test: `docs/03_dev_process.md`.
- Data contract (the app↔data interface): `docs/05_data_contract.md`.

## Quick start
```bash
cd viz
npm install
npm run ingest -- ../Results        # every dataset under Results/<project>/<dataset>/
npm run dev            # http://localhost:5173
```

## What this app is (one paragraph)
Pick a **dataset**, a **tier** (cell-type granularity: Tier1 `T Cell` → Tier2 `CD8 T Cell` →
Tier3 CD8 subtypes), and an **entity** (a metabolite today; gene pairs and, later, SpaceTravLR
gene/gene-sets). The main canvas draws a **graph of cell types** whose **edges are the
undirected interfaces** where that entity is exchanged, weighted by communication strength and
significance. A side panel lists/ranks entities (e.g. by T-cell involvement) so a user can ask
"which metabolites are influencing T cells?" — while the code stays use-case-agnostic.
The **`CT1→CT2` order is NOT a direction** (sorted-label artifact) — edges are undirected.
