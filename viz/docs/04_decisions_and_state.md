# 04 — Decisions & state (living)

**Update this file every session, before finishing.** It is the first thing to read after a
restart. Dates are absolute (project "today" was 2026-07-16 at kickoff).

## Current state
- **2026-07-16 — MVP built & verified end-to-end (playwright). Baseline committed.**
  - `viz/` initialized: Vite + React + TS + Cytoscape + Zustand. Deps installed.
  - Full docs set written (`docs/00`–`05`) + `viz/CLAUDE.md` + root-CLAUDE pointer.
  - Ingest adapter (`scripts/ingest.mjs`) works; `public/data/harreman/` generated & validated
    (160 metabolites, 416 gene pairs, Tier1–3 edges, gp↔metabolite many-to-many verified).
  - App: control bar (dataset/tier/entity-kind/non-sig toggle), ranked entity side panel,
    Cytoscape graph (undirected edges, log-width, sig/non-sig styling, self-loops), legend,
    entity-details overlay. `tsc`+`lint` clean, console clean.
  - **Playwright-verified**: loads Tier3 (5 CD8 subtypes), metabolites ranked by T-cell
    involvement (Iron→9 sig pairs, auto-selected), details show "9 sig / 15 interfaces";
    gene-pair toggle → 416 pairs ranked by metabolite count; selecting SLC1A5–SLC38A7 →
    L-Glutamine, 2 sig / 2 interfaces. Canvas renders (3 layers, full size). 0 console errors.
  - **Known gaps / next**: see roadmap. Screenshots via the playwright MCP return to the client
    but aren't written to a readable path here — verification used the a11y tree + canvas probes.
    Theme is dark/light via CSS vars but the Cytoscape stylesheet is built once per render (no
    live re-theme on OS switch) — acceptable for MVP.

## Decisions log
| # | Decision | Rationale |
|---|---|---|
| A1 | App lives in `viz/` inside the repo | Foster's call; committed here, isolated in web-dev mode. |
| A2 | Stack = Vite + React + TS, Cytoscape.js, Zustand, Papa Parse (ingest only) | See `01_architecture.md`. Cytoscape's style mappers fit the future width+color two-value edges. |
| A3 | **Build-time ingest adapter**, not live client CSV parsing | Decouples app from messy CSV; deployable; new source = new adapter, same contract. |
| A4 | Data contract is entity-agnostic (`metabolite`/`gene_pair`, extensible) | Gene pairs wanted now; SpaceTravLR genes/gene-sets later — same edge model. |
| A5 | Edges are **undirected**; diagonal = self-loop | harreman `CT1→CT2` is a sorted-label artifact, not flow (parent doc 05 §3). |
| A6 | Edge magnitude on **log/normalized-per-view** scale | `C_np` spans orders of magnitude; raw linear is unreadable. |
| A7 | Committed the `harreman` ingested dataset; other datasets stay local | Fresh clone runs without source CSVs; large data shouldn't bloat git. |
| A8 | Tier parentage inferred (coarse→fine); cell-type crosswalk left `null` | No annotation map available yet; UI tolerates absence. |

## Open questions / needs Foster
- **Dataset naming/id** for the current data: used `id=harreman`, name "Harreman — metabolite
  crosstalk (Xenium)". Confirm or rename.
- **Cell-type parent crosswalk** (Tier3 subtype → Tier2 `CD8 T Cell` → Tier1 `T Cell`): not in
  the outputs. If Foster can emit it from `metab_processing/`, we can draw true parent grouping.
- Primary magnitude column: using `C_np` (non-parametric, matches harreman's significance).
  Confirm vs `Z`/`C_p` for the width encoding.

## Roadmap / TODO (designed-for)
- [ ] Gene-pair view toggle (data already ingested) + show served metabolites (many-to-many).
- [ ] Parent cell-type marking via tier hierarchy (needs crosswalk above).
- [ ] Multiple datasets + `<root>/<datasetName>/easy_download` deploy layout (ingest already
      auto-detects this; app dataset switcher needed).
- [ ] SpaceTravLR signed two-value edges: width=|value|, color=sign (`--val-pos`/`--val-neg`).
- [ ] Metabolite ranking controls in the side panel (by T-cell involvement / significance).
- [ ] Static deploy of `dist/` when Foster wants it.

## Smoke checklist (re-verify after changes, via playwright MCP)
1. `npm run dev`; open http://localhost:5173 — no console errors.
2. Dataset/tier/metabolite selectors populate from `public/data/manifest.json`.
3. Selecting a tier renders its cell-type nodes; selecting a metabolite draws its edges.
4. Significant edges are visually distinct; diagonal shown as self-loop; legend present.
5. Empty state shows a message when a metabolite has no edges at a tier.
6. Screenshot captured as evidence.

## Changelog
- 2026-07-16: Scaffold, docs, ingest adapter, data validated. (this file created)
