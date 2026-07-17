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
- [x] Gene-pair view toggle (data already ingested) + show served metabolites (many-to-many).
      (MVP; details overlay lists served metabolites.)
- [x] Metabolite → gene-pair **expansion** (panel list + on-graph fan-out; per-edge + all).
      (Unit 3, 2026-07-17.)
- [ ] Parent cell-type marking via tier hierarchy (needs crosswalk above).
- [ ] Multiple datasets + `<root>/<datasetName>/easy_download` deploy layout (ingest already
      auto-detects this; app dataset switcher needed).
- [ ] SpaceTravLR signed two-value edges: width=|value|, color=sign (`--val-pos`/`--val-neg`).
      NOTE: these are **directed scalers** on edges — the fan-out gp sub-edges and the reserved
      `value`/`sign` edge fields are the intended carrier; likely a per-edge multiplier applied
      on top of the existing width scale + diverging color.
- [ ] Metabolite ranking controls in the side panel (by T-cell involvement / significance).
- [ ] Bottom overlays (legend / EdgeDetails) can overlap on very narrow canvases — reposition
      or hide on small screens (Unit 2 review Low; desktop-first so deferred).
- [ ] Static deploy of `dist/` when Foster wants it (see `06_hosting.md`).

## Smoke checklist (re-verify after changes, via playwright MCP)
1. `npm run dev`; open http://localhost:5173 — no console errors.
2. Dataset/tier/metabolite selectors populate from `public/data/manifest.json`.
3. Selecting a tier renders its cell-type nodes; selecting a metabolite draws its edges.
4. Significant edges are visually distinct; diagonal shown as self-loop; legend present.
5. Empty state shows a message when a metabolite has no edges at a tier.
6. Screenshot captured as evidence.

## Changelog
- 2026-07-17: **Unit 6 (gene-pair tabs).** A tab strip over the canvas (`GenePairTabs`) lets you
  view the selected metabolite's transporter pairs individually: "All (metabolite)" plus one
  tab per gene pair significant at the current tier (with its interface count, sorted by
  strength). Picking a pair tab (`store.gpTab`) isolates just that pair's interfaces on the
  graph — a per-pair drill-down complementing the on-graph fan-out and the panel list. While a
  tab is active the metabolite-only controls (expand mode / expand-all / non-sig) hide, the
  EdgeDetails panel resolves against the pair's edges and labels it "gene pair X interface",
  and hover shows the pair. Playwright-verified: 7 pair tabs for Iron@Tier3, CUBN–CUBN isolates
  11 interfaces, controls hidden, panel + hover correct; 0 console errors.
- 2026-07-17: **Unit 5 (full metabolite support).** All 160 network metabolites already show
  in the side panel (none were cut); the 17 that are insignificant everywhere (no global sig +
  no sig pair at any tier) are now marked `eliminated`, greyed, hinted "eliminated — not
  significant", and sorted below a divider ("Eliminated — not significant (full network
  support)"), so the complete network support is visible while the significant ones stay on
  top. `metaboliteInvolvedAnywhere` is tier-independent (a metabolite significant at another
  tier is not eliminated). Playwright-verified: 17 greyed items under the divider.
- 2026-07-17: **Unit 4 (edge legibility).** Fixed edge clumping (Foster feedback, incl. the
  non-significant-included case). Nodes 40→56px and max edge width 22→15px so strong edges
  anchor cleanly instead of blobbing into small nodes; self-loops arc up-and-out
  (`loop-direction -90deg`, `loop-sweep 80deg`) instead of sitting as a blob on the node;
  parallel edges (gene-pair fan-outs) spread via `control-point-step-size: 55`; non-significant
  edges recede to a thin (1.5px) faint (opacity 0.3) dashed line so they stop competing with
  significant interfaces. Playwright-verified both the non-sig metabolite view and the
  expand-all fan-out; 0 console errors. Also: `.playwright-mcp/` and `*.tsbuildinfo` gitignored
  (the two tsbuildinfo files untracked); root `.gitignore` touched by explicit request.
- 2026-07-17: **Unit 3 review follow-up.** Addressed the review's Medium + 2 Lows: metabolite
  edges now scale over the full view set (not just the unfanned fallback remainder), so a lone
  fallback interface no longer jumps to max width; `genePairsAtInterface` dedups on a canonical
  (sorted) key (hardens against a metabolite ever listing both gene orders); the panel notes
  the fan is a "significant subset — pair strengths need not sum to the metabolite total"; and
  the legend shows a "gene-pair sub-edge (own scale)" key in graph-expand mode. tsc+lint clean,
  playwright-verified, 0 console errors. (Unit 3 review: no Critical/High; data join verified
  correct against raw data — 96 primary-key hits, 0 fallback, no double-count.)
- 2026-07-17: **Unit 3 (gp-aware metabolite expansion).** A metabolite edge can be broken
  into its contributing transporter **gene pairs**, two ways, toggled in the control bar
  ("Gene pairs: In panel / On graph"):
  - **In panel**: clicking a metabolite edge lists the gene pairs significant at that
    interface (name + C_np, sorted) in EdgeDetails.
  - **On graph**: the picked interface (or, with "Expand all interfaces", every interface)
    fans out into parallel gene-pair sub-edges between the same cell-type nodes (width =
    per-pair strength, own width scale, translucent `.gp` style); hover a sub-edge for its
    gene pair + strength; clicking selects the whole fan.
  Data is client-side only (`genePairsAtInterface` joins a metabolite's `genePairs` against
  the same-tier gene_pair bundle, now loaded alongside in metabolite mode) — **no ingest or
  source-pipeline change** (A3/A4 hold). GraphView refactored so layout/fit run ONLY on tier
  change (a primitive `expansionKey` drives edge rebuilds), so expanding/picking never
  relayouts. Also **fixed the Unit 2 review Medium**: EdgeDetails now resolves the pick
  against the same *visible* (tier+significance-filtered) edge set as the graph, so toggling
  non-significant off no longer orphans the panel; tooltip a11y contradiction removed.
  Playwright-verified: panel breakdown (Iron@Effector↔other → CUBN–CUBN 6128); graph
  expand-all (9 metabolite edges → 17 gp sub-edges); per-edge fan (only clicked interface);
  gp hover shows "CUBN – CUBN … Cnp 309"; non-sig-off clears orphan panel; Tier1 relayout
  clean (no node stacking); gp controls hidden in gene-pair mode; 0 console errors.
- 2026-07-17: **Unit 2 (edge interaction).** Click an edge → new bottom-right **EdgeDetails**
  panel (interface cell types, C_np strength, significance/FDR, parametric C_p/Z, undirected
  note; × or empty-canvas click clears). Hover an edge → floating tooltip with the numeric
  C_np strength (mutates a ref'd DOM node, not React state, so mousemove doesn't thrash).
  Store gains `selectedEdge` + `selectEdge`, cleared on any dataset/tier/kind/entity change.
  Cytoscape `autounselectify` on; highlight is store-driven via a `.picked` overlay halo
  (dropped the dead `:selected` rule). Added `src/data/format.ts` (strength/FDR formatting)
  and `sameInterface` (order-agnostic edge match). Legend updated to "relative strength in
  view (log)" + hover/click hint (addresses Unit 1 review: per-view scale isn't absolute —
  raw value now reachable via hover/panel). Playwright-verified: click→panel with correct
  scores, exactly-one `.picked`, background-click clears, real canvas hover shows the tip;
  0 console errors. Folds in Unit 1 review LOWs (legend text, annotated dangling ink tokens).
- 2026-07-17: **Unit 1 (graph legibility).** (1) Cell-type labels moved BELOW the node
  (`text-valign: bottom` + canvas-colored halo) so long names ("Proliferating CD8 T cell")
  no longer overflow the circle. (2) Edge-width scale now normalizes between the view's
  log-min and log-max (was 0..max) and widened px range 2–22 (was 1.5–14), so strengths are
  visibly distinguishable instead of hugging the floor; a lone/all-equal edge maps to max.
  Added a **dev-only `window.__cy` test handle** (stripped in prod) so the playwright MCP can
  assert node/label/edge geometry against the canvas. Playwright-verified: 5 labels render
  below their nodes; edge widths span 2→22 with good spread; 0 console errors.
- 2026-07-17: Wrote `docs/06_hosting.md` — static-hosting guide (Firebase Hosting steps +
  GitHub Pages / Cloudflare Pages / Netlify alternatives, access-control/privacy note,
  "when data grows" future note, cost summary). No code or deploy; docs only.
- 2026-07-16: Scaffold, docs, ingest adapter, data validated. (this file created)
- 2026-07-16: MVP built + committed (638bf1f). Ran the dev/review loop once: review
  sub-agent audited the baseline; dev sub-agent fixed all findings — H1 (tier-switch stale
  edges crashed Cytoscape → clear stale bundle + filter edges to in-tier endpoints), M2
  (hide non-sig toggle in gene-pair view; gp tables are significant-only), M3 (loading
  overlay), L4 (color tokens: added `--on-accent`), L5 (dead code in ingest), L6/L7 (store
  async race + rejection hardening). Re-verified with playwright: 0 console errors on the
  Tier3→Tier1 switch; non-sig toggle absent in gp mode. Review confirmed undirected/selected
  semantics, entity-id integrity, and width-scale robustness are correct — do not change those.
