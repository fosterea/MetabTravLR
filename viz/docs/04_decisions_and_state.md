# 04 — Decisions & state (living)

**Update this file every session, before finishing.** It is the first thing to read after a
restart. Dates are absolute (project "today" was 2026-07-16 at kickoff).

## Current state
- **2026-07-22 — Adjustable FDR significance cutoff (slider) on both graph views. Playwright-verified.**
  - A discrete slider in the control bar sets the FDR_np cutoff that calls an interface
    "significant". Stops `[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2]` (log-ish; FDR_np is itself
    discrete with a ~0.003 floor), default **0.05**. Shown only in the Graph view, for **both**
    metabolite and gene-pair kinds.
  - No ingest/schema change: FDR_np was already on every edge. Significance is now derived at
    render time via `isSelected(scores, threshold)` (harreman's own `FDR_np < thr AND C_np > 0`),
    replacing reads of the baked-in `scores.selected`. At 0.05 the two are **identical** — verified
    across all 20,658 ingested edges (0 mismatches) — so the default is a true no-op. See A19.
  - Threading: `fdrThreshold` in the store (a global preference, NOT reset on dataset/tier/kind).
    Used by the graph filter+styling, EdgeDetails (resolve + label), EntityDetails counts, the
    panel ranking *hints* (metabolite SORT stays on the summary field, so order/auto-select are
    unchanged), and the Legend ("significant interface (FDR < X)").
  - Because harreman's per-tier gene-pair tables are significant-only (FDR < 0.05), the slider can
    only **tighten** gene-pair edges below 0.05; stops above 0.05 reveal nothing new (correct — no
    such rows exist). Metabolite tables carry the full FDR range, so it works both ways there.
  - `tsc` + `lint` + `build` clean; 0 console errors. Numerically verified: graph edge counts
    match an independent bundle computation at every stop (metabolite 0/3/4/4/5/5/5; gene-pair
    caps at 5); the non-sig toggle composes with the cutoff (0.2 + show-non-sig → 5 sig/5 non-sig,
    tighten to 0.01 → 4 sig/6 non-sig, all 10 still drawn); slider absent in the Environment view.
- **2026-07-19 — SpaceTravLR gene-pair coefficients in the Environment view. Playwright-verified.**
  - Ingest (schemaVersion **3**) now also reads `easy_download/metabtravlr_outputs/<Tier>/
    gene_pairs.csv` — a sibling of `harreman_outputs/` — into `<id>/beta/<Tier>.json`, with
    `hasBeta` on `Dataset`/`DatasetRef`. Only **Primary Dermal Melanoma** has a SpaceTravLR run
    today; the other datasets emit no `beta/` and simply don't show the section (+340 KB).
  - The "Neighborhoods" view is renamed **Environment** and now holds two sections: the existing
    harreman neighborhood bars, then a new `BetaPanel` heatmap of the SpaceTravLR coefficients.
    Both metabolites and gene pairs are supported; a cell-type picker shows all breakdowns
    stacked or narrows to one. See decisions A14–A17.
  - `tsc` + `lint` + `build` clean; **0 console errors/warnings** across the full pass.
  - Verified numerically, not just visually: all 1,555 Tier3 rows round-trip CSV→bundle, and a
    DOM-vs-bundle diff of 125 rendered cells found 0 mismatches in value, sign, hue or ≈0 floor.
- **2026-07-18 — Multi-dataset + neighborhood view + cross-navigation. Playwright-verified.**
  - Data source moved to `Results/<project>/<dataset>/easy_download/harreman_outputs`. Ingest
    now emits **3 datasets** (Human Lung, Human Prostate Adenocarcinoma, Primary Dermal
    Melanoma); the 2 unfinished runs (Human Breast, Human Cervical Cancer) degrade to
    `available: false` manifest entries and appear **disabled** in the picker. `public/data` is
    4.5 MB and fully committed (the Pages deploy serves it).
  - New **Neighborhoods view** (`NeighborhoodView`) beside the graph — per-cell-type bars for
    the selected entity, 3 metrics, "thin" flags on small-n rows. See decisions A9/A10.
  - Entity selection is **remembered per kind**, gene pairs get a **default selection**, and
    metabolites ⇄ gene pairs are **click-through linked** (A11).
  - The gene-pair list stays in the details panel in "On graph" mode, linked to the fan-out by
    a shared highlight (A12).
  - `tsc` + `lint` + `build` clean; **0 console errors/warnings** across the full smoke pass.
  - **Two latent bugs found and fixed while testing** (both only reachable once the view switch
    and dataset picker existed): the "already laid out" ref outlived its Cytoscape instance
    (blank-graph crash on remount), and it was keyed on tier id alone — every dataset names its
    finest tier `Tier3`, so switching dataset kept the previous dataset's nodes and threw
    `nonexistent source`. Now keyed on `(datasetId, tierId)` and cleared with the instance.
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
| A7 | Commit the ingested `public/data` (all datasets) | Fresh clone runs without source CSVs, and the Pages deploy serves this folder verbatim. ~4.5 MB today; revisit if an ingest ever gets to tens of MB. (Superseded the earlier harreman-only allowlist, 2026-07-18.) |
| A8 | Tier parentage inferred (coarse→fine); cell-type crosswalk left `null` | No annotation map available yet; UI tolerates absence. |
| A9 | Neighborhood scores get their **own view**, never an edge encoding | They are not an interface statistic (cells bucketed by their own label, no direction). Drawing them on the cell-type graph would assert an interface that the numbers do not support. Parent doc 05 §5a. |
| A10 | nbhd chart = horizontal bars, one hue; enrichment gets a diverging pair + neutral zero | One series ⇒ no legend needed; enrichment is the only signed metric. Palette validated with the dataviz validator against `--bg-canvas` in both themes. New `--nbhd-*` tokens rather than reusing the RESERVED `--val-*`. |
| A11 | Entity selection is remembered **per kind**; cross-links jump between kinds | Foster: switching to gene pairs had no default and lost your place. Memory makes the metabolite⇄pair links safe to follow — the way back is one click. |
| A12 | The gene-pair list stays in the panel in "On graph" mode (was either/or) | The two are complements, not alternatives: the fan shows *where*, the list shows *which + how much* and is where the links live. A shared highlight (`pinnedGp` + `hoverGp` → `selectFocusedGp`) ties them together; hover previews without destroying a pinned choice. |
| A13 | Incomplete datasets are listed **disabled**, not hidden | Silently dropping a dataset Foster knows he ran looks like data loss; the reason travels in the manifest. |
| A14 | SpaceTravLR betas get their **own bundle** (`beta/<Tier>.json`), NOT `EntityEdge.value`/`sign` | They are not an interface statistic: a row is (target gene, directed transporter pair, cell type), with no CT↔CT pair anywhere in it. Same reasoning as A9 for nbhd. `value`/`sign` stay reserved for genuinely directed signed *edges*. |
| A15 | Beta magnitude = **log ramp over a fixed 4 decades below the view max**; sign is a separate channel | Foster's catch: naive `sign(x)·log10|x|` turns −1e−7 into +7 — it flips AND inflates. Splitting the channels makes that class of bug impossible: `norm()` is always ≥0 and independent of sign. |
| A16 | Anything below the floor renders **untinted + labelled `≈0`**, with no minimum bar | Foster: "some things can be almost zero and that should be communicable." A min-anchored scale gives the view's smallest value a visible pedestal, so negligible reads as "weak but real". Absence of color is the honest encoding for absence of effect. |
| A17 | Both directions of a pair are **separate rows**, never merged | `env→cell` and `cell→env` are independent coefficients (70 of 107 pairs have both). Merging would invent a symmetry the model doesn't claim. This is the one place in the app where direction is real — hence the explicit "environment → cell" column header. |
| A18 | The view is renamed **Environment** and enabled when `hasNbhd \|\| hasBeta` | It now holds two different per-cell measurements, both about a cell's surroundings rather than an interface. "Neighborhoods" named only the first one. |
| A19 | Significance is **derived from FDR_np at a UI-set threshold** (`isSelected(scores, thr)`), not read from the baked-in `scores.selected` | Lets a slider control the cutoff with no ingest change. Reproduces `selected` exactly at the 0.05 default (0/20,658 mismatches), so it's a safe drop-in. `scores.selected` stays in the contract as harreman's own 0.05 call, but the app no longer reads it for graph significance. Slider stops are discrete because FDR_np is discrete with a low floor; ranking SORT deliberately does NOT depend on the threshold, so moving it never re-sorts the panel or changes auto-selection. |

## Open questions / needs Foster
- **Dataset naming**: ids/names now come straight from the `Results/` folder names
  ("Human_Lung" → "Human Lung"), with the project folder as a group label. Resolved — but say
  if you want prettier display names.
- **Human Breast / Human Cervical Cancer** only have `harreman_network.json`; they show as
  disabled. Re-run ingest after those finish and they light up on their own.
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
- [x] Multiple datasets + `<root>/<project>/<datasetName>/easy_download` layout, with a dataset
      switcher and graceful handling of incomplete runs. (2026-07-18.)
- [x] Neighborhood scores surfaced as their own view. (2026-07-18.)
- [ ] Compare datasets side by side (e.g. one metabolite's neighborhood profile across Lung /
      Prostate / Melanoma). The data is all client-side now, so this is a UI-only change.
- [x] SpaceTravLR gene-pair coefficients surfaced per cell type in the Environment view.
      (2026-07-19; `--val-pos`/`--val-neg` now in use for the heatmap.)
- [ ] SpaceTravLR signed two-value **edges**: width=|value|, color=sign. Still open and still
      unclaimed by A14 — the beta table has no cell-type *pair* in it, so nothing in
      `metabtravlr_outputs/` can populate `EntityEdge.value`/`sign` yet. Needs a source table
      keyed by (CT1, CT2) before this is buildable.
- [ ] Beta panel follow-ups Foster may want: the 4-decade window (`BETA_DECADES`) currently sends
      53% of Tier3 coefficients to ≈0 — widen if that hides signal he cares about; and a
      "sort rows by target gene" control if 24-direction metabolites get unwieldy.
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
6. Switch dataset — nodes/edges must become the NEW dataset's cell types (tier ids repeat
   across datasets, so this is the case that used to break); incomplete datasets are disabled.
7. Switch View → Neighborhoods and back — the graph must rebuild, not come back blank.
8. Switch Metabolite ⇄ Gene pair — each kind keeps its own selection; gene pairs have a default.
9. "On graph" mode: the pair list stays in the details panel; clicking a sub-edge pins its row;
   hovering a row highlights that pair on the graph and releases back to the pinned one.
10. Screenshot captured as evidence.
11. Environment view on **Primary Dermal Melanoma**: nbhd bars AND a "SpaceTravLR coefficients"
    heatmap. Cell-type picker narrows to one block and rescales the ramp. A bidirectional pair
    (e.g. SLC16A4 – SLCO2B1) shows **two** rows. A pair with no betas shows the "no coefficients"
    message, not an empty grid. On Human Lung the beta section is absent but the bars still render.
12b. FDR slider (Graph view): default reads "FDR < 0.05" and matches the old `selected`-based
    counts. Dragging left tightens (fewer/no significant edges); dragging right loosens for
    metabolites but never exceeds the gene-pair bundle's significant-only set. Legend + EntityDetails
    counts track it. Absent in the Environment view.
12. Beta correctness (worth re-running after any scale change — this is where the bugs are):
    in the console, diff the DOM against `beta/<Tier>.json` and assert, per cell, that the label
    matches source, a negative never renders `+`, hue matches sign, and `≈0` cells carry no tint.

## Changelog
- 2026-07-22: **Adjustable FDR significance cutoff (slider).** Added `isSelected(scores, thr)` to
  `src/data/scales.ts` (harreman's `FDR_np < thr AND C_np > 0`, `DEFAULT_FDR = 0.05`) and a
  `fdrThreshold` store field + `setFdrThreshold`. A discrete `<input type=range>` over log-ish
  stops (`FDR_STOPS`, default 0.05) in `ControlBar`, shown in the Graph view for both kinds.
  Replaced every graph-significance read of `scores.selected` (GraphView filter + edge class,
  EdgeDetails resolve + label, EntityDetails counts, ranking hints) with `isSelected(…, threshold)`;
  Legend now prints the live cutoff. Metabolite ranking sort stays on the summary field (stable
  order/auto-select); gene-pair sort follows the cutoff but is bounded by the significant-only
  bundle. No ingest/schema change. Playwright-verified numerically (see Current state); tsc+lint+
  build clean, 0 console errors. See A19.
- 2026-07-19: **SpaceTravLR gene-pair coefficients in the Environment view.**
  (1) **Ingest** (schemaVersion 3): `buildBeta()` reads the `metabtravlr_outputs/` sibling of
  `harreman_outputs/` and emits `<id>/beta/<Tier>.json` = `{ tier, cellTypes, targetGenes,
  byPair }`, plus `hasBeta` on `Dataset`/`DatasetRef`. Keyed on an order-independent **sorted**
  pair key (`betaKey`), because the network lists some pairs in both orders and a *directed* beta
  row can't be attributed to one `gpId`; the direction survives inside each row as `env`/`cell`.
  Rows for pairs absent from the network `gp` list are dropped with a warning, never invented.
  Melanoma only (+340 KB); other datasets emit nothing and degrade silently.
  (2) **`BetaPanel`**: a per-cell-type heatmap — rows = directed pairs (`environment → cell`),
  columns = the 4 target genes (CD3E/CD4/ENTPD1/IL2RA), one block per cell type with a picker for
  All vs one. Color = sign (`--val-*`, the reserved diverging pair, finally used), tint depth =
  log magnitude on **one scale shared across every block** so cell types are comparable, and the
  number is printed in every cell so color never carries the value alone. Small-n cell types get
  the same word-tag `thin` treatment as the nbhd bars.
  (3) **The scale is the interesting part** (`src/data/betaScale.ts`, A15/A16). Betas span ~6
  decades and are signed; naive `sign(x)·log10|x|` maps −1e−7 to +7, flipping and inflating.
  Magnitude and sign therefore ride separate channels, and the floor is anchored a fixed 4
  decades below the view max rather than at the view min — so a near-zero coefficient renders
  untinted and labelled `≈0` instead of getting a visible pedestal that would read as a small
  real effect.
  (4) **View renamed Neighborhoods → Environment**, enabled on `hasNbhd || hasBeta`, so a dataset
  with only one of the two still works.
  (5) **Verified**: 1,555/1,555 Tier3 rows round-trip CSV→bundle; a DOM-vs-bundle diff of 125
  rendered cells found 0 mismatches in value, sign, hue or floor; both directions of a
  bidirectional pair kept; tier/dataset/kind switching, empty states, and a no-beta dataset all
  behave; contrast of the strongest tint vs its text ≥6.8:1 in both themes; no clipping or
  horizontal overflow down to a 1100px viewport; 0 console errors.
- 2026-07-18: **Multi-dataset source, neighborhood-scores view, selection memory + cross-links.**
  (1) **Ingest** (`scripts/ingest.mjs`, schemaVersion 2): walks `Results/<project>/<dataset>/`,
  tags each dataset with its project, prettifies ids ("Human_Lung" → "Human Lung"), and never
  hard-fails — an unfinished run (network JSON, no tier tables) or any per-dataset exception
  becomes an `available: false` manifest entry with a reason. Also fixed discovery preferring a
  stray `easy_download/harreman_network.json` over the real `harreman_outputs/` child, which had
  made Human Lung look incomplete. Emits `nbhd/<Tier>.json` from the
  `[nbhd_scores][summary_{m,gp}]` tables, resolving the tables' ambiguous single-underscore gene
  -pair keys through the network's own `gp` list.
  (2) **Neighborhoods view**: a second canvas view (control-bar "View" switch) showing, per cell
  type, how much that type's own cells sit in high-scoring neighborhoods for the selected entity
  — bars over 3 metrics (significant share / mean score / log₂ enrichment), all stats in-row,
  small-n rows flagged "thin" (<25 significant cells, per the parent docs' instability warning),
  and a standing caveat that this is NOT an interface statistic. Deliberately not an edge
  encoding (A9). Works for metabolites and gene pairs; disabled for datasets without the scores.
  (3) **Selection memory + defaults**: gene-pair view now auto-selects its top-ranked pair (it
  waits for the tier bundle, since gp ranking is bundle-derived), and each kind remembers its
  last entity across switches.
  (4) **Panel + graph linked**: the transporter-pair list stays in the details panel in "On
  graph" mode; clicking a fanned sub-edge pins its row, hovering either side previews (the pair
  lights up, siblings recede), and clicking a row opens that pair's own view. Metabolite chips in
  the entity panel link the other way. Playwright-verified end to end: 57 sub-edges fanned,
  pin/hover/fallback correct, click-through + return-to-Iron, all 3 datasets and all 3 tiers,
  0 console errors.
  (5) **Fixed two latent graph bugs** the new navigation exposed — see Current state.
- 2026-07-17: **Reverted gene-pair color-coding + background de-emphasis; fixed fan-out
  overlap with curvature instead (Foster feedback).** (1) The color-coding didn't help — an
  interface fan's sub-edges overlapped into one line, and with >8 pairs two pairs shared a
  color anyway (review Medium). **Removed** all gene-pair coloring (theme `--gp-*` tokens,
  `edge.gp-slot-*`, tab/panel swatches, `slot`/`gpSlotMap`). Root fix: the fan-out sub-edges
  now spread apart geometrically — non-self edges bow via a per-edge `control-point-distances`
  (`unbundled-bezier`, spacing capped at 200px so dense fans stay on-screen); self-loops fan
  around the node as separate "petals" via per-edge `loop-direction`. Playwright-verified:
  L-Glutamine's 11-pair self-interface renders as an 11-petal flower, clearly separable with no
  color. Color can be re-added later only if still needed. (2) **Removed the T-cell background
  greying** — keep the app agnostic to the activity; nothing T-cell-specific without asking.
  (3) **Fixed 2-node tiers** (Tier1/Tier2) to lay out side by side (grid, 1 row) instead of the
  circle layout stacking them vertically. (4) Applied the review Lows: metabolite ranking stays
  on the summary sig-pair count so App's auto-select agrees with the panel's top row; the
  gene-pair count lookup is now order-tolerant. tsc+lint clean, 0 console errors.
- 2026-07-17: **Usability pass (drove the app for "which metabolites influence T cells?").**
  Top finding + fix: the background `other↔other` self-interaction dwarfs the T-cell interfaces
  (Tier1 Iron: bg C_np 194,927 vs T↔other 14,116, T↔T 2,590) and visually dominated. Now
  background-only significant interfaces (neither endpoint a focal/T-cell node, via
  `classifyCellType` — not a hard-coded string) render recessive (`edge.sig.bg` opacity 0.32)
  so T-cell interfaces read first; fan-out/gene-pair views are exempt (there you want all pairs
  vivid). Playwright-verified Tier1 (bg self-loop faded, T-cell edges bold) and Tier3 (only the
  lone other↔other dims). **Other findings logged, not yet actioned** (see Open questions):
  2-node tiers waste horizontal space (circle layout stacks them vertically); expand-mode/
  expand-all persist across metabolite+tier changes; top-left gp-tabs vs top-right entity panel
  can crowd on a narrow window.
- 2026-07-17: **Gene-pair color-coding (differentiate the fan-out).** Each transporter pair now
  gets a stable categorical color (theme.css `--gp-1..8`, validated colorblind-safe dataviz
  palette; adjacent CVD in the 6–8 band → legal because the **labelled tabs + hover** are the
  required secondary encoding, plus all 8 clear 3:1 contrast on both themes). Shared ordering
  (`metaboliteSigPairsAtTier` → slot) means a pair's color is identical across the on-graph
  fan-out sub-edges, its tab swatch, the isolated-pair view, and the panel breakdown swatches.
  Playwright-verified: 7 pairs → 7 distinct edge colors matching their tab swatches; 0 console
  errors. Also folded in the Unit-6 review L2 (gpTabInfo guards on gpBundle) and made gpTab
  lookups order-tolerant (`gpEdgesInTier`).
- 2026-07-17: **Sig-interaction counts + full gene-pair support.** Side panel now shows each
  entity's # significant cell-type interfaces at the current tier (from the loaded bundle):
  metabolites read "N sig interfaces · T-cell"; the **gene-pair view now lists all 416 pairs**
  (was significant-only), **sorted by sig-interaction count desc**, each "N sig interfaces · M
  metabolites", with the 280 pairs that have no significant interaction at the tier greyed below
  a "No significant interactions at this tier (full support)" divider — same treatment as the
  eliminated metabolites. Counts/sorting/greying update per tier. Playwright-verified: 416
  listed, 280 greyed at Tier3, top sorted CUBN–CUBN (11) → CD38–CD38 (7) → …
- 2026-07-17: **GitHub Pages deploy wired up.** Added `.github/workflows/deploy-viz.yml` (repo
  root — the sanctioned out-of-`viz/` file, per Foster's explicit deploy request): builds
  `viz/` and deploys `viz/dist` to Pages on push to `release` touching `viz/**` + manual
  dispatch. Verified `npm run build` locally (dist 615KB JS / 198KB gz, relative `./assets`,
  `public/data` bundled). **Foster's one-time step:** Settings → Pages → Source: GitHub
  Actions. Site → `https://fosterea.github.io/MetabTravLR/`. Docs in `06_hosting.md`.
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
