# 04 — Decisions & state (living)

**Update this file every session, before finishing.** It is the first thing to read after a
restart. Dates are absolute (project "today" was 2026-07-16 at kickoff).

## Current state
- **2026-08-01 — Cell-type-major matrix layout (Cycle 3 / REVISION 1). Playwright-verified.**
  - Foster's feedback: the channel-major `BetaSection` stacking repeated each cell type per factor and
    the columns didn't line up. Both beta views are now **cell-type-major**: each cell type appears
    ONCE, and the factor groups (metabolite pairs / LR / L-TF / TF / all-metabolic) are stacked
    beneath that one cell-type label, sharing ONE ordered set of **union** target-gene columns.
  - New `src/components/BetaMatrix.tsx` (+ `BetaMatrix.module.css`) REPLACES `BetaSection` (deleted).
    Props: `groups: BetaFactorGroup[]` (`{key, channel, label, rows}`), `cellTypes`, optional
    `allowedGenes`. Logic: `genesShown` = sorted UNION of target genes across all groups' rows in the
    shown cell types (∩ `allowedGenes` when given); per-GROUP scale (`makeBetaScale` over that group's
    shown rows — each factor keeps its own magnitude scale); ONE CSS grid for the WHOLE matrix
    (`minmax(200px,max-content) repeat(genesShown, 84px)`) so the feature-identity column and every
    gene column line up across every cell-type block and factor group — cell-type headers and
    factor-group dividers are full-width spanning rows (`grid-column: 1 / -1`) within it, and a gene
    header row repeats per block (still inside the one grid, so still aligned).
    **Missing coefficient = BLANK whitespace** (slot kept, `title="no coefficient"`); `≈0`
    (measured-but-negligible) stays visually distinct (subtle fill). One shared sign-only ramp legend
    at the top; each group states its own numeric range on its divider. See A26.
  - `SpaceTravlrView` now builds groups from the selected channels and renders ONE `<BetaMatrix
    allowedGenes={selectedGenes} cellTypes={shown}/>` (Sections/gene chips/cell-type select kept).
    `BetaPanel` builds groups = primary metab group (rows = the metabolite's pairs) + one per `added`
    id (rows = full `metab.rows` for "All metabolic transporters", else `ch.rows`), `allowedGenes`
    undefined (union columns, so a gene only a comparison factor covers still appears), renders ONE
    `<BetaMatrix cellTypes={shown}/>`. Compare chip bar / Add-comparison menu / cell-type select /
    direction caveat all kept; removal stays via the top chips' ✕ (the matrix has no per-section ✕);
    `added` still resets on entity/tier/dataset, not on cell-type. `availableCellTypes` now = cell
    types present in ANY shown group.
  - No ingest/schema/contract change — pure UI. Dead CSS from `BetaSection` removed from
    `BetaPanel.module.css` (`.section/.sectionHead/.remove/.grid/.cell/.foot/…`).
  - **Verified (Playwright, Melanoma):** Environment view — 5 cell types each shown once; primary
    transporter group + added comparison groups stacked beneath each; union gene columns in the SAME
    position across every group/cell type; adding TF surfaced TF-only columns (CTLA4/FOXP3/IL10) with
    the metab group **blank** there (ENTPD1 still `≈0`, distinct); per-group scales (metab 4.5e-3, TF
    4.0e0); ✕ on a chip removed the factor group and contracted the union columns; cell-type filter
    narrowed to one block; switching metabolite/tier cleared the added groups. Standalone view — same
    cell-type-major layout, all 4 dividers with per-group scales; section toggle off dropped that
    factor + its unique columns (TF off → IL10 gone); gene chip off removed a union column.
    **DOM-vs-bundle:** 2,145 value cells across both factors round-trip to `formatBeta(bundle,
    per-group scale reconstructed from the DOM)` with **0 text / 0 sign / 0 hue / 0 source
    mismatches**; 6,010 blank cells all carry `title="no coefficient"` and no number (blank ≠ ≈0).
    `tsc -b` + `lint` clean; **0 console errors/warnings** on a clean load. Screenshots
    `viz/rev1-environment-celltype-major.png`, `viz/rev1-standalone-celltype-major.png`.
  - **Cycle-3 review Low fix (single grid):** the first cut used one grid PER cell-type block, so the
    `max-content` first column could differ between blocks and the gene columns didn't line up
    vertically across blocks — violating Foster's "same order AND place for each cell type". Now the
    whole matrix is ONE grid (headers/dividers span full width), so the shared `max-content` first
    column and all gene columns align across every block automatically. Removed the dead empty
    `.cornerHead` ruleset. Playwright cross-block check: the same gene-header column has an IDENTICAL
    x across all 5 blocks in BOTH views (e.g. Environment CD3E left=546px ×5, IL2RA=1062px ×5;
    Standalone CD3E=234 ×5 … IL2RA=750 ×5; all `allEqual`), cell-type headers + all dividers 802px
    full-width; blank/≈0/tint unchanged; 0 console errors. Screenshots
    `viz/rev1b-environment-onegrid.png`, `viz/rev1b-standalone-onegrid.png`.
- **2026-08-01 — BetaPanel comparison chips (Cycle 2). Playwright-verified.**
  - The Environment view's SpaceTravLR section now lets you stack the OTHER feature channels beneath
    the metabolite's own transporter pairs, for a like-for-like read. `BetaPanel` gained: a primary
    section title ("This {metabolite|gene pair}'s transporter pairs"); a "Compare" bar (below the
    caveat, above the primary heatmap) with removable chips + an **"Add comparison ▾"** `<select>`
    (options = channels in the bundle not already added; `metab` labelled **"All metabolic
    transporters"** since the primary already shows this metabolite's own pairs, the others use
    `channel.label`); and one `BetaSection` per added channel BELOW the primary — `rows` = the WHOLE
    `channel.rows` (the superset), `genes` = `metab.targetGenes ∩ channel.targetGenes` (columns
    aligned to the metabolite's genes), `cellTypes` = the same `shown` filter, `onRemove` = drop it.
    Local `added: BetaChannelId[]` (order preserved) resets on entity/tier/dataset change (effect
    keyed on `entityId + tierId + datasetId`) but NOT on a cell-type-filter change. See A25.
  - No ingest/schema/contract change — this is pure UI over the v4 bundle + the Cycle-1 `BetaSection`.
  - **Verified (Playwright, Melanoma Environment):** primary titled correctly; Add-comparison adds
    LR / L-TF / TF / All-metabolic as removable chips below, in add order; each has its **own** scale
    + legend (metab 4.5e-3, lr 8.7e-3, ltf 1.6e-5, tf 4.0e0); columns align to the metabolite's genes
    (CD3E/CD4/ENTPD1/IL2RA; the L-TF comparison correctly narrows to its single overlapping gene
    CD3E); the menu empties as all four are added and an option returns when its ✕ removes the
    section; the cell-type filter narrows ALL sections at once and does NOT clear the chips; switching
    metabolite (Iron→Fatty acid) AND switching tier (Tier3→Tier2) both clear the added sections.
    `tsc -b` + `lint` clean; **0 console errors/warnings**. Screenshot
    `viz/betapanel-comparisons-melanoma.png`.
- **2026-08-01 — Multi-channel beta bundle + standalone SpaceTravLR view (Cycle 1). schemaVersion 4. Playwright-verified.**
  - **Ingest / contract (v4):** `beta/<Tier>.json` now holds ALL FOUR SpaceTravLR feature channels
    (`metab`/`lr`/`ltf`/`tf`) as a self-describing `BetaChannel[]`, replacing the v3 transporter-pair
    -only `byPair` index. `BetaRow` generalized `{env,cell}` → `{a,b}` (member `b` is null for the
    single-member `tf` channel). `buildBeta` drives a config loop over the four CSVs
    (`gene_pairs`/`ligand_receptor`/`ligand_tf`/`transcription_factor`), computing each channel's own
    `cellTypes`/`targetGenes` and sorting rows strongest-first. The `knownBetaKeys` network-membership
    drop is **gone** — the generic view shows raw features; a pair outside the network simply never
    matches a metabolite's pair keys. Re-ran `npm run ingest -- ../Results`: **only Melanoma
    `beta/{Tier1,Tier2,Tier3}.json` + `manifest.json` (schemaVersion 3→4) changed**; no
    edges/nbhd/dataset churn. Tier3 channel rows: metab 1555, lr 2820, ltf 100, tf 2020.
  - **`betaScale.ts`:** scale math (`makeBetaScale`/`formatBeta`/`formatMagnitude`/`BETA_DECADES`)
    unchanged; `groupBeta(byPair,pairKeys,cellTypes)` → `groupChannel(rows,cellTypes)` on the new
    `{a,b}` shape (id `${a}__${b??''}`), and `betaTooltip(r, channel)` now takes channel meta so it
    labels members per channel (export/import vs ligand/receptor vs ligand/TF vs TF).
  - **`BetaSection` (new):** the single reusable per-channel heatmap, with its OWN per-section scale
    (critical — TF is order 1e0, metab 1e-6). Used by both BetaPanel and the new view. `BetaPanel`
    reworked minimally to source its primary heatmap from the `metab` channel via `BetaSection`
    (rows filtered by `betaKey(r.a,r.b) ∈ pairKeysFor(entity)`); its caveat / cell-type select /
    empty states kept. **No comparison chips yet — that's Cycle 2.**
  - **Task 2 — standalone `SpaceTravlrView` (new):** a full-canvas, entity-INDEPENDENT view reading
    the whole beta bundle. Section toggles (per channel), target-gene chips (union across selected
    channels), and a shared cell-type select — all tracked as EXCLUSION sets so new tiers/datasets
    default everything ON. One `BetaSection` per selected channel, each on its own scale. New third
    View button **"SpaceTravLR"** (disabled when `!hasBeta`); `App` hides `EntityPanel` and adds
    `app__body--full` (single-column grid) for it; `selectDataset` retains the view only if valid
    (`graph` always / `nbhd` iff `hasEnvView` / `spacetravlr` iff `hasBeta`). See A21–A23.
  - **Verified (Playwright, Melanoma Tier3):** SpaceTravLR button disabled on no-beta datasets
    (FF Ovarian, and it is disabled wherever `hasBeta` is false), enabled on Melanoma; all 4 channels
    render; per-channel scale proof — own-scale maxes **metab 4.5e-3 / lr 8.7e-3 / ltf 1.6e-5 / tf
    4.0e0**, and the TF section shows order-1 values with spread (1914 numeric cells) while metab
    shows 1e-3..1e-6 (730 numeric) — neither collapses to all ≈0. **DOM-vs-bundle: all 6,495 rendered
    cells' text equals an independent `formatBeta(bundle, per-channel scale)`; 0 text / 0 sign / 0
    hue / 0 missing mismatches** (a negative never renders `+`; hue matches sign; ≈0 carry no tint).
    Section toggle hides/shows a channel; a gene chip removes that column from every section; the
    cell-type select narrows all sections to one block. BetaPanel unchanged behavior (125
    metabolite-scoped cells, 4 metab genes). `tsc -b` + `lint` + `build` clean; **0 console
    errors/warnings**. Screenshot `viz/spacetravlr-view-melanoma.png`.
- **2026-08-01 — Re-ingested: new SpaceTravLR melanoma run (7 target genes, was 4). Data-only; Playwright-verified.**
  - `npm run ingest -- ../Results` over the same 7 datasets. **Only the 3 Melanoma `beta/*.json`
    changed** — the harreman side of every dataset is byte-identical, so no edges/nbhd/manifest churn.
  - What grew: target genes **4 → 7** (added **HAVCR2, HIF1A, MYC** alongside CD3E/CD4/ENTPD1/IL2RA),
    pairs 72 → 76, Tier3 rows 1,555 → 2,680 (Tier1/Tier2 622 → 1,072). The BetaPanel heatmap is
    column-count-agnostic, so **no code change was needed** — 7 columns render with no horizontal
    overflow. Ingest reported no dropped rows (no pair missing from the network `gp` list).
  - Re-ran the smoke item 12 DOM-vs-bundle diff on Melanoma Tier3: **195/195 rendered values
    round-trip** to the bundle, 0 label/sign/hue/floor mismatches; 120 cells correctly read `—`
    (the new genes aren't fit for every direction), 16 exact `0`, 92 `≈0`. `tsc`+`lint`+`build`
    clean, 0 console errors/warnings.
  - Note for later: the new genes are **sparser** than the original 4 — 120 of 315 cells in the
    default view have no coefficient at all. That's honest (`—` ≠ `≈0`), but if Foster wants the
    grid denser we could hide all-missing columns per block.
- **2026-07-23 — Fixed: a selected gene-pair tab vanished when the cutoff emptied it. Playwright-verified.**
  - Foster: dragging the FDR slider until the *selected* pair had no significant interface bounced
    him back to the metabolite ("All") view and lost his place. Wanted behavior: tabs may appear and
    disappear with the cutoff (already correct), but **a pair you picked stays picked** — flagged,
    not removed — and only an explicit tab switch may release it.
  - Root cause: the release-to-"All" `useEffect` added earlier the same day (a fix for "stranded on
    a tab that no longer renders"). It solved the stranding by discarding the selection; the better
    answer is to keep rendering the tab.
  - Fix (all in `GenePairTabs`): the effect is gone. A `tabs` memo appends the selected `gpTab` back
    onto `pairs` as an `nInterfaces: 0` entry when the cutoff has emptied it (guarded on the pair
    really belonging to this metabolite); the tab renders dashed with the word-tag **"not
    significant"** instead of a count, and the empty-strip guard now tests `tabs`, not `pairs`. An
    active-tab `scrollIntoView` keeps it visible — the retained tab sorts last (maxC 0) in a strip
    that overflows at ~16 tabs, and a selected-but-off-screen tab would recreate the same confusion.
  - Verified across all 7 stops with a tab active: at 0.001 the pair is retained, selected, dashed,
    "not significant", graph 0 edges; at 0.005–0.2 it returns to its real count (3) and the other
    tabs' counts still track the cutoff (SLCO2B1–SLCO2B1 9/10/13/14). Survives unrelated re-renders;
    cleared by switching tab and by a metabolite change; with 16 tabs overflowing, the active tab is
    auto-scrolled into view at every stop. 0 console errors/warnings; tsc+lint+build clean.
- **2026-07-23 — Fixed: the FDR slider did nothing on the gene-pair drill-down paths. Playwright-verified.**
  - Foster hit it by picking a gene-pair **tab** in the metabolite view: the graph froze at the
    0.05 edge set no matter where the slider went.
  - Root cause (one assumption, four call sites): the gene-pair helpers in `data/genePairs.ts`
    treated **"present in the gp bundle" as "significant"**, which is only true at harreman's own
    0.05. So `gpEdgesInTier` / `metaboliteSigPairsAtTier` / `genePairsAtInterface` never looked at
    FDR_np, and every consumer of them ignored the cutoff: the GraphView `gpTab` branch (which
    early-returns *before* the metabolite filter), the matching EdgeDetails `gpTab` branch, the tab
    strip's counts, and the on-graph fan-out + panel pair list.
  - Fix: all three helpers now take a `threshold` (defaulted to `DEFAULT_FDR`) and filter with
    `isSelected`; GraphView, EdgeDetails and GenePairTabs pass the live `fdrThreshold`. Tightening
    can now empty a pair, so GenePairTabs also **releases a stale `gpTab` back to "All"** rather
    than stranding you on a tab it no longer renders (guarded on the bundle being loaded, so it
    can't fire during load when `pairs` is legitimately `[]`).
  - Verified: with a tab active, edges now track the cutoff **0/9/10/13/14/14/14** across the stops
    (was pinned at 14), matching an independent bundle computation exactly, and the tab counts
    follow; fan-out gp sub-edges go 0/25/25/28/32/32/32. 0 console errors; tsc+lint+build clean.
  - Lesson worth keeping: "significant-only table" is a property of *how it was generated*, not an
    invariant the UI may lean on once significance became adjustable. See A19.
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
| A20 | An **explicit selection outranks a filter**: a picked gene-pair tab is retained (flagged "not significant") when the FDR cutoff empties it, never auto-released | Foster: being forwarded back to the metabolite view mid-drag loses your place and reads as a bug. The cutoff is an *exploration* control — the user asking "is this pair still significant at 0.01?" needs the answer *in place*, and "no interfaces at this cutoff" is a real answer worth showing, not a reason to discard the question. Filters may add/remove things the user did NOT choose (unselected tabs still come and go); only the user retires their own selection. Generalizes beyond tabs: prefer flagging a chosen thing over silently dropping it. |
| A19 | Significance is **derived from FDR_np at a UI-set threshold** (`isSelected(scores, thr)`), not read from the baked-in `scores.selected` | Lets a slider control the cutoff with no ingest change. Reproduces `selected` exactly at the 0.05 default (0/20,658 mismatches), so it's a safe drop-in. `scores.selected` stays in the contract as harreman's own 0.05 call, but the app no longer reads it for graph significance. Slider stops are discrete because FDR_np is discrete with a low floor; ranking SORT deliberately does NOT depend on the threshold, so moving it never re-sorts the panel or changes auto-selection. |
| A21 | The beta bundle holds **all four SpaceTravLR channels** (`BetaChannel[]`), replacing the transporter-pair-only `byPair` index | The new run emits four feature kinds (metab/lr/ltf/tf), each with its own target-gene set. A `byPair` map keyed on sorted transporter pairs can't represent single-member TFs or ligand→receptor roles, and the generic view needs all channels anyway. `BetaChannel` is self-describing (label/kind/memberLabels/rowHeader) so components hardcode no per-channel meta. Metabolite lookup moves from `byPair[key]` to filtering the `metab` channel's rows by `betaKey(r.a,r.b)`. |
| A22 | **Per-channel scales are mandatory**; each factor group builds its own from its own rows (originally `BetaSection`, now `BetaMatrix` per group) | TF means are order 1e0, metab 1e-6, LR 1e-7, L-TF 1e-9..1e-7. A single shared scale (the natural instinct when stacking channels in one view) would send whole channels to the ≈0 floor — a *correctness* failure, not a styling one. So the renderer, not the view, owns a scale per factor group, and each group states its own range. |
| A23 | Network-membership drop (`knownBetaKeys`) removed from ingest | v3 dropped beta rows for transporter pairs absent from the harreman network, because they had no entity to hang off. The standalone view shows raw features with no entity, so there's nothing to drop against; a pair outside the network simply never matches a metabolite's pair keys downstream (harmless). Keeps ingest honest to the source and avoids silently hiding features. |
| A24 | Standalone SpaceTravLR view is **entity-independent and full-width** (hides `EntityPanel`) | It answers "what does the model say across all features?", not "…for this metabolite" — the side panel's entity selection is irrelevant, so it's hidden and the canvas fills the width (`app__body--full`). Toggles are EXCLUSION sets (empty = all shown) so new channels/genes/tiers never need re-selecting. `selectDataset` keeps the view only if the new dataset can still show it, else falls back to `graph`. |
| A25 | BetaPanel comparisons: added channels use an **INCLUSION** list (`added: BetaChannelId[]`, starts empty), the inverse of the standalone view's exclusion sets | Opposite defaults for opposite intents: the standalone view wants everything shown by default (survey), while the metabolite view is *about that metabolite* — comparisons are a deliberate opt-in the user adds one at a time, so an empty inclusion list is the right default. The compared rows are the channel's full superset (not entity-filtered). Reset on entity/tier/dataset (stale otherwise) but not on cell-type change (that's a lens on the same comparison, per A20's "filters may change unchosen things; only the user retires their own selection"). |
| A26 | **Cell-type-major** matrix (`BetaMatrix`), UNION gene columns, BLANK ≠ `≈0` — replaces the channel-major `BetaSection` stacking | Foster's REVISION 1: grouping by factor channel repeated each cell type per factor and columns didn't align, so cross-factor reading was hard. Cell-type-major shows each cell type once with the factor groups stacked beneath and ALL genes in a fixed column order — so "for this cell type, which factors move which genes?" is one glance. Columns are the UNION (superseding Cycle-2's intersect-to-metab-genes) so a gene only a comparison factor covers still appears; where a group has no coefficient the slot is BLANK whitespace (kept, not dropped/shifted), kept honestly distinct from `≈0` (measured-but-negligible). Scales stay **per factor group** (A22): TF ~1e0 and metab ~1e-6 in the same block would otherwise collapse to one floor. |

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
12c. FDR slider vs the gene-pair drill-downs (the 2026-07-23 regression — re-check all four):
    with a **gene-pair tab** active the graph must re-filter as you drag (not freeze at the 0.05
    set), the tab **counts** must change, the **on-graph fan-out** sub-edge count must change, and
    the EdgeDetails **"Carried by N significant pairs"** list must shrink. Tighten until the
    **selected** pair has no interfaces left: its tab **stays, stays selected, and reads "not
    significant"** (dashed) with an empty graph — it must NOT fall back to "All" (see A20).
    Unselected pairs that empty still drop out of the strip. Loosening restores its count; clicking
    another tab, or changing metabolite/tier/dataset/kind, is the only thing that clears it.
12. Beta correctness (worth re-running after any scale change — this is where the bugs are):
    in the console, diff the DOM against `beta/<Tier>.json` and assert, per cell, that the label
    matches source, a negative never renders `+`, hue matches sign, and `≈0` cells carry no tint.
    Cycle 1: 6,495 Tier3 cells across all four channels round-trip to `formatBeta(bundle, per-channel
    scale)` with 0 mismatches.
13. **Standalone SpaceTravLR view** (Melanoma): the third View button is enabled only when the
    dataset has betas (disabled on e.g. FF Ovarian / Human Lung). Clicking it hides the entity panel
    and fills the width. Layout is **cell-type-major** (`BetaMatrix`): each cell type once, the
    selected factor channels stacked beneath as divider-labelled groups, union target-gene columns in
    fixed position. **Per-group scales:** the TF group shows order-1 values while metab shows ~1e-6 —
    each divider prints its own "own scale |β| ≤ …". Toggling a **Section** chip adds/removes that
    factor group (and any columns only it contributed); toggling a **Target gene** chip adds/removes
    that union column; the **Cell type** select narrows to one block. Switching to a no-beta dataset
    while in this view falls back to the Graph view. 0 console errors.
14. **BetaPanel comparison chips + cell-type-major matrix** (Melanoma Environment view): each cell
    type appears ONCE; beneath its label the primary group ("This metabolite's transporter pairs") and
    any added comparison groups are stacked, sharing ONE union set of gene columns in fixed position.
    "Add comparison ▾" adds LR / L-TF / TF / "All metabolic transporters" as removable chips (removal
    via the chip ✕ — the matrix has no per-section ✕). Adding TF surfaces TF-only columns
    (CTLA4/FOXP3/IL10) as **blank whitespace** in the metab group (blank ≠ `≈0`); each factor group
    keeps its own scale (metab ~1e-6, TF ~1e0). The cell-type filter narrows ALL groups at once and
    does NOT clear the added ones. Switching metabolite OR tier clears the added groups.

## Changelog
- 2026-08-01: **Cell-type-major matrix layout (Cycle 3 / REVISION 1).** New `BetaMatrix.tsx`
  (+ `.module.css`) replaces `BetaSection` (deleted): each cell type shown ONCE, factor groups stacked
  beneath, ONE union set of target-gene columns in fixed position, per-group scales, blank whitespace
  for absent coefficients (kept distinct from `≈0`). `SpaceTravlrView` renders one `<BetaMatrix
  allowedGenes={selectedGenes}/>`; `BetaPanel` builds primary + `added` groups and renders one
  `<BetaMatrix/>` with union columns (`allowedGenes` undefined) — kept the Compare chip bar, cell-type
  select, caveat, and the entity/tier/dataset reset. Dead `BetaSection` CSS removed from
  `BetaPanel.module.css`. UI-only, no ingest/schema/contract change. Playwright-verified on Melanoma
  (both views cell-type-major; union columns incl. TF-only genes with metab blank there; per-group
  scales; add/remove factor + gene chips; cell-type filter; metabolite/tier reset; 2,145 value cells
  round-trip to `formatBeta(bundle)` with 0 mismatches, 6,010 blanks all "no coefficient"). tsc+lint
  clean, 0 console errors. See A26; smoke items 13/14 updated.
- 2026-08-01: **BetaPanel comparison chips (Cycle 2).** `BetaPanel` now titles its primary section
  ("This {metabolite|gene pair}'s transporter pairs") and adds a "Compare" bar with removable chips +
  an "Add comparison ▾" `<select>` (metab → "All metabolic transporters", others → `channel.label`).
  Each added channel renders a `BetaSection` below the primary with the whole channel's rows, columns
  intersected to the metabolite's target genes, the same cell-type filter, and a ✕ to remove. Local
  `added` list resets on entity/tier/dataset change but not on cell-type change (A25). UI-only — no
  ingest/schema/contract change. Playwright-verified on Melanoma (own per-channel scales, column
  alignment incl. L-TF narrowing to CD3E, add/remove ↔ menu, cell-type filter across all sections,
  metabolite+tier reset); tsc+lint clean, 0 console errors. Smoke item 14 added.
- 2026-08-01: **Multi-channel beta bundle + standalone SpaceTravLR view (Cycle 1).** schemaVersion 4.
  (1) **Ingest/contract:** `beta/<Tier>.json` = `{ tier, cellTypes, targetGenes, channels: BetaChannel[] }`
  covering all four channels (metab/lr/ltf/tf); `BetaRow` `{env,cell}`→`{a,b}` (`b` null for tf);
  `byPair` and the `knownBetaKeys` network drop removed (A21, A23). Re-ingest changed only Melanoma
  `beta/*` + manifest schemaVersion. (2) **`betaScale.ts`:** `groupBeta`→`groupChannel(rows,cellTypes)`;
  `betaTooltip(r, channel)` takes channel meta; scale math untouched. (3) **`BetaSection` (new)** =
  the reusable per-channel heatmap with its OWN per-section scale (A22); **`BetaPanel`** reworked to
  render its primary heatmap through it from the `metab` channel (no comparison chips yet — Cycle 2).
  (4) **`SpaceTravlrView` (new)** = full-canvas, entity-independent view with Sections/Target-gene/
  Cell-type controls (exclusion sets), one `BetaSection` per selected channel (A24); third View
  button "SpaceTravLR" (disabled `!hasBeta`), `EntityPanel` hidden + `app__body--full`, `selectDataset`
  view-retention guard. Playwright-verified on Melanoma Tier3: per-channel scales proven (metab
  4.5e-3 / lr 8.7e-3 / ltf 1.6e-5 / tf 4.0e0; TF order-1 with spread, none collapsed to ≈0), all
  6,495 rendered cells match `formatBeta(bundle)` (0 text/sign/hue/missing mismatches), toggles +
  cell-type filter work, button disabled on no-beta datasets. tsc+lint+build clean, 0 console errors.
- 2026-08-01: **Data refresh — new SpaceTravLR melanoma run re-ingested.** `npm run ingest -- ../Results`;
  only `Primary_Dermal_Melanoma/beta/{Tier1,Tier2,Tier3}.json` changed. Target genes 4 → 7 (+HAVCR2,
  HIF1A, MYC), pairs 72 → 76, Tier3 rows 1,555 → 2,680. No code, schema, contract or ingest change —
  `BetaPanel` already derives its columns from `targetGenes`. Verified per smoke item 12 (195/195
  cells match the bundle, 0 mismatches) and item 11; build + console clean.
- 2026-07-23: **Fix — a selected gene-pair tab vanished (and forwarded you to "All") once the FDR
  cutoff left it with no significant interface.** Removed the release-to-"All" `useEffect` added
  earlier the same day and replaced it with retention: `GenePairTabs` now renders a `tabs` memo =
  `pairs` plus, when the cutoff has emptied the selected `gpTab`, that pair re-appended as an
  `nInterfaces: 0` entry (guarded on it belonging to the current metabolite; genes ordered to match
  the id so the label is unchanged). The retained tab is dashed (`.emptied`) and reads "not
  significant" in place of its count, with an explanatory `title`. The empty-strip guard now tests
  `tabs` instead of `pairs`, and an active-tab `scrollIntoView` keeps the selection visible in the
  overflowing strip. Unselected pairs that empty still disappear as before. No store, data, ingest
  or contract change — `gpTab` is still cleared by tab switch / metabolite / tier / dataset / kind.
  See A20; smoke item 12c updated.
- 2026-07-23: **Fix — FDR slider was a no-op on every gene-pair drill-down path.** `gpEdgesInTier`,
  `metaboliteSigPairsAtTier` and `genePairsAtInterface` (`data/genePairs.ts`) all equated "in the gp
  bundle" with "significant" and now take a `threshold` (default `DEFAULT_FDR`), filtering with
  `isSelected`. Callers updated: GraphView (`gpTab` branch + fan-out), EdgeDetails (`gpTab` resolve
  + pair breakdown), GenePairTabs (tab list/counts). GenePairTabs additionally clears a `gpTab` that
  the cutoff has emptied, falling back to "All". No ingest/schema/contract change.
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
