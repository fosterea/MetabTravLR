# 00 — Overview

## What this is
A **local single-page web app** to visually explore **cell–cell metabolite crosstalk** inferred
by harreman (and, later, SpaceTravLR). The core artifact on screen is a **graph**:
- **Nodes = cell types** (within a chosen tier).
- **Edges = the undirected interfaces** between two cell types where a chosen **entity** (a
  metabolite, or a transporter gene pair) is significantly co-exchanged, weighted by
  communication strength and significance.

## Who it's for / the driving use case
Foster's friend wants to answer *"which metabolites are influencing T cells?"* The app makes
that answerable — pick the T-cell tier, rank metabolites by T-cell involvement, click one, see
which cell-type interfaces light up — **but the code is deliberately use-case-agnostic**: it
knows about datasets, tiers, cell types, and entities, not about T cells specifically.

## The domain data model (what the numbers mean)
Source: harreman spatial cell–cell communication on Xenium data (see the parent project's
`DataForClaude/documentation/05_harreman_reference.md`). Key facts baked into the design:

- **Dataset** = one `easy_download` folder of harreman outputs. Today there is exactly one.
  Eventually many, laid out as `<root>/<datasetName>/easy_download/…`.
- **Tier** = one cell-type annotation granularity. They form a **parent hierarchy of T-cell
  resolution**:
  - Tier1: `T Cell` vs `other`
  - Tier2: `CD8 T Cell` vs `other`  (refines Tier1's T Cell)
  - Tier3: `Effector CD8` / `Gamma delta` / `ISG CD8` / `Proliferating CD8` vs `other`
    (refines Tier2's CD8 T Cell)
- **Edge = an undirected cell-type interface.** harreman reports one row per *unordered* pair
  `(CT1, CT2)` incl. self-pairs (the diagonal `CT==CT` = within-cell-type). **`CT1→CT2` is NOT
  a direction** — it is a sorted-label artifact. Render edges undirected; render the diagonal
  as a self-loop / node-level mark.
- **Entity-agnostic edges.** An edge is keyed on an *entity*:
  - **metabolite** — the primary view (`…cell_com_df_m.csv`).
  - **gene pair** — the transporter pair that carries a metabolite (`…gp_sig.csv`).
    **gp ↔ metabolite is many-to-many**: one pair can serve several metabolites and vice
    versa, so a gene-pair view cannot be collapsed 1-to-1 onto a metabolite.
- **Scores per edge**: `C_np` (non-parametric strength, the primary magnitude), `FDR_np`
  (significance), `selected` (harreman's call: `FDR_np < thr AND C_np > 0`), plus the
  parametric `C_p`, `Z`, `Z_FDR`. **Magnitudes span orders of magnitude** (self / `other↔other`
  edges dwarf the interesting T-cell interfaces) → views must normalize per-view or use a log
  scale, and should let the user emphasize the interface edges.

## Product vision
### MVP (this milestone)
- Select dataset → tier → metabolite (from a rankable side panel) → render the cell-type graph
  with edge width ∝ strength and significance encoded (color/solidity), diagonal self-loops
  shown, T-cell interfaces emphasized.
- A details panel for the selected metabolite (its gene pairs, per-tier T-cell involvement).
- Verified end-to-end in a real browser via the playwright MCP server.

### Roadmap (designed-for, not yet built) — see `04_decisions_and_state.md`
- **Gene-pair view** toggle (data already ingested); show the metabolites a pair serves.
- **Parent cell-type marking**: use the tier hierarchy to relate a Tier3 subtype back to its
  Tier2/Tier1 parent (needs an annotation crosswalk — currently absent, tolerate absence).
- **Multiple datasets** and the `<root>/<datasetName>/easy_download` deploy layout.
- **SpaceTravLR integration**: select genes / gene sets / precomputed contradicting gene sets
  (positive vs negative), where an edge carries **two values** → encode as **width + diverging
  color**. The edge model already reserves `value`/`sign` for this.
- Easy static deployment (the build is already relative-path / self-contained-friendly).
