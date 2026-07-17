# 02 — Style & conventions

Consistency rules for every session and sub-agent. If you deviate, record why in
`04_decisions_and_state.md`.

## Design tokens (single source of truth = `src/theme.css`)
All colors/spacing/typography come from CSS custom properties on `:root`. **Never hard-code a
hex value in a component** — add or reuse a token. The app is **dark-first** (data viz reads
better on dark) with a light override via `@media (prefers-color-scheme: light)`.

Token groups (see `theme.css` for the actual values):
- `--bg-*` surfaces (app, panel, canvas), `--border-*`, `--text-*` (primary/secondary/muted).
- `--accent` (interactive/selection), `--accent-weak`.
- **Domain colors** — keep semantic and colorblind-safe:
  - `--cell-tcell` / `--cell-other` node families (T-cell lineage vs background).
  - Edge significance: `--edge-sig` (selected/significant) vs `--edge-nonsig` (muted).
  - **Reserved diverging pair** `--val-pos` / `--val-neg` for the future SpaceTravLR signed
    (two-value) edges — do not repurpose.
- Spacing scale `--sp-1..6` (4/8/12/16/24/32), radius `--radius`, `--font-sans`, `--font-mono`.

## Visual encoding rules (the graph is the product — keep it legible)
- **Edge width ∝ communication strength.** Magnitudes span orders of magnitude → map on a
  **log / normalized-per-view scale** (see `src/data/scales.ts`), never raw linear. Clamp to a
  sensible px range so one giant self-edge can't swamp the view.
- **Significance is encoded separately from strength**: significant (`selected`) edges are
  solid/opaque in `--edge-sig`; non-significant edges are thin/faded/dashed in `--edge-nonsig`.
  Default the view to significant-only with a toggle to reveal the rest.
- **The diagonal (`source===target`) is a self-loop / within-cell mark**, visually distinct
  from between-type interfaces.
- **Emphasize the interesting edges**: T-cell interfaces should be easy to pick out (the app
  may highlight them) without hard-coding "T cell" into logic — derive from node identity
  passed in, not from string checks scattered in components.
- **Legend is mandatory** whenever an encoding is on screen (width scale + significance +
  colors). Accessibility: don't rely on color alone — pair with width/opacity/dash.
- Future two-value edges: **width = |value|, color = sign** (`--val-pos`/`--val-neg` diverging).
  Design components so adding this is a scale swap, not a rewrite.

> When building any chart/legend/color choice, consult the `dataviz` skill first for palette
> and encoding discipline; keep results theme-aware (dark + light).

## Code conventions
- **TypeScript strict.** No `any` in app code (adapter script may be looser). Types for data
  live in `src/data/types.ts` and must stay in sync with `05_data_contract.md`.
- **Imports** use the `@/` alias for `src/` (configured in `vite.config.ts` + `tsconfig.json`).
- **Components** are function components; presentational components take props and read the
  store via hooks — no data fetching inside leaf components. Keep files small and single-purpose.
- **State** lives in `useVizStore` (Zustand). Derive, don't duplicate: selectors compute
  view-model from store + loaded data. No prop-drilling of global selection.
- **Cytoscape** lives behind a thin React wrapper in `src/graph/`. The rest of the app passes it
  plain data + a stylesheet; components never reach into the cy instance ad hoc.
- **No dead code / no `console.log`** left in committed code (a dev-only logger is fine).
- **Naming**: `camelCase` vars, `PascalCase` components/types, `SCREAMING_SNAKE` consts, files
  match their default export (`GraphView.tsx`, `useVizStore.ts`).
- Run `npm run lint` and `npm run format` before committing. Lint must be clean (0 warnings).

## Accessibility & UX baseline
- Keyboard-reachable controls; visible focus rings (use `--accent`).
- Sufficient contrast in both themes; test both.
- Loading and empty states are explicit ("no significant edges for this metabolite at this
  tier"), never a blank canvas.
