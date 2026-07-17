# MetabTravLR crosstalk explorer (`viz/`)

A local single-page app for exploring cell–cell **metabolite crosstalk** graphs from harreman
(and later SpaceTravLR) outputs. Pick a dataset → tier → metabolite and see which cell-type
interfaces exchange it.

> This is a self-contained side project. Its design, conventions, and state live in `docs/`.
> If you're an agent, read `CLAUDE.md` first. **Only edit files under `viz/`.**

## Quick start
```bash
cd viz
npm install

# Generate app data from the harreman outputs (re-run when the source data changes):
npm run ingest -- ../easy_download --id harreman --name "Harreman — metabolite crosstalk (Xenium)"

npm run dev        # http://localhost:5173
```

## Scripts
- `npm run dev` — Vite dev server (used for playwright testing).
- `npm run ingest -- <path>` — build the app data JSON from an `easy_download` folder (or a root
  of `<datasetName>/easy_download/…` folders). See `docs/05_data_contract.md`.
- `npm run build` / `npm run preview` — static production build / preview.
- `npm run lint` / `npm run format` — ESLint / Prettier.

## Docs
See [`docs/README.md`](docs/README.md). Start with `docs/00_overview.md`.

## Stack
Vite · React + TypeScript · Cytoscape.js (graph) · Zustand (state) · Papa Parse (ingest only).
Testing via the **playwright MCP server** only.
