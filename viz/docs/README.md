# viz/ docs — index

Documentation for the **MetabTravLR crosstalk explorer** — a local single-page app for
visualizing cell–cell metabolite-crosstalk graphs. This folder is the source of truth for the
app's design, conventions, and state. It is separate from the parent project's
`DataForClaude/documentation/` (which covers the data pipeline / research); read that only when
you need to understand where the *data* comes from.

## Read in this order
- **`00_overview.md`** — what the app is, who it's for, the domain data model, the product
  vision (MVP → roadmap). Start here.
- **`01_architecture.md`** — tech stack + *why*, folder layout, data flow, the ingest adapter,
  deployment posture.
- **`02_style_and_conventions.md`** — visual style rules + code conventions. Follow these so
  everything stays consistent across sessions and sub-agents.
- **`03_dev_process.md`** — the dev/review sub-agent loop and how to test with the playwright
  MCP server.
- **`04_decisions_and_state.md`** — **living** decision log + current state + roadmap/TODOs.
  Update this every session.
- **`05_data_contract.md`** — the normalized JSON schema the app consumes and how the adapter
  maps harreman outputs onto it.
- **`06_hosting.md`** — how to deploy the static build (Firebase / GitHub Pages / Cloudflare /
  Netlify), access-control options, and why it's effectively free at this scale.

## Parent-project references (read-only, for data understanding)
- `../../DataForClaude/documentation/05_harreman_reference.md` — what harreman outputs mean;
  the **`CT1→CT2` arrow is undirected**; gp↔metabolite is many-to-many.
- `../../DataForClaude/documentation/README.md` — the broader MetabTravLR adaptation.
