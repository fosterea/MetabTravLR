# 03 — Dev process & testing

## The dev/review sub-agent loop
Work proceeds in small **change units** (one component / one feature / one fix), each run
through a two-agent loop. Sub-agents inherit the golden rules in `viz/CLAUDE.md` — most
importantly **they may only touch `viz/`** and **test only via the playwright MCP server**.

1. **Plan (orchestrator).** Define the change unit + acceptance criteria. Point the agent at
   the relevant docs (`01`, `02`, `05`) and files.
2. **Dev agent.** Implements the change unit within `viz/`, keeps to the conventions in
   `02_style_and_conventions.md`, runs `npm run lint` + `tsc`, and — for anything with a visual
   surface — drives the dev server with the playwright MCP tools to confirm it renders and
   behaves. Reports a precise diff summary + what it verified.
3. **Review agent.** Adversarially reviews the diff for: correctness (esp. the data mapping and
   the **undirected-edge / log-scale** rules), scope creep (touched anything outside the unit?
   outside `viz/`?), convention/style violations, dead code, and test adequacy. Reports
   severity-ranked findings. Does **not** rewrite — it reports.
4. **Orchestrator** curates findings, has the dev agent fix the real ones, re-verifies, commits.

Briefs for the two roles live in this section — reuse them verbatim so behavior is consistent:

**Dev brief skeleton:**
> You are a web-dev sub-agent for the `viz/` app (MetabTravLR crosstalk explorer). Read
> `viz/CLAUDE.md` and `viz/docs/01_architecture.md`, `02_style_and_conventions.md`,
> `05_data_contract.md` first. **Only edit files under `viz/`.** Implement: <change unit +
> acceptance criteria>. Follow the conventions exactly (design tokens, log-scale edges,
> undirected edges, Zustand store, Cytoscape wrapper). Run `npm run lint` and `tsc -b` clean.
> If there is a visual surface, start `npm run dev` and use the **playwright MCP** tools to load
> http://localhost:5173, exercise the flow, and take a screenshot as evidence. Report a concise
> diff summary + exactly what you verified (with the screenshot).

**Review brief skeleton:**
> You are a review sub-agent. Review ONLY this diff (files under `viz/`): <diff/paths>. Check
> correctness of the data→visual mapping, adherence to `02_style_and_conventions.md`, that
> edges are treated as **undirected** and magnitudes use the **log/normalized scale**, scope
> creep (nothing outside the change unit or outside `viz/`), dead code, and whether the
> playwright verification actually exercised the feature. Report severity-ranked findings
> (Critical/High/Medium/Low) with file:line and a concrete failure scenario. Do not edit code.

## Testing with the playwright MCP server
- Start the app: `npm run dev` (serves http://localhost:5173; leave it running in the
  background).
- Use the `playwright` MCP tools to navigate, snapshot the accessibility tree, click/select,
  read console messages, and screenshot. **No other test framework is installed or allowed.**
- Minimum verification for any UI change:
  1. App loads with no console errors.
  2. The target flow works (e.g. select tier → select metabolite → graph updates).
  3. A screenshot is captured as evidence and referenced in the change summary.
- Keep a short manual **smoke checklist** current in `04_decisions_and_state.md` so any session
  can re-verify quickly.

## Definition of done (per change unit)
- `tsc -b` and `npm run lint` clean (0 warnings).
- Playwright smoke passes; screenshot captured.
- Conventions honored; no files touched outside `viz/`.
- `04_decisions_and_state.md` updated (state/roadmap/decisions).
- Committed with a scoped message (`viz:` prefix), staging only `viz/**` (+ root-CLAUDE pointer).
