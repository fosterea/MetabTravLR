# CLAUDE.md — start here

This is Foster's fork of **SpaceTravLR**, being adapted into **MetabTravLR**: a tool that
scores **which metabolites (harreman import/export transporter gene pairs) modulate which
target gene sets**, by **analyzing the model's learned coefficients directly** (no
perturbation prediction).

**All project context, plans, and decisions live in `DataForClaude/documentation/` — read
that folder's `README.md` first, then `00_overview.md`.** We keep durable notes there (in
the repo, transportable) rather than in Claude's private memory.

Working agreements:
- The user goes by **Foster**. Build an accurate mental model of the code before writing code.
- Changes to the existing package must be **minimal and surgical**; project-specific logic
  goes in separate scripts (e.g. `metab_processing/`). Don't touch the repo's `docs/` Sphinx
  site. **Don't install heavy/unnecessary dependencies.**
- Dev/review agent loop + testing regimen: see
  `DataForClaude/documentation/03_dev_process_and_testing.md`.
