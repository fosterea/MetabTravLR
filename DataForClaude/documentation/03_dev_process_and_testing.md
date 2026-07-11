# Dev/review loop + testing regimen

> How we build MetabTravLR safely: a **dev → review → fix loop** driven by two subagents,
> backed by a **layered test regimen** designed for the fact that the real data + heavy env
> live on Savio, not here. This is a starting process — **we will refine it as we go.**
> Guiding constraints: minimal/surgical package edits, few dependencies, be smart about
> long-running tests.

---

## 1. The dev/review loop

Main loop (Opus, orchestrator = me) holds the plan and runs this cycle per change unit
(one small, self-contained edit + its tests):

```
 spec ──► [metab-dev] implements + writes/updates tests + runs Tier-0/1 tests
              │  returns: diff summary, what it changed, test output, self-assessment
              ▼
        [metab-review] adversarially reviews the diff (correctness, numerics, scope,
              │        test adequacy, tractability) ──► findings (severity-ranked)
              ▼
   findings empty / all "no-change-needed"?  ── no ──► feed findings back to metab-dev ──┐
              │ yes                                                                        │
              ▼                                                                            │
   main loop: run heavier tests (Tier 2 if env available) + human checkpoint     ◄────────┘
```

Rules that keep it honest:
- **Dev and reviewer are separate agents** (fresh context) so the reviewer isn't anchored to
  the dev's assumptions. Reviewer is told to *try to break it*, default to skepticism, and
  cite `file:line` + a concrete failure scenario.
- **Bounded loop:** max ~3 dev↔review passes per unit; if not converged, escalate to me/Foster
  rather than thrash.
- **Small units:** one surgical change at a time (e.g. "thread `extra_lr` through `SpaceShip`")
  — never a big-bang. Each unit must leave the suite green.
- **The reviewer also guards scope:** flags any edit that isn't minimal/surgical, adds a
  dependency, or belongs in a separate script instead of the package.

### Model choice (Foster asked "sonnet?") — my recommendation, with pushback
- **metab-dev = Sonnet.** Agree — fast, economical, plenty capable for surgical edits + tests.
- **metab-review = Sonnet at *high* reasoning effort by default, and I'd escalate the
  reviewer to Opus for numerically-sensitive changes** (the O(N²)→sparse kernel, the β/`splash`
  math, the metabolite flux term). Reason to push back on "Sonnet for everything": review is
  where a cheap catch prevents an expensive wrong run on the cluster, and diversity between
  the dev and reviewer models catches blind spots a same-model pair shares. Cost is low
  because review reads a small diff, not the whole repo.
- **Orchestration/planning stays Opus** (me).
- Net: Sonnet-dev + Sonnet-high-review to start; opt the reviewer up to Opus on numeric units.
  Easy to dial — revisit once we see quality.

---

## 2. Testing regimen (layered by dependency weight & runtime)

The heavy scientific env (torch, scanpy, celloracle_tmp, group_lasso, commot, magic, numba…)
is a **cluster** concern. So we tier tests by what they need, and run the cheap tiers in the
loop, the expensive tiers on Savio.

| Tier | Name | Deps | Runtime | Runs where / when |
|---|---|---|---|---|
| **0** | Pure-logic / spoof | numpy, pandas only | seconds | in the dev/review loop, anywhere |
| **1** | Unit (light model) | + torch/sklearn (no scanpy/CO) | seconds–1 min | loop, if env present |
| **2** | Real demo-data smoke | full env + `data/*.h5ad` | minutes | Savio (or a full local env), pre-merge |
| **3** | Full integration | full env + big data | long | Savio, milestone only |

Follow the existing style in `tests/` (unittest + synthetic `make_regression`/`np.random`
AnnData, `sys.path` to `src`); no new test framework or plugins.

### Tier 0 — spoof-data logic tests (our first line of defense)
Test our *new, pure* logic with tiny hand-built inputs and **known-answer assertions**, no
model training:
- **Harreman loaders:** given a tiny fake `harreman_network.json` + `_m` CSV, assert we
  select the right significant metabolites for a tier and expand them to the correct gene
  pairs (incl. **both orientations**, homotypic pairs, FDR threshold behavior, the
  `cell_com_df_gp_sig`-duplicate quirk).
- **extra_lr builder:** metabolite selection → `list[(export, import)]`, deduped, filtered to
  genes present in `adata.var_names`, both directions present.
- **Gene-set parser:** JSON/dict `{label:[genes]}` → validated, missing genes reported.
- **Signed aggregation:** given a fake betadata frame with known `beta_<e>$<i>` columns,
  assert `score = mean_pos − mean_neg` computes as expected, including sign and NaN handling.
- **Column-name round-trips:** `beta_<e>$<i>` parse/format matches `beta.py::BetaFrame`
  conventions (guard against `$`/`#` mixups).

### Tier 1 — light model units
- `extra_lr` actually produces a `beta_<e>$<i>` **modulator column** in a `SpatialCellularProgramsEstimator`
  built on a tiny synthetic adata (extend `tests/test_spacetravlr.py`).
- Gene-subset training: a trainer seeded with 2–3 target genes writes exactly those
  `*_betadata.parquet` and nothing else.
- **No-TF orphan path:** a target gene with metabolite modulators but no TF regulators —
  assert current behavior (orphan-skipped) and, once we relax it, that it trains.
- **Numeric guards for the O(N²) fixes:** if/when we swap in sparse radius-neighbors, a
  Tier-1 test asserts the sparse received-ligand matrix matches the dense reference **within
  tolerance** on a few-hundred-cell synthetic set (correctness before scale).

### Tier 2 — real demo-data smoke (gene-focused = the speed trick)
Runs on the tonsil demo (`data/Slidetags_human_tonsil.h5ad`, human, has the paper's genes)
but **only trains a tiny curated gene list** so it finishes in minutes:
- **Focus gene list (test fixture):** a handful of well-characterized tonsil genes as
  targets — e.g. `PAX5, BCL6, FOXO1, IL21, CD40` (from the paper's figures). Keep this list
  in `tests/fixtures/` so Tier-2 runs are fast and deterministic. *(We'll also want a real
  metabolite/target list from Foster; until then, tests spoof metabolite edges from two
  expressed tonsil genes to exercise the plumbing end-to-end.)*
- **Assertions:** setup produces the expected artifacts; training the focus genes yields
  betadata with sane shapes and finite β's; a spoofed metabolite pair shows up as a
  `beta_<e>$<i>` column; the coefficient-read + signed-aggregation path returns a ranking.
- Keep N small (subsample the demo to ~2–5k cells) so even the dense O(N²) path is fine here
  — Tier 2 validates *logic on real data*, not scale. Scale is Tier 3 on Savio.

### Tier 3 — full integration (Savio, milestones)
Real Xenium subset with real harreman edges + the actual gene sets; validates the O(N²)
mitigations at 100k+ cells and end-to-end runtime/memory. Driven by our scripts + SLURM.

### Being smart about long runs (Foster's concern)
- Default to Tier 0/1 in the loop; gate Tier 2/3 behind explicit invocation.
- **Gene-focusing is the primary lever** — every real-data test names its target genes.
- Cache setup artifacts (`_adata.h5ad`, `celloracle_links.pkl`) between test runs; never
  recompute the GRN in a fast test.
- Mark slow tests (e.g. a `@slow` / separate dir) so Tier 0/1 stay a quick default.

---

## 3. Organization conventions (keep it tidy)
- **Package edits** (`src/SpaceTravLR/…`): only the minimal surgical hooks (thread `extra_lr`,
  optional gene subset, relax orphan skip, later the sparse kernel). Each behind a default
  that preserves current behavior.
- **Our logic** (`metab_processing/…` and/or a new `metabtravlr/` script package): harreman
  loading, edge building, gene-set parsing, coefficient reading + aggregation, plotting,
  SLURM driver. This is where most code lives.
- **Tests** (`tests/`): mirror the layout; Tier-0/1 alongside existing tests, fixtures in
  `tests/fixtures/`, Tier-2/3 clearly marked slow.
- **Docs** (`DataForClaude/documentation/`): update `04_decisions_and_state.md` whenever a
  decision changes; this process doc evolves with the workflow.
- Agent specs live in `.claude/agents/metab-dev.md` and `.claude/agents/metab-review.md`.

## 4. Run log & learnings

### Run 1 — CU-3 gene-focus (`SpaceShip(genes=…)`), 2026-07-11
The first live use of the dev/review loop. What we learned (leanings to carry forward):

**What worked well — keep doing:**
- **Baseline before dispatch.** Establishing the env + green baseline ourselves (19 fast tests,
  `spacetravlr_env`) and identifying the reusable test fixtures (`test_oracle.py`'s
  `MockRegulatoryFactory` + `generate_realistic_data()`; `test_spacetravlr.py`'s
  `create_test_adata` + mocks) gave the dev agent a known-good starting point and made its
  tests idiomatic. **Always baseline + scout fixtures first.**
- **Sonnet for BOTH dev and review was sufficient** for this plumbing change. Dev produced a
  clean surgical diff + 13 tests and *self-flagged two pre-existing bugs*; review
  *independently reproduced* both bugs and returned calibrated, non-rubber-stamp nits.
  → Leaning: **Sonnet/Sonnet for plumbing/logic CUs; escalate the reviewer to Opus only for
  numerically-sensitive CUs** (the O(N²)→sparse kernel, β/splash/flux math, metabolite CU-1).
- **The adversarial review earned its cost** — it re-derived the logic by hand and empirically
  reproduced the bug claims rather than trusting the dev. High value; keep for substantive CUs.

**Process improvements to adopt:**
- **Agent invocation:** we dispatched via `subagent_type: general-purpose` + `model` override +
  inline role/spec (pointing at `.claude/agents/*.md`), because it was unclear whether the
  custom `metab-dev`/`metab-review` types register mid-session. TODO: confirm whether
  `.claude/agents/*.md` are directly invocable as `subagent_type`; if so, use them (auto-applies
  their system prompt + model). Until confirmed, the general-purpose + inline-spec path is the
  reliable default.
- **Don't spawn a full review agent for trivial nits.** For the nit-fix pass, a self-run of the
  suite is enough; reserve the review agent for substantive change. (Applied on Run 1's 2nd pass.)
- **Continue the same dev agent via SendMessage** for follow-up passes (preserves its context)
  rather than a fresh Agent call.
- Timings: dev ≈ 20 min, review ≈ 12 min (synchronous). Fine for one CU; for independent CUs,
  dispatch in the background/parallel.

**Testing gaps this run exposed (act on before the metabolite CU):**
- Real-training tests on tiny/noisy data land in the group-lasso **`R² < 0.15` → zeroed-anchors
  shortcut**, so they exercise the queue/estimator/betadata-write plumbing but NOT the CNN
  training inner loop or real β emission. For CU-1 (metabolite group) we need a fixture that
  actually trains (R² ≥ 0.15) so we can assert a real `beta_<export>@<import>` column with a
  finite learned value.
- Pre-existing bugs will bite our real runs — see `04_decisions_and_state.md` "Known pre-existing
  bugs". The `activation`-kwarg crash in particular blocks any real `fit()` that hits the
  poor-fit branch; decide whether to fix it (tiny, core) before scaling metabolite training.

Refine again after CU-1.
