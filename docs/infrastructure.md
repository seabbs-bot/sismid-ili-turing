# Infrastructure

Tooling, data, scoring, and workflow.
Filled in as each piece lands, so this document tracks what actually exists.

## Status of pieces

| Piece | State | Notes |
|---|---|---|
| Repo on `seabbs-bot` | done | `seabbs-bot/sismid-ili-turing`, local at `~/code/seabbs/sismid-ili-turing` |
| Target hub clone | done | `~/code/external/sismid-ili-forecasting-sandbox` |
| Docs | done | brief, plan, infrastructure, steer-log, eda, reports index and template |
| Julia project | done | Turing, Mooncake, Pathfinder, ScoringRules, Arrow, DataFrames |
| Data export to Arrow | done | from the course R package `.rda` objects |
| WIS scoring helper | done | natural and log scale, on `ScoringRules.jl`; mean WIS ~0.29 on a real season-1 validation split |
| Base model | done | Phase 1; fits via Pathfinder + Mooncake |
| Forecast + hubverse writer | done | Phase 1; produces a valid 11x4x23 hub table end to end |
| Bayesian workflow checks | todo | prior/posterior predictive, residuals; per candidate, Phase 2 |
| Tree search (Phase 2) | in progress | Round 1: `v1-ar-high`, `v2-mvn-season`, `v3-diff`, `v4-tv-ar`, `v5-backfill`, via a resilient round-runner engine |
| Submission smoke-test PR | done | draft PR #59 on the reichlab hub passed the hub's validate-submission CI |
| Reproducible submission tooling | in progress | `scripts/submit.jl` + `docs/submission.md`; dress-rehearsed locally with the real base model output; external PR stays paused for Sam's go-ahead |
| Documenter docs site | todo | renders Julia API docs; links brief, plan, infrastructure, contracts, steer-log, `docs/eda/`, and per-loop reports |

## Data

Source objects live in the course package
(`~/code/nfidd/sismid-forecasting/data`) and are exported to Arrow under
`data/` here:

- `flu_data_hhs` — finalized wILI% series, 11 locations, weekly.
- `flu_data_hhs_versions` — every reported version with an `as_of` date; the
  raw material for the backfill model.
- `flu_data_hhs_tscv_season1..5` — expanding-window cross-validation splits with
  vintage values per forecast date, one per season.
- `copycat_db` — analogue library (available if needed).

The scoring truth is the hub's oracle output (each season's settled value),
taken from the hub clone, not from finalized cross-season revisions.

## Scoring

`ScoringRules.jl` (EpiAware org) provides `interval_score` and `quantile_score`.
WIS is the weighted mean of interval scores across the central intervals plus
the median term. The helper computes WIS on the natural scale (primary) and on
the log scale (report-only), and summarises mean and standard deviation across
origin dates and locations.

## Inference

- Turing.jl with Mooncake as the reverse-mode AD backend for all models.
- Pathfinder for a fast first pass over candidates; full MCMC for finalists.
- Turing callbacks stream fit progress for live monitoring.
- FlexiChains is the chains backend.
  Turing returns FlexiChains `VNChain` objects by default here, and the code
  works with that type directly rather than converting to MCMCChains.
- The base model fits fine via Pathfinder + Mooncake.
  An earlier apparent segfault in `test_model.jl` was traced to box
  OOM/congestion (99% swap, 15+ Julia processes), not a model or AD bug;
  no AD rewrite was needed.

## Testing

`EpiAwarePackageTools.jl` (EpiAware org) provides the modular test-suite
scaffolding used across `test/`.

## Workflow

- Parallel subagents run implement-and-review loops, at least 10 rounds, with
  5 to 10 subagents running per round. Each round runs several implementers
  (lower-power models are fine) proposing competing implementations, and a
  reviewer selects the preferred one. Changes merge to main by pull request or
  directly, depending on how the agents are coordinated.
- A few core jobs always run regardless of shared-box load; more work spins out
  when there is headroom.
- Every checkpoint commits and pushes to `seabbs-bot`, updates the README
  status, and (for search loops) writes a report under `reports/`.
