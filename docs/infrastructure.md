# Infrastructure

Tooling, data, scoring, and workflow.
Filled in as each piece lands, so this document tracks what actually exists.

## Status of pieces

| Piece | State | Notes |
|---|---|---|
| Repo on `seabbs-bot` | done | `seabbs-bot/sismid-ili-turing`, local at `~/code/seabbs/sismid-ili-turing` |
| Target hub clone | done | `~/code/external/sismid-ili-forecasting-sandbox` |
| Docs | done | brief, plan, infrastructure, reports index and template |
| Julia project | todo | Turing, Mooncake, Pathfinder, ScoringRules, Arrow, DataFrames |
| Data export to Arrow | todo | from the course R package `.rda` objects |
| WIS scoring helper | todo | natural and log scale, on `ScoringRules.jl` |
| Base model | todo | Phase 1 |
| Forecast + hubverse writer | todo | Phase 1 |

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

## Workflow

- Parallel subagents run implement-and-review loops. Changes merge to main by
  pull request or directly, depending on how the agents are coordinated.
- A few core jobs always run regardless of shared-box load; more work spins out
  when there is headroom.
- Every checkpoint commits and pushes to `seabbs-bot`, updates the README
  status, and (for search loops) writes a report under `reports/`.
