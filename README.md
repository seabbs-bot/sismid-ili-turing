# sismid-ili-turing

Hierarchical Bayesian forecasting of weighted influenza-like illness (wILI%)
for the SISMID reichlab sandbox hub, built in Julia with Turing.jl.

The model partially pools information across the 11 US/HHS locations, learns
seasonality as a random-effect structure that varies by location and season,
adds autoregressive dynamics, and jointly models reporting backfill.
Candidate formulations are searched, scored by the weighted interval score
(WIS), and the best are submitted to the online hub.

## Goal

Build a custom model, validate and select it following the two SISMID local-hub
sessions, and submit the finalist(s) to
[`reichlab/sismid-ili-forecasting-sandbox`](https://github.com/reichlab/sismid-ili-forecasting-sandbox)
as a hubverse pull request so the forecasts appear on the
[online dashboard](https://reichlab.io/sismid-ili-forecasting-dashboard/forecast.html).

- **Validation seasons** (fit and search here): 2015/16, 2016/17.
- **Testing seasons** (measure selected models here): 2017/18, 2018/19, 2019/20.
- **Target**: quantile forecasts of wILI%, 1–4 weeks ahead, 11 locations,
  23 quantiles per task.

## Documentation

Read these in order.

1. [`docs/brief.md`](docs/brief.md) — the requirements from Sam (what to build
   and how to work). This is the authoritative brief.
2. [`docs/plan.md`](docs/plan.md) — the model design, search and selection
   strategy, and the phased plan.
3. [`docs/infrastructure.md`](docs/infrastructure.md) — tooling, data, scoring,
   and the parallel-agent workflow, filled in as each piece lands.
4. [`docs/steer-log.md`](docs/steer-log.md) — a running record of Sam's
   guidance and the action taken in response.
5. [`docs/eda/`](docs/eda/) — exploratory analysis, revisited across search
   rounds rather than fixed once, with committed plots under
   `docs/eda/figures/`.
6. [`reports/`](reports/) — one report per iteration loop, plus a running index.

A Documenter site (see `docs/infrastructure.md`) will render the Julia API
docs and link all of the above in one nav.

## Status

Phase 0 (setup) and Phase 1 (machinery) are done.
The base model fits via Pathfinder and Mooncake, and the full
fit -> forecast -> score pipeline has run end to end on real data
(a season-1 validation split, all 11 locations), producing a valid
11x4x23 hub table with mean WIS ~0.29.
Phase 2 (tree search) is in progress: Round 1 runs five variants
(`v1-ar-high`, `v2-mvn-season`, `v3-diff`, `v4-tv-ar`, `v5-backfill`)
through a resilient round-runner engine.
See [`docs/plan.md`](docs/plan.md) for the full phase checklist.

## Repository layout

```
src/          front-runner Julia model, forecasting, hubverse I/O, scoring;
              always loadable and ready to run and submit
scripts/      entry points for data export, screening runs, submission
data/         Julia-ready data (Arrow), exported from the course R package
experiments/  candidate model definitions and run configs for the search;
              winners are promoted into src/
reports/      one markdown report per iteration loop (+ index and template)
docs/         brief, plan, infrastructure, steer-log, eda
test/         package tests
```
