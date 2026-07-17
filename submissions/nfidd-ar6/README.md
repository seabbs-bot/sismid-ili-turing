# nfidd-ar6

Simple, fast baseline submitted to the hub alongside the slower
hierarchical Turing model from the same team.

## Model

Independent AR(6) per location, fit by ordinary least squares on the
fourth-root-transformed vintage series available at each forecast
origin.
No hierarchy, no seasonality term, no backfill model: deliberately
simple.
Note this is plainer than the AR(6)+Fourier design first discussed;
a Fourier week-of-season term was dropped to keep the first submission
fast and simple, and was not added back in afterwards.

Probabilistic forecasts come from simulating 1000 Gaussian-innovation
sample paths forward per location and horizon using the fitted
residual standard deviation, taking the hub's 23 quantile levels, and
back-transforming to the natural wILI percentage scale, clamped at 0.

The whole submission (both validation seasons, all 11 locations, all
4 horizons) builds in a few seconds: no MCMC, no `SismidILITuring`
Turing/Mooncake dependency, only `CSV`, `DataFrames`, `Dates`,
`Statistics`, `Random`, `LinearAlgebra`.

## Usage

```
julia --project=<sismid-ili-turing repo> generate_forecasts.jl <hub_path>
```

Reads `data/flu_data_hhs_tscv_season{1,2}.csv` from the package's own
`data/` directory (via `src/data.jl`'s `training_splits`), fits and
forecasts every split, and writes
`<hub_path>/model-output/nfidd-ar6/<origin_date>-nfidd-ar6.csv` plus
`<hub_path>/model-metadata/nfidd-ar6.yml` when `hub_path` is given.

## Coverage

- 59 forecast origins (validation seasons 2015/16 and 2016/17; the
  three held-out test seasons are not included).
- All 11 hub locations, all 4 horizons, all 23 quantile levels.
- 2596 scored tasks (59 origins x 11 locations x 4 horizons).

## Performance

Scored against `target-data/oracle-output.csv` in the hub clone with
`src/scoring.jl`'s `score_forecasts`/`wis_summary`: mean WIS 0.368
(sd 0.471) on the natural wILI-percentage scale, mean WIS 0.106
(sd 0.103) on the report-only log1p scale, across all 2596 tasks.

## Submission

PR: https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/62
Branch: `submit-nfidd-ar6` in the `seabbs-bot` fork.
