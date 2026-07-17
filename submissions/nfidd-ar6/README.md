# nfidd-ar6

Simple, fast baseline submitted to the hub alongside the slower
hierarchical Turing model from the same team.

## Model

Two variants were built; see [Performance](#performance) for why the
plain AR(6) is the one actually merged.

`generate_forecasts.jl` (**merged in PR #62, live in the hub**):
independent AR(6) per location, fit by ordinary least squares on the
fourth-root-transformed vintage series available at each forecast
origin.
No hierarchy, no seasonality term, no backfill model: deliberately
simple.

`generate_forecasts_fourier.jl` (**built, validated, scored -- not
submitted**): the same AR(6) design with 3 added Fourier harmonic
pairs (sin/cos) of week-of-season, 52-week period, per location.
Scores worse than the plain version (see below), so it was not pushed
to the hub.

Both: probabilistic forecasts come from simulating 1000
Gaussian-innovation sample paths forward per location and horizon
using the fitted residual standard deviation, taking the hub's 23
quantile levels, and back-transforming to the natural wILI percentage
scale, clamped at 0.
Each builds the whole submission (both validation seasons, all 11
locations, all 4 horizons) in a few seconds: no MCMC, no
`SismidILITuring` Turing/Mooncake dependency, only `CSV`,
`DataFrames`, `Dates`, `Statistics`, `Random`, `LinearAlgebra`.

## Usage

```
julia --project=<sismid-ili-turing repo> generate_forecasts.jl <hub_path>
julia --project=<sismid-ili-turing repo> generate_forecasts_fourier.jl <hub_path>
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
`src/scoring.jl`'s `score_forecasts`/`wis_summary`, all 2596 tasks:

| Variant | Mean WIS (natural) | SD | Mean WIS (log1p, report-only) | SD |
|---|---|---|---|---|
| plain AR(6) -- merged | 0.368 | 0.471 | 0.106 | 0.103 |
| AR(6) + Fourier(3) -- not submitted | 0.412 | 0.521 | 0.118 | 0.122 |

Adding the Fourier seasonality term made the score worse on both
scales (about +12% natural, +11% log), likely because the added
seasonal parameters overfit given the `window_weeks=104` training
history and AR(6) already captures most of the short-term structure.
The plain AR(6) is the version actually merged into the hub.

## Submission

PR: https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/62
(plain AR(6) only) -- **MERGED** into `reichlab/sismid-ili-forecasting-sandbox`
main, `validate-submission` CI green.
Branch: `submit-nfidd-ar6` in the `seabbs-bot` fork.
The AR(6)+Fourier variant was never pushed or submitted.
