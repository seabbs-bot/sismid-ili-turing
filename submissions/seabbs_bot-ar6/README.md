# seabbs_bot-ar6

Correctly-named, full-5-season resubmission of the nfidd-ar6 baseline.

## Why a rename

The hub's `model-metadata` schema requires `team_abbr` to match
`^[a-zA-Z0-9_+]+$`.
A hyphen is invalid there, so `team_abbr: "seabbs-bot"` cannot be
used.
`seabbs_bot` (underscore) is valid.
`model_id` is `team_abbr-model_abbr`, so this becomes
`seabbs_bot-ar6` -- the hyphen there is the `model_id`'s own
separator, not part of either abbreviation.

`nfidd-ar6` (PR #62, merged, validation seasons only; PR #70,
test-season follow-up) stays in the hub as a historical entry.
`seabbs_bot-ar6` is the canonical, correctly-named, full-coverage
baseline going forward.

## Model

Independent AR(6) per location, fit by ordinary least squares on the
fourth-root-transformed vintage series available at each forecast
origin.
No hierarchy, no seasonality term, no backfill model: identical
model and code to `nfidd-ar6` (see `../nfidd-ar6/generate_forecasts.jl`).

Probabilistic forecasts come from simulating 1000 Gaussian-innovation
sample paths forward per location and horizon using the fitted
residual standard deviation, taking the hub's 23 quantile levels, and
back-transforming to the natural wILI percentage scale, clamped at 0.

Builds all 5 seasons (both validation and test) in a few seconds: no
MCMC, no `SismidILITuring` Turing/Mooncake dependency, only `CSV`,
`DataFrames`, `Dates`, `Statistics`, `Random`, `LinearAlgebra`, `JSON`
(the last only to read the hub's own round list -- see Coverage
below).

## Usage

```
julia --project=scripts/validate generate_forecasts.jl <hub_path>
```

(run from the `sismid-ili-turing` repo root; `scripts/validate`'s
environment already has `JSON` alongside `CSV`/`DataFrames`, and
`Statistics`/`Random`/`LinearAlgebra` are stdlib).

Reads `data/flu_data_hhs_tscv_season{1..5}.csv` from the package's own
`data/` directory (via `src/data.jl`'s `training_splits`, with
`allow_test_season=true` for seasons 3-5), fits and forecasts every
split, filters out any origin dates the hub's own
`hub-config/tasks.json` doesn't recognise, and writes
`<hub_path>/model-output/seabbs_bot-ar6/<origin_date>-seabbs_bot-ar6.csv`
plus `<hub_path>/model-metadata/seabbs_bot-ar6.yml`.

## Coverage

- 146 forecast origins exist across all 5 tscv seasons; 140 are
  submitted.
- 6 origin dates from season 5's (2019/20) splits (2020-03-28 through
  2020-05-02) are excluded: they postdate the hub's `tasks.json`
  round list (max 2020-03-21), most likely because 2019/20 season
  ILI surveillance was disrupted by the onset of COVID-19.
  `generate_forecasts.jl` detects and drops these automatically
  (`hub_round_dates`), rather than hardcoding the exclusion.
- All 11 hub locations, all 4 horizons, all 23 quantile levels for
  every submitted origin.
- 6160 scored tasks (140 origins x 11 locations x 4 horizons).

## Performance

Scored against `target-data/oracle-output.csv` in the hub clone with
`src/scoring.jl`'s `score_forecasts`/`wis_summary`, all 6160 tasks
(all 5 seasons, validation + test): mean WIS 0.515 (sd 0.719) on the
natural wILI-percentage scale, mean WIS 0.117 (sd 0.122) on the
report-only log1p scale.
For comparison, `nfidd-ar6`'s validation-seasons-only score (2596
tasks) was 0.368 natural / 0.106 log1p; the higher full-coverage mean
here reflects the test seasons' harder-to-forecast periods being
included, not a change in the model.

## Submission

Branch: `submit-seabbs_bot-ar6` in the `seabbs-bot` fork
(`sismid-ili-sandbox-fork-seabbsbot-ar6` clone).
PR: see repo root or ask -- filled in once opened.
