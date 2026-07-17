# nfidd-ar6bf

Iterates on `nfidd-ar6` (`../nfidd-ar6/`, merged, PR #62) by adding a
backfill correction, as the first improvement step on the simple
baseline.

## Model

Same independent per-location AR(6) as `nfidd-ar6`, fit by OLS on the
fourth-root-transformed vintage series, no hierarchy, no seasonality
term.
`nfidd-ar6` itself is the plain design: an earlier Fourier variant was
tried and dropped before submission (see `../nfidd-ar6/README.md`), so
this iterates on the merged plain-AR(6) model, not the discarded
Fourier one.

Added: a **backfill correction** step, applied to the vintage series
before fitting. The most recent weeks at each forecast origin are only
partially reported and tend to be revised, so they are nudged towards
their expected settled value using an empirical location x delay
revision profile estimated from `data/flu_data_hhs_versions.csv`.

For each `(location, delay)` with `delay = weeks(as_of - origin_date)`,
the correction is the median of `fourthroot(settled) -
fourthroot(vintage)` across matching `(location, origin_date)` pairs,
where `settled` is the value at that pair's largest tracked `as_of`
(the settled-value proxy used throughout
[docs/eda/02-backfill.md](../../docs/eda/02-backfill.md)). This is
deliberately not a monotonic reporting-CDF completion factor:
docs/eda/02-backfill.md shows revisions change sign across both delay
and location (e.g. HHS Region 2 revises up ~84% of the time at delay
1, HHS Region 9 revises down ~61% of the time), so a location-varying
empirical profile is used instead of a single shared curve. The
profile is estimated only from training-set origin dates (pre-2015
history plus the two validation seasons); no test-season data is used
anywhere, and applying it only ever looks at `as_of <= forecast_origin`
(enforced inside `build_model_data`), so no future revision leaks into
a split's own correction.

Everything else (AR order, transform, path simulation, quantile
levels, seed) is identical to `nfidd-ar6`, isolating the effect of the
backfill correction in the comparison below.

## Usage

```
julia --project=<sismid-ili-turing repo> generate_forecasts.jl <hub_path>
```

Reads `data/flu_data_hhs_tscv_season{1,2}.csv` and
`data/flu_data_hhs_versions.csv` from the package's own `data/`
directory, fits and forecasts every split, and writes
`<hub_path>/model-output/nfidd-ar6bf/<origin_date>-nfidd-ar6bf.csv`
plus `<hub_path>/model-metadata/nfidd-ar6bf.yml` when `hub_path` is
given.

## Coverage

- 59 forecast origins (validation seasons 2015/16 and 2016/17; the
  three held-out test seasons are not included).
- All 11 hub locations, all 4 horizons, all 23 quantile levels.
- 2596 scored tasks (59 origins x 11 locations x 4 horizons).
- Validated locally with `scripts/validate_submission.jl`: PASS.

## Performance

Scored against `target-data/oracle-output.csv` in the hub clone with
`src/scoring.jl`'s `score_forecasts`/`wis_summary`, all 2596 tasks:

| Model | Mean WIS (natural) | SD | Mean WIS (log1p, report-only) | SD |
|---|---|---|---|---|
| nfidd-ar6 | 0.368 | 0.471 | 0.106 | 0.103 |
| nfidd-ar6bf | 0.359 | 0.452 | 0.103 | 0.098 |

The backfill correction **improves** mean WIS by about 2.5% on both
scales, and also reduces the SD (a more stable improvement, not just
a mean shift from a few tasks).

Breaking the improvement down:

- **By location**: concentrated in the two locations with the
  largest, most consistently-signed revision profiles (see
  docs/eda/02-backfill.md): HHS Region 2 (mean per-task WIS improves
  by 0.057) and HHS Region 9 (0.026). Smaller improvements in most
  other locations (Region 7: -0.010, Region 3: -0.005, Region 6/10/1/5
  and US National: all around -0.0005 to -0.002). Essentially flat in
  HHS Region 8 (+0.00007) and a small regression in HHS Region 4
  (+0.0011).
- **By season**: most of the gain is in 2015/16 (-0.017 mean per-task
  WIS) versus 2016/17 (-0.001, near flat).
- **By horizon**: small, fairly even improvement across all four
  horizons (h=1: -0.014, h=2: -0.010, h=3: -0.008, h=4: -0.005 mean
  WIS).
- **Task-level**: 1373 of 2596 (53%) individual forecast tasks
  improved; the correction is not universally better, but net
  positive.

Caveat: only two seasons have tracked revision history in
`flu_data_hhs_versions.csv` (docs/eda/02-backfill.md), so the
correction profile is estimated from the same two validation seasons
it is scored on here — there is no other historical revision data to
hold out from profile estimation. Read this as "does correcting
towards a known revision profile help", not as an unbiased estimate of
out-of-sample gain. The held-out test seasons are untouched either
way.

## Submission

PR: https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/67
-- CI (`validate-submission`) green.
Branch: `submit-nfidd-ar6bf` in the `seabbs-bot` fork.
