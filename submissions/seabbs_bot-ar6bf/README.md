# seabbs_bot-ar6bf

Iterates on `nfidd-ar6` (`../nfidd-ar6/`, merged, PR #62) by adding a
backfill correction, as the first improvement step on the simple
baseline.

Submitted as `seabbs_bot-ar6bf` (`team_abbr = "seabbs_bot"`,
`model_abbr = "ar6bf"`): the hub's model-metadata schema forbids a
hyphen in `team_abbr` (`^[a-zA-Z0-9_+]+$`, underscore only), so this
cannot reuse `nfidd-ar6`'s `team_abbr`.

An earlier, validation-seasons-only version of this same model was
briefly submitted as `nfidd-ar6bf`
(PR [#67](https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/67),
merged) before the naming and full-coverage requirements below were
set; that PR predates this one and is superseded by it, though it
could not be closed after the fact since it had already been merged.

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

Reads `data/flu_data_hhs_tscv_season{1,2,3,4,5}.csv` and
`data/flu_data_hhs_versions.csv` from the package's own `data/`
directory, fits and forecasts every split of all five seasons (a
per-origin vintage fit, so covering the test seasons at generation
time never trains on or tunes against them), and writes
`<hub_path>/model-output/seabbs_bot-ar6bf/<origin_date>-seabbs_bot-ar6bf.csv`
plus `<hub_path>/model-metadata/seabbs_bot-ar6bf.yml` when `hub_path`
is given.

## Coverage

- All 5 seasons: validation (2015/16, 2016/17) and the three held-out
  test seasons (2017/18-2019/20), all 11 hub locations, all 4
  horizons, all 23 quantile levels.
- 140 of the hub's 142 declared valid `origin_date` values (all
  `optional`, none `required`, per `hub-config/tasks.json`), 61,600
  rows total (140 x 11 x 4 x 23).
- Two known, unavoidable gaps against the hub's full round list, both
  a property of this repo's own tscv data, not a modelling choice:
  - `2018-10-13` and `2019-10-12` (the first origin date of seasons
    4 and 5 respectively) are not present in
    `flu_data_hhs_tscv_season{4,5}.csv` at all -- each season's
    expanding-window split file starts one week later than the hub's
    declared round list, presumably because the first split would
    have too little lag history to be usable.
  - Conversely, `flu_data_hhs_tscv_season5.csv` itself runs six weeks
    *past* the hub's declared round list (`2020-03-28` through
    `2020-05-02`); those splits were generated but then dropped before
    submission since `hub-config/tasks.json` does not list them as
    valid `origin_date` values at all (real-world 2019/20 reporting
    was disrupted early by COVID-19, most likely the reason the hub's
    round list ends where it does).
  - `scripts/validate_submission.jl` flags both of these explicitly as
    "data/hub-config drift", i.e. a mismatch between this repo's own
    source data and the hub's declared task list, not a submission
    defect; since none of the hub's origin dates are `required`, a
    140/142 submission is valid.
- Validated locally with `scripts/validate_submission.jl`
  (`SEASONS=1,2,3,4,5`): the two coverage warnings above, no other
  problems.

## Performance

The backfill-vs-baseline selection decision is made on the
**validation seasons only** (2015/16, 2016/17, 2596 tasks), per
docs/contracts.md experimental integrity; the additional test-season
coverage above is generated for the hub submission only and is not
used in this comparison.

Scored against `target-data/oracle-output.csv` in the hub clone with
`src/scoring.jl`'s `score_forecasts`/`wis_summary`:

| Model | Mean WIS (natural) | SD | Mean WIS (log1p, report-only) | SD |
|---|---|---|---|---|
| nfidd-ar6 | 0.368 | 0.471 | 0.106 | 0.103 |
| seabbs_bot-ar6bf | 0.359 | 0.452 | 0.103 | 0.098 |

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
it is scored on here -- there is no other historical revision data to
hold out from profile estimation. Read this as "does correcting
towards a known revision profile help", not as an unbiased estimate of
out-of-sample gain. The held-out test seasons are untouched by the
profile estimate either way, and this comparison never uses the test
seasons.

## Submission

PR: https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/74
Branch: `submit-seabbs_bot-ar6bf` in the `seabbs-bot` fork.

Supersedes the earlier `nfidd-ar6bf`
(PR [#67](https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/67),
merged, validation-seasons-only, wrong `team_abbr` naming for this
project's convention).
