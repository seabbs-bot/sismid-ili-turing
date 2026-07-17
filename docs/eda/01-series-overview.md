# 01. Series overview

Source: `data/flu_data_hhs.csv` (finalized series).
Computed in Julia (CSV.jl, DataFrames.jl, Statistics, StatsBase).

**Experimental integrity: this and every report in `docs/eda/` uses
only pre-2015 history plus the two validation seasons (`2015/16`,
`2016/17`).**
The three testing seasons (`2017/18`-`2019/20`) are excluded from
every summary below, per `docs/contracts.md`: they are held out for
final evaluation and must not inform model design.
If you extend this analysis, filter to `season_year <= 2016` first
(see week-of-season definition below).

## Coverage

- 11 locations: `US National` and `HHS Region 1`..`HHS Region 10`.
- Weekly reference dates (Saturdays).
- After the validation cutoff: 2003-08-30 to 2017-07-22, 726 rows
  per location, 7,986 rows total.
- No missing values in the finalized series (0 everywhere) and
  every location has exactly the same 726 dates in this window.
- Full history (including the held-out test seasons, not analysed
  here) runs to 2020-08-29; do not load past 2017-07-22 for any
  design decision.

## Week-of-season definition

The mean wILI curve pooled across all years and locations has its
minimum around late July (confirmed on the pre-cutoff data: national
minima each year fall between early July and late August, clustering
around day-of-year ~205-220).
We define the season boundary at day-of-year 205 (24 July), i.e.
a season runs from late July of year `Y` to late July of year `Y+1`
and is labelled by its start year `Y` (`season_year`).
Week-of-season `woy` is the rank of `origin_date` within that season,
starting at 1.
Seasons run 52 weeks in most years and 53 weeks when a season
straddles an extra ISO week (`2004`, `2010`, `2016` are 53-week
seasons in the pre-cutoff data; all others are 52 weeks).
The first season in the file (`2003`, 47 weeks) is partial and
should be dropped or handled carefully in any season-indexed model.

This `woy`/`season_year` definition matches the tscv split files:
`season1` in `flu_data_hhs_tscv_season1.csv` runs to a forecast
origin of 2016-05-07 (`season_year` 2015, i.e. `2015/16`), and
`season2`..`season5` correspond to `season_year` 2016..2019
(`2016/17`..`2019/20`).
So `season1,2` = the validation seasons analysed here, and
`season3-5` = the held-out testing seasons excluded from all EDA.

## Scale (wILI, percentage points), validation-period data only

| location | min | median | mean | max | n zero |
|---|---|---|---|---|---|
| US National | 0.35 | 1.35 | 1.78 | 7.72 | 0 |
| HHS Region 1 | 0.05 | 0.70 | 0.99 | 9.69 | 0 |
| HHS Region 2 | 0.14 | 1.49 | 1.85 | 8.12 | 0 |
| HHS Region 3 | 0.12 | 1.54 | 1.95 | 10.53 | 0 |
| HHS Region 4 | 0.23 | 1.18 | 1.57 | 7.52 | 0 |
| HHS Region 5 | 0.15 | 1.04 | 1.43 | 9.85 | 0 |
| HHS Region 6 | 0.33 | 2.10 | 2.81 | 13.38 | 0 |
| HHS Region 7 | 0.00 | 0.79 | 1.28 | 11.45 | 5 |
| HHS Region 8 | 0.01 | 0.77 | 1.05 | 9.75 | 0 |
| HHS Region 9 | 0.17 | 1.86 | 2.12 | 9.11 | 0 |
| HHS Region 10 | 0.00 | 0.89 | 1.33 | 9.23 | 5 |

Scales differ a lot by location: Region 6 runs 2-3x higher than
Region 1 or Region 8 on both median and max, so per-location
intercepts (not a single shared level) are needed regardless of
transform.
Ten rows in this window are exactly 0.0, all in Region 7 and
Region 10 (5 each), always in summer troughs.
Any log-family transform needs an offset or a `log1p`-style
formulation to handle these exact zeros.

## Transform comparison

Skewness of the raw series and three candidate transforms
(min-to-max across the 11 locations, validation-period data):

| transform | skewness range | notes |
|---|---|---|
| raw (%) | 1.51 to 3.45 | strongly right-skewed everywhere |
| `log1p` | 0.26 to 1.26 | skew cut roughly in half to two-thirds, still positive |
| logit (on wili/100) | -0.57 to 0.61 | closest to symmetric, small in magnitude |
| fourth-root | 0.02 to 1.04 | between log1p and logit |

Logit gives the most symmetric residual distribution location by
location, but wILI is a percentage that in this window never
exceeds ~13.4%, so the logit transform is operating far from its
natural bound at 100% for the whole range: it behaves close to a
log transform in the observed regime and mostly earns its
"most symmetric" ranking by correcting the heavy right tail rather
than by using genuine boundedness information.
`log1p` (equivalently `log` with a small additive offset,
`to_scale(w, :log)` with an epsilon) is the pragmatic default: it
handles the exact zeros without a separate boundary process, is
simple to invert, and variance-stabilises well even though some
right skew remains.
Fourth-root is a reasonable middle-ground alternative if `log1p`
under-corrects skew for the highest-variance locations (Region 5,
Region 6, both above 1.0 on both log1p and fourth-root).

## Implications for the model

- Location-specific intercepts/scales are essential: raw ranges
  span roughly 3x between the lowest and highest locations.
- Use `log1p`-style log (`to_scale` with a small offset) as the
  primary modelling scale; it is simple, invertible, and handles
  the 10 exact zeros (Region 7, Region 10) without a special case.
  Keep logit or fourth-root as a fallback if diagnostics on
  Region 5/6 show residual skew.
- Define `woy` (week-of-season) using a season boundary at
  late July (day-of-year ≈ 205), consistent with the tscv season
  files; treat 52- vs 53-week seasons and the partial first season
  explicitly in `build_model_data`.
- Any further EDA extension must keep filtering to
  `season_year <= 2016` (or equivalently the `season1`/`season2`
  tscv files) — do not load `season3-5` or dates past 2017-07-22.
