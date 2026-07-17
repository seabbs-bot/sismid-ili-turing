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

## Transform comparison: variance stabilisation (primary criterion)

The transform choice is decided on variance-stabilisation grounds,
not skewness alone, since skewness only tells us the marginal shape
is asymmetric, not whether the noise scale tracks the mean the way
each candidate transform assumes.
`log` variance-stabilises only if `SD(wILI) ∝ mean(wILI)^1`
(a lognormal-like law); `sqrt` if `SD ∝ mean^0.5` (Poisson-like);
`fourth-root` sits in between.
We estimate the actual mean-variance law empirically (Taylor's power
law) and via Box-Cox, both on validation-period data only.

**Method.** For each (location, `woy`) cell, we have up to 13
independent observations (one per season, 2004-2016).
We compute the local mean and local variance of wILI within each
cell (n >= 8 seasons required) and regress `log(variance)` on
`log(mean)`: the slope `lambda` implies `variance ∝ mean^lambda`,
and the variance-stabilising power transform is `p = 1 - lambda/2`
(`p=0` is log, `p=0.25` is fourth-root, `p=0.5` is sqrt, `p=1` is no
transform).
Separately, we fit a Box-Cox `lambda` by direct maximum likelihood
on the positive wILI values (excluding the 10 exact zeros).

**Results.**

| method | pooled estimate | implied power `p` | per-location range |
|---|---|---|---|
| Taylor's power law (`lambda`) | 1.77 (R² = 0.55) | 0.12 | 0.91 (Region 9) to 3.17 (Region 5) |
| Box-Cox `lambda` (MLE) | 0.13 | 0.13 | -0.40 (Region 5) to 0.29 (Region 9) |

Both methods agree closely: the pooled fitted power is ~0.12-0.13,
between `log` (`p=0`) and fourth-root (`p=0.25`), notably closer to
`log`.
So the textbook "Poisson-like, use sqrt" story is not quite right
for this data — the pooled mean-variance law (`lambda` ~1.8) is
closer to lognormal (`lambda=2`) than Poisson (`lambda=1`) — but
there is real per-location heterogeneity: Region 9 alone is close to
genuinely Poisson-like (`lambda=0.91`, `p=0.55`), while Region 5 and
Region 6 are more extreme than lognormal (`lambda` 2.8-3.2,
`p` negative).

**Direct flatness comparison.** We also transformed the raw values
under each candidate and re-measured local-cell SD against the raw
local mean (same cells as above), reporting the regression slope of
`log(local SD)` on `log(raw local mean)` (0 = perfectly flat) and the
ratio of mean local SD between the highest and lowest mean quintile
(1 = perfectly flat):

| transform | slope | quintile max/min ratio |
|---|---|---|
| fitted Box-Cox power (`p=0.13`) | -0.02 | 1.11 |
| fourth-root (`p=0.25`) | 0.11 | 1.18 |
| logit (on wili/100) | -0.16 | 1.48 |
| `log` (`to_scale(w, :log)`, EPS floor) | -0.18 | 1.52 |
| `log1p` | 0.35 | 1.77 |
| sqrt (`p=0.5`, not in `TRANSFORMS`) | 0.38 | 1.93 |
| identity (no transform) | 0.88 | 4.51 |

**Fourth-root is the flattest of the transforms currently in
`src/core.jl`'s `TRANSFORMS`**, clearly better than `log` (slope
magnitude 0.11 vs 0.18, quintile ratio 1.18 vs 1.52) and much better
than `log1p` (0.35, ratio 1.77 — `log1p`'s `+1` shift matters at
these small percentage-scale values and pulls its behaviour toward
`identity`/`sqrt`, which is why it under-stabilises here despite
cutting skewness well in the section below).
`logit` performs similarly to `log` (both roughly comparable, both
clearly worse than fourth-root) for the same reason noted before:
in this data it never approaches its 100% bound, so it behaves like
a log transform rather than exploiting genuine boundedness.
The best-fitting continuous power (`p~0.13`) would do slightly
better still, but that transform does not exist in `TRANSFORMS`
(only `log`, `log1p`, `logit`, `fourthroot` are implemented) and is
close enough to `log` that it is not worth adding a fifth transform
for.
`sqrt` is *not* in `TRANSFORMS` either, and this data does not
support adding it: it is empirically worse than fourth-root and
even slightly worse than `log1p` here, because the pooled
mean-variance law is closer to lognormal than Poisson.

**Recommendation: use fourth-root (`to_scale(w, :fourthroot)`) as
the primary modelling scale, not `log`/`log1p`.**
This matches the original fable session's choice and Sam's prior
that `log` over-transforms wILI's mean-variance relationship.
Given the substantial per-location heterogeneity in the fitted power
(Region 9 wants something closer to sqrt; Region 5/Region 6 want
something more aggressive than log), no single global transform is
exactly right everywhere; if the model architecture can support a
location-varying observation-noise scale (e.g. a per-location
dispersion/scale parameter on top of a shared fourth-root mean
transform) that is more principled than searching for one perfect
global power.

## Transform comparison: skewness (secondary evidence)

Skewness of the raw series and three candidate transforms
(min-to-max across the 11 locations, validation-period data):

| transform | skewness range | notes |
|---|---|---|
| raw (%) | 1.51 to 3.45 | strongly right-skewed everywhere |
| `log1p` | 0.26 to 1.26 | skew cut roughly in half to two-thirds, still positive |
| logit (on wili/100) | -0.57 to 0.61 | closest to symmetric, small in magnitude |
| fourth-root | 0.02 to 1.04 | between log1p and logit |

Logit gives the most symmetric residual distribution location by
location, but as noted above it is not exploiting genuine
boundedness in this data, and it is not the flattest on
variance-stabilisation grounds either.
`log1p` cuts skew a lot but is the *worst* of the implemented
transforms on variance-stabilisation grounds (see above), so
skewness and variance-stabilisation point in different directions
here — we prioritise variance-stabilisation per Sam's steer, since
that is what the model's observation-noise assumption actually
relies on, and treat any residual skew after fourth-root as
something the AR/seasonal noise distribution should absorb (e.g. a
Student-t observation noise) rather than something to fix by
choosing a more skew-corrective transform.

## Implications for the model

- Location-specific intercepts/scales are essential: raw ranges
  span roughly 3x between the lowest and highest locations.
- **Use fourth-root (`to_scale(w, :fourthroot)`) as the primary
  modelling scale**, chosen on variance-stabilisation grounds (see
  above): it is the flattest of the four implemented transforms and
  matches Sam's prior and the original fable session.
  It already handles the 10 exact zeros (Region 7, Region 10)
  without a special case (`max(w, 0.0)^0.25`).
- `sqrt` is not implemented in `TRANSFORMS` and this analysis does
  not support adding it — it is empirically worse than fourth-root
  here.
- Given real per-location heterogeneity in the fitted power
  (0.91-3.17 on the Taylor's-law `lambda` scale), consider whether
  the model can support a location-varying observation-noise scale
  on top of the shared fourth-root transform, rather than relying on
  one global transform to be exactly right everywhere.
- Define `woy` (week-of-season) using a season boundary at
  late July (day-of-year ≈ 205), consistent with the tscv season
  files; treat 52- vs 53-week seasons and the partial first season
  explicitly in `build_model_data`.
- Any further EDA extension must keep filtering to
  `season_year <= 2016` (or equivalently the `season1`/`season2`
  tscv files) — do not load `season3-5` or dates past 2017-07-22.
