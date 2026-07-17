# 06. Regional heterogeneity at a glance

Source: `data/flu_data_hhs.csv`.
Computed in Julia (CSV.jl, DataFrames.jl, Statistics, StatsBase).
This report collects three per-location summary statistics already
touched on separately in [[01-series-overview]], [[03-seasonality]]
and [[05-autocorrelation]] into one comparison, so the shape of
regional heterogeneity is visible at a glance rather than only in
separate tables.

**Experimental integrity: filtered to `season_year` 2004-2016** (13
full seasons, `2004/05`-`2016/17`), the full set of complete
training-set seasons, i.e. pre-2015 history plus the two validation
seasons.
The partial first season (`2003`) and the three held-out testing
seasons (`2017/18`-`2019/20`) are excluded throughout.

## Three statistics, three different orderings

- **Off-season baseline**: median wILI over the off-season weeks
  (`woy` 1-8 and 45-52), the same window used for the onset/offset
  heuristic in [[03-seasonality]].
- **Mean seasonal amplitude**: for each season, that season's peak
  wILI minus the location's own baseline, averaged across the 13
  seasons (a simplified version of the amplitude in
  [[03-seasonality]], recomputed here directly against the baseline
  above rather than trough-to-peak within season).
- **Differenced-series volatility**: the SD of the week-to-week
  first difference of `log(wili)`, i.e. how noisy the series is
  once the trend/level is removed (related to, but not the same
  statistic as, the AR-order search in [[05-autocorrelation]]).

| location | baseline (%) | amplitude (pp) | volatility (SD Δlog) |
|---|---|---|---|
| HHS Region 6 | 1.33 | 7.08 | 0.203 |
| HHS Region 9 | 1.26 | 3.89 | 0.213 |
| HHS Region 3 | 1.03 | 4.63 | 0.203 |
| US National | 0.85 | 3.90 | 0.122 |
| HHS Region 2 | 0.85 | 3.78 | 0.234 |
| HHS Region 4 | 0.64 | 4.26 | 0.171 |
| HHS Region 5 | 0.60 | 4.01 | 0.174 |
| HHS Region 1 | 0.40 | 2.96 | 0.267 |
| HHS Region 10 | 0.40 | 4.28 | 0.430 |
| HHS Region 8 | 0.39 | 3.14 | 0.339 |
| HHS Region 7 | 0.28 | 4.73 | 0.413 |

(sorted by baseline, descending)

![Off-season baseline, mean seasonal amplitude and differenced-series
volatility, each shown as an ordered dot chart so the ranking by
location is visible directly; the three rankings are not the same
(validation + history
only)](figures/06_regional_heterogeneity.png)

Baseline spans a >4x range (0.28 to 1.33), consistent with the
level differences already noted in [[01-series-overview]].
Amplitude spans a >2x range and correlates moderately with baseline
(r = 0.54, Region 6 is both the highest-baseline and
highest-amplitude location) but is not simply proportional to it:
Region 7 has the third-lowest baseline yet the second-highest
amplitude, and Region 1 has a below-median baseline but the lowest
amplitude of all 11 locations.
Volatility correlates **negatively** with baseline (r = -0.63,
almost no relationship with amplitude, r = -0.10): the
lowest-baseline locations (Region 7, Region 10, Region 8) are also
the noisiest week to week on the log scale, while US National (the
aggregate, hence smoothed by pooling) is both mid-baseline and by
far the least volatile (0.122, versus 0.20-0.43 elsewhere).
This is the same "small counts amplify proportional noise"
mechanism already noted for backfill revisions in [[02-backfill]],
showing up here in the finalized series itself, not only in the
reporting process.

## Implications for the model

- No single location statistic predicts the others: a location that
  is calm on one axis (e.g. low amplitude) is not necessarily calm
  on another (e.g. Region 7's amplitude is high despite its baseline
  being nearly the lowest).
  Location-level random effects should be allowed to vary somewhat
  independently across level, amplitude and noise scale, rather than
  parameterised as a single "how big is this location" effect that
  drives all three together.
- The negative baseline-volatility correlation supports a
  location-varying observation/process noise scale (already flagged
  in [[01-series-overview]] from the Taylor's-law heterogeneity):
  smaller locations are not just scaled-down versions of larger
  ones, they are proportionally noisier, so a shared noise scale
  fitted mostly to the larger locations (US National, Region 6)
  would likely under-cover the smaller, choppier ones (Region 7,
  Region 8, Region 10).
- US National's low volatility (0.122, roughly half the next-lowest
  region) is an averaging artefact from aggregating the 10 regions,
  not evidence that the national series is intrinsically easier to
  forecast; treat it as its own location rather than a proxy for
  "typical region" noise.
