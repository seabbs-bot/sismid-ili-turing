# 03. Seasonality

Source: `data/flu_data_hhs.csv`, with the `woy` (week-of-season)
definition from [[01-series-overview]] (season boundary at
day-of-year ≈ 205, late July).
Computed in Julia (CSV.jl, DataFrames.jl, Statistics, StatsBase).

**Experimental integrity: filtered to `season_year` 2004-2016**
(13 full seasons: `2004/05`..`2016/17`), the full set of complete
training-set seasons, i.e. pre-2015 history plus the two validation
seasons.
The partial first season (`2003`, missing its first ~5 weeks of
data) and the three held-out testing seasons (`2017/18`-`2019/20`)
are excluded throughout.

This report leads with **peak timing, peak height, and season
onset/offset**, per location and per season, as cross-season
features for the search agents to use directly (e.g. as informative
priors on a season-level random effect, or as sanity checks on
posterior seasonal-curve draws).
Treat every number here as a living estimate from the full 13-season
training set: re-check it as the search progresses, and do not
hard-code it as ground truth for the held-out test seasons.

## Peak timing: national, by season

| season | peak `woy` | peak wILI |
|---|---|---|
| 2004/05 | 31 | 5.44 |
| 2005/06 | 23 | 3.28 |
| 2006/07 | 30 | 3.58 |
| 2007/08 | 30 | 5.98 |
| 2008/09 | 30 | 3.57 |
| 2009/10 | 14 | 7.72 |
| 2010/11 | 29 | 4.55 |
| 2011/12 | 34 | 2.39 |
| 2012/13 | 23 | 6.06 |
| 2013/14 | 23 | 4.59 |
| 2014/15 | 23 | 5.98 |
| 2015/16 | 34 | 3.56 |
| 2016/17 | 30 | 5.06 |

| statistic | value |
|---|---|
| mean peak `woy` | 27.2 |
| SD of peak `woy` | 5.7 |
| range of peak `woy` | 14 to 34 (roughly early-Dec to mid-Feb) |
| mean peak wILI | 4.75 |
| SD of peak wILI | 1.47 |

A ~6-week SD on a ~52-week season is substantial: the peak is not
fixed to a calendar week (2009/10's pandemic-associated peak at
`woy` 14 is a clear outlier, but even the non-pandemic seasons span
`woy` 23-34, an 11-week range).
A fixed (Fourier-term) seasonal curve would systematically misalign
in a large share of seasons.

## Peak timing and height, by location (across the 13 seasons)

| location | mean peak `woy` | SD peak `woy` | min-max `woy` | mean peak wILI | SD peak wILI |
|---|---|---|---|---|---|
| HHS Region 1 | 28.5 | 5.2 | 16-34 | 3.36 | 2.12 |
| HHS Region 2 | 27.1 | 7.6 | 16-45 | 4.63 | 1.84 |
| HHS Region 3 | 28.9 | 7.9 | 15-48 | 5.67 | 2.35 |
| HHS Region 4 | 25.8 | 6.7 | 8-34 | 4.89 | 1.54 |
| HHS Region 5 | 28.3 | 5.9 | 14-34 | 4.61 | 1.90 |
| HHS Region 6 | 26.9 | 5.8 | 13-34 | 8.42 | 2.53 |
| HHS Region 7 | 28.0 | 5.7 | 14-34 | 5.02 | 2.51 |
| HHS Region 8 | 26.5 | 7.5 | 11-34 | 3.54 | 2.09 |
| HHS Region 9 | 27.5 | 6.5 | 13-41 | 5.15 | 1.25 |
| HHS Region 10 | 28.8 | 6.4 | 13-39 | 4.67 | 2.18 |
| US National | 27.2 | 5.7 | 14-34 | 4.75 | 1.47 |

Per-location peak-timing SD ranges 5.2-7.9 weeks (Region 1 tightest,
Region 3 loosest), all comparably or more variable than the national
aggregate — location-level peak timing is not simply a noisier copy
of the national signal, it has genuine location-specific spread.

## Season onset and offset, by location

Onset/offset defined as the first/last week of a run of >= 2
consecutive weeks above 1.5x the location's own off-season baseline
(median wILI over `woy` 1-8 and 45-52).
This is a simple heuristic, not the CDC epidemic-threshold method;
treat the exact week numbers as indicative rather than precise, and
note it can behave oddly for locations with an already-elevated or
noisy off-season baseline (worth a sanity check against posterior
draws rather than used as a hard prior).

| location | median onset `woy` | SD onset | median offset `woy` | SD offset | median duration (weeks) |
|---|---|---|---|---|---|
| HHS Region 1 | 12 | 3.8 | 41 | 4.4 | 28 |
| HHS Region 2 | 12 | 7.2 | 41 | 6.6 | 33 |
| HHS Region 3 | 15 | 7.0 | 39 | 5.4 | 30 |
| HHS Region 4 | 11 | 4.4 | 42 | 5.6 | 35 |
| HHS Region 5 | 12 | 2.9 | 41 | 3.4 | 29 |
| HHS Region 6 | 14 | 5.6 | 40 | 3.7 | 26 |
| HHS Region 7 | 6 | 5.1 | 44 | 4.1 | 39 |
| HHS Region 8 | 9 | 6.4 | 46 | 4.6 | 36 |
| HHS Region 9 | 17 | 7.7 | 39 | 6.7 | 27 |
| HHS Region 10 | 12 | 6.3 | 45 | 4.8 | 35 |
| US National | 13 | 3.3 | 40 | 4.0 | 27 |

National onset/offset by season (`woy`): 2004/05 17/39, 2005/06
13/39, 2006/07 15/38, 2007/08 13/38, 2008/09 19/52, 2009/10 5/37
(pandemic season, early onset), 2010/11 16/40, 2011/12 13/42,
2012/13 12/38, 2013/14 13/44, 2014/15 12/43, 2015/16 13/44, 2016/17
15/42.
Onset SD (3-8 weeks) is generally tighter than offset SD across
locations, i.e. the season's decline back to baseline is more
variable in timing than its rise, consistent with a longer, more
variable tail after the peak than build-up before it.

## Shape of the curve

Pooled national mean wILI by week-of-season is a single smooth
hump: flat and low from `woy` 1-6 (late Jul-Sep, ~0.74-0.9), rising
from `woy` ~10 through a broad late-autumn/winter climb, peaking
around `woy` 28-31 (roughly mid-late January), then declining back
to trough levels by `woy` ~45-53.
This one-hump-per-season shape holds at every location; no location
shows a secondary within-season peak in the pooled mean curve.

## Amplitude varies by location, less so within a location

| location | mean amplitude (peak-trough) | SD amplitude | CV |
|---|---|---|---|
| HHS Region 6 | 7.52 | 2.46 | 0.33 |
| HHS Region 3 | 5.06 | 2.42 | 0.48 |
| HHS Region 7 | 4.90 | 2.45 | 0.50 |
| HHS Region 4 | 4.47 | 1.51 | 0.34 |
| HHS Region 10 | 4.47 | 2.11 | 0.47 |
| HHS Region 9 | 4.38 | 1.24 | 0.28 |
| HHS Region 5 | 4.17 | 1.91 | 0.46 |
| HHS Region 2 | 4.13 | 1.72 | 0.42 |
| US National | 4.10 | 1.41 | 0.34 |
| HHS Region 8 | 3.31 | 2.08 | 0.63 |
| HHS Region 1 | 3.11 | 2.14 | 0.69 |

Mean amplitude spans more than 2x across locations (3.1 to 7.5),
consistent with the scale differences in [[01-series-overview]].
The coefficient of variation of amplitude across seasons ranges
0.28-0.69, i.e. season-to-season amplitude variability is
proportionally largest for the smallest-amplitude locations
(Region 1, Region 8) — those locations have quieter, more
inconsistent seasons relative to their own scale.

## Cross-location peak-timing agreement

Correlation of each location's peak `woy` with the national peak
`woy`, across the 13 seasons, ranges from 0.37 (Region 2, weakest)
to 0.98 (Region 6, near-perfectly synchronised with the national
signal), with most locations between 0.61 and 0.89.
So peak timing is a mix of a shared national driver plus real
location-specific idiosyncrasy, especially for Region 2 (0.37) and
Region 1/Region 9 (both ~0.61).

![Seasonal wILI curves by location, 2004/05-2016/17 seasons overlaid
(sequential colour = chronological order); dots mark each season's
peak week](figures/03_seasonal_curves.png)

## Implications for the model

- A random-effect seasonal curve indexed by `woy`, allowed to shift
  in timing/amplitude by season (as the brief specifies), is well
  supported: peak timing SD of ~5-8 weeks across seasons, and across
  locations, is too large for a fixed Fourier curve to track well.
- Peak timing and onset/offset numbers above are useful as
  informative priors (e.g. a season-level peak-`woy` random effect
  centred near `woy` 27 with SD ~6) and as posterior-predictive
  sanity checks (does a fitted season's implied peak/onset/offset
  fall in a plausible range for that location?).
- Partially pool the seasonal curve across locations (shared
  large-scale hump shape, e.g. via the national mean) but let
  amplitude scale by location (2x+ range) and allow enough
  location-specific deviation to capture the weaker-than-national
  peak-timing correlation in Region 2 especially.
- Amplitude scaling should probably track each location's own level
  (e.g. multiplicative on the log scale) rather than an additive
  shared-magnitude random effect, given the amplitude range tracks
  the level range from [[01-series-overview]].
- Offset timing is more variable than onset timing across the
  board; if the model separates a rise phase from a decline phase
  (e.g. asymmetric seasonal curve), allow more flexibility on the
  decline side.
- These are full-training-set estimates (13 seasons, still a modest
  n per location); revisit them as the search progresses rather than
  treating them as fixed targets, and never extend this analysis
  into the three held-out test seasons.
