# 07. Region 9 deep dive

Source: `data/flu_data_hhs.csv`, `data/flu_data_hhs_versions.csv`.
Computed in Julia (CSV.jl, DataFrames.jl, Statistics, StatsBase).
Region 9 has come up as an outlier, in passing, in several separate
reports: the most Poisson-like transform power in
[[01-series-overview]], the strongest downward backfill bias in
[[02-backfill]], and the weakest amplitude correlation in
[[04-cross-location]].
This report pulls those threads together in one place and checks
whether Region 9 is a genuine, consistent outlier, or whether it is
flagged for different, unrelated reasons each time — a distinction
that matters for whether the model needs one Region-9-specific
mechanism or several independent ones.

**Experimental integrity: filtered to `season_year <= 2016`** (see
[[01-series-overview]] for the definition), the full training set —
pre-2015 history plus the two validation seasons.
The three held-out testing seasons are excluded throughout.

## Where Region 9 is an outlier

![Region 9 across three axes: Taylor's power law (left), seasonal
amplitude vs `US National` by season (middle), and delay-1 backfill
revision direction vs the pooled rate (right); Region 9 is a clear
outlier on the first and third, only moderately decoupled on the
second](figures/12_region9_deepdive.png)

- **Transform power (variance-stabilisation).**
  Region 9's fitted Taylor's-law slope is `lambda=0.91`, against a
  pooled `lambda=1.77` and a per-location range of 0.91-3.17 — Region
  9 is the location that defines the low end of that range, sitting
  close to genuinely Poisson-like (`lambda~1` implies `sqrt`,
  `p=0.5`) while every other location is closer to lognormal.
  Its Box-Cox `lambda` (0.29) is likewise the upper end of the
  per-location range in [[01-series-overview]] (-0.40 to 0.29).
  This is a real, well-evidenced difference in Region 9's own
  mean-variance relationship, not noise: the left panel above shows
  its local cells visibly following a shallower slope than the
  pooled fit.
- **Backfill revision direction.**
  Region 9 has the second-largest (after Region 2) mean signed
  delay-1 revision in [[02-backfill]] (-0.271, only 39% of first
  reports revised upward, pooled over the full tracked period) —
  i.e. Region 9's first reports systematically *overcount*, more
  strongly than any location except Region 2 undercounts.
  Split by tracked season (right panel above): 27% up in `2015/16`
  (a strong downward bias, well below the ~59-60% pooled rate that
  season) shifting to 51% up in `2016/17` (roughly neutral, in line
  with pooled) — a real within-Region-9 shift, though milder than
  Region 1/Region 4's full reversal in the same two seasons.
- **Baseline level.**
  Region 9's off-season baseline (1.26, [[06-regional-heterogeneity]])
  is the second-highest of all 11 locations, behind only Region 6.

## Where Region 9 is not an outlier

- **Seasonal amplitude correlation.**
  Region 9's amplitude correlates with `US National` at r=0.43
  (middle panel above) and with other locations at 0.25-0.59
  ([[04-cross-location]]) — the weakest in the 11x11 matrix, but
  "weakest of a mean-0.68 matrix" is a real but moderate outlier, not
  a location whose amplitude is unrelated to the rest (0.43 is still
  a positive, meaningful correlation, just the smallest one).
  The middle panel shows Region 9 and `US National` amplitude broadly
  tracking each other in most seasons, with a few clear misses
  (season 5 and season 16 in particular).
- **Peak-timing correlation.**
  Region 9's peak-`woy` correlates with the national peak at r~0.61
  ([[03-seasonality]]) — mid-pack, not the weakest (that is Region 2
  at 0.37).
- **Differenced-series volatility.**
  Region 9's SD of `Delta log(wili)` is 0.213
  ([[06-regional-heterogeneity]]), close to the middle of the
  0.122-0.430 range across locations, not the noisiest.
- **AR order and the differencing signature.**
  Region 9 selects a low AIC AR order (4, tied with `US National` and
  Region 8, [[05-autocorrelation]]) and is one of only three
  locations (with Region 7 and Region 10) that show the
  negative-lag-1 "AR beats differencing" signature in *every one* of
  the 13 training-set seasons — i.e. on this axis Region 9 is one of
  the most *consistent*, least anomalous locations, not an outlier.
- **Onset timing.**
  Region 9's onset-week correlation with the national pattern is
  r=0.57 ([[03-seasonality]] round-2 addition), close to the median
  across locations, not a standout either way.

## Implications for the model

- Region 9 needs at least two, probably unrelated,
  location-specific adjustments rather than one general "Region 9 is
  different" mechanism: a shallower observation-noise power (closer
  to `sqrt` than the shared fourth-root default) and a stronger,
  season-varying downward backfill bias.
  Bundling these into a single "Region 9 effect" would risk fitting
  the wrong shared cause to two independent phenomena.
- Region 9's amplitude decoupling (r=0.43 vs a 0.68 mean) is real but
  much milder than its backfill/transform anomalies; a season-level
  severity effect (per [[04-cross-location]]) can still include
  Region 9, just with more per-location slack than for the
  tightly-coupled core (Region 5, Region 6, `US National`).
- More generally: cross-report location outliers should be checked
  individually before being treated as a single "difficult location"
  flag.
  Region 9 is genuinely anomalous on some axes (transform power,
  backfill direction, baseline level) and unremarkable on others
  (volatility, AR order, differencing stability, onset timing,
  peak-timing correlation); a model that special-cases Region 9
  broadly (e.g. a location-level "difficulty" random effect shared
  across all these mechanisms) would be solving a problem that
  Region 9's data does not actually pose on most axes.
- These are full-training-set observations (13 seasons for the
  seasonal/transform statistics, two tracked seasons for backfill);
  revisit if the training window changes, and never draw on the
  three held-out test seasons.
