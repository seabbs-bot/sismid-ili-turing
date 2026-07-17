# 04. Cross-location structure

Source: `data/flu_data_hhs.csv`, log scale (`log(wili + 0.01)`).
Computed in Julia (CSV.jl, DataFrames.jl, Statistics, StatsBase).

**Experimental integrity: filtered to `season_year <= 2016`** (see
[[01-series-overview]] for the definition), i.e. pre-2015 history
plus the two validation seasons only.
The three held-out testing seasons are excluded throughout.

## Correlation of levels

Pairwise correlation of `log(wili)` across the 11 locations is high
everywhere: off-diagonal correlations range 0.57-0.95, mean 0.76.
`US National` correlates 0.78-0.95 with every region (it is close
to a weighted average of them), and the strongest region-region
pairs (Region 4-Region 5 at 0.87, Region 5-Region 6 at 0.87) sit
next to the weakest (Region 2-Region 10 at 0.57).
Most of this correlation is the shared seasonal cycle from
[[03-seasonality]]: every location has one broad winter hump, so
levels co-move strongly regardless of any genuine coupling in the
residual dynamics.

## Correlation of week-to-week changes

Correlation of the first difference of `log(wili)` (i.e. after
removing most of the seasonal level) drops a lot: off-diagonal
correlations range 0.03-0.69, mean 0.24.
`US National` still correlates strongly with the larger/more
central regions (Region 4: 0.69, Region 5: 0.69, Region 6: 0.60)
but weakly with some smaller or more idiosyncratic regions
(Region 8: 0.31, Region 10: 0.26).
Region-to-region differenced correlations are mostly modest
(0.03-0.37), with a few closer pairs (Region 4-Region 5: 0.52,
Region 3-Region 5: 0.37, Region 3-Region 4: 0.33).
So once the shared seasonal signal is removed, genuine
contemporaneous co-movement in week-to-week changes is real but
moderate, and uneven across the location pairs — not simply "one
correlation for everyone".

## Lead-lag

Cross-correlating each region's differenced `log(wili)` against
the national differenced series at lags -3..+3 weeks, every region
peaks at **lag 0** (contemporaneous), with the correlation falling
off (not always symmetrically, but never exceeding the lag-0 value)
at ±1-3 weeks.
There is no evidence of a region systematically leading or lagging
the national aggregate by one or more weeks; the coupling across
locations in this data is contemporaneous, not propagating
geographically week over week.

## Implications for the model

- The moderate, uneven, but genuinely non-zero correlation of
  differenced series across locations (mean 0.24, up to 0.69 for
  closely related regions) supports a correlated
  (multivariate-normal) location-effect / AR-innovation structure
  over fully independent per-location effects, at least for the
  more central/larger regions.
- Because the coupling is contemporaneous (lag 0) with no lead-lag,
  a VAR with cross-location lagged terms is unlikely to add much
  over a contemporaneous-correlation (MVN innovations) formulation;
  the cheaper "independent-per-location AR with correlated
  innovations" design is a reasonable default, with full VAR as a
  candidate to test rather than an obvious win.
- Region 2, Region 8, Region 9, Region 10 correlate more weakly with
  the rest (both in levels and differences) and may be better served
  by a looser prior on their location-effect correlation, or nearly
  independent treatment, rather than forcing them into the same
  correlation structure as the tightly-coupled core (Region 4,
  Region 5, Region 6, US National).
- These are validation-period-only correlations; re-check them once
  more seasons of data become available for design decisions (never
  by looking at the held-out test seasons themselves, only by
  re-running this same script if the validation window is extended).
