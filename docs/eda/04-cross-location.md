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

![Cross-location correlation heatmaps: levels (left, high
throughout, shared seasonality) vs week-to-week changes (right,
moderate and uneven once
differenced)](figures/04_cross_location_correlation.png)

## Correlation of seasonal amplitude: a shared severity-year effect

The two correlation types above are both computed on the weekly
series (levels, then differences).
This section asks a season-level version of the same question: is a
season that is unusually severe (high amplitude) at one location
also unusually severe elsewhere, using the per-season amplitude
(season peak minus that location's own off-season baseline, as
defined in [[03-seasonality]] and [[06-regional-heterogeneity]])
correlated pairwise across the 13 training-set seasons.

![Cross-location correlation of per-season seasonal amplitude
(left), and each location's amplitude (z-scored) plotted against
the cross-location mean amplitude for that season (right); a
moderate-to-strong shared severity-year signal, clearest for the
larger/central regions and weakest for Region 9 (validation +
history only, n=13 seasons per
pair)](figures/11_amplitude_correlation.png)

Off-diagonal amplitude correlation is mean 0.68, range 0.25
(Region 2-Region 9) to 0.94 (Region 5-`US National`,
Region 6-`US National`) — noticeably **stronger** than the
differenced weekly-series correlation above (mean 0.24), i.e. the
shared signal across locations is much more visible at the level of
"how bad was this whole season" than at the level of "did this
specific week move together".
`US National`'s amplitude correlates 0.72-0.94 with every region
except Region 9 (0.43) and Region 10 (0.60), so most seasons that
are severe nationally are severe at nearly every region too, with
Region 9 standing out as the location whose severity is least
coupled to the rest.
This is a different weak link from the peak-*timing* correlation in
[[03-seasonality]], where Region 2 was the outlier (0.37) and
Region 9 was only mildly weak (~0.61): a location can be
well-synchronised on severity but not on timing, or vice versa, so
these are genuinely separate axes of cross-location structure worth
tracking independently rather than assuming one location is simply
"the odd one out" across the board.

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
- These are full-training-set correlations (13 seasons, `2004/05`
  -`2016/17`); re-check them if the training window is later extended
  with more history, but never by drawing on the three held-out test
  seasons before final evaluation.
- The seasonal-amplitude correlation is materially stronger (mean
  0.68) than the differenced weekly-series correlation (mean 0.24)
  above, so a season-level shared severity effect (e.g. a
  season-level random effect on amplitude, correlated or shared
  across locations, on top of the per-location amplitude scaling
  already noted in [[03-seasonality]]) is worth adding as a
  candidate structure alongside the week-to-week location
  correlation, not as a substitute for it — they capture different
  parts of the cross-location coupling.
  Region 9's amplitude is the one clearly under-coupled to the rest
  (0.25-0.59 with other locations) and may need its own, looser
  severity-year term.
