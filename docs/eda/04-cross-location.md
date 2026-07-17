# 04. Cross-location structure

Source: `data/flu_data_hhs.csv`, log scale (`log(wili + 0.01)`).

## Correlation of levels

Pairwise correlation of `log(wili)` across the 11 locations is high
everywhere: off-diagonal correlations range 0.61-0.96, mean 0.78.
`US National` correlates 0.83-0.96 with every region (it is close
to a weighted average of them), and the strongest region-region
pairs (Region 4-Region 5 at 0.88, Region 5-Region 6 at 0.89) sit
next to the weakest (Region 2-Region 10 at 0.61).
Most of this correlation is the shared seasonal cycle from
[[03-seasonality]]: every location has one broad winter hump, so
levels co-move strongly regardless of any genuine coupling in the
residual dynamics.

## Correlation of week-to-week changes

Correlation of the first difference of `log(wili)` (i.e. after
removing most of the seasonal level) drops a lot: off-diagonal
correlations range 0.05-0.72, mean 0.26.
`US National` still correlates strongly with the larger/more
central regions (Region 4: 0.72, Region 5: 0.70, Region 6: 0.63)
but weakly with some smaller or more idiosyncratic regions
(Region 8: 0.34, Region 10: 0.28).
Region-to-region differenced correlations are mostly modest
(0.05-0.4), with a few closer pairs (Region 4-Region 5: 0.52,
Region 3-Region 5: 0.40, Region 4-Region 6: 0.40).
So once the shared seasonal signal is removed, genuine
contemporaneous co-movement in week-to-week changes is real but
moderate, and uneven across the location pairs — not simply "one
correlation for everyone".

## Lead-lag

Cross-correlating each region's differenced `log(wili)` against
the national differenced series at lags -3..+3 weeks, every region
peaks at **lag 0** (contemporaneous), with the correlation falling
off symmetrically at ±1-3 weeks.
There is no evidence of a region systematically leading or lagging
the national aggregate by one or more weeks; the coupling across
locations in this data is contemporaneous, not propagating
geographically week over week.

## Implications for the model

- The moderate, uneven, but genuinely non-zero correlation of
  differenced series across locations (mean 0.26, up to 0.72 for
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
