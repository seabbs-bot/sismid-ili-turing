# EDA reports index

Exploratory analysis of the ILI forecasting data in `data/`, written
to inform the joint Turing model design in `docs/brief.md` and
`docs/contracts.md`.
All analysis is done in Julia (CSV.jl, DataFrames.jl, Statistics,
StatsBase — no R).
Each report is self-contained and citable by number.

**Experimental integrity: every report below is computed only on
pre-2015 history plus the two validation seasons (`2015/16`,
`2016/17`), i.e. `season_year <= 2016`** under the week-of-season
definition in [[01-series-overview]].
The three held-out testing seasons (`2017/18`-`2019/20`) are
deliberately excluded from all EDA, so model selection cannot be
biased by their structure.
Treat these reports as living inputs: the search will revisit and
re-run this analysis across rounds rather than treating any number
here as fixed, and reports should be extended (not just read) as
new questions come up during the search.

- [[01-series-overview]] - the 11 locations, scales, season
  definition, missingness, and a comparison of `log1p` / logit /
  fourth-root transforms.
  `log1p`-style log is the recommended default scale.
- [[02-backfill]] - the reporting/revision process from the versions
  data: revision size by delay, non-monotonic direction, and
  variation by location and season phase.
  Revisions are non-monotonic and location-varying, settling by
  roughly delay 10-15 weeks.
- [[03-seasonality]] - **peak timing, peak height, and season
  onset/offset per location and per season**, plus curve shape and
  amplitude, as cross-season features for the model agents.
  Peak timing SD is ~5-8 weeks across seasons and locations,
  supporting a season-varying random-effect seasonal curve over a
  fixed Fourier curve; offset timing is more variable than onset.
- [[04-cross-location]] - correlation of levels and of week-to-week
  changes across the 11 locations, and lead-lag structure.
  Differenced-series correlation is moderate (mean 0.24) and
  contemporaneous (lag 0), supporting correlated
  (multivariate-normal) location effects over independent ones, and
  making a full VAR's lagged cross-terms a lower-priority
  refinement.
- [[05-autocorrelation]] - ACF/PACF of the (deseasonalised) log
  series and its first difference, per location.
  AIC-selected AR order on the deseasonalised residual has median 5
  (range 4-10) across locations; AR is preferred over differencing
  for most locations (differencing shows an over-differencing
  artefact, negative lag-1 ACF), except Region 4/Region 5.
