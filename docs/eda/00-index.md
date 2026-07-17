# EDA reports index

Exploratory analysis of the ILI forecasting data in `data/`, written
to inform the joint Turing model design in `docs/brief.md` and
`docs/contracts.md`.
Each report is self-contained and citable by number.

- [[01-series-overview]] - the 11 locations, scales, season
  definition, missingness, and a comparison of `log1p` / logit /
  fourth-root transforms.
  `log1p`-style log is the recommended default scale.
- [[02-backfill]] - the reporting/revision process from the versions
  data: revision size by delay, non-monotonic direction, and
  variation by location and season phase.
  Revisions are non-monotonic and location-varying, settling by
  roughly delay 10-15 weeks.
- [[03-seasonality]] - shape of the seasonal curve, and how much
  peak timing and amplitude vary across seasons and locations.
  Peak timing SD is ~5 weeks across seasons, supporting a
  season-varying random-effect seasonal curve over a fixed Fourier
  curve.
- [[04-cross-location]] - correlation of levels and of week-to-week
  changes across the 11 locations, and lead-lag structure.
  Differenced-series correlation is moderate (mean 0.26) and
  contemporaneous (lag 0), supporting correlated
  (multivariate-normal) location effects over independent ones, and
  making a full VAR's lagged cross-terms a lower-priority
  refinement.
- [[05-autocorrelation]] - ACF/PACF of the (deseasonalised) log
  series and its first difference, per location.
  AIC-selected AR order on the deseasonalised residual has median 6
  (range 2-10) across locations; AR is preferred over differencing,
  which shows an over-differencing artefact (negative lag-1 ACF).
