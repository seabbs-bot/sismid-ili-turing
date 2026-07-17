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
  definition, missingness, and a variance-stabilisation-led
  comparison of `log` / `log1p` / logit / fourth-root transforms
  (Taylor's power law and Box-Cox fits).
  Fourth-root is the recommended default scale: the fitted power
  (Taylor's-law `lambda~1.8`, Box-Cox `lambda~0.13`) sits between
  `log` and fourth-root, but a direct flatness comparison shows
  fourth-root clearly outperforms `log`/`log1p` at stabilising
  variance in this data.
  Pooled, per-location-standardised histograms and QQ-plots make the
  same variance-stabilisation/skewness story visible directly at the
  observation level.
- [[02-backfill]] - the reporting/revision process from the versions
  data: revision size by delay, non-monotonic direction, and
  variation by location and season phase.
  Revisions are non-monotonic and location-varying, settling by
  roughly delay 10-15 weeks.
  A per-location, per-season breakdown shows this is not a small
  pooled effect: Region 1 and Region 4 fully reverse the sign of
  their delay-1 revision bias between `2015/16` and `2016/17`, and a
  phase-crossed check confirms this is a genuine season-level shift,
  not a peak-vs-off-season artefact.
  On the fourth-root modelling scale the location-varying revision
  spread compresses from ~14x to ~7x, but Region 2 and Region 9 stay
  the two most heavily revised locations either way.
- [[03-seasonality]] - **peak timing, peak height, and season
  onset/offset per location and per season**, plus curve shape and
  amplitude, as cross-season features for the model agents.
  Peak timing SD is ~5-8 weeks across seasons and locations,
  supporting a season-varying random-effect seasonal curve over a
  fixed Fourier curve; offset timing is more variable than onset.
  Onset-week correlation across locations (mean 0.24) is much weaker
  than the amplitude correlation in [[04-cross-location]] (mean
  0.68): a severe season tends to be severe everywhere, but does not
  necessarily start early everywhere.
- [[04-cross-location]] - correlation of levels and of week-to-week
  changes across the 11 locations, and lead-lag structure.
  Differenced-series correlation is moderate (mean 0.24) and
  contemporaneous (lag 0), supporting correlated
  (multivariate-normal) location effects over independent ones, and
  making a full VAR's lagged cross-terms a lower-priority
  refinement.
  Correlation of per-season seasonal *amplitude* is much stronger
  (mean 0.68), a shared severity-year signal distinct from the
  week-to-week coupling above; Region 9 is the weak link on this
  axis, Region 2 on peak timing in [[03-seasonality]] — different
  locations are the outlier depending on which structure is tested.
  Season *duration* (offset minus onset) is the weakest of all three
  shared-year tests (mean r=0.06, essentially independent per
  location), completing a clear ordering: amplitude (strong) >
  onset timing (weak) > duration (~none).
- [[05-autocorrelation]] - ACF/PACF of the (deseasonalised) log
  series and its first difference, per location.
  AIC-selected AR order on the deseasonalised residual has median 5
  (range 4-10) across locations; AR is preferred over differencing
  for most locations (differencing shows an over-differencing
  artefact, negative lag-1 ACF), except Region 4/Region 5.
  A per-season breakdown shows persistence is stable season to
  season but the differencing sign is close to a coin flip for most
  locations; only Region 7/Region 9/Region 10 show it in every
  season.
- [[06-regional-heterogeneity]] - three per-location summary
  statistics (off-season baseline, mean seasonal amplitude,
  differenced-series volatility) compared side by side.
  The three rankings are not the same; volatility correlates
  negatively with baseline (r = -0.63), so smaller locations are
  proportionally noisier week to week, not just scaled-down copies
  of the larger ones.
- [[07-region9-deepdive]] - collects every passing mention of
  Region 9 as an outlier (across reports 01, 02, 03, 04, 06) into one
  place and checks each one directly.
  Region 9 is a genuine, well-evidenced outlier on transform power
  (most Poisson-like, `lambda=0.91`) and backfill direction
  (strongest downward bias after Region 2), but unremarkable on
  volatility, AR order, differencing stability, onset timing and
  peak-timing correlation — it needs targeted fixes, not one general
  "difficult location" mechanism.
