# seabbs_bot-season

Merged as [reichlab PR #79](https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/79)
(validation WIS 0.3004). Archived here from
`experiments/simple-round/season/generate.jl`, its original home (this
directory did not previously exist — the archiving step was missed at
submission time).

> **Backfill leakage note.** Like the other pre-`conformal-pooled`
> models in this repo, the backfill revision profile below is fit
> ONCE from `season_year <= 2016` (pre-2015 history plus both
> validation seasons), not rebuilt per forecast origin — so it leaks
> a validation split's own season's future weeks into its correction.
> The per-location climatology term is unaffected (it is already
> rebuilt leak-free at every origin, from `origin_date < forecast
> origin` only). See `submissions/README.md`'s "Hub submissions
> PAUSED" note for the recorded leak-free follow-up.

## Model

AR(6) per location, fourth-root transform, fit by OLS, with a
**per-location climatology** term added as one extra regressor
alongside the AR(6) lags, plus the same backfill correction as
`seabbs_bot-ar6bf`.

**Climatology**: unlike `seabbs_bot-seasonpool`/`seabbs_bot-seasstack`
(one shape pooled across all 11 locations), this is a separate curve
per location, built ONLY from that location's own history strictly
before the split's own forecast origin $\tau$ (leak-free):

$$
m_l(w) = \mathrm{median}\left\{\, g(\mathrm{wILI}_{l,t}) : w_t \equiv w \!\!\pmod{52},\ t < \tau \,\right\},
$$

then smoothed with a 5-week circular moving average to keep it
regularised (one number per calendar week borrowed from all available
history, not a free parameter fit to the ~2-season AR window):

$$
\mathrm{clim}_l(w) = \frac{1}{5}\sum_{o=-2}^{2} m_l\!\left((w + o) \bmod 52\right).
$$

**Backfill correction**: identical additive per-`(location, delay)`
median profile $r_{l,d}$ as `seabbs_bot-ar6bf` (see
`../seabbs_bot-ar6bf/README.md`), applied to the vintage series before
fitting.

**AR(6) + climatology, single regression**: the climatology value
enters as one more OLS column alongside the AR(6) lags, so its
coefficient $\gamma_l$ is free to shrink to ~0 itself if uninformative
for a given location/split (unlike a free Fourier harmonic, there is
no separate frequency or phase left for the fit to overfit):

$$
\tilde y_{l,t} = c_l + \sum_{k=1}^{6} \phi_{l,k}\, \tilde y_{l,t-k} + \gamma_l\, \mathrm{clim}_l(w_t) + \varepsilon_{l,t},
\qquad \varepsilon_{l,t} \sim \mathrm{N}\!\left(0, \sigma_l^2\right),
$$

where $\tilde y_{l,t}$ is the backfill-corrected series.

**Forecast**: 1000 Gaussian-innovation sample paths are simulated
forward from this recursion, with $\mathrm{clim}_l$ evaluated at each
future week-of-season, feeding each simulated value back in as a lag
for later horizons. The 23 hub quantile levels are taken across the
simulated paths and back-transformed with $g^{-1}(x) = x^4$, clamped
at 0.

## Usage

```
julia --project=<sismid-ili-turing repo> generate_forecasts.jl <hub_path>
```

Reads `data/flu_data_hhs_tscv_season{1,2,3,4,5}.csv`,
`data/flu_data_hhs_versions.csv`, and `data/flu_data_hhs.csv` (the
per-location climatology's full history), fits and forecasts every
split of all five seasons, and writes
`<hub_path>/model-output/seabbs_bot-season/<origin_date>-seabbs_bot-season.csv`
plus `<hub_path>/model-metadata/seabbs_bot-season.yml` when `hub_path`
is given.

## Performance

Selection is on the validation seasons only
(`experiments/simple-round/season/score.txt`):

| Model | Mean WIS (natural) | SD |
|---|---|---|
| nfidd-ar6 | 0.3684 | 0.4708 |
| seabbs_bot-ar6bf | 0.3590 | 0.4521 |
| seabbs_bot-season (climatology + backfill) | **0.3004** | 0.3890 |

Best of every seasonal form tried in this family (plain climatology,
Fourier(1)/(2)/(3) at various ridge penalties, and combinations);
wins in 10 of 11 locations and at every horizon individually, not
just on average.
