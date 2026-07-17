# seabbs_bot-seasstack

Merged as [reichlab PR #80](https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/80).

> **LEAKAGE NOTE.** The `generate.jl`/`generate_forecasts.jl` archived in this
> directory are the ORIGINAL build, which estimated the pooled seasonal and
> backfill profiles once from all `season_year <= 2016` data — for validation
> origins that leaked those seasons' own future weeks. The advertised 0.2601 is
> leak-inflated. The **honest leak-free** score for this exact model (profiles
> rebuilt per-origin from strictly-prior data) is **0.2891** (cov50 0.521,
> cov90 0.914) — still a genuine −3.8% over the clean `season` model (0.3004),
> so the log + Student-t + AR-pooling stack holds up. The leak-free rebuild
> lives in `experiments/simple-round/round2-stack/` (see its `score.txt`). A
> clean submission driver is being centralised via the leak-free `src/`
> profile functions; do not resubmit from the drivers in this dir as-is.

## Method

The pooled-seasonal + backfill core, with three stacked wins:

- Pooled week-of-season climatology (shared shape, pooled deviations across
  all 11 locations) on a **log** scale.
- Non-monotonic delay-indexed **backfill** correction (per location, additive,
  median, 8-week window).
- Per-location **AR(6)** on the deseasonalised residual, partially **pooled**
  toward an all-locations anchor (blend weight w=0.9).
- **Student-t(df=10)** innovations, variance-matched then scaled 1.4, to fix
  the under-coverage of raw Gaussian AR intervals.

## Model

Log transform: $y_{l,t} = \log\!\left(\max(\mathrm{wILI}_{l,t}, \epsilon)\right)$.

**Pooled seasonal shape** (log scale): each location's series is
centred on its own mean, $y_{l,t} - \bar y_l$, and the centred
deviations are pooled across all 11 locations by week-of-season $w$,
giving a raw per-week mean, circularly smoothed (span 3) and
re-centred to mean zero:

$$
s(w) = \mathrm{smooth}_3\!\left(\, \mathrm{mean}\left\{\, y_{l,t} - \bar y_l : w_t = w \,\right\} \right).
$$

Deseasonalised residual: $r_{l,t} = y_{l,t} - \bar y_l - s(w_t)$.

**Backfill correction**: identical additive per-`(location, delay)`
median profile $r_{l,d}$ as `seabbs_bot-ar6bf`/`seabbs_bot-seasonpool`
(see `../seabbs_bot-ar6bf/README.md`), applied on the log scale before
deseasonalising.

**AR(6) with coefficient pooling**: each location's AR(6) is fit to
$r_{l,t}$ by OLS, then blended toward one all-locations pooled fit
$\phi^{\mathrm{pool}}$ (OLS on every location's design rows stacked
together):

$$
\phi_l^{\mathrm{blend}} = (1 - w)\, \phi_l + w\, \phi^{\mathrm{pool}},
\qquad w = 0.9.
$$

**Student-t innovations**: variance-matched to the blended fit's
residual SD $\sigma_l$, then scaled by 1.4 to fix under-coverage:

$$
\varepsilon_{l,t} = \sigma_l \sqrt{\frac{\nu - 2}{\nu}} \times 1.4 \times T_t,
\qquad T_t \sim t_\nu,\ \ \nu = 10.
$$

**Forecast**: 1000 sample paths of $r_{l,t}$ are simulated forward
with $\phi_l^{\mathrm{blend}}$ and the Student-t innovations above,
then the location level and seasonal term are added back at the
future week-of-season, and back-transformed with $\exp(\cdot)$,
clamped at 0.

## Files

- `generate.jl` — the round-2 stack experiment (scores the ablation sweep on
  the validation seasons; see `score.txt`). Source of the winning combo.
- `generate_forecasts.jl` — the submission driver. Runs the winning combo
  across all five seasons and writes the hubverse submission:
  `julia --project=<repo> generate_forecasts.jl <hub_path>`. Each test-season
  origin is a per-week vintage fit (data up to that origin only), never used
  for selection.
- `score.txt` — the validation ablation table.

## Regenerate

```
julia --project=. submissions/seabbs_bot-seasstack/generate_forecasts.jl <hub_clone>
```

then prune to the hub's allowed origin dates (drops the 6 post-2020-03-21
dates outside the hub's task set) and validate before opening the PR.
