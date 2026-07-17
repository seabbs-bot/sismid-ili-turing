# seabbs_bot-seasonpool

Archived alongside the merged `seabbs_bot-season` model (PR
[#79](https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/79),
per-location climatology, `experiments/simple-round/season/`), but a
distinct design: this one pools the seasonal *shape* itself across all
11 locations rather than fitting a separate climatology curve per
location.
Validation-season result 0.3049 vs 0.3004 for `seabbs_bot-season`; not
itself the subject of a hub PR (see the "Wide simple round" table in
`submissions/README.md`).

## Model

AR(6) per location, fit by OLS on the fourth-root-transformed vintage
series, plus the same backfill correction as `seabbs_bot-ar6bf`, plus
a **pooled** week-of-season seasonal term.

**Backfill correction**: identical to `seabbs_bot-ar6bf`, an additive
per-`(location, delay)` median correction $r_{l,d}$, applied to the
vintage series for delays up to 8 weeks before anything else is fit
(see `../seabbs_bot-ar6bf/README.md` for the full derivation).

**Pooled seasonal shape**: a single shared week-of-season shape,
3-harmonic Fourier on week-of-season $w$ (period 52 weeks):

$$
s(w) = \sum_{k=1}^{3} \left[ a_k \sin\!\left(\frac{2\pi k w}{52}\right) + b_k \cos\!\left(\frac{2\pi k w}{52}\right) \right].
$$

The 6 coefficients $a_k, b_k$ are fit ONCE by no-intercept OLS,
pooling all 11 locations' fourth-root-scale deviations from their own
mean, over `season_year <= 2014` history only (true pre-2015 history,
disjoint from both validation and test seasons; about 6,700 pooled
observations behind 6 parameters).
This avoids the overfitting a free per-location Fourier(3) fit showed
(`nfidd-ar6`'s discarded Fourier variant, 6 parameters fit to only
~2 seasons of window, scored worse than plain AR(6)).

Per split, per location, a 2-parameter regression adapts the shared
shape to that location's own level and amplitude:

$$
y_{l,t} = \alpha_l + \beta_l\, s(w_t) + u_{l,t},
$$

fit by OLS on that split's own `window_weeks=104` window, giving the
deseasonalised residual $u_{l,t}$.

**AR(6) on the residual**: fit to $u_{l,t}$ exactly as in `nfidd-ar6`:

$$
u_{l,t} = c_l + \sum_{k=1}^{6} \phi_{l,k}\, u_{l,t-k} + \varepsilon_{l,t},
\qquad \varepsilon_{l,t} \sim \mathrm{N}\left(0, \sigma_l^2\right).
$$

**Forecast**: 1000 Gaussian-innovation sample paths of the residual
recursion are simulated forward, then the seasonal term at the
(known) future week-of-season is added back:

$$
\hat y_{l,T+h} = \alpha_l + \beta_l\, s(w_{T+h}) + \hat u_{l,T+h}.
$$

Only the AR component propagates simulated uncertainty forward; the
23 hub quantile levels are taken across the simulated paths and
back-transformed with $g^{-1}(x) = x^4$, clamped at 0.

## Usage

```
julia --project=<sismid-ili-turing repo> generate_forecasts.jl <hub_path>
```

Reads `data/flu_data_hhs_tscv_season{1,2,3,4,5}.csv`,
`data/flu_data_hhs_versions.csv`, and `data/flu_data_hhs.csv` (the
pooled shape's full history), fits and forecasts every split of all
five seasons, and writes
`<hub_path>/model-output/seabbs_bot-seasonpool/<origin_date>-seabbs_bot-seasonpool.csv`
plus `<hub_path>/model-metadata/seabbs_bot-seasonpool.yml` when
`hub_path` is given.

## Performance

Selection is on the validation seasons only
(`experiments/simple-round/seasonpool/score.txt`):

| Model | Mean WIS (natural) |
|---|---|
| nfidd-ar6 | 0.368 |
| seabbs_bot-ar6bf | 0.359 |
| seabbs_bot-seasonpool | 0.3049 |

15% better than `seabbs_bot-ar6bf`, with the gain concentrated at
longer horizons and in HHS Regions 9, 6, and 2 — the same regions that
show the strongest seasonal signal and the largest backfill gains.
