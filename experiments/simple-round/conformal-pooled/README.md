# conformal-pooled

Leak-free pooled-seasonal point forecast wrapped in split-conformal
intervals.
Honest best candidate in the repo (see `submissions/README.md`'s
"Honest leak-free leaderboard"): with a 208-week AR window and a 0.9
interval width scale, validation WIS **0.2730** (cov50 0.498, cov90
0.907) — not itself submitted to the hub (submissions are paused; see
`submissions/README.md`).

## Model

Log transform: $y_{l,t} = \log\!\left(\max(\mathrm{wILI}_{l,t}, \epsilon)\right)$.

**Leak-free pooled seasonal shape**: rebuilt at every forecast origin
from only the rows with `origin_date` strictly before that origin (no
fixed cutoff year). Each location is centred on its own mean over
that available history, deviations are pooled across all 11 locations
by week-of-season $w$, and circularly smoothed (span 3), giving the
same shared shape $s(w)$ as `seabbs_bot-seasstack`, but re-estimated
per split rather than once.

**Leak-free backfill correction**: the same additive
per-`(location, delay)` median profile $r_{l,d}$ as
`seabbs_bot-ar6bf`, also rebuilt at every origin from vintage rows
strictly before that origin.

**AR(6) + climatology, single regression**: unlike `seabbs_bot-season`
and `seabbs_bot-seasstack` (which deseasonalise first, then fit AR to
the residual), here the seasonal shape enters as one extra regressor
alongside the AR(6) lags, fit jointly by OLS:

$$
y_{l,t} = c_l + \sum_{k=1}^{6} \phi_{l,k}\, y_{l,t-k} + \gamma_l\, s(w_t) + \varepsilon_{l,t}.
$$

Writing $\theta_l = (c_l, \phi_{l,1}, \ldots, \phi_{l,6}, \gamma_l)$
for the full coefficient vector, each location's OLS fit is blended
toward one all-locations pooled fit $\theta^{\mathrm{pool}}$ (OLS on
every location's design rows stacked together):

$$
\theta_l^{\mathrm{blend}} = (1 - w)\, \theta_l + w\, \theta^{\mathrm{pool}},
\qquad w = 0.3.
$$

Fit over a 208-week AR window (~4 seasons; the validation optimum of a
leak-free sweep over $\{104,130,156,182,208\}$), not the 104-week
window used elsewhere in this repo.

**Point forecast**: a deterministic recursion (no innovation noise),
carrying $\theta_l^{\mathrm{blend}}$ forward and feeding each
predicted value back in as a lag:

$$
\hat y_{l,T+h} = c_l + \sum_{k=1}^{6} \phi_{l,k}\, \hat y_{l,T+h-k} + \gamma_l\, s(w_{T+h}).
$$

**Split-conformal intervals**: at every origin, a per-horizon pool of
past forecast errors is maintained, pooled across all 11 locations —
each error is $e = y_{\mathrm{actual}} - \hat y_{\mathrm{point}}$ on
the log scale, added to horizon $h$'s pool once its target date has
passed and the outcome is knowable. The calibrated quantile at level
$\tau$ is read directly off this empirical distribution, scaled by a
global width constant $c_w = 0.9$:

$$
\hat q_{l,h}(\tau) = \hat y_{l,T+h} + c_w \cdot Q_h(\tau),
$$

where $Q_h(\tau)$ is the empirical $\tau$-quantile of horizon $h$'s
pooled error distribution.
Before a horizon's pool reaches 10 observations (only the first few
origins of season 1), a Student-t(df=10, scale=1.4) fallback is used
in place of $Q_h(\tau)$.
Quantiles are back-transformed with $\exp(\cdot)$, clamped at 0.

## Files

- `generate.jl` — builds the leak-free pooled point forecast, wraps it
  in split-conformal intervals, scores it on the validation seasons,
  and writes `score.txt`; optionally writes a full 5-season hub
  submission if given a `hub_path`.
- `score.txt` — the validation scoring output, including the AR window
  and interval-width sweeps behind the 208-week/0.9-scale pick.
- `sweep_stack.jl` / `sweep_stack_results.txt` — the stacked-lever
  sweep (AR window, damped-trend blend, interval width) referenced
  above.

## Regenerate

```
julia --project=. experiments/simple-round/conformal-pooled/generate.jl [<hub_path>]
```

Always writes `score.txt`; additionally writes a full 5-season hub
submission if `hub_path` is given.
`WINDOW_WEEKS`, `WIDTH_SCALE`, and `POOL_WEIGHT` are environment-variable
overridable to reproduce the sweeps in `score.txt`.
