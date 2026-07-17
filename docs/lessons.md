# Lessons

Read this before writing any driver that fits a model and produces a
forecast. Every item here was a real bug or a real slowdown hit at
least once; the fixes are cheap, refitting is not.

## 1. Use the shared tools; do not hand-roll the forecast path

Turning `(build_model, data, model_id)` into a hub forecast table is
ONE correct sequence: build the model, fit it with Pathfinder, pull
its `generated_draws`, project them forward with `forecast_quantiles`.
Three call sites (`src/pipeline.jl`'s `produce_submission`,
`experiments/run_round.jl`'s `run_candidate`, and
`experiments/round1_run.jl`'s `fit_one_split`) each wrote this
sequence out by hand, and the same bug (see next item) had to be
found and fixed in each of them separately.

[`fit_and_forecast`](../src/pipeline.jl) (exported from
`SismidILITuring`) is now the single source of truth:

```julia
fit_and_forecast(build_model, data, model_id;
    project=base_project, ndraws=200, nruns=1,
    transform=data.transform) -> (forecast, model, fit, draws)
```

`produce_submission` and `round1_run.jl`'s `fit_one_split` both call
it. If you are writing a new driver (a new round, a new smoke test, a
one-off diagnostic script), call `fit_and_forecast` — do not write
`model = build_model(...); fit = fit_pathfinder(...); ...` yourself.
If it does not do what you need, extend it or ask; do not fork it.

## 2. ALWAYS use `generated_draws`, never `posterior_draws`, for forecasting

[`posterior_draws(fit)`](../src/inference.jl) exposes only the raw
sampled `~` sites. [`generated_draws(model, fit)`](../src/inference.jl)
re-evaluates the model's RETURN value for every draw, and that return
value is where `base_model` (and every `project_vN`) puts the derived
quantities `forecast_quantiles`'s `project` functions actually consume:
`mu0`, `mu_w`, `delta`, `season_eff`, `phi`, `sigma_ar`, `residual`,
`latent`, etc. `posterior_draws` does not carry these.

Passing `posterior_draws(fit)` into `forecast_quantiles` does not
error — it silently forecasts garbage (missing fields either error
deep in `project` or, worse, resolve to the wrong sampled site with a
similar name). This exact bug shipped independently in
`produce_submission`, `run_candidate`, and `fit_one_split` before
`fit_and_forecast` existed. Now that all three call the shared
function, fixing it once fixes it everywhere — keep it that way.

## 3. Compile once per process; never a cold Julia subprocess per fit

Turing + Mooncake + Pathfinder pay a real time-to-first-gradient cost
on first use in a process (package precompilation plus the first
AD trace). `round1_run.jl` already gets this right by design: one OS
process per CANDIDATE (for crash isolation against shared-host
segfaults), looping over all 8 origin-date splits for that candidate
INSIDE that one process, so the compile cost is paid once and
amortised over 8 fits, not paid 8 times.

Do not shell out to `julia script.jl` per split or per fit — that pays
the full Turing/Mooncake compile tax on every single fit. If you need
per-fit crash isolation, isolate at the same granularity round1_run.jl
does (one process per candidate, looping splits inside), not finer.

## 4. Do not over-parallelise the shared box

Each concurrent `julia` process pays its own independent
precompilation of Turing/Mooncake/Pathfinder; running many of them at
once multiplies memory pressure rather than sharing compiled code, and
this box swaps hard under that load (`~/.claude/hooks/compute-budget.sh`
has read swap_used at 99% while several round1 candidates ran at
once). One compile-once, appropriately-threaded job beats a swarm of
cold processes. Check `compute-budget.sh`'s verdict before starting
more concurrent fits; when it is not green, run fewer candidates at
once rather than more.

## 5. Why a single fit takes ~10 minutes, and fast-screening defaults

`base_model` is a joint model across all 11 locations with
`window_weeks=104` (two seasons) of weekly history. The dominant cost
is parameter count, not draw count or Pathfinder's path count:

- `eps_raw` is `T x L` — with `window_weeks=104`, `T ~ 104`, so this
  alone is ~1,144 unconstrained parameters, and it scales linearly
  with `window_weeks`. This is the single biggest lever available.
- `delta_raw` is `W x L` (`W` = weeks-in-season, ~52) — ~572
  parameters, but `W` does NOT scale with `window_weeks`, so trimming
  `window_weeks` does not shrink this block.
- The observation loop in `base_model` (`src/model.jl`) is a manual
  `for l in 1:L, t in 1:T` with a scalar `~ Normal(...)` per cell
  (~1,144 individual observe statements at `window_weeks=104`) rather
  than one vectorised likelihood call. Mooncake must trace/replay this
  every gradient evaluation. This is a real overhead source but is a
  `src/model.jl` architecture question, out of scope here — flagged
  as a follow-up for whoever next touches `base_model`, not fixed in
  this pass.
- `ndraws` (Pathfinder draws taken from the fitted approximation) and
  `nruns` (single- vs multi-path) are NOT the bottleneck: sampling
  from the fitted approximation is cheap MVN sampling that happens
  once optimisation has already converged. `nruns=1` (single-path,
  already the default everywhere in this repo) is already the fast
  choice; never set `nruns>1` for screening, only for a finalist
  suspected of a multimodal posterior.

**Fast-screening defaults for NEW search work** (do not retrofit onto
an in-flight round — round1 is committed to `window_weeks=104,
ndraws=150, Dmax=12` for comparability across its candidates, see
`experiments/round1/_results/*/meta.txt`):

- `window_weeks` ~52 (one season) for a coarse first pass to rank
  structurally different candidates before committing full compute.
  Trade-off: with only one season in the window, `season_eff` and the
  per-location `delta` seasonal deviation are far less identifiable
  (there is no second season to partially pool against), so WIS from
  a `window_weeks=52` screen is not comparable to, and should not
  replace, the `window_weeks=104` numbers used for actual candidate
  selection. Use it only to cheaply discard clearly-broken candidates.
- `ndraws=100` (down from 150-200) trims the downstream
  `forecast_quantiles`/`bayesian_checks` summarising cost a little,
  though this is secondary to `window_weeks`.
- `nruns=1` always for screening.
- `Dmax=12` unchanged — `(Dmax+1) x L = 143` parameters is small
  next to `eps_raw`, and `Dmax` is a data/backfill-realism choice
  (docs/contracts.md), not a speed lever worth compromising.

## 6. The forecast table schema

`fit_and_forecast(...).forecast` / `forecast_quantiles(...)` returns
exactly this schema (docs/contracts.md has the authoritative copy):

| column | type | notes |
|---|---|---|
| model_id | String | e.g. `nfidd-turing` |
| location | String | formal name in `LOCATIONS` |
| origin_date | Date | forecast origin (reference date) |
| horizon | Int | 1..4 |
| target_end_date | Date | origin_date + 7*horizon |
| target | String | `"ili perc"` |
| output_type | String | `"quantile"` |
| output_type_id | Float64 | a value in `QUANTILE_LEVELS` |
| value | Float64 | predicted wILI %, natural scale, non-negative |

## 7. The current transform is `fourthroot`, not `log`

`docs/contracts.md` still lists `:log` as the "favoured default" and
`src/pipeline.jl`'s `produce_submission` still defaults to `:log1p` —
both predate the search. `experiments/round1_run.jl`'s
`PRIMARY_TRANSFORM = :fourthroot` is what every round1 candidate
except the explicit `nfidd-base-log` control actually fits on (see
`experiments/round1/_results/*/meta.txt`). Do not assume an older
"favoured default" comment is still current — check
`PRIMARY_TRANSFORM` / the latest round's `summary.txt` for the
transform actually in use before adding a new candidate or driver.
