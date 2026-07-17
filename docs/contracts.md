# Interface contracts

Shared interfaces so modules built in parallel integrate cleanly.
All code: max 80 chars per line, no trailing whitespace, Mooncake AD backend
for any Turing sampling. The package module is `SismidILITuring`; component
files are `include`d by `src/SismidILITuring.jl` and share its scope, so
module-level types and constants below are visible without importing.

## Constants (defined in `src/SismidILITuring.jl`)

- `QUANTILE_LEVELS::Vector{Float64}` — the 23 hub quantiles:
  `[0.01, 0.025, 0.05, 0.10, …, 0.90, 0.95, 0.975, 0.99]`.
- `LOCATIONS::Vector{String}` — canonical location order (formal names):
  `"US National"`, `"HHS Region 1"`, …, `"HHS Region 10"`.
- `TARGET::String = "ili perc"`, `HORIZONS = 1:4`.
- `to_scale(w, t)`, `from_scale(x, t)` with `t` a transform symbol; wILI CSV
  values are percentages (0–100). Favoured default `:log`; `:logit`,
  `:fourthroot`, `:log1p` are candidate alternatives. See `src/core.jl`.

## Experimental integrity (do not cheat)

- Tune and select only on the validation seasons (2015/16, 2016/17). The three
  testing seasons (2017/18–2019/20) are held out; do not fit to, look at, or
  select on them until the finalists are locked.
- Use only vintage data available at each forecast origin (the tscv splits and
  version snapshots). Never let finalized or future values leak into a fit.
- Score against the hub oracle output (settled season values), matching the
  session workflow, so comparisons stay fair against the existing hub models.

## Data schema (input CSVs in `data/`)

- `flu_data_hhs.csv`: `location, origin_date, wili` — finalized series.
- `flu_data_hhs_versions.csv`: `location, origin_date, as_of, wili` — every
  reported version. Reporting delay = `as_of - origin_date` in weeks. This is
  the reporting triangle for the backfill model.
- `flu_data_hhs_tscv_seasonN.csv`: `location, origin_date, wili, .split` — one
  expanding-window split per forecast date; `wili` is the vintage value known at
  that split. The forecast origin for a split is `max(origin_date)` within it.

`origin_date` here is a *reference date* (Saturday ending an epiweek). Hub
`target_end_date = forecast_origin + 7 * horizon` days.

## `ModelData` (defined in `src/SismidILITuring.jl`, built by `src/data.jl`)

Matrix form, time × location, for one forecast origin (one split):

```julia
struct ModelData
    Y::Matrix{Union{Missing,Float64}}  # (T×L) vintage obs, modelling scale
    delay::Matrix{Int}                 # (T×L) reporting delay weeks, 0=newest,
                                       #       capped at Dmax; -1 where missing
    woy::Vector{Int}                   # (T) week-of-season index 1..W
    season::Vector{Int}                # (T) season index 1..S (training seasons)
    dates::Vector{Date}                # (T) reference dates, ascending
    L::Int; T::Int; W::Int; S::Int; Dmax::Int
    origin_date::Date                  # forecast origin for this split
end
```

`src/data.jl` exposes:
- `load_series(name)::DataFrame` — read a CSV from `data/`.
- `build_model_data(split_df; Dmax=6, window_weeks=104)::ModelData` — from one
  split's long DataFrame; `window_weeks` caps training history length.
- `training_splits(season::Int)::Vector{DataFrame}` — split DataFrames for a
  season, in forecast-origin order.

## Forecast quantile table (the interface between forecast, scoring, hub I/O)

A tidy `DataFrame` with exactly these columns:

| column | type | notes |
|---|---|---|
| model_id | String | e.g. `nfidd-seasarpp` |
| location | String | formal name in `LOCATIONS` |
| origin_date | Date | forecast origin (reference date) |
| horizon | Int | 1..4 |
| target_end_date | Date | origin_date + 7*horizon |
| target | String | `"ili perc"` |
| output_type | String | `"quantile"` |
| output_type_id | Float64 | a value in `QUANTILE_LEVELS` |
| value | Float64 | predicted wILI **percentage**, natural scale, non-negative |

## Module functions by file

- `src/forecast.jl`: `forecast_quantiles(fit, data, model_id; horizons=HORIZONS,
  levels=QUANTILE_LEVELS)::DataFrame` in the schema above. Draws from the
  posterior predictive, projects dynamics forward, maps back to the natural
  scale, clamps at 0.
- `src/scoring.jl`: `wis(obs, forecast_df)` and `score_forecasts(forecast_df,
  truth_df; scale=:natural)`. Truth table columns: `location, target_end_date,
  value`. Returns per-task WIS and a summary with mean and SD of WIS, on natural
  and log scale (`log1p` of wILI%). Built on `ScoringRules.interval_score`. WIS =
  `(1/(K+0.5)) * (0.5*|y-median| + Σ_k (α_k/2)·IS_{α_k})` over the K interval
  levels implied by `QUANTILE_LEVELS`.
- `src/hubio.jl`: `write_submission(forecast_df, hub_path; designated=true)` —
  one CSV per origin_date under `model-output/<model_id>/`, columns in hub order
  (`origin_date, location, target, horizon, target_end_date, output_type,
  output_type_id, value`, no `model_id`; verified against the real hist-avg
  files), plus `model-metadata/<model_id>.yml`
  (`team_abbr`, `model_abbr`, `designated_model`).
- `src/inference.jl`: `fit_pathfinder(model; draws=1000)` and
  `fit_mcmc(model; samples=1000, chains=2, adtype=AutoMooncake(), cb=nothing)` —
  wrappers with a Turing progress callback for live monitoring.
- `src/model.jl`: `base_model(d::ModelData)` — the base joint Turing model
  (partially-pooled week-of-season seasonality, per-location AR(1) residual,
  non-monotonic delay-indexed backfill revision), returning a model that can be
  fit and forecast.
