# Posterior-predictive forecasting: projects latent dynamics beyond the
# forecast origin, maps back to the natural wILI percentage scale, and
# summarises into the hub's tidy quantile schema. See docs/contracts.md.
#
# Assumes `src/core.jl` is already loaded (QUANTILE_LEVELS, LOCATIONS,
# TARGET, HORIZONS, to_scale/from_scale, ModelData).

using DataFrames
using Dates
using Statistics: quantile, mean

"""
    default_project(draw, data, horizons)

Default `project` function used by [`forecast_quantiles`](@ref).
Reconstructs a partially-pooled seasonal + per-location AR(1) latent
path, on the modelling scale, for `h = 1:maximum(horizons)` weeks beyond
`data.origin_date`. This is the fallback used when no model-specific
`project` is supplied.

`draw` is a single posterior draw, as a `NamedTuple` or `Dict{Symbol}`
(whichever `src/model.jl` finds convenient). It must provide:

- `seasonal::AbstractMatrix` — (W × L) week-of-season effect by
  location, on the modelling scale. The forecast week for horizon `h`
  looks up row `mod1(last_woy + h, data.W)`, where
  `last_woy = data.woy[end]`.
- `ar_coef::AbstractVector` — (L) per-location AR(1) coefficient.
- `resid_sd::AbstractVector` — (L) per-location AR(1) innovation sd.
- `last_resid::AbstractVector` — (L) post-seasonal AR(1) residual at
  the final training week, i.e. the state the AR(1) recursion is
  seeded from for the h = 1 step.

If `src/model.jl` names its draw fields differently, pass a matching
`project` keyword to `forecast_quantiles` instead of renaming its
model's draws.

Returns an `(L × maximum(horizons))` `Matrix{Float64}` of latent-scale
(pre-`from_scale`) forecasts. Each call draws fresh AR(1) innovation
noise, so repeated calls with the same `draw` give different
posterior-predictive realisations, matching what an MCMC/Pathfinder
draw represents.
"""
function default_project(draw, data::ModelData, horizons)
    L = data.L
    H = maximum(horizons)
    seasonal = _field(draw, :seasonal)
    ar = _field(draw, :ar_coef)
    sd = _field(draw, :resid_sd)
    resid = copy(_field(draw, :last_resid))
    w0 = data.woy[end]
    latent = Matrix{Float64}(undef, L, H)
    for h in 1:H
        w = mod1(w0 + h, data.W)
        resid .= ar .* resid .+ sd .* randn(L)
        latent[:, h] .= view(seasonal, w, :) .+ resid
    end
    return latent
end

# Read a field from a draw that may be a NamedTuple or a Dict{Symbol}.
_field(draw, key::Symbol) =
    draw isa AbstractDict ? draw[key] : getproperty(draw, key)

"""
    base_project(draw, data, horizons; difference=false)

`project` function bridging one `base_model` posterior draw to the
`(L × maximum(horizons))` latent-scale forecast matrix that
[`forecast_quantiles`](@ref) consumes.

`draw` is a raw sampled-site draw as returned by `posterior_draws(fit)`
(src/inference.jl): it carries `base_model`'s `~` sites, not its derived
return quantities. This function therefore recomputes the derived
quantities from the raw sites, mirroring src/model.jl exactly, so there
is no dependence on generated-quantities extraction. Required fields:
`mu0`, `sigma_season_pop`, `mu_w_raw` (length `W`), `sigma_season_loc`,
`delta_raw` (`W × L`), `sigma_season_time`, `season_eff_raw` (length
`S`), `phi_pop_mean`, `phi_pop_sd`, `phi_raw` (`L`), `mu_log_sigma_ar`,
`tau_log_sigma_ar`, `z_sigma_ar` (`L`), `eps_raw` (`T × L`).

For each horizon `h = 1:maximum(horizons)`:

- **Seasonality** for future week-of-season `w = mod1(data.woy[end] + h,
  data.W)` is rebuilt as `mu0 + mu_w[w] + delta[w, l] + season_eff[s]`,
  with `s = data.season[end]` (the current season's level shift held for
  the short within-season horizons).
- **Residual** is seeded from the last training-week residual (the tail
  of [`ar_or_diff`](@ref) rebuilt from `eps_raw`) and forward-simulated
  with fresh innovations: `resid_h = phi * resid_{h-1} + sigma_ar *
  randn()` for the default AR(1), or `resid_h = resid_{h-1} + sigma_ar *
  randn()` when `difference` is `true`. Fresh (not decayed) innovations
  keep predictive intervals calibrated; each call is a fresh
  posterior-predictive realisation for the draw.

`difference` must match how `base_model` was fit (its default is
`false`); [`forecast_quantiles`](@ref) calls `project` with three
positional arguments, so the default is used in the front-runner path.
No backfill revision is applied to future weeks (revision -> 0), matching
the observed-vs-forecast distinction in docs/contracts.md.
"""
function base_project(draw, data::ModelData, horizons;
                      difference::Bool=false)
    L = data.L
    W = data.W
    H = maximum(horizons)

    # --- Seasonality (mirror src/model.jl transforms exactly) ---
    mu0 = _field(draw, :mu0)
    mu_w_unc = cumsum(_field(draw, :mu_w_raw)) .* _field(draw,
        :sigma_season_pop)
    mu_w = mu_w_unc .- mean(mu_w_unc)                     # (W,)
    delta = _field(draw, :delta_raw) .* _field(draw, :sigma_season_loc)
    season_eff = _field(draw, :season_eff_raw) .*
        _field(draw, :sigma_season_time)                 # (S,)
    seff = season_eff[data.season[end]]

    # --- AR(1) residual dynamics (mirror src/model.jl) ---
    phi = tanh.(_field(draw, :phi_pop_mean) .+
                _field(draw, :phi_pop_sd) .* _field(draw, :phi_raw))
    sigma_ar = exp.(_field(draw, :mu_log_sigma_ar) .+
                    _field(draw, :tau_log_sigma_ar) .*
                    _field(draw, :z_sigma_ar))
    eps_raw = _field(draw, :eps_raw)                     # (T × L)
    resid = Float64[
        ar_or_diff(view(eps_raw, :, l), sigma_ar[l], phi[l], difference)[end]
        for l in 1:L
    ]

    latent = Matrix{Float64}(undef, L, H)
    for h in 1:H
        w = mod1(data.woy[end] + h, W)
        for l in 1:L
            innov = sigma_ar[l] * randn()
            resid[l] = difference ? resid[l] + innov :
                       phi[l] * resid[l] + innov
            latent[l, h] = mu0 + mu_w[w] + delta[w, l] + seff + resid[l]
        end
    end
    return latent
end

"""
    _draws(fit)

Normalise `fit` into a `Vector` of per-draw values that `project` can
consume. If `fit` is already a `Vector` (e.g. in tests, or a
pre-extracted list of draws) it is used as-is. Otherwise `fit` is
assumed to be Tables.jl-compatible (an MCMCChains `Chains` object, a
`DataFrame`, or similar) and each row becomes a `NamedTuple` draw.
"""
function _draws(fit)
    fit isa AbstractVector && return fit
    df = DataFrame(fit)
    return [NamedTuple(row) for row in eachrow(df)]
end

"""
    forecast_quantiles(fit, data, model_id; horizons=HORIZONS,
                        levels=QUANTILE_LEVELS, project=default_project)

Build the hub quantile forecast table (schema in docs/contracts.md)
from posterior draws in `fit`. For each draw, `project(draw, data,
horizons)` returns an `(L × maximum(horizons))` matrix of latent-scale
forecasts; these are mapped to the natural wILI percentage scale with
`from_scale(x, data.transform)`, clamped at 0, and summarised into
`levels` quantiles per location and horizon.

`fit` can be a `Vector` of per-draw `NamedTuple`/`Dict` values, or
anything `DataFrame`-convertible (e.g. an MCMCChains `Chains` object
from `fit_mcmc`/`fit_pathfinder`), in which case each row becomes a
`NamedTuple` draw. Vintage backfill revision is not applied to these
forecasts: future weeks are pure latent-series projections (revision
-> 0), matching the observed-vs-forecast distinction in
docs/contracts.md.

Set `project` to a model-specific function when `src/model.jl`'s draw
field names do not match [`default_project`](@ref)'s expectations.
"""
function forecast_quantiles(
    fit,
    data::ModelData,
    model_id::AbstractString;
    horizons=HORIZONS,
    levels=QUANTILE_LEVELS,
    project::Function=default_project,
)::DataFrame
    draws = _draws(fit)
    hs = collect(horizons)
    H = maximum(hs)
    L = data.L
    nd = length(draws)

    # (L x H x ndraws) posterior-predictive forecasts, natural scale.
    vals = Array{Float64}(undef, L, H, nd)
    for (d, draw) in enumerate(draws)
        latent = project(draw, data, hs)
        @views vals[:, :, d] .= max.(from_scale.(latent, data.transform), 0.0)
    end

    locs = LOCATIONS[1:L]
    n = L * length(hs) * length(levels)
    model_ids = Vector{String}(undef, n)
    locations = Vector{String}(undef, n)
    origin_dates = Vector{Date}(undef, n)
    horizon_col = Vector{Int}(undef, n)
    target_end_dates = Vector{Date}(undef, n)
    targets = Vector{String}(undef, n)
    output_types = Vector{String}(undef, n)
    output_type_ids = Vector{Float64}(undef, n)
    values = Vector{Float64}(undef, n)

    i = 0
    for (li, loc) in enumerate(locs)
        for (hi, h) in enumerate(hs)
            qs = quantile(view(vals, li, hi, :), levels)
            for (qi, lvl) in enumerate(levels)
                i += 1
                model_ids[i] = model_id
                locations[i] = loc
                origin_dates[i] = data.origin_date
                horizon_col[i] = h
                target_end_dates[i] = data.origin_date + Day(7 * h)
                targets[i] = TARGET
                output_types[i] = "quantile"
                output_type_ids[i] = lvl
                values[i] = qs[qi]
            end
        end
    end

    return DataFrame(
        model_id=model_ids,
        location=locations,
        origin_date=origin_dates,
        horizon=horizon_col,
        target_end_date=target_end_dates,
        target=targets,
        output_type=output_types,
        output_type_id=output_type_ids,
        value=values,
    )
end
