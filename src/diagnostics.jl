# Bayesian workflow checks: prior predictive, posterior predictive, and
# residual diagnostics. Every candidate model runs these before scoring
# (docs/brief.md). Kept loosely coupled to src/model.jl and src/inference.jl,
# which are developed in parallel: callers supply a `predict` closure that
# turns one draw into a natural-scale replicate, and a `draws` collection (or
# a `posterior_draws(fit)` function, if one is in scope) rather than this
# file reaching into model or fit internals. See docs/contracts.md.

using DataFrames
using Statistics
using StatsBase

"""
Plausible natural-scale wILI% range used to flag prior/posterior predictive
pathologies. See docs/brief.md.
"""
const PLAUSIBLE_RANGE = (0.0, 15.0)

# -- draw/orientation helpers ------------------------------------------------

"""
    _resolve_draws(fit)

Turn `fit` into a `Vector` of draws. If a `posterior_draws(fit)` function is
in scope (defined by src/inference.jl or src/model.jl), it is used; this is
resolved dynamically (`Base.invokelatest`) so include order does not matter.
Otherwise `fit` itself is treated as a `Vector` of draws (e.g. NamedTuples).
"""
function _resolve_draws(fit)
    if isdefined(@__MODULE__, :posterior_draws)
        return Base.invokelatest(posterior_draws, fit)
    elseif fit isa AbstractVector
        return fit
    else
        error(
            "Cannot resolve draws from `fit` of type $(typeof(fit)); " *
            "pass `draws=...` explicitly or define `posterior_draws(fit)`.",
        )
    end
end

"""
    _subsample(draws, ndraws)

Thin `draws` to at most `ndraws` elements (without replacement), leaving it
unchanged if it is already that size or smaller.
"""
function _subsample(draws::AbstractVector, ndraws::Int)
    length(draws) <= ndraws && return draws
    idx = StatsBase.sample(1:length(draws), ndraws; replace=false)
    return draws[idx]
end

"""
    _as_TxL(rep, data)

Orient a replicate matrix to (T, L), matching `data.Y`. Contracts differ on
whether `predict` returns (T, L) or (L, T); accept either and transpose as
needed rather than depending on which convention model.jl settles on.
"""
function _as_TxL(rep::AbstractMatrix, data::ModelData)
    sz = size(rep)
    sz == (data.T, data.L) && return rep
    sz == (data.L, data.T) && return permutedims(rep)
    throw(DimensionMismatch(
        "predict() returned a $(sz) matrix; expected ($(data.T), " *
        "$(data.L)) (T x L) or ($(data.L), $(data.T)) (L x T).",
    ))
end

"""
    default_predict(draw, data)

Model-agnostic placeholder `predict` used only when no real predictive
function is supplied (e.g. before src/model.jl lands, or in smoke tests). If
`draw` already carries a matrix-like replicate (a bare matrix, or a
NamedTuple field named `Y_rep`/`Yrep`/`y_rep`/`yrep`/`replicate`), that is
used directly. Otherwise falls back to a naive per-location climatology:
each location's mean of its own observed values, repeated over all weeks.
Real usage should pass the fitted model's own `predict`.
"""
function default_predict(draw, data::ModelData)
    if draw isa AbstractMatrix
        return _as_TxL(draw, data)
    end
    if draw isa NamedTuple
        for key in (:Y_rep, :Yrep, :y_rep, :yrep, :replicate)
            haskey(draw, key) && return _as_TxL(getfield(draw, key), data)
        end
    end
    obs_natural = from_scale.(data.Y, data.transform)
    means = map(1:data.L) do l
        vals = collect(skipmissing(view(obs_natural, :, l)))
        isempty(vals) ? 0.0 : mean(vals)
    end
    return repeat(reshape(means, 1, data.L), data.T, 1)
end

# -- prior predictive ---------------------------------------------------------

"""
    prior_predictive(model, data::ModelData; ndraws=200,
        predict=default_predict)

Draw `ndraws` fresh samples from `model`'s prior (each a plain call `model()`,
which resamples any latent site, including entries of `Y` that are `missing`
in `data`), turn each into a natural-scale replicate via `predict(draw,
data)`, and summarise plausibility.

Returns a `NamedTuple`:
- `summary`: pooled `mean`, `sd`, `min`, `max`, 2.5/50/97.5% quantiles of all
  simulated values, the fraction outside `PLAUSIBLE_RANGE` (roughly 0-15%
  wILI), and the fraction non-finite (NaN/Inf pathologies).
- `simulated`: the `Vector` of simulated (T, L) matrices, one per draw.
"""
function prior_predictive(
    model, data::ModelData; ndraws::Int=200, predict=default_predict,
)
    draws = [model() for _ in 1:ndraws]
    sims = [_as_TxL(predict(draw, data), data) for draw in draws]
    pooled = reduce(vcat, vec(s) for s in sims)
    finite = filter(isfinite, pooled)
    n = length(pooled)
    n_nonfinite = n - length(finite)
    lo, hi = PLAUSIBLE_RANGE
    frac_outside = isempty(finite) ? NaN :
        count(v -> v < lo || v > hi, finite) / length(finite)
    summary = (
        n_draws=ndraws,
        n_values=n,
        mean=isempty(finite) ? NaN : mean(finite),
        sd=isempty(finite) ? NaN : std(finite),
        min=isempty(finite) ? NaN : minimum(finite),
        max=isempty(finite) ? NaN : maximum(finite),
        q025=isempty(finite) ? NaN : quantile(finite, 0.025),
        q50=isempty(finite) ? NaN : quantile(finite, 0.50),
        q975=isempty(finite) ? NaN : quantile(finite, 0.975),
        frac_outside_plausible_range=frac_outside,
        frac_nonfinite=n_nonfinite / n,
    )
    return (summary=summary, simulated=sims)
end

# -- posterior predictive -----------------------------------------------------

"""
    posterior_predictive(fit, model, data::ModelData; ndraws=200,
        draws=nothing, predict=default_predict)

Replicate the observed vintage data (`data.Y`, mapped to the natural scale)
from the posterior. `model` is accepted for interface symmetry with
`prior_predictive` and future use, but is not required by the default
`predict`. Draws are `draws` if given, else `_resolve_draws(fit)`
(`posterior_draws(fit)` if defined, else `fit` itself as a `Vector`),
thinned to at most `ndraws`.

Returns a `NamedTuple`:
- `per_observation`: a `DataFrame` with one row per non-missing observed
  cell (`t`, `location`, `week_of_season`, `obs`, `pred_mean`, `lower50`,
  `upper50`, `in50`, `lower90`, `upper90`, `in90`).
- `calibration`: empirical 50% and 90% coverage (`coverage50`, `coverage90`)
  over those cells, plus `n`, for calibration at a glance (should be close
  to 0.5 and 0.9 if well calibrated).
"""
function posterior_predictive(
    fit, model, data::ModelData;
    ndraws::Int=200, draws=nothing, predict=default_predict,
)
    ds = _subsample(draws === nothing ? _resolve_draws(fit) : draws, ndraws)
    sims = [_as_TxL(predict(draw, data), data) for draw in ds]
    obs_natural = from_scale.(data.Y, data.transform)

    rows = NamedTuple[]
    for l in 1:data.L, t in 1:data.T
        obs = obs_natural[t, l]
        ismissing(obs) && continue
        vals = sort([s[t, l] for s in sims])
        pred_mean = mean(vals)
        lower50, upper50 = quantile(vals, 0.25), quantile(vals, 0.75)
        lower90, upper90 = quantile(vals, 0.05), quantile(vals, 0.95)
        push!(rows, (
            t=t,
            location=l <= length(LOCATIONS) ? LOCATIONS[l] : l,
            week_of_season=data.woy[t],
            obs=obs,
            pred_mean=pred_mean,
            lower50=lower50, upper50=upper50, in50=lower50 <= obs <= upper50,
            lower90=lower90, upper90=upper90, in90=lower90 <= obs <= upper90,
        ))
    end
    per_observation = DataFrame(rows)
    n = nrow(per_observation)
    calibration = (
        coverage50=n == 0 ? NaN : mean(per_observation.in50),
        coverage90=n == 0 ? NaN : mean(per_observation.in90),
        n=n,
    )
    return (per_observation=per_observation, calibration=calibration)
end

# -- residual diagnostics -----------------------------------------------------

"""
    residual_summary(fit, model, data::ModelData; ndraws=200, draws=nothing,
        predict=default_predict)

Standardised residuals of observed `Y` (modelling scale) against the
posterior mean and SD of the replicate (predict output mapped back to the
modelling scale via `to_scale`), by location and by week-of-season, plus
residual autocorrelation per location to reveal unmodelled temporal
structure, and a simple linear trend of residuals over time per location.

Returns a `NamedTuple`:
- `by_location`: `DataFrame` (`location`, `mean_resid`, `sd_resid`,
  `trend_slope`, `n`).
- `by_week_of_season`: `DataFrame` (`week_of_season`, `mean_resid`, `n`).
- `autocorrelation`: `DataFrame` (`location`, `lag`, `acf`), lags `1:min(10,
  n_t-1)` per location.
- `residuals`: the (T, L) standardised residual matrix (`missing` where `Y`
  is missing).
"""
function residual_summary(
    fit, model, data::ModelData;
    ndraws::Int=200, draws=nothing, predict=default_predict,
)
    ds = _subsample(draws === nothing ? _resolve_draws(fit) : draws, ndraws)
    sims_scale = [
        to_scale.(_as_TxL(predict(draw, data), data), data.transform)
        for draw in ds
    ]

    resid = Matrix{Union{Missing,Float64}}(missing, data.T, data.L)
    for l in 1:data.L, t in 1:data.T
        obs = data.Y[t, l]
        ismissing(obs) && continue
        vals = [s[t, l] for s in sims_scale]
        pred_sd = std(vals)
        resid[t, l] = pred_sd > 0 ? (obs - mean(vals)) / pred_sd : missing
    end

    by_location_rows = NamedTuple[]
    acf_rows = NamedTuple[]
    for l in 1:data.L
        col = resid[:, l]
        idx = findall(!ismissing, col)
        n = length(idx)
        vals = Float64.(col[idx])
        mean_r = n == 0 ? NaN : mean(vals)
        sd_r = n == 0 ? NaN : std(vals)
        slope = n < 2 ? NaN : _trend_slope(Float64.(idx), vals)
        loc = l <= length(LOCATIONS) ? LOCATIONS[l] : l
        push!(by_location_rows, (
            location=loc, mean_resid=mean_r, sd_resid=sd_r,
            trend_slope=slope, n=n,
        ))
        maxlag = max(0, min(10, n - 1))
        if maxlag >= 1
            acfs = autocor(vals, 1:maxlag)
            for (k, a) in zip(1:maxlag, acfs)
                push!(acf_rows, (location=loc, lag=k, acf=a))
            end
        end
    end
    by_location = DataFrame(by_location_rows)
    autocorrelation = DataFrame(acf_rows)

    woy_groups = Dict{Int,Vector{Float64}}()
    for l in 1:data.L, t in 1:data.T
        r = resid[t, l]
        ismissing(r) && continue
        push!(get!(woy_groups, data.woy[t], Float64[]), r)
    end
    by_week_of_season = DataFrame(
        week_of_season=collect(keys(woy_groups)),
        mean_resid=[mean(v) for v in values(woy_groups)],
        n=[length(v) for v in values(woy_groups)],
    )
    sort!(by_week_of_season, :week_of_season)

    return (
        by_location=by_location,
        by_week_of_season=by_week_of_season,
        autocorrelation=autocorrelation,
        residuals=resid,
    )
end

"""
    _trend_slope(x, y)

Ordinary least squares slope of `y` on `x`, for a simple linear trend check.
"""
function _trend_slope(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    vx = var(x)
    vx == 0 ? NaN : cov(x, y) / vx
end

# -- convenience --------------------------------------------------------------

"""
    bayesian_checks(fit, model, data::ModelData; ndraws=200,
        draws=nothing, predict=default_predict)

Run `prior_predictive`, `posterior_predictive`, and `residual_summary` and
collect their results into one `NamedTuple` (`prior`, `posterior`,
`residuals`), suitable for writing directly into an experiment report.
"""
function bayesian_checks(
    fit, model, data::ModelData;
    ndraws::Int=200, draws=nothing, predict=default_predict,
)
    prior = prior_predictive(model, data; ndraws=ndraws, predict=predict)
    posterior = posterior_predictive(
        fit, model, data; ndraws=ndraws, draws=draws, predict=predict,
    )
    residuals = residual_summary(
        fit, model, data; ndraws=ndraws, draws=draws, predict=predict,
    )
    return (prior=prior, posterior=posterior, residuals=residuals)
end
