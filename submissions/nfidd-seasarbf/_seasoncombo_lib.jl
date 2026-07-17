#!/usr/bin/env julia
# seasonal combination sweep -- simple-round, SEASONAL COMBINATION family.
#
# `nfidd-ar6/generate_forecasts_fourier.jl` tried per-location seasonality
# (3 Fourier harmonic pairs of week-of-season, fit independently for each
# of the 11 locations) and made things WORSE than the plain AR(6)
# baseline: 0.412 vs 0.368 mean WIS (submissions/nfidd-ar6/README.md).
# The likely cause: fitting a location-specific seasonal shape from only
# `window_weeks=104` (two seasons) of that one location's history
# overfits -- docs/eda/03-seasonality.md shows peak timing alone has a
# 5-8 week SD per location, so two seasons of one series barely pin down
# a shape.
#
# This sweep tries a different design: estimate ONE POOLED seasonal
# shape -- a week-of-season climatology shared across all 11 locations
# AND the full ~13-season training history (`data/flu_data_hhs.csv`,
# pre-2015 history plus the two validation seasons) -- then combine that
# shared shape with several different residual/AR mechanisms, per
# location. Pooling over 11 locations x 13 seasons gives ~140
# observations per week-of-season bin instead of ~2, which should make
# the shape itself far more stable, even though it can no longer capture
# any location-specific timing or amplitude quirk on its own (those are
# added back separately, see the amplitude-scaling combo below).
#
# Four seasonal combinations, all built on the SAME pooled seasonal
# shape (`build_seasonal_profile`), varying only the mechanism combined
# with it:
#
#   1. core:      pooled-seasonal + per-location AR(6) + backfill
#                 correction (additive/per-location/median/window=8,
#                 the same design as `seabbs_bot-ar6bf`,
#                 submissions/seabbs_bot-ar6bf/README.md, 0.359 mean
#                 WIS) -- isolates "does pooled seasonality on top of
#                 the best plain-AR baseline help".
#   2. tvar:       pooled-seasonal + a time-varying (discounted /
#                 exponentially-weighted) per-location AR(6) residual,
#                 no backfill correction -- isolates whether allowing
#                 the AR dynamics themselves to drift over the training
#                 window helps once seasonality is removed.
#   3. ridgevar:  pooled-seasonal + a ridge-regularised VAR(p) residual
#                 fit JOINTLY across all 11 locations (one equation per
#                 location, ridge penalty because 11*p+1 regressors is
#                 large next to ~100 training rows), no backfill
#                 correction -- isolates whether cross-location
#                 residual coupling (e.g. a region running a week
#                 ahead/behind its neighbours) adds anything beyond
#                 independent per-location AR(6).
#   4. amp:       the pooled seasonal shape, but scaled per location by
#                 a partially-shrunk amplitude factor (OLS slope of
#                 that location's own deviation on the pooled shape,
#                 shrunk toward 1.0), still with plain per-location
#                 AR(6), no backfill correction -- isolates whether a
#                 SMALL, shrunk per-location adjustment on top of the
#                 shared shape helps without re-introducing the
#                 per-location overfitting that sank the Fourier
#                 variant.
#
# Each combo (2-4) is swept over a small hyperparameter grid; the best
# setting within each family is compared to combo 1 and to each other.
# A bonus combo 5 (`amp+backfill`) then stacks combo 4's winning
# shrinkage on top of combo 1's backfill correction, since nothing about
# either design precludes the other.
# LIGHT + ANALYTIC: CSV/DataFrames/Statistics/LinearAlgebra only, no
# Turing.
#
# Scope: VALIDATION SEASONS (1, 2) ONLY, scored against the local hub
# clone's oracle (docs/contracts.md experimental integrity) -- this is a
# tuning sweep, not a submission driver. The pooled seasonal shape, the
# amplitude scales, and the backfill profile are all estimated only from
# `season_year <= 2016` (pre-2015 history plus the two validation
# seasons), matching the discipline in
# `experiments/simple-round/backfill/generate.jl`: no test-season data
# anywhere.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl
# writes score.txt alongside this file; does not write a hub submission
# (no hub_path argument -- this is exploratory, not a candidate for
# `submissions/`).

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
const HERE = @__DIR__
include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))
include(joinpath(PKG_DIR, "src", "hubio.jl"))
include(joinpath(PKG_DIR, "src", "scoring.jl"))

const TRANSFORM = :fourthroot
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12                 # matches ar6bf's build_model_data Dmax
const WINDOW_WEEKS = 104
const MIN_SUPPORT = 5           # min sample size per profile bin to trust
const SMOOTH_WINDOW = 3         # circular smoothing span for the profile
const VALIDATION_ONLY = (1, 2)
const MAX_TRAIN_SEASON_YEAR = 2016  # pre-2015 history + validation seasons
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")

# Reference backfill design (`seabbs_bot-ar6bf`), reused unchanged for
# the "core" combo only.
const BF_MODE = :additive
const BF_WINDOW = 8
const BF_STAT = :median

# ---------------------------------------------------------------------
# Pooled seasonal shape
# ---------------------------------------------------------------------

"""
    build_seasonal_profile(hist; transform, max_season_year, min_support,
                            smooth_window) -> Dict{Int,Float64}

Pooled week-of-season climatology on the `transform` scale, estimated
ONCE from the full historical series (`data/flu_data_hhs.csv`),
restricted to `season_year(origin_date) <= max_season_year`.

Each location's `wili` is transformed and centred on that location's
OWN mean over the restricted history; the profile at week-of-season `w`
is the mean of these centred values pooled across ALL locations and
matching weeks -- one shared shape estimated from
`11 locations x ~13 seasons` of data, not fit separately per location
(the per-location Fourier fit in
`submissions/nfidd-ar6/generate_forecasts_fourier.jl` overfit this way
and scored worse than plain AR(6): 0.412 vs 0.368). Weeks with fewer
than `min_support` pooled observations fall back to 0.0 (no seasonal
adjustment) before smoothing.

The raw per-week means are then smoothed with a circular moving
average of span `smooth_window` (week 1 and the last week are treated
as adjacent, since week-of-season wraps around the year), and
re-centred to have zero mean across the cycle so that adding the
profile never shifts a location's overall level -- only its within-
year shape.
"""
function build_seasonal_profile(
    hist::DataFrame; transform::Symbol, max_season_year::Int,
    min_support::Int, smooth_window::Int,
)
    h = hist[season_year.(hist.origin_date) .<= max_season_year, :]
    x = to_scale.(h.wili, transform)
    locs = h.location
    woys = week_of_season.(h.origin_date)

    levels = Dict{String,Float64}()
    for loc in unique(locs)
        levels[loc] = mean(x[locs .== loc])
    end
    dev = [x[i] - levels[locs[i]] for i in eachindex(x)]

    Wmax = maximum(woys)
    raw = [Float64[] for _ in 1:Wmax]
    for i in eachindex(dev)
        push!(raw[woys[i]], dev[i])
    end
    means = [length(v) >= min_support ? mean(v) : 0.0 for v in raw]

    half = div(smooth_window - 1, 2)
    smoothed = similar(means)
    for w in 1:Wmax
        idxs = [mod1(w + off, Wmax) for off in (-half):half]
        smoothed[w] = mean(means[idxs])
    end
    smoothed .-= mean(smoothed)

    return Dict(w => smoothed[w] for w in 1:Wmax)
end

"""
    build_amplitude_scales(hist, profile; transform, max_season_year,
                            shrink) -> Vector{Float64}

Per-location amplitude scale for the `amp` combination, in `LOCATIONS`
order. For each location, `b_l` is the no-intercept OLS slope of that
location's own (transform-scale, own-mean-centred) deviation on the
pooled `profile` value at the matching week-of-season, estimated over
the same restricted history as the profile itself. The returned scale
is `1.0 + shrink * (b_l - 1.0)`: `shrink = 0.0` reproduces the
unscaled pooled shape (`amp = 1` everywhere, same as the `core`/`tvar`/
`ridgevar` combos); `shrink = 1.0` uses the raw per-location slope with
no shrinkage (the per-location-overfitting risk this combo is designed
to avoid); intermediate `shrink` partially pools toward the shared
shape.
"""
function build_amplitude_scales(
    hist::DataFrame, profile::Dict{Int,Float64}; transform::Symbol,
    max_season_year::Int, shrink::Float64,
)
    h = hist[season_year.(hist.origin_date) .<= max_season_year, :]
    scales = ones(length(LOCATIONS))
    for (li, loc) in enumerate(LOCATIONS)
        sub = h[h.location .== loc, :]
        isempty(sub) && continue
        x = to_scale.(sub.wili, transform)
        dev = x .- mean(x)
        s = [get(profile, week_of_season(d), 0.0) for d in sub.origin_date]
        denom = sum(abs2, s)
        b = denom > 1e-8 ? sum(dev .* s) / denom : 1.0
        scales[li] = 1.0 + shrink * (b - 1.0)
    end
    return scales
end

"""
    deseasonalize(Y, woy, profile, amp) -> (R, level)

Remove each location's own mean level and the (`amp`-scaled) pooled
seasonal shape from `Y` (T x L, modelling scale), returning the
residual matrix `R` and the per-location `level` used (so callers can
add it back at forecast time). `Y` is assumed complete (no missing) --
true for every `build_model_data` window used here (checked directly:
`window_weeks=104` splits are fully populated).
"""
function deseasonalize(
    Y::AbstractMatrix, woy::Vector{Int}, profile::Dict{Int,Float64},
    amp::Vector{Float64},
)
    T, L = size(Y)
    level = zeros(L)
    for l in 1:L
        level[l] = mean(Float64.(Y[:, l]))
    end
    R = Matrix{Float64}(undef, T, L)
    for l in 1:L, t in 1:T
        s = get(profile, woy[t], 0.0)
        R[t, l] = Y[t, l] - level[l] - amp[l] * s
    end
    return R, level
end

# ---------------------------------------------------------------------
# Backfill correction (identical to `seabbs_bot-ar6bf` / the backfill
# sweep's reference variant; used only by the `core` combo)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support,
                            mode, pooled, stat) -> Dict

Empirical per-`(location, delay)` revision profile, identical in design
to `experiments/simple-round/backfill/generate.jl`'s function of the
same name -- see that file for the full description. Reused here
unchanged (`mode=:additive, pooled=false, stat=:median`) for the `core`
combo only.
"""
function build_revision_profile(
    versions::DataFrame; transform::Symbol, max_delay::Int,
    min_support::Int, mode::Symbol, pooled::Bool, stat::Symbol,
)
    raw = Dict{Any,Vector{Float64}}()
    for g in groupby(versions, [:location, :origin_date])
        settled_idx = argmax(g.as_of)
        settled = to_scale(g.wili[settled_idx], transform)
        settled_as_of = g.as_of[settled_idx]
        loc = g.location[1]
        for row in eachrow(g)
            row.as_of == settled_as_of && continue
            delay = div(Dates.value(row.as_of - row.origin_date), 7)
            (delay < 0 || delay > max_delay) && continue
            vintage = to_scale(row.wili, transform)
            if mode == :multiplicative && abs(vintage) < 1e-6
                continue
            end
            val = mode == :additive ? settled - vintage : settled / vintage
            key = pooled ? delay : (loc, delay)
            push!(get!(raw, key, Float64[]), val)
        end
    end
    profile = Dict{Any,Float64}()
    for (key, vals) in raw
        length(vals) < min_support && continue
        profile[key] = stat == :median ? median(vals) : mean(vals)
    end
    return profile
end

"""
    apply_backfill_correction!(data, profile; mode, pooled, delay_cutoff)

Identical to `experiments/simple-round/backfill/generate.jl`'s function
of the same name: nudges `data.Y` in place using the empirical
revision `profile`.
"""
function apply_backfill_correction!(
    data::ModelData, profile::Dict; mode::Symbol, pooled::Bool,
    delay_cutoff::Int,
)
    for l in 1:data.L, t in 1:data.T
        d = data.delay[t, l]
        (d < 0 || d > delay_cutoff) && continue
        ismissing(data.Y[t, l]) && continue
        key = pooled ? d : (LOCATIONS[l], d)
        haskey(profile, key) || continue
        c = profile[key]
        data.Y[t, l] = mode == :additive ? data.Y[t, l] + c : data.Y[t, l] * c
    end
    return data
end

# ---------------------------------------------------------------------
# Per-location AR(6): plain OLS and discounted (time-varying) WLS
# ---------------------------------------------------------------------

"""
    fit_ar(y, order) -> (coef, resid_sd)

Plain OLS fit of an AR(`order`) model with intercept, identical to
`nfidd-ar6`/`seabbs_bot-ar6bf`.
"""
function fit_ar(y::AbstractVector{Float64}, order::Int)
    n = length(y)
    nobs = n - order
    nobs >= order + 2 ||
        error("series too short for AR($order): n=$n, nobs=$nobs")
    X = ones(nobs, order + 1)
    yresp = Vector{Float64}(undef, nobs)
    for (row, t) in enumerate((order + 1):n)
        yresp[row] = y[t]
        for lag in 1:order
            X[row, lag + 1] = y[t - lag]
        end
    end
    coef = X \ yresp
    resid = yresp .- X * coef
    dof = max(nobs - (order + 1), 1)
    resid_sd = sqrt(sum(abs2, resid) / dof)
    return coef, resid_sd
end

"""
    fit_ar_discounted(y, order, discount) -> (coef, resid_sd)

Weighted (discounted / exponentially-forgetting) least-squares fit of
an AR(`order`) model with intercept: row `t` (ascending in time) gets
weight `discount^(nobs - row)`, so the most recent training row has
weight 1 and older rows are geometrically downweighted. `discount = 1`
reproduces plain OLS (`fit_ar`). Lets the fitted AR dynamics track
recent behaviour more closely than a flat-weighted fit, at the cost of
using less effective history -- the trade-off `discount` sweeps.
"""
function fit_ar_discounted(
    y::AbstractVector{Float64}, order::Int, discount::Float64,
)
    n = length(y)
    nobs = n - order
    nobs >= order + 2 ||
        error("series too short for AR($order): n=$n, nobs=$nobs")
    X = ones(nobs, order + 1)
    yresp = Vector{Float64}(undef, nobs)
    w = Vector{Float64}(undef, nobs)
    for (row, t) in enumerate((order + 1):n)
        yresp[row] = y[t]
        for lag in 1:order
            X[row, lag + 1] = y[t - lag]
        end
        w[row] = discount^(nobs - row)
    end
    Xw = X .* sqrt.(w)
    yw = yresp .* sqrt.(w)
    coef = Xw \ yw
    resid = yresp .- X * coef
    wdof = max(sum(w) - (order + 1), 1.0)
    resid_sd = sqrt(sum(w .* abs2.(resid)) / wdof)
    return coef, resid_sd
end

"""
    simulate_paths(y, coef, resid_sd, order, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Identical to `nfidd-ar6`'s function of the same name: simulate
Gaussian-innovation AR(`order`) sample paths forward from the end of
`y`, for each horizon in `horizons`. Used for both `fit_ar` and
`fit_ar_discounted` fits (same AR(`order`) coefficient layout).
"""
function simulate_paths(
    y::AbstractVector{Float64}, coef::Vector{Float64}, resid_sd::Float64,
    order::Int, horizons, npaths::Int; rng::Random.AbstractRNG,
)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    tail0 = y[(end - order + 1):end]
    for s in 1:npaths
        tail = copy(tail0)
        for h in 1:hmax
            pred = coef[1]
            for lag in 1:order
                pred += coef[lag + 1] * tail[end - lag + 1]
            end
            val = pred + resid_sd * randn(rng)
            if h in horizons
                out[h][s] = val
            end
            push!(tail, val)
            popfirst!(tail)
        end
    end
    return out
end

# ---------------------------------------------------------------------
# Ridge-regularised VAR(p) across locations, fit on the deseasonalized
# residual matrix
# ---------------------------------------------------------------------

"""
    fit_ridge_var(R, p, lambda) -> (Beta, resid_sd)

Per-equation ridge regression VAR(`p`) fit across all `L` columns of
the deseasonalized residual matrix `R` (T x L): each location's column
is regressed on an intercept plus the `p` most recent lags of ALL `L`
columns (`1 + p*L` regressors), ridge-penalised by `lambda` (needed
because `p*L` regressors against ~100 training rows is otherwise
poorly conditioned/overfit for `L=11`). `Beta` is `(1 + p*L) x L`
(column `l` is location `l`'s equation); `resid_sd` is the per-location
in-sample residual standard deviation, used as INDEPENDENT (diagonal)
simulation noise -- the cross-location coupling this combo targets
lives entirely in `Beta`, not in a full noise covariance, to keep the
simulator simple.
"""
function fit_ridge_var(R::AbstractMatrix{Float64}, p::Int, lambda::Float64)
    T, L = size(R)
    nobs = T - p
    ncoef = 1 + p * L
    nobs >= ncoef + 2 ||
        error("series too short for ridge-VAR(p=$p): T=$T, L=$L")
    X = ones(nobs, ncoef)
    Yresp = Matrix{Float64}(undef, nobs, L)
    for (row, t) in enumerate((p + 1):T)
        Yresp[row, :] = R[t, :]
        for k in 1:p
            X[row, (2 + (k - 1) * L):(1 + k * L)] = R[t - k, :]
        end
    end
    reg = X' * X + lambda * I(ncoef)
    Beta = reg \ (X' * Yresp)
    resid = Yresp - X * Beta
    resid_sd = vec(sqrt.(sum(abs2, resid; dims=1) ./ max(nobs - ncoef, 1)))
    return Beta, resid_sd
end

"""
    simulate_var_paths(tail0, Beta, resid_sd, p, L, horizons, npaths; rng)
        -> Dict{Int,Matrix{Float64}}

Simulate `npaths` joint sample paths of the ridge-VAR(`p`) forward from
`tail0` (`tail0[k]` = the residual vector `p - k + 1` steps before the
forecast origin, so `tail0[1]` is the most recent), for each horizon in
`horizons`. Returns, per horizon, an `npaths x L` matrix. Innovations
are independent Normal(0, `resid_sd[l]`) draws per location per step
(see `fit_ridge_var` docstring for why the noise is diagonal).
"""
function simulate_var_paths(
    tail0::Vector{Vector{Float64}}, Beta::Matrix{Float64},
    resid_sd::Vector{Float64}, p::Int, L::Int, horizons, npaths::Int;
    rng::Random.AbstractRNG,
)
    hmax = maximum(horizons)
    out = Dict(h => Matrix{Float64}(undef, npaths, L) for h in horizons)
    for s in 1:npaths
        buf = deepcopy(tail0)
        for h in 1:hmax
            xrow = ones(1 + p * L)
            for k in 1:p
                xrow[(2 + (k - 1) * L):(1 + k * L)] = buf[k]
            end
            pred = vec(xrow' * Beta)
            val = pred .+ resid_sd .* randn(rng, L)
            if h in horizons
                out[h][s, :] = val
            end
            pushfirst!(buf, val)
            pop!(buf)
        end
    end
    return out
end

# ---------------------------------------------------------------------
# Forecast table builder (shared by all four combos, selected by flags)
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, versions_full, profile, amp; kwargs...)
        -> DataFrame

Fit and forecast one seasonal combination for every cross-validation
split of every season in `seasons`. `profile`/`amp` give the pooled
(and optionally per-location-scaled) seasonal shape removed before
fitting and added back at forecast time. Exactly one of the residual
mechanisms is active:

  - default (`var_order=0`, `ar_mode=:plain`): independent per-location
    plain AR(`AR_ORDER`) (`core`/`amp` combos).
  - `ar_mode=:discounted`: independent per-location discounted AR
    (`tvar` combo).
  - `var_order > 0`: joint ridge-VAR(`var_order`) across locations
    (`ridgevar` combo).

`backfill_profile` (if given) is applied via `apply_backfill_correction!`
before deseasonalizing -- used only by the `core` combo.
"""
function build_forecast_table(
    seasons, versions_full, profile::Dict{Int,Float64}, amp::Vector{Float64};
    backfill_profile::Union{Nothing,Dict}=nothing,
    backfill_window::Int=0, ar_mode::Symbol=:plain, discount::Float64=1.0,
    var_order::Int=0, var_lambda::Float64=1.0, model_id::String,
)
    rng = MersenneTwister(SEED)
    rows = DataFrame(
        model_id=String[], location=String[], origin_date=Date[],
        horizon=Int[], target_end_date=Date[], target=String[],
        output_type=String[], output_type_id=Float64[], value=Float64[],
    )
    for season in seasons
        splits = training_splits(
            season; allow_test_season=(season in TEST_SEASONS),
        )
        for split in splits
            data = build_model_data(
                split; Dmax=DMAX, transform=TRANSFORM,
                window_weeks=WINDOW_WEEKS, versions=versions_full,
            )
            if backfill_profile !== nothing
                apply_backfill_correction!(
                    data, backfill_profile; mode=BF_MODE, pooled=false,
                    delay_cutoff=backfill_window,
                )
            end
            R, level = deseasonalize(data.Y, data.woy, profile, amp)
            origin = data.origin_date

            if var_order > 0
                Beta, resid_sd = fit_ridge_var(R, var_order, var_lambda)
                tail0 = [R[end - k + 1, :] for k in 1:var_order]
                paths = simulate_var_paths(
                    tail0, Beta, resid_sd, var_order, data.L, HORIZONS,
                    NPATHS; rng=rng,
                )
                for (li, loc) in enumerate(LOCATIONS)
                    for h in HORIZONS
                        target_end = origin + Day(7 * h)
                        s = get(profile, week_of_season(target_end), 0.0)
                        vals = paths[h][:, li] .+ level[li] .+ amp[li] * s
                        for q in QUANTILE_LEVELS
                            qval = quantile(vals, q)
                            nat = max(from_scale(qval, TRANSFORM), 0.0)
                            push!(rows, (
                                model_id, loc, origin, h, target_end,
                                TARGET, "quantile", q, nat,
                            ))
                        end
                    end
                end
            else
                for (li, loc) in enumerate(LOCATIONS)
                    y = R[:, li]
                    coef, resid_sd = ar_mode == :discounted ?
                        fit_ar_discounted(y, AR_ORDER, discount) :
                        fit_ar(y, AR_ORDER)
                    paths = simulate_paths(
                        y, coef, resid_sd, AR_ORDER, HORIZONS, NPATHS;
                        rng=rng,
                    )
                    for h in HORIZONS
                        target_end = origin + Day(7 * h)
                        s = get(profile, week_of_season(target_end), 0.0)
                        vals = paths[h] .+ level[li] .+ amp[li] * s
                        for q in QUANTILE_LEVELS
                            qval = quantile(vals, q)
                            nat = max(from_scale(qval, TRANSFORM), 0.0)
                            push!(rows, (
                                model_id, loc, origin, h, target_end,
                                TARGET, "quantile", q, nat,
                            ))
                        end
                    end
                end
            end
        end
    end
    return rows
end

"""Hub oracle (`target-data/oracle-output.csv`) as a scoring truth
table."""
function load_oracle(hub_path)
    path = joinpath(hub_path, "target-data", "oracle-output.csv")
    oracle = CSV.read(path, DataFrame)
    truth = DataFrame(
        location=String.(oracle.location),
        target_end_date=Date.(oracle.target_end_date),
        value=Float64.(oracle.oracle_value),
    )
    return dropmissing(truth)
end

score_one(forecast, truth) = wis_summary(score_forecasts(
    forecast, truth; scale=:natural,
))[1, :]

# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------

const DISCOUNTS = (0.95, 0.97, 0.99, 0.995, 1.0)
const VAR_ORDERS = (1, 2)
const VAR_LAMBDAS = (1.0, 5.0, 20.0)
const SHRINKS = (0.0, 0.25, 0.5, 0.75, 1.0)

function main()
    t0 = time()
    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")
    truth = load_oracle(HUB_PATH)
    ones_amp = ones(length(LOCATIONS))

    profile = build_seasonal_profile(
        hist; transform=TRANSFORM, max_season_year=MAX_TRAIN_SEASON_YEAR,
        min_support=MIN_SUPPORT, smooth_window=SMOOTH_WINDOW,
    )

    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= MAX_TRAIN_SEASON_YEAR, :,
    ]
    backfill_profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_WINDOW,
        min_support=5, mode=BF_MODE, pooled=false, stat=BF_STAT,
    )

    # Sanity baseline: no seasonality at all (empty profile -> plain
    # AR(6), no backfill correction) -- reproduces nfidd-ar6 (0.368).
    empty_profile = Dict{Int,Float64}()
    no_season = build_forecast_table(
        VALIDATION_ONLY, versions_full, empty_profile, ones_amp;
        model_id="no-season-baseline",
    )
    no_season_summ = score_one(no_season, truth)
    println("no-season baseline (plain AR(6)): " *
            "mean_wis=$(round(no_season_summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(no_season_summ.sd_wis; digits=4))")

    # --- combo 1: core = pooled-seasonal + AR(6) + backfill ---
    core = build_forecast_table(
        VALIDATION_ONLY, versions_full, profile, ones_amp;
        backfill_profile=backfill_profile, backfill_window=BF_WINDOW,
        model_id="seasoncombo-core",
    )
    core_summ = score_one(core, truth)
    println("combo 1 (core: season+AR6+backfill): " *
            "mean_wis=$(round(core_summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(core_summ.sd_wis; digits=4))")

    # --- combo 2: pooled-seasonal + discounted (time-varying) AR ---
    tvar_results = NamedTuple[]
    for d in DISCOUNTS
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, ones_amp;
            ar_mode=:discounted, discount=d, model_id="seasoncombo-tvar",
        )
        summ = score_one(fc, truth)
        push!(tvar_results, (
            discount=d, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("combo 2 (tvar) discount=$d -> " *
                "mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(tvar_results; by=r -> r.mean_wis)
    tvar_best = tvar_results[1]

    # --- combo 3: pooled-seasonal + ridge-VAR(p) across locations ---
    var_results = NamedTuple[]
    for p in VAR_ORDERS, lambda in VAR_LAMBDAS
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, ones_amp;
            var_order=p, var_lambda=lambda, model_id="seasoncombo-ridgevar",
        )
        summ = score_one(fc, truth)
        push!(var_results, (
            p=p, lambda=lambda, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("combo 3 (ridgevar) p=$p lambda=$lambda -> " *
                "mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(var_results; by=r -> r.mean_wis)
    var_best = var_results[1]

    # --- combo 4: pooled-seasonal + shrunk per-location amplitude ---
    amp_results = NamedTuple[]
    for shrink in SHRINKS
        amp = build_amplitude_scales(
            hist, profile; transform=TRANSFORM,
            max_season_year=MAX_TRAIN_SEASON_YEAR, shrink=shrink,
        )
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, amp;
            model_id="seasoncombo-amp",
        )
        summ = score_one(fc, truth)
        push!(amp_results, (
            shrink=shrink, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("combo 4 (amp) shrink=$shrink -> " *
                "mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(amp_results; by=r -> r.mean_wis)
    amp_best = amp_results[1]

    # --- bonus combo 5: amp's best shrink + backfill together ---
    # combos 1 (backfill) and 4 (amplitude) each help alone; since
    # neither combo's design precludes the other, check whether
    # stacking them (the amp combo's winning shrink on top of the
    # SAME backfill correction as combo 1) beats either alone.
    amp_bf_scale = build_amplitude_scales(
        hist, profile; transform=TRANSFORM,
        max_season_year=MAX_TRAIN_SEASON_YEAR, shrink=amp_best.shrink,
    )
    amp_bf = build_forecast_table(
        VALIDATION_ONLY, versions_full, profile, amp_bf_scale;
        backfill_profile=backfill_profile, backfill_window=BF_WINDOW,
        model_id="seasoncombo-amp-backfill",
    )
    amp_bf_summ = score_one(amp_bf, truth)
    println("bonus combo 5 (amp+backfill) shrink=$(amp_best.shrink) -> " *
            "mean_wis=$(round(amp_bf_summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(amp_bf_summ.sd_wis; digits=4))")

    combos = [
        (name="core", mean_wis=core_summ.mean_wis, sd_wis=core_summ.sd_wis,
         detail="backfill window=$BF_WINDOW ($BF_MODE/per-loc/$BF_STAT)"),
        (name="tvar", mean_wis=tvar_best.mean_wis, sd_wis=tvar_best.sd_wis,
         detail="discount=$(tvar_best.discount)"),
        (name="ridgevar", mean_wis=var_best.mean_wis, sd_wis=var_best.sd_wis,
         detail="p=$(var_best.p) lambda=$(var_best.lambda)"),
        (name="amp", mean_wis=amp_best.mean_wis, sd_wis=amp_best.sd_wis,
         detail="shrink=$(amp_best.shrink)"),
        (name="amp+backfill", mean_wis=amp_bf_summ.mean_wis,
         sd_wis=amp_bf_summ.sd_wis,
         detail="shrink=$(amp_best.shrink), backfill window=$BF_WINDOW " *
                 "($BF_MODE/per-loc/$BF_STAT)"),
    ]
    sort!(combos; by=r -> r.mean_wis)
    winner = combos[1]

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "seasonal combination sweep -- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "reference points:")
        println(io, "  nfidd-ar6 (plain AR6, no season)          = 0.368 " *
                     "(sd 0.471)")
        println(io, "  seabbs_bot-ar6bf (AR6 + backfill)          = 0.359 " *
                     "(sd 0.452)")
        println(io, "  nfidd-ar6 fourier (per-location season)    = 0.412 " *
                     "(sd 0.521, WORSE -- overfit)")
        println(io, "  local no-season sanity rerun (this script) = " *
                     "$(round(no_season_summ.mean_wis; digits=4)) " *
                     "(sd $(round(no_season_summ.sd_wis; digits=4)))")
        println(io)
        println(io, "=== combo 1: core (season + AR6 + backfill) ===")
        println(io, "mean_wis=$(round(core_summ.mean_wis; digits=4)) " *
                     "sd_wis=$(round(core_summ.sd_wis; digits=4)) " *
                     "n_tasks=$(core_summ.n_tasks)")
        println(io)
        println(io, "=== combo 2: tvar (season + discounted AR) sweep ===")
        for r in tvar_results
            println(io, "  discount=$(r.discount) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: discount=$(tvar_best.discount) " *
                     "mean_wis=$(round(tvar_best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(tvar_best.sd_wis; digits=4))")
        println(io)
        println(io, "=== combo 3: ridgevar (season + cross-location " *
                     "ridge-VAR) sweep ===")
        for r in var_results
            println(io, "  p=$(r.p) lambda=$(r.lambda) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: p=$(var_best.p) lambda=$(var_best.lambda) " *
                     "mean_wis=$(round(var_best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(var_best.sd_wis; digits=4))")
        println(io)
        println(io, "=== combo 4: amp (season + shrunk per-location " *
                     "amplitude) sweep ===")
        for r in amp_results
            println(io, "  shrink=$(r.shrink) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: shrink=$(amp_best.shrink) " *
                     "mean_wis=$(round(amp_best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(amp_best.sd_wis; digits=4))")
        println(io)
        println(io, "=== bonus combo 5: amp+backfill (combo 4's best " *
                     "shrink stacked on combo 1's backfill) ===")
        println(io, "shrink=$(amp_best.shrink) backfill window=$BF_WINDOW " *
                     "($BF_MODE/per-loc/$BF_STAT)")
        println(io, "mean_wis=$(round(amp_bf_summ.mean_wis; digits=4)) " *
                     "sd_wis=$(round(amp_bf_summ.sd_wis; digits=4)) " *
                     "n_tasks=$(amp_bf_summ.n_tasks)")
        println(io)
        println(io, "=== overall comparison (best of each combo) ===")
        for r in combos
            println(io, rpad(r.name, 10) *
                         "mean_wis=$(rpad(round(r.mean_wis; digits=4), 8)) " *
                         "sd_wis=$(rpad(round(r.sd_wis; digits=4), 8)) " *
                         r.detail)
        end
        println(io)
        println(io, "=== winner: $(winner.name) ===")
        println(io, "mean_wis=$(round(winner.mean_wis; digits=4)) " *
                     "sd_wis=$(round(winner.sd_wis; digits=4)) " *
                     "($(winner.detail))")
        vs_ref = 0.359 - winner.mean_wis
        vs_pct = 100 * vs_ref / 0.359
        println(io, "vs AR(6)+backfill reference (0.359): " *
                     "$(round(vs_ref; digits=4)) ($(round(vs_pct; digits=2))%)")
    end

    dt = round(time() - t0; digits=1)
    println("\nwinner: $(winner.name) mean_wis=" *
            "$(round(winner.mean_wis; digits=4)) " *
            "sd_wis=$(round(winner.sd_wis; digits=4))")
    println("wrote score.txt in $(dt)s total")
    return combos
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
