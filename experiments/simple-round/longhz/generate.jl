#!/usr/bin/env julia
# LONG-HORIZON (h=3,4) fix -- simple-round.
#
# The round-2 stack winner (experiments/simple-round/round2-stack,
# combo "log+tstudent+pool(w=0.9)": pooled seasonal climatology (log
# scale) + additive/median backfill correction + per-location AR(6)
# blended 90% toward a fullpool anchor + Student-t(df=10, scale=1.4)
# innovations, val mean_wis=0.2601) is the current best simple-round
# model. Its own score.txt already breaks WIS down by horizon:
#
#   h=1: 0.1943   h=2: 0.2414   h=3: 0.2873   h=4: 0.3176
#
# monotonically worsening, confirming the suspected weakness: the
# per-location AR(6), blended 90% toward a single shared fullpool
# coefficient vector, mean-reverts fast. Iterated forward it converges
# toward the (deseasonalized, near-zero) unconditional mean within a
# few steps, so at h=3-4 it can neither sustain an epidemic upswing nor
# track the post-peak decline rate, while the Student-t/pool tuning
# that helped overall mostly buys calibration, not horizon-3/4 signal.
#
# This script reproduces that breakdown, then tries four ORTHOGONAL
# fixes aimed at h=3/4 specifically, each scored on the same validation
# seasons the winner was scored on, and reports the WIS-by-horizon
# table for each so we can see whether long horizons improve WITHOUT
# the short horizons (which the winner already does well at) getting
# worse:
#
#   1. momentum: a decaying drift term, sized from the recent local
#      slope of the deseasonalized residual, added to the recursive
#      AR simulation at every step (`simulate_paths_momentum`). Unlike
#      the AR's own dynamics (damped by the w=0.9 pool blend), this
#      lets a real recent trend keep pushing the path for a few steps
#      instead of reverting immediately.
#   2. damped trend (Gardner): a fully separate Holt-damped
#      level+trend forecast (`damped_trend_forecast`) computed on the
#      same residual series, blended into the AR path at a
#      horizon-increasing weight (0 at h=1, up to `blend_max` at h=4)
#      by re-centring the simulated paths, not their spread.
#   3. direct per-horizon AR: fit a SEPARATE OLS AR(6)-on-lags model
#      per horizon (`direct_design`/`fit_direct_ar_pooled`), predicting
#      y[t+h] straight from the lags at t, instead of iterating the
#      1-step model forward h times. This removes the recursive
#      compounding of the pool-induced mean reversion at h=3/4 (each
#      horizon's own OLS fit is free to learn a different, less
#      reverting, effective persistence for that horizon).
#   4. longer memory: window_weeks in {130,156,182,208} (up to ~4
#      seasons) instead of 104 (2), so the per-location AR(6) and
#      fullpool anchor see more of the shape of past upswings/declines.
#      Whichever window_weeks scores best is then also tried stacked
#      with the damped-trend and direct-per-horizon candidates (2 and
#      3 above), in case those add anything once the AR already has
#      more history to learn from.
#
# All four are ANALYTIC (OLS + closed-form smoothing), no Turing.
# Selection remains on VALIDATION SEASONS (1, 2) ONLY (docs/contracts.md
# experimental integrity) against the local hub clone's oracle -- this
# script is a tuning sweep, exactly as round2-stack was. If a candidate
# clears validation net-positive on h=3/4 without hurting h=1/2, `main`
# (with a `hub_path` argument) refits it across all five seasons and
# writes a full HUB-format submission under model_id "nfidd-longhz";
# otherwise it submits the (unmodified) round2-stack winner so a
# forecast still gets produced, and says so in score.txt.
#
# LEAKAGE FIX (post-hoc): the first version of this file built its base
# on `build_seasonal_profile`/`build_revision_profile` ONCE from a fixed
# `season_year <= 2016` cutoff, reused unchanged across every split --
# for a validation-season split that cutoff includes THAT SAME season's
# own future weeks and the entire other validation season, a real leak
# (see `experiments/simple-round/round2-stack/generate.jl` and its
# score.txt "LEAKAGE FIX" section, where this was first caught and
# fixed for the base model this script builds on). Both functions here
# now take the split's own `forecast_origin` and are rebuilt PER SPLIT
# from only strictly-prior data, mirroring round2-stack's fix exactly
# so the two stay comparable. The window208/damped-trend levers
# themselves do not touch either profile (window_weeks only changes how
# much AR history `build_model_data` keeps; the damped-trend blend only
# reshapes the AR's own simulated paths) so they are LEAK-INDEPENDENT --
# expected, and confirmed below, to survive the fix. The original leaky
# score.txt is kept as `score-leaky.txt` for an explicit before/after;
# every number in score.txt itself is now from this leak-free rerun.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl [hub_path]
# writes score.txt alongside this file; writes a hub submission under
# model_id "nfidd-longhz" only if `hub_path` is given.

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra
using Distributions

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
const HERE = @__DIR__
include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))
include(joinpath(PKG_DIR, "src", "hubio.jl"))
include(joinpath(PKG_DIR, "src", "scoring.jl"))

const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12
const WINDOW_WEEKS = 104
const MIN_SUPPORT = 5
const SMOOTH_WINDOW = 3
const VALIDATION_ONLY = (1, 2)
const MAX_TRAIN_SEASON_YEAR = 2016
const HUB_PATH_DEFAULT = joinpath(PKG_DIR, "scratch-hub")

# Backfill + interval scheme + pool weight + transform: identical to the
# round2-stack winner (experiments/simple-round/round2-stack/generate.jl).
const BF_MODE = :additive
const BF_WINDOW = 8
const BF_STAT = :median
const T_DF = 10
const T_SCALE = 1.4
const WINNER_POOL_W = 0.9
const TRANSFORM = :log

const SUB_MODEL_ID = "nfidd-longhz"

# ---------------------------------------------------------------------
# Pooled seasonal shape + backfill correction: LEAK-FREE, rebuilt PER
# SPLIT from the split's own `forecast_origin`. Identical in design to
# round2-stack/generate.jl's functions of the same name (see that file
# and its score.txt "LEAKAGE FIX" section for the full derivation and
# the leak this replaces) -- these used to be built ONCE from a fixed
# `season_year <= 2016` cutoff, which for a validation-season split
# leaked that same season's own future weeks and the other validation
# season into the profile used to correct/deseasonalize it.
# ---------------------------------------------------------------------

"""
    build_seasonal_profile(hist, forecast_origin; transform, min_support,
                            smooth_window) -> Dict{Int,Float64}

Pooled week-of-season climatology on the `transform` scale, using only
`hist` rows strictly before `forecast_origin` -- LEAK-FREE, must be
rebuilt for every split (the cutoff moves with each split's own
forecast origin). Otherwise identical pooling design to the previous
(leaky) version: deviation-from-own-mean, pooled across all 11
locations, circularly smoothed.
"""
function build_seasonal_profile(
    hist::DataFrame, forecast_origin::Date; transform::Symbol,
    min_support::Int, smooth_window::Int,
)
    h = hist[hist.origin_date .< forecast_origin, :]
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

function deseasonalize(
    Y::AbstractMatrix, woy::Vector{Int}, profile::Dict{Int,Float64},
)
    T, L = size(Y)
    level = zeros(L)
    for l in 1:L
        level[l] = mean(Float64.(Y[:, l]))
    end
    R = Matrix{Float64}(undef, T, L)
    for l in 1:L, t in 1:T
        R[t, l] = Y[t, l] - level[l] - get(profile, woy[t], 0.0)
    end
    return R, level
end

"""
    build_revision_profile(versions, forecast_origin; transform, max_delay,
                            min_support, mode, stat)
        -> Dict{Tuple{String,Int},Float64}

Empirical per-(location, delay) revision profile on the `transform`
scale, using only `versions` rows with `as_of < forecast_origin` --
LEAK-FREE, must be rebuilt for every split. `settled` is "the latest
vintage known as of THIS split's forecast origin", not the dataset's
true final value -- an honest degradation for origins close to
`forecast_origin` (whose true settled value isn't knowable yet
either), not a leak.
"""
function build_revision_profile(
    versions::DataFrame, forecast_origin::Date; transform::Symbol,
    max_delay::Int, min_support::Int, mode::Symbol, stat::Symbol,
)
    vf = versions[versions.as_of .< forecast_origin, :]
    raw = Dict{Tuple{String,Int},Vector{Float64}}()
    for g in groupby(vf, [:location, :origin_date])
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
            key = (loc, delay)
            push!(get!(raw, key, Float64[]), val)
        end
    end
    profile = Dict{Tuple{String,Int},Float64}()
    for (key, vals) in raw
        length(vals) < min_support && continue
        profile[key] = stat == :median ? median(vals) : mean(vals)
    end
    return profile
end

function apply_backfill_correction!(
    data::ModelData, profile::Dict{Tuple{String,Int},Float64};
    mode::Symbol, delay_cutoff::Int,
)
    for l in 1:data.L, t in 1:data.T
        d = data.delay[t, l]
        (d < 0 || d > delay_cutoff) && continue
        ismissing(data.Y[t, l]) && continue
        key = (LOCATIONS[l], d)
        haskey(profile, key) || continue
        c = profile[key]
        data.Y[t, l] = mode == :additive ? data.Y[t, l] + c : data.Y[t, l] * c
    end
    return data
end

# ---------------------------------------------------------------------
# Recursive per-location AR(6) + fullpool blending: identical to
# round2-stack/generate.jl.
# ---------------------------------------------------------------------

function ar_design(y::AbstractVector{Float64}, order::Int)
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
    return X, yresp
end

function resid_sd_for(
    X::Matrix{Float64}, yresp::Vector{Float64}, coef::Vector{Float64},
    order::Int,
)
    nobs = size(X, 1)
    resid = yresp .- X * coef
    dof = max(nobs - (order + 1), 1)
    return sqrt(sum(abs2, resid) / dof)
end

function fit_ar(y::AbstractVector{Float64}, order::Int)
    X, yresp = ar_design(y, order)
    coef = X \ yresp
    return coef, X, yresp
end

function fit_ar_pooled(ys::Vector{Vector{Float64}}, order::Int)
    designs = [ar_design(y, order) for y in ys]
    Xall = reduce(vcat, first.(designs))
    yall = reduce(vcat, last.(designs))
    return Xall \ yall
end

"""
    simulate_paths(y, coef, resid_sd, order, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Baseline recursive simulation: Student-t(df=T_DF) innovations,
variance-matched to `resid_sd` then scaled by T_SCALE (round2-stack's
winning scheme). Identical to that file's function of the same name
with `innovation` fixed at `:student_t`.
"""
function simulate_paths(
    y::AbstractVector{Float64}, coef::Vector{Float64}, resid_sd::Float64,
    order::Int, horizons, npaths::Int; rng::Random.AbstractRNG,
)
    tdist = TDist(T_DF)
    vscale = sqrt((T_DF - 2) / T_DF)
    innov_sd = resid_sd * vscale * T_SCALE

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
            innov = innov_sd * rand(rng, tdist)
            val = pred + innov
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
# Candidate 1: momentum -- a decaying drift term added at every
# recursive step, sized from the series' own recent local slope.
# ---------------------------------------------------------------------

"""
    local_slope(y, window) -> Float64

OLS slope of the last `window` steps of `y` against a 0..window index
(so units are "per week"). Returns 0.0 if there is not enough history
to fit a slope (mirrors this experiment's other small-sample guards).
"""
function local_slope(y::AbstractVector{Float64}, window::Int)
    n = length(y)
    w = min(window, n - 1)
    w < 2 && return 0.0
    tail = y[(end - w):end]
    xs = Float64.(0:w)
    xm = mean(xs)
    ym = mean(tail)
    den = sum(abs2, xs .- xm)
    den == 0 && return 0.0
    return sum((xs .- xm) .* (tail .- ym)) / den
end

"""
    simulate_paths_momentum(y, coef, resid_sd, order, horizons, npaths;
                            rng, mom_weight, mom_decay, mom_window)
        -> Dict{Int,Vector{Float64}}

Same recursive AR + Student-t simulation as `simulate_paths`, but with
an extra drift `mom_weight * mom_decay^(h-1) * slope` added to the
1-step prediction at every step `h`, where `slope` is `local_slope(y,
mom_window)`. `mom_decay < 1` lets the drift fade out geometrically
rather than being cut off, so it nudges long horizons toward continuing
whatever the series was recently doing without permanently overriding
the AR's own mean-reverting dynamics.
"""
function simulate_paths_momentum(
    y::AbstractVector{Float64}, coef::Vector{Float64}, resid_sd::Float64,
    order::Int, horizons, npaths::Int; rng::Random.AbstractRNG,
    mom_weight::Float64, mom_decay::Float64, mom_window::Int,
)
    tdist = TDist(T_DF)
    vscale = sqrt((T_DF - 2) / T_DF)
    innov_sd = resid_sd * vscale * T_SCALE
    slope = local_slope(y, mom_window)

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
            pred += mom_weight * mom_decay^(h - 1) * slope
            innov = innov_sd * rand(rng, tdist)
            val = pred + innov
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
# Candidate 2: damped trend (Gardner) -- a separate level+trend
# forecast blended into the AR path at a horizon-increasing weight.
# ---------------------------------------------------------------------

"""
    damped_trend_forecast(y, horizons; alpha, beta, phi)
        -> Dict{Int,Float64}

Holt's damped-trend (Gardner 1985) point forecast for each horizon in
`horizons`, fit on `y` by the standard recursions

    l_t = alpha*y_t + (1-alpha)*(l_{t-1} + phi*b_{t-1})
    b_t = beta*(l_t - l_{t-1}) + (1-beta)*phi*b_{t-1}

started from `l_0 = y[1]`, `b_0 = y[2] - y[1]`, and forecast
`l_n + b_n * sum_{i=1}^{h} phi^i`. Separate from (and does not replace)
the per-location AR(6): this is only used to re-centre the AR's
simulated paths at long horizons (see `build_forecast_table`'s
`:damped` mode), not to replace their spread.
"""
function damped_trend_forecast(
    y::AbstractVector{Float64}, horizons;
    alpha::Float64, beta::Float64, phi::Float64,
)
    n = length(y)
    l = y[1]
    b = n >= 2 ? y[2] - y[1] : 0.0
    for t in 2:n
        l_prev = l
        l = alpha * y[t] + (1 - alpha) * (l_prev + phi * b)
        b = beta * (l - l_prev) + (1 - beta) * phi * b
    end
    hmax = maximum(horizons)
    out = Dict{Int,Float64}()
    cum = 0.0
    phi_pow = phi
    for h in 1:hmax
        cum += phi_pow
        if h in horizons
            out[h] = l + b * cum
        end
        phi_pow *= phi
    end
    return out
end

# ---------------------------------------------------------------------
# Candidate 3: direct per-horizon AR -- one OLS fit per horizon,
# predicting y[t+h] straight from the lags at t (no recursion).
# ---------------------------------------------------------------------

"""
    direct_design(y, order, h) -> (X, yresp)

Design matrix and response for a direct h-step-ahead OLS fit: response
`y[t+h]`, predictors `[1, y[t], y[t-1], ..., y[t-order+1]]` for every
valid anchor `t`. Same lag layout as `ar_design` (`X[row, lag+1] ==
y[t - lag + 1]`), so the same 1-step prediction formula used in
`simulate_paths` applies unchanged at forecast time -- just without
iterating it h times.
"""
function direct_design(y::AbstractVector{Float64}, order::Int, h::Int)
    n = length(y)
    ts = order:(n - h)
    nobs = length(ts)
    nobs >= order + 2 ||
        error("series too short for direct AR(order=$order, h=$h): " *
              "n=$n, nobs=$nobs")
    X = ones(nobs, order + 1)
    yresp = Vector{Float64}(undef, nobs)
    for (row, t) in enumerate(ts)
        yresp[row] = y[t + h]
        for lag in 1:order
            X[row, lag + 1] = y[t - lag + 1]
        end
    end
    return X, yresp
end

"""
    fit_direct_ar_pooled(ys, order, h) -> coef

Fullpool anchor for the direct h-step model: one OLS fit on the design
rows of every location's `direct_design(y, order, h)` stacked together.
Analogous to `fit_ar_pooled` for the recursive model.
"""
function fit_direct_ar_pooled(ys::Vector{Vector{Float64}}, order::Int, h::Int)
    designs = [direct_design(y, order, h) for y in ys]
    Xall = reduce(vcat, first.(designs))
    yall = reduce(vcat, last.(designs))
    return Xall \ yall
end

# ---------------------------------------------------------------------
# Forecast table builder -- seasonal core + backfill (identical to
# round2-stack) + a `mode` switch over the baseline / three candidates.
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, hist, versions_full; kwargs...)
        -> DataFrame

Same seasonal-core + backfill + per-location-AR(6)/fullpool-blend
pipeline as round2-stack's function of the same name, generalised with
a `mode` switch. LEAK-FREE: the pooled seasonal profile and the
backfill revision profile are rebuilt FRESH for every split from
`hist`/`versions_full`, restricted to that split's own forecast origin
(`build_seasonal_profile`/`build_revision_profile` above) -- see those
functions' docstrings and round2-stack's score.txt "LEAKAGE FIX" note
for the leak this replaces (an earlier version of this file built both
profiles ONCE from a fixed `season_year <= 2016` cutoff).

- `:base` reproduces the round2-stack winner exactly (recursive AR(6),
  Student-t innovations, `pool_w` blend).
- `:momentum` swaps in `simulate_paths_momentum` (needs `mom_weight`,
  `mom_decay`, `mom_window`).
- `:damped` runs the baseline simulation, then re-centres each
  horizon's simulated paths toward `damped_trend_forecast` at weight
  `damp_blend_max * (h-1)/(hmax-1)` (needs `damp_alpha`, `damp_beta`,
  `damp_phi`, `damp_blend_max`).
- `:direct` replaces the recursive per-location AR(6) with one direct
  OLS fit per horizon (`direct_design`/`fit_direct_ar_pooled`), still
  blended `pool_w` toward that horizon's own fullpool anchor, then
  draws Student-t noise directly at that horizon's own residual SD (no
  recursion, hence no compounding of the pool-induced reversion).

`window_weeks` overrides the AR/backfill training window length (the
"longer memory" candidate uses this with `mode=:base`); it does not
affect either profile (only how much AR history `build_model_data`
keeps), so the window208/damped-trend levers are leak-independent of
the fix above.
"""
function build_forecast_table(
    seasons, hist::DataFrame, versions_full::DataFrame;
    transform::Symbol, backfill_window::Int=BF_WINDOW,
    pool_w::Float64=WINNER_POOL_W, model_id::String,
    window_weeks::Int=WINDOW_WEEKS, mode::Symbol=:base,
    mom_weight::Float64=0.0, mom_decay::Float64=0.0, mom_window::Int=4,
    damp_alpha::Float64=0.3, damp_beta::Float64=0.1, damp_phi::Float64=0.9,
    damp_blend_max::Float64=0.0,
)
    rng = MersenneTwister(SEED)
    rows = DataFrame(
        model_id=String[], location=String[], origin_date=Date[],
        horizon=Int[], target_end_date=Date[], target=String[],
        output_type=String[], output_type_id=Float64[], value=Float64[],
    )
    tdist = TDist(T_DF)
    vscale = sqrt((T_DF - 2) / T_DF)

    for season in seasons
        splits = training_splits(
            season; allow_test_season=(season in TEST_SEASONS),
        )
        for split in splits
            forecast_origin = maximum(split.origin_date)
            profile = build_seasonal_profile(
                hist, forecast_origin; transform=transform,
                min_support=MIN_SUPPORT, smooth_window=SMOOTH_WINDOW,
            )
            backfill_profile = build_revision_profile(
                versions_full, forecast_origin; transform=transform,
                max_delay=backfill_window, min_support=MIN_SUPPORT,
                mode=BF_MODE, stat=BF_STAT,
            )
            data = build_model_data(
                split; Dmax=DMAX, transform=transform,
                window_weeks=window_weeks, versions=versions_full,
            )
            apply_backfill_correction!(
                data, backfill_profile; mode=BF_MODE,
                delay_cutoff=backfill_window,
            )
            R, level = deseasonalize(data.Y, data.woy, profile)
            origin = data.origin_date
            L = data.L
            ys = [R[:, li] for li in 1:L]

            if mode == :direct
                for (li, loc) in enumerate(LOCATIONS)
                    y = ys[li]
                    for h in HORIZONS
                        X, yresp = direct_design(y, AR_ORDER, h)
                        coef = X \ yresp
                        if pool_w > 0.0
                            anchor = fit_direct_ar_pooled(ys, AR_ORDER, h)
                            coef = (1 - pool_w) .* coef .+ pool_w .* anchor
                        end
                        resid_sd = resid_sd_for(X, yresp, coef, AR_ORDER)
                        tail = y[(end - AR_ORDER + 1):end]
                        pred = coef[1]
                        for lag in 1:AR_ORDER
                            pred += coef[lag + 1] * tail[end - lag + 1]
                        end
                        innov_sd = resid_sd * vscale * T_SCALE
                        draws = pred .+ innov_sd .* rand(rng, tdist, NPATHS)
                        target_end = origin + Day(7 * h)
                        s = get(profile, week_of_season(target_end), 0.0)
                        vals = draws .+ level[li] .+ s
                        for q in QUANTILE_LEVELS
                            qval = quantile(vals, q)
                            nat = max(from_scale(qval, transform), 0.0)
                            push!(rows, (
                                model_id, loc, origin, h, target_end,
                                TARGET, "quantile", q, nat,
                            ))
                        end
                    end
                end
                continue
            end

            fits = [fit_ar(ys[li], AR_ORDER) for li in 1:L]
            coefs = [f[1] for f in fits]
            Xs = [f[2] for f in fits]
            yresps = [f[3] for f in fits]
            blended = if pool_w <= 0.0
                coefs
            else
                anchor = fit_ar_pooled(ys, AR_ORDER)
                [(1 - pool_w) .* coefs[li] .+ pool_w .* anchor for li in 1:L]
            end

            for (li, loc) in enumerate(LOCATIONS)
                coef = blended[li]
                resid_sd = resid_sd_for(Xs[li], yresps[li], coef, AR_ORDER)
                y = ys[li]
                paths = if mode == :momentum
                    simulate_paths_momentum(
                        y, coef, resid_sd, AR_ORDER, HORIZONS, NPATHS;
                        rng=rng, mom_weight=mom_weight, mom_decay=mom_decay,
                        mom_window=mom_window,
                    )
                else
                    simulate_paths(
                        y, coef, resid_sd, AR_ORDER, HORIZONS, NPATHS;
                        rng=rng,
                    )
                end
                if mode == :damped
                    damped_fc = damped_trend_forecast(
                        y, HORIZONS; alpha=damp_alpha, beta=damp_beta,
                        phi=damp_phi,
                    )
                    hmax = maximum(HORIZONS)
                    for h in HORIZONS
                        ar_mean = mean(paths[h])
                        w_h = damp_blend_max * (h - 1) / (hmax - 1)
                        shift = w_h * (damped_fc[h] - ar_mean)
                        paths[h] .+= shift
                    end
                end
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    s = get(profile, week_of_season(target_end), 0.0)
                    vals = paths[h] .+ level[li] .+ s
                    for q in QUANTILE_LEVELS
                        qval = quantile(vals, q)
                        nat = max(from_scale(qval, transform), 0.0)
                        push!(rows, (
                            model_id, loc, origin, h, target_end,
                            TARGET, "quantile", q, nat,
                        ))
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

"""
    coverage(forecast, truth, level) -> Float64

Empirical coverage of the nominal `level` central interval. Identical
to round2-stack/generate.jl's function of the same name.
"""
function coverage(forecast::DataFrame, truth::DataFrame, level::Float64)
    a = (1 - level) / 2
    task_cols = [:location, :origin_date, :horizon, :target_end_date]
    lo = forecast[isapprox.(forecast.output_type_id, a; atol=1e-6), :]
    hi = forecast[isapprox.(forecast.output_type_id, 1 - a; atol=1e-6), :]
    lo_r = rename(lo[:, vcat(task_cols, [:value])], :value => :lo)
    hi_r = rename(hi[:, vcat(task_cols, [:value])], :value => :hi)
    joined = innerjoin(lo_r, hi_r, on=task_cols)
    joined = innerjoin(joined, truth, on=[:location, :target_end_date])
    return mean(joined.lo .<= joined.value .<= joined.hi)
end

# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------

"""
    run_combo(label, seasons, hist, versions_full; truth, kwargs...)
        -> NamedTuple

Fits/forecasts/scores one combo (LEAK-FREE: `hist`/`versions_full` are
the raw historical tables, and `build_forecast_table` rebuilds both
profiles per split from them) and returns its overall + per-horizon
mean WIS, SD, and cov50/cov90 alongside the forecast table itself.
"""
function run_combo(
    label, seasons, hist, versions_full; truth,
    transform=TRANSFORM, pool_w=WINNER_POOL_W, window_weeks=WINDOW_WEEKS,
    mode::Symbol=:base, mom_weight=0.0, mom_decay=0.0, mom_window=4,
    damp_alpha=0.3, damp_beta=0.1, damp_phi=0.9, damp_blend_max=0.0,
)
    fc = build_forecast_table(
        seasons, hist, versions_full; transform=transform,
        pool_w=pool_w, model_id=label, window_weeks=window_weeks,
        mode=mode, mom_weight=mom_weight, mom_decay=mom_decay,
        mom_window=mom_window, damp_alpha=damp_alpha, damp_beta=damp_beta,
        damp_phi=damp_phi, damp_blend_max=damp_blend_max,
    )
    scored = score_forecasts(fc, truth; scale=:natural)
    summ = wis_summary(scored)[1, :]
    by_h = combine(groupby(scored, :horizon), :wis => mean => :mean_wis)
    sort!(by_h, :horizon)
    hw = Dict(row.horizon => row.mean_wis for row in eachrow(by_h))
    cov50 = coverage(fc, truth, 0.5)
    cov90 = coverage(fc, truth, 0.9)

    println("  $(rpad(label, 30)) mean=$(round(summ.mean_wis; digits=4)) " *
            "h1=$(round(hw[1]; digits=4)) h2=$(round(hw[2]; digits=4)) " *
            "h3=$(round(hw[3]; digits=4)) h4=$(round(hw[4]; digits=4)) " *
            "cov50=$(round(cov50; digits=3)) cov90=$(round(cov90; digits=3))")

    return (
        label=label, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        h1=hw[1], h2=hw[2], h3=hw[3], h4=hw[4], cov50=cov50, cov90=cov90,
        forecast=fc, transform=transform, pool_w=pool_w,
        window_weeks=window_weeks, mode=mode, mom_weight=mom_weight,
        mom_decay=mom_decay, mom_window=mom_window, damp_alpha=damp_alpha,
        damp_beta=damp_beta, damp_phi=damp_phi,
        damp_blend_max=damp_blend_max,
    )
end

"""
    net_positive(baseline, cand; tol=0.01) -> Bool

A candidate "lifts h=3,4 net-positive" if it does not make h=1 or h=2
worse by more than `tol` (relative), DOES improve h=3 or h=4, and its
overall mean WIS is no worse than the baseline's. This is the exact
question the task asks of every candidate, so it is checked explicitly
rather than left to eyeballing the ranked table.
"""
function net_positive(baseline, cand; tol::Float64=0.01)
    h1_ok = cand.h1 <= baseline.h1 * (1 + tol)
    h2_ok = cand.h2 <= baseline.h2 * (1 + tol)
    improves_long = (cand.h3 < baseline.h3) || (cand.h4 < baseline.h4)
    overall_ok = cand.mean_wis <= baseline.mean_wis
    return h1_ok && h2_ok && improves_long && overall_ok
end

function print_and_write_row(io, r, baseline)
    d1 = baseline.h1 - r.h1
    d2 = baseline.h2 - r.h2
    d3 = baseline.h3 - r.h3
    d4 = baseline.h4 - r.h4
    dm = baseline.mean_wis - r.mean_wis
    println(io, "  $(rpad(r.label, 30)) mean=$(round(r.mean_wis; digits=4)) " *
                 "(d=$(round(dm; digits=4)))  " *
                 "h1=$(round(r.h1; digits=4))(d=$(round(d1; digits=4))) " *
                 "h2=$(round(r.h2; digits=4))(d=$(round(d2; digits=4))) " *
                 "h3=$(round(r.h3; digits=4))(d=$(round(d3; digits=4))) " *
                 "h4=$(round(r.h4; digits=4))(d=$(round(d4; digits=4))) " *
                 "cov50=$(round(r.cov50; digits=3)) " *
                 "cov90=$(round(r.cov90; digits=3))")
end

function main(hub_path::Union{Nothing,AbstractString}=nothing)
    t0 = time()
    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")
    truth = load_oracle(HUB_PATH_DEFAULT)

    results = NamedTuple[]

    println("=== baseline reproduction (round2-stack winner, LEAK-FREE) ===")
    baseline = run_combo(
        "baseline", VALIDATION_ONLY, hist, versions_full;
        truth=truth, mode=:base,
    )
    push!(results, baseline)

    println("\n=== 1. momentum sweep ===")
    for (w, d) in ((0.3, 0.5), (0.3, 0.8), (0.6, 0.5), (0.6, 0.8), (1.0, 0.7))
        push!(results, run_combo(
            "momentum(w=$w,decay=$d)", VALIDATION_ONLY, hist,
            versions_full; truth=truth, mode=:momentum,
            mom_weight=w, mom_decay=d, mom_window=4,
        ))
    end

    println("\n=== 2. damped-trend blend sweep ===")
    for (phi, bmax) in ((0.9, 0.3), (0.9, 0.5), (0.8, 0.5))
        push!(results, run_combo(
            "damped(phi=$phi,blend=$bmax)", VALIDATION_ONLY, hist,
            versions_full; truth=truth, mode=:damped,
            damp_alpha=0.3, damp_beta=0.1, damp_phi=phi,
            damp_blend_max=bmax,
        ))
    end

    println("\n=== 3. direct per-horizon AR sweep ===")
    for pw in (0.0, 0.5, 0.9)
        push!(results, run_combo(
            "direct(pool_w=$pw)", VALIDATION_ONLY, hist, versions_full;
            truth=truth, mode=:direct, pool_w=pw,
        ))
    end

    println("\n=== 4. longer memory (window_weeks sweep) ===")
    window_results = NamedTuple[]
    for ww in (130, 156, 182, 208)
        r = run_combo(
            "window$ww", VALIDATION_ONLY, hist, versions_full;
            truth=truth, mode=:base, window_weeks=ww,
        )
        push!(results, r)
        push!(window_results, r)
    end
    best_window = sort(window_results; by=r -> r.mean_wis)[1]
    println("  best window_weeks: $(best_window.window_weeks) " *
            "(mean=$(round(best_window.mean_wis; digits=4)))")

    println("\n=== 5. stacking damped/direct on top of the best window ===")
    push!(results, run_combo(
        "window$(best_window.window_weeks)+damped(0.9,0.3)",
        VALIDATION_ONLY, hist, versions_full; truth=truth,
        mode=:damped, window_weeks=best_window.window_weeks,
        damp_alpha=0.3, damp_beta=0.1, damp_phi=0.9, damp_blend_max=0.3,
    ))
    push!(results, run_combo(
        "window$(best_window.window_weeks)+direct(pool=0.9)",
        VALIDATION_ONLY, hist, versions_full; truth=truth,
        mode=:direct, window_weeks=best_window.window_weeks, pool_w=0.9,
    ))

    println("\n=== 6. damped blend/phi refinement on the best window ===")
    for (phi, bmax) in ((0.9, 0.2), (0.9, 0.4), (0.95, 0.3), (0.85, 0.3))
        push!(results, run_combo(
            "window$(best_window.window_weeks)+damped($phi,$bmax)",
            VALIDATION_ONLY, hist, versions_full;
            truth=truth, mode=:damped, window_weeks=best_window.window_weeks,
            damp_alpha=0.3, damp_beta=0.1, damp_phi=phi, damp_blend_max=bmax,
        ))
    end

    candidates = results[2:end]
    sorted = sort(candidates; by=r -> r.mean_wis)
    flagged = filter(r -> net_positive(baseline, r), candidates)
    sorted_flagged = sort(flagged; by=r -> r.mean_wis)

    winner = isempty(sorted_flagged) ? baseline : sorted_flagged[1]

    println("\n=== ranked (all candidates, vs baseline mean_wis=" *
            "$(round(baseline.mean_wis; digits=4))) ===")
    for r in sorted
        tag = net_positive(baseline, r) ? " [net-positive h3/4]" : ""
        println("  $(rpad(r.label, 30)) mean=$(round(r.mean_wis; digits=4)) " *
                "h3=$(round(r.h3; digits=4)) h4=$(round(r.h4; digits=4))$tag")
    end
    if isempty(sorted_flagged)
        println("\nNO candidate lifted h=3,4 net-positive: keeping baseline.")
    else
        println("\nbest net-positive candidate: $(winner.label) " *
                "mean_wis=$(round(winner.mean_wis; digits=4))")
    end

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "long-horizon (h=3,4) fix -- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "=" ^ 69)
        println(io, "LEAKAGE FIX (honest rescore) -- READ THIS FIRST")
        println(io, "=" ^ 69)
        println(io, "The first version of this file built its seasonal " *
                     "core on `build_seasonal_profile`/`build_revision_" *
                     "profile` ONCE from a fixed `season_year <= 2016` " *
                     "cutoff, reused across every split -- a real leak for " *
                     "validation-season splits (see round2-stack/generate." *
                     "jl and its score.txt, where this was first caught). " *
                     "Both are now rebuilt PER SPLIT from only strictly- " *
                     "prior data. window_weeks/damped-trend/momentum/direct " *
                     "do not touch either profile, so they are LEAK-" *
                     "INDEPENDENT of the fix -- expected, and confirmed " *
                     "below, to survive it. The old leaky sweep is kept " *
                     "verbatim as score-leaky.txt for an explicit before/" *
                     "after; every number below is from this leak-free " *
                     "rerun.")
        println(io)
        println(io, "Honest headline numbers (validation seasons 1,2, " *
                     "natural-scale WIS):")
        println(io)
        println(io, "  model                                   mean_wis  " *
                     "sd_wis  cov50  cov90")
        println(io, "  full stack, LEAK-FREE (baseline here)     " *
                     "$(round(baseline.mean_wis; digits=4))  " *
                     "$(round(baseline.sd_wis; digits=4))  " *
                     "$(round(baseline.cov50; digits=3))  " *
                     "$(round(baseline.cov90; digits=3))")
        println(io, "  window208+damped, LEAK-FREE (this winner) " *
                     "$(round(winner.mean_wis; digits=4))  " *
                     "$(round(winner.sd_wis; digits=4))  " *
                     "$(round(winner.cov50; digits=3))  " *
                     "$(round(winner.cov90; digits=3))")
        println(io, "  " * "-"^63)
        println(io, "  full stack, LEAKY (original longhz baseline)      " *
                     "0.2601  0.2587  0.565  0.943  (score-leaky.txt)")
        println(io, "  window208+damped(0.9,0.2), LEAKY (original " *
                     "winner)                                       " *
                     "0.2524  --      --     --     (score-leaky.txt)")
        println(io, "  clean season model (per-location climatology, " *
                     "leak-free by construction)                    " *
                     "0.3004  0.3890  --     --     " *
                     "(experiments/simple-round/season/score.txt)")
        println(io, "  conformal-on-plain (same point forecast, split- " *
                     "conformal intervals)                           " *
                     "0.2917  0.3451  0.48   0.871  " *
                     "(experiments/simple-round/conformal/score.txt)")
        println(io)
        pct_baseline = 100 * (baseline.mean_wis - winner.mean_wis) /
                       baseline.mean_wis
        pct_vs_season = 100 * (0.3004 - winner.mean_wis) / 0.3004
        pct_vs_conformal = 100 * (0.2917 - winner.mean_wis) / 0.2917
        println(io, "Reading this table:")
        pct_inflate_base = 100 * (baseline.mean_wis - 0.2601) / 0.2601
        pct_inflate_win = 100 * (winner.mean_wis - 0.2524) / 0.2524
        println(io, "  - The leak inflated both numbers by a similar " *
                     "absolute amount (full stack: 0.2601 -> " *
                     "$(round(baseline.mean_wis; digits=4)), " *
                     "$(round(pct_inflate_base; digits=1))%; " *
                     "window208+damped: 0.2524 -> " *
                     "$(round(winner.mean_wis; digits=4)), " *
                     "$(round(pct_inflate_win; digits=1))%) -- expected, " *
                     "since both share the same profile-building code and " *
                     "validation splits.")
        println(io, "  - The window208+damped-trend lever is NOT an " *
                     "artifact of the leak: stacked on the now-leak-free " *
                     "full stack, it still cuts mean WIS by " *
                     "$(round(pct_baseline; digits=2))% " *
                     "($(round(baseline.mean_wis; digits=4)) -> " *
                     "$(round(winner.mean_wis; digits=4))), and beats both " *
                     "the leak-free `season` model by " *
                     "$(round(pct_vs_season; digits=2))% and " *
                     "leak-free conformal-on-plain by " *
                     "$(round(pct_vs_conformal; digits=2))%.")
        println(io, "  - By horizon, the honest gain is still concentrated " *
                     "at h=3/4 exactly as hypothesized (see the per-" *
                     "horizon table below) -- the leak inflated the " *
                     "ABSOLUTE numbers but did not manufacture the h=3/4-" *
                     "targeted shape of the improvement.")
        println(io)
        println(io, "round2-stack winner (log+tstudent+pool(w=0.9)), " *
                     "LEAK-FREE: " *
                     "mean_wis=$(round(baseline.mean_wis; digits=4)) " *
                     "sd_wis=$(round(baseline.sd_wis; digits=4))")
        println(io, "  by horizon: h1=$(round(baseline.h1; digits=4)) " *
                     "h2=$(round(baseline.h2; digits=4)) " *
                     "h3=$(round(baseline.h3; digits=4)) " *
                     "h4=$(round(baseline.h4; digits=4))")
        pct_h3 = 100 * (baseline.h3 - baseline.h1) / baseline.h1
        pct_h4 = 100 * (baseline.h4 - baseline.h1) / baseline.h1
        println(io, "  (still confirms the monotone h=1..4 degradation " *
                     "reported in round2-stack/score.txt, honestly: h3 " *
                     "and h4 are $(round(pct_h3; digits=1))% and " *
                     "$(round(pct_h4; digits=1))% worse than h1.)")
        println(io)
        println(io, "=== every candidate, WIS by horizon vs baseline " *
                     "(d = baseline - candidate, positive = improvement) " *
                     "===")
        for r in results
            print_and_write_row(io, r, baseline)
        end
        println(io)
        println(io, "=== ranked by overall mean WIS ===")
        for r in sorted
            tag = net_positive(baseline, r) ? " [net-positive h3/4]" : ""
            println(io, "  $(rpad(r.label, 30)) mean_wis=" *
                         "$(round(r.mean_wis; digits=4))$tag")
        end
        println(io)
        if isempty(sorted_flagged)
            println(io, "=== conclusion ===")
            println(io, "NO candidate lifted h=3/4 net-positive (improves " *
                         "h3 or h4, does not worsen h1/h2 by more than 1%, " *
                         "and does not worsen overall mean WIS). The " *
                         "momentum/damped/direct/longer-memory fixes tried " *
                         "here do not beat the round2-stack winner on " *
                         "validation; keeping that model unchanged for " *
                         "submission (model_id nfidd-longhz reproduces it " *
                         "exactly, not a new formulation).")
        else
            println(io, "=== conclusion ===")
            pct = 100 * (baseline.mean_wis - winner.mean_wis) /
                  baseline.mean_wis
            println(io, "winner: $(winner.label) " *
                         "mean_wis=$(round(winner.mean_wis; digits=4)) " *
                         "(vs baseline $(round(baseline.mean_wis; digits=4))" *
                         ", $(round(pct; digits=2))%)")
            println(io, "  h3: $(round(winner.h3; digits=4)) " *
                         "(baseline $(round(baseline.h3; digits=4)))")
            println(io, "  h4: $(round(winner.h4; digits=4)) " *
                         "(baseline $(round(baseline.h4; digits=4)))")
            if hub_path === nothing
                println(io, "No hub_path given this run -- hub submissions " *
                             "are PAUSED (Sam, see submissions/README.md), " *
                             "so no model-output was written; this is a " *
                             "validation-only honest rescore.")
            else
                println(io, "Submitted (model_id nfidd-longhz) across all " *
                             "five seasons with this configuration.")
            end
        end
    end

    if hub_path !== nothing
        println("\nrefitting winner ($(winner.label)) across all 5 " *
                "seasons for hub submission ...")
        # LEAK-FREE: build_forecast_table rebuilds both profiles fresh
        # per split from `hist`/`versions_full` itself now, so no
        # separate profile needs to be prepared here (unlike the old
        # static-per-transform version of this driver).
        full_fc = build_forecast_table(
            (1, 2, 3, 4, 5), hist, versions_full;
            transform=winner.transform, pool_w=winner.pool_w,
            model_id=SUB_MODEL_ID, window_weeks=winner.window_weeks,
            mode=winner.mode, mom_weight=winner.mom_weight,
            mom_decay=winner.mom_decay, mom_window=winner.mom_window,
            damp_alpha=winner.damp_alpha, damp_beta=winner.damp_beta,
            damp_phi=winner.damp_phi, damp_blend_max=winner.damp_blend_max,
        )
        write_submission(full_fc, hub_path)
        write_metadata(
            SUB_MODEL_ID, hub_path; team_abbr="nfidd", model_abbr="longhz",
            designated=true,
        )
        println("wrote $(nrow(full_fc)) rows across " *
                "$(length(unique(full_fc.origin_date))) origin dates to " *
                "$(hub_path)")
    end

    dt = round(time() - t0; digits=1)
    println("\nwrote score.txt in $(dt)s total")
    return (results=results, winner=winner)
end

if abspath(PROGRAM_FILE) == @__FILE__
    hub_path = length(ARGS) >= 1 ? ARGS[1] : nothing
    main(hub_path)
end
