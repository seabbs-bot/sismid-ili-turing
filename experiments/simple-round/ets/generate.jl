#!/usr/bin/env julia
# simple dynamics sweep -- simple-round, ALTERNATIVE SIMPLE DYNAMICS
# family (beyond plain AR).
#
# Reference point: `nfidd-ar6` (submissions/nfidd-ar6/generate_forecasts.jl),
# independent AR(6) per location fit by OLS on the fourth-root-transformed
# vintage series, scores 0.368 mean WIS on validation; adding a backfill
# correction (`seabbs_bot-ar6bf`) improves that to 0.359
# (submissions/seabbs_bot-ar6bf/README.md). This sweep asks whether a
# DIFFERENT simple per-location dynamic -- not AR(6), not its order --
# beats either number. AR order itself is a separate sweep
# (experiments/simple-round/ar-order/); this one holds order-6 AR as the
# reference and instead swaps in five alternative dynamics:
#
#   - `ses`     simple exponential smoothing (level only, no trend)
#   - `holt`    linear-trend (Holt) exponential smoothing
#   - `damped`  damped-trend exponential smoothing (Holt with phi < 1)
#   - `rwdrift` random walk with drift
#   - `ardiff`  AR(p) on first differences (ARIMA(p,1,0)-style), p chosen
#               per series by in-sample AIC over {1, 2, 3}
#
# Each is fit per (location, split) on the fourth-root modelling scale
# (docs/lessons.md item 7: the package's current favoured transform), with
# smoothing/AR parameters chosen per series by grid search minimising
# in-sample one-step-ahead SSE (`ses`/`holt`/`damped`) or AIC (`ardiff`).
# `rwdrift` has no free smoothing parameter (drift = mean first
# difference). Each is then tried both with and without the SAME additive,
# per-location, median backfill correction `seabbs_bot-ar6bf` uses
# (delay_cutoff=8, min_support=5) -- a fixed ingredient, not a re-sweep of
# the correction's own hyperparameters (that sweep is
# experiments/simple-round/backfill/).
#
# Forecasts come from simulating 1000 Gaussian-innovation sample paths
# forward per location/horizon (matching nfidd-ar6's approach), taking the
# hub's 23 quantile levels, and back-transforming to natural wILI%,
# clamped at 0.
#
# LIGHT + ANALYTIC: CSV/DataFrames/Statistics/LinearAlgebra only, no
# Turing. Scope: VALIDATION SEASONS (1, 2) ONLY, scored against the local
# hub clone's oracle (docs/contracts.md experimental integrity) -- this is
# a tuning sweep, not a submission driver.
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
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12
const VALIDATION_ONLY = (1, 2)
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")

# Backfill correction knobs -- fixed at seabbs_bot-ar6bf's chosen design
# (additive, per-location, median, 8-week cutoff); see that submission's
# README for why these were picked. Not re-swept here.
const BF_DELAY_CUTOFF = 8
const BF_MIN_SUPPORT = 5

# Grids for the per-series smoothing/AR-order parameter search. Kept
# coarse enough to stay fast (each combination is one O(T) filter pass,
# not a fit): SES gets a fine alpha grid since it is its only free
# parameter; Holt/damped share an alpha/beta grid, with damped adding a
# phi grid on top.
const SES_ALPHAS = 0.05:0.05:0.95
const HOLT_ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)
const HOLT_BETAS = (0.02, 0.05, 0.1, 0.2)
const DAMPED_PHIS = (0.8, 0.85, 0.9, 0.95, 0.98)
const ARDIFF_ORDERS = (1, 2, 3)

# ---------------------------------------------------------------------
# AR(order) on levels -- the nfidd-ar6 reference design, and (with
# order=1, coef fixed rather than fit) reused directly for rwdrift's
# forward simulation below.
# ---------------------------------------------------------------------

"""
    fit_ar(y, order) -> (coef, resid_sd)

OLS fit of an AR(`order`) model with intercept to `y` (ascending in
time, no missing values). `coef = [c, phi_1, ..., phi_order]`,
`resid_sd` the in-sample residual standard deviation. Identical to
`submissions/nfidd-ar6/generate_forecasts.jl`'s function of the same
name.
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
    simulate_paths(y, coef, resid_sd, order, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Simulate `npaths` Gaussian-innovation AR(`order`) sample paths forward
from the end of `y` (modelling scale). Used both for the AR(6)
reference design and, with `order=1` and `coef` fixed to `[drift,
1.0]`, for `rwdrift` (a random walk with drift is an AR(1) with unit
coefficient).
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
# ses -- simple exponential smoothing
# ---------------------------------------------------------------------

"""
    fit_ses(y; alphas=SES_ALPHAS) -> (alpha, level, resid_sd)

Fit `ETS(A,N,N)` (level only, no trend) by grid search over `alphas`,
picking the smoothing constant that minimises in-sample one-step-ahead
SSE. `level` is the final smoothed level (the flat point forecast for
every horizon); `resid_sd` the in-sample residual SD of the chosen fit.
"""
function fit_ses(y::AbstractVector{Float64}; alphas=SES_ALPHAS)
    n = length(y)
    best_sse, best_alpha, best_level = Inf, first(alphas), y[1]
    for alpha in alphas
        level = y[1]
        sse = 0.0
        for t in 2:n
            sse += (y[t] - level)^2
            level = alpha * y[t] + (1 - alpha) * level
        end
        if sse < best_sse
            best_sse, best_alpha, best_level = sse, alpha, level
        end
    end
    dof = max((n - 1) - 2, 1)
    resid_sd = sqrt(best_sse / dof)
    return (alpha=best_alpha, level=best_level, resid_sd=resid_sd)
end

"""
    simulate_ses_paths(level0, alpha, resid_sd, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Simulate `npaths` sample paths of `ETS(A,N,N)` forward from `level0`.
Each step draws a Gaussian innovation, adds it to the current level to
produce a simulated observation, then updates the level with the same
smoothing recursion used in fitting -- so forecast variance widens with
horizon exactly as the state-space form implies, without needing the
closed-form variance formula.
"""
function simulate_ses_paths(
    level0::Float64, alpha::Float64, resid_sd::Float64, horizons,
    npaths::Int; rng::Random.AbstractRNG,
)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    for s in 1:npaths
        level = level0
        for h in 1:hmax
            val = level + resid_sd * randn(rng)
            if h in horizons
                out[h][s] = val
            end
            level = alpha * val + (1 - alpha) * level
        end
    end
    return out
end

# ---------------------------------------------------------------------
# holt / damped -- linear- and damped-trend exponential smoothing
# ---------------------------------------------------------------------

"""
    _holt_filter(y, alpha, beta, phi) -> (sse, level, trend)

One in-sample pass of the damped-trend smoothing recursion (`phi = 1.0`
recovers plain linear-trend Holt): `pred_t = level + phi * trend`,
`new_level = alpha * y_t + (1 - alpha) * pred_t`, `new_trend = beta *
(new_level - level) + (1 - beta) * phi * trend`. Returns the one-step
SSE and the final `(level, trend)` state, shared by `fit_holt` and
`fit_damped`'s grid searches.
"""
function _holt_filter(
    y::AbstractVector{Float64}, alpha::Float64, beta::Float64, phi::Float64,
)
    n = length(y)
    level, trend = y[1], y[2] - y[1]
    sse = 0.0
    for t in 2:n
        pred = level + phi * trend
        sse += (y[t] - pred)^2
        new_level = alpha * y[t] + (1 - alpha) * pred
        new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level, trend = new_level, new_trend
    end
    return sse, level, trend
end

"""
    fit_holt(y; alphas=HOLT_ALPHAS, betas=HOLT_BETAS)
        -> (alpha, beta, phi, level, trend, resid_sd)

Fit linear-trend Holt (`ETS(A,A,N)`, `phi` fixed at 1.0) by grid search
over `(alpha, beta)` minimising in-sample SSE (`_holt_filter`).
"""
function fit_holt(y::AbstractVector{Float64}; alphas=HOLT_ALPHAS,
        betas=HOLT_BETAS)
    n = length(y)
    best = (sse=Inf, alpha=first(alphas), beta=first(betas), level=y[1],
            trend=0.0)
    for alpha in alphas, beta in betas
        sse, level, trend = _holt_filter(y, alpha, beta, 1.0)
        if sse < best.sse
            best = (sse=sse, alpha=alpha, beta=beta, level=level,
                     trend=trend)
        end
    end
    dof = max((n - 1) - 4, 1)
    resid_sd = sqrt(best.sse / dof)
    return (alpha=best.alpha, beta=best.beta, phi=1.0, level=best.level,
            trend=best.trend, resid_sd=resid_sd)
end

"""
    fit_damped(y; alphas=HOLT_ALPHAS, betas=HOLT_BETAS, phis=DAMPED_PHIS)
        -> (alpha, beta, phi, level, trend, resid_sd)

Fit damped-trend Holt (`ETS(A,Ad,N)`) by grid search over `(alpha, beta,
phi)` minimising in-sample SSE (`_holt_filter`).
"""
function fit_damped(y::AbstractVector{Float64}; alphas=HOLT_ALPHAS,
        betas=HOLT_BETAS, phis=DAMPED_PHIS)
    n = length(y)
    best = (sse=Inf, alpha=first(alphas), beta=first(betas),
            phi=first(phis), level=y[1], trend=0.0)
    for alpha in alphas, beta in betas, phi in phis
        sse, level, trend = _holt_filter(y, alpha, beta, phi)
        if sse < best.sse
            best = (sse=sse, alpha=alpha, beta=beta, phi=phi, level=level,
                     trend=trend)
        end
    end
    dof = max((n - 1) - 5, 1)
    resid_sd = sqrt(best.sse / dof)
    return (alpha=best.alpha, beta=best.beta, phi=best.phi, level=best.level,
            trend=best.trend, resid_sd=resid_sd)
end

"""
    simulate_holt_paths(level0, trend0, alpha, beta, phi, resid_sd,
                         horizons, npaths; rng) -> Dict{Int,Vector{Float64}}

Simulate `npaths` sample paths of (damped-)trend exponential smoothing
forward from `(level0, trend0)`, `phi = 1.0` for plain Holt. Same
draw-then-update logic as `simulate_ses_paths`, with the trend state
carried and damped by `phi` each step.
"""
function simulate_holt_paths(
    level0::Float64, trend0::Float64, alpha::Float64, beta::Float64,
    phi::Float64, resid_sd::Float64, horizons, npaths::Int;
    rng::Random.AbstractRNG,
)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    for s in 1:npaths
        level, trend = level0, trend0
        for h in 1:hmax
            pred = level + phi * trend
            val = pred + resid_sd * randn(rng)
            if h in horizons
                out[h][s] = val
            end
            new_level = alpha * val + (1 - alpha) * pred
            new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
            level, trend = new_level, new_trend
        end
    end
    return out
end

# ---------------------------------------------------------------------
# rwdrift -- random walk with drift
# ---------------------------------------------------------------------

"""
    fit_rwdrift(y) -> (coef, resid_sd)

Random walk with drift: `y_t = y_{t-1} + drift + eps_t`, `drift =
mean(diff(y))`. Returned as `coef = [drift, 1.0]` so the existing
`simulate_paths` (AR machinery, `order=1`) can be reused unchanged for
forward simulation -- an RW-with-drift is exactly an AR(1) with the
autoregressive coefficient fixed at 1 rather than fit.
"""
function fit_rwdrift(y::AbstractVector{Float64})
    d = diff(y)
    drift = mean(d)
    dof = max(length(d) - 1, 1)
    resid_sd = sqrt(sum(abs2, d .- drift) / dof)
    return (coef=[drift, 1.0], resid_sd=resid_sd)
end

# ---------------------------------------------------------------------
# ardiff -- AR(p) on first differences (ARIMA(p,1,0)-style)
# ---------------------------------------------------------------------

"""
    fit_ardiff(y; orders=ARDIFF_ORDERS) -> (coef, resid_sd, order)

Fit AR(`p`) (`fit_ar`) to the first-differenced series `diff(y)` for
each `p` in `orders`, picking the order that minimises in-sample AIC
(`n * log(SSE / n) + 2 * (p + 1)`). Differencing once removes a
stochastic trend that a plain AR(p) on levels does not; re-integrating
the simulated forward differences (`simulate_ar_diff_paths`) recovers
level forecasts.
"""
function fit_ardiff(y::AbstractVector{Float64}; orders=ARDIFF_ORDERS)
    d = diff(y)
    best_aic, best = Inf, nothing
    for p in orders
        nobs = length(d) - p
        nobs < p + 2 && continue
        coef, resid_sd = fit_ar(d, p)
        sse = resid_sd^2 * max(nobs - (p + 1), 1)
        aic = nobs * log(sse / nobs) + 2 * (p + 1)
        if aic < best_aic
            best_aic = aic
            best = (coef=coef, resid_sd=resid_sd, order=p)
        end
    end
    best === nothing && error("no ardiff order in $orders fit for n=$(length(y))")
    return best
end

"""
    simulate_ar_diff_paths(y, coef, resid_sd, order, horizons, npaths;
                            rng) -> Dict{Int,Vector{Float64}}

Simulate `npaths` sample paths of the fitted AR(`order`) on
first-differences forward, cumulatively summing each simulated
difference onto the last observed level `y[end]` to produce level
forecasts at each horizon (the ARIMA(p,1,0) re-integration step).
"""
function simulate_ar_diff_paths(
    y::AbstractVector{Float64}, coef::Vector{Float64}, resid_sd::Float64,
    order::Int, horizons, npaths::Int; rng::Random.AbstractRNG,
)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    d = diff(y)
    tail0 = d[(end - order + 1):end]
    level0 = y[end]
    for s in 1:npaths
        tail = copy(tail0)
        level = level0
        for h in 1:hmax
            pred = coef[1]
            for lag in 1:order
                pred += coef[lag + 1] * tail[end - lag + 1]
            end
            dval = pred + resid_sd * randn(rng)
            level += dval
            if h in horizons
                out[h][s] = level
            end
            push!(tail, dval)
            popfirst!(tail)
        end
    end
    return out
end

# ---------------------------------------------------------------------
# Backfill correction (fixed design, from seabbs_bot-ar6bf)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile (additive, median), on the
`transform` scale. Identical to
`submissions/seabbs_bot-ar6bf/generate_forecasts.jl`'s function of the
same name.
"""
function build_revision_profile(
    versions::DataFrame; transform::Symbol, max_delay::Int,
    min_support::Int,
)
    raw = Dict{Tuple{String,Int},Vector{Float64}}()
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
            key = (loc, delay)
            push!(get!(raw, key, Float64[]), settled - vintage)
        end
    end
    profile = Dict{Tuple{String,Int},Float64}()
    for (key, vals) in raw
        length(vals) >= min_support && (profile[key] = median(vals))
    end
    return profile
end

"""
    apply_backfill_correction!(data, profile)

Nudge `data.Y` in place wherever `0 <= delay <= BF_DELAY_CUTOFF` and a
matching profile entry exists. Identical to
`submissions/seabbs_bot-ar6bf/generate_forecasts.jl`'s function of the
same name.
"""
function apply_backfill_correction!(
    data::ModelData, profile::Dict{Tuple{String,Int},Float64},
)
    for l in 1:data.L, t in 1:data.T
        d = data.delay[t, l]
        (d < 0 || d > BF_DELAY_CUTOFF) && continue
        ismissing(data.Y[t, l]) && continue
        key = (LOCATIONS[l], d)
        haskey(profile, key) || continue
        data.Y[t, l] += profile[key]
    end
    return data
end

# ---------------------------------------------------------------------
# Shared forecast-table builder
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, model_id, fit_fn, sim_fn; backfill_profile,
                          versions_full) -> DataFrame

Fit and forecast one candidate (`fit_fn(y) -> fit`, `sim_fn(y, fit,
horizons, npaths, rng) -> Dict{Int,Vector{Float64}}`) for every
cross-validation split of every season in `seasons`, returning the
combined hub quantile table (docs/contracts.md schema). If
`backfill_profile` is not `nothing`, `apply_backfill_correction!` is
run on each split's data before fitting (and `versions_full` must be
supplied so `build_model_data` can compute true `as_of`-based delay).
"""
function build_forecast_table(
    seasons, model_id::String, fit_fn::Function, sim_fn::Function;
    backfill_profile::Union{Nothing,Dict}=nothing,
    versions_full::Union{Nothing,DataFrame}=nothing,
)
    rng = MersenneTwister(SEED)
    rows = DataFrame(
        model_id=String[], location=String[], origin_date=Date[],
        horizon=Int[], target_end_date=Date[], target=String[],
        output_type=String[], output_type_id=Float64[], value=Float64[],
    )
    for season in seasons
        for split in training_splits(season)
            data = build_model_data(
                split; Dmax=DMAX, transform=TRANSFORM, window_weeks=104,
                versions=(backfill_profile === nothing ? nothing :
                          versions_full),
            )
            if backfill_profile !== nothing
                apply_backfill_correction!(data, backfill_profile)
            end
            origin = data.origin_date
            for (li, loc) in enumerate(LOCATIONS)
                y = Float64.(data.Y[:, li])
                fit = fit_fn(y)
                paths = sim_fn(y, fit, HORIZONS, NPATHS, rng)
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    vals = paths[h]
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

# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------

# Ordered (not Dict) so sweep output/report order is deterministic.
const FAMILIES = [
    ("ses", fit_ses,
     (y, fit, h, np, rng) -> simulate_ses_paths(
         fit.level, fit.alpha, fit.resid_sd, h, np; rng=rng)),
    ("holt", fit_holt,
     (y, fit, h, np, rng) -> simulate_holt_paths(
         fit.level, fit.trend, fit.alpha, fit.beta, fit.phi, fit.resid_sd,
         h, np; rng=rng)),
    ("damped", fit_damped,
     (y, fit, h, np, rng) -> simulate_holt_paths(
         fit.level, fit.trend, fit.alpha, fit.beta, fit.phi, fit.resid_sd,
         h, np; rng=rng)),
    ("rwdrift", fit_rwdrift,
     (y, fit, h, np, rng) -> simulate_paths(
         y, fit.coef, fit.resid_sd, 1, h, np; rng=rng)),
    ("ardiff", fit_ardiff,
     (y, fit, h, np, rng) -> simulate_ar_diff_paths(
         y, fit.coef, fit.resid_sd, fit.order, h, np; rng=rng)),
]

function run_family(fname, fit_fn, sim_fn, truth; backfill_profile=nothing,
        versions_full=nothing)
    forecast = build_forecast_table(
        VALIDATION_ONLY, "tmp-$fname", fit_fn, sim_fn;
        backfill_profile=backfill_profile, versions_full=versions_full,
    )
    scored = score_forecasts(forecast, truth; scale=:natural)
    summ = wis_summary(scored)
    return (summary=summ[1, :], scored=scored)
end

function main()
    t0 = time()
    truth = load_oracle(HUB_PATH)

    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_DELAY_CUTOFF,
        min_support=BF_MIN_SUPPORT,
    )
    println("backfill profile: $(length(profile)) (location, delay) " *
            "entries with >= $(BF_MIN_SUPPORT) observations")

    println("=== reference: plain AR(6), no backfill (nfidd-ar6 design) ===")
    ar6_fit = y -> fit_ar(y, 6)
    ar6_sim = (y, fit, h, np, rng) -> simulate_paths(
        y, fit[1], fit[2], 6, h, np; rng=rng)
    ar6 = run_family("ar6", ar6_fit, ar6_sim, truth)
    println("mean_wis=$(round(ar6.summary.mean_wis; digits=4)) " *
            "sd_wis=$(round(ar6.summary.sd_wis; digits=4)) " *
            "n_tasks=$(ar6.summary.n_tasks)  " *
            "(reference: nfidd-ar6 README reports 0.368/0.471)")

    results = NamedTuple[]
    scored_by_key = Dict{Tuple{String,Bool},DataFrame}()
    for (fname, fit_fn, sim_fn) in FAMILIES
        for bf in (false, true)
            bp = bf ? profile : nothing
            r = run_family(
                fname, fit_fn, sim_fn, truth; backfill_profile=bp,
                versions_full=versions_full,
            )
            push!(results, (
                family=fname, backfill=bf, mean_wis=r.summary.mean_wis,
                sd_wis=r.summary.sd_wis, n_tasks=r.summary.n_tasks,
            ))
            scored_by_key[(fname, bf)] = r.scored
            println("family=$(rpad(fname, 8)) backfill=$(rpad(bf, 6)) -> " *
                    "mean_wis=$(round(r.summary.mean_wis; digits=4)) " *
                    "sd_wis=$(round(r.summary.sd_wis; digits=4))")
        end
    end

    sort!(results; by=r -> r.mean_wis)
    best = results[1]
    best_scored = scored_by_key[(best.family, best.backfill)]
    println("\n=== best variant ===")
    println(best)

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "simple dynamics sweep -- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "reference points:")
        println(io, "  plain AR(6), this run, no backfill: " *
                     "mean_wis=$(round(ar6.summary.mean_wis; digits=4)) " *
                     "sd_wis=$(round(ar6.summary.sd_wis; digits=4))")
        println(io, "  nfidd-ar6 (submissions/nfidd-ar6/README.md): " *
                     "0.368 (sd 0.471)")
        println(io, "  seabbs_bot-ar6bf, AR(6) + backfill " *
                     "(submissions/seabbs_bot-ar6bf/README.md): " *
                     "0.359 (sd 0.452) -- the number to beat")
        println(io)
        println(io, "full sweep (sorted by mean_wis, ascending):")
        println(io, rpad("family", 10) * rpad("backfill", 10) *
                     rpad("mean_wis", 12) * "sd_wis")
        for r in results
            println(io,
                rpad(r.family, 10) * rpad(string(r.backfill), 10) *
                rpad(string(round(r.mean_wis; digits=4)), 12) *
                string(round(r.sd_wis; digits=4)),
            )
        end
        println(io)
        println(io, "=== best variant ===")
        println(io, "family=$(best.family) backfill=$(best.backfill)")
        println(io, "mean_wis=$(round(best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(best.sd_wis; digits=4)) " *
                     "n_tasks=$(best.n_tasks)")
        vs_ar6bf = 0.359 - best.mean_wis
        vs_ar6bf_pct = 100 * vs_ar6bf / 0.359
        println(io, "vs seabbs_bot-ar6bf (0.359): " *
                     "$(round(vs_ar6bf; digits=4)) " *
                     "($(round(vs_ar6bf_pct; digits=2))%, " *
                     "positive = best variant wins)")

        println(io)
        println(io, "-- breakdown by location (best variant vs plain AR6) --")
        by_loc = combine(groupby(best_scored, :location),
            :wis => mean => :mean_wis, nrow => :n)
        base_by_loc = combine(groupby(ar6.scored, :location),
            :wis => mean => :mean_wis)
        merged_loc = innerjoin(
            by_loc, base_by_loc; on=:location, renamecols="_best" => "_base",
        )
        merged_loc.improvement = merged_loc.mean_wis_base .-
                                  merged_loc.mean_wis_best
        sort!(merged_loc, :improvement; rev=true)
        for row in eachrow(merged_loc)
            println(io, rpad(row.location, 16) *
                         "best=$(round(row.mean_wis_best; digits=4)) " *
                         "ar6=$(round(row.mean_wis_base; digits=4)) " *
                         "improvement=$(round(row.improvement; digits=4))")
        end

        println(io)
        println(io, "-- breakdown by origin_date (best variant vs plain " *
                     "AR6) --")
        by_time = combine(groupby(best_scored, :origin_date),
            :wis => mean => :mean_wis, :wis => std => :sd_wis, nrow => :n)
        sort!(by_time, :mean_wis; rev=true)
        for row in first(eachrow(by_time), 10)
            println(io, "$(row.origin_date): " *
                         "mean_wis=$(round(row.mean_wis; digits=4)) " *
                         "sd_wis=$(round(row.sd_wis; digits=4)) n=$(row.n)")
        end

        println(io)
        println(io, "-- breakdown by horizon (best variant vs plain AR6) --")
        by_h = combine(groupby(best_scored, :horizon),
            :wis => mean => :mean_wis, nrow => :n)
        base_by_h = combine(groupby(ar6.scored, :horizon),
            :wis => mean => :mean_wis)
        merged_h = innerjoin(
            by_h, base_by_h; on=:horizon, renamecols="_best" => "_base",
        )
        merged_h.improvement = merged_h.mean_wis_base .- merged_h.mean_wis_best
        sort!(merged_h, :horizon)
        for row in eachrow(merged_h)
            println(io, "h=$(row.horizon): " *
                         "best=$(round(row.mean_wis_best; digits=4)) " *
                         "ar6=$(round(row.mean_wis_base; digits=4)) " *
                         "improvement=$(round(row.improvement; digits=4))")
        end
    end

    dt = round(time() - t0; digits=1)
    println("\nwrote score.txt in $(dt)s total")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
