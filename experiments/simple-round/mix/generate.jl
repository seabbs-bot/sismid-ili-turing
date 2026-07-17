#!/usr/bin/env julia
# mix_ridgevar_season.jl -- ridge-regularised VAR-style cross-location
# AR(6) with a light seasonal harmonic, per location, fit by ridge OLS.
# LIGHT + ANALYTIC (no Turing): CSV/DataFrames/Statistics/LinearAlgebra
# only, same dependency footprint as submissions/nfidd-ar6.
#
# Origin: a "mixtures" sweep of analytic mechanisms for the simple-
# model wide round (docs/brief.md, submissions/README.md's "wide
# simple round" tracker), tasked with combining promising single-
# mechanism ideas -- time-varying (discounted) AR, cross-location
# predictors, pooled/shrunk AR, VAR, ridge, backfill, seasonality --
# to see whether combining them compounds the gains over the single-
# mechanism baselines (seabbs_bot-ar6bf: AR(6)+backfill, 0.359 mean
# validation WIS; the team's separately-tracked AR(12)+backfill,
# 0.3518).
#
# What won the sweep: NOT a combination of everything. Ridge-penalised
# cross-location lag-1 predictors (a light VAR: each location's AR(6)
# gets one extra lag-1 predictor per OTHER location, `L-1 = 10` extra
# columns, kept in check by an L2 penalty on every non-intercept
# coefficient) plus a single sin/cos seasonal harmonic, with NO
# backfill correction and NO discounted/time-varying weighting,
# clearly beat every mixture that added those extra mechanisms on top.
# See the README in this directory for the full sweep table and the
# ablation that isolated this.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl <hub_path>

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))
include(joinpath(PKG_DIR, "src", "hubio.jl"))

const MODEL_ID = "simple-mix-ridgevar-season"
const TRANSFORM = :fourthroot
const OWN_ORDER = 6      # own-location AR lags
const CROSS_ORDER = 1    # lag-1 of every OTHER location (the "VAR" part)
const RIDGE_LAMBDA = 2.0 # L2 penalty on every non-intercept coefficient
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12
const WINDOW_WEEKS = 104

# ---------------------------------------------------------------------
# Regression machinery: a ridge-penalised OLS solve, a design matrix
# with own AR lags + cross-location lag-1 predictors + a light seasonal
# harmonic pair, and a joint (all-locations-at-once) path simulator --
# joint because a cross-location predictor means each location's
# h-step-ahead forecast depends on every OTHER location's simulated
# future path too, not just its own.
# ---------------------------------------------------------------------

"""
    ridge_ls(X, y, lambda) -> coef

Ridge-penalised OLS: `(X'X + lambda*P) \\ (X'y)`, where `P` is the
identity with the intercept (first column) excluded from the penalty.
`lambda = 0` is plain OLS.
"""
function ridge_ls(X::Matrix{Float64}, y::Vector{Float64}, lambda::Float64)
    P = Diagonal([i == 1 ? 0.0 : 1.0 for i in 1:size(X, 2)])
    return (X' * X + lambda * P) \ (X' * y)
end

"""
    build_design(Y, li, own_order, cross_order; woy, W)
        -> (X, yresp)

Design matrix for location `li`'s regression: intercept, `own_order`
own lags, then one lag per week in `1:cross_order` for every OTHER
location (columns ordered by location, matching `simulate_paths`'s
read-back), then a `sin`/`cos` pair at `2*pi*woy[t]/W` (a single
seasonal harmonic, evaluated at the RESPONSE week `t`, not lagged).
`Y` is the full (T x L) matrix on the modelling scale, no missing
values.
"""
function build_design(Y::Matrix{Float64}, li::Int, own_order::Int,
        cross_order::Int, woy::Vector{Int}, W::Int)
    T, L = size(Y)
    cross_locs = filter(!=(li), 1:L)
    maxlag = max(own_order, cross_order)
    nobs = T - maxlag
    ncols = 1 + own_order + cross_order * length(cross_locs) + 2
    X = ones(nobs, ncols)
    yresp = Vector{Float64}(undef, nobs)
    for (row, t) in enumerate((maxlag + 1):T)
        yresp[row] = Y[t, li]
        col = 2
        for lag in 1:own_order
            X[row, col] = Y[t - lag, li]
            col += 1
        end
        for cl in cross_locs, lag in 1:cross_order
            X[row, col] = Y[t - lag, cl]
            col += 1
        end
        X[row, col] = sin(2 * pi * woy[t] / W)
        X[row, col + 1] = cos(2 * pi * woy[t] / W)
    end
    return X, yresp
end

"""
    fit_ridgevar_season(Y, own_order, cross_order, ridge_lambda, woy, W)
        -> Vector{NamedTuple}

Fit one ridge-penalised regression per location (`coef`, `resid_sd`,
plus enough of the design layout for [`simulate_paths`](@ref) to read
`coef` back correctly). `cross_locs` is every OTHER location, in
ascending index order, matching `build_design`.
"""
function fit_ridgevar_season(Y::Matrix{Float64}, own_order::Int,
        cross_order::Int, ridge_lambda::Float64, woy::Vector{Int}, W::Int)
    L = size(Y, 2)
    models = Vector{NamedTuple}(undef, L)
    for li in 1:L
        X, yresp = build_design(Y, li, own_order, cross_order, woy, W)
        coef = ridge_ls(X, yresp, ridge_lambda)
        resid = yresp .- X * coef
        dof = max(length(yresp) - size(X, 2), 1)
        resid_sd = sqrt(sum(abs2, resid) / dof)
        models[li] = (; coef, resid_sd, cross_locs=filter(!=(li), 1:L))
    end
    return models
end

"""
    simulate_paths(Y, models, own_order, cross_order, horizons, npaths,
                   woy_last, W; rng)
        -> Dict{Int,Matrix{Float64}}   # h -> (npaths x L)

Simulate `npaths` joint Gaussian-innovation sample paths forward from
the end of `Y`, for every location AT ONCE (needed because each
location's predictor set includes every other location's lags): at
each step, every location's next value is predicted from the CURRENT
tail (before any location in this step is updated), one independent
Normal(0, resid_sd) innovation is added per location, and only then are
all `L` tails advanced together.
"""
function simulate_paths(Y::Matrix{Float64}, models, own_order::Int,
        cross_order::Int, horizons, npaths::Int, woy_last::Int, W::Int;
        rng::Random.AbstractRNG)
    T, L = size(Y)
    hmax = maximum(horizons)
    maxlag = max(own_order, cross_order)
    out = Dict(h => Matrix{Float64}(undef, npaths, L) for h in horizons)
    tail0 = Y[(end - maxlag + 1):end, :]
    for s in 1:npaths
        tail = copy(tail0)
        for h in 1:hmax
            newrow = Vector{Float64}(undef, L)
            woy_h = mod1(woy_last + h, W)
            for li in 1:L
                m = models[li]
                pred = m.coef[1]
                col = 2
                for lag in 1:own_order
                    pred += m.coef[col] * tail[end - lag + 1, li]
                    col += 1
                end
                for cl in m.cross_locs, lag in 1:cross_order
                    pred += m.coef[col] * tail[end - lag + 1, cl]
                    col += 1
                end
                pred += m.coef[col] * sin(2 * pi * woy_h / W)
                pred += m.coef[col + 1] * cos(2 * pi * woy_h / W)
                newrow[li] = pred + m.resid_sd * randn(rng)
            end
            if h in horizons
                out[h][s, :] .= newrow
            end
            tail = vcat(tail[2:end, :], reshape(newrow, 1, L))
        end
    end
    return out
end

"""
    build_forecast_table(seasons) -> DataFrame

Fit and forecast the ridge-VAR+season mixture for every cross-
validation split of every season in `seasons`, returning the combined
hub quantile table (docs/contracts.md schema). Training discipline:
`build_model_data` caps each split's data at its own forecast origin,
and `window_weeks=104` further caps history to the most recent two
seasons -- identical discipline to nfidd-ar6 / seabbs_bot-ar6bf.
"""
function build_forecast_table(seasons)
    rng = MersenneTwister(SEED)
    rows = DataFrame(
        model_id=String[], location=String[], origin_date=Date[],
        horizon=Int[], target_end_date=Date[], target=String[],
        output_type=String[], output_type_id=Float64[], value=Float64[],
    )
    for season in seasons
        splits = training_splits(season)
        for split in splits
            data = build_model_data(
                split; Dmax=DMAX, transform=TRANSFORM,
                window_weeks=WINDOW_WEEKS,
            )
            Y = Float64.(data.Y)
            origin = data.origin_date
            models = fit_ridgevar_season(
                Y, OWN_ORDER, CROSS_ORDER, RIDGE_LAMBDA, data.woy, data.W,
            )
            paths = simulate_paths(
                Y, models, OWN_ORDER, CROSS_ORDER, HORIZONS, NPATHS,
                data.woy[end], data.W; rng=rng,
            )
            for h in HORIZONS
                target_end = origin + Day(7 * h)
                vals = paths[h]
                for (li, loc) in enumerate(LOCATIONS)
                    v = @view vals[:, li]
                    for q in QUANTILE_LEVELS
                        qval = quantile(v, q)
                        nat = max(from_scale(qval, TRANSFORM), 0.0)
                        push!(rows, (
                            MODEL_ID, loc, origin, h, target_end,
                            TARGET, "quantile", q, nat,
                        ))
                    end
                end
            end
        end
    end
    return rows
end

function main()
    hub_path = length(ARGS) >= 1 ? ARGS[1] : nothing
    t0 = time()
    forecast = build_forecast_table((1, 2))
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    if hub_path !== nothing
        write_submission(forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="seabbs_bot", model_abbr="mixridgevarseason",
            designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
