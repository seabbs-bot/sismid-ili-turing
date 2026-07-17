#!/usr/bin/env julia
# ar6_fourier.jl -- AR(6) + Fourier seasonality per-location baseline
# for the sismid-ili-turing hub session (nfidd-ar6), replacing the plain
# AR(6) version already merged as PR #62.
#
# Per location, on the fourth-root scale: OLS regression of y_t on an
# intercept, 6 AR lags, and 3 Fourier harmonic pairs (sin/cos) of
# week-of-season with a 52-week period. Forecasts come from simulating
# 1000 Gaussian-innovation paths forward h=1..4, taking the 23 hub
# quantile levels, back-transforming and clamping at 0.
#
# Deliberately avoids `using SismidILITuring` -- see ar6_baseline.jl
# (the plain-AR6 predecessor) for the rationale; only src/core.jl,
# src/data.jl, src/hubio.jl are included, bringing in CSV/DataFrames/
# Dates only.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> ar6_fourier.jl <hub_path>

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

const MODEL_ID = "nfidd-ar6"
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const N_HARMONICS = 3
const SEASON_PERIOD = 52.0
const NPATHS = 1000
const SEED = 20260717

"""
    fourier_features(woy, K, period) -> Vector{Float64}

`2K` Fourier features `[sin(2*pi*1*woy/period), cos(2*pi*1*woy/period),
sin(2*pi*2*woy/period), cos(2*pi*2*woy/period), ...]` for `K` harmonics
of week-of-season `woy` at the given `period` (weeks).
"""
function fourier_features(woy::Real, K::Int, period::Float64)
    feats = Vector{Float64}(undef, 2K)
    for k in 1:K
        ang = 2 * pi * k * woy / period
        feats[2k - 1] = sin(ang)
        feats[2k] = cos(ang)
    end
    return feats
end

"""
    fit_ar_fourier(y, dates, order, K, period) -> (coef, resid_sd)

OLS fit of an AR(`order`) + `K`-harmonic Fourier seasonality model with
intercept. `coef = [c, phi_1, ..., phi_order, s_1, c_1, ..., s_K, c_K]`;
`phi_1` multiplies the most recent lag (`y[t-1]`), `s_k`/`c_k` multiply
the sin/cos Fourier features of week-of-season at time `t` (the response
time, not the lag times -- seasonality explains the level being
predicted, not the predictors). `dates[t]` must align with `y[t]`.
"""
function fit_ar_fourier(
    y::AbstractVector{Float64}, dates::AbstractVector{Date},
    order::Int, K::Int, period::Float64,
)
    n = length(y)
    length(dates) == n || error("y and dates must have the same length")
    nobs = n - order
    ncoef = 1 + order + 2K
    nobs >= ncoef + 2 ||
        error("series too short for AR($order)+Fourier($K): n=$n, " *
              "nobs=$nobs, ncoef=$ncoef")
    X = ones(nobs, ncoef)
    yresp = Vector{Float64}(undef, nobs)
    for (row, t) in enumerate((order + 1):n)
        yresp[row] = y[t]
        for lag in 1:order
            X[row, 1 + lag] = y[t - lag]
        end
        feats = fourier_features(week_of_season(dates[t]), K, period)
        X[row, (2 + order):(1 + order + 2K)] .= feats
    end
    coef = X \ yresp
    resid = yresp .- X * coef
    dof = max(nobs - ncoef, 1)
    resid_sd = sqrt(sum(abs2, resid) / dof)
    return coef, resid_sd
end

"""
    simulate_paths_fourier(y, coef, resid_sd, order, K, period, origin,
                            horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Simulate `npaths` Gaussian-innovation AR(`order`)+Fourier(`K`) sample
paths forward from the end of `y` (modelling scale), for each horizon in
`horizons`. The Fourier term at each simulated step uses the true
calendar week-of-season for `origin + 7*h` days (known in advance, not
simulated), so only the AR component propagates simulated uncertainty
forward.
"""
function simulate_paths_fourier(
    y::AbstractVector{Float64}, coef::Vector{Float64}, resid_sd::Float64,
    order::Int, K::Int, period::Float64, origin::Date, horizons,
    npaths::Int; rng::Random.AbstractRNG,
)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    tail0 = y[(end - order + 1):end]  # most recent `order` obs, ascending
    future_feats = [
        fourier_features(week_of_season(origin + Day(7h)), K, period)
        for h in 1:hmax
    ]
    for s in 1:npaths
        tail = copy(tail0)
        for h in 1:hmax
            pred = coef[1]
            for lag in 1:order
                pred += coef[lag + 1] * tail[end - lag + 1]
            end
            feats = future_feats[h]
            for j in 1:2K
                pred += coef[1 + order + j] * feats[j]
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

"""
    build_forecast_table(seasons) -> DataFrame

Fit and forecast the AR(6)+Fourier(3) baseline for every cross-validation
split of every season in `seasons`, returning the combined hub quantile
table (docs/contracts.md schema in sismid-ili-turing). Training
discipline: `build_model_data` caps each split's data at its own forecast
origin, and `window_weeks=104` further caps history to the most recent
two seasons.
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
                split; Dmax=12, transform=TRANSFORM, window_weeks=104,
            )
            origin = data.origin_date
            for (li, loc) in enumerate(LOCATIONS)
                y = Float64.(data.Y[:, li])
                coef, resid_sd = fit_ar_fourier(
                    y, data.dates, AR_ORDER, N_HARMONICS, SEASON_PERIOD,
                )
                paths = simulate_paths_fourier(
                    y, coef, resid_sd, AR_ORDER, N_HARMONICS,
                    SEASON_PERIOD, origin, HORIZONS, NPATHS; rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    vals = paths[h]
                    for q in QUANTILE_LEVELS
                        qval = quantile(vals, q)
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
            team_abbr="nfidd", model_abbr="ar6", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
