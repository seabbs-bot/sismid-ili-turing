#!/usr/bin/env julia
# ar6_baseline.jl -- simple, fast, analytic AR(6) per-location baseline
# for the sismid-ili-turing hub session (nfidd-ar6).
#
# Deliberately simple: an independent AR(6) per location, no hierarchy,
# no seasonality term, no backfill model. Fit by ordinary least squares
# on the fourth-root-transformed vintage series (docs/lessons.md item 7
# in sismid-ili-turing: :fourthroot is the package's current favoured
# transform, not :log). Probabilistic forecasts come from simulating
# S=1000 Gaussian-innovation paths forward h=1..4 using the fitted
# residual standard deviation, taking the 23 hub quantile levels per
# (location, horizon), and back-transforming with `from_scale`, clamped
# at 0.
#
# Deliberately avoids `using SismidILITuring`: that module's `include`
# chain (src/SismidILITuring.jl) pulls in Turing/Mooncake/Pathfinder
# (src/model.jl, src/inference.jl) purely to fit an OLS regression,
# which is unnecessary weight on a shared, busy box. `src/core.jl`,
# `src/data.jl`, and `src/hubio.jl` are each documented as
# standalone-includable in their own header comments and bring in only
# CSV/DataFrames/Dates -- everything this driver needs.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> ar6_baseline.jl <hub_path>

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
const NPATHS = 1000
const SEED = 20260717

"""
    fit_ar(y, order) -> (coef, resid_sd)

OLS fit of an AR(`order`) model with intercept to `y` (ascending in
time, no missing values). `coef = [c, phi_1, ..., phi_order]`, where
`phi_1` multiplies the most recent lag (`y[t-1]`) and `phi_order` the
most distant (`y[t-order]`). `resid_sd` is the in-sample residual
standard deviation with `nobs - (order + 1)` degrees of freedom.
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
from the end of `y` (modelling scale), for each horizon in `horizons`.
Each path draws one fresh Normal(0, resid_sd) innovation per step and
feeds simulated values back in as lags for later horizons (proper
forward propagation of forecast uncertainty).
"""
function simulate_paths(
    y::AbstractVector{Float64}, coef::Vector{Float64}, resid_sd::Float64,
    order::Int, horizons, npaths::Int; rng::Random.AbstractRNG,
)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    tail0 = y[(end - order + 1):end]  # most recent `order` obs, ascending
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

"""
    build_forecast_table(seasons) -> DataFrame

Fit and forecast the AR(6) baseline for every cross-validation split of
every season in `seasons`, returning the combined hub quantile table
(docs/contracts.md schema in sismid-ili-turing). Training discipline:
`build_model_data` caps each split's data at its own forecast origin
(the tscv split itself never carries future/finalized values), and
`window_weeks=104` further caps history to the most recent two seasons.
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
                coef, resid_sd = fit_ar(y, AR_ORDER)
                paths = simulate_paths(
                    y, coef, resid_sd, AR_ORDER, HORIZONS, NPATHS;
                    rng=rng,
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
