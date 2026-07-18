#!/usr/bin/env julia
# gen_ar6.jl -- TEST-SEASON (3, 4, 5) forecasts for the plain nfidd-ar6
# baseline, for the held-out test-season evaluation (reports/
# test-evaluation.md). Model logic is verbatim from
# submissions/nfidd-ar6/generate_forecasts.jl (independent AR(6) per
# location, fourthroot scale, no seasonality, no backfill -- nothing to
# leak). The only change from that file: `training_splits` is called
# with `allow_test_season=true` for seasons 3-5, and only those three
# seasons are generated (this is a REPORTING run against the locked
# selection, not a new submission).
#
# Model SELECTION happened on validation seasons (1, 2) only, per
# docs/brief.md/docs/contracts.md; this script does not tune anything,
# it reruns the already-locked nfidd-ar6 design on the held-out test
# seasons for scoring in reports/test-evaluation.md.
#
# Usage: julia --project=<sismid-ili-turing repo> gen_ar6.jl

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

const MODEL_ID = "nfidd-ar6"
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717

"""Identical to submissions/nfidd-ar6/generate_forecasts.jl's function of
the same name."""
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

"""Identical to submissions/nfidd-ar6/generate_forecasts.jl's function of
the same name."""
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

"""
    build_forecast_table(seasons) -> DataFrame

TEST-SEASON version of submissions/nfidd-ar6/generate_forecasts.jl's
function of the same name: identical model, but `training_splits` is
called with `allow_test_season=true` so seasons 3-5 are accepted. Each
split is still just a per-origin vintage fit capped at its own forecast
origin -- no test-season data is fit to or tuned against, only scored.
"""
function build_forecast_table(seasons)
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
    t0 = time()
    forecast = build_forecast_table(TEST_SEASONS)
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) TEST-season " *
            "origin date(s) in $(dt)s")
    outpath = joinpath(HERE, "out", "nfidd-ar6.csv")
    CSV.write(outpath, forecast)
    println("wrote $(outpath)")
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
