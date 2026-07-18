#!/usr/bin/env julia
# gen_ar6bf.jl -- TEST-SEASON (3, 4, 5) forecasts for nfidd-ar6bf
# (AR(6) + backfill correction), for the held-out test-season
# evaluation (reports/test-evaluation.md).
#
# Model logic is otherwise identical to
# submissions/seabbs_bot-ar6bf/generate_forecasts.jl (AR(6), fourthroot
# scale, per-(location,delay) backfill correction), BUT the backfill
# revision profile there was estimated ONCE from a fixed
# `season_year <= 2016` cutoff and reused unchanged across every split.
# For scoring the TEST seasons (2017/18-2019/20) that fixed cutoff is
# already leak-free (2016 is strictly before every test-season origin --
# see submissions/README.md's "Hub submissions PAUSED" note), but this
# script instead rebuilds the profile PER SPLIT from only
# `as_of < forecast_origin` via the canonical leak-free builder in
# `src/seasonal.jl` (`build_revision_profile`/`apply_backfill_correction!`),
# for two reasons: (1) it is strictly more correct -- later test splits
# get to use more of their own already-elapsed history, not a profile
# frozen in 2016 -- and (2) it keeps every model in this evaluation on
# the same, auditable, per-origin discipline (docs/steer-log.md instruction
# for this report), rather than mixing a fixed-cutoff model in with the
# rest.
#
# Usage: julia --project=<sismid-ili-turing repo> gen_ar6bf.jl

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
include(joinpath(PKG_DIR, "src", "seasonal.jl"))
# ^ per-origin build_revision_profile / apply_backfill_correction!

const MODEL_ID = "nfidd-ar6bf"
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12
const DELAY_CUTOFF = 8   # weeks; matches seabbs_bot-ar6bf
const MIN_SUPPORT = 5
const SMOOTH_WINDOW = 3  # unused by the revision profile, kept for parity

"""Identical to submissions/seabbs_bot-ar6bf/generate_forecasts.jl's
function of the same name."""
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

"""Identical to submissions/seabbs_bot-ar6bf/generate_forecasts.jl's
function of the same name."""
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
    build_forecast_table(seasons, versions_full) -> DataFrame

TEST-SEASON AR(6)+backfill forecast table. The backfill revision
profile is rebuilt PER SPLIT from `versions_full` restricted to
`as_of < forecast_origin` (`src/seasonal.jl`'s `build_revision_profile`)
-- leak-free -- then applied with `apply_backfill_correction!` before
the per-location AR(6) fit. Everything else (AR order, transform,
window, simulation) is identical to seabbs_bot-ar6bf.
"""
function build_forecast_table(seasons, versions_full::DataFrame)
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
                split; Dmax=DMAX, transform=TRANSFORM, window_weeks=104,
                versions=versions_full,
            )
            origin = data.origin_date
            profile = build_revision_profile(
                versions_full, origin; transform=TRANSFORM,
                max_delay=DELAY_CUTOFF, min_support=MIN_SUPPORT,
                mode=:additive, stat=:median,
            )
            apply_backfill_correction!(
                data, profile; mode=:additive, delay_cutoff=DELAY_CUTOFF,
            )
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
    versions_full = load_series("flu_data_hhs_versions")
    forecast = build_forecast_table(TEST_SEASONS, versions_full)
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) TEST-season " *
            "origin date(s) in $(dt)s")
    outpath = joinpath(HERE, "out", "nfidd-ar6bf.csv")
    CSV.write(outpath, forecast)
    println("wrote $(outpath)")
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
