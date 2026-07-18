#!/usr/bin/env julia
# gen_season.jl -- TEST-SEASON (3, 4, 5) forecasts for seabbs_bot-season
# (per-location climatology + AR(6) + backfill, the merged hub PR #79
# submission), for the held-out test-season evaluation (reports/
# test-evaluation.md).
#
# Model logic is otherwise identical to
# experiments/simple-round/season/generate.jl (MODEL_ID
# "seabbs_bot-season", the validation-selected "climatology-backfill"
# form, mean validation WIS 0.3004): per-location climatology term
# (`build_climatology`, already rebuilt per split from only
# `origin_date < forecast_origin` -- no change needed there) + AR(6) +
# backfill correction, fourthroot scale.
#
# The ONE change from that file: the backfill revision profile there
# was estimated ONCE from a fixed `season_year <= 2016` cutoff (already
# leak-free for TEST-season scoring, since 2016 is strictly before every
# test-season origin -- see submissions/README.md's "Hub submissions
# PAUSED" note). This script instead rebuilds it PER SPLIT via the
# canonical `src/seasonal.jl` builder (`as_of < forecast_origin`), for
# the same reasons as gen_ar6bf.jl: strictly more correct (later test
# splits get their own elapsed history, not a profile frozen in 2016),
# and keeps every model in this evaluation on the same per-origin
# discipline.
#
# Usage: julia --project=<sismid-ili-turing repo> gen_season.jl

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

const MODEL_ID = "seabbs_bot-season"
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12
const WINDOW_WEEKS = 104
const SEASON_PERIOD = 52
const DELAY_CUTOFF = 8
const MIN_SUPPORT = 5

"""Identical to experiments/simple-round/season/generate.jl's function
of the same name -- per-location climatology, already per-origin
(restricted to `origin_date < forecast_origin` inside)."""
function build_climatology(
    loc_hist::DataFrame, forecast_origin::Date;
    period::Int=SEASON_PERIOD, smooth_window::Int=5,
)
    sub = loc_hist[loc_hist.origin_date .< forecast_origin, :]
    bins = [Float64[] for _ in 1:period]
    for row in eachrow(sub)
        b = mod1(week_of_season(row.origin_date), period)
        push!(bins[b], to_scale(row.wili, TRANSFORM))
    end
    allvals = reduce(vcat, bins; init=Float64[])
    overall = isempty(allvals) ? 0.0 : median(allvals)
    raw = [isempty(b) ? overall : median(b) for b in bins]
    half = smooth_window ÷ 2
    smoothed = similar(raw)
    for i in 1:period
        idxs = [mod1(i + o, period) for o in (-half):half]
        smoothed[i] = mean(raw[idxs])
    end
    return smoothed
end

"""Identical to experiments/simple-round/season/generate.jl's function
of the same name."""
function fit_ar_clim(
    y::AbstractVector{Float64}, woy::AbstractVector{Int}, order::Int,
    clim::Vector{Float64},
)
    n = length(y)
    nobs = n - order
    nobs >= order + 3 ||
        error("series too short for AR($order)+clim: n=$n, nobs=$nobs")
    ncols = order + 2
    X = ones(nobs, ncols)
    yresp = Vector{Float64}(undef, nobs)
    for (row, t) in enumerate((order + 1):n)
        yresp[row] = y[t]
        for lag in 1:order
            X[row, lag + 1] = y[t - lag]
        end
        X[row, ncols] = clim[mod1(woy[t], SEASON_PERIOD)]
    end
    coef = X \ yresp
    resid = yresp .- X * coef
    dof = max(nobs - ncols, 1)
    resid_sd = sqrt(sum(abs2, resid) / dof)
    return coef, resid_sd
end

"""Identical to experiments/simple-round/season/generate.jl's function
of the same name."""
function simulate_paths_clim(
    y::AbstractVector{Float64}, future_woy::Vector{Int},
    coef::Vector{Float64}, resid_sd::Float64, order::Int,
    clim::Vector{Float64}, horizons, npaths::Int;
    rng::Random.AbstractRNG,
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
            pred += coef[order + 2] * clim[mod1(future_woy[h], SEASON_PERIOD)]
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
    build_forecast_table(seasons, versions_full, hist_by_loc) -> DataFrame

TEST-SEASON climatology+AR(6)+backfill forecast table. Backfill profile
rebuilt PER SPLIT (leak-free, `src/seasonal.jl`); climatology already
per-origin. Everything else identical to seabbs_bot-season.
"""
function build_forecast_table(seasons, versions_full, hist_by_loc)
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
            origin = data.origin_date
            profile = build_revision_profile(
                versions_full, origin; transform=TRANSFORM,
                max_delay=DELAY_CUTOFF, min_support=MIN_SUPPORT,
                mode=:additive, stat=:median,
            )
            apply_backfill_correction!(
                data, profile; mode=:additive, delay_cutoff=DELAY_CUTOFF,
            )
            future_woy = [
                week_of_season(origin + Day(7 * h)) for h in HORIZONS
            ]
            for (li, loc) in enumerate(LOCATIONS)
                y = Float64.(data.Y[:, li])
                clim = build_climatology(hist_by_loc[loc], origin)
                coef, resid_sd = fit_ar_clim(y, data.woy, AR_ORDER, clim)
                paths = simulate_paths_clim(
                    y, future_woy, coef, resid_sd, AR_ORDER, clim,
                    HORIZONS, NPATHS; rng=rng,
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
    hist_full = load_series("flu_data_hhs")
    hist_by_loc = Dict(
        loc => hist_full[hist_full.location .== loc, [:origin_date, :wili]]
        for loc in LOCATIONS
    )
    forecast = build_forecast_table(TEST_SEASONS, versions_full, hist_by_loc)
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) TEST-season " *
            "origin date(s) in $(dt)s")
    outpath = joinpath(HERE, "out", "seabbs_bot-season.csv")
    CSV.write(outpath, forecast)
    println("wrote $(outpath)")
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
