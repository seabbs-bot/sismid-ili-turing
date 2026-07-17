#!/usr/bin/env julia
# sweep.jl -- AR ORDER sweep for the simple-model wide round.
#
# Same family as submissions/nfidd-ar6/generate_forecasts.jl (plain
# per-location AR(p), OLS, fourthroot, 1000 simulated paths): the only
# thing varied here is the AR order p, plus one crossed lever (the
# empirical backfill correction from submissions/seabbs_bot-ar6bf, WIS
# 0.359 vs nfidd-ar6's 0.368 baseline).
#
# Deliberately avoids `using SismidILITuring` (Turing/Mooncake/Pathfinder
# weight, see nfidd-ar6's own header comment) -- CSV/DataFrames/Dates/
# Statistics/LinearAlgebra/ScoringRules only. `src/scoring.jl` needs
# `ScoringRules`, a scoring-only package with no Turing dependency.
#
# Sweeps p in {2, 4, 6, 8, 10, 12}, crossed with {no backfill, backfill},
# fits and forecasts every split of the VALIDATION seasons only (1, 2 --
# docs/contracts.md experimental integrity), and scores against the hub
# oracle (`target-data/oracle-output.csv` in the local hub clone).
#
# Usage:
#   julia --project=<sismid-ili-turing repo> sweep.jl

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra
using ScoringRules

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))
include(joinpath(PKG_DIR, "src", "hubio.jl"))
include(joinpath(PKG_DIR, "src", "scoring.jl"))

const HUB_PATH = joinpath(
    homedir(), "code", "external", "sismid-ili-forecasting-sandbox",
)
const TRANSFORM = :fourthroot
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12
const DELAY_CUTOFF = 8
const MIN_SUPPORT = 5
const ORDERS = [2, 4, 6, 8, 10, 12]
const VALIDATION = (1, 2)

# ---------------------------------------------------------------------
# AR(p) fit + forecast -- identical in form to nfidd-ar6 / ar6bf, just
# parameterised on `order`.
# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------
# Backfill correction -- lifted unchanged from
# submissions/seabbs_bot-ar6bf/generate_forecasts.jl, crossed here with
# AR order rather than isolated on its own.
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile on the `transform` scale
(docs/eda/02-backfill.md). `versions` must already be filtered by the
caller to training-set origin dates only (no test seasons).
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

Nudge `data.Y` in place at every `(t, l)` with `0 <= delay <=
DELAY_CUTOFF` and a matching `profile` entry.
"""
function apply_backfill_correction!(
    data::ModelData, profile::Dict{Tuple{String,Int},Float64},
)
    for l in 1:data.L, t in 1:data.T
        d = data.delay[t, l]
        (d < 0 || d > DELAY_CUTOFF) && continue
        ismissing(data.Y[t, l]) && continue
        key = (LOCATIONS[l], d)
        haskey(profile, key) || continue
        data.Y[t, l] += profile[key]
    end
    return data
end

# ---------------------------------------------------------------------
# Forecast table builder, parameterised on AR order + optional profile
# ---------------------------------------------------------------------

"""
    build_forecast_table(model_id, seasons, order; profile, versions_full)
        -> DataFrame

Fit and forecast an AR(`order`) baseline (with backfill correction
applied first if `profile !== nothing`) for every split of every season
in `seasons`.
"""
function build_forecast_table(
    model_id, seasons, order::Int;
    profile::Union{Nothing,Dict{Tuple{String,Int},Float64}}=nothing,
    versions_full::Union{Nothing,DataFrame}=nothing,
)
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
                split; Dmax=DMAX, transform=TRANSFORM, window_weeks=104,
                versions=versions_full,
            )
            profile !== nothing && apply_backfill_correction!(data, profile)
            origin = data.origin_date
            for (li, loc) in enumerate(LOCATIONS)
                y = Float64.(data.Y[:, li])
                coef, resid_sd = fit_ar(y, order)
                paths = simulate_paths(
                    y, coef, resid_sd, order, HORIZONS, NPATHS; rng=rng,
                )
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

"""Hub oracle (`target-data/oracle-output.csv`) as a scoring truth table."""
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

function main()
    t0 = time()
    truth = load_oracle(HUB_PATH)

    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=DELAY_CUTOFF,
        min_support=MIN_SUPPORT,
    )
    println("revision profile: $(length(profile)) (location, delay) " *
            "entries with >= $(MIN_SUPPORT) observations")

    results = DataFrame(
        order=Int[], backfill=Bool[], mean_wis=Float64[], sd_wis=Float64[],
    )
    scored_by_variant = Dict{Tuple{Int,Bool},DataFrame}()

    for order in ORDERS, backfill in (false, true)
        model_id = "ar$(order)" * (backfill ? "bf" : "")
        prof = backfill ? profile : nothing
        forecast = build_forecast_table(
            model_id, VALIDATION, order; profile=prof,
            versions_full=versions_full,
        )
        scored = score_forecasts(forecast, truth; scale=:natural)
        summ = wis_summary(scored)
        push!(results, (order, backfill, summ.mean_wis[1], summ.sd_wis[1]))
        scored_by_variant[(order, backfill)] = scored
        println("order=$(order) backfill=$(backfill): " *
                "mean_wis=$(round(summ.mean_wis[1]; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis[1]; digits=4)) " *
                "($(round(time() - t0; digits=1))s elapsed)")
    end

    sort!(results, :mean_wis)
    println("\n=== ranked (validation seasons $(VALIDATION)) ===")
    println(results)

    best = results[1, :]
    println("\nbest: order=$(best.order) backfill=$(best.backfill) " *
            "mean_wis=$(round(best.mean_wis; digits=4)) " *
            "sd_wis=$(round(best.sd_wis; digits=4))")
    println("baselines: nfidd-ar6 (order=6, no backfill) = 0.368; " *
            "seabbs_bot-ar6bf (order=6, backfill) = 0.359")

    # Breakdown of the best variant by location and horizon.
    best_scored = scored_by_variant[(best.order, best.backfill)]
    by_loc = combine(groupby(best_scored, :location),
        :wis => mean => :mean_wis, nrow => :n)
    sort!(by_loc, :mean_wis)
    println("\n=== best variant: mean WIS by location ===")
    println(by_loc)

    by_h = combine(groupby(best_scored, :horizon),
        :wis => mean => :mean_wis, nrow => :n)
    sort!(by_h, :horizon)
    println("\n=== best variant: mean WIS by horizon ===")
    println(by_h)

    # Also break down by season (origin year) since only 2 validation
    # seasons are in scope.
    best_scored.season_year = season_year.(best_scored.origin_date)
    by_season = combine(groupby(best_scored, :season_year),
        :wis => mean => :mean_wis, nrow => :n)
    sort!(by_season, :season_year)
    println("\n=== best variant: mean WIS by season ===")
    println(by_season)

    println("\ntotal time: $(round(time() - t0; digits=1))s")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
