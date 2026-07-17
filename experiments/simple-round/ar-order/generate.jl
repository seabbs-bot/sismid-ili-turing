#!/usr/bin/env julia
# generate.jl -- winning variant of the AR-ORDER sweep (sweep.jl, same
# directory): per-location AR(12) + the seabbs_bot-ar6bf backfill
# correction.
#
# Family: submissions/nfidd-ar6/generate_forecasts.jl (plain AR(p) per
# location, OLS, fourthroot, 1000 simulated paths -- CSV/DataFrames/
# Statistics/LinearAlgebra only, no `SismidILITuring`/Turing). This
# script only changes AR_ORDER (6 -> 12) relative to nfidd-ar6 and adds
# the unchanged backfill correction from seabbs_bot-ar6bf.
#
# Sweep result (sweep.jl, validation seasons 1-2 only, scored against
# the hub oracle): order=12+backfill was the best of 12 combinations
# (order in {2,4,6,8,10,12} x {no backfill, backfill}), mean WIS
# 0.3518 vs nfidd-ar6's 0.368 and seabbs_bot-ar6bf's 0.359. See
# score.txt for the full ranked table and the by-location/by-horizon/
# by-season breakdown. order=8+backfill (0.3525) and order=10+backfill
# (0.3534) are close seconds with marginally lower SD; order=12 was
# chosen because it has the best mean WIS, the primary ranking metric
# used throughout this repo (docs/contracts.md, nfidd-ar6/README.md).
#
# Generates all 5 seasons (1,2 validation + 3,4,5 test, as ar6bf) for
# the hub submission: each split is still just a per-origin vintage
# fit capped at its own forecast origin, so covering the test seasons
# at generation time never trains on or tunes against them -- the
# order/backfill choice itself was locked on the validation seasons
# only (sweep.jl).
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl <hub_path>
#
# With no `hub_path`, only builds and times the forecast table (no
# files written) -- matches nfidd-ar6/ar6bf's own driver convention.

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

const MODEL_ID = "ar-order-12bf"
const TRANSFORM = :fourthroot
const AR_ORDER = 12
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12
const DELAY_CUTOFF = 8
const MIN_SUPPORT = 5

# ---------------------------------------------------------------------
# Backfill correction -- unchanged from
# submissions/seabbs_bot-ar6bf/generate_forecasts.jl
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
# AR(12) fit + forecast -- identical in form to nfidd-ar6/ar6bf
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

"""
    build_forecast_table(seasons, profile, versions_full) -> DataFrame

Fit and forecast the AR(12)+backfill-correction model for every
cross-validation split of every season in `seasons`, returning the
combined hub quantile table (docs/contracts.md schema). Training
discipline: `build_model_data` caps each split's data at its own
forecast origin, and `window_weeks=104` further caps history to the
most recent two seasons. Seasons in `TEST_SEASONS` are fetched with
`allow_test_season=true` (as ar6bf): each split is still just a
per-origin vintage fit capped at its own forecast origin, not
training on or tuning against the test season -- the order/backfill
selection itself was made on the validation seasons only (sweep.jl).
"""
function build_forecast_table(seasons, profile, versions_full)
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
            apply_backfill_correction!(data, profile)
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

    # All 5 seasons for the submission (validation 1,2 + test 3,4,5,
    # as ar6bf): the order/backfill choice was selected on the
    # validation seasons only (sweep.jl), never on the test seasons.
    forecast = build_forecast_table((1, 2, 3, 4, 5), profile, versions_full)
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    if hub_path !== nothing
        write_submission(forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="seabbs_bot", model_abbr="arorder12bf",
            designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
