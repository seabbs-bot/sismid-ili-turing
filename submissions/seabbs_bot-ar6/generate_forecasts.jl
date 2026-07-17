#!/usr/bin/env julia
# generate_forecasts.jl -- full 5-season, correctly-named resubmission
# of the plain AR(6) per-location baseline (model_id "seabbs_bot-ar6";
# same fitting/forecasting logic as nfidd-ar6, PR #62 merged +
# nfidd-ar6 test-season follow-up PR #70).
#
# Renamed because the hub's model-metadata schema requires team_abbr to
# match ^[a-zA-Z0-9_+]+$ -- a hyphen is invalid, so "seabbs-bot" cannot
# be a team_abbr. "seabbs_bot" (underscore) is valid; model_id becomes
# "seabbs_bot-ar6" (the hyphen there is the model_id's own
# team_abbr-model_abbr separator, not part of either abbreviation).
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
# Covers all 5 seasons (2015/16 .. 2019/20): validation seasons 1, 2
# fit normally; test seasons 3, 4, 5 pass `allow_test_season=true` to
# `training_splits` -- each split is still capped at its own forecast
# origin by `build_model_data`, so this is a per-week vintage fit, not
# training on test-season outcomes (docs/contracts.md experimental
# integrity).
#
# Deliberately avoids `using SismidILITuring`: that module's `include`
# chain (src/SismidILITuring.jl) pulls in Turing/Mooncake/Pathfinder
# (src/model.jl, src/inference.jl) purely to fit an OLS regression,
# which is unnecessary weight on a shared, busy box. `src/core.jl`,
# `src/data.jl`, and `src/hubio.jl` are each documented as
# standalone-includable in their own header comments and bring in only
# CSV/DataFrames/Dates -- plus JSON here, to read the hub's own round
# list and filter out any origin dates the hub doesn't recognise (see
# `hub_round_dates`).
#
# Usage:
#   julia --project=scripts/validate generate_forecasts.jl <hub_path>
# (needs JSON in addition to CSV/DataFrames/Dates/Statistics/Random/
# LinearAlgebra; scripts/validate's environment already has it)

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra
using JSON

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))
include(joinpath(PKG_DIR, "src", "hubio.jl"))

const MODEL_ID = "seabbs_bot-ar6"
const TEAM_ABBR = "seabbs_bot"
const MODEL_ABBR = "ar6"
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const ALL_SEASONS = (1, 2, 3, 4, 5)
const TEST_SEASONS_SET = (3, 4, 5)

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
(docs/contracts.md schema). Training discipline: `build_model_data`
caps each split's data at its own forecast origin, and
`window_weeks=104` further caps history to the most recent two
seasons. `allow_test_season=true` is passed for `season in
TEST_SEASONS_SET` (3, 4, 5) -- deliberate, see file header.
"""
function build_forecast_table(seasons)
    rng = MersenneTwister(SEED)
    rows = DataFrame(
        model_id=String[], location=String[], origin_date=Date[],
        horizon=Int[], target_end_date=Date[], target=String[],
        output_type=String[], output_type_id=Float64[], value=Float64[],
    )
    for season in seasons
        allow_test = season in TEST_SEASONS_SET
        splits = training_splits(season; allow_test_season=allow_test)
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

"""
    hub_round_dates(hub_path) -> Set{Date}

The forecast-origin dates the hub's `hub-config/tasks.json` actually
recognises. Some cross-validation splits extend past the hub's round
list (season 5's 2019/20 splits run to 2020-05-02; the hub's rounds
stop at 2020-03-21, likely because 2019/20 ILI surveillance was
disrupted by COVID-19 onset) -- forecasts for those dates would
reference a round the hub doesn't have, so they must be filtered out
before writing the submission. See docs/contracts.md.
"""
function hub_round_dates(hub_path::AbstractString)::Set{Date}
    path = joinpath(hub_path, "hub-config", "tasks.json")
    cfg = JSON.parsefile(path)
    task = cfg["rounds"][1]["model_tasks"][1]["task_ids"]
    return Set(Date(d) for d in task["origin_date"]["optional"])
end

function main()
    hub_path = length(ARGS) >= 1 ? ARGS[1] : nothing
    t0 = time()
    forecast = build_forecast_table(ALL_SEASONS)
    dt = round(time() - t0; digits=2)
    n_origins_built = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins_built) " *
            "origin date(s) in $(dt)s (seasons $(ALL_SEASONS))")

    if hub_path !== nothing
        allowed = hub_round_dates(hub_path)
        built_dates = Set(forecast.origin_date)
        excluded = sort(collect(setdiff(built_dates, allowed)))
        if !isempty(excluded)
            println("excluding $(length(excluded)) origin date(s) not " *
                     "in the hub's round list: $excluded")
            forecast = forecast[in.(forecast.origin_date, Ref(allowed)), :]
        end
        n_origins_final = length(unique(forecast.origin_date))
        println("writing $(nrow(forecast)) rows across " *
                 "$(n_origins_final) origin date(s)")

        write_submission(forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr=TEAM_ABBR, model_abbr=MODEL_ABBR, designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
