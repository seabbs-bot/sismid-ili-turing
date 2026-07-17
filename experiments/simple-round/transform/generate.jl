#!/usr/bin/env julia
# generate.jl -- simple, fast, analytic AR(6) per-location baseline for
# the simple-model wide round (sismid-ili-turing), TRANSFORM family.
#
# Identical in every respect to
# submissions/nfidd-ar6/generate_forecasts.jl (independent OLS-fit
# AR(6) per location, Gaussian-innovation path simulation, no
# hierarchy/seasonality/backfill) EXCEPT the modelling-scale transform:
#
#   nfidd-ar6:    fourth-root (`:fourthroot`), every location
#   this file:    log (`:log`), every location, EXCEPT HHS Region 9
#                 which uses sqrt (`:sqrt`)
#
# See sweep.jl in this directory for the full sweep this choice comes
# from (identity, sqrt, fourth-root, log, log1p, a pooled fitted
# Box-Cox power, a fully-per-location fitted Box-Cox power, and two
# log+Region-9-override mixes), and score.txt for the validation-season
# WIS results. Summary: `log` (not `fourthroot`, the EDA's flattest-
# variance pick) wins on WIS by ~4%, because WIS is scored on the
# NATURAL scale and rewards a tighter fit to the bulk of the
# distribution more than perfectly flat variance does; a fully
# per-location Box-Cox power is numerically unstable (many locations'
# fitted powers are negative, and the unshifted `w^p` family diverges
# near-zero for p < 0) and scores far worse than any single global
# choice; overriding just HHS Region 9 -- the one location the EDA
# consistently flags as needing a higher, more Poisson-like power
# (docs/eda/07-region9-deepdive.md) -- with `sqrt` gives a further
# small, real improvement with no added instability.
#
# Deliberately avoids `using SismidILITuring`: see
# submissions/nfidd-ar6/generate_forecasts.jl's header for why (Turing/
# Mooncake/Pathfinder are unnecessary weight for an OLS regression on a
# shared, busy box). Only CSV/DataFrames/Dates/Statistics/Random/
# LinearAlgebra.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl <hub_path> [seasons]
#
# <hub_path>: hub clone to write model-output/model-metadata into
#   (omit to just build and print row counts, no write).
# [seasons]: comma-separated season ids, default "1,2,3,4,5" (full hub
#   coverage: validation + the three held-out test seasons). Model
#   selection itself used ONLY seasons 1,2 -- see score.txt.

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

const MODEL_ID = "ar6-log-r9sqrt"
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717

# `:sqrt` is not one of `src/core.jl`'s `TRANSFORMS` (only `log`,
# `log1p`, `logit`, `fourthroot` are). Adding it here mutates only this
# process's in-memory `TRANSFORMS` Dict, not the file on disk -- safe
# alongside other agents editing the repo concurrently this round (see
# sweep.jl's `power_pair` for the general power-transform family this
# is drawn from).
TRANSFORMS[:sqrt] = (w -> sqrt(max(w, 0.0)), x -> x >= 0 ? x^2 : 0.0)

# Likewise `:identity` (raw wILI, no rescaling): `build_forecast_table`
# below always builds `ModelData` at `:identity` and applies each
# location's own transform (`transform_for`) column-wise afterwards,
# since `build_model_data` only accepts one shared transform per call.
TRANSFORMS[:identity] = (w -> Float64(w), x -> x)

"""Per-location modelling-scale transform: `:log` everywhere except
`HHS Region 9`, which uses `:sqrt` (see file header)."""
transform_for(loc::AbstractString) = loc == "HHS Region 9" ? :sqrt : :log

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

Data is built once at `transform=:identity` (raw wILI) and then each
location's column is mapped to its own modelling scale via
`transform_for` -- `:log` everywhere except a `:sqrt` override for HHS
Region 9 -- rather than passing one shared `transform` Symbol into
`build_model_data` as nfidd-ar6 does, since that function only accepts
a single transform for the whole call.
"""
function build_forecast_table(seasons)
    rng = MersenneTwister(SEED)
    rows = DataFrame(
        model_id=String[], location=String[], origin_date=Date[],
        horizon=Int[], target_end_date=Date[], target=String[],
        output_type=String[], output_type_id=Float64[], value=Float64[],
    )
    for season in seasons
        # `training_splits` refuses TEST_SEASONS (3, 4, 5) unless told
        # explicitly (docs/brief.md experimental integrity guard in
        # src/data.jl); the model/transform choice above was selected
        # on VALIDATION_SEASONS (1, 2) ONLY (see score.txt), so passing
        # test seasons here is for full hub coverage at generation time
        # only, matching seabbs_bot-ar6bf's pattern -- it never feeds
        # back into model selection.
        splits = training_splits(
            season; allow_test_season=(season in TEST_SEASONS),
        )
        for split in splits
            data = build_model_data(
                split; Dmax=12, transform=:identity, window_weeks=104,
            )
            origin = data.origin_date
            for (li, loc) in enumerate(LOCATIONS)
                tform = transform_for(loc)
                y = to_scale.(Float64.(data.Y[:, li]), tform)
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
                        nat = max(from_scale(qval, tform), 0.0)
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
    # Full 5-season coverage for the hub write (validation 1,2 + the
    # three held-out test seasons 3,4,5); scoring/selection (score.txt)
    # uses seasons (1, 2) only, never these test seasons.
    seasons = length(ARGS) >= 2 ? Tuple(parse.(Int, split(ARGS[2], ","))) :
        (1, 2, 3, 4, 5)
    t0 = time()
    forecast = build_forecast_table(seasons)
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    if hub_path !== nothing
        write_submission(forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="nfidd", model_abbr="ar6logr9", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
