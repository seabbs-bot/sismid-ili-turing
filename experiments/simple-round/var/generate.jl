#!/usr/bin/env julia
# generate.jl -- winning variant of the multi-location VAR sweep
# (sweep.jl, same directory): a ridge-penalised VAR(1) across all 11
# locations, layered on top of each location's own AR(6).
#
# Family: submissions/nfidd-ar6/generate_forecasts.jl (plain per-
# location AR(p), OLS, fourthroot, 1000 simulated Gaussian-innovation
# paths -- CSV/DataFrames/Statistics/LinearAlgebra only, no
# `SismidILITuring`/Turing). This script keeps that per-location AR(6)
# unchanged and ADDS every other location's lag-1 value as an extra,
# ridge-penalised predictor.
#
# Sweep result (sweep.jl, validation seasons 1-2 only, scored against
# the hub oracle): a dense, UNPENALISED VAR(1) badly overfits
# (var1_ols: mean WIS 0.4214, WORSE than the plain AR(6) baseline's
# 0.3684) -- exactly what docs/eda/04-cross-location.md's warning
# about a dense 11-location VAR predicts. Ridge-penalising just the
# cross-location coefficients (own AR(6) lags are never penalised)
# fixes this: `lambda_cross=1.5` was the best of 8 ridge levels tried
# (0, 0.5, 1, 1.5, 2, 3, 5), mean WIS 0.3383, though 1.0-2.0 are all
# within Monte-Carlo noise of each other (0.3384-0.3389) -- a flat
# optimum, not a sharp one. A VAR(2) variant (also including each
# other location's lag-2 value) never beat the best VAR(1) variant, in
# line with docs/eda/04-cross-location.md's lead-lag finding (every
# region's cross-correlation peaks at lag 0, so there is no genuine
# lagged information to recover beyond lag 1). See score.txt for the
# full ranked table and the by-location/by-horizon/by-season
# breakdown against the plain AR(6) baseline.
#
# Generates all 5 seasons (1,2 validation + 3,4,5 test) for the hub
# submission: each split is still just a per-origin vintage fit capped
# at its own forecast origin, so covering the test seasons at
# generation time never trains on or tunes against them -- the ridge
# level was locked on the validation seasons only (sweep.jl).
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

const MODEL_ID = "var1-ridge"
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const RIDGE_LAMBDA = 1.5  # penalty on cross-location coefficients only
const NPATHS = 1000
const SEED = 20260717
const L = length(LOCATIONS)
const ALL_OTHER_IDX = [[j for j in 1:L if j != l] for l in 1:L]

# ---------------------------------------------------------------------
# Ridge-penalised VAR(1)-on-AR(6): each location's own AR(6), plus
# every OTHER location's lag-1 value with a ridge penalty.
# ---------------------------------------------------------------------

"""
    build_row(tail, l) -> Vector{Float64}

One predictor row for location `l`: intercept, `AR_ORDER` own lags
(most recent first), then the lag-1 value of every OTHER location.
`tail` holds the most recent `AR_ORDER` rows (all locations), most
recent LAST.
"""
function build_row(tail::AbstractMatrix{Float64}, l::Int)
    x = Float64[1.0]
    for k in 1:AR_ORDER
        push!(x, tail[end - k + 1, l])
    end
    for j in ALL_OTHER_IDX[l]
        push!(x, tail[end, j])
    end
    return x
end

"""
    fit_var1_ridge(Y, l) -> (coef, resid_sd)

Ridge fit of location `l`'s predictor row (`build_row`) against
`Y[:, l]`, over every time step with `AR_ORDER` lags of history
available. The intercept and the `AR_ORDER` own-AR coefficients are
never penalised; the `L - 1` cross-location coefficients are
penalised by `RIDGE_LAMBDA` (see the module docstring for why: an
unpenalised dense VAR(1) over 11 locations overfits badly).
"""
function fit_var1_ridge(Y::Matrix{Float64}, l::Int)
    T = size(Y, 1)
    nobs = T - AR_ORDER
    ncol = 1 + AR_ORDER + length(ALL_OTHER_IDX[l])
    X = Matrix{Float64}(undef, nobs, ncol)
    yresp = Vector{Float64}(undef, nobs)
    for (row, t) in enumerate((AR_ORDER + 1):T)
        X[row, :] = build_row((@view Y[(t - AR_ORDER):(t - 1), :]), l)
        yresp[row] = Y[t, l]
    end
    lam = fill(RIDGE_LAMBDA, ncol)
    lam[1:(1 + AR_ORDER)] .= 0.0  # intercept + own AR(6): unpenalised
    coef = (X'X + Diagonal(lam)) \ (X'yresp)
    resid = yresp .- X * coef
    dof = max(nobs - ncol, 1)
    resid_sd = sqrt(sum(abs2, resid) / dof)
    return coef, resid_sd
end

"""
    simulate_joint(Y, coefs, resid_sds, horizons, npaths; rng)
        -> Dict{Int,Vector{Vector{Float64}}}

Simulate `npaths` Gaussian-innovation sample paths forward from the
end of `Y`, JOINTLY across all `L` locations, for each horizon in
`horizons`: every location's step at time `t` uses the SAME simulated
tail (all locations' values at `t-1`), so cross-location feedback
propagates correctly into later horizons -- true VAR forward
simulation, not `L` independent per-location simulations bolted
together. Returns `out[h][l]`, a length-`npaths` vector of simulated
values for location `l` at horizon `h`.
"""
function simulate_joint(
    Y::Matrix{Float64}, coefs::Vector{Vector{Float64}},
    resid_sds::Vector{Float64}, horizons, npaths::Int;
    rng::Random.AbstractRNG,
)
    hmax = maximum(horizons)
    out = Dict(h => [Vector{Float64}(undef, npaths) for _ in 1:L]
               for h in horizons)
    tail0 = Y[(end - AR_ORDER + 1):end, :]
    for s in 1:npaths
        tail = copy(tail0)
        for h in 1:hmax
            newrow = Vector{Float64}(undef, L)
            for l in 1:L
                pred = dot(coefs[l], build_row(tail, l))
                newrow[l] = pred + resid_sds[l] * randn(rng)
            end
            if h in horizons
                for l in 1:L
                    out[h][l][s] = newrow[l]
                end
            end
            tail = vcat(tail[2:end, :], reshape(newrow, 1, L))
        end
    end
    return out
end

"""
    build_forecast_table(seasons) -> DataFrame

Fit and jointly forecast the ridge-VAR(1) model for every cross-
validation split of every season in `seasons`, returning the combined
hub quantile table (docs/contracts.md schema). Training discipline:
`build_model_data` caps each split's data at its own forecast origin,
and `window_weeks=104` further caps history to the most recent two
seasons.
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
            Y = Float64.(data.Y)
            coefs = Vector{Vector{Float64}}(undef, L)
            resid_sds = Vector{Float64}(undef, L)
            for l in 1:L
                coefs[l], resid_sds[l] = fit_var1_ridge(Y, l)
            end
            paths = simulate_joint(
                Y, coefs, resid_sds, HORIZONS, NPATHS; rng=rng,
            )
            for h in HORIZONS
                target_end = origin + Day(7 * h)
                for (li, loc) in enumerate(LOCATIONS)
                    vals = paths[h][li]
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

    # All 5 seasons for the submission (validation 1,2 + test 3,4,5):
    # the ridge level was selected on the validation seasons only
    # (sweep.jl), never on the test seasons.
    forecast = build_forecast_table((1, 2, 3, 4, 5))
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    if hub_path !== nothing
        write_submission(forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="seabbs_bot", model_abbr="var1ridge",
            designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
