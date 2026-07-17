#!/usr/bin/env julia
# simple-round candidate "intervals": tunes the PREDICTIVE-INTERVAL /
# uncertainty scheme for the simple AR(6)+backfill model, holding the
# POINT forecast fixed. Point forecast (unchanged from
# submissions/seabbs_bot-ar6bf/generate_forecasts.jl): independent
# AR(6) per location, fit by OLS on the fourth-root-transformed,
# backfill-corrected vintage series, no hierarchy, no seasonality term.
#
# Baseline to beat: seabbs_bot-ar6bf's own Gaussian-innovation
# `simulate_paths` scheme, mean WIS (natural, validation seasons 1,2)
# = 0.359.
#
# Schemes tried (see score.txt for the full sweep; this file only
# implements the winner):
#   - Gaussian innovations at the raw fitted `resid_sd` (reproduces the
#     0.359 baseline exactly) and at inflation scales 0.7-2.5.
#   - Student-t innovations (df 3-20), variance-matched to `resid_sd`
#     so a given df's spread is comparable to the Gaussian case, at the
#     same scale grid.
#   - Empirical (iid) bootstrap of each fit's own in-sample AR(6)
#     residuals, at the same scale grid.
#   - A per-horizon variance-growth multiplier on top of the best flat
#     scale (tests whether later horizons need extra widening beyond
#     what the AR recursion already gives them for free).
#
# Result: ALL THREE distributional families (Gaussian, Student-t,
# bootstrap) are essentially flat-uninformative at unit scale -- the
# raw fitted `resid_sd` under-covers substantially (50% nominal
# interval covers ~41%, 90% nominal covers ~78%), so every family's
# mean WIS keeps falling as its dispersion is inflated well past
# scale=1.0, bottoming out around scale~1.4-1.5 for all three. That
# bottoming point is WHERE variance inflation trades off correctly
# against interval-width penalty (sharpness); past it, WIS gets worse
# again as dispersion overshoots (see score.txt scale=2.0+ rows).
#
# At its own optimal scale each family reaches a similar WIS (Gaussian
# 0.3491 @ s=1.4, bootstrap 0.3489 @ s=1.5), but STUDENT-T WINS
# outright: at moderate df (8-20 all near-identical) and scale~1.4, it
# reaches 0.3481-0.3483 -- both the best raw score and, unlike the
# Gaussian/bootstrap optima (which land noticeably OVER-covered,
# cov50~0.53-0.57 vs nominal 0.50), the best-CALIBRATED point tested:
# cov50=0.512, cov90=0.892 against nominal 0.50/0.90. The heavier tail
# does the work that a Gaussian needs raw over-dispersion to fake, so
# less inflation is needed overall and the interval ends up both
# sharper AND better calibrated. df is a flat direction in this range
# (8, 10, 12, 15, 20 all within ~0.0002 mean WIS of each other); df=10
# is used below as an unremarkable, commonly-used choice within that
# flat region, not because it is uniquely best.
#
# Per-horizon variance growth was tested on top of the best flat scale
# and made things WORSE monotonically (0.3509 -> 0.3568 as the growth
# factor increased) -- a genuine null result, not a knob left at zero
# for lack of trying: per-horizon coverage at the chosen scheme is
# already flat across h=1..4 (cov50 in [0.512, 0.553], cov90 in
# [0.884, 0.906], see score.txt), so the AR(6) recursion already
# propagates uncertainty across horizons correctly on its own; an
# added multiplicative horizon term only over-widens the horizons that
# did not need it.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl <hub_path>

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra
using Distributions

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))
include(joinpath(PKG_DIR, "src", "hubio.jl"))

const MODEL_ID = "simple-ar6bf-t10"
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12
const DELAY_CUTOFF = 8   # weeks; profile is ~0 beyond this, docs/eda/02
const MIN_SUPPORT = 5    # min sample size per (location, delay) to trust

# Tuned interval scheme: Student-t(df) innovations, variance-matched to
# the fitted `resid_sd` (so `T_SCALE=1.0` alone would have the same
# spread as the Gaussian baseline), then inflated by `T_SCALE`. Chosen
# by validation-seasons-only sweep, see score.txt.
const T_DF = 10
const T_SCALE = 1.4

# ---------------------------------------------------------------------
# Backfill correction profile (verbatim from seabbs_bot-ar6bf)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile on the `transform` scale,
identical to `seabbs_bot-ar6bf`'s function of the same name -- see
that file's docstring for the full derivation
(docs/eda/02-backfill.md).
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

Nudge `data.Y` in place, identical to `seabbs_bot-ar6bf`.
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
# Fixed point forecast: AR(6) OLS (unchanged from seabbs_bot-ar6bf)
# ---------------------------------------------------------------------

"""
    fit_ar(y, order) -> (coef, resid_sd)

OLS fit of an AR(`order`) model with intercept, identical to
`seabbs_bot-ar6bf`/`nfidd-ar6`. `coef = [c, phi_1, ..., phi_order]`,
`resid_sd` the in-sample residual standard deviation.
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

# ---------------------------------------------------------------------
# Tuned interval scheme: Student-t(T_DF), variance-matched then scaled
# ---------------------------------------------------------------------

"""
    simulate_paths(y, coef, resid_sd, order, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Simulate `npaths` sample paths forward, exactly as
`seabbs_bot-ar6bf`'s function of the same name EXCEPT the innovation
at each step is drawn from a Student-t(`T_DF`) distribution,
variance-matched to `resid_sd` (`Var(T_DF) = df / (df - 2)`, so the
raw variance-matched draw alone reproduces the Gaussian case's
spread), then inflated by `T_SCALE`. Both `T_DF` and `T_SCALE` were
selected by a validation-seasons-only sweep (score.txt); the point
forecast (the AR(6) recursion itself) is untouched.
"""
function simulate_paths(
    y::AbstractVector{Float64}, coef::Vector{Float64}, resid_sd::Float64,
    order::Int, horizons, npaths::Int; rng::Random.AbstractRNG,
)
    vscale = sqrt((T_DF - 2) / T_DF)
    innov_sd = resid_sd * vscale * T_SCALE
    tdist = TDist(T_DF)

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
            val = pred + innov_sd * rand(rng, tdist)
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

Fit and forecast the AR(6)+backfill+tuned-interval model for every
cross-validation split of every season in `seasons`. Structurally
identical to `seabbs_bot-ar6bf`'s function of the same name (see that
file for the training-discipline notes: `build_model_data` caps each
split at its own forecast origin, `window_weeks=104`, and the revision
profile is estimated training-set-only). Seasons in `TEST_SEASONS` are
fetched with `allow_test_season=true`, exactly as `seabbs_bot-ar6bf`
does: each split is still just a per-origin vintage fit capped at its
own forecast origin, not training on the test season, and neither
`T_DF`/`T_SCALE` nor the point forecast were tuned against them --
tuning/scoring used seasons (1, 2) only (score.txt).
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

    # T_DF/T_SCALE (and the point forecast) were selected on VALIDATION
    # seasons (1, 2) ONLY (docs/contracts.md experimental integrity,
    # score.txt). Generation here still covers all five seasons, same
    # as seabbs_bot-ar6bf: each split is an independent per-origin
    # vintage fit, so covering the held-out test seasons at generation
    # time never trains on or tunes against them.
    forecast = build_forecast_table((1, 2, 3, 4, 5), profile, versions_full)
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    if hub_path !== nothing
        write_submission(forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="seabbs_bot", model_abbr="ar6bft10", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
