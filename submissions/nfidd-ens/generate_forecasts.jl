#!/usr/bin/env julia
# nfidd-ens -- ENSEMBLE submission: plain average of four simple,
# independent per-location analytic models (fourthroot scale):
#   ar6         -- plain AR(6), OLS + Gaussian-innovation path sim
#                  (submissions/nfidd-ar6)
#   ar6bf       -- same AR(6) but fit on backfill-corrected vintage
#                  (submissions/seabbs_bot-ar6bf)
#   climatology -- seasonal-naive: empirical quantiles of history at
#                  the same week-of-season (+/- a small band), no
#                  fitting
#   ses         -- simple exponential smoothing (level only), Gaussian
#                  path sim with variance growing with horizon
#
# Selected on VALIDATION SEASONS (1, 2) ONLY in
# experiments/simple-round/ensemble/{generate.jl,score.txt}: this
# ens-mean combination scored 0.2902 mean WIS (sd 0.3334) there, vs
# 0.359 for ar6bf alone and 0.368 for ar6 alone -- a ~19% improvement,
# and the most stable of any variant tried (lowest WIS SD). See that
# score.txt for the full breakdown (by location/horizon/season) and
# for the ens-median / ens-wis-weighted alternatives that did worse.
#
# LIGHT + ANALYTIC: no Turing. Standalone, like nfidd-ar6/ar6bf:
# includes src/core.jl, src/data.jl, src/hubio.jl, src/scoring.jl (the
# last for parity with the validation sweep; this driver only
# generates and writes, it does not score).
#
# Combination: plain, unweighted mean of the four members' quantile
# VALUES at each (location, horizon, target_end_date, quantile level)
# -- provably monotone in the quantile level (the mean of several
# componentwise-ordered, non-decreasing sequences is itself
# non-decreasing), so no re-sorting is needed.
#
# Coverage: all 5 seasons (1,2 validation + 3,4,5 held-out test).
# `allow_test_season=true` on `training_splits` for seasons in
# TEST_SEASONS governs the TUNING gate only (docs/contracts.md
# experimental integrity); the member designs and the ens-mean
# combination were locked using validation data only, and every
# split's fit here is still capped at its own forecast origin via
# `build_model_data` -- generating a forecast for a test-season split
# never trains on or tunes against that season. The backfill revision
# profile is estimated ONLY from `season_year <= 2016` origin dates
# (pre-2015 history plus the two validation seasons), exactly as
# `submissions/seabbs_bot-ar6bf/generate_forecasts.jl`.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate_forecasts.jl <hub_path>

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
include(joinpath(PKG_DIR, "src", "scoring.jl"))

const MODEL_ID = "nfidd-ens"
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12              # matches ar6bf's build_model_data Dmax
const DELAY_CUTOFF = 8       # ar6bf's chosen backfill window
const MIN_SUPPORT = 5        # min sample size per profile key to trust
const CLIMO_BAND = 2         # +/- weeks-of-season pooled for climatology
const CLIMO_MIN_N = 8        # min pooled n before falling back to full history
const SES_ALPHAS = 0.1:0.1:0.9

# ---------------------------------------------------------------------
# ar6 / ar6bf members: AR(6) OLS fit + Gaussian path simulation, and
# the ar6bf backfill correction (identical to their submissions).
# ---------------------------------------------------------------------

"""
    fit_ar(y, order) -> (coef, resid_sd)

OLS fit of an AR(`order`) model with intercept to `y` (ascending in
time, no missing values). `coef = [c, phi_1, ..., phi_order]`,
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
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile on the `transform` scale,
identical to `submissions/seabbs_bot-ar6bf/generate_forecasts.jl`: for
each `(location, delay)` with at least `min_support` recorded
versions at that delay, the median of `to_scale(settled, transform) -
to_scale(vintage, transform)`, where `settled` is the value at each
`(location, origin_date)` group's largest tracked `as_of`. `versions`
must already be filtered by the caller to the desired origin dates.
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

Nudge `data.Y` in place, at every `(t, l)` with `0 <= data.delay[t, l]
<= DELAY_CUTOFF` and a matching `profile` entry, by adding the
profile's location/delay correction.
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
# ses member: simple exponential smoothing (level only, no trend).
# ---------------------------------------------------------------------

"""
    fit_ses(y, alphas) -> (level, resid_sd, alpha)

Simple exponential smoothing (level only) on `y` (ascending in time),
fit by grid search over `alphas`: picks the value minimising in-sample
one-step-ahead SSE. `level` is the final smoothed level; `resid_sd` is
the in-sample one-step residual standard deviation at the chosen
`alpha`.
"""
function fit_ses(y::AbstractVector{Float64}, alphas)
    best_alpha = first(alphas)
    best_sse = Inf
    best_level = y[1]
    best_resid = Float64[]
    for alpha in alphas
        level = y[1]
        resid = Vector{Float64}(undef, length(y) - 1)
        for t in 2:length(y)
            resid[t - 1] = y[t] - level
            level = alpha * y[t] + (1 - alpha) * level
        end
        sse = sum(abs2, resid)
        if sse < best_sse
            best_sse = sse
            best_alpha = alpha
            best_level = level
            best_resid = resid
        end
    end
    dof = max(length(best_resid) - 1, 1)
    resid_sd = sqrt(sum(abs2, best_resid) / dof)
    return best_level, resid_sd, best_alpha
end

"""
    simulate_ses_paths(level, resid_sd, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Simulate `npaths` Gaussian sample paths for a level-only forecast:
`level + resid_sd * sqrt(h) * randn()` per horizon `h` (a simplifying
random-walk-like heuristic for how one-step forecast variance
accumulates over the horizon).
"""
function simulate_ses_paths(
    level::Float64, resid_sd::Float64, horizons, npaths::Int;
    rng::Random.AbstractRNG,
)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    for h in horizons, s in 1:npaths
        out[h][s] = level + resid_sd * sqrt(h) * randn(rng)
    end
    return out
end

# ---------------------------------------------------------------------
# climatology member: seasonal-naive empirical quantiles, no fitting.
# ---------------------------------------------------------------------

"""
    climatology_quantiles(split_df, loc, target_end_date, transform;
                          band, min_n) -> Vector{Float64}

Seasonal-naive/climatology forecast: pools every historical `wili`
observation at `loc` in `split_df` (already capped at the split's own
forecast origin -- no future leakage) whose week-of-season is within
`band` weeks of `target_end_date`'s week-of-season (a circular
distance on the ~52-week season cycle), transforms the pool to
`transform` scale, and returns the 23 `QUANTILE_LEVELS` of that pooled
empirical distribution, back-transformed to the natural scale. Falls
back to the location's full history if fewer than `min_n`
observations are pooled at that band.
"""
function climatology_quantiles(
    split_df::DataFrame, loc::AbstractString, target_end_date::Date,
    transform::Symbol; band::Int, min_n::Int,
)
    target_woy = week_of_season(target_end_date)
    loc_df = split_df[split_df.location .== loc, :]
    isempty(loc_df) && error("no history for location $loc")
    woy = week_of_season.(loc_df.origin_date)
    dist = [min(abs(w - target_woy), 52 - abs(w - target_woy)) for w in woy]
    pooled = loc_df.wili[dist .<= band]
    length(pooled) < min_n && (pooled = loc_df.wili)
    vals = to_scale.(pooled, transform)
    return [max(from_scale(quantile(vals, q), transform), 0.0)
            for q in QUANTILE_LEVELS]
end

# ---------------------------------------------------------------------
# ens-mean: plain average of the four members, all 5 seasons.
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, profile, versions_full) -> DataFrame

Fit and forecast all four members for every cross-validation split of
every season in `seasons`, averaging their quantile values into the
`nfidd-ens` submission (docs/contracts.md hub quantile schema). Seasons
in `TEST_SEASONS` are fetched with `allow_test_season=true`: each split
is still just a per-origin vintage fit capped at its own forecast
origin, not training on the test season (see file header).
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
            data_bf = deepcopy(data)
            apply_backfill_correction!(data_bf, profile)
            origin = data.origin_date
            for (li, loc) in enumerate(LOCATIONS)
                y = Float64.(data.Y[:, li])
                y_bf = Float64.(data_bf.Y[:, li])

                coef, resid_sd = fit_ar(y, AR_ORDER)
                paths_ar6 = simulate_paths(
                    y, coef, resid_sd, AR_ORDER, HORIZONS, NPATHS; rng=rng,
                )
                coef_bf, resid_sd_bf = fit_ar(y_bf, AR_ORDER)
                paths_ar6bf = simulate_paths(
                    y_bf, coef_bf, resid_sd_bf, AR_ORDER, HORIZONS, NPATHS;
                    rng=rng,
                )
                level, resid_sd_ses, _ = fit_ses(y, SES_ALPHAS)
                paths_ses = simulate_ses_paths(
                    level, resid_sd_ses, HORIZONS, NPATHS; rng=rng,
                )

                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    climo_vals = climatology_quantiles(
                        split, loc, target_end, TRANSFORM; band=CLIMO_BAND,
                        min_n=CLIMO_MIN_N,
                    )
                    for (qi, q) in enumerate(QUANTILE_LEVELS)
                        nat_ar6 = max(
                            from_scale(quantile(paths_ar6[h], q), TRANSFORM),
                            0.0,
                        )
                        nat_ar6bf = max(
                            from_scale(quantile(paths_ar6bf[h], q), TRANSFORM),
                            0.0,
                        )
                        nat_ses = max(
                            from_scale(quantile(paths_ses[h], q), TRANSFORM),
                            0.0,
                        )
                        nat_climo = climo_vals[qi]
                        ens_val = (nat_ar6 + nat_ar6bf + nat_ses +
                                   nat_climo) / 4
                        push!(rows, (
                            MODEL_ID, loc, origin, h, target_end,
                            TARGET, "quantile", q, ens_val,
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

    forecast = build_forecast_table((1, 2, 3, 4, 5), profile, versions_full)

    # `flu_data_hhs_tscv_season5.csv` runs six weeks past the hub's
    # declared round list (2020-03-28 through 2020-05-02, real-world
    # 2019/20 reporting was disrupted early by COVID-19) -- drop those
    # splits before writing, matching
    # `submissions/seabbs_bot-ar6bf/README.md`'s documented "data/
    # hub-config drift" gap (none of the hub's origin dates are
    # `required`, so this is a valid, if partial, submission either
    # way).
    n_before = length(unique(forecast.origin_date))
    filter!(:origin_date => <=(Date(2020, 3, 21)), forecast)
    n_after = length(unique(forecast.origin_date))
    n_before != n_after && println("dropped $(n_before - n_after) origin " *
        "date(s) past the hub's declared round list (season 5 overrun)")

    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    if hub_path !== nothing
        write_submission(forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="nfidd", model_abbr="ens", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
