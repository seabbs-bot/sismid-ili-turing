#!/usr/bin/env julia
# seasonpool_baseline.jl -- AR(6) per location + additive/per-location/
# median backfill correction (as seabbs_bot-ar6bf) + a POOLED seasonal
# climatology term, for the sismid-ili-turing hub session.
#
# Submitted as `seabbs_bot-seasonpool` (team_abbr "seabbs_bot": the
# hub's model-metadata schema forbids a hyphen in team_abbr,
# `^[a-zA-Z0-9_+]+$`, underscore only).
#
# Design (see experiments/simple-round/seasonpool/generate.jl and
# score.txt for the validation-season comparison this is based on):
# a NAIVE per-location Fourier(3) fit
# (submissions/nfidd-ar6/generate_forecasts_fourier.jl) made AR(6)
# WORSE (0.412 vs 0.368) -- 6 seasonal parameters per location,
# estimated on only ~2 seasons of window, overfits. The fix pools the
# seasonal shape across all 11 locations instead of fitting it per
# location:
#
#   1. ONE shared week-of-season shape (`shape(woy)`), fourth-root
#      scale, 3-harmonic Fourier (6 parameters total), fit by
#      no-intercept OLS to `data/flu_data_hhs.csv` restricted to
#      `season_year <= 2014` (true pre-2015 history, disjoint from
#      both validation and test seasons), pooling all 11 locations
#      after centring each on its own mean over that window. ~6,700
#      pooled (location, week) observations behind 6 parameters: no
#      realistic overfitting risk.
#   2. Per split, per location: a 2-parameter regression (intercept +
#      amplitude) adapts the shared shape to that location's own level
#      and seasonal swing, using only that split's own
#      `build_model_data` window (never future data).
#   3. The backfill correction (identical to seabbs_bot-ar6bf:
#      additive, per-location, median, delay cutoff 8) is applied to
#      the vintage series first; the seasonal term is then fit and
#      removed; AR(6) is fit to the deseasonalised residual.
#   4. Forecast = per-location seasonal term at the (known) future
#      week-of-season + simulated AR(6) residual paths. Only the AR
#      component propagates simulated uncertainty forward.
#
# Validation-season result (experiments/simple-round/seasonpool/
# score.txt): mean_wis=0.3049 vs 0.359 for seabbs_bot-ar6bf and 0.368
# for nfidd-ar6 (15% and 17% improvement respectively), with the gain
# concentrated at longer horizons and in HHS Regions 9/6/2.
#
# Coverage: generates forecasts for every origin date in ALL FIVE
# seasons (1-2 validation, 3-5 held-out test), matching
# seabbs_bot-ar6bf's full-season driver -- each split is still just a
# per-origin vintage fit capped at its own forecast origin. Neither
# the backfill profile (`season_year <= 2016`) nor the pooled seasonal
# shape (`season_year <= 2014`) can leak the test seasons
# (`season_year >= 2017`).
#
# LIGHT + ANALYTIC: CSV/DataFrames/Statistics/LinearAlgebra only, no
# Turing.
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

const MODEL_ID = "seabbs_bot-seasonpool"
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const N_HARMONICS = 3
const SEASON_PERIOD = 52.0
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12               # matches seabbs_bot-ar6bf
const DELAY_CUTOFF = 8        # backfill window; matches ar6bf
const MIN_SUPPORT = 5         # min sample size per (location, delay)
const CLIMATOLOGY_YEAR = 2014 # pooled shape uses season_year <= this

# ---------------------------------------------------------------------
# Backfill correction profile (identical to seabbs_bot-ar6bf)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile on the `transform` scale,
identical to `seabbs_bot-ar6bf`. For each `(location, delay)` with at
least `min_support` recorded versions at that delay, the median of
`settled - vintage` (both on `transform` scale) across matching
`(location, origin_date)` groups. `versions` must already be filtered
by the caller (here: `season_year <= 2016`, no test-season data).
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
profile's location/delay correction. Missing entries and delays
outside the profile's support are left untouched.
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
# Pooled seasonal climatology
# ---------------------------------------------------------------------

"""
    fourier_features(woy, K, period) -> Vector{Float64}

`2K` Fourier features `[sin(2*pi*1*woy/period), cos(2*pi*1*woy/period),
...]` for `K` harmonics of week-of-season `woy` at the given `period`
(weeks).
"""
function fourier_features(woy::Real, K::Int, period::Float64)
    feats = Vector{Float64}(undef, 2K)
    for k in 1:K
        ang = 2 * pi * k * woy / period
        feats[2k - 1] = sin(ang)
        feats[2k] = cos(ang)
    end
    return feats
end

"""
    fit_pooled_shape(history; transform, K, period, cutoff_year)
        -> Vector{Float64}

Fit ONE shared `K`-harmonic week-of-season shape, pooling all 11
locations, from `history` (the `flu_data_hhs.csv` schema: `location,
origin_date, wili`) restricted to `season_year(origin_date) <=
cutoff_year`. Each location's `transform`-scale series is centred on
its own mean over this window first (so a location that simply runs
at a different level doesn't bias the shared shape); a no-intercept
OLS regression of the pooled centred values on `fourier_features` then
gives the shared shape's `2K` coefficients.
"""
function fit_pooled_shape(
    history::DataFrame; transform::Symbol, K::Int, period::Float64,
    cutoff_year::Int,
)
    hist = history[season_year.(history.origin_date) .<= cutoff_year, :]
    centred = Vector{Float64}(undef, nrow(hist))
    for g in groupby(hist, :location)
        vals = to_scale.(g.wili, transform)
        centred[parentindices(g)[1]] = vals .- mean(vals)
    end
    X = Matrix{Float64}(undef, nrow(hist), 2K)
    for (i, d) in enumerate(hist.origin_date)
        X[i, :] = fourier_features(week_of_season(d), K, period)
    end
    return X \ centred
end

"""
    shape_value(woy, shape_coef, K, period) -> Float64

Shared pooled seasonal shape (deviation from a location's own mean, on
the `transform` scale) at week-of-season `woy`.
"""
function shape_value(woy::Real, shape_coef::Vector{Float64}, K::Int,
        period::Float64)
    return dot(fourier_features(woy, K, period), shape_coef)
end

"""
    fit_seasonal_level(y, woy_vec, shape_coef, K, period) -> (alpha, beta)

Per-location OLS fit of `y_t = alpha + beta * shape(woy_t) + resid`:
the small per-location amplitude/level scaling of the shared shape.
Only 2 parameters, fit on that split's own training window.
"""
function fit_seasonal_level(
    y::AbstractVector{Float64}, woy_vec::AbstractVector{Int},
    shape_coef::Vector{Float64}, K::Int, period::Float64,
)
    n = length(y)
    X = ones(n, 2)
    for (i, w) in enumerate(woy_vec)
        X[i, 2] = shape_value(w, shape_coef, K, period)
    end
    alpha, beta = X \ y
    return alpha, beta
end

# ---------------------------------------------------------------------
# AR(6) fit + forecast (identical to nfidd-ar6 / seabbs_bot-ar6bf,
# applied to the deseasonalised residual)
# ---------------------------------------------------------------------

"""
    fit_ar(y, order) -> (coef, resid_sd)

OLS fit of an AR(`order`) model with intercept to `y` (ascending in
time, no missing values).
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
from the end of `y` (deseasonalised residual scale), for each horizon
in `horizons`.
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
    build_forecast_table(seasons, profile, versions_full, shape_coef)
        -> DataFrame

Fit and forecast the AR(6)+backfill+pooled-season model for every
cross-validation split of every season in `seasons`, returning the
combined hub quantile table (docs/contracts.md schema). Seasons in
`TEST_SEASONS` are fetched with `allow_test_season=true`: each split
is still just a per-origin vintage fit capped at its own forecast
origin, not training on the test season.
"""
function build_forecast_table(seasons, profile, versions_full, shape_coef)
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
                alpha, beta = fit_seasonal_level(
                    y, data.woy, shape_coef, N_HARMONICS, SEASON_PERIOD,
                )
                seasonal_now = [
                    alpha + beta * shape_value(
                        w, shape_coef, N_HARMONICS, SEASON_PERIOD,
                    ) for w in data.woy
                ]
                resid = y .- seasonal_now
                coef, resid_sd = fit_ar(resid, AR_ORDER)
                paths = simulate_paths(
                    resid, coef, resid_sd, AR_ORDER, HORIZONS, NPATHS;
                    rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    seasonal_h = alpha + beta * shape_value(
                        week_of_season(target_end), shape_coef,
                        N_HARMONICS, SEASON_PERIOD,
                    )
                    vals = paths[h] .+ seasonal_h
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

    history = load_series("flu_data_hhs")
    shape_coef = fit_pooled_shape(
        history; transform=TRANSFORM, K=N_HARMONICS, period=SEASON_PERIOD,
        cutoff_year=CLIMATOLOGY_YEAR,
    )
    println("pooled shape ($(N_HARMONICS) harmonics, season_year <= " *
            "$(CLIMATOLOGY_YEAR)): coef=$(round.(shape_coef; digits=4))")

    forecast = build_forecast_table(
        (1, 2, 3, 4, 5), profile, versions_full, shape_coef,
    )
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    if hub_path !== nothing
        write_submission(forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="seabbs_bot", model_abbr="seasonpool", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
