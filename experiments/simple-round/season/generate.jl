#!/usr/bin/env julia
# generate.jl -- AR(6) baseline (submissions/nfidd-ar6/generate_forecasts.jl)
# plus a REGULARISED seasonal term and the backfill correction
# (submissions/seabbs_bot-ar6bf/generate_forecasts.jl), for the
# simple-model wide round's SEASONALITY family.
#
# A naive Fourier(3) term made AR(6) worse (0.412 vs 0.368 mean
# validation WIS): free harmonic coefficients, fit OLS on only ~2
# seasons of AR training data (`window_weeks=104`), overfit the two
# validation seasons' particular peak timing/height rather than
# learning the recurring annual shape. This driver instead adds a
# CLIMATOLOGY term: a single extra regressor per location, equal to a
# smoothed circular week-of-season median curve built from the FULL
# historical series (`data/flu_data_hhs.csv`, every season strictly
# before the split's own forecast origin -- no test-season or future
# leakage), not just the 2-season AR window. The smoothing (a 5-week
# circular moving average over the median-by-week-of-season, itself
# already a robust-to-peak-noise summary across every available
# historical season) is what keeps this regularised: it borrows a
# single well-conditioned number per calendar week from all of
# history, rather than fitting a free per-week or per-harmonic
# parameter to the ~2 seasons the AR(6) fit itself sees. That single
# climatology value enters the per-location regression as one more
# OLS coefficient alongside the AR(6) lags, so the fit can shrink it
# to ~0 itself if it is not informative for a given location/split --
# unlike the free Fourier harmonics, there is no separate frequency or
# phase for the fit to overfit.
#
# Sweep (validation seasons 1, 2 only, docs/contracts.md experimental
# integrity; see experiments/simple-round/season/score.txt for the
# full table): fewer harmonics (Fourier(1), Fourier(2)), ridge-shrunk
# Fourier(3) at several penalties, and this climatology term were all
# tried, alone and combined with the backfill correction. The
# climatology term was the only seasonal form that reliably beat both
# the plain AR(6) baseline (0.368) and AR(6)+backfill (0.359) instead
# of overfitting; combined with the backfill correction it does best
# overall (0.300), and wins in 10 of 11 locations and at every horizon
# individually (not just on average) -- see score.txt.
#
# Deliberately avoids `using SismidILITuring` for the same reason as
# nfidd-ar6: this only needs `src/core.jl`, `src/data.jl`, and
# `src/hubio.jl` (each standalone-includable, CSV/DataFrames/Dates
# only), not the Turing/Mooncake/Pathfinder weight of the full module.
#
# Coverage: like seabbs_bot-ar6bf, generates forecasts for every origin
# date in ALL FIVE seasons (1-2 validation, 3-5 held-out test) when
# writing a hub submission -- each split is still just a per-origin
# vintage fit capped at its own forecast origin. Model selection and
# scoring above used the VALIDATION seasons only.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl <hub_path>

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

const MODEL_ID = "seabbs_bot-season"
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12            # matches nfidd-ar6's build_model_data Dmax
const WINDOW_WEEKS = 104   # matches nfidd-ar6: caps AR history at 2 seasons
const SEASON_PERIOD = 52   # canonical annual cycle length for the climatology
const DELAY_CUTOFF = 8     # weeks; backfill profile is ~0 beyond this
const MIN_SUPPORT = 5      # min sample size per (location, delay) to trust

# ---------------------------------------------------------------------
# Backfill correction (identical to seabbs_bot-ar6bf)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile on the `transform` scale.
See `submissions/seabbs_bot-ar6bf/generate_forecasts.jl` for the full
derivation; identical here. `versions` must already be filtered by the
caller to the training set only (no test seasons).
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

Nudge `data.Y` in place wherever `0 <= delay <= DELAY_CUTOFF` and a
matching `profile` entry exists. Identical to seabbs_bot-ar6bf.
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
# Climatology term
# ---------------------------------------------------------------------

"""
    build_climatology(loc_hist, forecast_origin;
                       period=SEASON_PERIOD, smooth_window=5)
        -> Vector{Float64}

Smoothed circular week-of-season climatology curve for one location,
on the `TRANSFORM` scale, length `period`. Built ONLY from `loc_hist`
rows strictly before `forecast_origin` (no leakage of the split's own
or future observations); `loc_hist` is that location's full historical
`(origin_date, wili)` rows, unfiltered by date -- filtering happens
here, against each split's own forecast origin.

Each circular bin (`mod1(week_of_season(d), period)`) is the MEDIAN
`to_scale` value across every matching historical week (robust to a
single season's peak noise), then smoothed with a `period`-wrapped
`smooth_window`-wide moving average. This smoothing is what keeps the
term regularised: one number per calendar week borrowed from every
available historical season, not a free parameter fit to the AR(6)
window's ~2 seasons.
"""
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

# ---------------------------------------------------------------------
# AR(6) + climatology fit and forward simulation
# ---------------------------------------------------------------------

"""
    fit_ar_clim(y, woy, order, clim) -> (coef, resid_sd)

OLS fit of an AR(`order`) model with intercept and one extra
regressor, the climatology value `clim[mod1(woy[t], period)]` at each
response time `t`. `coef = [c, phi_1, ..., phi_order, gamma]`; `gamma`
is the climatology coefficient, free to shrink to ~0 itself if
uninformative for this location/split. `resid_sd` is the in-sample
residual standard deviation with degrees of freedom adjusted for the
extra column.
"""
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

"""
    simulate_paths_clim(y, future_woy, coef, resid_sd, order, clim,
                        horizons, npaths; rng) -> Dict{Int,Vector{Float64}}

Simulate `npaths` Gaussian-innovation AR(`order`)+climatology sample
paths forward from the end of `y`, for each horizon in `horizons`.
`future_woy[h]` is the week-of-season at horizon `h`. Identical
forward-propagation structure to nfidd-ar6's `simulate_paths`, with the
climatology term added to each step's prediction.
"""
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
    build_forecast_table(seasons, profile, versions_full, hist_by_loc)
        -> DataFrame

Fit and forecast the AR(6)+climatology+backfill model for every
cross-validation split of every season in `seasons`. Training
discipline as nfidd-ar6/seabbs_bot-ar6bf: `build_model_data` caps each
split at its own forecast origin and `window_weeks=104` further caps
AR history to 2 seasons; the climatology term separately draws on the
FULL historical series but is still capped, inside
`build_climatology`, at each split's own forecast origin. `season` in
`TEST_SEASONS` (3, 4, 5) is fetched with `allow_test_season=true` --
this was tuned and selected on `VALIDATION_SEASONS` (1, 2) only (see
score.txt); a per-origin fit here is still just a vintage fit capped
at its own forecast origin, not training on the test seasons.
"""
function build_forecast_table(seasons, profile, versions_full, hist_by_loc)
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
            apply_backfill_correction!(data, profile)
            origin = data.origin_date
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

    hist_full = load_series("flu_data_hhs")
    hist_by_loc = Dict(
        loc => hist_full[hist_full.location .== loc, [:origin_date, :wili]]
        for loc in LOCATIONS
    )

    forecast = build_forecast_table(
        (1, 2, 3, 4, 5), profile, versions_full, hist_by_loc,
    )
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    if hub_path !== nothing
        write_submission(forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="seabbs_bot", model_abbr="season", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
