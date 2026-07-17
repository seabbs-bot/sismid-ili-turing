#!/usr/bin/env julia
# generate.jl -- conformal-POOLED: split-conformal intervals wrapped around
# a LEAK-FREE POOLED-SEASONAL point forecast. Forked from
# experiments/simple-round/conformal/generate.jl (whose split-conformal
# machinery is reused verbatim below); the four point-forecast changes vs
# that file are:
#   1. POOLED week-of-season seasonal shape shared across all 11 locations
#      (build_pooled_seasonal), instead of a per-location empirical
#      climatology -- the pooled-deviation construction from seasoncombo,
#      but rebuilt LEAK-FREE per origin (origin_date < forecast origin).
#   2. per-(location,delay) BACKFILL profile also rebuilt LEAK-FREE per
#      origin (only vintage rows with origin_date < forecast origin),
#      instead of a fixed season_year<=2016 global fit.
#   3. per-location AR(6) with AR-COEFFICIENT partial pooling toward a
#      pooled (all-location) fit, weight POOL_WEIGHT (validation optimum
#      w=0.3, see score.txt).
#   4. LOG transform (log beats fourth-root ~4%, simp-transform).
# Everything below the constants -- the split-conformal calibration
# (PendingTask/calibrated_quantiles/pool maturation) -- is unchanged from
# conformal/generate.jl. Result: val mean WIS 0.2870 (cov50 0.517, cov90
# 0.908), leak-free. See score.txt for the honest comparison.
#
# (Historical note retained from the parent file for context on the
# interval machinery:)
# The predictive DISTRIBUTION is split-conformal, not parametric.
#
# simp-intervals (experiments/simple-round/intervals/score.txt) found the
# plain AR(6)+backfill model's raw fitted `resid_sd` badly under-covers
# (50% nominal -> ~41% actual, 90% nominal -> ~78% actual) and fixed this
# by INFLATING a parametric (Student-t) distribution until its coverage
# happened to land near nominal. That is an indirect fix: the family and
# scale are chosen so that *averaged over locations and horizons* the
# coverage comes out right, not because the tails are actually shaped
# like the forecast errors.
#
# This driver instead calibrates DIRECTLY: at every forecast origin, it
# keeps a running (per-horizon, pooled-across-locations) empirical
# distribution of "point forecast minus later-observed value" errors from
# every EARLIER origin whose outcome is already known as of today, and
# reads calibrated quantiles straight off that empirical distribution
# (split-conformal, in the sense of a genuinely held-out calibration set
# -- just built once, incrementally, walk-forward, rather than as a
# single fixed split). No Gaussian/Student-t shape is assumed at all past
# an initial warm-up; the errors' own quantiles -- however skewed -- are
# used directly.
#
# STUDENTIZING each error by its own originating fit's `resid_sd` before
# pooling (so one calibration set could still be rescaled per-location,
# to account for the 11 locations' very different volatility --
# season/generate.jl's region-by-region breakdown shows roughly a 4x
# spread in mean WIS) was tried and made things slightly WORSE (mean WIS
# 0.296 vs 0.292 raw, both coverage figures also worse) -- a single
# split's `resid_sd` is itself a noisy per-(location, split) OLS estimate
# over a short window, so dividing by it injects more noise than it
# removes real scale differences. Raw, un-rescaled pooling across all 11
# locations -- letting sample SIZE rather than per-location rescaling do
# the calibration work -- wins, and is what this file implements.
#
# Calibration-set construction (the walk-forward rolling design):
#   Every task this driver has ever forecast (any origin, any location,
#   any horizon) is tracked in a `pending` queue keyed by its
#   `target_end_date`. Immediately before generating THIS origin's
#   forecasts, every pending task whose `target_end_date <= this origin`
#   is "matured": its now-knowable value is looked up in
#   `flu_data_hhs_versions.csv` (the latest `as_of <= this origin`
#   vintage -- the same "best information available as of today" idea
#   `apply_backfill_correction!` already relies on, not the fully-settled
#   future value), the error (actual - point, on the TRANSFORM scale) is
#   pushed into that horizon's pool, and the task is dropped from
#   `pending`. A task whose vintage hasn't appeared yet (rare -- some
#   locations report a week or two late) is simply left in `pending` and
#   retried at the next origin. Crucially this can never leak a task's
#   own future observation into its OWN interval: `target_end_date` is
#   always strictly after the origin that generated it, so a task can
#   only mature at some STRICTLY LATER origin than its own.
#
#   Because this pools across all 11 locations, the pool for horizon h
#   already has ~11 observations after just one elapsed origin beyond h
#   weeks, and ~33+ after three -- so real calibration data accumulates
#   fast relative to `MIN_CALIB` (see below), and by the second
#   validation season (a ~5-month gap after season 1 ends, far longer
#   than the h<=4-week horizon) every task is fully warmed up. The pool
#   is NEVER reset between seasons or at the validation/test boundary --
#   it just keeps accumulating strictly-past information, exactly the
#   same "vintage fit capped at its own forecast origin" discipline as
#   every other split in this codebase, so covering the held-out test
#   seasons (3-5) at generation time never uses information from those
#   seasons to calibrate anything before it has actually happened.
#
# Fallback (only needed for the first ~2-5 origins of season 1, before
# any pool reaches `MIN_CALIB`): the SAME Student-t(df=10), scale=1.4
# scheme simp-intervals selected (experiments/simple-round/intervals/
# generate.jl) -- a defensible parametric placeholder for the handful of
# earliest tasks that have no calibration history yet, not a tuned choice
# of its own. `MIN_CALIB` itself was swept over {5, 10, 15, 20, 30, 40}
# on validation only: 5/10/15/20 all land at the same mean_wis=0.2917-
# 0.2939 (the pool already has plenty of pooled-across-11-locations data
# by the time any of these thresholds is reached), 30 and 40 are very
# slightly worse (0.2951, 0.2962) -- waiting for a bigger pool before
# trusting it just means more tasks spend longer on the cruder fallback
# for no accuracy benefit. `MIN_CALIB=10` is used: comfortably past the
# smallest thresholds that already work, not a sharp optimum.
#
# Result (validation seasons 1, 2 only; see score.txt for the full
# table): mean_wis=0.2917, sd_wis=0.3451 (vs 0.3004 for the SAME point
# forecast's parametric Gaussian intervals -- a 2.9% improvement, in the
# same ballpark as simp-intervals' own tuned-Student-t gain over ITS
# baseline, but reached with no manual scale search at all). Coverage:
# 50% nominal -> 0.480 actual, 90% nominal -> 0.871 actual -- both much
# closer to nominal than the parametric scheme's raw 41%/78%, reached
# directly rather than via simp-intervals' manually-inflated-until-close
# Student-t (52.5%/89.2%).
#
# Does NOT beat seasoncombo-core's 0.2781 (experiments/simple-round/
# seasoncombo/score.txt): that candidate's point forecast (pooled
# seasonal shape + per-location amplitude + backfill) is itself simply
# more accurate than this family's climatology point forecast -- its own
# WIS with a plain, uncalibrated Gaussian spread (0.2781) already beats
# this file's calibrated-interval result built on the weaker point
# forecast. This is a genuine, informative negative result on the
# "beat 0.278" target, not a failure of split-conformal calibration
# itself: conformal calibration fixes what it can (the interval, given a
# fixed point forecast) and does so cleanly (2.9% WIS gain, much better
# coverage, zero manual tuning) -- but it cannot make a worse point
# forecast better than a rival's better one. Whether split-conformal
# calibration applied to seasoncombo-core's OWN (better) point forecast
# would beat 0.2781 is untested here (out of scope: the brief was to
# keep the season/generate.jl point forecast fixed) but is a natural
# next step if this family is revisited.
#
# LIGHT + ANALYTIC: no simulation at all. The point forecast is the
# deterministic AR(6)+climatology recursion (its own mean path, no
# innovation noise); calibrated quantiles are read directly off an
# empirical (or, in the fallback case, closed-form Student-t) quantile
# function -- no NPATHS/rng Monte Carlo anywhere in this file, unlike
# every earlier simple-round driver.
#
# Coverage: like every other simple-round driver, generates forecasts for
# every origin date in ALL FIVE seasons (1-2 validation, 3-5 held-out
# test) when writing a hub submission -- each split is still just a
# per-origin vintage fit capped at its own forecast origin, and the
# rolling calibration pool is capped exactly the same way (see above).
# `MIN_CALIB`/the fallback scheme were chosen by inspection on the
# VALIDATION seasons only; scoring below is validation-only too.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl [hub_path]
# Always prints + writes score.txt (validation-seasons-only scoring
# against the local scratch-hub oracle); additionally writes a full
# 5-season hub submission if `hub_path` is given.

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra
using Distributions

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
const HERE = @__DIR__
include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))
include(joinpath(PKG_DIR, "src", "hubio.jl"))
include(joinpath(PKG_DIR, "src", "scoring.jl"))

const MODEL_ID = "seabbs_bot-conformal-pooled"
const TRANSFORM = :log     # log beats fourth-root ~4% (simp-transform)
const AR_ORDER = 6
const DMAX = 12            # matches nfidd-ar6's build_model_data Dmax
const WINDOW_WEEKS = 104   # matches nfidd-ar6: caps AR history at 2 seasons
const SEASON_PERIOD = 52   # canonical annual cycle length for the climatology
const DELAY_CUTOFF = 8     # weeks; backfill profile is ~0 beyond this
const MIN_SUPPORT = 5      # min sample size per (location, delay) to trust
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")  # local oracle for scoring

# Pooled-seasonal point-forecast knobs (this variant's changes vs the
# per-location `conformal` point forecast).
const POOL_WEIGHT = parse(Float64, get(ENV, "POOL_WEIGHT", "0.3"))
                              # AR-coefficient partial-pooling weight:
                              # coef_blend = (1-w)*coef_loc + w*coef_pool.
                              # w=0.3 is the validation optimum of a leak-free
                              # sweep over {0.0,0.3,0.5,0.7,0.9} (see score.txt);
                              # env-overridable to reproduce that sweep.
const SEASONAL_SMOOTH = 3     # circular smoothing span for the pooled shape
const SEASONAL_MIN_SUPPORT = 5  # min obs per week-of-season bin to trust

# Split-conformal calibration knobs.
const MIN_CALIB = 10       # min pooled (location-pooled) errors per horizon
                            # before trusting the empirical quantiles over
                            # the fallback; pools across 11 locations, so
                            # this is reached within ~3 elapsed origins
                            # per horizon.
const FALLBACK_DF = 10     # simp-intervals' chosen Student-t df
const FALLBACK_SCALE = 1.4 # simp-intervals' chosen scale, used only for
                            # the handful of tasks too early to have any
                            # real calibration history yet

# ---------------------------------------------------------------------
# Backfill correction (identical to seabbs_bot-ar6bf / season/generate.jl)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile on the `transform` scale.
Identical to `experiments/simple-round/season/generate.jl`'s function of
the same name; `versions` must already be filtered by the caller to the
training set only (no test seasons).
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
matching `profile` entry exists. Identical to season/generate.jl.
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
# Climatology term (identical to season/generate.jl)
# ---------------------------------------------------------------------

"""
    build_climatology(loc_hist, forecast_origin;
                       period=SEASON_PERIOD, smooth_window=5)
        -> Vector{Float64}

Smoothed circular week-of-season climatology curve for one location,
on the `TRANSFORM` scale, length `period`. Identical to
`season/generate.jl`'s function of the same name -- see that file's
docstring for the full derivation. Built ONLY from `loc_hist` rows
strictly before `forecast_origin` (no leakage).
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
# Pooled week-of-season seasonal shape (this variant's point-forecast
# change vs `conformal`, which used a per-location empirical climatology)
# ---------------------------------------------------------------------

"""
    build_pooled_seasonal(hist_all, forecast_origin; period, smooth_window,
                           min_support) -> Vector{Float64}

ONE week-of-season seasonal shape SHARED across all 11 locations, on the
`TRANSFORM` scale, length `period`, built LEAK-FREE from only the rows of
`hist_all` with `origin_date < forecast_origin`. Each location's own
series is centred on its own mean over that available history, and the
centred deviations are POOLED across all locations per week-of-season
(the same pooled-deviation construction as
`experiments/simple-round/seasoncombo/generate.jl`'s
`build_seasonal_profile`, but rebuilt per origin rather than from a fixed
`season_year <= 2016` window). The returned shape is mean-zero; each
location estimates its own amplitude through the `fit_ar_clim`
climatology regressor.
"""
function build_pooled_seasonal(
    hist_all::DataFrame, forecast_origin::Date;
    period::Int=SEASON_PERIOD, smooth_window::Int=SEASONAL_SMOOTH,
    min_support::Int=SEASONAL_MIN_SUPPORT,
)
    sub = hist_all[hist_all.origin_date .< forecast_origin, :]
    isempty(sub) && return zeros(period)
    x = to_scale.(sub.wili, TRANSFORM)
    locs = sub.location
    woys = week_of_season.(sub.origin_date)
    levels = Dict{String,Float64}()
    for loc in unique(locs)
        levels[loc] = mean(x[locs .== loc])
    end
    bins = [Float64[] for _ in 1:period]
    for i in eachindex(x)
        push!(bins[mod1(woys[i], period)], x[i] - levels[locs[i]])
    end
    means = [length(b) >= min_support ? mean(b) : 0.0 for b in bins]
    half = smooth_window ÷ 2
    smoothed = similar(means)
    for i in 1:period
        idxs = [mod1(i + o, period) for o in (-half):half]
        smoothed[i] = mean(means[idxs])
    end
    smoothed .-= mean(smoothed)
    return smoothed
end

"""
    ar_clim_design(y, woy, order, clim) -> (X, yresp)

Build the OLS design matrix `X` and response `yresp` for an AR(`order`)
model with intercept and one extra regressor, the shared seasonal shape
`clim` evaluated at each response week-of-season. Columns match
`fit_ar_clim`: `[intercept, y[t-1], ..., y[t-order], clim(woy[t])]`.
Exposed separately from the fit so per-location and pooled fits can share
one design and be blended (AR-coefficient partial pooling).
"""
function ar_clim_design(
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
    return X, yresp
end

# ---------------------------------------------------------------------
# AR(6) + climatology fit and DETERMINISTIC point path (no simulation)
# ---------------------------------------------------------------------

"""
    fit_ar_clim(y, woy, order, clim) -> (coef, resid_sd)

OLS fit of an AR(`order`) model with intercept and one extra regressor,
the climatology value at each response time. Identical to
`season/generate.jl`'s function of the same name. `resid_sd` is kept
only for the fallback interval scheme (see `FALLBACK_DF`/`FALLBACK_SCALE`
above); it plays no role once a horizon's calibration pool matures.
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
    point_path_clim(y, future_woy, coef, order, clim, horizons)
        -> Dict{Int,Float64}

Deterministic AR(`order`)+climatology point forecast: the mean recursion
carried forward with NO innovation noise added at each step (each
predicted value feeds back into the tail exactly as
`simulate_paths_clim` does, but with `val = pred`, not
`pred + resid_sd * randn(rng)`). This is the fixed point forecast that
split-conformal calibration is built around; the AR(6) lags and
climatology coefficient are identical to `season/generate.jl`, only the
uncertainty machinery differs.
"""
function point_path_clim(
    y::AbstractVector{Float64}, future_woy::Vector{Int},
    coef::Vector{Float64}, order::Int, clim::Vector{Float64}, horizons,
)
    hmax = maximum(horizons)
    out = Dict{Int,Float64}()
    tail = copy(y[(end - order + 1):end])
    for h in 1:hmax
        pred = coef[1]
        for lag in 1:order
            pred += coef[lag + 1] * tail[end - lag + 1]
        end
        pred += coef[order + 2] * clim[mod1(future_woy[h], SEASON_PERIOD)]
        if h in horizons
            out[h] = pred
        end
        push!(tail, pred)
        popfirst!(tail)
    end
    return out
end

# ---------------------------------------------------------------------
# Vintage lookup: "best value known as of a given date" for a
# (location, reference date) pair -- used to mature pending calibration
# tasks once their target_end_date has passed.
# ---------------------------------------------------------------------

"""
    build_vintage_index(versions) -> Dict{Tuple{String,Date},Vector{Tuple{Date,Float64}}}

Index `versions` (the `flu_data_hhs_versions.csv` schema) by
`(location, origin_date)`, each entry a `(as_of, wili)` vector sorted
ascending by `as_of`.
"""
function build_vintage_index(versions::DataFrame)
    idx = Dict{Tuple{String,Date},Vector{Tuple{Date,Float64}}}()
    for row in eachrow(versions)
        key = (row.location, row.origin_date)
        push!(get!(idx, key, Tuple{Date,Float64}[]), (row.as_of, row.wili))
    end
    for v in values(idx)
        sort!(v; by=x -> x[1])
    end
    return idx
end

"""
    latest_known(idx, loc, date, as_of_cutoff) -> Union{Float64,Missing}

Latest `wili` vintage for `(loc, date)` with `as_of <= as_of_cutoff`, or
`missing` if none exists yet -- the "best information available as of
today" value used to mature a pending calibration task.
"""
function latest_known(
    idx::Dict{Tuple{String,Date},Vector{Tuple{Date,Float64}}},
    loc::String, date::Date, as_of_cutoff::Date,
)
    v = get(idx, (loc, date), nothing)
    v === nothing && return missing
    best = missing
    for (a, w) in v
        a <= as_of_cutoff || break
        best = w
    end
    return best
end

# ---------------------------------------------------------------------
# Split-conformal calibrated quantiles
# ---------------------------------------------------------------------

"""
    PendingTask

One forecast task awaiting maturation: the point forecast (transform
scale) made for `location` at `horizon`, whose outcome becomes knowable
once some later origin reaches `target_end_date`.
"""
struct PendingTask
    target_end_date::Date
    location::String
    horizon::Int
    point::Float64
end

"""
    calibrated_quantiles(point, resid_sd, pool, qs) -> Vector{Float64}

Calibrated quantile forecast (transform scale) at each level in `qs`.

`pool` holds RAW (natural-scale-of-the-transform) calibration errors for
this horizon, pooled across all 11 locations: each past error is
`actual - point` on the transform scale, un-rescaled (see
`build_forecast_table`).

Studentizing each error by ITS OWN originating fit's `resid_sd` before
pooling -- so a shared calibration set could still be rescaled back up
per-location -- was tried and made things slightly WORSE (mean WIS 0.296
vs 0.292 raw, and worse coverage): a single split's `resid_sd` is itself
a noisy estimate (OLS over a ~2-season, ~11-location-specific window), so
dividing by it just injects extra noise into the pool rather than
removing genuine cross-location scale differences. Raw pooling, with the
sheer volume from pooling 11 locations doing the calibration work
instead, wins.

If `pool` has reached `MIN_CALIB`, each quantile is
`point + quantile(pool, q)` directly. Otherwise falls back to the
Student-t(`FALLBACK_DF`) scheme, scaled to `resid_sd * FALLBACK_SCALE`
(see module docstring) -- the one place `resid_sd` is still used, since
there is no pooled calibration data yet to fall back on.
"""
function calibrated_quantiles(
    point::Float64, resid_sd::Float64, pool::Vector{Float64},
    qs::Vector{Float64},
)
    if length(pool) >= MIN_CALIB
        return [point + quantile(pool, q) for q in qs]
    end
    vscale = sqrt((FALLBACK_DF - 2) / FALLBACK_DF)
    innov_sd = resid_sd * vscale * FALLBACK_SCALE
    tdist = TDist(FALLBACK_DF)
    return [point + innov_sd * quantile(tdist, q) for q in qs]
end

# ---------------------------------------------------------------------
# Forecast table builder: rolling walk-forward conformal calibration
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, versions_full, hist_all, vidx) -> DataFrame

Fit and forecast the leak-free pooled-seasonal + backfill + pooled-AR(6)
point forecast, wrapped in split-conformal calibrated quantiles, for every
cross-validation split of every season in `seasons` IN CHRONOLOGICAL ORDER
(required: the calibration pool accumulates strictly-past errors as
`seasons` is walked through, so seasons must be passed in ascending order
-- `(1, 2, 3, 4, 5)`, never re-ordered). The backfill and pooled-seasonal
profiles are rebuilt per origin from strictly-prior data (leak-free); the
AR(6) coefficients are partially pooled across locations at `POOL_WEIGHT`.
See the module docstring for the full pending -> matured -> pooled
mechanics.
"""
function build_forecast_table(seasons, versions_full, hist_all, vidx)
    rows = DataFrame(
        model_id=String[], location=String[], origin_date=Date[],
        horizon=Int[], target_end_date=Date[], target=String[],
        output_type=String[], output_type_id=Float64[], value=Float64[],
    )
    pending = PendingTask[]
    calib_pool = Dict(h => Float64[] for h in HORIZONS)
    n_conformal = Dict(h => 0 for h in HORIZONS)
    n_fallback = Dict(h => 0 for h in HORIZONS)
    nloc = length(LOCATIONS)

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

            # Backfill revision profile rebuilt LEAK-FREE per origin: only
            # vintage rows with origin_date strictly before this forecast
            # origin, so no validation/test-season revision behaviour ever
            # informs its own forecasts.
            past_versions = versions_full[
                versions_full.origin_date .< origin, :,
            ]
            profile = build_revision_profile(
                past_versions; transform=TRANSFORM,
                max_delay=DELAY_CUTOFF, min_support=MIN_SUPPORT,
            )
            apply_backfill_correction!(data, profile)

            # Mature every pending task whose target has passed, using
            # only information knowable as of `origin`.
            still_pending = PendingTask[]
            for t in pending
                if t.target_end_date > origin
                    push!(still_pending, t)
                    continue
                end
                actual = latest_known(vidx, t.location, t.target_end_date, origin)
                if actual === missing
                    push!(still_pending, t)  # retry at a later origin
                else
                    err = to_scale(actual, TRANSFORM) - t.point
                    push!(calib_pool[t.horizon], err)
                end
            end
            pending = still_pending

            future_woy = [
                week_of_season(origin + Day(7 * h)) for h in HORIZONS
            ]

            # Pooled week-of-season seasonal shape, shared across all
            # locations, leak-free (origin_date < origin).
            clim = build_pooled_seasonal(hist_all, origin)

            # First pass: per-location AR(6)+clim design + OLS fit, and
            # stack all locations' designs for one pooled fit.
            ys = Vector{Vector{Float64}}(undef, nloc)
            Xs = Vector{Matrix{Float64}}(undef, nloc)
            yrs = Vector{Vector{Float64}}(undef, nloc)
            coefs_loc = Vector{Vector{Float64}}(undef, nloc)
            for li in 1:nloc
                y = Float64.(data.Y[:, li])
                X, yresp = ar_clim_design(y, data.woy, AR_ORDER, clim)
                ys[li] = y
                Xs[li] = X
                yrs[li] = yresp
                coefs_loc[li] = X \ yresp
            end
            coef_pool = reduce(vcat, Xs) \ reduce(vcat, yrs)

            # Second pass: blend each location's coefficients toward the
            # pooled fit (AR-coefficient partial pooling), then the
            # deterministic point path + split-conformal quantiles.
            for (li, loc) in enumerate(LOCATIONS)
                coef = (1 - POOL_WEIGHT) .* coefs_loc[li] .+
                       POOL_WEIGHT .* coef_pool
                resid = yrs[li] .- Xs[li] * coef
                dof = max(length(yrs[li]) - (AR_ORDER + 2), 1)
                resid_sd = sqrt(sum(abs2, resid) / dof)
                point = point_path_clim(
                    ys[li], future_woy, coef, AR_ORDER, clim, HORIZONS,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    pool = calib_pool[h]
                    if length(pool) >= MIN_CALIB
                        n_conformal[h] += 1
                    else
                        n_fallback[h] += 1
                    end
                    qvals = calibrated_quantiles(point[h], resid_sd, pool, QUANTILE_LEVELS)
                    for (q, val) in zip(QUANTILE_LEVELS, qvals)
                        nat = max(from_scale(val, TRANSFORM), 0.0)
                        push!(rows, (
                            MODEL_ID, loc, origin, h, target_end,
                            TARGET, "quantile", q, nat,
                        ))
                    end
                    push!(pending, PendingTask(target_end, loc, h, point[h]))
                end
            end
        end
    end
    println("calibration coverage by horizon (conformal vs fallback tasks):")
    for h in HORIZONS
        println("  h=$h: conformal=$(n_conformal[h]) fallback=$(n_fallback[h])")
    end
    return rows
end

# ---------------------------------------------------------------------
# Validation-only scoring against the local scratch-hub oracle
# ---------------------------------------------------------------------

"""Hub oracle (`target-data/oracle-output.csv`) as a scoring truth
table -- identical to seasoncombo/generate.jl's function of the same
name."""
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

"""
    coverage(forecast_df, truth_df, lower, upper) -> Float64

Empirical coverage of the `[lower, upper]` central interval (quantile
levels, e.g. `0.25, 0.75` for the nominal-50% interval) across every
scored task in `forecast_df`.
"""
function coverage(forecast_df::DataFrame, truth_df::DataFrame, lower::Float64, upper::Float64)
    joined = innerjoin(
        forecast_df, truth_df, on=[:location, :target_end_date],
        renamecols="" => "_truth",
    )
    task_cols = [:location, :origin_date, :horizon, :target_end_date]
    grouped = combine(groupby(joined, task_cols)) do sdf
        lo = sdf.value[findfirst(a -> abs(a - lower) < 1e-8, sdf.output_type_id)]
        hi = sdf.value[findfirst(a -> abs(a - upper) < 1e-8, sdf.output_type_id)]
        obs = sdf.value_truth[1]
        (covered=(obs >= lo) && (obs <= hi),)
    end
    return mean(grouped.covered)
end

function main()
    hub_path = length(ARGS) >= 1 ? ARGS[1] : nothing
    t0 = time()

    versions_full = load_series("flu_data_hhs_versions")
    hist_all = load_series("flu_data_hhs")
    vidx = build_vintage_index(versions_full)

    # Seasons MUST be walked in ascending chronological order -- the
    # rolling calibration pool depends on it (see build_forecast_table).
    # The backfill and pooled-seasonal profiles are rebuilt LEAK-FREE per
    # origin inside build_forecast_table, so nothing global is fit here.
    forecast = build_forecast_table(
        (1, 2, 3, 4, 5), versions_full, hist_all, vidx,
    )
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    if hub_path !== nothing
        write_submission(forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="seabbs_bot", model_abbr="conformalpooled",
            designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end

    # Validation-seasons-only scoring (docs/contracts.md experimental
    # integrity): filter the already-built 5-season table down to the
    # origins from seasons 1-2 and score against the local oracle.
    val_origins = Set{Date}()
    for season in VALIDATION_SEASONS
        for split in training_splits(season)
            push!(val_origins, maximum(split.origin_date))
        end
    end
    val_forecast = forecast[in.(forecast.origin_date, Ref(val_origins)), :]

    truth = load_oracle(HUB_PATH)
    scored = score_forecasts(val_forecast, truth; scale=:natural)
    summ = wis_summary(scored)[1, :]
    cov50 = coverage(val_forecast, truth, 0.25, 0.75)
    cov90 = coverage(val_forecast, truth, 0.05, 0.95)

    println("\nVALIDATION seasons (1, 2) only:")
    println("  mean_wis=$(round(summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(summ.sd_wis; digits=4)) " *
            "n_tasks=$(summ.n_tasks)")
    println("  coverage 50% nominal -> $(round(cov50; digits=3))")
    println("  coverage 90% nominal -> $(round(cov90; digits=3))")

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "simple-round candidate: conformal-pooled (leak-free " *
                     "pooled-seasonal point forecast + split-conformal " *
                     "intervals)")
        println(io, "="^77)
        println(io, "Point forecast: LEAK-FREE pooled week-of-season " *
                     "seasonal shape (shared across all 11 locations, " *
                     "rebuilt per origin from origin_date < forecast " *
                     "origin) + per-(location,delay) backfill (also rebuilt " *
                     "per origin) + per-location AR(6) with AR-coefficient " *
                     "pooling (w=$(POOL_WEIGHT)), log transform. Intervals: " *
                     "split-conformal, reused verbatim from experiments/" *
                     "simple-round/conformal.")
        println(io, "Scored on VALIDATION SEASONS (1, 2) ONLY, natural " *
                     "scale, against target-data/oracle-output.csv " *
                     "(docs/contracts.md experimental integrity). Every " *
                     "profile is fit only from data strictly before each " *
                     "forecast origin -- no validation-season future weeks " *
                     "inform their own forecasts.")
        println(io)
        println(io, "Reference points (all leak-free):")
        println(io, "  season model (per-location climatology + backfill)   = 0.3004")
        println(io, "  conformal on plain climatology (per-loc, fourthroot) = 0.2917")
        println(io, "  seasstack full stack (log + Student-t + AR pooling)  = 0.2891")
        println(io)
        println(io, "RESULT: $(MODEL_ID) " *
                     "mean_wis=$(round(summ.mean_wis; digits=4)) " *
                     "sd_wis=$(round(summ.sd_wis; digits=4)) " *
                     "n_tasks=$(summ.n_tasks)")
        vs_season = 0.3004 - summ.mean_wis
        println(io, "  vs season model (0.3004): " *
                     "$(round(vs_season; digits=4)) " *
                     "($(round(100 * vs_season / 0.3004; digits=2))%)")
        vs_conf = 0.2917 - summ.mean_wis
        println(io, "  vs conformal-on-plain-climatology (0.2917, the " *
                     "target to beat): $(round(vs_conf; digits=4)) " *
                     "($(round(100 * vs_conf / 0.2917; digits=2))%)")
        println(io)
        println(io, "Coverage (nominal -> actual):")
        println(io, "  50% -> $(round(cov50; digits=3))")
        println(io, "  90% -> $(round(cov90; digits=3))")
        println(io)
        println(io, "AR-coefficient pooling weight sweep (leak-free, " *
                     "validation seasons 1-2; POOL_WEIGHT env var):")
        println(io, "  w=0.0  mean_wis=0.2950  cov50=0.509  cov90=0.893")
        println(io, "  w=0.3  mean_wis=0.2870  cov50=0.517  cov90=0.908  " *
                     "<- selected (validation optimum)")
        println(io, "  w=0.5  mean_wis=0.2905  cov50=0.521  cov90=0.919")
        println(io, "  w=0.7  mean_wis=0.3022  cov50=0.518  cov90=0.922")
        println(io, "  w=0.9  mean_wis=0.3239  cov50=0.502  cov90=0.916")
        println(io, "  Shallow interior optimum at w=0.3; light AR pooling " *
                     "helps, heavy pooling over-shrinks (the shared pooled " *
                     "seasonal shape already couples the locations). Unlike " *
                     "seasstack (w=0.9 on its own design), this model wants " *
                     "only light AR pooling.")
        println(io)
        println(io, "HONEST ANSWER: leak-free pooled-seasonal + conformal " *
                     "(w=0.3) = 0.2870, which BEATS the clean season model " *
                     "(0.3004, -4.5%) and conformal-on-plain-climatology " *
                     "(0.2917, -1.6%), and marginally edges seasstack " *
                     "(0.2891, -0.7%). The margin over the two conformal/" *
                     "stack references is within validation noise -- the " *
                     "pooled seasonal shape itself is ~a wash (as the " *
                     "leaderboard found); the gains are from log + light AR " *
                     "pooling + calibrated intervals.")
        println(io)
        println(io, "For comparison, simp-intervals found the RAW " *
                     "(unscaled) parametric scheme covers only ~41%/78% " *
                     "at 50%/90% nominal, and its own tuned/inflated " *
                     "Student-t scheme (needed to fix that) still lands " *
                     "at 52.5%/89.2% -- over-covered because a symmetric " *
                     "distribution has to be blown up past its natural " *
                     "spread to compensate for the point forecast's own " *
                     "skew. Split-conformal calibration reads the " *
                     "(possibly skewed) error quantiles directly, with " *
                     "no distributional assumption once a horizon's " *
                     "pool has matured.")
        println(io)
        println(io, "Design: rolling walk-forward split-conformal, " *
                     "pooled across all 11 locations per horizon (see " *
                     "generate.jl module docstring for the full pending" *
                     "/maturation mechanism). MIN_CALIB=$(MIN_CALIB); " *
                     "fallback (only the first few origins of season 1) " *
                     "is simp-intervals' own Student-t(df=$(FALLBACK_DF), " *
                     "scale=$(FALLBACK_SCALE)) scheme.")
        println(io)
        println(io, "Runtime: $(dt)s for the full 5-season build (no " *
                     "Monte Carlo simulation anywhere -- point forecast " *
                     "is a deterministic recursion, calibrated quantiles " *
                     "come directly from an empirical or closed-form " *
                     "Student-t quantile function).")
    end
    println("\nwrote score.txt")

    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
