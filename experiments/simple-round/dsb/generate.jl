#!/usr/bin/env julia
# generate.jl -- DIFFERENCING + SEASONALITY + BACKFILL, all three
# combined -- simple-round front-runner combination.
#
# Three components, each the best-scoring variant of its own family
# sweep in this hub:
#
#   1. SEASONALITY: a single POOLED week-of-season shape shared across
#      all 11 locations, built from pre-2015 history
#      (`season_year <= 2014`, matching seasonpool/generate.jl's
#      cutoff). Unlike seasonpool (a pooled `N_HARMONICS`-harmonic
#      Fourier regression) this uses a smoothed pooled CLIMATOLOGY, in
#      the spirit of season/generate.jl's per-location climatology
#      (median-by-week-of-season, then a period-wrapped moving-average
#      smooth) but pooled across locations instead of fit per location:
#      each location's fourth-root series is first centred on its own
#      pre-2015 mean (as seasonpool does, so a location that simply
#      runs at a different level doesn't bias the shared shape), then
#      the centred values from all 11 locations are pooled into one set
#      of circular week-of-season bins, summarised by the MEDIAN per
#      bin (robust to a single season's peak noise), then smoothed with
#      a 5-week circular moving average. Per split, per location, a
#      2-parameter OLS regression (intercept + amplitude scaling of the
#      shared shape onto that split's own training window) adapts the
#      pooled curve to that location's own level and seasonal swing --
#      identical in form to seasonpool's `fit_seasonal_level`.
#   2. BACKFILL: the backfill sweep's best variant
#      (experiments/simple-round/backfill/score.txt) --
#      MULTIPLICATIVE, per-location, MEDIAN revision profile with a
#      6-week window (mean WIS 0.3586 alone vs 0.359 for ar6bf's
#      additive/window-8 choice), applied to `data.Y` before any
#      fitting.
#   3. DIFFERENCING: rather than fitting AR(p) directly to the
#      deseasonalised, backfill-corrected residual level (as every
#      other AR-family model in this hub does), this differences that
#      residual once and fits a low-order AR(q) to the FIRST DIFFERENCE
#      (an ARIMA(q,1,0)-style model). Forecasting integrates back:
#      simulated Gaussian-innovation paths of the differenced series
#      are cumulatively summed from the residual's last observed level
#      to reconstruct simulated residual LEVEL paths at each horizon.
#      The differencing order is chosen by a small internal sweep (see
#      `DIFF_ORDER_SWEEP` below) rather than fixed a priori.
#
# Forecast = pooled seasonal term at the (known) future week-of-season
# + integrated differenced-residual simulated paths, on top of a
# training series that already has the backfill nowcast correction
# applied to its recent weeks.
#
# This script runs a 4-way ABLATION (the full DSB combination, and each
# of the three components dropped in turn) plus the plain AR(6)
# baseline, all on IDENTICAL data/RNG plumbing for a fair comparison,
# and writes score.txt. "Drop differencing" replaces the differenced-
# residual AR(q) with a plain AR(12)-on-level fit (the best AR order
# found by experiments/simple-round/ar-order/sweep.jl), so that
# ablation is a like-for-like comparison against the strongest
# available non-differenced alternative, not a strawman AR(6).
#
# Scope: VALIDATION SEASONS (1, 2) ONLY, scored against the local hub
# clone's oracle (docs/contracts.md experimental integrity) -- this is
# a tuning/comparison experiment, not a submission driver. The backfill
# profile is built only from origin dates with `season_year <= 2016`
# (matches ar6bf/backfill sweep); the pooled seasonal shape only from
# `season_year <= 2014` (matches seasonpool) -- neither can leak the
# held-out TEST seasons (3-5).
#
# LIGHT + ANALYTIC: CSV/DataFrames/Statistics/LinearAlgebra only, no
# Turing.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl
# writes score.txt alongside this file; does not write a hub submission
# (no hub_path argument -- exploratory, not a `submissions/` candidate).

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
const HERE = @__DIR__
include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))
include(joinpath(PKG_DIR, "src", "hubio.jl"))
include(joinpath(PKG_DIR, "src", "scoring.jl"))

const TRANSFORM = :fourthroot
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12                # matches ar6bf/seasonpool's Dmax
const WINDOW_WEEKS = 104        # matches every other AR-family model
const BF_WINDOW = 6             # backfill sweep's best window
const MIN_SUPPORT = 5           # min sample size per (location, delay)
const SEASON_PERIOD = 52        # canonical annual cycle length (weeks)
const CLIMATOLOGY_YEAR = 2014   # pooled shape uses season_year <= this
const SMOOTH_WINDOW = 5         # circular moving-average width (weeks)
const AR_ORDER_LEVEL = 12       # "drop differencing" ablation's AR order
                                 # (ar-order sweep's best: 0.3518)
const DIFF_ORDER_SWEEP = (1, 2, 3, 4, 6)  # candidate AR(q) orders for the
                                           # differenced residual
const VALIDATION_ONLY = (1, 2)
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")

# ---------------------------------------------------------------------
# Backfill correction -- multiplicative / per-location / median,
# window 6 (experiments/simple-round/backfill/score.txt's best variant)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile on the `transform` scale:
for each `(location, delay)` with at least `min_support` recorded
versions at that delay, the MEDIAN of `settled / vintage` (both on
`transform` scale) across matching `(location, origin_date)` groups --
the multiplicative form, the backfill sweep's best-scoring mode.
`versions` must already be filtered by the caller to training-set
origin dates only (no test seasons). Rows with `|vintage| < 1e-6` are
skipped to avoid a near-zero-division blow-up.
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
            abs(vintage) < 1e-6 && continue
            key = (loc, delay)
            push!(get!(raw, key, Float64[]), settled / vintage)
        end
    end
    profile = Dict{Tuple{String,Int},Float64}()
    for (key, vals) in raw
        length(vals) >= min_support && (profile[key] = median(vals))
    end
    return profile
end

"""
    apply_backfill_correction!(data, profile; delay_cutoff)

Nudge `data.Y` in place, MULTIPLICATIVELY, at every `(t, l)` with
`0 <= data.delay[t, l] <= delay_cutoff` and a matching `profile` entry.
`profile` may be empty (the "drop backfill" ablation), in which case
this is a no-op.
"""
function apply_backfill_correction!(
    data::ModelData, profile::Dict{Tuple{String,Int},Float64};
    delay_cutoff::Int,
)
    for l in 1:data.L, t in 1:data.T
        d = data.delay[t, l]
        (d < 0 || d > delay_cutoff) && continue
        ismissing(data.Y[t, l]) && continue
        key = (LOCATIONS[l], d)
        haskey(profile, key) || continue
        data.Y[t, l] *= profile[key]
    end
    return data
end

# ---------------------------------------------------------------------
# Pooled seasonal climatology (shared shape, not per-location Fourier)
# ---------------------------------------------------------------------

"""
    build_pooled_shape(history; transform, period, cutoff_year,
                        smooth_window) -> Vector{Float64}

ONE shared week-of-season climatology curve, pooled across all 11
locations, on the `transform` scale, length `period`. Built only from
`history` (the `flu_data_hhs.csv` schema) restricted to
`season_year(origin_date) <= cutoff_year` (pre-2015 history, matching
seasonpool's cutoff).

Each location's `transform`-scale series is first centred on its own
mean over this window (so a location that simply runs at a different
level doesn't bias the shared shape -- seasonpool's centring step),
then every location's centred values are pooled into circular
week-of-season bins (`mod1(week_of_season(d), period)`), summarised by
the MEDIAN per bin (robust to a single season's peak noise, as
season/generate.jl's per-location climatology), and finally smoothed
with a `period`-wrapped `smooth_window`-wide moving average. With
~6,700 pooled (location, week) observations behind `period` bins, this
is far too well-supported to overfit the way a per-location Fourier fit
did (nfidd-ar6 + per-location Fourier(3): 0.412, worse than the plain
AR(6) baseline).
"""
function build_pooled_shape(
    history::DataFrame; transform::Symbol, period::Int, cutoff_year::Int,
    smooth_window::Int,
)
    hist = history[season_year.(history.origin_date) .<= cutoff_year, :]
    bins = [Float64[] for _ in 1:period]
    for g in groupby(hist, :location)
        vals = to_scale.(g.wili, transform)
        centred = vals .- mean(vals)
        for (v, d) in zip(centred, g.origin_date)
            b = mod1(week_of_season(d), period)
            push!(bins[b], v)
        end
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

"""
    shape_value(woy, shape, period) -> Float64

Shared pooled seasonal shape (deviation from a location's own mean, on
the `transform` scale) at week-of-season `woy`.
"""
shape_value(woy::Real, shape::Vector{Float64}, period::Int) =
    shape[mod1(round(Int, woy), period)]

"""
    fit_seasonal_level(y, woy_vec, shape, period) -> (alpha, beta)

Per-location OLS fit of `y_t = alpha + beta * shape(woy_t) + resid`:
the small per-location amplitude/level scaling of the shared pooled
shape onto that split's own training window (`y`, `woy_vec`). Only 2
parameters, so this never leaks future data and cannot meaningfully
overfit. Identical in form to seasonpool/generate.jl's
`fit_seasonal_level`.
"""
function fit_seasonal_level(
    y::AbstractVector{Float64}, woy_vec::AbstractVector{Int},
    shape::Vector{Float64}, period::Int,
)
    n = length(y)
    X = ones(n, 2)
    for (i, w) in enumerate(woy_vec)
        X[i, 2] = shape_value(w, shape, period)
    end
    alpha, beta = X \ y
    return alpha, beta
end

# ---------------------------------------------------------------------
# AR(order) fit -- shared by the "drop differencing" ablation (fit
# directly to the residual level) and the differencing branch (fit to
# the first difference of the residual)
# ---------------------------------------------------------------------

"""
    fit_ar(y, order) -> (coef, resid_sd)

OLS fit of an AR(`order`) model with intercept to `y` (ascending in
time, no missing values). `coef = [c, phi_1, ..., phi_order]`;
`resid_sd` is the in-sample residual standard deviation.
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
    simulate_paths_level(y, coef, resid_sd, order, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Simulate `npaths` Gaussian-innovation AR(`order`) sample paths forward
from the end of `y` (residual LEVEL scale), for each horizon in
`horizons`. Used by the "drop differencing" ablation.
"""
function simulate_paths_level(
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
    simulate_paths_diff(resid, coef, resid_sd, order, horizons, npaths;
                         rng) -> Dict{Int,Vector{Float64}}

Simulate `npaths` Gaussian-innovation AR(`order`) sample paths of the
FIRST DIFFERENCE of `resid` forward from its end, then integrate
(cumulative sum from `resid[end]`) to reconstruct simulated residual
LEVEL paths at each horizon -- the ARIMA(order,1,0)-style forecast used
by the differencing component. `coef`/`resid_sd` come from
`fit_ar(diff(resid), order)`; the AR lags driving each step are the
most recent `order` DIFFERENCES of `resid` (not levels).
"""
function simulate_paths_diff(
    resid::AbstractVector{Float64}, coef::Vector{Float64},
    resid_sd::Float64, order::Int, horizons, npaths::Int;
    rng::Random.AbstractRNG,
)
    d = diff(resid)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    tail0 = d[(end - order + 1):end]
    level0 = resid[end]
    for s in 1:npaths
        tail = copy(tail0)
        level = level0
        for h in 1:hmax
            pred = coef[1]
            for lag in 1:order
                pred += coef[lag + 1] * tail[end - lag + 1]
            end
            dval = pred + resid_sd * randn(rng)
            level += dval
            if h in horizons
                out[h][s] = level
            end
            push!(tail, dval)
            popfirst!(tail)
        end
    end
    return out
end

# ---------------------------------------------------------------------
# Forecast table builder -- toggles each of the three components
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, profile, versions_full, shape;
                          use_season, use_backfill, use_diff,
                          diff_order, model_id) -> DataFrame

Fit and forecast for every cross-validation split of every season in
`seasons`, toggling each of the three DSB components independently so
the same driver serves the full combination and every ablation:

  - `use_backfill`: apply the multiplicative/per-location/median
    backfill correction (window `BF_WINDOW`) to `data.Y` before
    fitting. When `false`, `profile` is ignored (pass an empty dict).
  - `use_season`: fit and remove the pooled seasonal term (2-parameter
    per-location adaptation of `shape`) before modelling the residual,
    adding it back at forecast time for the (known) future
    week-of-season. When `false`, the residual is the raw
    (backfill-corrected, if applicable) series.
  - `use_diff`: model the FIRST DIFFERENCE of the residual with
    AR(`diff_order`) and integrate simulated paths back to the level
    scale (`simulate_paths_diff`). When `false`, fit AR(`AR_ORDER_LEVEL`)
    directly to the residual level (`simulate_paths_level`) -- the
    "drop differencing" ablation.
"""
function build_forecast_table(
    seasons, profile, versions_full, shape; use_season::Bool,
    use_backfill::Bool, use_diff::Bool, diff_order::Int, model_id::String,
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
                split; Dmax=DMAX, transform=TRANSFORM,
                window_weeks=WINDOW_WEEKS, versions=versions_full,
            )
            if use_backfill
                apply_backfill_correction!(
                    data, profile; delay_cutoff=BF_WINDOW,
                )
            end
            origin = data.origin_date
            future_woy = [
                week_of_season(origin + Day(7 * h)) for h in HORIZONS
            ]
            for (li, loc) in enumerate(LOCATIONS)
                y = Float64.(data.Y[:, li])
                alpha, beta = 0.0, 0.0
                resid = y
                if use_season
                    alpha, beta = fit_seasonal_level(
                        y, data.woy, shape, SEASON_PERIOD,
                    )
                    seasonal_now = [
                        alpha + beta * shape_value(w, shape, SEASON_PERIOD)
                        for w in data.woy
                    ]
                    resid = y .- seasonal_now
                end

                if use_diff
                    d = diff(resid)
                    coef, resid_sd = fit_ar(d, diff_order)
                    paths = simulate_paths_diff(
                        resid, coef, resid_sd, diff_order, HORIZONS,
                        NPATHS; rng=rng,
                    )
                else
                    coef, resid_sd = fit_ar(resid, AR_ORDER_LEVEL)
                    paths = simulate_paths_level(
                        resid, coef, resid_sd, AR_ORDER_LEVEL, HORIZONS,
                        NPATHS; rng=rng,
                    )
                end

                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    seasonal_h = use_season ? alpha + beta * shape_value(
                        future_woy[h], shape, SEASON_PERIOD,
                    ) : 0.0
                    vals = paths[h] .+ seasonal_h
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

"""Hub oracle (`target-data/oracle-output.csv`) as a scoring truth
table."""
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

# ---------------------------------------------------------------------
# Run the ablation
# ---------------------------------------------------------------------

function run_variant(
    versions_full, profile, shape, truth; use_season, use_backfill,
    use_diff, diff_order, model_id,
)
    forecast = build_forecast_table(
        VALIDATION_ONLY, profile, versions_full, shape;
        use_season=use_season, use_backfill=use_backfill,
        use_diff=use_diff, diff_order=diff_order, model_id=model_id,
    )
    scored = score_forecasts(forecast, truth; scale=:natural)
    summ = wis_summary(scored)
    return (summary=summ[1, :], scored=scored)
end

function main()
    t0 = time()

    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_WINDOW,
        min_support=MIN_SUPPORT,
    )
    println("backfill profile (multiplicative/per-loc/median, window " *
            "$(BF_WINDOW)): $(length(profile)) (location, delay) " *
            "entries with >= $(MIN_SUPPORT) observations")

    history = load_series("flu_data_hhs")
    shape = build_pooled_shape(
        history; transform=TRANSFORM, period=SEASON_PERIOD,
        cutoff_year=CLIMATOLOGY_YEAR, smooth_window=SMOOTH_WINDOW,
    )
    println("pooled climatology shape (season_year <= " *
            "$(CLIMATOLOGY_YEAR), smoothed over $(SMOOTH_WINDOW) " *
            "weeks): range $(round(minimum(shape); digits=4)) to " *
            "$(round(maximum(shape); digits=4))")

    truth = load_oracle(HUB_PATH)
    empty_profile = Dict{Tuple{String,Int},Float64}()

    println("\n=== baseline-ar6: plain AR(6), no season/backfill/diff ===")
    # baseline-ar6 must use AR(6), not AR_ORDER_LEVEL=12 (that constant
    # is reserved for the "drop differencing" ablation), so it is built
    # as a one-off variant matching nfidd-ar6 exactly, rather than via
    # `run_variant`/`build_forecast_table` above.
    function run_ar6_baseline()
        rng = MersenneTwister(SEED)
        rows = DataFrame(
            model_id=String[], location=String[], origin_date=Date[],
            horizon=Int[], target_end_date=Date[], target=String[],
            output_type=String[], output_type_id=Float64[],
            value=Float64[],
        )
        for season in VALIDATION_ONLY
            for split in training_splits(season)
                data = build_model_data(
                    split; Dmax=DMAX, transform=TRANSFORM,
                    window_weeks=WINDOW_WEEKS,
                )
                origin = data.origin_date
                for (li, loc) in enumerate(LOCATIONS)
                    y = Float64.(data.Y[:, li])
                    coef, resid_sd = fit_ar(y, 6)
                    paths = simulate_paths_level(
                        y, coef, resid_sd, 6, HORIZONS, NPATHS; rng=rng,
                    )
                    for h in HORIZONS
                        target_end = origin + Day(7 * h)
                        vals = paths[h]
                        for q in QUANTILE_LEVELS
                            qval = quantile(vals, q)
                            nat = max(from_scale(qval, TRANSFORM), 0.0)
                            push!(rows, (
                                "baseline-ar6", loc, origin, h,
                                target_end, TARGET, "quantile", q, nat,
                            ))
                        end
                    end
                end
            end
        end
        scored = score_forecasts(rows, truth; scale=:natural)
        summ = wis_summary(scored)
        return (summary=summ[1, :], scored=scored)
    end
    ar6 = run_ar6_baseline()
    println("mean_wis=$(round(ar6.summary.mean_wis; digits=4)) " *
            "sd_wis=$(round(ar6.summary.sd_wis; digits=4)) " *
            "n_tasks=$(ar6.summary.n_tasks)")

    println("\n=== differencing-order sweep (full DSB combination) ===")
    diff_sweep = NamedTuple[]
    for order in DIFF_ORDER_SWEEP
        r = run_variant(
            versions_full, profile, shape, truth; use_season=true,
            use_backfill=true, use_diff=true, diff_order=order,
            model_id="dsb-sweep-$(order)",
        )
        push!(diff_sweep, (order=order, mean_wis=r.summary.mean_wis,
            sd_wis=r.summary.sd_wis))
        println("diff_order=$(order) -> " *
                "mean_wis=$(round(r.summary.mean_wis; digits=4)) " *
                "sd_wis=$(round(r.summary.sd_wis; digits=4))")
    end
    sort!(diff_sweep; by=r -> r.mean_wis)
    best_diff_order = diff_sweep[1].order
    println("best differencing order: $(best_diff_order)")

    println("\n=== full DSB: differencing + seasonality + backfill ===")
    full = run_variant(
        versions_full, profile, shape, truth; use_season=true,
        use_backfill=true, use_diff=true, diff_order=best_diff_order,
        model_id="dsb-full",
    )
    println("mean_wis=$(round(full.summary.mean_wis; digits=4)) " *
            "sd_wis=$(round(full.summary.sd_wis; digits=4)) " *
            "n_tasks=$(full.summary.n_tasks)")

    println("\n=== drop differencing: season + backfill, AR($(AR_ORDER_LEVEL)) " *
            "on residual level ===")
    drop_diff = run_variant(
        versions_full, profile, shape, truth; use_season=true,
        use_backfill=true, use_diff=false, diff_order=best_diff_order,
        model_id="drop-diff",
    )
    println("mean_wis=$(round(drop_diff.summary.mean_wis; digits=4)) " *
            "sd_wis=$(round(drop_diff.summary.sd_wis; digits=4)) " *
            "n_tasks=$(drop_diff.summary.n_tasks)")

    println("\n=== drop seasonality: differencing + backfill, no season ===")
    drop_season = run_variant(
        versions_full, profile, shape, truth; use_season=false,
        use_backfill=true, use_diff=true, diff_order=best_diff_order,
        model_id="drop-season",
    )
    println("mean_wis=$(round(drop_season.summary.mean_wis; digits=4)) " *
            "sd_wis=$(round(drop_season.summary.sd_wis; digits=4)) " *
            "n_tasks=$(drop_season.summary.n_tasks)")

    println("\n=== drop backfill: differencing + seasonality, no backfill ===")
    drop_backfill = run_variant(
        versions_full, empty_profile, shape, truth; use_season=true,
        use_backfill=false, use_diff=true, diff_order=best_diff_order,
        model_id="drop-backfill",
    )
    println("mean_wis=$(round(drop_backfill.summary.mean_wis; digits=4)) " *
            "sd_wis=$(round(drop_backfill.summary.sd_wis; digits=4)) " *
            "n_tasks=$(drop_backfill.summary.n_tasks)")

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "differencing + seasonality + backfill (DSB) -- " *
                     "simple-round front-runner combination")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "backfill profile: multiplicative / per-location / " *
                     "median, window $(BF_WINDOW) " *
                     "($(length(profile)) (location, delay) entries)")
        println(io, "pooled climatology shape: season_year <= " *
                     "$(CLIMATOLOGY_YEAR), smoothed over " *
                     "$(SMOOTH_WINDOW) weeks, range " *
                     "$(round(minimum(shape); digits=4)) to " *
                     "$(round(maximum(shape); digits=4))")
        println(io, "AR_ORDER_LEVEL (drop-differencing ablation): " *
                     "$(AR_ORDER_LEVEL)")
        println(io, "best differencing order (sweep over " *
                     "$(DIFF_ORDER_SWEEP)): $(best_diff_order)")
        println(io)
        println(io, "reference points from other experiments/READMEs " *
                     "(validation seasons 1,2):")
        println(io, "  nfidd-ar6 (plain AR(6))                    = 0.368")
        println(io, "  seabbs_bot-ar6bf (AR(6)+backfill, additive) = 0.359")
        println(io, "  ar-order (AR(12)+backfill)                 = 0.3518")
        println(io)
        println(io, "this run's own numbers (identical data/RNG " *
                     "plumbing across all variants, fair comparison):")
        println(io, "differencing-order sweep (full DSB combination):")
        println(io, rpad("order", 8) * rpad("mean_wis", 12) * "sd_wis")
        for r in diff_sweep
            println(io, rpad(string(r.order), 8) *
                         rpad(string(round(r.mean_wis; digits=4)), 12) *
                         string(round(r.sd_wis; digits=4)))
        end
        println(io)
        for (label, r) in (
            ("baseline-ar6", ar6), ("dsb-full", full),
            ("drop-differencing", drop_diff),
            ("drop-seasonality", drop_season),
            ("drop-backfill", drop_backfill),
        )
            println(io, rpad(label, 20) *
                         "mean_wis=$(round(r.summary.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.summary.sd_wis; digits=4)) " *
                         "n_tasks=$(r.summary.n_tasks)")
        end
        println(io)
        vs_ar6 = ar6.summary.mean_wis - full.summary.mean_wis
        vs_ar6_pct = 100 * vs_ar6 / ar6.summary.mean_wis
        vs_arorderbf = 0.3518 - full.summary.mean_wis
        vs_arorderbf_pct = 100 * vs_arorderbf / 0.3518
        println(io, "dsb-full vs plain AR(6) baseline: " *
                     "$(round(vs_ar6; digits=4)) ($(round(vs_ar6_pct; digits=2))%)")
        println(io, "dsb-full vs AR(12)+backfill (0.3518): " *
                     "$(round(vs_arorderbf; digits=4)) " *
                     "($(round(vs_arorderbf_pct; digits=2))%)")
        println(io)
        println(io, "-- ablation: contribution of each component --")
        println(io, "(positive = removing the component makes WIS worse, " *
                     "i.e. the component helps)")
        println(io, "drop differencing: " *
                     "$(round(drop_diff.summary.mean_wis - full.summary.mean_wis; digits=4))")
        println(io, "drop seasonality:  " *
                     "$(round(drop_season.summary.mean_wis - full.summary.mean_wis; digits=4))")
        println(io, "drop backfill:     " *
                     "$(round(drop_backfill.summary.mean_wis - full.summary.mean_wis; digits=4))")

        println(io)
        println(io, "-- breakdown by location (dsb-full vs drop-differencing) --")
        by_loc = combine(groupby(full.scored, :location),
            :wis => mean => :mean_wis, nrow => :n)
        base_by_loc = combine(groupby(drop_diff.scored, :location),
            :wis => mean => :mean_wis)
        merged_loc = innerjoin(
            by_loc, base_by_loc; on=:location,
            renamecols="_full" => "_dropdiff",
        )
        merged_loc.improvement = merged_loc.mean_wis_dropdiff .-
                                  merged_loc.mean_wis_full
        sort!(merged_loc, :improvement; rev=true)
        for row in eachrow(merged_loc)
            println(io, rpad(row.location, 16) *
                         "full=$(round(row.mean_wis_full; digits=4)) " *
                         "dropdiff=$(round(row.mean_wis_dropdiff; digits=4)) " *
                         "improvement=$(round(row.improvement; digits=4))")
        end

        println(io)
        println(io, "-- breakdown by season (dsb-full vs drop-differencing) --")
        full.scored.season_year = season_year.(full.scored.origin_date)
        drop_diff.scored.season_year = season_year.(drop_diff.scored.origin_date)
        by_season = combine(groupby(full.scored, :season_year),
            :wis => mean => :mean_wis, nrow => :n)
        base_by_season = combine(groupby(drop_diff.scored, :season_year),
            :wis => mean => :mean_wis)
        merged_season = innerjoin(
            by_season, base_by_season; on=:season_year,
            renamecols="_full" => "_dropdiff",
        )
        merged_season.improvement = merged_season.mean_wis_dropdiff .-
                                     merged_season.mean_wis_full
        for row in eachrow(merged_season)
            println(io, "season $(row.season_year): " *
                         "full=$(round(row.mean_wis_full; digits=4)) " *
                         "dropdiff=$(round(row.mean_wis_dropdiff; digits=4)) " *
                         "improvement=$(round(row.improvement; digits=4))")
        end

        println(io)
        println(io, "-- breakdown by horizon (dsb-full vs drop-differencing) --")
        by_h = combine(groupby(full.scored, :horizon),
            :wis => mean => :mean_wis, nrow => :n)
        base_by_h = combine(groupby(drop_diff.scored, :horizon),
            :wis => mean => :mean_wis)
        merged_h = innerjoin(
            by_h, base_by_h; on=:horizon,
            renamecols="_full" => "_dropdiff",
        )
        merged_h.improvement = merged_h.mean_wis_dropdiff .-
                                merged_h.mean_wis_full
        sort!(merged_h, :horizon)
        for row in eachrow(merged_h)
            println(io, "h=$(row.horizon): " *
                         "full=$(round(row.mean_wis_full; digits=4)) " *
                         "dropdiff=$(round(row.mean_wis_dropdiff; digits=4)) " *
                         "improvement=$(round(row.improvement; digits=4))")
        end

        println(io)
        winner = argmin([
            ar6.summary.mean_wis, full.summary.mean_wis,
            drop_diff.summary.mean_wis, drop_season.summary.mean_wis,
            drop_backfill.summary.mean_wis, 0.3518,
        ])
        labels = ("baseline-ar6", "dsb-full", "drop-differencing",
            "drop-seasonality", "drop-backfill", "ar-order12bf(0.3518)")
        println(io, "does DSB beat the AR(12)+backfill/ar6bf/ar6 " *
                     "front-runners? " *
                     "$(full.summary.mean_wis < 0.3518 ? "YES" : "NO") " *
                     "(dsb-full=$(round(full.summary.mean_wis; digits=4)) " *
                     "vs ar-order12bf=0.3518)")
        println(io, "overall best of this run's variants: $(labels[winner])")
    end

    dt = round(time() - t0; digits=1)
    println("\nwrote score.txt in $(dt)s total")
    return (ar6=ar6, full=full, drop_diff=drop_diff,
        drop_season=drop_season, drop_backfill=drop_backfill)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
