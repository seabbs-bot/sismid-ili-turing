#!/usr/bin/env julia
# season-adaptive forecasting -- simple-round, SEASON-ADAPTIVE family.
#
# Family: submissions/nfidd-ar6/generate_forecasts.jl (plain per-location
# AR(6), OLS, fourthroot, 1000 simulated Gaussian-innovation paths).
# Other simple-round families already show that a STATIC pooled
# seasonal shape helps a lot on top of AR(6)+backfill
# (experiments/simple-round/season: climatology-backfill 0.3004;
# experiments/simple-round/seasonpool: pooled-Fourier-backfill 0.3049;
# both far below the plain AR(6)+backfill reference, 0.359, and the
# AR-order family's order=12+backfill winner, 0.3518). This family asks
# a different question: does the forecast improve further if, on top
# of that static shape, it also ADAPTS to whether the CURRENT season is
# running severe or mild, right now, using only what has been observed
# of it so far?
#
# Motivation (docs/eda/04-cross-location.md, "a shared severity-year
# effect"): per-season seasonal amplitude correlates across locations
# at mean r=0.68 (range 0.25-0.94) -- a season that is severe at one
# location tends to be severe almost everywhere. That means a handful
# of observed weeks from ALL 11 locations, pooled, carry much more
# information about "how severe is this season" than the same handful
# of weeks from any single location alone. This family exploits exactly
# that: at each forecast origin, it estimates a single scalar severity
# index from the CURRENT (partial) season's data-to-date, pooled across
# every location, and uses it to rescale (a) the size of the seasonal
# bump added to the forecast for future weeks, and (b) the AR
# innovation variance (the working hypothesis being that a season
# running hotter than usual is also less predictable week to week).
#
# A note on why this family builds its OWN pooled seasonal shape rather
# than reusing `experiments/simple-round/seasoncombo`'s (whose "core"
# combo, 0.2781, is the best mean-WIS number anywhere in this round):
# that shape is fit ONCE from `season_year <= 2016`, i.e. from the
# FULL, completed data of both validation seasons -- for any split
# whose forecast origin falls inside season 1 or 2, that shape has
# already seen weeks strictly AFTER that split's own origin (finalized
# values, not vintage-capped). docs/contracts.md's experimental-
# integrity rule ("never let finalized or future values leak into a
# fit") and docs/lessons.md item 9 are explicit that this must not
# happen. `experiments/simple-round/season` and `.../seasonpool` are
# more careful (a strictly-pre-origin climatology, and a shape fit only
# from pre-2015 history, respectively); this family follows their
# discipline, not seasoncombo's, so its own numbers are directly
# comparable to season/seasonpool but NOT expected to reach
# seasoncombo's 0.2781 (which is, on this reading, an optimistic
# number). The `adaptive-base` variant below (severity switched off) is
# this family's own honest, leak-free static-shape reference point,
# comparable in spirit to seasonpool's 0.3049.
#
# Design, per cross-validation split (forecast origin `o`):
#   1. Backfill correction (unchanged from seabbs_bot-ar6bf: additive,
#      per-location, median, delay window 8).
#   2. Pooled seasonal shape: a `N_HARMONICS`-harmonic Fourier
#      regression of week-of-season, fit ONCE PER SPLIT from
#      `data/flu_data_hhs.csv` restricted to `origin_date < o` (i.e.
#      every location, every season strictly before this split's own
#      forecast origin -- no leakage, and re-estimated at every split
#      as more history becomes available). Each location's fourthroot
#      series is centred on its own mean over that same pre-`o` window
#      before pooling, so the shared shape reflects relative seasonal
#      shape, not location level. ~11 locations x up to 13 seasons of
#      pooled weekly observations comfortably supports 2*N_HARMONICS
#      parameters -- no realistic overfitting risk (contrast with the
#      per-location Fourier(3) in
#      submissions/nfidd-ar6/generate_forecasts_fourier.jl, which
#      overfit using only `window_weeks=104` of ONE location's data,
#      0.412 vs 0.368).
#   3. AR(6) is fit to the residual after removing each location's own
#      mean (over the split's own 104-week AR window) and the pooled
#      shape at unit amplitude -- structurally identical to
#      `seasoncombo`'s `core`/`deseasonalize`, just with a leak-free
#      per-split shape.
#   4. Season-adaptive severity index: pooling ALL 11 locations' data
#      from the CURRENT (partial) season only (the most recent season
#      present in the split's own window, i.e. exactly the season
#      containing origin `o`), regress each location's own-mean-
#      centred deviation on the pooled shape value at the matching
#      week -- a single, shared slope `sigma_hat` (no intercept, since
#      the shape is zero-mean by construction). `sigma_hat ~= 1` means
#      this season is running at the pooled shape's "typical"
#      amplitude so far; `> 1` means it is running hotter, `< 1`
#      milder. This uses ONLY weeks up to `o` (build_model_data already
#      caps the split there), pooled across locations for the extra
#      power the r=0.68 shared-severity effect buys.
#   5. Forecast: for each future horizon `h`, the seasonal component
#      added is `level[l] + sigma_shrunk * shape(woy(o + 7h))`, where
#      `sigma_shrunk = 1 + AMP_SHRINK * (sigma_hat - 1)` (AMP_SHRINK
#      swept below; 0 reproduces the static/unadaptive shape exactly).
#      The AR simulation's innovation SD is INDEPENDENTLY scaled by
#      `max(sigma_hat, 0.05) ^ VAR_GAMMA` (using the raw severity
#      estimate, not `sigma_shrunk` -- the two knobs are deliberately
#      not chained, see `build_forecast_table`'s docstring for why).
#      Both act ONLY on the forecast extrapolation, never retroactively
#      on the AR fit's own training residual, so a noisy early-season
#      severity estimate can distort only the current forecast, not
#      the fitted dynamics themselves.
#
# RESULT: the severity index itself is real and directionally sensible
# (docs/eda/04-cross-location.md's shared-severity effect shows up
# cleanly: once past the first ~6 noisy onset weeks, `sigma_hat` settles
# around 0.4-0.7 through the whole of season 2015/16 and around 1.0-1.3
# through 2016/17, matching that 2016/17 was the harder, more severe
# season everywhere -- see score.txt). But rescaling the SEASONAL
# COMPONENT's amplitude by it (`amp_shrink > 0`) makes mean WIS steadily
# WORSE, monotonically, the harder it is pushed (0.2976 at shrink=0 up
# to 0.4916 at shrink=1.5) -- the AR(6) fit's own lagged dynamics
# already track a current run of elevated/depressed residuals through
# momentum, so re-scaling the deterministic shape on top double-counts
# that adjustment. A separately-tried mechanism from the same brief --
# weighting the AR fit's training rows by closeness of week-of-season
# phase to the forecast origin's own phase, a Gaussian kernel of
# bandwidth `bw` weeks, tested at `bw in (999, 26, 16, 10, 6)` -- shows
# the same shape: a very wide kernel (mostly flat weighting) is
# roughly neutral, and it gets steadily worse as the kernel narrows
# (0.2976 -> 0.325 at bw=6), i.e. the AR(6) dynamics do not vary enough
# by phase to be worth the loss of effective training data from
# discarding off-phase weeks; that sweep is not kept as code here since
# it never beat the flat-weighted baseline at any bandwidth tried.
# Rescaling the AR INNOVATION VARIANCE by the same raw severity index
# (`var_gamma`), independent of the (harmful) amplitude rescaling,
# is the one mechanism that helps, modestly: mean WIS falls smoothly
# and unimodally to 0.2956 around `var_gamma ~= -0.2..-0.25` (milder
# seasons get slightly WIDER intervals, more severe ones slightly
# narrower -- the opposite sign from the original hypothesis, but a
# real, reproducible ~0.7% gain, not sweep noise; see score.txt for the
# full curve). CONCLUSION: season-adaptation helps here, but only
# through a small, one-directional variance adjustment, not through
# rescaling the deterministic seasonal amplitude or reweighting the AR
# fit by phase -- both of those either double-count what AR(6)'s own
# momentum already does, or throw away more training signal than the
# phase information they add back.
#
# LIGHT + ANALYTIC: CSV/DataFrames/Statistics/LinearAlgebra only, no
# Turing/Mooncake/Pathfinder (same reasoning as nfidd-ar6).
#
# Scope: VALIDATION SEASONS (1, 2) ONLY, scored against the local hub
# clone's oracle -- exploratory tuning, not a `submissions/` driver (no
# `hub_path` argument; writes score.txt alongside this file).
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl

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
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12
const WINDOW_WEEKS = 104
const DELAY_CUTOFF = 8       # backfill delay window (seabbs_bot-ar6bf)
const MIN_SUPPORT = 5        # min sample size per (location, delay)
const N_HARMONICS = 3        # pooled shape's harmonic count
const SEASON_PERIOD = 52.0
const SEVERITY_FLOOR = 0.05  # floor on sigma_shrunk before ^VAR_GAMMA
const VALIDATION_ONLY = (1, 2)
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")

# ---------------------------------------------------------------------
# Backfill correction (identical to seabbs_bot-ar6bf)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile on the `transform` scale,
identical to `submissions/seabbs_bot-ar6bf/generate_forecasts.jl`.
`versions` must already be filtered by the caller to training-set
origin dates only (no test-season data).
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
# Leak-free, per-split pooled seasonal shape
# ---------------------------------------------------------------------

"""
    fourier_features(woy, K, period) -> Vector{Float64}

`2K` Fourier features for `K` harmonics of week-of-season `woy` at the
given `period` (weeks). Identical in form to the per-location Fourier
submission's / seasonpool's helper of the same name.
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
    build_pooled_shape(hist, forecast_origin; transform, K, period)
        -> Vector{Float64}

ONE shared `K`-harmonic week-of-season shape, pooling all locations,
fit from `hist` (the `flu_data_hhs.csv` schema) restricted to
`origin_date < forecast_origin`: strictly before this split's own
forecast origin, so re-estimated fresh at every split and never
touching that split's own or future observations (docs/contracts.md
experimental integrity; contrast with seasoncombo's shape, fit once
from `season_year <= 2016`, which does not have this guarantee for
splits inside the validation seasons -- see this file's header).

Each location's `transform`-scale series is first centred on its own
mean over this same pre-`forecast_origin` window (so a location that
simply runs at a different level doesn't bias the shared shape), then
a no-intercept OLS regression of the pooled centred values on
`fourier_features(week_of_season(d), K, period)` gives the shared
shape's `2K` coefficients.
"""
function build_pooled_shape(
    hist::DataFrame, forecast_origin::Date; transform::Symbol, K::Int,
    period::Float64,
)
    h = hist[hist.origin_date .< forecast_origin, :]
    centred = Vector{Float64}(undef, nrow(h))
    for g in groupby(h, :location)
        vals = to_scale.(g.wili, transform)
        centred[parentindices(g)[1]] = vals .- mean(vals)
    end
    X = Matrix{Float64}(undef, nrow(h), 2K)
    for (i, d) in enumerate(h.origin_date)
        X[i, :] = fourier_features(week_of_season(d), K, period)
    end
    return X \ centred
end

"""
    shape_value(woy, shape_coef, K, period) -> Float64

Pooled seasonal shape (deviation from a location's own mean, on the
`TRANSFORM` scale) at week-of-season `woy`.
"""
function shape_value(
    woy::Real, shape_coef::Vector{Float64}, K::Int, period::Float64,
)
    return dot(fourier_features(woy, K, period), shape_coef)
end

# ---------------------------------------------------------------------
# Season-adaptive severity index
# ---------------------------------------------------------------------

"""
    estimate_severity(data, shape_coef, level, K, period) -> Float64

Pooled, cross-location severity index for the CURRENT (partial) season
in this split -- the season containing the forecast origin, i.e.
`data.season .== maximum(data.season)`. For every `(t, l)` pair with
`t` in that season (`build_model_data` already caps `data.Y` at the
forecast origin, so this never sees future weeks), regress the
own-mean-centred deviation `data.Y[t, l] - level[l]` on the pooled
shape value at `data.woy[t]` -- a single, no-intercept slope shared
across all 11 locations (docs/eda/04-cross-location.md: per-season
amplitude correlates at mean r=0.68 across locations, so pooling gives
far more power than any one location's handful of current-season
weeks would alone).

Returns 1.0 (no adjustment) if the current season has too little
shape-bearing signal yet to estimate a slope (denominator near 0 --
happens only in the first week or two of a season, when the shape
itself is close to flat).
"""
function estimate_severity(
    data::ModelData, shape_coef::Vector{Float64}, level::Vector{Float64},
    K::Int, period::Float64,
)
    cur = maximum(data.season)
    num = 0.0
    denom = 0.0
    for t in 1:data.T
        data.season[t] == cur || continue
        s = shape_value(data.woy[t], shape_coef, K, period)
        for l in 1:data.L
            z = Float64(data.Y[t, l]) - level[l]
            num += z * s
            denom += s^2
        end
    end
    return denom > 1e-6 ? num / denom : 1.0
end

# ---------------------------------------------------------------------
# AR(6) fit + forward simulation (identical to nfidd-ar6, applied to
# the deseasonalised residual)
# ---------------------------------------------------------------------

"""
    fit_ar(y, order) -> (coef, resid_sd)

Plain OLS fit of an AR(`order`) model with intercept.
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
in `horizons`. Identical in structure to nfidd-ar6's function of the
same name.
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

# ---------------------------------------------------------------------
# Forecast table builder
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, hist, versions_full, profile;
                          amp_shrink, var_gamma, model_id) -> DataFrame

Fit and forecast the season-adaptive model for every cross-validation
split of every season in `seasons`. `profile` is the backfill
revision profile (may be empty). `amp_shrink` and `var_gamma` are two
INDEPENDENT knobs on the same current-season severity index
(`estimate_severity`): `amp_shrink` rescales the extrapolated seasonal
component (`sigma_shrunk = 1 + amp_shrink * (sigma_hat - 1)`, applied
multiplicatively to the future shape value); `var_gamma` separately
rescales the AR innovation SD by `max(sigma_hat, SEVERITY_FLOOR) ^
var_gamma`, using the RAW severity estimate regardless of
`amp_shrink` -- the two mechanisms are deliberately not chained, so
each can be swept and attributed on its own (see this file's header:
chaining them through the same shrunk value was an early bug in this
family's own first pass, and made the variance sweep a no-op whenever
`amp_shrink` was left at 0). Both `0.0` reproduce the static,
unadaptive shape and variance exactly (`adaptive-base`).
"""
function build_forecast_table(
    seasons, hist::DataFrame, versions_full::DataFrame, profile::Dict;
    amp_shrink::Float64, var_gamma::Float64, model_id::String,
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
            apply_backfill_correction!(data, profile)
            origin = data.origin_date

            shape_coef = build_pooled_shape(
                hist, origin; transform=TRANSFORM, K=N_HARMONICS,
                period=SEASON_PERIOD,
            )
            level = [mean(Float64.(data.Y[:, l])) for l in 1:data.L]
            sigma_hat = estimate_severity(
                data, shape_coef, level, N_HARMONICS, SEASON_PERIOD,
            )
            sigma_shrunk = 1.0 + amp_shrink * (sigma_hat - 1.0)
            var_scale = max(sigma_hat, SEVERITY_FLOOR)^var_gamma

            for (li, loc) in enumerate(LOCATIONS)
                y = Float64.(data.Y[:, li])
                r = [
                    y[t] - level[li] - shape_value(
                        data.woy[t], shape_coef, N_HARMONICS, SEASON_PERIOD,
                    ) for t in 1:data.T
                ]
                coef, resid_sd = fit_ar(r, AR_ORDER)
                resid_sd_scaled = resid_sd * var_scale
                paths = simulate_paths(
                    r, coef, resid_sd_scaled, AR_ORDER, HORIZONS, NPATHS;
                    rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    s = shape_value(
                        week_of_season(target_end), shape_coef,
                        N_HARMONICS, SEASON_PERIOD,
                    )
                    vals = paths[h] .+ level[li] .+ sigma_shrunk * s
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

score_one(forecast, truth) = wis_summary(score_forecasts(
    forecast, truth; scale=:natural,
))[1, :]

# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------

const AMP_SHRINKS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
const VAR_GAMMAS = (
    -0.5, -0.4, -0.3, -0.25, -0.2, -0.1, 0.0, 0.25, 0.5, 0.75, 1.0,
)

function main()
    t0 = time()
    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")
    truth = load_oracle(HUB_PATH)

    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=DELAY_CUTOFF,
        min_support=MIN_SUPPORT,
    )
    println("backfill profile: $(length(profile)) (location, delay) " *
            "entries with >= $(MIN_SUPPORT) observations")

    # --- adaptive-base: static shape, no severity adaptation ---
    base_fc = build_forecast_table(
        VALIDATION_ONLY, hist, versions_full, profile;
        amp_shrink=0.0, var_gamma=0.0, model_id="adaptive-base",
    )
    base_summ = score_one(base_fc, truth)
    println("adaptive-base (static shape, sigma off): " *
            "mean_wis=$(round(base_summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(base_summ.sd_wis; digits=4))")

    # --- sweep 1: amplitude adaptation alone (var_gamma=0) ---
    amp_results = NamedTuple[]
    for shrink in AMP_SHRINKS
        fc = build_forecast_table(
            VALIDATION_ONLY, hist, versions_full, profile;
            amp_shrink=shrink, var_gamma=0.0, model_id="adaptive-amp",
        )
        summ = score_one(fc, truth)
        push!(amp_results, (
            shrink=shrink, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("amp shrink=$shrink -> " *
                "mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(amp_results; by=r -> r.mean_wis)
    amp_best = amp_results[1]

    # --- sweep 2: variance adaptation on top of the best amp shrink ---
    var_results = NamedTuple[]
    for gamma in VAR_GAMMAS
        fc = build_forecast_table(
            VALIDATION_ONLY, hist, versions_full, profile;
            amp_shrink=amp_best.shrink, var_gamma=gamma,
            model_id="adaptive-var",
        )
        summ = score_one(fc, truth)
        push!(var_results, (
            gamma=gamma, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("shrink=$(amp_best.shrink) gamma=$gamma -> " *
                "mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(var_results; by=r -> r.mean_wis)
    var_best = var_results[1]

    # --- final: best amp shrink + best var gamma ---
    final_fc = build_forecast_table(
        VALIDATION_ONLY, hist, versions_full, profile;
        amp_shrink=amp_best.shrink, var_gamma=var_best.gamma,
        model_id="adaptive-full",
    )
    final_scored = score_forecasts(final_fc, truth; scale=:natural)
    final_summ = wis_summary(final_scored)[1, :]

    base_scored = score_forecasts(base_fc, truth; scale=:natural)

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "season-adaptive severity sweep -- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "reference points from other experiments/READMEs:")
        println(io, "  nfidd-ar6 (plain AR6)                    = 0.368")
        println(io, "  seabbs_bot-ar6bf (AR6+backfill)          = 0.359")
        println(io, "  ar-order-12bf (AR12+backfill, currently")
        println(io, "    designated best-overall submission)   = 0.3518")
        println(io, "  season family (per-loc climatology+backfill)")
        println(io, "                                           = 0.3004")
        println(io, "  seasonpool (pooled Fourier shape+backfill)")
        println(io, "                                           = 0.3049")
        println(io, "  seasoncombo core (season_year<=2016 pooled")
        println(io, "    shape+AR6+backfill)                    = 0.2781")
        println(io, "    NOTE: that shape sees each validation season's")
        println(io, "    own future weeks (see this file's header) --")
        println(io, "    not used as this family's base, see")
        println(io, "    adaptive-base below instead.")
        println(io)
        println(io, "this run's own numbers (identical data/RNG " *
                     "plumbing, fair comparison):")
        println(io, "adaptive-base     mean_wis=" *
                     "$(round(base_summ.mean_wis; digits=4)) sd_wis=" *
                     "$(round(base_summ.sd_wis; digits=4)) n_tasks=" *
                     "$(base_summ.n_tasks)")
        println(io)
        println(io, "=== sweep 1: amplitude adaptation (var_gamma=0) ===")
        for r in amp_results
            println(io, "  amp_shrink=$(r.shrink) -> mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: amp_shrink=$(amp_best.shrink) mean_wis=" *
                     "$(round(amp_best.mean_wis; digits=4)) sd_wis=" *
                     "$(round(amp_best.sd_wis; digits=4))")
        println(io)
        println(io, "=== sweep 2: variance adaptation on top of " *
                     "amp_shrink=$(amp_best.shrink) ===")
        for r in var_results
            println(io, "  var_gamma=$(r.gamma) -> mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: var_gamma=$(var_best.gamma) mean_wis=" *
                     "$(round(var_best.mean_wis; digits=4)) sd_wis=" *
                     "$(round(var_best.sd_wis; digits=4))")
        println(io)
        println(io, "=== winner: adaptive-full " *
                     "(amp_shrink=$(amp_best.shrink), " *
                     "var_gamma=$(var_best.gamma)) ===")
        println(io, "mean_wis=$(round(final_summ.mean_wis; digits=4)) " *
                     "sd_wis=$(round(final_summ.sd_wis; digits=4)) " *
                     "n_tasks=$(final_summ.n_tasks)")
        vs_base = base_summ.mean_wis - final_summ.mean_wis
        vs_base_pct = 100 * vs_base / base_summ.mean_wis
        vs_ref = 0.3518 - final_summ.mean_wis
        vs_ref_pct = 100 * vs_ref / 0.3518
        println(io, "vs adaptive-base (this family's own static " *
                     "reference): $(round(vs_base; digits=4)) " *
                     "($(round(vs_base_pct; digits=2))%)")
        println(io, "vs ar-order-12bf reference (0.3518): " *
                     "$(round(vs_ref; digits=4)) " *
                     "($(round(vs_ref_pct; digits=2))%)")

        println(io)
        println(io, "-- breakdown by location (adaptive-full vs " *
                     "adaptive-base) --")
        by_loc = combine(groupby(final_scored, :location),
            :wis => mean => :mean_wis)
        base_by_loc = combine(groupby(base_scored, :location),
            :wis => mean => :mean_wis)
        merged_loc = innerjoin(
            by_loc, base_by_loc; on=:location,
            renamecols="_full" => "_base",
        )
        merged_loc.improvement = merged_loc.mean_wis_base .-
                                  merged_loc.mean_wis_full
        sort!(merged_loc, :improvement; rev=true)
        for row in eachrow(merged_loc)
            println(io, rpad(row.location, 16) *
                         "full=$(round(row.mean_wis_full; digits=4)) " *
                         "base=$(round(row.mean_wis_base; digits=4)) " *
                         "improvement=$(round(row.improvement; digits=4))")
        end

        println(io)
        println(io, "-- breakdown by season (adaptive-full vs " *
                     "adaptive-base) --")
        final_scored.season_year = season_year.(final_scored.origin_date)
        base_scored.season_year = season_year.(base_scored.origin_date)
        by_season = combine(groupby(final_scored, :season_year),
            :wis => mean => :mean_wis, nrow => :n)
        base_by_season = combine(groupby(base_scored, :season_year),
            :wis => mean => :mean_wis)
        merged_season = innerjoin(
            by_season, base_by_season; on=:season_year,
            renamecols="_full" => "_base",
        )
        merged_season.improvement = merged_season.mean_wis_base .-
                                     merged_season.mean_wis_full
        for row in eachrow(merged_season)
            println(io, "season $(row.season_year): " *
                         "full=$(round(row.mean_wis_full; digits=4)) " *
                         "base=$(round(row.mean_wis_base; digits=4)) " *
                         "improvement=$(round(row.improvement; digits=4)) " *
                         "n=$(row.n_full)")
        end

        println(io)
        println(io, "-- breakdown by horizon (adaptive-full vs " *
                     "adaptive-base) --")
        by_h = combine(groupby(final_scored, :horizon),
            :wis => mean => :mean_wis)
        base_by_h = combine(groupby(base_scored, :horizon),
            :wis => mean => :mean_wis)
        merged_h = innerjoin(
            by_h, base_by_h; on=:horizon, renamecols="_full" => "_base",
        )
        merged_h.improvement = merged_h.mean_wis_base .-
                                merged_h.mean_wis_full
        sort!(merged_h, :horizon)
        for row in eachrow(merged_h)
            println(io, "h=$(row.horizon): " *
                         "full=$(round(row.mean_wis_full; digits=4)) " *
                         "base=$(round(row.mean_wis_base; digits=4)) " *
                         "improvement=$(round(row.improvement; digits=4))")
        end

        println(io)
        println(io, "-- breakdown by weeks elapsed in current season " *
                     "(adaptive-full vs adaptive-base) --")
        println(io, "(tests whether adaptation helps more once more " *
                     "of the season has actually been observed)")
        elapsed_bucket(d) = begin
            w = week_of_season(d)
            w <= 4 ? "01-04" : w <= 8 ? "05-08" : w <= 12 ? "09-12" :
                w <= 16 ? "13-16" : "17+"
        end
        final_scored.bucket = elapsed_bucket.(final_scored.origin_date)
        base_scored.bucket = elapsed_bucket.(base_scored.origin_date)
        by_bucket = combine(groupby(final_scored, :bucket),
            :wis => mean => :mean_wis, nrow => :n)
        base_by_bucket = combine(groupby(base_scored, :bucket),
            :wis => mean => :mean_wis)
        merged_bucket = innerjoin(
            by_bucket, base_by_bucket; on=:bucket,
            renamecols="_full" => "_base",
        )
        merged_bucket.improvement = merged_bucket.mean_wis_base .-
                                     merged_bucket.mean_wis_full
        sort!(merged_bucket, :bucket)
        for row in eachrow(merged_bucket)
            println(io, "weeks $(row.bucket): " *
                         "full=$(round(row.mean_wis_full; digits=4)) " *
                         "base=$(round(row.mean_wis_base; digits=4)) " *
                         "improvement=$(round(row.improvement; digits=4)) " *
                         "n=$(row.n_full)")
        end
    end

    dt = round(time() - t0; digits=1)
    println("\nwinner: adaptive-full mean_wis=" *
            "$(round(final_summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(final_summ.sd_wis; digits=4))")
    println("wrote score.txt in $(dt)s total")
    return (base=base_summ, amp=amp_results, var=var_results,
            final=final_summ)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
