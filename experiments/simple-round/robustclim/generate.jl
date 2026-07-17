#!/usr/bin/env julia
# generate.jl -- ROBUST CLIMATOLOGY ESTIMATION, simple-round SEASONALITY
# family. Builds directly on
# experiments/simple-round/season/generate.jl (MODEL_ID
# "seabbs_bot-season", this family's pick so far: per-location
# climatology + AR(6) + backfill, 0.3004 mean validation WIS --
# experiments/simple-round/season/score.txt). A different residual
# mechanism (pooled shape + backfill on a per-location AR(6), no
# per-location climatology regressor) does better still
# (experiments/simple-round/seasoncombo "core" combo, 0.2781), and its
# own follow-up (experiments/simple-round/seasondrift) shows a
# RECENCY-WEIGHTED version of that pooled shape does better again
# (0.2602 at decay=0.4) -- both useful reference points, but neither is
# what this file tunes: this file only touches season/generate.jl's own
# CLIMATOLOGY TERM (the smoothed per-location median-by-week-of-season
# curve), everything else (AR(6), backfill correction, fourthroot
# scale, OLS fit, Gaussian-innovation simulation) is unchanged from that
# file.
#
# The original climatology estimator is: bin historical (pre-forecast-
# origin) observations by week-of-season, take the MEDIAN per bin
# (equal weight to every historical season, robust to a single season's
# peak noise), then smooth with a 5-week circular moving average. Four
# ways to improve that estimator are swept here, each isolated then
# stacked (season/seasondrift/smoother's staged pattern):
#
#   1. RECENCY WEIGHTING (`decay`): weight historical seasons by
#      `decay^(seasons_ago)` before summarising each bin, instead of
#      the original's implicit equal weighting -- severity and peak
#      timing drift over the 17 years of history
#      (data/flu_data_hhs.csv spans 2002-2019, docs/eda/03-
#      seasonality.md), so recent seasons may be a better guide to the
#      upcoming one than distant ones.
#   2. ROBUST ESTIMATION (`trim_frac`): replace the plain median with a
#      (recency-)weighted TRIMMED MEAN, dropping the most extreme
#      `trim_frac` share of a bin's total weight from each tail before
#      averaging -- a trimmed mean's `trim_frac -> 0.5` limit is the
#      median itself, so this generalises rather than replaces the
#      original's robustness, and makes it tunable instead of fixed.
#   3. SMOOTHER SPLINE (`smooth_method = :spline`): replace the fixed-
#      width circular moving average with a circular Whittaker/P-spline
#      smoother (`whittaker_smooth_circular`) -- a single smoothness
#      penalty (`spline_lambda`) on the curve's second difference,
#      solved once per split as a 52x52 linear system, rather than a
#      fixed local window.
#   4. PER-LOCATION VS POOLED BLEND (`blend`): re-level the pooled
#      (cross-location, seasoncombo-style) shape to each location's own
#      median and linearly blend it with that location's own
#      climatology curve -- `blend=0` is the original per-location-only
#      design, `blend=1` is pure pooled-shape-at-this-location's-level,
#      values between borrow strength across locations without
#      discarding location-specific timing entirely.
#
# All four knobs share one `ClimConfig` and are estimated ONLY from
# history strictly before each split's own forecast origin (no test-
# season or future leakage), exactly as the original.
#
# Scope: this file both TUNES (validation seasons 1, 2 only,
# docs/contracts.md experimental integrity -- see the staged sweep
# below, captured verbatim into score.txt) and, given a `hub_path`
# argument, WRITES the full 5-season hub submission (1-2 validation,
# 3-5 held-out test) for whatever configuration the validation sweep
# itself picks as the winner -- test-season data is never used for
# tuning, only a per-origin vintage fit capped at its own forecast
# origin (same discipline as season/generate.jl).
#
# Deliberately LIGHT + ANALYTIC (no Turing): CSV/DataFrames/Statistics/
# LinearAlgebra only.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl [hub_path]
# With no hub_path, only runs the validation sweep (redirect stdout to
# reproduce score.txt). With hub_path, additionally writes the winning
# configuration's full 5-season hub-format submission.

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

const MODEL_ID = "seabbs_bot-robustclim"
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12            # matches season/generate.jl's build_model_data Dmax
const WINDOW_WEEKS = 104   # caps AR history at 2 seasons, as nfidd-ar6
const SEASON_PERIOD = 52   # canonical annual cycle length for climatology
const DELAY_CUTOFF = 8     # weeks; backfill profile is ~0 beyond this
const MIN_SUPPORT = 5      # min sample size per (location, delay) to trust
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")  # local oracle for scoring

# ---------------------------------------------------------------------
# Backfill correction (identical to season/generate.jl)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile on the `transform` scale.
Identical to `experiments/simple-round/season/generate.jl`; see that
file for the full derivation. `versions` must already be filtered by
the caller to the training set only (no test seasons).
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
# Robust, recency-weighted, spline-smoothed climatology
# ---------------------------------------------------------------------

"""
    ClimConfig

Bundle of the four climatology-estimation knobs this experiment sweeps,
kept together so `build_climatology`/`build_pooled_deviation`/the sweep
loop below share one object instead of five positional arguments.

  - `decay`: per-season recency weight, `decay^(seasons_ago)`, applied
    to every historical observation before it enters a week-of-season
    bin. `1.0` recovers equal weighting of every historical season
    (the original climatology's implicit behaviour); `< 1.0` favours
    recent seasons.
  - `trim_frac`: fraction of a bin's TOTAL WEIGHT trimmed from each
    tail before averaging (see `trimmed_weighted_mean`) -- the ROBUST
    estimation axis. `0.0` is a plain (recency-)weighted mean;
    approaching `0.5` approaches the (recency-weighted) MEDIAN the
    original climatology used, since a median is a trimmed mean's
    `trim_frac -> 0.5` limit.
  - `smooth_method`: `:movavg` (the original circular moving average)
    or `:spline` (a circular Whittaker/P-spline smoother, see
    `whittaker_smooth_circular`).
  - `smooth_window`: circular moving-average span, used when
    `smooth_method == :movavg`.
  - `spline_lambda`: second-difference penalty weight, used when
    `smooth_method == :spline`.
  - `blend`: `0.0` uses the per-location climatology alone (the
    original design); `1.0` uses the POOLED (cross-location) shape,
    re-levelled to this location's own median, alone; values between
    linearly blend the two (see `climatology_for`).
"""
struct ClimConfig
    decay::Float64
    trim_frac::Float64
    smooth_method::Symbol
    smooth_window::Int
    spline_lambda::Float64
    blend::Float64
end

"""
    trimmed_weighted_mean(vals, weights, trim_frac) -> Float64

Recency-weighted, trimmed-mean summary of one climatology bin.
`vals[i]` carries weight `weights[i]`. Values are sorted, the extreme
`trim_frac` share of TOTAL WEIGHT is dropped from each tail (a value
straddling a trim boundary contributes only its within-bounds weight
fraction, so this is continuous in `trim_frac`, not a hard include/
exclude per season), and the remainder is combined by weight. This is
what "trims outlier seasons" per bin: an unusually early/late/severe
season's contribution to a week is downweighted or dropped rather than
pulling the mean toward it. `trim_frac = 0` is a plain weighted mean;
`trim_frac -> 0.5` approaches the weighted median.
"""
function trimmed_weighted_mean(
    vals::Vector{Float64}, weights::Vector{Float64}, trim_frac::Float64,
)
    n = length(vals)
    n == 0 && return 0.0
    ord = sortperm(vals)
    v = vals[ord]
    w = weights[ord]
    total = sum(w)
    total <= 0 && return mean(vals)
    cum = cumsum(w)
    lo = trim_frac * total
    hi = (1 - trim_frac) * total
    num = 0.0
    den = 0.0
    for i in 1:n
        c0 = i == 1 ? 0.0 : cum[i - 1]
        c1 = cum[i]
        overlap = min(c1, hi) - max(c0, lo)
        if overlap > 0
            num += v[i] * overlap
            den += overlap
        end
    end
    return den > 0 ? num / den : sum(v .* w) / total
end

"""
    circular_movavg(raw, window) -> Vector{Float64}

The original climatology's smoother: a `window`-wide circular moving
average (week 1 and the last week are adjacent, since week-of-season
wraps around the year).
"""
function circular_movavg(raw::Vector{Float64}, window::Int)
    period = length(raw)
    half = window ÷ 2
    smoothed = similar(raw)
    for i in 1:period
        idxs = [mod1(i + o, period) for o in (-half):half]
        smoothed[i] = mean(raw[idxs])
    end
    return smoothed
end

"""
    whittaker_smooth_circular(raw, lambda) -> Vector{Float64}

Circular Whittaker/P-spline smoother: solves
`(I + lambda * D'D) x = raw` for `x`, where `D` is the discrete second-
difference operator on the circular (period-wrapped) week index. This
is the natural smoothing-spline replacement for a fixed-width moving
average: rather than a local window, it penalises the SECOND
DIFFERENCE of the whole curve (its curvature) and solves for the
single curve of the chosen smoothness that best follows the raw
per-week statistic everywhere at once, still wrapping week
`period` back to week 1. `lambda` plays the role `smooth_window` did
for the moving average: larger `lambda` gives a smoother, flatter
curve. `period` (52) is small enough that building and solving the
dense linear system once per split is cheap.
"""
function whittaker_smooth_circular(raw::Vector{Float64}, lambda::Float64)
    n = length(raw)
    D = zeros(n, n)
    for i in 1:n
        D[i, mod1(i - 1, n)] += 1.0
        D[i, i] += -2.0
        D[i, mod1(i + 1, n)] += 1.0
    end
    A = Matrix{Float64}(I, n, n) + lambda * (D' * D)
    return A \ raw
end

"""
    smooth_curve(raw, cfg) -> Vector{Float64}

Dispatch to `circular_movavg` or `whittaker_smooth_circular` per
`cfg.smooth_method`.
"""
function smooth_curve(raw::Vector{Float64}, cfg::ClimConfig)
    return cfg.smooth_method == :spline ?
        whittaker_smooth_circular(raw, cfg.spline_lambda) :
        circular_movavg(raw, cfg.smooth_window)
end

"""
    bin_weighted(sub, cur_season, cfg; period) -> Vector{Float64}

Shared binning step for both the per-location and pooled climatology:
buckets `sub`'s `(origin_date, value)` rows (already on the modelling
scale) by circular week-of-season, weights each by
`cfg.decay^(cur_season - season_year(origin_date))`, and reduces each
bin with `trimmed_weighted_mean`. Empty bins fall back to the pooled-
across-all-bins trimmed weighted mean.
"""
function bin_weighted(
    dates::AbstractVector{Date}, values::AbstractVector{Float64},
    cur_season::Int, cfg::ClimConfig; period::Int=SEASON_PERIOD,
)
    bin_vals = [Float64[] for _ in 1:period]
    bin_wts = [Float64[] for _ in 1:period]
    for (d, x) in zip(dates, values)
        b = mod1(week_of_season(d), period)
        w = cfg.decay^(cur_season - season_year(d))
        push!(bin_vals[b], x)
        push!(bin_wts[b], w)
    end
    allvals = reduce(vcat, bin_vals; init=Float64[])
    allwts = reduce(vcat, bin_wts; init=Float64[])
    overall = isempty(allvals) ? 0.0 :
        trimmed_weighted_mean(allvals, allwts, cfg.trim_frac)
    raw = [
        isempty(bin_vals[b]) ? overall :
            trimmed_weighted_mean(bin_vals[b], bin_wts[b], cfg.trim_frac)
        for b in 1:period
    ]
    return raw
end

"""
    build_climatology(loc_hist, forecast_origin, cfg; period) -> Vector{Float64}

Recency-weighted, trimmed, smoothed circular week-of-season
climatology curve for ONE location, on the `TRANSFORM` scale, length
`period`. Built ONLY from `loc_hist` rows strictly before
`forecast_origin` (no leakage); `loc_hist` is that location's full
historical `(origin_date, wili)` rows, unfiltered by date -- filtering
happens here.
"""
function build_climatology(
    loc_hist::DataFrame, forecast_origin::Date, cfg::ClimConfig;
    period::Int=SEASON_PERIOD,
)
    sub = loc_hist[loc_hist.origin_date .< forecast_origin, :]
    isempty(sub) && return zeros(period)
    cur_season = season_year(forecast_origin)
    values = to_scale.(sub.wili, TRANSFORM)
    raw = bin_weighted(sub.origin_date, values, cur_season, cfg; period=period)
    return smooth_curve(raw, cfg)
end

"""
    build_pooled_deviation(hist_by_loc, forecast_origin, cfg; period)
        -> Vector{Float64}

Pooled (cross-location) week-of-season DEVIATION shape: each location's
history is first centred on that location's own trimmed-weighted-mean
level (so absolute-scale differences across locations,
docs/eda/03-seasonality.md, don't bias the pooled shape), the centred
values from ALL locations are pooled into one set of circular bins, and
reduced/smoothed exactly as `build_climatology`. One call per split
(not per location) -- reused by `climatology_for` for every location's
blend.
"""
function build_pooled_deviation(
    hist_by_loc::Dict{String,DataFrame}, forecast_origin::Date,
    cfg::ClimConfig; period::Int=SEASON_PERIOD,
)
    cur_season = season_year(forecast_origin)
    all_dates = Date[]
    all_dev = Float64[]
    for loc in LOCATIONS
        sub = hist_by_loc[loc]
        sub = sub[sub.origin_date .< forecast_origin, :]
        isempty(sub) && continue
        values = to_scale.(sub.wili, TRANSFORM)
        weights = [cfg.decay^(cur_season - season_year(d)) for d in sub.origin_date]
        level = trimmed_weighted_mean(values, weights, cfg.trim_frac)
        append!(all_dates, sub.origin_date)
        append!(all_dev, values .- level)
    end
    isempty(all_dates) && return zeros(period)
    raw = bin_weighted(all_dates, all_dev, cur_season, cfg; period=period)
    return smooth_curve(raw, cfg)
end

"""
    climatology_for(loc, loc_hist, forecast_origin, cfg, pooled_dev)
        -> Vector{Float64}

Per-location climatology, optionally blended with the pooled shape
(`pooled_dev`, from `build_pooled_deviation`, computed once per split
and passed in). `cfg.blend == 0` returns the per-location curve
unchanged (the original design); otherwise the pooled deviation is
re-levelled to this location's own trimmed-weighted-mean level and
linearly blended in.
"""
function climatology_for(
    loc_hist::DataFrame, forecast_origin::Date, cfg::ClimConfig,
    pooled_dev::Vector{Float64},
)
    per_loc = build_climatology(loc_hist, forecast_origin, cfg)
    cfg.blend <= 0.0 && return per_loc
    sub = loc_hist[loc_hist.origin_date .< forecast_origin, :]
    if isempty(sub)
        level = 0.0
    else
        cur_season = season_year(forecast_origin)
        values = to_scale.(sub.wili, TRANSFORM)
        weights = [cfg.decay^(cur_season - season_year(d)) for d in sub.origin_date]
        level = trimmed_weighted_mean(values, weights, cfg.trim_frac)
    end
    pooled_abs = level .+ pooled_dev
    return (1 - cfg.blend) .* per_loc .+ cfg.blend .* pooled_abs
end

# ---------------------------------------------------------------------
# AR(6) + climatology fit and forward simulation (identical to
# season/generate.jl)
# ---------------------------------------------------------------------

"""
    fit_ar_clim(y, woy, order, clim) -> (coef, resid_sd)

OLS fit of an AR(`order`) model with intercept and one extra
regressor, the climatology value `clim[mod1(woy[t], period)]` at each
response time `t`. Identical to season/generate.jl.
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

Identical to season/generate.jl's function of the same name.
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
    build_forecast_table(seasons, cfg, profile, versions_full, hist_by_loc;
                          model_id) -> DataFrame

Fit and forecast the AR(6)+robust-climatology+backfill model for every
cross-validation split of every season in `seasons`, using climatology
configuration `cfg`. Training discipline as season/generate.jl:
`build_model_data` caps each split at its own forecast origin and
`window_weeks=104` further caps AR history to 2 seasons; the
climatology terms separately draw on the FULL historical series but
are still capped, inside `build_climatology`/`build_pooled_deviation`,
at each split's own forecast origin. `season in TEST_SEASONS` (3, 4, 5)
is fetched with `allow_test_season=true` -- callers must only pass
those when writing the final hub submission for an already-selected
`cfg`, never while tuning.
"""
function build_forecast_table(
    seasons, cfg::ClimConfig, profile, versions_full::DataFrame,
    hist_by_loc::Dict{String,DataFrame}; model_id::String,
)
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
            pooled_dev = cfg.blend > 0.0 ?
                build_pooled_deviation(hist_by_loc, origin, cfg) :
                Float64[]
            for (li, loc) in enumerate(LOCATIONS)
                y = Float64.(data.Y[:, li])
                clim = climatology_for(
                    hist_by_loc[loc], origin, cfg, pooled_dev,
                )
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

# ---------------------------------------------------------------------
# Scoring helpers (identical pattern to seasoncombo/generate.jl)
# ---------------------------------------------------------------------

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
# Staged sweep grids
# ---------------------------------------------------------------------

const DECAYS = (
    1.0, 0.995, 0.99, 0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,
)
const TRIM_FRACS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.49, 0.499)
const MOVAVG_WINDOWS = (1, 3, 5, 7, 9, 11)
const SPLINE_LAMBDAS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0)
const BLENDS = (0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)

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

    hist_full = load_series("flu_data_hhs")
    hist_by_loc = Dict(
        loc => hist_full[hist_full.location .== loc, [:origin_date, :wili]]
        for loc in LOCATIONS
    )

    truth = load_oracle(HUB_PATH)

    println("robust climatology sweep -- simple-round")
    println("validation seasons $(VALIDATION_SEASONS) only, " *
            "natural-scale WIS")
    println()
    println("reference points:")
    println("  season/generate.jl climatology-backfill (per-location, " *
            "equal-weight median, movavg5) = 0.3004 (sd 0.3890)")
    println("  seasoncombo core (pooled shape, backfill, no clim term) " *
            "= 0.2781 (sd 0.3341)")
    println("  seasondrift best (recency-weighted pooled shape, " *
            "decay=0.4)                    = 0.2602 (sd 0.3071)")
    println()

    score_cfg(cfg, label) = begin
        fc = build_forecast_table(
            VALIDATION_SEASONS, cfg, profile, versions_full, hist_by_loc;
            model_id=label,
        )
        summ = score_one(fc, truth)
        mw = round(summ.mean_wis; digits=4)
        sw = round(summ.sd_wis; digits=4)
        println("  $(label) -> mean_wis=$(mw) sd_wis=$(sw)")
        return summ.mean_wis
    end

    println("=== stage 0: near-median sanity check " *
            "(decay=1.0, trim=0.45, movavg5, blend=0) ===")
    base_cfg = ClimConfig(1.0, 0.45, :movavg, 5, 50.0, 0.0)
    base_wis = score_cfg(base_cfg, "sanity(trim=0.45)")
    println("(trim=0.45 is close to, but not bit-identical to, the " *
            "original plain per-week MEDIAN; a median is a trimmed " *
            "mean's trim_frac -> 0.5 limit -- this just confirms the " *
            "generalised estimator lands in the same neighbourhood as " *
            "season/generate.jl's 0.3004 before any tuning.)")
    println()

    println("=== stage 1: recency-weighting sweep " *
            "(trim=0.45, movavg5, blend=0) ===")
    best_decay = 1.0
    best_decay_wis = base_wis
    for d in DECAYS
        d == 1.0 && continue
        w = score_cfg(ClimConfig(d, 0.45, :movavg, 5, 50.0, 0.0), "decay=$d")
        if w < best_decay_wis
            best_decay_wis = w
            best_decay = d
        end
    end
    println("best: decay=$(best_decay) mean_wis=$(round(best_decay_wis; digits=4))")
    println()

    println("=== stage 2: robust trimming sweep " *
            "(decay=$(best_decay), movavg5, blend=0) ===")
    best_trim = 0.45
    best_trim_wis = best_decay_wis
    for tf in TRIM_FRACS
        tf == 0.45 && continue
        w = score_cfg(
            ClimConfig(best_decay, tf, :movavg, 5, 50.0, 0.0), "trim=$tf",
        )
        if w < best_trim_wis
            best_trim_wis = w
            best_trim = tf
        end
    end
    println("best: trim=$(best_trim) mean_wis=$(round(best_trim_wis; digits=4))")
    println()

    println("=== stage 3: smoothing sweep " *
            "(decay=$(best_decay), trim=$(best_trim), blend=0) ===")
    best_smooth_method = :movavg
    best_smooth_param = 5.0
    best_smooth_wis = best_trim_wis
    for win in MOVAVG_WINDOWS
        win == 5 && continue
        w = score_cfg(
            ClimConfig(best_decay, best_trim, :movavg, win, 50.0, 0.0),
            "movavg(w=$win)",
        )
        if w < best_smooth_wis
            best_smooth_wis = w
            best_smooth_method = :movavg
            best_smooth_param = Float64(win)
        end
    end
    for lam in SPLINE_LAMBDAS
        w = score_cfg(
            ClimConfig(best_decay, best_trim, :spline, 5, lam, 0.0),
            "spline(lambda=$lam)",
        )
        if w < best_smooth_wis
            best_smooth_wis = w
            best_smooth_method = :spline
            best_smooth_param = lam
        end
    end
    println("best: $(best_smooth_method)($(best_smooth_param)) " *
            "mean_wis=$(round(best_smooth_wis; digits=4))")
    println()

    win = best_smooth_method == :movavg ? Int(best_smooth_param) : 5
    lam = best_smooth_method == :spline ? best_smooth_param : 50.0

    println("=== stage 4: per-location vs pooled blend sweep " *
            "(decay=$(best_decay), trim=$(best_trim), " *
            "$(best_smooth_method)) ===")
    best_blend = 0.0
    best_blend_wis = best_smooth_wis
    for b in BLENDS
        b == 0.0 && continue
        w = score_cfg(
            ClimConfig(best_decay, best_trim, best_smooth_method, win, lam, b),
            "blend=$b",
        )
        if w < best_blend_wis
            best_blend_wis = w
            best_blend = b
        end
    end
    println("best: blend=$(best_blend) mean_wis=$(round(best_blend_wis; digits=4))")
    println()

    winner_cfg = ClimConfig(
        best_decay, best_trim, best_smooth_method, win, lam, best_blend,
    )
    println("=== overall winner ===")
    println("decay=$(best_decay) trim=$(best_trim) " *
            "smooth=$(best_smooth_method)($(best_smooth_method == :movavg ? win : lam)) " *
            "blend=$(best_blend)")
    println("mean_wis=$(round(best_blend_wis; digits=4)) " *
            "vs season/generate.jl reference (0.3004): " *
            "$(round(0.3004 - best_blend_wis; digits=4)) " *
            "($(round(100 * (0.3004 - best_blend_wis) / 0.3004; digits=2))%)")
    println()

    fc_winner = build_forecast_table(
        VALIDATION_SEASONS, winner_cfg, profile, versions_full, hist_by_loc;
        model_id=MODEL_ID,
    )
    scored = score_forecasts(fc_winner, truth; scale=:natural)

    println("-- breakdown by location (winner, mean WIS) --")
    by_loc = sort(
        combine(groupby(scored, :location), :wis => mean => :mean_wis),
        :mean_wis,
    )
    for row in eachrow(by_loc)
        println("  $(row.location)  $(round(row.mean_wis; digits=4))")
    end
    println()

    println("-- breakdown by horizon (winner, mean WIS) --")
    by_h = sort(
        combine(groupby(scored, :horizon), :wis => mean => :mean_wis),
        :horizon,
    )
    for row in eachrow(by_h)
        println("  h=$(row.horizon): $(round(row.mean_wis; digits=4))")
    end
    println()

    println("-- breakdown by validation season (winner, mean WIS) --")
    scored.season_year = season_year.(scored.origin_date)
    by_season = sort(
        combine(groupby(scored, :season_year), :wis => mean => :mean_wis),
        :season_year,
    )
    for row in eachrow(by_season)
        println("  season $(row.season_year): $(round(row.mean_wis; digits=4))")
    end
    println()

    dt = round(time() - t0; digits=2)
    println("validation sweep runtime: $(dt)s")

    if hub_path !== nothing
        t1 = time()
        full = build_forecast_table(
            (1, 2, 3, 4, 5), winner_cfg, profile, versions_full,
            hist_by_loc; model_id=MODEL_ID,
        )
        write_submission(full, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="seabbs_bot", model_abbr="robustclim", designated=true,
        )
        dt2 = round(time() - t1; digits=2)
        n_origins = length(unique(full.origin_date))
        println("wrote full 5-season submission ($(nrow(full)) rows, " *
                "$(n_origins) origin dates) + metadata to $(hub_path) " *
                "in $(dt2)s")
    end
    return winner_cfg
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
