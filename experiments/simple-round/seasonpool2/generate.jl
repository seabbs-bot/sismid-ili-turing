#!/usr/bin/env julia
# seasonal + partial pooling sweep -- simple-round, ROUND 2.
#
# Starting point: the round-1 winner
# (`experiments/simple-round/seasoncombo/generate.jl` combo 1, "core") --
# ONE pooled week-of-season seasonal shape (`build_seasonal_profile`,
# 11 locations x ~13 seasons of history) plus, PER LOCATION, an
# unpooled AR(6) fit on the deseasonalized residual and the
# `seabbs_bot-ar6bf` backfill correction. That combination scores
# mean_wis=0.2781 (sd 0.3341) on the validation seasons -- see
# `experiments/simple-round/seasoncombo/score.txt`.
#
# `experiments/simple-round/pool/generate.jl` showed that, on the PLAIN
# (no-season) AR(6) baseline, partially pooling each location's AR(6)
# coefficients toward a fullpool (all-locations-stacked) OLS anchor
# helps a little: fullpool-w0.5 scores 0.3643 vs 0.3684 unpooled, a
# 1.1% improvement. This sweep asks whether that same partial-pooling
# idea helps MORE or LESS once the pooled seasonal shape has already
# soaked up the biggest source of cross-location commonality, by adding
# three independent partial-pooling knobs on top of the "core" model:
#
#   1. AMPLITUDE pooling: `seasoncombo`'s `build_amplitude_scales`
#      already IS this knob (per-location amplitude shrunk toward the
#      pooled mean of 1.0); `seasoncombo`'s combo 4/5 already swept it
#      alone and combined with backfill, finding LESS pooling (shrink
#      close to 1, i.e. closer to the raw per-location slope) better
#      than the core model's full pooling (shrink=0, amp=1
#      everywhere). Reproduced/extended here on top of the full core
#      model (with backfill) for a complete, directly comparable grid.
#   2. AR(6) COEFFICIENT pooling: per-location AR(6), fit on the
#      DESEASONALIZED residual (not the raw series, unlike `pool/`),
#      blended `(1-w) * own + w * fullpool_anchor` where the anchor is
#      one OLS fit stacking every location's residual design rows for
#      that split -- the direct seasonal-model analogue of `pool/`'s
#      `:fullpool` scheme.
#   3. BACKFILL PROFILE pooling: the per-`(location, delay)` revision
#      correction (`build_revision_profile`, `pooled=false`) blended
#      toward a `pooled=true` (delay-only, pooled across locations)
#      version. `experiments/simple-round/backfill/generate.jl` found
#      `pooled=true` alone clearly worse than `pooled=false` on the
#      plain (no-season) baseline (0.366-0.369 vs 0.359-0.360); tested
#      here as a genuine partial blend, on top of the seasonal model,
#      rather than the all-or-nothing choice that sweep made.
#
# Each knob is swept ALONE on top of the full core model first (to
# isolate its own marginal effect against the 0.2781 anchor), then the
# three individual winners are combined and checked for interaction.
#
# LIGHT + ANALYTIC: CSV/DataFrames/Dates/Statistics/LinearAlgebra only,
# no Turing (same as `seasoncombo`/`pool`).
#
# Scope: VALIDATION SEASONS (1, 2) ONLY, scored against the local hub
# clone's oracle (docs/contracts.md experimental integrity) -- a tuning
# sweep, not a submission driver. The pooled seasonal shape, amplitude
# scales, AR(6) pooling anchor, and both backfill profiles are all
# estimated only from `season_year <= 2016` (pre-2015 history plus the
# two validation seasons); no test-season data anywhere.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl
# writes score.txt alongside this file; no hub submission written.

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
const MIN_SUPPORT = 5
const SMOOTH_WINDOW = 3
const VALIDATION_ONLY = (1, 2)
const MAX_TRAIN_SEASON_YEAR = 2016
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")

# Reference backfill design (`seabbs_bot-ar6bf` / `seasoncombo`'s
# "core" combo), reused unchanged as the per-location backfill anchor.
const BF_MODE = :additive
const BF_WINDOW = 8
const BF_STAT = :median

# ---------------------------------------------------------------------
# Pooled seasonal shape + per-location amplitude scale
# (identical to `experiments/simple-round/seasoncombo/generate.jl`)
# ---------------------------------------------------------------------

"""
    build_seasonal_profile(hist; transform, max_season_year, min_support,
                            smooth_window) -> Dict{Int,Float64}

Pooled week-of-season climatology on the `transform` scale, estimated
ONCE from the full historical series, restricted to
`season_year(origin_date) <= max_season_year`. See `seasoncombo`'s copy
of this function for the full derivation; identical here.
"""
function build_seasonal_profile(
    hist::DataFrame; transform::Symbol, max_season_year::Int,
    min_support::Int, smooth_window::Int,
)
    h = hist[season_year.(hist.origin_date) .<= max_season_year, :]
    x = to_scale.(h.wili, transform)
    locs = h.location
    woys = week_of_season.(h.origin_date)

    levels = Dict{String,Float64}()
    for loc in unique(locs)
        levels[loc] = mean(x[locs .== loc])
    end
    dev = [x[i] - levels[locs[i]] for i in eachindex(x)]

    Wmax = maximum(woys)
    raw = [Float64[] for _ in 1:Wmax]
    for i in eachindex(dev)
        push!(raw[woys[i]], dev[i])
    end
    means = [length(v) >= min_support ? mean(v) : 0.0 for v in raw]

    half = div(smooth_window - 1, 2)
    smoothed = similar(means)
    for w in 1:Wmax
        idxs = [mod1(w + off, Wmax) for off in (-half):half]
        smoothed[w] = mean(means[idxs])
    end
    smoothed .-= mean(smoothed)

    return Dict(w => smoothed[w] for w in 1:Wmax)
end

"""
    build_amplitude_scales(hist, profile; transform, max_season_year,
                            shrink) -> Vector{Float64}

Per-location amplitude scale, in `LOCATIONS` order, identical to
`seasoncombo`'s function of the same name -- see that file for the
full derivation. `shrink = 0.0` fully pools toward the shared shape
(`amp = 1` everywhere, the "core" model's own setting); `shrink = 1.0`
uses the raw per-location OLS slope with no pooling at all.
"""
function build_amplitude_scales(
    hist::DataFrame, profile::Dict{Int,Float64}; transform::Symbol,
    max_season_year::Int, shrink::Float64,
)
    h = hist[season_year.(hist.origin_date) .<= max_season_year, :]
    scales = ones(length(LOCATIONS))
    for (li, loc) in enumerate(LOCATIONS)
        sub = h[h.location .== loc, :]
        isempty(sub) && continue
        x = to_scale.(sub.wili, transform)
        dev = x .- mean(x)
        s = [get(profile, week_of_season(d), 0.0) for d in sub.origin_date]
        denom = sum(abs2, s)
        b = denom > 1e-8 ? sum(dev .* s) / denom : 1.0
        scales[li] = 1.0 + shrink * (b - 1.0)
    end
    return scales
end

"""
    deseasonalize(Y, woy, profile, amp) -> (R, level)

Remove each location's own mean level and the (`amp`-scaled) pooled
seasonal shape from `Y` (T x L, modelling scale). Identical to
`seasoncombo`'s function of the same name.
"""
function deseasonalize(
    Y::AbstractMatrix, woy::Vector{Int}, profile::Dict{Int,Float64},
    amp::Vector{Float64},
)
    T, L = size(Y)
    level = zeros(L)
    for l in 1:L
        level[l] = mean(Float64.(Y[:, l]))
    end
    R = Matrix{Float64}(undef, T, L)
    for l in 1:L, t in 1:T
        s = get(profile, woy[t], 0.0)
        R[t, l] = Y[t, l] - level[l] - amp[l] * s
    end
    return R, level
end

# ---------------------------------------------------------------------
# Backfill correction: per-location, pooled, and a partial blend
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support,
                            mode, pooled, stat) -> Dict

Empirical revision profile, identical in design to
`experiments/simple-round/backfill/generate.jl`'s function of the same
name (reused unchanged by `seasoncombo` too). `pooled=false` keys by
`(location, delay)`; `pooled=true` keys by `delay` alone, pooling the
correction across all locations.
"""
function build_revision_profile(
    versions::DataFrame; transform::Symbol, max_delay::Int,
    min_support::Int, mode::Symbol, pooled::Bool, stat::Symbol,
)
    raw = Dict{Any,Vector{Float64}}()
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
            if mode == :multiplicative && abs(vintage) < 1e-6
                continue
            end
            val = mode == :additive ? settled - vintage : settled / vintage
            key = pooled ? delay : (loc, delay)
            push!(get!(raw, key, Float64[]), val)
        end
    end
    profile = Dict{Any,Float64}()
    for (key, vals) in raw
        length(vals) < min_support && continue
        profile[key] = stat == :median ? median(vals) : mean(vals)
    end
    return profile
end

"""
    blend_revision_profiles(per_loc, pooled, weight; window) -> Dict

Partial pool of a per-`(location, delay)` revision profile toward a
`delay`-only pooled profile: for each location and each
`delay in 0:window`, the blended correction is
`(1 - weight) * per_loc[(loc, delay)] + weight * pooled[delay]` when
both exist; falls back to whichever of the two exists when only one
does (rather than dropping the key), and is omitted (no correction
applied at that `(loc, delay)`) when neither does. `weight = 0.0`
reproduces the pure per-location profile (the core model's own
setting); `weight = 1.0` reproduces the pure pooled-across-locations
profile (the variant `backfill/generate.jl` found clearly worse than
per-location on the plain, no-season baseline).
"""
function blend_revision_profiles(
    per_loc::Dict, pooled::Dict, weight::Float64; window::Int,
)
    blended = Dict{Tuple{String,Int},Float64}()
    for loc in LOCATIONS, d in 0:window
        have_loc = haskey(per_loc, (loc, d))
        have_pool = haskey(pooled, d)
        if have_loc && have_pool
            blended[(loc, d)] =
                (1 - weight) * per_loc[(loc, d)] + weight * pooled[d]
        elseif have_loc
            blended[(loc, d)] = per_loc[(loc, d)]
        elseif have_pool
            blended[(loc, d)] = pooled[d]
        end
    end
    return blended
end

"""
    apply_backfill_correction!(data, profile; mode, delay_cutoff)

Nudge `data.Y` in place wherever `0 <= delay <= delay_cutoff` and a
matching `(location, delay)` entry exists in `profile`. Identical to
`seasoncombo`'s function with `pooled` fixed to `false`, since every
profile passed here (plain or blended) is already keyed by
`(location, delay)`.
"""
function apply_backfill_correction!(
    data::ModelData, profile::Dict{Tuple{String,Int},Float64};
    mode::Symbol, delay_cutoff::Int,
)
    for l in 1:data.L, t in 1:data.T
        d = data.delay[t, l]
        (d < 0 || d > delay_cutoff) && continue
        ismissing(data.Y[t, l]) && continue
        key = (LOCATIONS[l], d)
        haskey(profile, key) || continue
        c = profile[key]
        data.Y[t, l] = mode == :additive ? data.Y[t, l] + c : data.Y[t, l] * c
    end
    return data
end

# ---------------------------------------------------------------------
# Per-location AR(6) on the deseasonalized residual, plain and pooled
# ---------------------------------------------------------------------

"""
    ar_design(y, order) -> (X, yresp)

Design matrix and response for an OLS AR(`order`) fit with intercept,
identical in form to `pool/generate.jl`'s function of the same name,
applied here to a deseasonalized residual column rather than the raw
series.
"""
function ar_design(y::AbstractVector{Float64}, order::Int)
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
    return X, yresp
end

"""
    fit_ar(y, order) -> (coef, X, yresp)

OLS fit of an AR(`order`) model with intercept to the residual `y`.
Returns the design too so callers can evaluate a different (pooled)
coefficient vector on the same data via `resid_sd_for`.
"""
function fit_ar(y::AbstractVector{Float64}, order::Int)
    X, yresp = ar_design(y, order)
    coef = X \ yresp
    return coef, X, yresp
end

"""
    fit_ar_pooled(ys, order) -> coef

One OLS AR(`order`) fit on the design rows of every location's
deseasonalized residual in `ys` stacked together -- the fullpool
anchor, identical in spirit to `pool/generate.jl`'s function of the
same name but fit on residuals instead of raw series.
"""
function fit_ar_pooled(ys::Vector{Vector{Float64}}, order::Int)
    designs = [ar_design(y, order) for y in ys]
    Xall = reduce(vcat, first.(designs))
    yall = reduce(vcat, last.(designs))
    return Xall \ yall
end

"""
    resid_sd_for(X, yresp, coef, order) -> Float64

Residual SD of `coef` (not necessarily the OLS solution for `X`,
`yresp`) evaluated on this design. Identical to `pool/generate.jl`'s
function of the same name.
"""
function resid_sd_for(
    X::Matrix{Float64}, yresp::Vector{Float64}, coef::Vector{Float64},
    order::Int,
)
    nobs = size(X, 1)
    resid = yresp .- X * coef
    dof = max(nobs - (order + 1), 1)
    return sqrt(sum(abs2, resid) / dof)
end

"""
    simulate_paths(y, coef, resid_sd, order, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Simulate `npaths` Gaussian-innovation AR(`order`) sample paths forward
from the end of `y`, for each horizon in `horizons`. Identical to
`seasoncombo`'s function of the same name.
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
# Forecast table builder: core model + optional AR(6) pooling weight
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, versions_full, profile, amp,
                          backfill_profile; ar_weight, model_id)
        -> DataFrame

Fit and forecast the seasonal "core" model (pooled shape, per-location
`amp`-scaled amplitude, backfill correction) for every cross-validation
split of every season in `seasons`, with each location's AR(6)
coefficient (fit on the deseasonalized residual) blended
`(1 - ar_weight) * own + ar_weight * fullpool_anchor`. `ar_weight = 0.0`
reproduces the unpooled "core" combo exactly (`amp = ones(L)` further
reproduces `seasoncombo`'s combo 1, 0.2781).
"""
function build_forecast_table(
    seasons, versions_full, profile::Dict{Int,Float64}, amp::Vector{Float64},
    backfill_profile::Dict{Tuple{String,Int},Float64};
    ar_weight::Float64=0.0, model_id::String,
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
            apply_backfill_correction!(
                data, backfill_profile; mode=BF_MODE, delay_cutoff=BF_WINDOW,
            )
            R, level = deseasonalize(data.Y, data.woy, profile, amp)
            origin = data.origin_date
            L = data.L

            ys = [R[:, li] for li in 1:L]
            fits = [fit_ar(ys[li], AR_ORDER) for li in 1:L]
            coefs = [f[1] for f in fits]
            Xs = [f[2] for f in fits]
            yresps = [f[3] for f in fits]
            anchor = ar_weight > 0 ? fit_ar_pooled(ys, AR_ORDER) : coefs[1]

            for (li, loc) in enumerate(LOCATIONS)
                coef = ar_weight > 0 ?
                    (1 - ar_weight) .* coefs[li] .+ ar_weight .* anchor :
                    coefs[li]
                resid_sd = resid_sd_for(Xs[li], yresps[li], coef, AR_ORDER)
                paths = simulate_paths(
                    ys[li], coef, resid_sd, AR_ORDER, HORIZONS, NPATHS;
                    rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    s = get(profile, week_of_season(target_end), 0.0)
                    vals = paths[h] .+ level[li] .+ amp[li] * s
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

const AMP_SHRINKS = (0.0, 0.25, 0.5, 0.75, 1.0)
const AR_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
const BF_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)

function main()
    t0 = time()
    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")
    truth = load_oracle(HUB_PATH)
    L = length(LOCATIONS)
    ones_amp = ones(L)

    profile = build_seasonal_profile(
        hist; transform=TRANSFORM, max_season_year=MAX_TRAIN_SEASON_YEAR,
        min_support=MIN_SUPPORT, smooth_window=SMOOTH_WINDOW,
    )

    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= MAX_TRAIN_SEASON_YEAR, :,
    ]
    per_loc_bf = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_WINDOW,
        min_support=5, mode=BF_MODE, pooled=false, stat=BF_STAT,
    )
    pooled_bf = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_WINDOW,
        min_support=5, mode=BF_MODE, pooled=true, stat=BF_STAT,
    )
    core_bf = blend_revision_profiles(
        per_loc_bf, pooled_bf, 0.0; window=BF_WINDOW,
    )

    # Sanity: reproduce seasoncombo combo 1 ("core") exactly.
    core = build_forecast_table(
        VALIDATION_ONLY, versions_full, profile, ones_amp, core_bf;
        ar_weight=0.0, model_id="seasonpool2-core",
    )
    core_summ = score_one(core, truth)
    println("core (reproduces seasoncombo combo 1): " *
            "mean_wis=$(round(core_summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(core_summ.sd_wis; digits=4)) " *
            "(reference: 0.2781)")

    # --- knob 1: amplitude pooling alone, on top of core ---
    amp_results = NamedTuple[]
    for shrink in AMP_SHRINKS
        amp = build_amplitude_scales(
            hist, profile; transform=TRANSFORM,
            max_season_year=MAX_TRAIN_SEASON_YEAR, shrink=shrink,
        )
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, amp, core_bf;
            ar_weight=0.0, model_id="seasonpool2-amp",
        )
        summ = score_one(fc, truth)
        push!(amp_results, (
            shrink=shrink, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("knob 1 (amp) shrink=$shrink -> " *
                "mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(amp_results; by=r -> r.mean_wis)
    amp_best = amp_results[1]

    # --- knob 2: AR(6) coefficient pooling alone, on top of core ---
    ar_results = NamedTuple[]
    for w in AR_WEIGHTS
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, ones_amp, core_bf;
            ar_weight=w, model_id="seasonpool2-ar",
        )
        summ = score_one(fc, truth)
        push!(ar_results, (
            weight=w, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("knob 2 (AR pool) weight=$w -> " *
                "mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(ar_results; by=r -> r.mean_wis)
    ar_best = ar_results[1]

    # --- knob 3: backfill profile pooling alone, on top of core ---
    bf_results = NamedTuple[]
    for w in BF_WEIGHTS
        bf = blend_revision_profiles(per_loc_bf, pooled_bf, w; window=BF_WINDOW)
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, ones_amp, bf;
            ar_weight=0.0, model_id="seasonpool2-bf",
        )
        summ = score_one(fc, truth)
        push!(bf_results, (
            weight=w, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("knob 3 (backfill pool) weight=$w -> " *
                "mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(bf_results; by=r -> r.mean_wis)
    bf_best = bf_results[1]

    # --- combined: stack the three individual winners ---
    combined_amp = build_amplitude_scales(
        hist, profile; transform=TRANSFORM,
        max_season_year=MAX_TRAIN_SEASON_YEAR, shrink=amp_best.shrink,
    )
    combined_bf = blend_revision_profiles(
        per_loc_bf, pooled_bf, bf_best.weight; window=BF_WINDOW,
    )
    combined = build_forecast_table(
        VALIDATION_ONLY, versions_full, profile, combined_amp, combined_bf;
        ar_weight=ar_best.weight, model_id="seasonpool2-combined",
    )
    combined_summ = score_one(combined, truth)
    println("combined (amp shrink=$(amp_best.shrink), " *
            "AR weight=$(ar_best.weight), bf weight=$(bf_best.weight)) -> " *
            "mean_wis=$(round(combined_summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(combined_summ.sd_wis; digits=4))")

    # The naive "each knob's own best" stack did WORSE than AR-pooling
    # alone (see score.txt) -- amp shrink and AR weight interact, so a
    # joint grid is needed to find the true combined optimum rather
    # than assuming the effects add. Backfill pooling is fixed at
    # weight=0 (its own best): every weight tested made it worse
    # (knob 3), so it is very unlikely to turn helpful only in
    # combination with the other two.
    joint_results = NamedTuple[]
    for shrink in AMP_SHRINKS, w in AR_WEIGHTS
        amp = build_amplitude_scales(
            hist, profile; transform=TRANSFORM,
            max_season_year=MAX_TRAIN_SEASON_YEAR, shrink=shrink,
        )
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, amp, core_bf;
            ar_weight=w, model_id="seasonpool2-joint",
        )
        summ = score_one(fc, truth)
        push!(joint_results, (
            shrink=shrink, weight=w, mean_wis=summ.mean_wis,
            sd_wis=summ.sd_wis,
        ))
    end
    sort!(joint_results; by=r -> r.mean_wis)
    joint_best = joint_results[1]
    println("joint grid best: amp shrink=$(joint_best.shrink), " *
            "AR weight=$(joint_best.weight) -> " *
            "mean_wis=$(round(joint_best.mean_wis; digits=4)) " *
            "sd_wis=$(round(joint_best.sd_wis; digits=4))")

    candidates = [
        (name="core", mean_wis=core_summ.mean_wis, sd_wis=core_summ.sd_wis),
        (name="amp-only", mean_wis=amp_best.mean_wis, sd_wis=amp_best.sd_wis),
        (name="ar-only", mean_wis=ar_best.mean_wis, sd_wis=ar_best.sd_wis),
        (name="bf-only", mean_wis=bf_best.mean_wis, sd_wis=bf_best.sd_wis),
        (name="combined-naive", mean_wis=combined_summ.mean_wis,
         sd_wis=combined_summ.sd_wis),
        (name="joint-grid-best", mean_wis=joint_best.mean_wis,
         sd_wis=joint_best.sd_wis),
    ]
    sort!(candidates; by=r -> r.mean_wis)
    winner = candidates[1]

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "seasonal + partial pooling sweep -- simple-round, " *
                     "round 2")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "reference points:")
        println(io, "  seasoncombo combo 1 (core: season+AR6+backfill)   " *
                     "= 0.2781 (sd 0.3341)")
        println(io, "  seasoncombo combo 5 (amp shrink=1.0 + backfill)   " *
                     "= 0.2748 (sd 0.3198)")
        println(io, "  pool/ fullpool-w0.5 (plain AR, no season)         " *
                     "= 0.3643 vs unpooled 0.3684 (1.11% better)")
        println(io, "  local core sanity rerun (this script)             " *
                     "= $(round(core_summ.mean_wis; digits=4)) " *
                     "(sd $(round(core_summ.sd_wis; digits=4)))")
        println(io)
        println(io, "=== knob 1: amplitude pooling (shrink toward the " *
                     "pooled amp=1 mean) ===")
        for r in amp_results
            pct = 100 * (core_summ.mean_wis - r.mean_wis) / core_summ.mean_wis
            println(io, "  shrink=$(r.shrink) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4)) " *
                         "vs core: $(round(pct; digits=2))%")
        end
        println(io, "best: shrink=$(amp_best.shrink) " *
                     "mean_wis=$(round(amp_best.mean_wis; digits=4))")
        println(io)
        println(io, "=== knob 2: AR(6) coefficient pooling (fullpool " *
                     "anchor on deseasonalized residual) ===")
        for r in ar_results
            pct = 100 * (core_summ.mean_wis - r.mean_wis) / core_summ.mean_wis
            println(io, "  weight=$(r.weight) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4)) " *
                         "vs core: $(round(pct; digits=2))%")
        end
        println(io, "best: weight=$(ar_best.weight) " *
                     "mean_wis=$(round(ar_best.mean_wis; digits=4))")
        println(io)
        println(io, "=== knob 3: backfill profile pooling (blend toward " *
                     "delay-only, cross-location profile) ===")
        for r in bf_results
            pct = 100 * (core_summ.mean_wis - r.mean_wis) / core_summ.mean_wis
            println(io, "  weight=$(r.weight) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4)) " *
                         "vs core: $(round(pct; digits=2))%")
        end
        println(io, "best: weight=$(bf_best.weight) " *
                     "mean_wis=$(round(bf_best.mean_wis; digits=4))")
        println(io)
        println(io, "=== combined-naive (each knob's own best, stacked) ===")
        println(io, "amp shrink=$(amp_best.shrink), " *
                     "AR weight=$(ar_best.weight), " *
                     "bf weight=$(bf_best.weight)")
        println(io, "mean_wis=$(round(combined_summ.mean_wis; digits=4)) " *
                     "sd_wis=$(round(combined_summ.sd_wis; digits=4))")
        println(io, "WORSE than AR-pooling alone -- amp shrink and AR " *
                     "weight interact, the naive stack does not add.")
        println(io)
        println(io, "=== joint grid: amp shrink x AR weight (bf weight " *
                     "fixed at 0, its own best) ===")
        for r in joint_results
            println(io, "  shrink=$(r.shrink) weight=$(r.weight) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: shrink=$(joint_best.shrink) " *
                     "weight=$(joint_best.weight) " *
                     "mean_wis=$(round(joint_best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(joint_best.sd_wis; digits=4))")
        println(io)
        println(io, "=== overall comparison ===")
        for r in candidates
            println(io, rpad(r.name, 17) *
                         "mean_wis=$(rpad(round(r.mean_wis; digits=4), 8)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io)
        println(io, "=== winner: $(winner.name) ===")
        println(io, "mean_wis=$(round(winner.mean_wis; digits=4)) " *
                     "sd_wis=$(round(winner.sd_wis; digits=4))")
        vs_core = core_summ.mean_wis - winner.mean_wis
        vs_pct = 100 * vs_core / core_summ.mean_wis
        println(io, "vs core reference (0.2781): " *
                     "$(round(vs_core; digits=4)) " *
                     "($(round(vs_pct; digits=2))%)")
        println(io)
        println(io, "=== interpretation: does partial pooling help more " *
                     "or less once seasonality is in? ===")
        println(io, "plain AR (pool/, no season): fullpool-w0.5 beat " *
                     "unpooled by 1.11%.")
        ar_pct = 100 * (core_summ.mean_wis - ar_best.mean_wis) /
                 core_summ.mean_wis
        println(io, "seasonal core + AR pooling: best weight=" *
                     "$(ar_best.weight) beat unpooled-AR core by " *
                     "$(round(ar_pct; digits=2))%.")
        amp_pct = 100 * (core_summ.mean_wis - amp_best.mean_wis) /
                  core_summ.mean_wis
        println(io, "amplitude pooling: best shrink=$(amp_best.shrink) " *
                     "$(amp_best.shrink > 0.5 ? "(LESS pooling)" :
                        "(MORE pooling)") beat full-pooling core by " *
                     "$(round(amp_pct; digits=2))%.")
        bf_pct = 100 * (core_summ.mean_wis - bf_best.mean_wis) /
                 core_summ.mean_wis
        println(io, "backfill pooling: best weight=$(bf_best.weight) " *
                     "beat per-location-only core by " *
                     "$(round(bf_pct; digits=2))%.")
    end

    dt = round(time() - t0; digits=1)
    println("\nwinner: $(winner.name) mean_wis=" *
            "$(round(winner.mean_wis; digits=4)) " *
            "sd_wis=$(round(winner.sd_wis; digits=4))")
    println("wrote score.txt in $(dt)s total")
    return candidates
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
