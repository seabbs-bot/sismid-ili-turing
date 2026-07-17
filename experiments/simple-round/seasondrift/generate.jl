#!/usr/bin/env julia
# seasonal DRIFT / adaptation sweep -- simple-round, iterating on the
# round-1 winner (experiments/simple-round/seasoncombo/generate.jl
# combo 1 "core": pooled-seasonal + per-location AR(6) + backfill,
# mean_wis=0.2781 on validation seasons 1, 2).
#
# `core`'s pooled seasonal shape is a single FIXED climatology, shared
# across all 11 locations and all 13 training seasons alike.
# docs/eda/03-seasonality.md shows two things that fixed shape cannot
# capture: peak TIMING drifts across seasons (national peak `woy` SD
# ~5.7 weeks, per-location 5.2-7.9 weeks) and peak AMPLITUDE varies
# season-to-season (per-location amplitude CV 0.28-0.69). This sweep
# asks whether letting the shape ADAPT -- its phase and/or amplitude
# estimated from THIS season's trajectory so far, or the pooled shape
# itself drifting slowly across training seasons by weighting recent
# ones more -- beats the fixed-shape 0.2781.
#
# Three kinds of drift, all layered on top of the unchanged `core`
# design (same pooled-profile construction, same backfill correction,
# same per-location AR(6), same everything else):
#
#   1. profile decay: `build_seasonal_profile` reweighted so recent
#      training seasons count more (`decay < 1` downweights older
#      seasons geometrically); decay=1.0 exactly reproduces `core`'s
#      profile. Tests whether the pooled SHAPE itself should track
#      recent seasons more than distant ones.
#   2. within-season phase/amplitude adaptation: for each split, the
#      CURRENT season's already-observed rows (`data.season ==
#      maximum(data.season)`, i.e. this season's trajectory before the
#      forecast origin) are grid-searched against integer week shifts
#      of the pooled profile, with the OLS amplitude slope fit
#      (shrunk toward 1.0) at each candidate shift; the best-fitting
#      (phi, amp) is used to reconstruct the seasonal term for both
#      the current season's own residuals (so the AR(6) fit sees a
#      consistent deseasonalization right up to the forecast origin)
#      and the future horizons being forecast. Below `min_obs`
#      season-so-far observations, falls back to (phi=0, amp=1) --
#      early in a season there is too little signal to estimate either
#      safely. Tried both per-location (each location's own phase) and
#      POOLED (one shared phase fit by minimising total SSE across all
#      11 locations at once) -- docs/eda/03-seasonality.md's
#      cross-location peak-timing correlation (mean 0.37-0.80,
#      location-dependent) suggests a shared national phase might not
#      transfer to every region.
#   3. stacking whichever of (1) and (2) helps alone.
#
# LIGHT + ANALYTIC: CSV/DataFrames/Statistics/LinearAlgebra only, no
# Turing. Scored on VALIDATION SEASONS (1, 2) ONLY against the local
# hub clone's oracle (docs/contracts.md experimental integrity) -- this
# is a tuning sweep, not a submission driver.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl
# writes score.txt alongside this file; does not write a hub
# submission.

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

# Reference backfill design, identical to seasoncombo's `core` combo.
const BF_MODE = :additive
const BF_WINDOW = 8
const BF_STAT = :median

# ---------------------------------------------------------------------
# Pooled seasonal shape, with an optional recency-decay reweighting
# ---------------------------------------------------------------------

"""
    build_seasonal_profile(hist; transform, max_season_year, min_support,
                            smooth_window, decay=1.0) -> Dict{Int,Float64}

Pooled week-of-season climatology on the `transform` scale, identical
in construction to `seasoncombo/generate.jl`'s function of the same
name (own-location-mean-centred deviations, pooled across all
locations, circular moving average, re-centred to zero mean) with one
addition: each observation from training season `sy` is weighted
`decay^(max_season_year - sy)` before averaging within a
week-of-season bin, so `decay < 1` lets recent seasons dominate the
shape and `decay = 1.0` (the default) reproduces the plain,
equally-weighted `core` profile exactly.
"""
function build_seasonal_profile(
    hist::DataFrame; transform::Symbol, max_season_year::Int,
    min_support::Int, smooth_window::Int, decay::Float64=1.0,
)
    h = hist[season_year.(hist.origin_date) .<= max_season_year, :]
    x = to_scale.(h.wili, transform)
    locs = h.location
    woys = week_of_season.(h.origin_date)
    syears = season_year.(h.origin_date)

    levels = Dict{String,Float64}()
    for loc in unique(locs)
        levels[loc] = mean(x[locs .== loc])
    end
    dev = [x[i] - levels[locs[i]] for i in eachindex(x)]
    w = [decay^(max_season_year - sy) for sy in syears]

    Wmax = maximum(woys)
    sumw = zeros(Wmax)
    sumwv = zeros(Wmax)
    cnt = zeros(Int, Wmax)
    for i in eachindex(dev)
        b = woys[i]
        sumw[b] += w[i]
        sumwv[b] += w[i] * dev[i]
        cnt[b] += 1
    end
    means = [cnt[b] >= min_support ? sumwv[b] / sumw[b] : 0.0 for b in 1:Wmax]

    half = div(smooth_window - 1, 2)
    smoothed = similar(means)
    for wk in 1:Wmax
        idxs = [mod1(wk + off, Wmax) for off in (-half):half]
        smoothed[wk] = mean(means[idxs])
    end
    smoothed .-= mean(smoothed)

    return Dict(wk => smoothed[wk] for wk in 1:Wmax)
end

# ---------------------------------------------------------------------
# Within-season phase/amplitude adaptation
# ---------------------------------------------------------------------

"""
    estimate_drift(Y, woy, season, profile, level, W; phi_grid, min_obs,
                   shrink_amp, do_phase, do_amp, pooled_phase=false)
        -> (phis::Vector{Int}, amps::Vector{Float64})

Estimate this split's within-season phase shift and/or amplitude scale
from the CURRENT season's already-observed rows
(`season .== maximum(season)`), relative to the pooled `profile`
(zero-mean, `level`-independent by construction). For each location
(or once, pooled across all locations, if `pooled_phase`), grid-
searches `phi_grid` (integer week shifts) for the shift that, jointly
with the no-intercept OLS amplitude slope `b` at that shift, minimises
the sum of squared deviations of `Y[t,l] - level[l]` from
`b * profile[mod1(woy[t] - phi, W)]` over the season-so-far rows;
`b` is partially shrunk toward 1.0 by `shrink_amp` (as
`seasoncombo`'s cross-season `build_amplitude_scales` does) and
clamped to `[0, 3]` to guard against a wild slope from very few points.
`do_phase`/`do_amp` gate which piece is actually estimated -- the other
stays at its neutral value (`phi=0`/`amp=1`). Below `min_obs`
season-so-far observations for a location (or overall, when
`pooled_phase`), falls back to `(phi=0, amp=1)`: early in a season
there are too few points to estimate either safely, so this collapses
back to the unadapted `core` shape.
"""
function estimate_drift(
    Y::AbstractMatrix, woy::Vector{Int}, season::Vector{Int},
    profile::Dict{Int,Float64}, level::Vector{Float64}, W::Int;
    phi_grid, min_obs::Int, shrink_amp::Float64, do_phase::Bool,
    do_amp::Bool, pooled_phase::Bool=false,
)
    T, L = size(Y)
    cur = maximum(season)
    idx = findall(==(cur), season)
    phis = zeros(Int, L)
    amps = ones(L)
    (!do_phase && !do_amp) && return phis, amps
    length(idx) < min_obs && return phis, amps

    phi_candidates = do_phase ? collect(phi_grid) : [0]
    slope(dev, s) = (denom = sum(abs2, s); denom > 1e-8 ?
        sum(dev .* s) / denom : 0.0)
    shrink(b) = clamp(1.0 + shrink_amp * (b - 1.0), 0.0, 3.0)

    if pooled_phase
        devs = [[Float64(Y[t, l]) - level[l] for t in idx] for l in 1:L]
        best_phi, best_sse = 0, Inf
        for phi in phi_candidates
            sse = 0.0
            for l in 1:L
                s = [get(profile, mod1(woy[t] - phi, W), 0.0) for t in idx]
                b_use = do_amp ? shrink(slope(devs[l], s)) : 1.0
                sse += sum(abs2, devs[l] .- b_use .* s)
            end
            sse < best_sse && ((best_phi, best_sse) = (phi, sse))
        end
        for l in 1:L
            s = [get(profile, mod1(woy[t] - best_phi, W), 0.0) for t in idx]
            phis[l] = best_phi
            amps[l] = do_amp ? shrink(slope(devs[l], s)) : 1.0
        end
        return phis, amps
    end

    for l in 1:L
        dev = [Float64(Y[t, l]) - level[l] for t in idx]
        woy_l = [woy[t] for t in idx]
        best_phi, best_amp, best_sse = 0, 1.0, Inf
        for phi in phi_candidates
            s = [get(profile, mod1(w - phi, W), 0.0) for w in woy_l]
            b_use = do_amp ? shrink(slope(dev, s)) : 1.0
            sse = sum(abs2, dev .- b_use .* s)
            sse < best_sse && ((best_phi, best_amp, best_sse) =
                (phi, b_use, sse))
        end
        phis[l] = best_phi
        amps[l] = best_amp
    end
    return phis, amps
end

"""
    deseasonalize_adaptive(Y, woy, season, profile, amp, phi, W)
        -> (R, level)

Like `seasoncombo`'s `deseasonalize`, but the CURRENT season's rows
(`season .== maximum(season)`) use the per-location adapted `(phi,
amp)` from `estimate_drift`, while every earlier (complete) season's
rows use the plain, unadapted profile (`phi=0, amp=1`) -- past seasons
already anchor the pooled shape itself (via `build_seasonal_profile`)
and should not be retroactively reshaped by an estimate drawn from a
DIFFERENT (the current, partial) season's trajectory. This keeps the
AR(6) fit's residual history built on the same footing as `core`,
while the current season's most recent residuals (and the future
horizons reconstructed from them) reflect the adaptation.
"""
function deseasonalize_adaptive(
    Y::AbstractMatrix, woy::Vector{Int}, season::Vector{Int},
    profile::Dict{Int,Float64}, amp::Vector{Float64}, phi::Vector{Int},
    W::Int,
)
    T, L = size(Y)
    cur = maximum(season)
    level = zeros(L)
    for l in 1:L
        level[l] = mean(Float64.(Y[:, l]))
    end
    R = Matrix{Float64}(undef, T, L)
    for l in 1:L, t in 1:T
        if season[t] == cur
            s = get(profile, mod1(woy[t] - phi[l], W), 0.0)
            R[t, l] = Y[t, l] - level[l] - amp[l] * s
        else
            s = get(profile, woy[t], 0.0)
            R[t, l] = Y[t, l] - level[l] - s
        end
    end
    return R, level
end

# ---------------------------------------------------------------------
# Backfill correction (identical to seasoncombo's `core` combo)
# ---------------------------------------------------------------------

function build_revision_profile(
    versions::DataFrame; transform::Symbol, max_delay::Int,
    min_support::Int, mode::Symbol, stat::Symbol,
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
            val = mode == :additive ? settled - vintage : settled / vintage
            key = (loc, delay)
            push!(get!(raw, key, Float64[]), val)
        end
    end
    profile = Dict{Tuple{String,Int},Float64}()
    for (key, vals) in raw
        length(vals) < min_support && continue
        profile[key] = stat == :median ? median(vals) : mean(vals)
    end
    return profile
end

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
# Plain per-location AR(6) (identical to seasoncombo/nfidd-ar6)
# ---------------------------------------------------------------------

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
    build_forecast_table(seasons, versions_full, profile; kwargs...)
        -> DataFrame

Fit and forecast pooled-seasonal + AR(6) + backfill for every
cross-validation split of every season in `seasons`, exactly as
seasoncombo's `core` combo when `do_phase=do_amp=false` (the neutral
`estimate_drift` fallback then makes `deseasonalize_adaptive`
identical to plain `deseasonalize`) -- this is the sanity check that
this driver reproduces 0.2781 before any drift is switched on.
"""
function build_forecast_table(
    seasons, versions_full, profile::Dict{Int,Float64};
    backfill_profile::Dict{Tuple{String,Int},Float64},
    do_phase::Bool=false, do_amp::Bool=false, phi_grid=(-6:6),
    min_obs::Int=6, shrink_amp::Float64=1.0, pooled_phase::Bool=false,
    model_id::String,
)
    rng = MersenneTwister(SEED)
    rows = DataFrame(
        model_id=String[], location=String[], origin_date=Date[],
        horizon=Int[], target_end_date=Date[], target=String[],
        output_type=String[], output_type_id=Float64[], value=Float64[],
    )
    W = maximum(keys(profile))
    for season in seasons
        splits = training_splits(season)
        for split in splits
            data = build_model_data(
                split; Dmax=DMAX, transform=TRANSFORM,
                window_weeks=WINDOW_WEEKS, versions=versions_full,
            )
            apply_backfill_correction!(
                data, backfill_profile; mode=BF_MODE,
                delay_cutoff=BF_WINDOW,
            )
            level0 = [mean(Float64.(data.Y[:, l])) for l in 1:data.L]
            phis, amps = estimate_drift(
                data.Y, data.woy, data.season, profile, level0, W;
                phi_grid=phi_grid, min_obs=min_obs, shrink_amp=shrink_amp,
                do_phase=do_phase, do_amp=do_amp, pooled_phase=pooled_phase,
            )
            R, level = deseasonalize_adaptive(
                data.Y, data.woy, data.season, profile, amps, phis, W,
            )
            origin = data.origin_date
            for (li, loc) in enumerate(LOCATIONS)
                y = R[:, li]
                coef, resid_sd = fit_ar(y, AR_ORDER)
                paths = simulate_paths(
                    y, coef, resid_sd, AR_ORDER, HORIZONS, NPATHS; rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    w = mod1(week_of_season(target_end) - phis[li], W)
                    s = get(profile, w, 0.0)
                    vals = paths[h] .+ level[li] .+ amps[li] * s
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

const DECAYS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
const PHASE_SETTINGS = (
    (min_obs=6, phi_grid=-4:4), (min_obs=6, phi_grid=-8:8),
    (min_obs=10, phi_grid=-8:8),
)
const AMP_SETTINGS = (
    (min_obs=6, shrink=0.1), (min_obs=6, shrink=0.25),
    (min_obs=6, shrink=0.5), (min_obs=6, shrink=1.0),
    (min_obs=10, shrink=1.0),
)

function main()
    t0 = time()
    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")
    truth = load_oracle(HUB_PATH)

    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= MAX_TRAIN_SEASON_YEAR, :,
    ]
    backfill_profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_WINDOW,
        min_support=5, mode=BF_MODE, stat=BF_STAT,
    )

    profile0 = build_seasonal_profile(
        hist; transform=TRANSFORM, max_season_year=MAX_TRAIN_SEASON_YEAR,
        min_support=MIN_SUPPORT, smooth_window=SMOOTH_WINDOW, decay=1.0,
    )

    # --- sanity check: reproduce seasoncombo's `core` (0.2781) ---
    core = build_forecast_table(
        VALIDATION_ONLY, versions_full, profile0;
        backfill_profile=backfill_profile, model_id="core-reproduced",
    )
    core_summ = score_one(core, truth)
    println("sanity core reproduction: " *
            "mean_wis=$(round(core_summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(core_summ.sd_wis; digits=4)) " *
            "(target: 0.2781, seasoncombo/score.txt)")

    # --- 1: profile decay only (recency-weighted pooled shape) ---
    decay_results = NamedTuple[]
    for decay in DECAYS
        profile_d = decay == 1.0 ? profile0 : build_seasonal_profile(
            hist; transform=TRANSFORM,
            max_season_year=MAX_TRAIN_SEASON_YEAR, min_support=MIN_SUPPORT,
            smooth_window=SMOOTH_WINDOW, decay=decay,
        )
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile_d;
            backfill_profile=backfill_profile, model_id="decay-$decay",
        )
        summ = score_one(fc, truth)
        push!(decay_results, (
            decay=decay, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("decay=$decay -> mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(decay_results; by=r -> r.mean_wis)
    decay_best = decay_results[1]

    # --- 2a: within-season PHASE adaptation only, per-location ---
    phase_results = NamedTuple[]
    for cfg in PHASE_SETTINGS, pooled in (false, true)
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile0;
            backfill_profile=backfill_profile, do_phase=true,
            phi_grid=cfg.phi_grid, min_obs=cfg.min_obs,
            pooled_phase=pooled, model_id="phase",
        )
        summ = score_one(fc, truth)
        push!(phase_results, (
            min_obs=cfg.min_obs, phi_grid=cfg.phi_grid, pooled=pooled,
            mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("phase min_obs=$(cfg.min_obs) phi_grid=$(cfg.phi_grid) " *
                "pooled=$pooled -> " *
                "mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(phase_results; by=r -> r.mean_wis)
    phase_best = phase_results[1]

    # --- 2b: within-season AMPLITUDE adaptation only ---
    amp_results = NamedTuple[]
    for cfg in AMP_SETTINGS
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile0;
            backfill_profile=backfill_profile, do_amp=true,
            min_obs=cfg.min_obs, shrink_amp=cfg.shrink, model_id="amp",
        )
        summ = score_one(fc, truth)
        push!(amp_results, (
            min_obs=cfg.min_obs, shrink=cfg.shrink,
            mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("amp min_obs=$(cfg.min_obs) shrink=$(cfg.shrink) -> " *
                "mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(amp_results; by=r -> r.mean_wis)
    amp_best = amp_results[1]

    # --- 2c: phase + amplitude combined, best settings of each ---
    combined = build_forecast_table(
        VALIDATION_ONLY, versions_full, profile0;
        backfill_profile=backfill_profile, do_phase=true, do_amp=true,
        phi_grid=phase_best.phi_grid, min_obs=min(
            phase_best.min_obs, amp_best.min_obs,
        ), shrink_amp=amp_best.shrink, pooled_phase=phase_best.pooled,
        model_id="phase-amp",
    )
    combined_summ = score_one(combined, truth)
    println("phase+amp combined -> " *
            "mean_wis=$(round(combined_summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(combined_summ.sd_wis; digits=4))")

    # --- 3: best of (1) stacked on best of (2) ---
    profile_best = decay_best.decay == 1.0 ? profile0 : build_seasonal_profile(
        hist; transform=TRANSFORM, max_season_year=MAX_TRAIN_SEASON_YEAR,
        min_support=MIN_SUPPORT, smooth_window=SMOOTH_WINDOW,
        decay=decay_best.decay,
    )
    decay_best_fc = build_forecast_table(
        VALIDATION_ONLY, versions_full, profile_best;
        backfill_profile=backfill_profile, model_id="decay-best",
    )

    # --- breakdown: fixed-shape core vs the profile-decay winner, by
    #     location and by validation season -- checks whether the
    #     decay closes the harder-2016/17 gap flagged in
    #     seasoncombo/score.txt and docs/eda/03-seasonality.md, not
    #     just the mean. ---
    scored_core = score_forecasts(core, truth; scale=:natural)
    scored_decay = score_forecasts(decay_best_fc, truth; scale=:natural)
    scored_core.season_yr = season_year.(scored_core.origin_date)
    scored_decay.season_yr = season_year.(scored_decay.origin_date)

    loc_a = combine(groupby(scored_core, :location), :wis => mean => :core)
    loc_b = combine(groupby(scored_decay, :location), :wis => mean => :decay)
    loc_breakdown = innerjoin(loc_a, loc_b, on=:location)
    loc_breakdown.improvement = loc_breakdown.core .- loc_breakdown.decay
    sort!(loc_breakdown, :improvement; rev=true)

    yr_a = combine(groupby(scored_core, :season_yr), :wis => mean => :core)
    yr_b = combine(groupby(scored_decay, :season_yr), :wis => mean => :decay)
    yr_breakdown = innerjoin(yr_a, yr_b, on=:season_yr)
    yr_breakdown.improvement = yr_breakdown.core .- yr_breakdown.decay
    sort!(yr_breakdown, :season_yr)

    stacked = build_forecast_table(
        VALIDATION_ONLY, versions_full, profile_best;
        backfill_profile=backfill_profile, do_phase=true, do_amp=true,
        phi_grid=phase_best.phi_grid, min_obs=min(
            phase_best.min_obs, amp_best.min_obs,
        ), shrink_amp=amp_best.shrink, pooled_phase=phase_best.pooled,
        model_id="stacked",
    )
    stacked_summ = score_one(stacked, truth)
    println("stacked (decay=$(decay_best.decay) + phase+amp) -> " *
            "mean_wis=$(round(stacked_summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(stacked_summ.sd_wis; digits=4))")

    candidates = [
        (name="core (fixed shape, no drift)", mean_wis=core_summ.mean_wis,
         sd_wis=core_summ.sd_wis, detail="reference, from seasoncombo"),
        (name="profile decay only", mean_wis=decay_best.mean_wis,
         sd_wis=decay_best.sd_wis, detail="decay=$(decay_best.decay)"),
        (name="phase only", mean_wis=phase_best.mean_wis,
         sd_wis=phase_best.sd_wis,
         detail="min_obs=$(phase_best.min_obs) " *
                "phi_grid=$(phase_best.phi_grid) pooled=$(phase_best.pooled)"),
        (name="amplitude only", mean_wis=amp_best.mean_wis,
         sd_wis=amp_best.sd_wis,
         detail="min_obs=$(amp_best.min_obs) shrink=$(amp_best.shrink)"),
        (name="phase+amplitude", mean_wis=combined_summ.mean_wis,
         sd_wis=combined_summ.sd_wis, detail="best settings of each"),
        (name="stacked (decay+phase+amp)", mean_wis=stacked_summ.mean_wis,
         sd_wis=stacked_summ.sd_wis, detail="all three together"),
    ]
    sort!(candidates; by=r -> r.mean_wis)
    winner = candidates[1]

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "seasonal drift/adaptation sweep -- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "reference: seasoncombo core (fixed pooled shape, " *
                     "AR6, backfill) = 0.2781 (sd 0.3341)")
        println(io, "sanity reproduction here: " *
                     "mean_wis=$(round(core_summ.mean_wis; digits=4)) " *
                     "sd_wis=$(round(core_summ.sd_wis; digits=4))")
        println(io)
        println(io, "=== 1: profile decay (recency-weighted pooled " *
                     "shape) sweep ===")
        for r in decay_results
            println(io, "  decay=$(r.decay) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: decay=$(decay_best.decay) " *
                     "mean_wis=$(round(decay_best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(decay_best.sd_wis; digits=4))")
        println(io)
        println(io, "-- breakdown by location (core vs decay=" *
                     "$(decay_best.decay)) --")
        for r in eachrow(loc_breakdown)
            println(io, "  $(rpad(r.location, 15)) " *
                         "core=$(round(r.core; digits=4)) " *
                         "decay=$(round(r.decay; digits=4)) " *
                         "improvement=$(round(r.improvement; digits=4))")
        end
        println(io)
        println(io, "-- breakdown by validation season (core vs decay=" *
                     "$(decay_best.decay)) --")
        for r in eachrow(yr_breakdown)
            println(io, "  season_year=$(r.season_yr) " *
                         "core=$(round(r.core; digits=4)) " *
                         "decay=$(round(r.decay; digits=4)) " *
                         "improvement=$(round(r.improvement; digits=4))")
        end
        println(io)
        println(io, "=== 2a: within-season phase adaptation sweep ===")
        for r in phase_results
            println(io, "  min_obs=$(r.min_obs) phi_grid=$(r.phi_grid) " *
                         "pooled=$(r.pooled) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: min_obs=$(phase_best.min_obs) " *
                     "phi_grid=$(phase_best.phi_grid) " *
                     "pooled=$(phase_best.pooled) " *
                     "mean_wis=$(round(phase_best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(phase_best.sd_wis; digits=4))")
        println(io)
        println(io, "=== 2b: within-season amplitude adaptation sweep ===")
        for r in amp_results
            println(io, "  min_obs=$(r.min_obs) shrink=$(r.shrink) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: min_obs=$(amp_best.min_obs) " *
                     "shrink=$(amp_best.shrink) " *
                     "mean_wis=$(round(amp_best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(amp_best.sd_wis; digits=4))")
        println(io)
        println(io, "=== 2c: phase+amplitude combined (best settings " *
                     "of each) ===")
        println(io, "mean_wis=$(round(combined_summ.mean_wis; digits=4)) " *
                     "sd_wis=$(round(combined_summ.sd_wis; digits=4)) " *
                     "n_tasks=$(combined_summ.n_tasks)")
        println(io)
        println(io, "=== 3: stacked (best decay + best phase+amp) ===")
        println(io, "decay=$(decay_best.decay), " *
                     "phase phi_grid=$(phase_best.phi_grid) " *
                     "pooled=$(phase_best.pooled), " *
                     "amp shrink=$(amp_best.shrink)")
        println(io, "mean_wis=$(round(stacked_summ.mean_wis; digits=4)) " *
                     "sd_wis=$(round(stacked_summ.sd_wis; digits=4)) " *
                     "n_tasks=$(stacked_summ.n_tasks)")
        println(io)
        println(io, "=== overall comparison ===")
        for r in candidates
            println(io, rpad(r.name, 30) *
                         "mean_wis=$(rpad(round(r.mean_wis; digits=4), 8)) " *
                         "sd_wis=$(rpad(round(r.sd_wis; digits=4), 8)) " *
                         r.detail)
        end
        println(io)
        println(io, "=== winner: $(winner.name) ===")
        println(io, "mean_wis=$(round(winner.mean_wis; digits=4)) " *
                     "sd_wis=$(round(winner.sd_wis; digits=4)) " *
                     "($(winner.detail))")
        vs_ref = 0.2781 - winner.mean_wis
        vs_pct = 100 * vs_ref / 0.2781
        println(io, "vs fixed-shape core reference (0.2781): " *
                     "$(round(vs_ref; digits=4)) ($(round(vs_pct; digits=2))%)")
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
