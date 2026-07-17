#!/usr/bin/env julia
# ensemble of SEASONAL+BACKFILL analytic models -- simple-round,
# ENSEMBLE family, seasonal-model iteration.
#
# `experiments/simple-round/ensemble/generate.jl` averaged four PLAIN
# members (ar6, ar6bf, climatology, ses) and only reached 0.290 mean
# WIS -- worse than the best single plain member, ar6bf's own 0.359 was
# already beaten by every seasonal model in this tree, so an ensemble
# of plain models was never going to threaten them. This experiment
# instead ensembles the STRONGEST seasonal+backfill variants already
# found by other single-lever sweeps in this same experiments/
# simple-round/ tree, each copied here largely unchanged from its own
# generate.jl (see the file header of each source for the full
# derivation; only repeated in brief below):
#
#   core     (experiments/simple-round/seasoncombo/generate.jl, combo
#             1): pooled additive week-of-season climatology
#             (`build_seasonal_profile`, season_year<=2016) removed
#             before an AR(6) fit, on top of the ar6bf additive/per-
#             location/median/window=8 backfill correction.
#             mean_wis=0.2781.
#   ampbf    (seasoncombo, bonus combo 5): `core` plus a per-location,
#             partially-shrunk (shrink=1.0, i.e. unshrunk) amplitude
#             scale on the pooled shape (`build_amplitude_scales`).
#             This was seasoncombo's own OVERALL WINNER, mean_wis=
#             0.2748 -- stronger than the `core` variant this
#             iteration's brief named as the round-1 baseline to beat
#             (0.2781); included here as the strongest single seasonal
#             member available, not just the one named in the brief.
#   amp      (seasoncombo, combo 4): the same amplitude-scaled pooled
#             shape as `ampbf`, but WITHOUT the backfill correction.
#             mean_wis=0.2827. Kept as a distinct member (backfill on/
#             off) even though it shares its seasonal-shape code with
#             `core`/`ampbf`, since dropping backfill changes which
#             training rows deseasonalisation and the AR fit see.
#   full     (experiments/simple-round/full/generate.jl, order=6): all
#             three of {pooled 3-harmonic FOURIER climatology
#             (season_year<=2014, a different season representation
#             from core/ampbf/amp's additive week-of-season profile),
#             MULTIPLICATIVE/per-location/median/window=6 backfill,
#             partial pooling of the per-location AR(6) coefficients
#             toward a `:fullpool` anchor at weight 0.5} stacked.
#             mean_wis=0.2997.
#   season   (submissions/seabbs_bot-season style, this experiment's
#             own `experiments/simple-round/season/generate.jl`): AR(6)
#             with ONE extra regressor per location -- a smoothed
#             circular climatology curve built from that location's own
#             FULL history strictly before the split's own forecast
#             origin (not a fixed pooled cutoff) -- plus the ar6bf
#             additive/window=8 backfill correction. mean_wis=0.3004.
#   seasonpool (experiments/simple-round/seasonpool/generate.jl): the
#             same pooled 3-harmonic Fourier shape as `full`, but with
#             only the additive/window=8 backfill correction and NO
#             partial pooling of AR coefficients. mean_wis=0.3049.
#
# `tvar` (seasoncombo combo 2) is DELIBERATELY NOT included as a
# seventh member: its own sweep's best setting is discount=1.0, which
# makes `fit_ar_discounted` mathematically identical to plain OLS
# `fit_ar` -- i.e. best-tuned `tvar` is just the pooled season (amp=1,
# no backfill) + plain AR(6) model, which is EXACTLY `amp`'s own
# shrink=0.0 sweep row (both score 0.2866 in seasoncombo/score.txt to
# 4dp). Including it here would silently double-count that one
# forecast in the pointwise mean/median instead of adding real
# diversity.
#
# Combination (pointwise, per location/origin_date/horizon/quantile
# level, exactly as in experiments/simple-round/ensemble/generate.jl):
#   ens-mean         -- simple average across the 6 members
#   ens-median       -- pointwise median across the 6 members
#   ens-wis-weighted -- weights = 1/mean_wis per member, normalised
#                       (tuned on validation only)
# All three preserve monotonicity in the quantile level for the same
# reason given in ensemble/generate.jl's header: each member's own
# quantile function is non-decreasing in the level, and an order
# statistic or convex combination of several componentwise-ordered
# vectors is itself non-decreasing.
#
# RESULT: none of the three 6-way combinations beat the strongest
# single member. `core`/`ampbf`/`amp` all share the same pooled
# additive-season code path and are highly correlated, so folding in
# the three visibly weaker, more diverse members (full/season/
# seasonpool, all ~0.30) pulls every 6-way combo's mean_wis UP, not
# down -- diversity here comes with a quality cost that outweighs the
# variance-reduction benefit. Two smaller subset combos were added to
# check whether restricting to just the strong, correlated members
# still helps at all: `ens-top2-mean` (core+ampbf) and `ens-top3-mean`/
# `ens-top3-weighted` (core+ampbf+amp). Both land BETWEEN `core` and
# `ampbf` -- i.e. still worse than simply picking `ampbf` alone. See
# score.txt: the overall best model in this experiment is the single
# member `ampbf` (mean_wis=0.2749), not any ensemble combination.
# `ampbf` is seasoncombo's own bonus combo 5 (amp+backfill), which was
# already the strongest thing in that sweep (0.2748) -- stronger than
# `core` (0.2781), the number this iteration's brief named as the
# round-1 winner to beat. This mirrors experiments/simple-round/
# ensemble/generate.jl's finding for the 4 PLAIN members: averaging
# correlated-but-uneven models does not beat picking the best one.
#
# LIGHT + ANALYTIC: CSV/DataFrames/Statistics/LinearAlgebra only, no
# Turing.
#
# Scope: VALIDATION SEASONS (1, 2) ONLY, scored against the local hub
# clone's oracle (docs/contracts.md experimental integrity) -- this is
# a tuning/selection sweep, not a submission driver. Every profile,
# shape, and pooling anchor below is estimated only from data available
# at or before each split's own forecast origin (see each member's
# description above for its own training cutoff), matching the
# discipline of every source experiment it is copied from.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl
# writes score.txt alongside this file; does not write a hub submission
# (no hub_path argument -- this is exploratory, not a `submissions/`
# candidate).

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra
using Printf

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
const HERE = @__DIR__
include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))
include(joinpath(PKG_DIR, "src", "hubio.jl"))
include(joinpath(PKG_DIR, "src", "scoring.jl"))

const TRANSFORM = :fourthroot
const AR_ORDER = 6
const N_HARMONICS = 3
const SEASON_PERIOD = 52.0
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12
const WINDOW_WEEKS = 104
const MIN_SUPPORT = 5
const VALIDATION_ONLY = (1, 2)
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")

# additive pooled-climatology profile (core/ampbf/amp): matches
# seasoncombo/generate.jl
const SEASON_CUTOFF_ADD = 2016
const SMOOTH_WINDOW_ADD = 3
const AMP_SHRINK = 1.0

# pooled Fourier shape (full/seasonpool): matches full+seasonpool's own
# CLIMATOLOGY_YEAR
const SEASON_CUTOFF_FOURIER = 2014

# backfill windows: additive (core/ampbf/amp/season/seasonpool, matches
# seabbs_bot-ar6bf) vs multiplicative (full, matches its own sweep pick)
const BF_ADD_WINDOW = 8
const BF_MULT_WINDOW = 6

# full model's partial-pooling weight, matches full/generate.jl
const POOL_WEIGHT = 0.5

const MEMBER_NAMES = ("core", "ampbf", "amp", "full", "season", "seasonpool")

# ---------------------------------------------------------------------
# Additive pooled week-of-season climatology + amplitude scaling
# (identical to experiments/simple-round/seasoncombo/generate.jl)
# ---------------------------------------------------------------------

"""
    build_seasonal_profile(hist; transform, max_season_year, min_support,
                            smooth_window) -> Dict{Int,Float64}

Pooled week-of-season climatology on the `transform` scale, identical
in design to seasoncombo/generate.jl's function of the same name: each
location's series is centred on its own mean, deviations are pooled
across all 11 locations and matching weeks, then circularly smoothed
and re-centred to zero mean.
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

Per-location amplitude scale, identical to seasoncombo/generate.jl's
function of the same name: `1.0 + shrink * (b_l - 1.0)`, `b_l` the no-
intercept OLS slope of that location's own centred deviation on the
pooled `profile` value at the matching week-of-season.
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

Identical to seasoncombo/generate.jl's function of the same name:
removes each location's own mean level and the (`amp`-scaled) pooled
seasonal shape from `Y`, returning the residual matrix and the per-
location level used.
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
# Pooled Fourier week-of-season shape (identical to full/seasonpool)
# ---------------------------------------------------------------------

"""
    fourier_features(woy, K, period) -> Vector{Float64}

`2K` Fourier features of week-of-season `woy`, identical to full/
seasonpool's helper.
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

Fit ONE shared `K`-harmonic week-of-season shape pooling all 11
locations, identical to full/seasonpool's function of the same name.
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

Pooled Fourier shape value at week-of-season `woy`.
"""
function shape_value(woy::Real, shape_coef::Vector{Float64}, K::Int,
        period::Float64)
    return dot(fourier_features(woy, K, period), shape_coef)
end

"""
    fit_seasonal_level(y, woy_vec, shape_coef, K, period) -> (alpha, beta)

Per-location, per-split 2-parameter (intercept + amplitude) OLS
adaptation of the pooled Fourier shape, identical to full/seasonpool's
function of the same name.
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
# Per-location climatology regressor (identical to
# experiments/simple-round/season/generate.jl)
# ---------------------------------------------------------------------

"""
    build_climatology(loc_hist, forecast_origin; period, smooth_window)
        -> Vector{Float64}

Smoothed circular week-of-season climatology curve for one location,
built ONLY from `loc_hist` rows strictly before `forecast_origin`.
Identical to season/generate.jl's function of the same name.
"""
function build_climatology(
    loc_hist::DataFrame, forecast_origin::Date;
    period::Int=Int(SEASON_PERIOD), smooth_window::Int=5,
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

"""
    fit_ar_clim(y, woy, order, clim) -> (coef, resid_sd)

OLS fit of AR(`order`) with intercept and one extra climatology
regressor, identical to season/generate.jl's function of the same
name.
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
        X[row, ncols] = clim[mod1(woy[t], Int(SEASON_PERIOD))]
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

Forward Gaussian-innovation AR(`order`)+climatology simulation,
identical to season/generate.jl's function of the same name.
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
            pred += coef[order + 2] *
                clim[mod1(future_woy[h], Int(SEASON_PERIOD))]
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
# Backfill correction profiles + application (additive and
# multiplicative variants, matching ar6bf/season/seasoncombo vs full)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support,
                            mode) -> Dict{Tuple{String,Int},Float64}

Empirical per-`(location, delay)` revision profile: median of `settled
- vintage` (additive) or `settled / vintage` (multiplicative), both on
the `transform` scale, identical in design to every backfill profile
builder elsewhere in this repo (see seasoncombo/full/generate.jl).
"""
function build_revision_profile(
    versions::DataFrame; transform::Symbol, max_delay::Int,
    min_support::Int, mode::Symbol,
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
            if mode == :multiplicative && abs(vintage) < 1e-6
                continue
            end
            val = mode == :additive ? settled - vintage : settled / vintage
            key = (loc, delay)
            push!(get!(raw, key, Float64[]), val)
        end
    end
    profile = Dict{Tuple{String,Int},Float64}()
    for (key, vals) in raw
        length(vals) >= min_support && (profile[key] = median(vals))
    end
    return profile
end

"""
    apply_backfill!(data, profile; mode, delay_cutoff)

Nudge `data.Y` in place (additive `+=` or multiplicative `*=`) wherever
`0 <= delay <= delay_cutoff` and a matching profile entry exists.
"""
function apply_backfill!(
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
# Plain per-location AR(order): OLS fit + Gaussian path simulation
# ---------------------------------------------------------------------

"""
    ar_design(y, order) -> (X, yresp)

Design matrix and response for an OLS AR(`order`) fit with intercept.
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
    fit_ar(y, order) -> (coef, resid_sd)

Plain OLS fit of an AR(`order`) model with intercept.
"""
function fit_ar(y::AbstractVector{Float64}, order::Int)
    X, yresp = ar_design(y, order)
    coef = X \ yresp
    resid = yresp .- X * coef
    dof = max(size(X, 1) - (order + 1), 1)
    resid_sd = sqrt(sum(abs2, resid) / dof)
    return coef, resid_sd
end

"""
    resid_sd_for(X, yresp, coef, order) -> Float64

Residual SD of `coef` (not necessarily `X`/`yresp`'s own OLS solution)
evaluated on this design -- used for the `full` member's blended
(partially-pooled) coefficients.
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
    fit_ar_pooled(ys, order) -> coef

One OLS AR(`order`) fit on the design rows of every series in `ys`
stacked together -- the `full` member's `:fullpool` anchor.
"""
function fit_ar_pooled(ys::Vector{Vector{Float64}}, order::Int)
    designs = [ar_design(y, order) for y in ys]
    Xall = reduce(vcat, first.(designs))
    yall = reduce(vcat, last.(designs))
    return Xall \ yall
end

"""
    simulate_paths(y, coef, resid_sd, order, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Gaussian-innovation AR(`order`) forward path simulation, identical to
every other plain-AR member's function of the same name in this repo.
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
# Build all six members' forecast tables in one pass over the splits.
# ---------------------------------------------------------------------

_empty_rows() = DataFrame(
    model_id=String[], location=String[], origin_date=Date[],
    horizon=Int[], target_end_date=Date[], target=String[],
    output_type=String[], output_type_id=Float64[], value=Float64[],
)

function _push_quantiles!(rows, model_id, loc, origin, h, target_end, vals)
    for q in QUANTILE_LEVELS
        qval = quantile(vals, q)
        nat = max(from_scale(qval, TRANSFORM), 0.0)
        push!(rows, (
            model_id, loc, origin, h, target_end, TARGET, "quantile", q,
            nat,
        ))
    end
end

"""
    build_member_forecasts(seasons, versions_full, hist, hist_by_loc,
        add_profile, mult_profile, season_profile, amp_scale,
        fourier_coef) -> Dict{String,DataFrame}

Fit and forecast all six seasonal+backfill members for every cross-
validation split of every season in `seasons`, sharing one raw
`ModelData` build (and its two backfilled copies) per split.
"""
function build_member_forecasts(
    seasons, versions_full, hist_by_loc, add_profile, mult_profile,
    season_profile, amp_scale, fourier_coef,
)
    rng = MersenneTwister(SEED)
    rows = Dict(m => _empty_rows() for m in MEMBER_NAMES)
    ones_amp = ones(length(LOCATIONS))
    L = length(LOCATIONS)

    for season in seasons
        splits = training_splits(season)
        for split in splits
            data_raw = build_model_data(
                split; Dmax=DMAX, transform=TRANSFORM,
                window_weeks=WINDOW_WEEKS, versions=versions_full,
            )
            data_add = deepcopy(data_raw)
            apply_backfill!(
                data_add, add_profile; mode=:additive,
                delay_cutoff=BF_ADD_WINDOW,
            )
            data_mult = deepcopy(data_raw)
            apply_backfill!(
                data_mult, mult_profile; mode=:multiplicative,
                delay_cutoff=BF_MULT_WINDOW,
            )
            origin = data_raw.origin_date
            future_woy = [
                week_of_season(origin + Day(7 * h)) for h in HORIZONS
            ]

            # -- core / ampbf: additive-backfilled data, pooled additive
            # season, amp=1 vs amp=shrunk-slope --
            R_core, level_core = deseasonalize(
                data_add.Y, data_add.woy, season_profile, ones_amp,
            )
            R_ampbf, level_ampbf = deseasonalize(
                data_add.Y, data_add.woy, season_profile, amp_scale,
            )
            # -- amp: raw (no-backfill) data, same amp-scaled season --
            R_amp, level_amp = deseasonalize(
                data_raw.Y, data_raw.woy, season_profile, amp_scale,
            )

            # -- full: multiplicative-backfilled data, pooled FOURIER
            # season (2-param per-location adaptation), collect every
            # location's residual first so the :fullpool AR anchor can
            # be fit across all of them jointly --
            full_alpha = zeros(L)
            full_beta = zeros(L)
            full_resid = Vector{Vector{Float64}}(undef, L)
            for li in 1:L
                y = Float64.(data_mult.Y[:, li])
                alpha, beta = fit_seasonal_level(
                    y, data_mult.woy, fourier_coef, N_HARMONICS,
                    SEASON_PERIOD,
                )
                full_alpha[li] = alpha
                full_beta[li] = beta
                seasonal_now = [
                    alpha + beta * shape_value(
                        w, fourier_coef, N_HARMONICS, SEASON_PERIOD,
                    ) for w in data_mult.woy
                ]
                full_resid[li] = y .- seasonal_now
            end
            full_fits = [fit_ar(full_resid[li], AR_ORDER) for li in 1:L]
            full_designs = [ar_design(full_resid[li], AR_ORDER) for li in 1:L]
            full_anchor = fit_ar_pooled(full_resid, AR_ORDER)
            full_blended = [
                (1 - POOL_WEIGHT) .* full_fits[li][1] .+
                    POOL_WEIGHT .* full_anchor
                for li in 1:L
            ]

            # -- seasonpool: additive-backfilled data, same pooled
            # FOURIER season, plain AR(6), no partial pooling --
            seasonpool_alpha = zeros(L)
            seasonpool_beta = zeros(L)
            seasonpool_resid = Vector{Vector{Float64}}(undef, L)
            for li in 1:L
                y = Float64.(data_add.Y[:, li])
                alpha, beta = fit_seasonal_level(
                    y, data_add.woy, fourier_coef, N_HARMONICS,
                    SEASON_PERIOD,
                )
                seasonpool_alpha[li] = alpha
                seasonpool_beta[li] = beta
                seasonal_now = [
                    alpha + beta * shape_value(
                        w, fourier_coef, N_HARMONICS, SEASON_PERIOD,
                    ) for w in data_add.woy
                ]
                seasonpool_resid[li] = y .- seasonal_now
            end

            for (li, loc) in enumerate(LOCATIONS)
                # core
                coef, resid_sd = fit_ar(R_core[:, li], AR_ORDER)
                paths = simulate_paths(
                    R_core[:, li], coef, resid_sd, AR_ORDER, HORIZONS,
                    NPATHS; rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    s = get(season_profile, week_of_season(target_end), 0.0)
                    vals = paths[h] .+ level_core[li] .+ s
                    _push_quantiles!(
                        rows["core"], "core", loc, origin, h, target_end,
                        vals,
                    )
                end

                # ampbf
                coef, resid_sd = fit_ar(R_ampbf[:, li], AR_ORDER)
                paths = simulate_paths(
                    R_ampbf[:, li], coef, resid_sd, AR_ORDER, HORIZONS,
                    NPATHS; rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    s = get(season_profile, week_of_season(target_end), 0.0)
                    vals = paths[h] .+ level_ampbf[li] .+ amp_scale[li] * s
                    _push_quantiles!(
                        rows["ampbf"], "ampbf", loc, origin, h, target_end,
                        vals,
                    )
                end

                # amp
                coef, resid_sd = fit_ar(R_amp[:, li], AR_ORDER)
                paths = simulate_paths(
                    R_amp[:, li], coef, resid_sd, AR_ORDER, HORIZONS,
                    NPATHS; rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    s = get(season_profile, week_of_season(target_end), 0.0)
                    vals = paths[h] .+ level_amp[li] .+ amp_scale[li] * s
                    _push_quantiles!(
                        rows["amp"], "amp", loc, origin, h, target_end, vals,
                    )
                end

                # full
                coef = full_blended[li]
                X, yresp = full_designs[li]
                resid_sd = resid_sd_for(X, yresp, coef, AR_ORDER)
                paths = simulate_paths(
                    full_resid[li], coef, resid_sd, AR_ORDER, HORIZONS,
                    NPATHS; rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    seasonal_h = full_alpha[li] + full_beta[li] * shape_value(
                        week_of_season(target_end), fourier_coef,
                        N_HARMONICS, SEASON_PERIOD,
                    )
                    vals = paths[h] .+ seasonal_h
                    _push_quantiles!(
                        rows["full"], "full", loc, origin, h, target_end,
                        vals,
                    )
                end

                # season (per-location climatology regressor)
                y = Float64.(data_add.Y[:, li])
                clim = build_climatology(hist_by_loc[loc], origin)
                coef, resid_sd = fit_ar_clim(y, data_add.woy, AR_ORDER, clim)
                paths = simulate_paths_clim(
                    y, future_woy, coef, resid_sd, AR_ORDER, clim, HORIZONS,
                    NPATHS; rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    _push_quantiles!(
                        rows["season"], "season", loc, origin, h,
                        target_end, paths[h],
                    )
                end

                # seasonpool
                coef, resid_sd = fit_ar(seasonpool_resid[li], AR_ORDER)
                paths = simulate_paths(
                    seasonpool_resid[li], coef, resid_sd, AR_ORDER, HORIZONS,
                    NPATHS; rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    seasonal_h = seasonpool_alpha[li] +
                        seasonpool_beta[li] * shape_value(
                            week_of_season(target_end), fourier_coef,
                            N_HARMONICS, SEASON_PERIOD,
                        )
                    vals = paths[h] .+ seasonal_h
                    _push_quantiles!(
                        rows["seasonpool"], "seasonpool", loc, origin, h,
                        target_end, vals,
                    )
                end
            end
        end
    end
    return rows
end

# ---------------------------------------------------------------------
# Quantile-level combination across members (identical to
# experiments/simple-round/ensemble/generate.jl).
# ---------------------------------------------------------------------

"""
    combine_members(all_df, method; weights=nothing, model_id) -> DataFrame

Combine member quantile forecasts into one ensemble forecast table,
pointwise per (location, origin_date, horizon, target_end_date,
output_type_id). `method` is `:mean`, `:median`, or `:weighted`
(`weights[model_id]`-weighted average; required for `:weighted`).
"""
function combine_members(
    all_df::DataFrame, method::Symbol;
    weights::Union{Nothing,Dict{String,Float64}}=nothing,
    model_id::String,
)
    group_cols = [:location, :origin_date, :horizon, :target_end_date,
                  :target, :output_type, :output_type_id]
    combined = combine(groupby(all_df, group_cols)) do sdf
        value = if method == :mean
            mean(sdf.value)
        elseif method == :median
            median(sdf.value)
        elseif method == :weighted
            weights === nothing && error("`:weighted` needs `weights`")
            w = [weights[m] for m in sdf.model_id]
            sum(w .* sdf.value) / sum(w)
        else
            error("unknown method $method")
        end
        (value=value,)
    end
    insertcols!(combined, 1, :model_id => model_id)
    return combined[:, [:model_id, :location, :origin_date, :horizon,
                         :target_end_date, :target, :output_type,
                         :output_type_id, :value]]
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

function main()
    t0 = time()
    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    add_profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_ADD_WINDOW,
        min_support=MIN_SUPPORT, mode=:additive,
    )
    mult_profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_MULT_WINDOW,
        min_support=MIN_SUPPORT, mode=:multiplicative,
    )

    hist = load_series("flu_data_hhs")
    season_profile = build_seasonal_profile(
        hist; transform=TRANSFORM, max_season_year=SEASON_CUTOFF_ADD,
        min_support=MIN_SUPPORT, smooth_window=SMOOTH_WINDOW_ADD,
    )
    amp_scale = build_amplitude_scales(
        hist, season_profile; transform=TRANSFORM,
        max_season_year=SEASON_CUTOFF_ADD, shrink=AMP_SHRINK,
    )
    fourier_coef = fit_pooled_shape(
        hist; transform=TRANSFORM, K=N_HARMONICS, period=SEASON_PERIOD,
        cutoff_year=SEASON_CUTOFF_FOURIER,
    )
    hist_by_loc = Dict(
        loc => hist[hist.location .== loc, [:origin_date, :wili]]
        for loc in LOCATIONS
    )

    truth = load_oracle(HUB_PATH)

    println("building 6 members' forecasts...")
    member_rows = build_member_forecasts(
        VALIDATION_ONLY, versions_full, hist_by_loc, add_profile,
        mult_profile, season_profile, amp_scale, fourier_coef,
    )
    all_members = vcat((member_rows[m] for m in MEMBER_NAMES)...)

    member_summary = Dict{String,NamedTuple}()
    member_scored = Dict{String,DataFrame}()
    for m in MEMBER_NAMES
        scored = score_forecasts(member_rows[m], truth; scale=:natural)
        summ = wis_summary(scored)[1, :]
        member_summary[m] = (
            mean_wis=summ.mean_wis, sd_wis=summ.sd_wis, n_tasks=summ.n_tasks,
        )
        member_scored[m] = scored
        @printf(
            "  %-12s mean_wis=%.4f  sd_wis=%.4f  n=%d\n", m,
            summ.mean_wis, summ.sd_wis, summ.n_tasks,
        )
    end

    # WIS-weighted combination: weight = 1/mean_wis per member,
    # normalised to sum to 1 (tuned on validation only -- this IS the
    # validation round; see docs/contracts.md experimental integrity).
    weights = Dict(m => 1 / member_summary[m].mean_wis for m in MEMBER_NAMES)
    wsum = sum(values(weights))
    weights = Dict(m => w / wsum for (m, w) in weights)

    combos = Dict(
        "ens-mean" => combine_members(
            all_members, :mean; model_id="ens-mean",
        ),
        "ens-median" => combine_members(
            all_members, :median; model_id="ens-median",
        ),
        "ens-wis-weighted" => combine_members(
            all_members, :weighted; weights=weights,
            model_id="ens-wis-weighted",
        ),
    )

    # The 6-way combos above pool 3 near-duplicate strong members (core/
    # ampbf/amp, all built on the same pooled additive season) with 3
    # visibly weaker, more diverse ones (full/season/seasonpool, all
    # ~0.30). Also check smaller subsets restricted to just the
    # strongest members, in case the weaker three are dragging the
    # pointwise average down more than their diversity is worth.
    top2 = filter(r -> r.model_id in ("core", "ampbf"), all_members)
    top3 = filter(r -> r.model_id in ("core", "ampbf", "amp"), all_members)
    top3_weights = Dict(
        m => weights[m] for m in ("core", "ampbf", "amp")
    )
    top3_wsum = sum(values(top3_weights))
    top3_weights = Dict(m => w / top3_wsum for (m, w) in top3_weights)
    merge!(
        combos, Dict(
            "ens-top2-mean" => combine_members(
                top2, :mean; model_id="ens-top2-mean",
            ),
            "ens-top3-mean" => combine_members(
                top3, :mean; model_id="ens-top3-mean",
            ),
            "ens-top3-weighted" => combine_members(
                top3, :weighted; weights=top3_weights,
                model_id="ens-top3-weighted",
            ),
        ),
    )

    combo_summary = Dict{String,NamedTuple}()
    combo_scored = Dict{String,DataFrame}()
    for (name, df_) in combos
        scored = score_forecasts(df_, truth; scale=:natural)
        summ = wis_summary(scored)[1, :]
        combo_summary[name] = (
            mean_wis=summ.mean_wis, sd_wis=summ.sd_wis, n_tasks=summ.n_tasks,
        )
        combo_scored[name] = scored
    end

    ranking = sort(
        vcat(
            [(model=m, member_summary[m]...) for m in MEMBER_NAMES],
            [(model=n, combo_summary[n]...) for n in keys(combo_summary)],
        );
        by=r -> r.mean_wis,
    )
    best = ranking[1]
    best_scored = best.model in MEMBER_NAMES ?
        member_scored[best.model] : combo_scored[best.model]

    println("\n=== seasonal ensemble round (validation seasons 1, 2 " *
            "only) ===")
    for r in ranking
        @printf(
            "%-18s mean_wis=%.4f  sd_wis=%.4f  n=%d\n", r.model, r.mean_wis,
            r.sd_wis, r.n_tasks,
        )
    end
    println("weights (1/mean_wis, normalised): ", weights)

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "ensemble of seasonal+backfill models -- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "members (each copied from its own source " *
                     "experiment, see file header for full derivation):")
        println(io, "  core       -- seasoncombo combo 1: pooled additive " *
                     "season + AR(6) + additive backfill (w=8)")
        println(io, "  ampbf      -- seasoncombo bonus combo 5 (its own " *
                     "OVERALL WINNER): core + per-location amplitude " *
                     "scale (shrink=1.0)")
        println(io, "  amp        -- seasoncombo combo 4: amplitude-" *
                     "scaled season + AR(6), NO backfill")
        println(io, "  full       -- full/generate.jl order=6: pooled " *
                     "Fourier season + multiplicative backfill (w=6) + " *
                     "fullpool AR(6) (weight 0.5)")
        println(io, "  season     -- season/generate.jl: per-location " *
                     "climatology regressor + AR(6) + additive backfill " *
                     "(w=8)")
        println(io, "  seasonpool -- seasonpool/generate.jl: pooled " *
                     "Fourier season + AR(6) + additive backfill (w=8)")
        println(io)
        println(io, "combination methods:")
        println(io, "  ens-mean         -- simple average across the 6 " *
                     "members")
        println(io, "  ens-median       -- pointwise median across the 6 " *
                     "members")
        println(io, "  ens-wis-weighted -- weights = 1/mean_wis per " *
                     "member, normalised (tuned on validation only)")
        println(io)
        println(io, "reference points:")
        println(io, "  round-1 winner (this iteration's brief)      = " *
                     "0.2781 (seasoncombo-core)")
        println(io, "  seasoncombo's OWN best single model          = " *
                     "0.2748 (ampbf, i.e. amp+backfill)")
        println(io, "  simp-ensemble (4 PLAIN members, mean/median/" *
                     "weighted) = 0.290 (worse than any seasonal member " *
                     "here)")
        println(io)
        println(io, "ranking (sorted by mean_wis, ascending):")
        println(io, rpad("model", 20) * rpad("mean_wis", 12) *
                     rpad("sd_wis", 12) * "n_tasks")
        for r in ranking
            println(io, rpad(r.model, 20) *
                         rpad(string(round(r.mean_wis; digits=4)), 12) *
                         rpad(string(round(r.sd_wis; digits=4)), 12) *
                         string(r.n_tasks))
        end
        println(io)
        println(io, "weights (1/mean_wis, normalised): $(weights)")
        println(io)
        println(io, "=== best: $(best.model) ===")
        println(io, "mean_wis=$(round(best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(best.sd_wis; digits=4)) " *
                     "n_tasks=$(best.n_tasks)")
        vs_core = 0.2781 - best.mean_wis
        vs_core_pct = 100 * vs_core / 0.2781
        println(io, "vs round-1 winner, seasoncombo-core (0.2781): " *
                     "$(round(vs_core; digits=4)) " *
                     "($(round(vs_core_pct; digits=2))%)")
        vs_ampbf = 0.2748 - best.mean_wis
        vs_ampbf_pct = 100 * vs_ampbf / 0.2748
        println(io, "vs seasoncombo's own best single model, ampbf " *
                     "(0.2748): $(round(vs_ampbf; digits=4)) " *
                     "($(round(vs_ampbf_pct; digits=2))%)")

        println(io)
        println(io, "-- breakdown by location (best) --")
        by_loc = combine(groupby(best_scored, :location),
            :wis => mean => :mean_wis, nrow => :n)
        sort!(by_loc, :mean_wis)
        for row in eachrow(by_loc)
            println(io, rpad(row.location, 16) *
                         "mean_wis=$(round(row.mean_wis; digits=4)) " *
                         "n=$(row.n)")
        end

        println(io)
        println(io, "-- breakdown by horizon (best) --")
        by_h = combine(groupby(best_scored, :horizon),
            :wis => mean => :mean_wis, nrow => :n)
        sort!(by_h, :horizon)
        for row in eachrow(by_h)
            println(io, "h=$(row.horizon): " *
                         "mean_wis=$(round(row.mean_wis; digits=4)) " *
                         "n=$(row.n)")
        end

        println(io)
        println(io, "-- breakdown by season (best) --")
        best_scored.season_year = season_year.(best_scored.origin_date)
        by_season = combine(groupby(best_scored, :season_year),
            :wis => mean => :mean_wis, nrow => :n)
        sort!(by_season, :season_year)
        for row in eachrow(by_season)
            println(io, "season $(row.season_year): " *
                         "mean_wis=$(round(row.mean_wis; digits=4)) " *
                         "n=$(row.n)")
        end
    end

    dt = round(time() - t0; digits=1)
    println("\nwrote score.txt in $(dt)s total")
    return ranking
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
