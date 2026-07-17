#!/usr/bin/env julia
# generate.jl -- CLOSE variant of the round-1 winner
# (experiments/simple-round/seasoncombo/generate.jl, combo 1 "core":
# pooled-seasonal + AR(6) + backfill, mean_wis=0.2781), for the
# LOCAL-LEVEL / RANDOM-WALK RESIDUAL family.
#
# Keeps the pooled climatology seasonality (`build_seasonal_profile`,
# `deseasonalize`) and the backfill correction
# (`build_revision_profile`/`apply_backfill_correction!`) of the
# round-1 winner EXACTLY -- both functions below are byte-for-byte
# copies of seasoncombo/generate.jl's. NOTE: the round-1 winner's
# "core" combo (the one scoring 0.2781, reproduced below as the
# `ar6-reference` sanity check) uses the ADDITIVE backfill correction
# (mode=:additive, window=8, per-location, median), not multiplicative
# -- this file matches that exactly so the comparison below isolates
# the residual-model change alone. (A separate sweep,
# experiments/simple-round/backfill/score.txt, found multiplicative
# slightly better in isolation -- 0.3586 vs 0.359 -- but the seasonal
# combo winner that set the 0.2781 reference point used additive, so
# additive is what this file reproduces and compares against; see
# experiments/simple-round/sesresid/generate.jl, the sibling non-AR-
# residual family, which made and documented the same choice.)
#
# What's DIFFERENT: the per-location AR(6) fit on the deseasonalised +
# backfilled residual is replaced by three genuine LOCAL-LEVEL state-
# space forms, none of them a lag regression:
#
#   1. rw:          driftless random walk -- the forecast simply
#                   PERSISTS the last observed residual at every
#                   horizon (`E[y_{t+h}] = y_t`), with independent
#                   innovations accumulating step by step, so
#                   simulated variance grows LINEARLY with horizon
#                   (`resid_sd^2 * h`, the classic driftless-RW
#                   growth rate -- no bound).
#   2. damped:      the same random walk, but the level is pulled
#                   toward ZERO (not the training-window mean, unlike
#                   `sesresid`'s mean-reverting form -- the residual
#                   here is already deseasonalised AND backfill-
#                   corrected, so there is no obvious non-zero anchor
#                   left to revert to) by a fixed decay factor `phi`
#                   applied every simulated step:
#                   `L_h = phi * L_{h-1} + innovation`. `phi` is swept
#                   directly as a hyperparameter against validation
#                   WIS (not fit by OLS per split/location -- that
#                   would just be AR(1), already covered by the
#                   `ar-order` family) -- the same "sweep a shared
#                   discount/decay knob" convention as `seasoncombo`'s
#                   `tvar` combo. `phi=1` reproduces `rw` exactly;
#                   `phi<1` caps simulated variance growth at a finite
#                   stationary limit (`resid_sd^2 / (1 - phi^2)`)
#                   instead of growing without bound.
#   3. locallinear: a local LINEAR trend, extrapolated a short way
#                   forward. The slope is a Theil-Sen estimator (the
#                   MEDIAN of all pairwise slopes over the last
#                   `window` residual points) -- robust to the single
#                   outlier point that an OLS slope over so short a
#                   window would be highly sensitive to. The forecast
#                   is the last observed residual plus `h` times that
#                   slope, with innovations accumulating exactly as in
#                   `rw` (so variance again grows linearly with
#                   horizon).
#
# Each form's own structural knob (none for `rw`; `phi` for `damped`;
# `window` for `locallinear`) is swept first against Gaussian-
# innovation validation WIS to isolate the state-space MECHANISM's own
# effect, then -- following experiments/simple-round/intervals/
# generate.jl's finding that variance-matched Student-t innovations
# (df=10) beat both Gaussian and empirical-bootstrap innovations for
# this project's AR(6)+backfill baseline -- each form's spread is
# RECALIBRATED with the same three innovation families (Gaussian,
# Student-t(df=10) variance-matched, and an empirical iid bootstrap of
# that fit's own in-sample one-step residuals), each at several
# dispersion scales, since a genuinely different residual mechanism
# has no reason to inherit `intervals`' AR(6)-tuned scale unchanged.
#
# LIGHT + ANALYTIC: CSV/DataFrames/Distributions/Statistics/
# LinearAlgebra only, no Turing (`Distributions` is used only for
# `TDist`, exactly as experiments/simple-round/intervals/generate.jl
# already does -- it is a project dependency, not part of the
# Turing/Mooncake/Pathfinder stack this sweep still avoids).
#
# Scope: VALIDATION SEASONS (1, 2) ONLY, scored against the local hub
# clone's oracle (docs/contracts.md experimental integrity) -- a
# tuning sweep, not a submission driver.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl
# writes score.txt alongside this file; does not write a hub
# submission (exploratory sweep, not a submissions/ candidate).

using CSV
using DataFrames
using Dates
using Distributions
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
const AR_ORDER = 6              # only used by the AR(6) reference rerun
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12
const WINDOW_WEEKS = 104
const MIN_SUPPORT = 5
const SMOOTH_WINDOW = 3
const VALIDATION_ONLY = (1, 2)
const MAX_TRAIN_SEASON_YEAR = 2016
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")

# Reference backfill design, identical to seasoncombo's "core" combo
# (the one that set the 0.2781 reference point).
const BF_MODE = :additive
const BF_WINDOW = 8
const BF_STAT = :median

# Student-t degrees of freedom for the recalibrated-spread sweep,
# fixed at the value experiments/simple-round/intervals/generate.jl
# found in the middle of its own flat-optimal range (8-20 all within
# ~0.0002 mean WIS of each other for the AR(6)+backfill baseline).
const T_DF = 10.0

# ---------------------------------------------------------------------
# Pooled seasonal shape (byte-for-byte from seasoncombo/generate.jl)
# ---------------------------------------------------------------------

"""
    build_seasonal_profile(hist; transform, max_season_year, min_support,
                            smooth_window) -> Dict{Int,Float64}

Pooled week-of-season climatology, identical to
`experiments/simple-round/seasoncombo/generate.jl`'s function of the
same name -- see that file for the full derivation. Kept unchanged
here: this experiment only varies the residual mechanism, not the
seasonal term.
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
    deseasonalize(Y, woy, profile, amp) -> (R, level)

Identical to seasoncombo/generate.jl's function of the same name;
`amp` is always `ones(L)` in this file (no per-location amplitude
scaling -- that was a separate combo, orthogonal to the residual-model
question this file asks).
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
# Backfill correction (byte-for-byte from seasoncombo/generate.jl,
# additive/per-location/median variant only)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support,
                            mode, pooled, stat) -> Dict

Identical to seasoncombo/generate.jl's function of the same name.
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
    apply_backfill_correction!(data, profile; mode, pooled, delay_cutoff)

Identical to seasoncombo/generate.jl's function of the same name.
"""
function apply_backfill_correction!(
    data::ModelData, profile::Dict; mode::Symbol, pooled::Bool,
    delay_cutoff::Int,
)
    for l in 1:data.L, t in 1:data.T
        d = data.delay[t, l]
        (d < 0 || d > delay_cutoff) && continue
        ismissing(data.Y[t, l]) && continue
        key = pooled ? d : (LOCATIONS[l], d)
        haskey(profile, key) || continue
        c = profile[key]
        data.Y[t, l] = mode == :additive ? data.Y[t, l] + c : data.Y[t, l] * c
    end
    return data
end

# ---------------------------------------------------------------------
# Reference per-location AR(6) (byte-for-byte from seasoncombo, used
# only to reproduce the 0.2781 sanity check for comparison)
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

function simulate_paths_ar(
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
# Innovation draws: Gaussian, variance-matched Student-t(T_DF), or an
# empirical iid bootstrap of a fit's own in-sample residuals
# ---------------------------------------------------------------------

"""
    draw_innovation(rng, mode, resid_sd, scale, pool, tdist) -> Float64

One innovation draw, dispatched on `mode`:

  - `:gaussian`:   `resid_sd * scale * randn(rng)`.
  - `:student_t`:  `resid_sd * scale` times a `tdist = TDist(T_DF)`
                   draw, variance-matched so `scale=1` has the same
                   variance as the Gaussian case
                   (`Var(TDist(df)) = df / (df - 2)`), exactly the
                   convention `experiments/simple-round/intervals/
                   generate.jl` used for the AR(6)+backfill baseline.
  - `:empirical`:  `scale` times a uniform-with-replacement draw from
                   `pool`, the fit's own in-sample one-step residuals
                   (already approximately zero-mean by construction --
                   see each `fit_*` function).

`pool` is unused (may be `nothing`) for `:gaussian`/`:student_t`.
"""
function draw_innovation(
    rng::Random.AbstractRNG, mode::Symbol, resid_sd::Float64,
    scale::Float64, pool::Union{Nothing,Vector{Float64}}, tdist::TDist,
)
    if mode == :gaussian
        return resid_sd * scale * randn(rng)
    elseif mode == :student_t
        vscale = sqrt((T_DF - 2) / T_DF)
        return resid_sd * vscale * scale * rand(rng, tdist)
    elseif mode == :empirical
        pool === nothing && error("empirical mode needs a residual pool")
        return scale * pool[rand(rng, 1:length(pool))]
    else
        error("unknown noise mode: $mode")
    end
end

# ---------------------------------------------------------------------
# Local-level residual model 1: driftless random walk (persist last
# residual)
# ---------------------------------------------------------------------

"""
    fit_rw(y) -> (anchor, resid_sd, pool)

Driftless random walk: `anchor = y[end]` is the flat forecast for
every horizon; `pool`/`resid_sd` are the in-sample first differences
`y[t] - y[t-1]`, the model's own one-step-ahead prediction errors
under a driftless RW (no fitting beyond that -- there is no free
parameter at all).
"""
function fit_rw(y::AbstractVector{Float64})
    n = length(y)
    n >= 3 || error("series too short for RW: n=$n")
    pool = diff(y)
    resid_sd = sqrt(mean(abs2, pool))
    return y[end], resid_sd, pool
end

"""
    simulate_paths_rw(anchor, resid_sd, horizons, npaths; rng, noise_mode,
                       scale, pool) -> Dict{Int,Vector{Float64}}

Forward Monte Carlo simulation of the driftless RW: at each step,
`val_h = val_{h-1} + innovation`, `val_0 = anchor`. Uncertainty
accumulates without bound (`Var(val_h) = h * resid_sd^2 * scale^2`
under Gaussian/Student-t innovations).
"""
function simulate_paths_rw(
    anchor::Float64, resid_sd::Float64, horizons, npaths::Int;
    rng::Random.AbstractRNG, noise_mode::Symbol, scale::Float64,
    pool::Union{Nothing,Vector{Float64}},
)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    tdist = TDist(T_DF)
    for s in 1:npaths
        val = anchor
        for h in 1:hmax
            val += draw_innovation(rng, noise_mode, resid_sd, scale, pool,
                                    tdist)
            if h in horizons
                out[h][s] = val
            end
        end
    end
    return out
end

# ---------------------------------------------------------------------
# Local-level residual model 2: damped random walk (decays toward
# zero, not toward a training-window mean)
# ---------------------------------------------------------------------

"""
    fit_damped_rw(y, phi) -> (anchor, resid_sd, pool)

Damped random walk with a FIXED decay `phi` applied every step toward
zero (not toward `mean(y)` -- the deseasonalised, backfill-corrected
residual has no obvious non-zero level left to revert to). `phi` is
swept as a hyperparameter, not fit by OLS (a per-split/location OLS
fit of `y[t]` on `y[t-1]` alone would just be AR(1), already covered
by the `ar-order` family). `pool`/`resid_sd` are the in-sample one-
step errors `y[t] - phi*y[t-1]` this fixed-`phi` model would have
made.
"""
function fit_damped_rw(y::AbstractVector{Float64}, phi::Float64)
    n = length(y)
    n >= 3 || error("series too short for damped RW: n=$n")
    pool = [y[t] - phi * y[t - 1] for t in 2:n]
    resid_sd = sqrt(mean(abs2, pool))
    return y[end], resid_sd, pool
end

"""
    simulate_paths_damped_rw(anchor, resid_sd, phi, horizons, npaths;
                              rng, noise_mode, scale, pool)
        -> Dict{Int,Vector{Float64}}

Forward simulation of the damped RW: `val_h = phi*val_{h-1} +
innovation`, `val_0 = anchor`. `phi=1` reproduces `simulate_paths_rw`
exactly; `phi<1` caps simulated variance at the AR(1)-style
stationary limit `resid_sd^2 * scale^2 / (1 - phi^2)` instead of
growing without bound.
"""
function simulate_paths_damped_rw(
    anchor::Float64, resid_sd::Float64, phi::Float64, horizons,
    npaths::Int; rng::Random.AbstractRNG, noise_mode::Symbol,
    scale::Float64, pool::Union{Nothing,Vector{Float64}},
)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    tdist = TDist(T_DF)
    for s in 1:npaths
        val = anchor
        for h in 1:hmax
            val = phi * val +
                  draw_innovation(rng, noise_mode, resid_sd, scale, pool,
                                  tdist)
            if h in horizons
                out[h][s] = val
            end
        end
    end
    return out
end

# ---------------------------------------------------------------------
# Local-level residual model 3: local-linear trend (Theil-Sen robust
# slope, short extrapolation)
# ---------------------------------------------------------------------

"""
    theil_sen_slope(vals) -> Float64

Theil-Sen slope estimator: the MEDIAN of `(vals[j] - vals[i]) / (j -
i)` over every pair `i < j`. Robust to a single outlier point, unlike
an OLS slope over so short a window -- appropriate here since `vals`
is only the last `window` residual points (see `fit_locallinear`).
"""
function theil_sen_slope(vals::AbstractVector{Float64})
    n = length(vals)
    slopes = Float64[]
    for i in 1:(n - 1), j in (i + 1):n
        push!(slopes, (vals[j] - vals[i]) / (j - i))
    end
    return median(slopes)
end

"""
    fit_locallinear(y, window) -> (anchor, slope, resid_sd, pool)

Local linear trend over the last `window` residual points (fewer if
`y` is shorter): `slope` is the Theil-Sen estimator of `vals =
y[(end - window + 1):end]`; `anchor = y[end]`. `pool`/`resid_sd` are
the in-sample one-step errors `vals[t] - vals[t-1] - slope` this
constant-slope model would have made over that same window -- a SHORT
window by design (this is a local, not global, trend), so the
residual pool it returns for the empirical bootstrap noise mode is
correspondingly small.
"""
function fit_locallinear(y::AbstractVector{Float64}, window::Int)
    n = length(y)
    w = min(window, n)
    w >= 4 || error("window too short for local-linear: window=$w")
    vals = y[(end - w + 1):end]
    slope = theil_sen_slope(vals)
    pool = [vals[t] - vals[t - 1] - slope for t in 2:w]
    resid_sd = sqrt(mean(abs2, pool))
    return y[end], slope, resid_sd, pool
end

"""
    simulate_paths_locallinear(anchor, slope, resid_sd, horizons, npaths;
                                rng, noise_mode, scale, pool)
        -> Dict{Int,Vector{Float64}}

Forward simulation of the local-linear trend: `val_h = val_{h-1} +
slope + innovation`, `val_0 = anchor`. Like `rw`, uncertainty grows
linearly with horizon; unlike `rw`, the mean forecast drifts by
`slope` each step rather than staying flat.
"""
function simulate_paths_locallinear(
    anchor::Float64, slope::Float64, resid_sd::Float64, horizons,
    npaths::Int; rng::Random.AbstractRNG, noise_mode::Symbol,
    scale::Float64, pool::Union{Nothing,Vector{Float64}},
)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    tdist = TDist(T_DF)
    for s in 1:npaths
        val = anchor
        for h in 1:hmax
            val += slope +
                   draw_innovation(rng, noise_mode, resid_sd, scale, pool,
                                   tdist)
            if h in horizons
                out[h][s] = val
            end
        end
    end
    return out
end

# ---------------------------------------------------------------------
# Forecast table builder
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, versions_full, profile, backfill_profile;
                          kwargs...) -> DataFrame

Fit and forecast one residual mechanism for every cross-validation
split of every season in `seasons`, on top of the pooled seasonal
shape `profile` and the additive backfill correction.
`residual` selects the mechanism: `:ar` (reference AR(6)), `:rw`,
`:damped`, or `:locallinear`. `noise_mode`/`scale` select the
innovation family (see `draw_innovation`); ignored (Gaussian, scale=1)
by `:ar`, which always reproduces the round-1 winner's own scheme.
"""
function build_forecast_table(
    seasons, versions_full, profile::Dict{Int,Float64},
    backfill_profile::Dict; residual::Symbol=:rw, phi::Float64=0.9,
    window::Int=8, noise_mode::Symbol=:gaussian, scale::Float64=1.0,
    model_id::String,
)
    rng = MersenneTwister(SEED)
    amp = ones(length(LOCATIONS))
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
                data, backfill_profile; mode=BF_MODE, pooled=false,
                delay_cutoff=BF_WINDOW,
            )
            R, level = deseasonalize(data.Y, data.woy, profile, amp)
            origin = data.origin_date

            for (li, loc) in enumerate(LOCATIONS)
                y = R[:, li]
                paths = if residual == :ar
                    coef, resid_sd = fit_ar(y, AR_ORDER)
                    simulate_paths_ar(
                        y, coef, resid_sd, AR_ORDER, HORIZONS, NPATHS;
                        rng=rng,
                    )
                elseif residual == :rw
                    anchor, resid_sd, pool = fit_rw(y)
                    simulate_paths_rw(
                        anchor, resid_sd, HORIZONS, NPATHS; rng=rng,
                        noise_mode=noise_mode, scale=scale, pool=pool,
                    )
                elseif residual == :damped
                    anchor, resid_sd, pool = fit_damped_rw(y, phi)
                    simulate_paths_damped_rw(
                        anchor, resid_sd, phi, HORIZONS, NPATHS; rng=rng,
                        noise_mode=noise_mode, scale=scale, pool=pool,
                    )
                elseif residual == :locallinear
                    anchor, slope, resid_sd, pool = fit_locallinear(y, window)
                    simulate_paths_locallinear(
                        anchor, slope, resid_sd, HORIZONS, NPATHS; rng=rng,
                        noise_mode=noise_mode, scale=scale, pool=pool,
                    )
                else
                    error("unknown residual model: $residual")
                end
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    s = get(profile, week_of_season(target_end), 0.0)
                    vals = paths[h] .+ level[li] .+ s
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

"""Per-location and per-horizon mean-WIS breakdown for one scored
forecast table, joined against `truth`."""
function breakdown(forecast, truth)
    scored = score_forecasts(forecast, truth; scale=:natural)
    by_loc = combine(groupby(scored, :location), :wis => mean => :mean_wis)
    sort!(by_loc, :mean_wis)
    by_h = combine(groupby(scored, :horizon), :wis => mean => :mean_wis)
    sort!(by_h, :horizon)
    return by_loc, by_h
end

# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------

const PHIS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0)
const WINDOWS = (4, 6, 8, 10, 12, 16)
const NOISE_MODES = (:gaussian, :student_t, :empirical)
const SCALES = (0.8, 1.0, 1.2, 1.4, 1.6, 2.0)

"""Sweep `NOISE_MODES` x `SCALES` for one already-structurally-tuned
residual form; returns the sorted results and the best entry."""
function sweep_noise(seasons, versions_full, profile, backfill_profile;
        residual::Symbol, phi::Float64=0.9, window::Int=8, tag::String)
    results = NamedTuple[]
    for mode in NOISE_MODES, sc in SCALES
        fc = build_forecast_table(
            seasons, versions_full, profile, backfill_profile;
            residual=residual, phi=phi, window=window, noise_mode=mode,
            scale=sc, model_id="locallevel-$tag-noise",
        )
        summ = score_one(fc, load_oracle(HUB_PATH))
        push!(results, (
            mode=mode, scale=sc, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
    end
    sort!(results; by=r -> r.mean_wis)
    return results, results[1]
end

function main()
    t0 = time()
    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")
    truth = load_oracle(HUB_PATH)

    profile = build_seasonal_profile(
        hist; transform=TRANSFORM, max_season_year=MAX_TRAIN_SEASON_YEAR,
        min_support=MIN_SUPPORT, smooth_window=SMOOTH_WINDOW,
    )

    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= MAX_TRAIN_SEASON_YEAR, :,
    ]
    backfill_profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_WINDOW,
        min_support=5, mode=BF_MODE, pooled=false, stat=BF_STAT,
    )

    # Sanity rerun: round-1 winner (AR(6) residual), should reproduce
    # 0.2781.
    ar_ref = build_forecast_table(
        VALIDATION_ONLY, versions_full, profile, backfill_profile;
        residual=:ar, model_id="locallevel-ar6-ref",
    )
    ar_summ = score_one(ar_ref, truth)
    println("round-1 winner sanity rerun (AR(6) residual): " *
            "mean_wis=$(round(ar_summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(ar_summ.sd_wis; digits=4))")

    # --- rw: driftless random walk, Gaussian-noise sanity point ---
    rw_gauss = build_forecast_table(
        VALIDATION_ONLY, versions_full, profile, backfill_profile;
        residual=:rw, model_id="locallevel-rw-gauss",
    )
    rw_gauss_summ = score_one(rw_gauss, truth)
    println("rw (Gaussian, scale=1) -> " *
            "mean_wis=$(round(rw_gauss_summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(rw_gauss_summ.sd_wis; digits=4))")

    # --- damped: phi sweep, Gaussian noise ---
    damped_phi_results = NamedTuple[]
    for phi in PHIS
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, backfill_profile;
            residual=:damped, phi=phi, model_id="locallevel-damped-phi",
        )
        summ = score_one(fc, truth)
        push!(damped_phi_results, (
            phi=phi, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("damped (Gaussian) phi=$phi -> " *
                "mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(damped_phi_results; by=r -> r.mean_wis)
    damped_phi_best = damped_phi_results[1]

    # --- locallinear: window sweep, Gaussian noise ---
    ll_window_results = NamedTuple[]
    for w in WINDOWS
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, backfill_profile;
            residual=:locallinear, window=w,
            model_id="locallevel-locallinear-window",
        )
        summ = score_one(fc, truth)
        push!(ll_window_results, (
            window=w, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("locallinear (Gaussian) window=$w -> " *
                "mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(ll_window_results; by=r -> r.mean_wis)
    ll_window_best = ll_window_results[1]

    # --- recalibrate spread (noise_mode x scale) for each structurally
    #     tuned form ---
    rw_noise_results, rw_noise_best = sweep_noise(
        VALIDATION_ONLY, versions_full, profile, backfill_profile;
        residual=:rw, tag="rw",
    )
    println("rw best recalibrated: mode=$(rw_noise_best.mode) " *
            "scale=$(rw_noise_best.scale) -> " *
            "mean_wis=$(round(rw_noise_best.mean_wis; digits=4)) " *
            "sd_wis=$(round(rw_noise_best.sd_wis; digits=4))")

    damped_noise_results, damped_noise_best = sweep_noise(
        VALIDATION_ONLY, versions_full, profile, backfill_profile;
        residual=:damped, phi=damped_phi_best.phi, tag="damped",
    )
    println("damped best recalibrated: phi=$(damped_phi_best.phi) " *
            "mode=$(damped_noise_best.mode) scale=$(damped_noise_best.scale)" *
            " -> mean_wis=$(round(damped_noise_best.mean_wis; digits=4)) " *
            "sd_wis=$(round(damped_noise_best.sd_wis; digits=4))")

    ll_noise_results, ll_noise_best = sweep_noise(
        VALIDATION_ONLY, versions_full, profile, backfill_profile;
        residual=:locallinear, window=ll_window_best.window, tag="locallinear",
    )
    println("locallinear best recalibrated: window=$(ll_window_best.window) " *
            "mode=$(ll_noise_best.mode) scale=$(ll_noise_best.scale) -> " *
            "mean_wis=$(round(ll_noise_best.mean_wis; digits=4)) " *
            "sd_wis=$(round(ll_noise_best.sd_wis; digits=4))")

    forms = [
        (name="ar6-reference", mean_wis=ar_summ.mean_wis,
         sd_wis=ar_summ.sd_wis, detail="round-1 winner, AR(6) residual"),
        (name="rw", mean_wis=rw_noise_best.mean_wis,
         sd_wis=rw_noise_best.sd_wis,
         detail="mode=$(rw_noise_best.mode) scale=$(rw_noise_best.scale)"),
        (name="damped", mean_wis=damped_noise_best.mean_wis,
         sd_wis=damped_noise_best.sd_wis,
         detail="phi=$(damped_phi_best.phi) mode=$(damped_noise_best.mode) " *
                 "scale=$(damped_noise_best.scale)"),
        (name="locallinear", mean_wis=ll_noise_best.mean_wis,
         sd_wis=ll_noise_best.sd_wis,
         detail="window=$(ll_window_best.window) " *
                 "mode=$(ll_noise_best.mode) scale=$(ll_noise_best.scale)"),
    ]
    ranked = sort(forms; by=r -> r.mean_wis)
    winner_nonar = ranked[1].name == "ar6-reference" ? ranked[2] : ranked[1]

    # Best-of-the-three-non-AR forecast table, kept for the region/time
    # breakdown vs the AR(6) reference.
    best_fc = if winner_nonar.name == "rw"
        build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, backfill_profile;
            residual=:rw, noise_mode=rw_noise_best.mode,
            scale=rw_noise_best.scale, model_id="locallevel-best",
        )
    elseif winner_nonar.name == "damped"
        build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, backfill_profile;
            residual=:damped, phi=damped_phi_best.phi,
            noise_mode=damped_noise_best.mode, scale=damped_noise_best.scale,
            model_id="locallevel-best",
        )
    else
        build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, backfill_profile;
            residual=:locallinear, window=ll_window_best.window,
            noise_mode=ll_noise_best.mode, scale=ll_noise_best.scale,
            model_id="locallevel-best",
        )
    end
    best_loc, best_h = breakdown(best_fc, truth)
    ar_loc, ar_h = breakdown(ar_ref, truth)

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "locallevel: local-level/random-walk residual on the " *
                     "round-1 winner's seasonal + backfill construction " *
                     "-- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "round-1 winner reference (seasoncombo core, " *
                     "pooled-seasonal + AR(6) + backfill): 0.2781")
        println(io, "sanity rerun here: " *
                     "mean_wis=$(round(ar_summ.mean_wis; digits=4)) " *
                     "sd_wis=$(round(ar_summ.sd_wis; digits=4))")
        println(io)
        println(io, "=== rw (driftless random walk) ===")
        println(io, "Gaussian/scale=1 sanity point: " *
                     "mean_wis=$(round(rw_gauss_summ.mean_wis; digits=4)) " *
                     "sd_wis=$(round(rw_gauss_summ.sd_wis; digits=4))")
        println(io, "noise x scale recalibration (best 8 shown):")
        for r in rw_noise_results[1:min(8, length(rw_noise_results))]
            println(io, "  mode=$(r.mode) scale=$(r.scale) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: mode=$(rw_noise_best.mode) " *
                     "scale=$(rw_noise_best.scale) " *
                     "mean_wis=$(round(rw_noise_best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(rw_noise_best.sd_wis; digits=4))")
        println(io)
        println(io, "=== damped (random walk decaying toward zero) ===")
        println(io, "phi sweep (Gaussian, scale=1):")
        for r in damped_phi_results
            println(io, "  phi=$(r.phi) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io, "best phi: $(damped_phi_best.phi) " *
                     "mean_wis=$(round(damped_phi_best.mean_wis; digits=4))")
        println(io, "noise x scale recalibration at phi=" *
                     "$(damped_phi_best.phi) (best 8 shown):")
        for r in damped_noise_results[1:min(8, length(damped_noise_results))]
            println(io, "  mode=$(r.mode) scale=$(r.scale) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: mode=$(damped_noise_best.mode) " *
                     "scale=$(damped_noise_best.scale) " *
                     "mean_wis=$(round(damped_noise_best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(damped_noise_best.sd_wis; digits=4))")
        println(io)
        println(io, "=== locallinear (Theil-Sen robust local trend) ===")
        println(io, "window sweep (Gaussian, scale=1):")
        for r in ll_window_results
            println(io, "  window=$(r.window) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io, "best window: $(ll_window_best.window) " *
                     "mean_wis=$(round(ll_window_best.mean_wis; digits=4))")
        println(io, "noise x scale recalibration at window=" *
                     "$(ll_window_best.window) (best 8 shown):")
        for r in ll_noise_results[1:min(8, length(ll_noise_results))]
            println(io, "  mode=$(r.mode) scale=$(r.scale) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: mode=$(ll_noise_best.mode) " *
                     "scale=$(ll_noise_best.scale) " *
                     "mean_wis=$(round(ll_noise_best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(ll_noise_best.sd_wis; digits=4))")
        println(io)
        println(io, "=== overall comparison (best of each form) ===")
        for r in ranked
            println(io, rpad(r.name, 16) *
                         "mean_wis=$(rpad(round(r.mean_wis; digits=4), 8)) " *
                         "sd_wis=$(rpad(round(r.sd_wis; digits=4), 8)) " *
                         r.detail)
        end
        println(io)
        println(io, "=== best local-level form: $(winner_nonar.name) ===")
        println(io, "mean_wis=$(round(winner_nonar.mean_wis; digits=4)) " *
                     "sd_wis=$(round(winner_nonar.sd_wis; digits=4)) " *
                     "($(winner_nonar.detail))")
        vs_ar = ar_summ.mean_wis - winner_nonar.mean_wis
        vs_pct = 100 * vs_ar / ar_summ.mean_wis
        println(io, "vs AR(6)-residual round-1 winner " *
                     "($(round(ar_summ.mean_wis; digits=4))): " *
                     "$(round(vs_ar; digits=4)) ($(round(vs_pct; digits=2))%)")
        println(io)
        println(io, "region/time breakdown, $(winner_nonar.name) vs " *
                     "AR(6)-residual reference (same validation tasks)")
        println(io)
        println(io, "  by location (mean WIS)")
        joined_loc = innerjoin(
            best_loc, ar_loc; on=:location, makeunique=true,
        )
        rename!(joined_loc, :mean_wis => :best, :mean_wis_1 => :ar6)
        sort!(joined_loc, :best)
        for row in eachrow(joined_loc)
            println(io, "    $(rpad(row.location, 16)) ar6=" *
                         "$(round(row.ar6; digits=4))  " *
                         "$(winner_nonar.name)=$(round(row.best; digits=4))")
        end
        println(io)
        println(io, "  by horizon (mean WIS)")
        for h in HORIZONS
            b = best_h[best_h.horizon .== h, :mean_wis][1]
            a = ar_h[ar_h.horizon .== h, :mean_wis][1]
            println(io, "    h=$h: ar6=$(round(a; digits=4)) -> " *
                         "$(winner_nonar.name)=$(round(b; digits=4))")
        end
        println(io)
        if winner_nonar.mean_wis <= ar_summ.mean_wis
            println(io, "CONCLUSION: the local-level residual model " *
                         "($(winner_nonar.name)) TIES/BEATS the AR(6) " *
                         "residual on validation WIS once the pooled " *
                         "seasonal term + backfill correction are held " *
                         "fixed.")
        else
            println(io, "CONCLUSION: the best local-level residual model " *
                         "($(winner_nonar.name)) does NOT beat the AR(6) " *
                         "residual on validation WIS once the pooled " *
                         "seasonal term + backfill correction are held " *
                         "fixed. Whether it is nonetheless a useful " *
                         "DIVERSE ensemble member depends on whether its " *
                         "errors are structurally different from the " *
                         "AR(6) reference's, not just worse on average " *
                         "-- see the region/time breakdown above: a " *
                         "state-space model with no lag structure at all " *
                         "(no fitted phi_1..phi_6) will tend to miss in " *
                         "different places/horizons than the AR(6) " *
                         "residual, which is the property that matters " *
                         "for ensembling even when the standalone mean " *
                         "WIS is worse.")
        end
    end

    dt = round(time() - t0; digits=1)
    println("\nbest local-level form: $(winner_nonar.name) " *
            "mean_wis=$(round(winner_nonar.mean_wis; digits=4)) " *
            "sd_wis=$(round(winner_nonar.sd_wis; digits=4))")
    println("AR(6) reference: mean_wis=$(round(ar_summ.mean_wis; digits=4))")
    println("wrote score.txt in $(dt)s total")
    return forms
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
