#!/usr/bin/env julia
# generate.jl -- CLOSE variant of the round-1 winner
# (experiments/simple-round/seasoncombo/generate.jl, combo 1 "core":
# pooled-seasonal + AR(6) + backfill, mean_wis=0.2781), for the
# NON-AR RESIDUAL family.
#
# Keeps the pooled climatology seasonality (`build_seasonal_profile`,
# `deseasonalize`) and the backfill correction
# (`build_revision_profile`/`apply_backfill_correction!`) of the
# round-1 winner EXACTLY -- both functions below are byte-for-byte
# copies of seasoncombo/generate.jl's. NOTE: the round-1 winner's
# "core" combo (the one scoring 0.2781, reproduced below as the
# `no-residual-change` sanity check) uses the ADDITIVE backfill
# correction (mode=:additive, window=8, per-location, median), not
# multiplicative -- this file matches that exactly so the comparison
# below isolates the residual-model change alone. (A separate sweep,
# experiments/simple-round/backfill/score.txt, found multiplicative
# slightly better in isolation -- 0.3586 vs 0.359 -- but the seasonal
# combo winner that set the 0.2781 reference point used additive, so
# additive is what this file reproduces and compares against.)
#
# What's DIFFERENT: the per-location AR(6) fit on the deseasonalised +
# backfilled residual is replaced by three NON-AR alternatives, each
# a genuine state-space smoother (not a lag regression):
#
#   1. ses:     simple exponential smoothing (ETS(A,N,N)) -- a single
#               level state, L_t = L_{t-1} + alpha*(y_t - L_{t-1}).
#               Forecast is FLAT at the final level for every horizon;
#               forward simulation compounds one-step Gaussian
#               innovations through the same recursive update, so
#               simulated h-step variance grows like
#               resid_sd^2 * (1 + (h-1)*alpha^2) (standard ETS(A,N,N)
#               growth), same mechanism nfidd-ar6's AR simulator uses
#               for its own lag state, just with a 1-parameter level
#               instead of 6 AR coefficients.
#   2. damped:  local level with mean reversion -- identical recursion
#               to `ses`, except the one-step prediction is pulled
#               toward the training-window mean `mu` at rate `phi`:
#               pred = mu + phi*(L_{t-1} - mu). `phi=1` reproduces
#               `ses` exactly; `phi<1` makes forecasts decay toward mu
#               as horizon grows and caps simulated variance growth
#               (it saturates rather than growing without bound) --
#               appropriate if the deseasonalised residual is already
#               short-range once the pooled seasonal term has absorbed
#               the slow (within-year) dynamics.
#   3. ewma:    exponentially-weighted mean -- a BATCH (not recursive)
#               weighted average of the whole training window,
#               weights w_k = rho^(n-k) (most recent observation
#               weight 1, geometrically downweighted history), used as
#               a FIXED anchor for every horizon. Unlike `ses`/`damped`
#               the anchor does not update as simulated values are
#               generated; uncertainty is a driftless random walk
#               around that fixed anchor, so simulated variance grows
#               like resid_sd^2 * h. This is the most mechanically
#               different of the three: no online state update at all,
#               just a single smoothed number plus additive white
#               noise that accumulates.
#
# Each of the three is grid-searched over its smoothing
# parameter(s) (alpha for `ses`; alpha, phi for `damped`; rho for
# `ewma`) directly against VALIDATION WIS, following the same
# tuning-by-direct-validation-WIS convention as seasoncombo's own
# tvar/ridgevar/amp sweeps (docs/contracts.md experimental integrity:
# validation seasons 1, 2 only, no test-season data anywhere in this
# file).
#
# LIGHT + ANALYTIC: CSV/DataFrames/Statistics/LinearAlgebra only, no
# Turing.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl
# writes score.txt alongside this file; does not write a hub
# submission (exploratory sweep, not a submissions/ candidate).

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
# Non-AR residual model 1: simple exponential smoothing (ETS(A,N,N))
# ---------------------------------------------------------------------

"""
    fit_ses(y, alpha) -> (level, resid_sd)

Simple exponential smoothing: a single level state updated by
`L_t = L_{t-1} + alpha*(y_t - L_{t-1})`, initialised at `y[1]`.
`resid_sd` is the RMS of the in-sample one-step-ahead errors
`y_t - L_{t-1}`. No AR lag structure at all -- the only free
parameter is `alpha`.
"""
function fit_ses(y::AbstractVector{Float64}, alpha::Float64)
    n = length(y)
    n >= 3 || error("series too short for SES: n=$n")
    L = y[1]
    resid2 = Float64[]
    for t in 2:n
        e = y[t] - L
        push!(resid2, e^2)
        L = L + alpha * e
    end
    resid_sd = sqrt(mean(resid2))
    return L, resid_sd
end

"""
    simulate_paths_ses(level0, resid_sd, alpha, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Forward Monte Carlo simulation of the SES state: at each step, draw a
Gaussian innovation around the current flat level, then update the
level exactly as in `fit_ses`. Because the level is re-estimated from
each simulated value, uncertainty compounds forward (variance grows
with horizon), the same forward-propagation pattern as the AR
simulator, just with a 1-parameter level state instead of `order` AR
lags.
"""
function simulate_paths_ses(
    level0::Float64, resid_sd::Float64, alpha::Float64, horizons,
    npaths::Int; rng::Random.AbstractRNG,
)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    for s in 1:npaths
        L = level0
        for h in 1:hmax
            val = L + resid_sd * randn(rng)
            if h in horizons
                out[h][s] = val
            end
            L = L + alpha * (val - L)
        end
    end
    return out
end

# ---------------------------------------------------------------------
# Non-AR residual model 2: damped local level (mean-reverting)
# ---------------------------------------------------------------------

"""
    fit_damped_level(y, alpha, phi) -> (level, mu, resid_sd)

Local level with mean reversion: the one-step prediction is pulled
toward the training-window mean `mu` at rate `phi` before the SES-style
update, `pred = mu + phi*(L_{t-1} - mu)`, `L_t = pred + alpha*(y_t -
pred)`. `phi = 1` reproduces `fit_ses` exactly; `phi < 1` caps how far
the level can drift from `mu`, appropriate if the deseasonalised
residual is already short-range (seasonality has absorbed the slow
dynamics, so what's left should hover near its own mean rather than
wander indefinitely as `fit_ses`'s pure random-walk level would).
"""
function fit_damped_level(y::AbstractVector{Float64}, alpha::Float64,
        phi::Float64)
    n = length(y)
    n >= 3 || error("series too short for damped level: n=$n")
    mu = mean(y)
    L = y[1]
    resid2 = Float64[]
    for t in 2:n
        pred = mu + phi * (L - mu)
        e = y[t] - pred
        push!(resid2, e^2)
        L = pred + alpha * e
    end
    resid_sd = sqrt(mean(resid2))
    return L, mu, resid_sd
end

"""
    simulate_paths_damped(level0, mu, resid_sd, alpha, phi, horizons,
                          npaths; rng) -> Dict{Int,Vector{Float64}}

Forward simulation of the damped local level, same structure as
`simulate_paths_ses` but with the mean-reverting prediction step:
`pred = mu + phi*(L - mu)`. Because `pred` is pulled toward `mu` every
step, simulated variance growth saturates with horizon rather than
compounding without bound (contrast `simulate_paths_ses`).
"""
function simulate_paths_damped(
    level0::Float64, mu::Float64, resid_sd::Float64, alpha::Float64,
    phi::Float64, horizons, npaths::Int; rng::Random.AbstractRNG,
)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    for s in 1:npaths
        L = level0
        for h in 1:hmax
            pred = mu + phi * (L - mu)
            val = pred + resid_sd * randn(rng)
            if h in horizons
                out[h][s] = val
            end
            L = pred + alpha * (val - pred)
        end
    end
    return out
end

# ---------------------------------------------------------------------
# Non-AR residual model 3: exponentially-weighted mean (batch, no
# online state update)
# ---------------------------------------------------------------------

"""
    fit_ewma_mean(y, rho) -> (level, resid_sd)

Batch exponentially-weighted mean over the WHOLE training window:
weight `w_k = rho^(n - k)` (the most recent observation gets weight 1,
older observations are geometrically downweighted), normalised to sum
to 1. `level = sum(w .* y)` is used as a FIXED anchor for every
forecast horizon -- unlike `fit_ses`/`fit_damped_level` there is no
online state update at all. `resid_sd` is the weighted RMS deviation
of `y` from this single anchor.
"""
function fit_ewma_mean(y::AbstractVector{Float64}, rho::Float64)
    n = length(y)
    n >= 3 || error("series too short for EWMA mean: n=$n")
    w = [rho^(n - t) for t in 1:n]
    w ./= sum(w)
    level = sum(w .* y)
    resid_sd = sqrt(sum(w .* (y .- level) .^ 2))
    return level, resid_sd
end

"""
    simulate_paths_ewma(level, resid_sd, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Forward simulation around the FIXED `fit_ewma_mean` anchor: a
driftless random walk of independent Gaussian increments accumulated
onto the anchor (`val_h = val_{h-1} + resid_sd*randn()`, `val_0 =
level`), so simulated variance grows linearly with horizon
(`resid_sd^2 * h`) around a centre that never itself updates -- the
most mechanically different of the three non-AR forms from the AR(6)
it replaces: no lag regression, no online level filter, just one
smoothed number plus accumulating noise.
"""
function simulate_paths_ewma(
    level::Float64, resid_sd::Float64, horizons, npaths::Int;
    rng::Random.AbstractRNG,
)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    for s in 1:npaths
        val = level
        for h in 1:hmax
            val = val + resid_sd * randn(rng)
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
    build_forecast_table(seasons, versions_full, profile; kwargs...)
        -> DataFrame

Fit and forecast one residual mechanism for every cross-validation
split of every season in `seasons`, on top of the pooled seasonal
shape `profile` and (if `backfill_profile` given) the additive
backfill correction. `residual` selects the mechanism:
`:ar` (reference AR(6)), `:ses`, `:damped`, or `:ewma`.
"""
function build_forecast_table(
    seasons, versions_full, profile::Dict{Int,Float64};
    backfill_profile::Union{Nothing,Dict}=nothing,
    backfill_window::Int=0, residual::Symbol=:ses,
    alpha::Float64=0.3, phi::Float64=0.9, rho::Float64=0.9,
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
            if backfill_profile !== nothing
                apply_backfill_correction!(
                    data, backfill_profile; mode=BF_MODE, pooled=false,
                    delay_cutoff=backfill_window,
                )
            end
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
                elseif residual == :ses
                    level0, resid_sd = fit_ses(y, alpha)
                    simulate_paths_ses(
                        level0, resid_sd, alpha, HORIZONS, NPATHS; rng=rng,
                    )
                elseif residual == :damped
                    level0, mu, resid_sd = fit_damped_level(y, alpha, phi)
                    simulate_paths_damped(
                        level0, mu, resid_sd, alpha, phi, HORIZONS, NPATHS;
                        rng=rng,
                    )
                elseif residual == :ewma
                    level0, resid_sd = fit_ewma_mean(y, rho)
                    simulate_paths_ewma(
                        level0, resid_sd, HORIZONS, NPATHS; rng=rng,
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

const ALPHAS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
const PHIS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
const RHOS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)

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
        VALIDATION_ONLY, versions_full, profile;
        backfill_profile=backfill_profile, backfill_window=BF_WINDOW,
        residual=:ar, model_id="sesresid-ar6-ref",
    )
    ar_summ = score_one(ar_ref, truth)
    println("round-1 winner sanity rerun (AR(6) residual): " *
            "mean_wis=$(round(ar_summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(ar_summ.sd_wis; digits=4))")

    # --- ses: simple exponential smoothing, alpha grid ---
    ses_results = NamedTuple[]
    for a in ALPHAS
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile;
            backfill_profile=backfill_profile, backfill_window=BF_WINDOW,
            residual=:ses, alpha=a, model_id="sesresid-ses",
        )
        summ = score_one(fc, truth)
        push!(ses_results, (
            alpha=a, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("ses alpha=$a -> mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(ses_results; by=r -> r.mean_wis)
    ses_best = ses_results[1]

    # --- damped: local level with mean reversion, alpha x phi grid ---
    damped_results = NamedTuple[]
    for a in ALPHAS, p in PHIS
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile;
            backfill_profile=backfill_profile, backfill_window=BF_WINDOW,
            residual=:damped, alpha=a, phi=p, model_id="sesresid-damped",
        )
        summ = score_one(fc, truth)
        push!(damped_results, (
            alpha=a, phi=p, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
    end
    sort!(damped_results; by=r -> r.mean_wis)
    damped_best = damped_results[1]
    println("damped best: alpha=$(damped_best.alpha) phi=$(damped_best.phi) " *
            "-> mean_wis=$(round(damped_best.mean_wis; digits=4)) " *
            "sd_wis=$(round(damped_best.sd_wis; digits=4))")

    # --- ewma: batch exponentially-weighted mean, rho grid ---
    ewma_results = NamedTuple[]
    for r in RHOS
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile;
            backfill_profile=backfill_profile, backfill_window=BF_WINDOW,
            residual=:ewma, rho=r, model_id="sesresid-ewma",
        )
        summ = score_one(fc, truth)
        push!(ewma_results, (
            rho=r, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("ewma rho=$r -> mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(ewma_results; by=r -> r.mean_wis)
    ewma_best = ewma_results[1]

    forms = [
        (name="ar6-reference", mean_wis=ar_summ.mean_wis,
         sd_wis=ar_summ.sd_wis, detail="round-1 winner, AR(6) residual"),
        (name="ses", mean_wis=ses_best.mean_wis, sd_wis=ses_best.sd_wis,
         detail="alpha=$(ses_best.alpha)"),
        (name="damped", mean_wis=damped_best.mean_wis,
         sd_wis=damped_best.sd_wis,
         detail="alpha=$(damped_best.alpha), phi=$(damped_best.phi)"),
        (name="ewma", mean_wis=ewma_best.mean_wis, sd_wis=ewma_best.sd_wis,
         detail="rho=$(ewma_best.rho)"),
    ]
    ranked = sort(forms; by=r -> r.mean_wis)
    winner_nonar = ranked[1].name == "ar6-reference" ? ranked[2] : ranked[1]

    # Best-of-the-three-non-AR forecast table, kept for the region/time
    # breakdown vs the AR(6) reference.
    best_fc = if winner_nonar.name == "ses"
        build_forecast_table(
            VALIDATION_ONLY, versions_full, profile;
            backfill_profile=backfill_profile, backfill_window=BF_WINDOW,
            residual=:ses, alpha=ses_best.alpha, model_id="sesresid-best",
        )
    elseif winner_nonar.name == "damped"
        build_forecast_table(
            VALIDATION_ONLY, versions_full, profile;
            backfill_profile=backfill_profile, backfill_window=BF_WINDOW,
            residual=:damped, alpha=damped_best.alpha, phi=damped_best.phi,
            model_id="sesresid-best",
        )
    else
        build_forecast_table(
            VALIDATION_ONLY, versions_full, profile;
            backfill_profile=backfill_profile, backfill_window=BF_WINDOW,
            residual=:ewma, rho=ewma_best.rho, model_id="sesresid-best",
        )
    end
    best_loc, best_h = breakdown(best_fc, truth)
    ar_loc, ar_h = breakdown(ar_ref, truth)

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "sesresid: non-AR residual on the round-1 winner's " *
                     "seasonal + backfill construction -- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "round-1 winner reference (seasoncombo core, " *
                     "pooled-seasonal + AR(6) + backfill): 0.2781")
        println(io, "sanity rerun here: " *
                     "mean_wis=$(round(ar_summ.mean_wis; digits=4)) " *
                     "sd_wis=$(round(ar_summ.sd_wis; digits=4))")
        println(io)
        println(io, "=== ses (simple exponential smoothing) alpha sweep ===")
        for r in ses_results
            println(io, "  alpha=$(r.alpha) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: alpha=$(ses_best.alpha) " *
                     "mean_wis=$(round(ses_best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(ses_best.sd_wis; digits=4))")
        println(io)
        println(io, "=== damped (mean-reverting local level) " *
                     "alpha x phi sweep (best 10 shown) ===")
        for r in damped_results[1:min(10, length(damped_results))]
            println(io, "  alpha=$(r.alpha) phi=$(r.phi) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: alpha=$(damped_best.alpha) phi=$(damped_best.phi) " *
                     "mean_wis=$(round(damped_best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(damped_best.sd_wis; digits=4))")
        println(io)
        println(io, "=== ewma (batch exponentially-weighted mean) " *
                     "rho sweep ===")
        for r in ewma_results
            println(io, "  rho=$(r.rho) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: rho=$(ewma_best.rho) " *
                     "mean_wis=$(round(ewma_best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(ewma_best.sd_wis; digits=4))")
        println(io)
        println(io, "=== overall comparison (best of each form) ===")
        for r in ranked
            println(io, rpad(r.name, 16) *
                         "mean_wis=$(rpad(round(r.mean_wis; digits=4), 8)) " *
                         "sd_wis=$(rpad(round(r.sd_wis; digits=4), 8)) " *
                         r.detail)
        end
        println(io)
        println(io, "=== best non-AR form: $(winner_nonar.name) ===")
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
            println(io, "CONCLUSION: the non-AR residual model " *
                         "($(winner_nonar.name)) TIES/BEATS the AR(6) " *
                         "residual on validation WIS once the pooled " *
                         "seasonal term + backfill correction are held " *
                         "fixed.")
        else
            println(io, "CONCLUSION: the best non-AR residual model " *
                         "($(winner_nonar.name)) does NOT beat the AR(6) " *
                         "residual on validation WIS once the pooled " *
                         "seasonal term + backfill correction are held " *
                         "fixed, but see the region/time breakdown above " *
                         "for whether it is usefully DIFFERENT (a " *
                         "candidate for ensembling even if not a like-" *
                         "for-like replacement).")
        end
    end

    dt = round(time() - t0; digits=1)
    println("\nbest non-AR form: $(winner_nonar.name) " *
            "mean_wis=$(round(winner_nonar.mean_wis; digits=4)) " *
            "sd_wis=$(round(winner_nonar.sd_wis; digits=4))")
    println("AR(6) reference: mean_wis=$(round(ar_summ.mean_wis; digits=4))")
    println("wrote score.txt in $(dt)s total")
    return forms
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
