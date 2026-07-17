#!/usr/bin/env julia
# search_grid.jl -- SYSTEMATIC 3x2x2x2x2 grid over every simple-round
# component family that has, on its own, beaten the AR(6) baseline:
#
#   AR order:      {6, 8, 12}          (experiments/simple-round/ar-order)
#   backfill:      {none, multiplicative w6 per-loc median}
#                                      (winning cell of
#                                      experiments/simple-round/backfill's
#                                      mode/window/pooling/stat sweep)
#   pooled season: {off, on}           (experiments/simple-round/seasonpool:
#                                      ONE shared week-of-season Fourier
#                                      shape fit across all 11 locations
#                                      + history, subtracted before the
#                                      AR fit -- NOT the per-location
#                                      Fourier that overfit, and NOT
#                                      season/'s per-location climatology)
#   AR pooling:    {off, fullpool w=0.5} (experiments/simple-round/pool)
#   differencing:  {off (AR on level), on (AR on first difference of
#                  the deseasonalised residual)} -- new for this grid
#
# 3*2*2*2*2 = 48 combinations, run serially in ONE process so the
# expensive, factor-independent parts -- CSV loads, the backfill
# revision profile, the pooled seasonal shape, and (per split/location)
# the backfilled+deseasonalised residual series -- are each computed
# ONCE and reused across every combination that shares them. Only 4
# distinct (backfill, season) data variants exist; each of the 12
# (ar_order, ar_pool, diff) model configurations is then a cheap
# per-split OLS fit + Monte Carlo simulation on the cached residual
# series, not a full data rebuild.
#
# Deliberately LIGHT + ANALYTIC, like every other simple-round script:
# CSV/DataFrames/Statistics/LinearAlgebra only, no Turing.
#
# SCORES ON VALIDATION SEASONS (1, 2) ONLY (docs/contracts.md
# experimental integrity); `training_splits` refuses seasons 3-5 unless
# `allow_test_season=true`, never passed here.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> search_grid.jl
# writes ranked_table.txt (all 48 rows) alongside this file.

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
const DMAX = 12
const WINDOW_WEEKS = 104
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")

# Winning cell of experiments/simple-round/backfill/score.txt's
# mode x window x pooling x stat sweep (mean_wis=0.3586, best of all 30
# combinations tried there).
const BF_MODE = :multiplicative
const BF_WINDOW = 6
const BF_POOLED = false
const BF_STAT = :median
const BF_MIN_SUPPORT = 5

# Pooled seasonal shape, identical settings to
# experiments/simple-round/seasonpool/generate.jl.
const N_HARMONICS = 3
const SEASON_PERIOD = 52.0
const CLIMATOLOGY_YEAR = 2014

# AR partial-pooling anchor + shrinkage weight, per the brief
# ("fullpool w0.5"); matches experiments/simple-round/pool's :fullpool
# scheme at w=0.5.
const POOL_WEIGHT = 0.5

const AR_ORDERS = (6, 8, 12)
const BACKFILL_OPTS = (false, true)
const SEASON_OPTS = (false, true)
const POOL_OPTS = (false, true)
const DIFF_OPTS = (false, true)

# ---------------------------------------------------------------------
# Backfill correction (generalised mode/window/pooling/stat, copied
# from experiments/simple-round/backfill/generate.jl; only the
# multiplicative/w6/per-loc/median cell is ever exercised here).
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support,
                            mode, pooled, stat) -> Dict

Empirical revision profile on the `transform` scale. See
experiments/simple-round/backfill/generate.jl for the full derivation
of the `mode`/`pooled`/`stat` generalisation; unchanged here.
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

Nudge `data.Y` in place wherever `0 <= delay <= delay_cutoff` and a
matching `profile` entry exists. Identical to
experiments/simple-round/backfill/generate.jl.
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
# Pooled seasonal climatology (copied from
# experiments/simple-round/seasonpool/generate.jl).
# ---------------------------------------------------------------------

"""
    fourier_features(woy, K, period) -> Vector{Float64}

`2K` Fourier features of week-of-season `woy` at the given `period`.
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

ONE shared `K`-harmonic week-of-season shape, pooling all 11 locations'
`transform`-scale series (each centred on its own mean first) over
`history` restricted to `season_year(origin_date) <= cutoff_year`.
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

"""Shared pooled seasonal shape value (deviation from a location's own
mean, `transform` scale) at week-of-season `woy`."""
shape_value(woy::Real, shape_coef::Vector{Float64}, K::Int, period::Float64) =
    dot(fourier_features(woy, K, period), shape_coef)

"""
    fit_seasonal_level(y, woy_vec, shape_coef, K, period) -> (alpha, beta)

Per-location OLS fit of `y_t = alpha + beta * shape(woy_t) + resid`: the
small per-location amplitude/level scaling of the shared shape.
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
# AR(p) fit (design-returning, so a shrunk/blended coefficient can be
# re-scored on the same design) + fullpool anchor + forward simulation
# with optional cumulative reconstruction (for the differenced case).
# Copied/merged from experiments/simple-round/pool/generate.jl.
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

"""Residual SD of `coef` (not necessarily the OLS solution for `X`,
`yresp`) evaluated on this design."""
function resid_sd_for(
    X::Matrix{Float64}, yresp::Vector{Float64}, coef::Vector{Float64},
    order::Int,
)
    nobs = size(X, 1)
    resid = yresp .- X * coef
    dof = max(nobs - (order + 1), 1)
    return sqrt(sum(abs2, resid) / dof)
end

"""OLS AR(`order`) fit; returns `(coef, X, yresp)`."""
function fit_ar(y::AbstractVector{Float64}, order::Int)
    X, yresp = ar_design(y, order)
    coef = X \ yresp
    return coef, X, yresp
end

"""One OLS AR(`order`) fit on the design rows of every series in `ys`
stacked together -- the `:fullpool` common-dynamics anchor."""
function fit_ar_pooled(ys::Vector{Vector{Float64}}, order::Int)
    designs = [ar_design(y, order) for y in ys]
    Xall = reduce(vcat, first.(designs))
    yall = reduce(vcat, last.(designs))
    return Xall \ yall
end

"""
    simulate_paths(series, coef, resid_sd, order, horizons, npaths;
                   rng, level0=nothing) -> Dict{Int,Vector{Float64}}

Simulate `npaths` Gaussian-innovation AR(`order`) sample paths forward
from the end of `series`, for each horizon in `horizons`.

When `level0 === nothing` (the non-differenced case), `series` already
lives on the level scale the forecast is wanted on, and `out[h][s]` is
simply the simulated value at step `h` -- identical to every other
simple-round script's `simulate_paths`.

When `level0` is given (the differenced case: `series` is
`diff(resid)`), each step's simulated value is a DIFFERENCE, and
`out[h][s]` instead accumulates `level0 + sum(d_1, ..., d_h)` PER PATH
-- horizon-`h` quantiles of a cumulative sum are not the cumulative sum
of each horizon's own quantiles, so the running level must be tracked
inside the same per-path loop that draws the innovations, not
reconstructed afterwards from `out[h]` in isolation.
"""
function simulate_paths(
    series::AbstractVector{Float64}, coef::Vector{Float64},
    resid_sd::Float64, order::Int, horizons, npaths::Int;
    rng::Random.AbstractRNG, level0::Union{Nothing,Float64}=nothing,
)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    tail0 = series[(end - order + 1):end]
    accumulate = level0 !== nothing
    for s in 1:npaths
        tail = copy(tail0)
        level = accumulate ? level0 : 0.0
        for h in 1:hmax
            pred = coef[1]
            for lag in 1:order
                pred += coef[lag + 1] * tail[end - lag + 1]
            end
            val = pred + resid_sd * randn(rng)
            accumulate && (level += val)
            if h in horizons
                out[h][s] = accumulate ? level : val
            end
            push!(tail, val)
            popfirst!(tail)
        end
    end
    return out
end

# ---------------------------------------------------------------------
# Data caching: build the (backfill, season) residual series ONCE per
# pair (4 total), reused across every (ar_order, pool, diff) combo that
# shares it.
# ---------------------------------------------------------------------

"""One cross-validation split's cached, backfilled, (optionally)
deseasonalised residual series -- everything the 12 (ar_order, pool,
diff) model configurations sharing this (backfill, season) pair need,
computed once."""
struct SplitCache
    origin_date::Date
    resid::Vector{Vector{Float64}}             # per location
    seasonal_offsets::Vector{Vector{Float64}}  # per location, per horizon
end

"""
    build_split_cache(versions_full, profile, shape_coef;
                       use_backfill, use_season) -> Vector{SplitCache}

Build the backfilled + (optionally) deseasonalised residual series for
every split of both validation seasons, in order. `profile` is the
multiplicative/w6/per-loc/median backfill correction (applied only
when `use_backfill`); `shape_coef` is the pooled seasonal shape
(subtracted, via a 2-parameter per-location OLS refit, only when
`use_season`).
"""
function build_split_cache(
    versions_full, profile, shape_coef; use_backfill::Bool,
    use_season::Bool,
)
    L = length(LOCATIONS)
    caches = SplitCache[]
    for season in VALIDATION_SEASONS
        for split in training_splits(season)
            data = build_model_data(
                split; Dmax=DMAX, transform=TRANSFORM,
                window_weeks=WINDOW_WEEKS,
                versions=use_backfill ? versions_full : nothing,
            )
            use_backfill && apply_backfill_correction!(
                data, profile; mode=BF_MODE, pooled=BF_POOLED,
                delay_cutoff=BF_WINDOW,
            )
            origin = data.origin_date
            resid = Vector{Vector{Float64}}(undef, L)
            offsets = Vector{Vector{Float64}}(undef, L)
            for li in 1:L
                y = Float64.(data.Y[:, li])
                if use_season
                    alpha, beta = fit_seasonal_level(
                        y, data.woy, shape_coef, N_HARMONICS, SEASON_PERIOD,
                    )
                    seasonal_now = [
                        alpha + beta * shape_value(
                            w, shape_coef, N_HARMONICS, SEASON_PERIOD,
                        ) for w in data.woy
                    ]
                    resid[li] = y .- seasonal_now
                    offsets[li] = [
                        alpha + beta * shape_value(
                            week_of_season(origin + Day(7 * h)), shape_coef,
                            N_HARMONICS, SEASON_PERIOD,
                        ) for h in HORIZONS
                    ]
                else
                    resid[li] = y
                    offsets[li] = zeros(length(HORIZONS))
                end
            end
            push!(caches, SplitCache(origin, resid, offsets))
        end
    end
    return caches
end

# ---------------------------------------------------------------------
# Forecast table for one (ar_order, backfill, season, pool, diff) cell
# ---------------------------------------------------------------------

"""
    build_forecast_table(model_id, split_caches, ar_order; use_pool,
                          use_diff) -> DataFrame

Fit and forecast one grid cell across every cached split. `use_diff`
fits AR(`ar_order`) to the first difference of each location's cached
residual series (reconstructing the residual-scale forecast via
`simulate_paths`'s per-path cumulative sum) rather than to the residual
level directly. `use_pool` blends each location's own AR coefficients
`(1 - POOL_WEIGHT) * own + POOL_WEIGHT * anchor` with a fullpool
common-dynamics anchor fit jointly on all 11 locations' series (in the
same, diff'd-or-not representation) for this split.
"""
function build_forecast_table(
    model_id::String, split_caches::Vector{SplitCache}, ar_order::Int;
    use_pool::Bool, use_diff::Bool,
)
    rng = MersenneTwister(SEED)
    L = length(LOCATIONS)
    rows = DataFrame(
        model_id=String[], location=String[], origin_date=Date[],
        horizon=Int[], target_end_date=Date[], target=String[],
        output_type=String[], output_type_id=Float64[], value=Float64[],
    )
    for sc in split_caches
        origin = sc.origin_date
        series = [
            use_diff ? diff(sc.resid[li]) : sc.resid[li] for li in 1:L
        ]
        fits = [fit_ar(series[li], ar_order) for li in 1:L]
        coefs = [f[1] for f in fits]
        Xs = [f[2] for f in fits]
        yresps = [f[3] for f in fits]

        blended = if use_pool
            anchor = fit_ar_pooled(series, ar_order)
            [
                (1 - POOL_WEIGHT) .* coefs[li] .+ POOL_WEIGHT .* anchor
                for li in 1:L
            ]
        else
            coefs
        end

        for li in 1:L
            loc = LOCATIONS[li]
            coef = blended[li]
            resid_sd = resid_sd_for(Xs[li], yresps[li], coef, ar_order)
            level0 = use_diff ? sc.resid[li][end] : nothing
            paths = simulate_paths(
                series[li], coef, resid_sd, ar_order, HORIZONS, NPATHS;
                rng=rng, level0=level0,
            )
            for h in HORIZONS
                target_end = origin + Day(7 * h)
                vals = paths[h] .+ sc.seasonal_offsets[li][h]
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
# Main: build the 4 cached data variants once, then run all 48 model
# configurations against them.
# ---------------------------------------------------------------------

function main()
    t0 = time()

    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_WINDOW,
        min_support=BF_MIN_SUPPORT, mode=BF_MODE, pooled=BF_POOLED,
        stat=BF_STAT,
    )
    println("backfill profile (multiplicative, w=$(BF_WINDOW), " *
            "per-loc, median): $(length(profile)) entries")

    history = load_series("flu_data_hhs")
    shape_coef = fit_pooled_shape(
        history; transform=TRANSFORM, K=N_HARMONICS, period=SEASON_PERIOD,
        cutoff_year=CLIMATOLOGY_YEAR,
    )
    println("pooled seasonal shape ($(N_HARMONICS) harmonics, " *
            "season_year <= $(CLIMATOLOGY_YEAR)): " *
            "coef=$(round.(shape_coef; digits=4))")

    truth = load_oracle(HUB_PATH)

    println("\nbuilding the 4 (backfill, season) cached data variants...")
    data_cache = Dict{Tuple{Bool,Bool},Vector{SplitCache}}()
    for bf in BACKFILL_OPTS, sn in SEASON_OPTS
        data_cache[(bf, sn)] = build_split_cache(
            versions_full, profile, shape_coef; use_backfill=bf,
            use_season=sn,
        )
    end
    dt_data = round(time() - t0; digits=1)
    println("data cached in $(dt_data)s")

    results = DataFrame(
        ar_order=Int[], backfill=Bool[], season=Bool[], ar_pool=Bool[],
        diff=Bool[], mean_wis=Float64[], sd_wis=Float64[], n_tasks=Int[],
    )

    n_done = 0
    n_total = length(AR_ORDERS) * length(BACKFILL_OPTS) *
              length(SEASON_OPTS) * length(POOL_OPTS) * length(DIFF_OPTS)
    for ar_order in AR_ORDERS, bf in BACKFILL_OPTS, sn in SEASON_OPTS,
        pl in POOL_OPTS, df in DIFF_OPTS

        model_id = "grid-ar$(ar_order)" * (bf ? "-bf" : "") *
            (sn ? "-sn" : "") * (pl ? "-pl" : "") * (df ? "-df" : "")
        forecast = build_forecast_table(
            model_id, data_cache[(bf, sn)], ar_order; use_pool=pl,
            use_diff=df,
        )
        scored = score_forecasts(forecast, truth; scale=:natural)
        summ = wis_summary(scored)
        push!(results, (
            ar_order, bf, sn, pl, df, summ.mean_wis[1], summ.sd_wis[1],
            summ.n_tasks[1],
        ))
        n_done += 1
        println("[$(n_done)/$(n_total)] order=$(ar_order) bf=$(bf) " *
                "sn=$(sn) pool=$(pl) diff=$(df) -> mean_wis=" *
                "$(round(summ.mean_wis[1]; digits=4)) sd_wis=" *
                "$(round(summ.sd_wis[1]; digits=4)) " *
                "($(round(time() - t0; digits=1))s elapsed)")
    end

    sort!(results, :mean_wis)
    dt = round(time() - t0; digits=1)
    println("\ngrid search done in $(dt)s total, sorted by mean WIS:")
    show(results; allrows=true, allcols=true)
    println()

    open(joinpath(HERE, "ranked_table.txt"), "w") do io
        println(io, "simple-round systematic grid: AR order x backfill x " *
                     "pooled season x AR pooling x differencing")
        println(io, "3x2x2x2x2 = $(nrow(results)) combinations, " *
                     "validation seasons $(VALIDATION_SEASONS) only, " *
                     "natural-scale WIS")
        println(io, "runtime: $(dt)s total ($(dt_data)s data caching)")
        println(io)
        println(io, "backfill = multiplicative, w=$(BF_WINDOW), " *
                     "per-loc, median (winning cell of " *
                     "experiments/simple-round/backfill/score.txt)")
        println(io, "season = pooled $(N_HARMONICS)-harmonic Fourier " *
                     "shape, season_year <= $(CLIMATOLOGY_YEAR) " *
                     "(experiments/simple-round/seasonpool)")
        println(io, "ar_pool = fullpool, w=$(POOL_WEIGHT) shrinkage " *
                     "(experiments/simple-round/pool)")
        println(io, "diff = AR fit to the first difference of the " *
                     "(optionally deseasonalised) residual, " *
                     "reconstructed via per-path cumulative sum")
        println(io)
        println(io, "reference points:")
        println(io, "  nfidd-ar6        (order=6,  no backfill)         " *
                     "= 0.368")
        println(io, "  seabbs_bot-ar6bf (order=6,  backfill)            " *
                     "= 0.359")
        println(io, "  ar-order best    (order=12, backfill)            " *
                     "= 0.3518")
        println(io, "  combo (per-loc Fourier season, ar4+bf)           " *
                     "= 0.3349")
        println(io, "  seasonpool (order=6, bf, pooled season)          " *
                     "= 0.3049")
        println(io)
        println(io, "full ranked table (sorted by mean_wis, ascending):")
        println(io, rpad("order", 7) * rpad("backfill", 10) *
                     rpad("season", 8) * rpad("ar_pool", 9) *
                     rpad("diff", 7) * rpad("mean_wis", 11) * "sd_wis")
        for row in eachrow(results)
            println(io,
                rpad(string(row.ar_order), 7) *
                rpad(string(row.backfill), 10) *
                rpad(string(row.season), 8) *
                rpad(string(row.ar_pool), 9) *
                rpad(string(row.diff), 7) *
                rpad(string(round(row.mean_wis; digits=4)), 11) *
                string(round(row.sd_wis; digits=4)),
            )
        end

        println(io)
        println(io, "-- main effects (mean_wis averaged over the other " *
                     "4 factors) --")
        for (factor, opts) in (
            (:ar_order, AR_ORDERS), (:backfill, BACKFILL_OPTS),
            (:season, SEASON_OPTS), (:ar_pool, POOL_OPTS),
            (:diff, DIFF_OPTS),
        )
            println(io, "  $(factor):")
            for opt in opts
                sub = results[results[!, factor] .== opt, :]
                println(io, "    $(opt): mean_wis=" *
                             "$(round(mean(sub.mean_wis); digits=4)) " *
                             "(over $(nrow(sub)) cells), mean_sd_wis=" *
                             "$(round(mean(sub.sd_wis); digits=4))")
            end
        end

        println(io)
        best = results[1, :]
        println(io, "best overall: order=$(best.ar_order) " *
                     "backfill=$(best.backfill) season=$(best.season) " *
                     "ar_pool=$(best.ar_pool) diff=$(best.diff)")
        println(io, "  mean_wis=$(round(best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(best.sd_wis; digits=4)) " *
                     "n_tasks=$(best.n_tasks)")
    end
    println("\nwrote ranked_table.txt to $(HERE)")

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
