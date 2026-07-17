#!/usr/bin/env julia
# ROUND 3 -- simple-round.
#
# Combines the two round-2 winners and checks whether stacking them
# beats either alone:
#
#   1. round2-stack (experiments/simple-round/round2-stack/):
#      pooled seasonal (LOG scale) + additive/per-location/median
#      backfill (window=8) + per-location AR(6) BLENDED toward a
#      fullpool anchor fit (pool_w=0.9) + Student-t(df=10, scale=1.4)
#      innovations. mean_wis=0.2601 (sd 0.2587), validation seasons
#      1, 2 -- see round2-stack/score.txt. This is the best simple-
#      round result so far and is reproduced here byte-for-byte as
#      the `ar-stack` member/reference point.
#   2. sesresid (experiments/simple-round/sesresid/): replacing the
#      AR(6) residual with a DAMPED LOCAL LEVEL (mean-reverting
#      state-space: pred = mu + phi*(L-mu), L updated SES-style at
#      rate alpha) on the round-1 core (FOURTHROOT scale, no
#      Student-t, no pooling). alpha=0.9, phi=0.9 beat the plain
#      AR(6) residual (0.2644 vs 0.2781) on that core.
#
# Two things sesresid's damped level was never tried with: the LOG
# scale (round2-stack's transform win) and Student-t innovations
# (round2-stack's interval win). Section 1 below re-grids alpha/phi
# for the damped residual on the LOG-scale core (backfill unchanged --
# additive/window=8/median, identical in both source experiments), and
# section 2 checks whether Student-t innovations, applied to the
# damped level the same variance-matched way as the AR stack, help it
# the way they helped AR(6).
#
# NOTE on "pooling" a damped level: AR(6) pooling blends each
# location's own OLS coefficient VECTOR toward a fullpool anchor fit
# (`fit_ar_pooled`) -- a per-location parameter to partially share.
# The damped level has no per-location coefficient of that kind: alpha
# and phi are already a single pair of hyperparameters grid-searched
# ONCE and then shared by all 11 locations (maximal pooling already);
# the only per-location quantities are `mu` (that location's own
# training-window mean) and the running level `L`, both of which have
# to stay location-specific for the recursion to mean-revert to the
# right place. So there is no separate pool_w knob to sweep here --
# "pooled damped level" already describes what `fit_damped_level` is.
#
# Section 3 builds an ENSEMBLE (pointwise quantile mean/median,
# `experiments/simple-round/ensemble/generate.jl`'s `combine_members`
# pattern) of the two FULL stacks -- ar-stack (Student-t/pool(0.9)
# AR(6) residual) and damped-stack (best damped-level config from
# sections 1-2) -- since they are mechanically different residual
# models built on the identical seasonal+backfill core and might
# average out each other's errors even if neither beats the other
# alone.
#
# Scored on VALIDATION SEASONS (1, 2) ONLY throughout sections 1-3
# (docs/contracts.md experimental integrity) -- no test-season data
# anywhere in the tuning/selection code below. The winning
# combination is then used, UNCHANGED, to build a full 5-season
# hub-format submission (`main`'s final block, only reached when a
# `hub_path` argument is given) -- same pattern as
# `experiments/simple-round/season/generate.jl`.
#
# LIGHT + ANALYTIC: CSV/DataFrames/Dates/Statistics/Random/
# LinearAlgebra/Distributions only, no Turing.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl [hub_path]
# writes score.txt alongside this file. If `hub_path` is given, also
# writes a hub-format submission (model_id "nfidd-round3") for all
# five seasons to `<hub_path>/model-output/` and
# `<hub_path>/model-metadata/`.

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

const MODEL_ID = "nfidd-round3"
const TRANSFORM = :log         # round2-stack's transform win
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

# Backfill design, identical to round2-stack and sesresid (both use
# this exact combination).
const BF_MODE = :additive
const BF_WINDOW = 8
const BF_STAT = :median

# round2-stack's winning interval scheme and AR pool weight.
const T_DF = 10
const T_SCALE = 1.4
const AR_POOL_W = 0.9

# Damped-level re-grid: centred on sesresid's fourthroot-core winner
# (alpha=0.9, phi=0.9), widened slightly since the log scale changes
# the residual's magnitude/spread.
const DAMPED_ALPHAS = (0.7, 0.8, 0.9, 0.95)
const DAMPED_PHIS = (0.8, 0.9, 0.95, 0.99)

# ---------------------------------------------------------------------
# Pooled seasonal shape + backfill correction (identical to
# round2-stack/generate.jl -- see that file for the full derivation)
# ---------------------------------------------------------------------

"""
    build_seasonal_profile(hist; transform, max_season_year, min_support,
                            smooth_window) -> Dict{Int,Float64}

Pooled week-of-season climatology on the `transform` scale. Identical
to `experiments/simple-round/round2-stack/generate.jl`'s function of
the same name.
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
    deseasonalize(Y, woy, profile) -> (R, level)

Remove each location's own mean level and the pooled seasonal shape
from `Y` (T x L, modelling scale). Identical to round2-stack's
function of the same name.
"""
function deseasonalize(Y::AbstractMatrix, woy::Vector{Int}, profile::Dict{Int,Float64})
    T, L = size(Y)
    level = zeros(L)
    for l in 1:L
        level[l] = mean(Float64.(Y[:, l]))
    end
    R = Matrix{Float64}(undef, T, L)
    for l in 1:L, t in 1:T
        R[t, l] = Y[t, l] - level[l] - get(profile, woy[t], 0.0)
    end
    return R, level
end

"""
    build_revision_profile(versions; transform, max_delay, min_support,
                            mode, stat) -> Dict{Tuple{String,Int},Float64}

Empirical per-(location, delay) revision profile. Identical to
round2-stack's function of the same name (`pooled=false` fixed).
"""
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
        length(vals) < min_support && continue
        profile[key] = stat == :median ? median(vals) : mean(vals)
    end
    return profile
end

"""
    apply_backfill_correction!(data, profile; mode, delay_cutoff)

Nudge `data.Y` in place. Identical to round2-stack's function of the
same name.
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
# AR(6) residual member (round2-stack's winner, byte-for-byte)
# ---------------------------------------------------------------------

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

function resid_sd_for(
    X::Matrix{Float64}, yresp::Vector{Float64}, coef::Vector{Float64},
    order::Int,
)
    nobs = size(X, 1)
    resid = yresp .- X * coef
    dof = max(nobs - (order + 1), 1)
    return sqrt(sum(abs2, resid) / dof)
end

function fit_ar(y::AbstractVector{Float64}, order::Int)
    X, yresp = ar_design(y, order)
    coef = X \ yresp
    return coef, X, yresp
end

function fit_ar_pooled(ys::Vector{Vector{Float64}}, order::Int)
    designs = [ar_design(y, order) for y in ys]
    Xall = reduce(vcat, first.(designs))
    yall = reduce(vcat, last.(designs))
    return Xall \ yall
end

"""
    simulate_paths_ar(y, coef, resid_sd, order, horizons, npaths;
                      rng, innovation, t_df, t_scale)
        -> Dict{Int,Vector{Float64}}

AR(`order`) forward simulation with Gaussian or variance-matched
Student-t innovations. Identical to round2-stack's `simulate_paths`.
"""
function simulate_paths_ar(
    y::AbstractVector{Float64}, coef::Vector{Float64}, resid_sd::Float64,
    order::Int, horizons, npaths::Int;
    rng::Random.AbstractRNG, innovation::Symbol,
    t_df::Int=T_DF, t_scale::Float64=T_SCALE,
)
    tdist = TDist(t_df)
    vscale = sqrt((t_df - 2) / t_df)
    innov_sd = innovation == :student_t ? resid_sd * vscale * t_scale : resid_sd

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
            innov = innovation == :student_t ?
                innov_sd * rand(rng, tdist) : innov_sd * randn(rng)
            val = pred + innov
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
# Damped local level residual member (sesresid's winner)
# ---------------------------------------------------------------------

"""
    fit_damped_level(y, alpha, phi) -> (level, mu, resid_sd)

Local level with mean reversion: `pred = mu + phi*(L-mu)`,
`L_t = pred + alpha*(y_t - pred)`, `mu = mean(y)`. Identical to
`experiments/simple-round/sesresid/generate.jl`'s function of the same
name.
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
                          npaths; rng, innovation, t_df, t_scale)
        -> Dict{Int,Vector{Float64}}

Forward simulation of the damped local level, extended (vs sesresid's
version) with the same Gaussian/Student-t innovation choice as
`simulate_paths_ar`, so the AR stack's interval win can be tried on
this residual too.
"""
function simulate_paths_damped(
    level0::Float64, mu::Float64, resid_sd::Float64, alpha::Float64,
    phi::Float64, horizons, npaths::Int; rng::Random.AbstractRNG,
    innovation::Symbol, t_df::Int=T_DF, t_scale::Float64=T_SCALE,
)
    tdist = TDist(t_df)
    vscale = sqrt((t_df - 2) / t_df)
    innov_sd = innovation == :student_t ? resid_sd * vscale * t_scale : resid_sd

    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    for s in 1:npaths
        L = level0
        for h in 1:hmax
            pred = mu + phi * (L - mu)
            innov = innovation == :student_t ?
                innov_sd * rand(rng, tdist) : innov_sd * randn(rng)
            val = pred + innov
            if h in horizons
                out[h][s] = val
            end
            L = pred + alpha * (val - pred)
        end
    end
    return out
end

# ---------------------------------------------------------------------
# Forecast table builder -- seasonal+backfill core, either residual
# member.
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, versions_full, profile, backfill_profile;
                          residual, innovation, pool_w, alpha, phi,
                          model_id) -> DataFrame

Fit and forecast one residual member for every cross-validation split
of every season in `seasons`, on top of the pooled seasonal
deseasonalization + backfill correction. `residual = :ar` uses the
per-location AR(6), optionally blended toward a fullpool anchor
(`pool_w > 0`); `residual = :damped` uses the damped local level
(`alpha`, `phi`). Both support `innovation in (:gaussian, :student_t)`.
"""
function build_forecast_table(
    seasons, versions_full, profile::Dict{Int,Float64},
    backfill_profile::Dict{Tuple{String,Int},Float64};
    residual::Symbol, innovation::Symbol=:gaussian, pool_w::Float64=0.0,
    alpha::Float64=0.9, phi::Float64=0.9, model_id::String,
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
            apply_backfill_correction!(
                data, backfill_profile; mode=BF_MODE,
                delay_cutoff=BF_WINDOW,
            )
            R, level = deseasonalize(data.Y, data.woy, profile)
            origin = data.origin_date
            L = data.L
            ys = [R[:, li] for li in 1:L]

            ar_blended = nothing
            ar_Xs = nothing
            ar_yresps = nothing
            if residual == :ar
                fits = [fit_ar(ys[li], AR_ORDER) for li in 1:L]
                coefs = [f[1] for f in fits]
                ar_Xs = [f[2] for f in fits]
                ar_yresps = [f[3] for f in fits]
                ar_blended = if pool_w <= 0.0
                    coefs
                else
                    anchor = fit_ar_pooled(ys, AR_ORDER)
                    [(1 - pool_w) .* coefs[li] .+ pool_w .* anchor for li in 1:L]
                end
            end

            for (li, loc) in enumerate(LOCATIONS)
                paths = if residual == :ar
                    coef = ar_blended[li]
                    resid_sd = resid_sd_for(ar_Xs[li], ar_yresps[li], coef, AR_ORDER)
                    simulate_paths_ar(
                        ys[li], coef, resid_sd, AR_ORDER, HORIZONS, NPATHS;
                        rng=rng, innovation=innovation,
                    )
                elseif residual == :damped
                    level0, mu, resid_sd = fit_damped_level(ys[li], alpha, phi)
                    simulate_paths_damped(
                        level0, mu, resid_sd, alpha, phi, HORIZONS, NPATHS;
                        rng=rng, innovation=innovation,
                    )
                else
                    error("unknown residual: $residual")
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

# ---------------------------------------------------------------------
# Ensemble combiner (pointwise quantile mean/median, same pattern as
# experiments/simple-round/ensemble/generate.jl's `combine_members`,
# specialised to exactly two member tables).
# ---------------------------------------------------------------------

"""
    combine_two(df_a, df_b, method; model_id) -> DataFrame

Pointwise-combine two forecast tables (identical task grids assumed),
per (location, origin_date, horizon, target_end_date,
output_type_id). `method` is `:mean` or `:median`. Monotone in the
quantile level with no post-hoc sorting, same argument as
`ensemble/generate.jl`'s `combine_members`.
"""
function combine_two(df_a::DataFrame, df_b::DataFrame, method::Symbol;
        model_id::String)
    a = rename(
        select(df_a, Not(:model_id)), :value => :value_a,
    )
    b = rename(
        select(df_b, Not(:model_id)), :value => :value_b,
    )
    task_cols = [:location, :origin_date, :horizon, :target_end_date,
                 :target, :output_type, :output_type_id]
    joined = innerjoin(a, b; on=task_cols)
    value = if method == :mean
        (joined.value_a .+ joined.value_b) ./ 2
    elseif method == :median
        [median((joined.value_a[i], joined.value_b[i]))
         for i in eachindex(joined.value_a)]
    else
        error("unknown method $method")
    end
    out = select(joined, task_cols)
    out.value = value
    insertcols!(out, 1, :model_id => model_id)
    return out[:, [:model_id, :location, :origin_date, :horizon,
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

score_one(forecast, truth) = wis_summary(score_forecasts(
    forecast, truth; scale=:natural,
))[1, :]

"""
    coverage(forecast, truth, level) -> Float64

Empirical coverage of the nominal `level` central interval. Identical
to round2-stack's function of the same name.
"""
function coverage(forecast::DataFrame, truth::DataFrame, level::Float64)
    a = (1 - level) / 2
    task_cols = [:location, :origin_date, :horizon, :target_end_date]
    lo = forecast[isapprox.(forecast.output_type_id, a; atol=1e-6), :]
    hi = forecast[isapprox.(forecast.output_type_id, 1 - a; atol=1e-6), :]
    lo_r = rename(lo[:, vcat(task_cols, [:value])], :value => :lo)
    hi_r = rename(hi[:, vcat(task_cols, [:value])], :value => :hi)
    joined = innerjoin(lo_r, hi_r, on=task_cols)
    joined = innerjoin(joined, truth, on=[:location, :target_end_date])
    return mean(joined.lo .<= joined.value .<= joined.hi)
end

function summarize(label, fc, truth)
    summ = score_one(fc, truth)
    cov50 = coverage(fc, truth, 0.5)
    cov90 = coverage(fc, truth, 0.9)
    println("  $(rpad(label, 40)) mean_wis=$(round(summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(summ.sd_wis; digits=4)) " *
            "cov50=$(round(cov50; digits=3)) cov90=$(round(cov90; digits=3))")
    return (label=label, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
            cov50=cov50, cov90=cov90, forecast=fc)
end

function main()
    hub_path = length(ARGS) >= 1 ? ARGS[1] : nothing
    t0 = time()
    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")
    truth = load_oracle(HUB_PATH)
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= MAX_TRAIN_SEASON_YEAR, :,
    ]

    profile = build_seasonal_profile(
        hist; transform=TRANSFORM, max_season_year=MAX_TRAIN_SEASON_YEAR,
        min_support=MIN_SUPPORT, smooth_window=SMOOTH_WINDOW,
    )
    bf_profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_WINDOW,
        min_support=MIN_SUPPORT, mode=BF_MODE, stat=BF_STAT,
    )

    println("=== sanity: reproduce round2-stack's winner " *
            "(log+tstudent+pool(w=0.9), ref mean_wis=0.2601) ===")
    ar_stack = summarize(
        "ar-stack (log+tstudent+pool(w=0.9))",
        build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, bf_profile;
            residual=:ar, innovation=:student_t, pool_w=AR_POOL_W,
            model_id="round3-ar-stack",
        ),
        truth,
    )

    println("\n=== section 1: damped-level alpha x phi re-grid, " *
            "LOG scale, Gaussian innovations ===")
    damped_grid = NamedTuple[]
    for a in DAMPED_ALPHAS, p in DAMPED_PHIS
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, bf_profile;
            residual=:damped, innovation=:gaussian, alpha=a, phi=p,
            model_id="round3-damped-grid",
        )
        summ = score_one(fc, truth)
        push!(damped_grid, (alpha=a, phi=p, mean_wis=summ.mean_wis,
                            sd_wis=summ.sd_wis))
        println("  alpha=$a phi=$p -> mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(damped_grid; by=r -> r.mean_wis)
    damped_best = damped_grid[1]
    println("best (Gaussian): alpha=$(damped_best.alpha) " *
            "phi=$(damped_best.phi) " *
            "mean_wis=$(round(damped_best.mean_wis; digits=4))")

    println("\n=== section 2: Student-t innovations on the damped " *
            "level (best alpha/phi from section 1) ===")
    damped_stack = summarize(
        "damped-stack (log+tstudent+damped($(damped_best.alpha),$(damped_best.phi)))",
        build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, bf_profile;
            residual=:damped, innovation=:student_t,
            alpha=damped_best.alpha, phi=damped_best.phi,
            model_id="round3-damped-stack",
        ),
        truth,
    )
    damped_gaussian = summarize(
        "damped-gaussian (log+damped($(damped_best.alpha),$(damped_best.phi)), no tstudent)",
        build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, bf_profile;
            residual=:damped, innovation=:gaussian,
            alpha=damped_best.alpha, phi=damped_best.phi,
            model_id="round3-damped-gaussian",
        ),
        truth,
    )

    println("\n=== section 3: ensemble of ar-stack and damped-stack ===")
    ens_mean = summarize(
        "ensemble-mean(ar-stack, damped-stack)",
        combine_two(
            ar_stack.forecast, damped_stack.forecast, :mean;
            model_id="round3-ensemble-mean",
        ),
        truth,
    )
    ens_median = summarize(
        "ensemble-median(ar-stack, damped-stack)",
        combine_two(
            ar_stack.forecast, damped_stack.forecast, :median;
            model_id="round3-ensemble-median",
        ),
        truth,
    )

    candidates = [ar_stack, damped_gaussian, damped_stack, ens_mean, ens_median]
    ranked = sort(candidates; by=r -> r.mean_wis)
    winner = ranked[1]

    println("\n=== ranked ===")
    for r in ranked
        println("  $(rpad(r.label, 55)) mean_wis=$(round(r.mean_wis; digits=4)) " *
                "sd_wis=$(round(r.sd_wis; digits=4))")
    end
    println("\nwinner: $(winner.label) mean_wis=$(round(winner.mean_wis; digits=4))")
    println("vs round-2 stack (0.2601): " *
            "$(round(0.2601 - winner.mean_wis; digits=4)) " *
            "($(round(100 * (0.2601 - winner.mean_wis) / 0.2601; digits=2))%)")

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "round 3 -- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "round-2 stack reference (log+tstudent+pool(w=0.9)): " *
                     "mean_wis=0.2601 sd_wis=0.2587 " *
                     "(experiments/simple-round/round2-stack/score.txt)")
        println(io, "sesresid reference (fourthroot core, damped(0.9,0.9), " *
                     "no tstudent/pooling): mean_wis=0.2644 " *
                     "(experiments/simple-round/sesresid/score.txt)")
        println(io)
        println(io, "=== sanity check: ar-stack reproduction ===")
        println(io, "  $(rpad(ar_stack.label, 55)) mean_wis=" *
                     "$(round(ar_stack.mean_wis; digits=4)) sd_wis=" *
                     "$(round(ar_stack.sd_wis; digits=4)) cov50=" *
                     "$(round(ar_stack.cov50; digits=3)) cov90=" *
                     "$(round(ar_stack.cov90; digits=3))")
        println(io)
        println(io, "=== section 1: damped-level alpha x phi re-grid " *
                     "(LOG scale, Gaussian innovations) ===")
        for r in damped_grid
            println(io, "  alpha=$(r.alpha) phi=$(r.phi) -> mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: alpha=$(damped_best.alpha) phi=$(damped_best.phi) " *
                     "mean_wis=$(round(damped_best.mean_wis; digits=4))")
        println(io)
        println(io, "=== section 2: Student-t on the damped level ===")
        for r in (damped_gaussian, damped_stack)
            println(io, "  $(rpad(r.label, 65)) mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4)) cov50=" *
                         "$(round(r.cov50; digits=3)) cov90=" *
                         "$(round(r.cov90; digits=3))")
        end
        tstudent_delta = damped_gaussian.mean_wis - damped_stack.mean_wis
        println(io, "Student-t delta on damped level: " *
                     "$(round(tstudent_delta; digits=4)) " *
                     "($(round(100 * tstudent_delta / damped_gaussian.mean_wis; digits=2))%)")
        println(io)
        println(io, "=== section 3: ensembles ===")
        for r in (ens_mean, ens_median)
            println(io, "  $(rpad(r.label, 55)) mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4)) cov50=" *
                         "$(round(r.cov50; digits=3)) cov90=" *
                         "$(round(r.cov90; digits=3))")
        end
        println(io)
        println(io, "=== ranked (all candidates) ===")
        for r in ranked
            println(io, "  $(rpad(r.label, 55)) mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4)) cov50=" *
                         "$(round(r.cov50; digits=3)) cov90=" *
                         "$(round(r.cov90; digits=3))")
        end
        println(io)
        println(io, "=== winner: $(winner.label) ===")
        println(io, "mean_wis=$(round(winner.mean_wis; digits=4)) " *
                     "sd_wis=$(round(winner.sd_wis; digits=4)) " *
                     "cov50=$(round(winner.cov50; digits=3)) " *
                     "cov90=$(round(winner.cov90; digits=3))")
        vs_ref = 0.2601 - winner.mean_wis
        vs_pct = 100 * vs_ref / 0.2601
        println(io, "vs round-2 stack (0.2601): $(round(vs_ref; digits=4)) " *
                     "($(round(vs_pct; digits=2))%)")
        println(io)
        if winner.label == ar_stack.label
            println(io, "CONCLUSION: neither the damped-level residual nor " *
                         "the ar-stack/damped-stack ensemble beats the " *
                         "round-2 ar-stack (0.2601) on validation WIS. The " *
                         "damped-level residual and the AR(6)+pool+" *
                         "Student-t stack do NOT compound -- the ar-stack " *
                         "remains the round-3 candidate, unchanged.")
        else
            println(io, "CONCLUSION: $(winner.label) BEATS the round-2 " *
                         "ar-stack (0.2601) on validation WIS -- see the " *
                         "delta above. This is the round-3 candidate.")
        end
    end

    if hub_path !== nothing
        println("\nbuilding full 5-season hub submission with winner: " *
                "$(winner.label) ...")
        full = if winner.label == ar_stack.label
            build_forecast_table(
                (1, 2, 3, 4, 5), versions_full, profile, bf_profile;
                residual=:ar, innovation=:student_t, pool_w=AR_POOL_W,
                model_id=MODEL_ID,
            )
        elseif winner.label == damped_stack.label
            build_forecast_table(
                (1, 2, 3, 4, 5), versions_full, profile, bf_profile;
                residual=:damped, innovation=:student_t,
                alpha=damped_best.alpha, phi=damped_best.phi,
                model_id=MODEL_ID,
            )
        elseif winner.label == damped_gaussian.label
            build_forecast_table(
                (1, 2, 3, 4, 5), versions_full, profile, bf_profile;
                residual=:damped, innovation=:gaussian,
                alpha=damped_best.alpha, phi=damped_best.phi,
                model_id=MODEL_ID,
            )
        else
            full_ar = build_forecast_table(
                (1, 2, 3, 4, 5), versions_full, profile, bf_profile;
                residual=:ar, innovation=:student_t, pool_w=AR_POOL_W,
                model_id=MODEL_ID,
            )
            full_damped = build_forecast_table(
                (1, 2, 3, 4, 5), versions_full, profile, bf_profile;
                residual=:damped, innovation=:student_t,
                alpha=damped_best.alpha, phi=damped_best.phi,
                model_id=MODEL_ID,
            )
            method = winner.label == ens_mean.label ? :mean : :median
            combine_two(full_ar, full_damped, method; model_id=MODEL_ID)
        end
        n_origins = length(unique(full.origin_date))
        println("built $(nrow(full)) rows across $(n_origins) origin " *
                "date(s)")
        write_submission(full, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="nfidd", model_abbr="round3", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end

    dt = round(time() - t0; digits=1)
    println("\nwrote score.txt in $(dt)s total")
    return ranked
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
