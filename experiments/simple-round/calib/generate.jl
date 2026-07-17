#!/usr/bin/env julia
# CALIBRATION -- coverage + bias fix on top of the round-2 stack winner
# (experiments/simple-round/round2-stack/generate.jl, combo
# "log+tstudent+pool(w=0.9)": pooled seasonal climatology (log scale) +
# per-location backfill correction + per-location AR(6) on the
# deseasonalized residual, coefficients partially pooled (w=0.9) toward
# a fullpool anchor, simulated forward with Student-t(df=10, scale=1.4)
# innovations). Validation WIS 0.2601, cov50=0.565, cov90=0.943 --
# reproduced here as the BEFORE baseline (round2-stack/score.txt).
#
# That winner is slightly OVER-wide at the two headline levels
# (cov50/cov90 both above nominal). This experiment:
#
#   1. DIAGNOSES coverage at every nominal decile (10..90) and the
#      signed bias (median forecast minus truth, on both the natural
#      and the model's own log scale, plus the WIS over/under-
#      prediction decomposition as a pinball-asymmetry proxy),
#      broken down by horizon, location, and season.
#   2. FITS three candidate post-hoc calibrations on top of the
#      (unchanged) point forecast and simulated quantiles, ALL on the
#      model's own log scale (recovered exactly from the output table,
#      since `value = exp(log-scale prediction)` for this transform):
#        - BIAS: a per-horizon additive shift of the median (closed
#          form: mean(log truth - log median) on a training fold),
#          propagated to every quantile so the whole distribution moves
#          with the debiased median;
#        - per-horizon multiplicative WIDTH scale (grid search
#          minimizing WIS around the median);
#        - per-location multiplicative WIDTH scale on top of that.
#      Each is validated LEAVE-ONE-SEASON-OUT within the two validation
#      seasons (fit on season A, apply+score on season B, and vice
#      versa) so the reported numbers are honest out-of-fold
#      performance, not the calibration re-marking its own homework.
#
#      RESULT: the bias correction does NOT generalize and is DROPPED.
#      The diagnosis shows a clear-looking per-horizon bias in-sample
#      (worsening under-prediction with horizon, -0.048 to -0.105
#      natural-scale), but broken down by season it is season 1:
#      +0.036 (slightly OVER-predicting), season 2: -0.2 (badly UNDER-
#      predicting) -- opposite signs. A single season's under/over-
#      prediction is season-severity noise (how bad that particular
#      flu season turned out to be), not a persistent model artifact,
#      so a bias term fit on one season actively hurts the other: OOF
#      mean WIS gets WORSE with any bias correction (0.2601 -> 0.2719
#      bias-only, -4.5%; see score.txt's ablation table). Per-horizon
#      width scaling adds nothing once location scaling is present
#      either (0.2574 with both vs 0.2566 location-only). The per-
#      LOCATION width scale alone is the only piece that actually
#      generalizes -- some regions (e.g. HHS 6/7/8/10) are reliably
#      over-wide in BOTH validation seasons, others (HHS 2/4, US
#      National) reliably under-wide, a stable pattern a fixed scalar
#      per location can safely exploit. FINAL FIX = per-location
#      multiplicative width scale only (delta=0, h_scale=1 everywhere).
#      The final calibration constants used for the hub submission are
#      refit on BOTH validation seasons combined (still never touching
#      the test seasons) and applied as frozen constants to all 5
#      seasons, the same pattern `experiments/simple-round/intervals`
#      and `round2-stack/submit.jl` already use for their own tuned
#      constants (T_DF, T_SCALE, pool weight).
#
# SCORED ON VALIDATION SEASONS (1, 2) ONLY throughout (diagnosis, LOSO
# fitting, and every number reported in score.txt); the full 5-season
# hub write at the end applies the validation-fitted constants
# mechanically, never scoring or selecting against the test seasons.
#
# LIGHT + ANALYTIC: CSV/DataFrames/Dates/Statistics/Random/
# LinearAlgebra/Distributions only, no Turing, no simulation beyond the
# round2-stack point-forecast machinery it reuses verbatim.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl [hub_path]
# writes score.txt alongside this file; if `hub_path` is given, also
# writes a full 5-season hub submission (model_id "nfidd-calib").

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

const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12
const WINDOW_WEEKS = 104
const MIN_SUPPORT = 5
const SMOOTH_WINDOW = 3
const VALIDATION_ONLY = (1, 2)
const MAX_TRAIN_SEASON_YEAR = 2016
const HUB_PATH_DEFAULT = joinpath(PKG_DIR, "scratch-hub")

# Winner's fixed design (round2-stack/score.txt: log+tstudent+pool(w=0.9)).
const TRANSFORM = :log
const BF_MODE = :additive
const BF_WINDOW = 8
const BF_STAT = :median
const T_DF = 10
const T_SCALE = 1.4
const POOL_W = 0.9

const MODEL_ID = "nfidd-calib"

# ---------------------------------------------------------------------
# Point-forecast machinery, verbatim from
# experiments/simple-round/round2-stack/generate.jl (see that file for
# full derivation docstrings) -- reproduced here rather than included
# because each experiments/simple-round/ script is self-contained.
# ---------------------------------------------------------------------

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

function simulate_paths(
    y::AbstractVector{Float64}, coef::Vector{Float64}, resid_sd::Float64,
    order::Int, horizons, npaths::Int;
    rng::Random.AbstractRNG, t_df::Int=T_DF, t_scale::Float64=T_SCALE,
)
    tdist = TDist(t_df)
    vscale = sqrt((t_df - 2) / t_df)
    innov_sd = resid_sd * vscale * t_scale

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
            innov = innov_sd * rand(rng, tdist)
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

"""
    build_forecast_table(seasons, versions_full, profile, backfill_profile;
        model_id) -> DataFrame

The round-2 stack winner's forecast table (transform=:log,
pool_w=`POOL_W`, Student-t(df=`T_DF`, scale=`T_SCALE`) intervals),
uncalibrated -- this experiment's BEFORE baseline and the input every
calibration stage below post-processes.
"""
function build_forecast_table(
    seasons, versions_full, profile::Dict{Int,Float64},
    backfill_profile::Dict{Tuple{String,Int},Float64}; model_id::String,
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
            fits = [fit_ar(ys[li], AR_ORDER) for li in 1:L]
            coefs = [f[1] for f in fits]
            Xs = [f[2] for f in fits]
            yresps = [f[3] for f in fits]

            anchor = fit_ar_pooled(ys, AR_ORDER)
            blended = [
                (1 - POOL_W) .* coefs[li] .+ POOL_W .* anchor for li in 1:L
            ]

            for (li, loc) in enumerate(LOCATIONS)
                coef = blended[li]
                resid_sd = resid_sd_for(Xs[li], yresps[li], coef, AR_ORDER)
                paths = simulate_paths(
                    ys[li], coef, resid_sd, AR_ORDER, HORIZONS, NPATHS;
                    rng=rng,
                )
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

# ---------------------------------------------------------------------
# Diagnosis: coverage at every nominal decile, and signed bias, broken
# down by horizon / location / season.
# ---------------------------------------------------------------------

const TASK_COLS = [:model_id, :location, :origin_date, :horizon,
                    :target_end_date]

"""
    coverage_by(forecast, truth, level; by=Symbol[]) -> DataFrame

Empirical coverage of the nominal `level` central interval (e.g.
`level=0.5` -> the `[0.25, 0.75]` quantile pair), optionally grouped by
`by` (extra columns already present on `forecast`, e.g. `[:horizon]` or
`[:season_num]`). Generalizes round2-stack's single-number `coverage`
helper to support the per-horizon/location/season breakdown this
experiment needs.
"""
function coverage_by(
    forecast::DataFrame, truth::DataFrame, level::Float64;
    by::Vector{Symbol}=Symbol[],
)
    a = (1 - level) / 2
    keep = vcat(TASK_COLS, filter(c -> !(c in TASK_COLS), by))
    lo = forecast[isapprox.(forecast.output_type_id, a; atol=1e-6), :]
    hi = forecast[isapprox.(forecast.output_type_id, 1 - a; atol=1e-6), :]
    lo_r = rename(lo[:, vcat(keep, [:value])], :value => :lo)
    hi_r = rename(hi[:, vcat(keep, [:value])], :value => :hi)
    joined = innerjoin(lo_r, hi_r, on=keep)
    joined = innerjoin(joined, truth, on=[:location, :target_end_date])
    joined.covered = (joined.lo .<= joined.value) .& (joined.value .<= joined.hi)
    if isempty(by)
        return DataFrame(
            nominal=[level], coverage=[mean(joined.covered)],
            n=[nrow(joined)],
        )
    end
    g = combine(groupby(joined, by), :covered => mean => :coverage,
        nrow => :n)
    g.nominal .= level
    sort!(g, by)
    return g
end

"""
    bias_table(forecast, truth; by=Symbol[]) -> DataFrame

Signed bias of the MEDIAN forecast (`output_type_id == 0.5`), on both
the natural scale (`bias_nat = median - truth`) and the model's own
`log` scale (`bias_log`, recovered exactly from the output value since
`value = exp(log-scale prediction)` for `TRANSFORM = :log` -- see
`to_log_scale`). Positive = over-prediction. Optionally grouped by `by`.
"""
function bias_table(forecast::DataFrame, truth::DataFrame; by::Vector{Symbol}=Symbol[])
    med = forecast[isapprox.(forecast.output_type_id, 0.5; atol=1e-6), :]
    joined = innerjoin(med, truth, on=[:location, :target_end_date],
        renamecols="" => "_truth")
    joined.bias_nat = joined.value .- joined.value_truth
    joined.bias_log = log.(max.(joined.value, EPS)) .-
                       log.(max.(joined.value_truth, EPS))
    cols = isempty(by) ? [] : by
    g = if isempty(by)
        DataFrame(
            mean_bias_nat=[mean(joined.bias_nat)],
            mean_bias_log=[mean(joined.bias_log)], n=[nrow(joined)],
        )
    else
        combine(groupby(joined, by),
            :bias_nat => mean => :mean_bias_nat,
            :bias_log => mean => :mean_bias_log, nrow => :n)
    end
    isempty(by) || sort!(g, by)
    return g
end

"""
    asymmetry_by(scored, by) -> DataFrame

Mean WIS over/under-prediction components (from `score_forecasts`),
grouped by `by` -- a pinball-asymmetry proxy: a model with
`mean_overprediction >> mean_underprediction` is systematically pricing
in more downside risk than upside (or vice versa), independent of the
signed bias of the median alone.
"""
function asymmetry_by(scored::DataFrame, by::Vector{Symbol})
    g = combine(groupby(scored, by),
        :overprediction => mean => :mean_over,
        :underprediction => mean => :mean_under, nrow => :n)
    sort!(g, by)
    return g
end

const NOMINAL_LEVELS = 0.1:0.1:0.9

function add_season_num!(df::DataFrame)
    df.season_num = [season_year(d) == 2015 ? 1 : 2 for d in df.origin_date]
    return df
end

function print_full_coverage(io, forecast, truth)
    println(io, "  nominal  empirical")
    for lvl in NOMINAL_LEVELS
        c = coverage_by(forecast, truth, lvl)
        println(io, "  $(Int(round(100*lvl; digits=0)))%      " *
                "$(round(c.coverage[1]; digits=3)) (n=$(c.n[1]))")
    end
end

function print_group_coverage(io, forecast, truth, groupcol; levels=NOMINAL_LEVELS)
    for lvl in levels
        g = coverage_by(forecast, truth, lvl; by=[groupcol])
        print(io, "  nominal $(Int(round(100*lvl)))%: ")
        println(io, join(
            ["$(row[groupcol])=$(round(row.coverage; digits=3))"
             for row in eachrow(g)], "  "))
    end
end

# ---------------------------------------------------------------------
# Fix: per-horizon bias correction + per-horizon/location width scaling,
# all applied on the model's own log scale.
# ---------------------------------------------------------------------

"""
    calibrate_forecast(forecast; delta, h_scale, loc_scale) -> DataFrame

Post-hoc calibration of an (already-built) forecast table, entirely on
the model's own log scale (recovered exactly from `value` since
`value = exp(x)` for `TRANSFORM = :log`): for each task, let `x_q =
log(value_q)` and `x_med = log(value at q=0.5)`. The calibrated
quantile is

    x_q' = (x_med + delta[h]) + (h_scale[h] * loc_scale[loc]) * (x_q - x_med)
    value_q' = exp(x_q')

i.e. shift the median by the horizon's bias correction, then scale its
deviation from that (new) median by the horizon x location width
factor. Missing dict entries default to 0.0 (bias) / 1.0 (scale) --
identity for that horizon/location.
"""
function calibrate_forecast(
    forecast::DataFrame; delta::Dict{Int,Float64},
    h_scale::Dict{Int,Float64}, loc_scale::Dict{String,Float64},
)
    med = forecast[isapprox.(forecast.output_type_id, 0.5; atol=1e-6),
                   vcat(TASK_COLS, [:value])]
    med = rename(med, :value => :median_value)
    joined = innerjoin(forecast, med, on=TASK_COLS)
    out = copy(joined)
    n = nrow(out)
    newval = Vector{Float64}(undef, n)
    for i in 1:n
        h = out.horizon[i]
        loc = out.location[i]
        x = log(max(out.value[i], EPS))
        xmed = log(max(out.median_value[i], EPS))
        d = get(delta, h, 0.0)
        m = get(h_scale, h, 1.0) * get(loc_scale, loc, 1.0)
        newval[i] = exp((xmed + d) + m * (x - xmed))
    end
    out.value = newval
    return select(out, vcat(TASK_COLS, [:target, :output_type,
                                         :output_type_id, :value]))
end

"""
    fit_bias(forecast, truth) -> Dict{Int,Float64}

Closed-form per-horizon bias correction: `delta[h] = mean(log(truth) -
log(median forecast))` over that horizon's tasks in `forecast`. Applying
this shift to the median (`calibrate_forecast`'s `delta`) makes the
median unbiased ON THIS SET, on the model's own log scale.
"""
function fit_bias(forecast::DataFrame, truth::DataFrame)
    med = forecast[isapprox.(forecast.output_type_id, 0.5; atol=1e-6), :]
    joined = innerjoin(med, truth, on=[:location, :target_end_date],
        renamecols="" => "_truth")
    delta = Dict{Int,Float64}()
    for h in HORIZONS
        sub = joined[joined.horizon .== h, :]
        delta[h] = mean(log.(max.(sub.value_truth, EPS)) .-
                         log.(max.(sub.value, EPS)))
    end
    return delta
end

const H_SCALE_GRID = 0.5:0.02:1.2
const LOC_SCALE_GRID = 0.7:0.02:1.3

"""
    fit_h_scale(forecast, truth, delta, grid) -> Dict{Int,Float64}

Per-horizon width multiplier: grid search over `grid`, holding `delta`
(bias) fixed, minimizing mean WIS on that horizon's tasks alone --
horizons don't interact through this transform, so each is optimized
independently on its own subset of `forecast`.
"""
function fit_h_scale(forecast::DataFrame, truth::DataFrame,
        delta::Dict{Int,Float64}, grid)
    best = Dict{Int,Float64}()
    for h in HORIZONS
        sub = forecast[forecast.horizon .== h, :]
        best_m, best_wis = 1.0, Inf
        for m in grid
            cal = calibrate_forecast(
                sub; delta=delta, h_scale=Dict(h => m),
                loc_scale=Dict{String,Float64}(),
            )
            scored = score_forecasts(cal, truth; scale=:natural)
            w = mean(scored.wis)
            if w < best_wis
                best_wis, best_m = w, m
            end
        end
        best[h] = best_m
    end
    return best
end

"""
    fit_loc_scale(forecast, truth, delta, h_scale, grid)
        -> Dict{String,Float64}

Per-location width multiplier ON TOP OF the per-horizon scale: grid
search over `grid`, holding `delta` and `h_scale` fixed, minimizing mean
WIS on that location's tasks alone (all horizons).
"""
function fit_loc_scale(forecast::DataFrame, truth::DataFrame,
        delta::Dict{Int,Float64}, h_scale::Dict{Int,Float64}, grid)
    best = Dict{String,Float64}()
    for loc in LOCATIONS
        sub = forecast[forecast.location .== loc, :]
        best_m, best_wis = 1.0, Inf
        for m in grid
            cal = calibrate_forecast(
                sub; delta=delta, h_scale=h_scale,
                loc_scale=Dict(loc => m),
            )
            scored = score_forecasts(cal, truth; scale=:natural)
            w = mean(scored.wis)
            if w < best_wis
                best_wis, best_m = w, m
            end
        end
        best[loc] = best_m
    end
    return best
end

const IDENTITY_DELTA = Dict(h => 0.0 for h in HORIZONS)
const IDENTITY_H_SCALE = Dict(h => 1.0 for h in HORIZONS)
const IDENTITY_LOC_SCALE = Dict(l => 1.0 for l in LOCATIONS)

"""
    fit_calibration(forecast, truth; use_bias, use_hscale, use_locscale)
        -> (delta, h_scale, loc_scale)

Coordinate-descent fit of whichever calibration stages are switched on
(bias closed-form, then horizon width grid with `delta` fixed, then
location width grid with `delta`/`h_scale` fixed); stages left off are
the identity (`delta=0`, `scale=1`). Used both for the three-way
ablation (which stage(s) actually generalize?) and, with the winning
combination, for the final frozen constants.
"""
function fit_calibration(
    forecast::DataFrame, truth::DataFrame;
    use_bias::Bool, use_hscale::Bool, use_locscale::Bool,
)
    delta = use_bias ? fit_bias(forecast, truth) : IDENTITY_DELTA
    h_scale = use_hscale ?
        fit_h_scale(forecast, truth, delta, H_SCALE_GRID) : IDENTITY_H_SCALE
    loc_scale = use_locscale ?
        fit_loc_scale(forecast, truth, delta, h_scale, LOC_SCALE_GRID) :
        IDENTITY_LOC_SCALE
    return delta, h_scale, loc_scale
end

"""
    loso_calibrate(before, truth; use_bias, use_hscale, use_locscale)
        -> DataFrame

Leave-one-season-out calibration: for each validation season, fit the
requested calibration stage(s) on the OTHER season and apply them to
this one, so every row in the returned table is calibrated by
parameters that never saw its own season -- honest out-of-fold
performance, not the calibration re-marking its own homework.
"""
function loso_calibrate(
    before::DataFrame, truth::DataFrame;
    use_bias::Bool, use_hscale::Bool, use_locscale::Bool,
)
    parts = DataFrame[]
    for test_season in VALIDATION_ONLY
        train_season = only(setdiff(VALIDATION_ONLY, (test_season,)))
        train_fc = before[before.season_num .== train_season, :]
        test_fc = before[before.season_num .== test_season, :]
        delta, h_scale, loc_scale = fit_calibration(
            train_fc, truth; use_bias=use_bias, use_hscale=use_hscale,
            use_locscale=use_locscale,
        )
        push!(parts, calibrate_forecast(
            test_fc; delta=delta, h_scale=h_scale, loc_scale=loc_scale,
        ))
    end
    return vcat(parts...)
end

fmt_dict(d) = join(["$(k)=$(round(v; digits=3))"
                     for (k, v) in sort(collect(d); by=first)], "  ")

function main()
    t0 = time()
    hub_path = length(ARGS) >= 1 ? ARGS[1] : nothing

    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")
    truth = load_oracle(HUB_PATH_DEFAULT)
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

    println("=== reproducing round2-stack winner on validation (1,2) ===")
    before = build_forecast_table(
        VALIDATION_ONLY, versions_full, profile, bf_profile;
        model_id="before",
    )
    add_season_num!(before)
    before_scored = score_forecasts(before, truth; scale=:natural)
    before_summary = wis_summary(before_scored)[1, :]
    println("before: mean_wis=$(round(before_summary.mean_wis; digits=4)) " *
            "sd_wis=$(round(before_summary.sd_wis; digits=4)) " *
            "($(round(time()-t0; digits=1))s)")

    # -------------------------------------------------------------
    # 1. DIAGNOSIS
    # -------------------------------------------------------------
    println("\n=== DIAGNOSIS (before calibration) ===")
    println("-- coverage at every nominal decile (pooled) --")
    print_full_coverage(stdout, before, truth)
    println("-- coverage by horizon (50%, 90%) --")
    print_group_coverage(stdout, before, truth, :horizon; levels=(0.5, 0.9))
    println("-- coverage by season (50%, 90%) --")
    print_group_coverage(stdout, before, truth, :season_num; levels=(0.5, 0.9))
    bias_h = bias_table(before, truth; by=[:horizon])
    bias_loc = bias_table(before, truth; by=[:location])
    bias_season = bias_table(before, truth; by=[:season_num])
    asym_h = asymmetry_by(before_scored, [:horizon])
    println("-- bias by horizon (median - truth) --")
    for row in eachrow(bias_h)
        println("  h=$(row.horizon): bias_nat=$(round(row.mean_bias_nat; digits=3)) " *
                "bias_log=$(round(row.mean_bias_log; digits=3)) (n=$(row.n))")
    end
    println("-- WIS over/under-prediction by horizon --")
    for row in eachrow(asym_h)
        println("  h=$(row.horizon): over=$(round(row.mean_over; digits=4)) " *
                "under=$(round(row.mean_under; digits=4))")
    end

    # -------------------------------------------------------------
    # 2. FIX -- three-way LOSO ablation: which calibration stage(s)
    #    actually generalize out-of-fold? (bias alone; bias+both
    #    widths; location width alone). See module docstring for why
    #    bias is expected to fail and gets dropped.
    # -------------------------------------------------------------
    println("\n=== FIX: leave-one-season-out ablation ===")
    variants = (
        (label="bias only", use_bias=true, use_hscale=false,
            use_locscale=false),
        (label="bias+hscale+locscale (naive full fix)", use_bias=true,
            use_hscale=true, use_locscale=true),
        (label="hscale+locscale (no bias)", use_bias=false,
            use_hscale=true, use_locscale=true),
        (label="locscale only (FINAL CHOICE)", use_bias=false,
            use_hscale=false, use_locscale=true),
    )
    ablation_rows = NamedTuple[]
    oof_final = DataFrame()
    for v in variants
        oof = loso_calibrate(
            before, truth; use_bias=v.use_bias, use_hscale=v.use_hscale,
            use_locscale=v.use_locscale,
        )
        scored = score_forecasts(oof, truth; scale=:natural)
        summ = wis_summary(scored)[1, :]
        cov50 = coverage_by(oof, truth, 0.5).coverage[1]
        cov90 = coverage_by(oof, truth, 0.9).coverage[1]
        println("  $(rpad(v.label, 40)) mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4)) cov50=" *
                "$(round(cov50; digits=3)) cov90=$(round(cov90; digits=3))")
        push!(ablation_rows, (
            label=v.label, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
            cov50=cov50, cov90=cov90,
        ))
        v.label == "locscale only (FINAL CHOICE)" && (oof_final = oof)
    end
    add_season_num!(oof_final)
    oof_summary = wis_summary(score_forecasts(oof_final, truth; scale=:natural))[1, :]

    println("\n-- coverage at every nominal decile (after, OOF, " *
            "locscale-only) --")
    print_full_coverage(stdout, oof_final, truth)
    println("-- coverage by horizon (50%, 90%), after OOF --")
    print_group_coverage(stdout, oof_final, truth, :horizon; levels=(0.5, 0.9))
    println("-- coverage by season (50%, 90%), after OOF --")
    print_group_coverage(stdout, oof_final, truth, :season_num; levels=(0.5, 0.9))
    bias_h_after = bias_table(oof_final, truth; by=[:horizon])
    println("-- bias by horizon, after OOF --")
    for row in eachrow(bias_h_after)
        println("  h=$(row.horizon): bias_nat=$(round(row.mean_bias_nat; digits=3)) " *
                "bias_log=$(round(row.mean_bias_log; digits=3)) (n=$(row.n))")
    end

    println("\nWIS before=$(round(before_summary.mean_wis; digits=4)) " *
            "after(OOF, locscale-only)=$(round(oof_summary.mean_wis; digits=4)) " *
            "delta=$(round(before_summary.mean_wis - oof_summary.mean_wis; digits=4))")

    cov50_before = coverage_by(before, truth, 0.5).coverage[1]
    cov90_before = coverage_by(before, truth, 0.9).coverage[1]
    cov50_after = coverage_by(oof_final, truth, 0.5).coverage[1]
    cov90_after = coverage_by(oof_final, truth, 0.9).coverage[1]

    # -------------------------------------------------------------
    # 3. Final calibration constants: refit the WINNING stage
    #    (location width only) on BOTH validation seasons combined --
    #    frozen constants for the actual hub submission.
    # -------------------------------------------------------------
    println("\n=== final calibration (fit on both validation seasons) ===")
    final_delta, final_h_scale, final_loc_scale = fit_calibration(
        before, truth; use_bias=false, use_hscale=false, use_locscale=true,
    )
    println("delta (bias):    $(fmt_dict(final_delta)) (identity -- dropped)")
    println("h_scale (width): $(fmt_dict(final_h_scale)) (identity -- dropped)")
    println("loc_scale (width): $(fmt_dict(final_loc_scale))")

    final_cal = calibrate_forecast(
        before; delta=final_delta, h_scale=final_h_scale,
        loc_scale=final_loc_scale,
    )
    final_scored = score_forecasts(final_cal, truth; scale=:natural)
    final_summary = wis_summary(final_scored)[1, :]
    cov50_final_insample = coverage_by(final_cal, truth, 0.5).coverage[1]
    cov90_final_insample = coverage_by(final_cal, truth, 0.9).coverage[1]
    println("in-sample (both seasons, same data fit on -- NOT the " *
            "selection number, see OOF above) mean_wis=" *
            "$(round(final_summary.mean_wis; digits=4)) cov50=" *
            "$(round(cov50_final_insample; digits=3)) cov90=" *
            "$(round(cov90_final_insample; digits=3))")

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "calibration (coverage + bias fix) on top of the " *
                "round2-stack winner -- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time()-t0; digits=1))s")
        println(io)
        println(io, "base model: round2-stack log+tstudent+pool(w=0.9) " *
                "(round2-stack/score.txt): mean_wis=0.2601 cov50=0.565 " *
                "cov90=0.943")
        println(io, "reproduced here: mean_wis=" *
                "$(round(before_summary.mean_wis; digits=4)) sd_wis=" *
                "$(round(before_summary.sd_wis; digits=4)) cov50=" *
                "$(round(cov50_before; digits=3)) cov90=" *
                "$(round(cov90_before; digits=3))")
        println(io)

        println(io, "=== DIAGNOSIS (before) ===")
        println(io, "-- coverage at every nominal decile (pooled) --")
        print_full_coverage(io, before, truth)
        println(io)
        println(io, "-- coverage by horizon --")
        print_group_coverage(io, before, truth, :horizon)
        println(io)
        println(io, "-- coverage by location (50%, 90%) --")
        print_group_coverage(io, before, truth, :location; levels=(0.5, 0.9))
        println(io)
        println(io, "-- coverage by season (50%, 90%) --")
        print_group_coverage(io, before, truth, :season_num; levels=(0.5, 0.9))
        println(io)
        println(io, "-- bias (median - truth) by horizon --")
        for row in eachrow(bias_h)
            println(io, "  h=$(row.horizon): bias_nat=" *
                    "$(round(row.mean_bias_nat; digits=3)) bias_log=" *
                    "$(round(row.mean_bias_log; digits=3)) (n=$(row.n))")
        end
        println(io, "-- bias by location --")
        for row in eachrow(bias_loc)
            println(io, "  $(rpad(row.location, 16)) bias_nat=" *
                    "$(round(row.mean_bias_nat; digits=3)) bias_log=" *
                    "$(round(row.mean_bias_log; digits=3)) (n=$(row.n))")
        end
        println(io, "-- bias by season --")
        for row in eachrow(bias_season)
            println(io, "  season $(row.season_num): bias_nat=" *
                    "$(round(row.mean_bias_nat; digits=3)) bias_log=" *
                    "$(round(row.mean_bias_log; digits=3)) (n=$(row.n))")
        end
        println(io, "-- WIS over/under-prediction by horizon (pinball " *
                "asymmetry proxy) --")
        for row in eachrow(asym_h)
            println(io, "  h=$(row.horizon): over=" *
                    "$(round(row.mean_over; digits=4)) under=" *
                    "$(round(row.mean_under; digits=4))")
        end
        println(io)

        println(io, "=== FIX: leave-one-season-out ablation (honest, " *
                "out-of-fold WIS/coverage for each candidate stage) ===")
        for r in ablation_rows
            println(io, "  $(rpad(r.label, 42)) mean_wis=" *
                    "$(round(r.mean_wis; digits=4)) sd_wis=" *
                    "$(round(r.sd_wis; digits=4)) cov50=" *
                    "$(round(r.cov50; digits=3)) cov90=" *
                    "$(round(r.cov90; digits=3))")
        end
        println(io, "-> BIAS DOES NOT GENERALIZE: any variant including it " *
                "scores worse OOF than the uncalibrated baseline (0.2601). " *
                "Season-level bias flips sign (season 1: +0.036, season 2: " *
                "-0.2, see bias-by-season above) -- that's season-severity " *
                "noise, not a persistent artifact a fixed constant can " *
                "correct for. Per-horizon width scaling adds nothing once " *
                "location scaling is present. FINAL FIX = per-location " *
                "width scale only.")
        println(io)
        println(io, "-- detail for the final choice (locscale only) --")
        println(io, "coverage at every nominal decile (after, OOF):")
        print_full_coverage(io, oof_final, truth)
        println(io)
        println(io, "coverage by horizon (50%, 90%), after OOF:")
        print_group_coverage(io, oof_final, truth, :horizon; levels=(0.5, 0.9))
        println(io)
        println(io, "coverage by season (50%, 90%), after OOF:")
        print_group_coverage(io, oof_final, truth, :season_num; levels=(0.5, 0.9))
        println(io)
        println(io, "bias by horizon, after OOF:")
        for row in eachrow(bias_h_after)
            println(io, "  h=$(row.horizon): bias_nat=" *
                    "$(round(row.mean_bias_nat; digits=3)) bias_log=" *
                    "$(round(row.mean_bias_log; digits=3)) (n=$(row.n))")
        end
        println(io)
        println(io, "WIS: before=$(round(before_summary.mean_wis; digits=4)) " *
                "after(OOF)=$(round(oof_summary.mean_wis; digits=4)) delta=" *
                "$(round(before_summary.mean_wis - oof_summary.mean_wis; digits=4)) " *
                "($(round(100*(before_summary.mean_wis - oof_summary.mean_wis)/before_summary.mean_wis; digits=2))%)")
        println(io, "coverage: cov50 $(round(cov50_before; digits=3)) -> " *
                "$(round(cov50_after; digits=3)) (nominal 0.50); cov90 " *
                "$(round(cov90_before; digits=3)) -> " *
                "$(round(cov90_after; digits=3)) (nominal 0.90)")
        println(io)

        println(io, "=== final calibration constants (fit on BOTH " *
                "validation seasons -- frozen for the hub submission) ===")
        println(io, "delta (log-scale bias, by horizon): " *
                "$(fmt_dict(final_delta)) (identity -- dropped, see " *
                "ablation above)")
        println(io, "h_scale (width multiplier, by horizon): " *
                "$(fmt_dict(final_h_scale)) (identity -- dropped, adds " *
                "nothing once loc_scale is present)")
        println(io, "loc_scale (width multiplier, by location): " *
                "$(fmt_dict(final_loc_scale))")
        println(io, "in-sample check (both seasons, NOT the honest " *
                "number -- see OOF above): mean_wis=" *
                "$(round(final_summary.mean_wis; digits=4)) cov50=" *
                "$(round(cov50_final_insample; digits=3)) cov90=" *
                "$(round(cov90_final_insample; digits=3))")
    end

    # -------------------------------------------------------------
    # Full 5-season hub-format output, calibration constants FROZEN
    # from validation seasons only.
    # -------------------------------------------------------------
    println("\n=== building full 5-season forecast for hub submission ===")
    full_forecast = build_forecast_table(
        (1, 2, 3, 4, 5), versions_full, profile, bf_profile;
        model_id=MODEL_ID,
    )
    full_calibrated = calibrate_forecast(
        full_forecast; delta=final_delta, h_scale=final_h_scale,
        loc_scale=final_loc_scale,
    )
    full_calibrated.model_id .= MODEL_ID
    n_origins = length(unique(full_calibrated.origin_date))
    println("built $(nrow(full_calibrated)) rows across $(n_origins) " *
            "origin date(s)")

    if hub_path !== nothing
        write_submission(full_calibrated, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="nfidd", model_abbr="calib", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end

    dt = round(time() - t0; digits=1)
    println("\nwrote score.txt in $(dt)s total")
    return (
        before_summary=before_summary, oof_summary=oof_summary,
        final_delta=final_delta, final_h_scale=final_h_scale,
        final_loc_scale=final_loc_scale, forecast=full_calibrated,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
