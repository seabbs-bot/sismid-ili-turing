#!/usr/bin/env julia
# generate.jl -- FEATURE-BASED RIDGE REGRESSION, simple-round FEATURE
# family. Unlike every other simple-round driver (all iterated AR(p),
# one lag structure applied recursively h times to reach horizon h),
# this fits a SEPARATE ridge regression PER HORIZON that regresses the
# h-step-ahead value DIRECTLY on a richer feature set:
#
#   - lags 1..LAG_ORDER of the (backfill-corrected) series
#   - a short-window trend/slope
#   - the pooled (cross-location) climatology value AT THE TARGET WEEK
#     (i.e. week-of-season origin_date + 7h falls in), adapted from
#     seasoncombo/generate.jl's `build_seasonal_profile` but
#     recomputed PER SPLIT, capped at that split's own forecast
#     origin -- safe to run across all 5 seasons (see docstring below)
#   - the current season's amplitude-so-far (peak-to-date relative to
#     the location's own running median)
#   - the national ("US National") level at the origin
#   - sin/cos encoding of the origin's own week-of-season (phase)
#
# All features standardised before a per-(location, horizon, split)
# ridge fit (penalty excludes the intercept); this is what makes
# "richer than AR" feasible on ~90 training rows per split without the
# naive-Fourier overfitting failure mode documented in
# experiments/simple-round/season/score.txt -- ridge shrinks any
# uninformative feature toward 0 itself. Predictive quantiles are
# Normal(point, resid_sd * SCALE) -- no path simulation needed (this is
# direct multi-horizon regression, not an iterated recursion), which is
# also why this is fast despite the richer feature set: no 1000-path
# loop, just one small linear solve per (location, horizon, split).
#
# Both the ridge penalty LAMBDA and a residual-inflation SCALE (every
# other simple-round driver that checked, `experiments/simple-round/
# intervals/score.txt`, found the raw in-sample resid_sd under-covers
# needing ~1.4-1.5x) are cross-validated by sweeping a joint grid,
# scored on VALIDATION SEASONS (1, 2) ONLY against the hub oracle
# (docs/contracts.md experimental integrity) -- see score.txt for the
# full grid and the feature-importance ranking.
#
# Deliberately LIGHT + ANALYTIC (no Turing/Mooncake/Pathfinder), like
# every other simple-round script: CSV/DataFrames/Dates/Statistics/
# LinearAlgebra/Distributions/ScoringRules only. `Distributions` (for
# `quantile(Normal(...), q)`) is already used this way elsewhere in
# this family (experiments/simple-round/intervals,
# experiments/simple-round/round2-stack) without pulling in Turing.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl [hub_path]
# With no `hub_path`, runs the validation sweep, writes score.txt, and
# stops (matches experiments/simple-round/backfill's convention).
# With `hub_path`, additionally builds the FULL 5-season (1-2
# validation + 3-5 test) hub submission at the winning (lambda, scale)
# -- each split's climatology and ridge fit are still capped at that
# split's own forecast origin, so covering the test seasons at
# generation time never trains on or tunes against them; the
# (lambda, scale) choice itself was locked on the validation seasons
# only (score.txt).

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

const MODEL_ID = "seabbs_bot-features"
const TRANSFORM = :fourthroot
const SEED = 20260717
const DMAX = 12
const WINDOW_WEEKS = 104
const DELAY_CUTOFF = 8      # backfill correction, matches ar6bf/season
const BF_MIN_SUPPORT = 5
const SEASON_PERIOD = 52
const CLIM_MIN_SUPPORT = 3  # min pooled obs per week-of-season bin
const CLIM_SMOOTH_WINDOW = 5
const LAG_ORDER = 6
const TREND_WINDOW = 4
const VALIDATION_ONLY = (1, 2)
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")

const FEATURE_NAMES = vcat(
    ["lag$(i)" for i in 1:LAG_ORDER],
    ["trend", "clim_target", "amp_so_far", "national_level",
     "woy_sin", "woy_cos"],
)
const NFEAT = length(FEATURE_NAMES)

# ---------------------------------------------------------------------
# Backfill correction (identical to seabbs_bot-ar6bf / season / ar6bf's
# other family reuses -- additive, per-location, median)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile on the `transform` scale.
Identical to seabbs_bot-ar6bf / experiments/simple-round/season's
function of the same name. `versions` must already be filtered by the
caller to the training set only (no test seasons).
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
matching `profile` entry exists.
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
# Pooled climatology, per split (adapted from seasoncombo/generate.jl's
# `build_seasonal_profile`; the difference is that seasoncombo builds
# ONE profile from a fixed pre-2017 cutoff and reuses it everywhere
# (fine there -- that file is validation-only, never used to cover the
# test seasons). Here the profile is instead capped at each split's OWN
# forecast origin, so it can be safely recomputed for TEST_SEASONS
# splits too without leaking anything -- same discipline as
# experiments/simple-round/season's `build_climatology`.)
# ---------------------------------------------------------------------

"""
    build_pooled_climatology(hist, forecast_origin;
        transform=TRANSFORM, min_support=CLIM_MIN_SUPPORT,
        smooth_window=CLIM_SMOOTH_WINDOW, period=SEASON_PERIOD)
        -> Vector{Float64}

Pooled (cross-location) week-of-season climatology curve, length
`period`, on the `transform` scale. Built ONLY from `hist` rows
strictly before `forecast_origin` (no leakage of the split's own or
future observations). Each location's history is first centred on
that location's OWN mean over the restricted history (so absolute-
level differences across locations, e.g. Region 6 vs Region 8, don't
bias the shared shape), then the centred deviations are pooled across
ALL locations and binned circularly by week-of-season (bins with fewer
than `min_support` pooled observations fall back to 0, i.e. no
adjustment), smoothed with a `smooth_window`-wide circular moving
average, and re-centred to zero mean across the cycle so adding it
never shifts a location's overall level -- only its within-year shape.
Indexed as `curve[mod1(week_of_season(date), period)]`.
"""
function build_pooled_climatology(
    hist::DataFrame, forecast_origin::Date;
    transform::Symbol=TRANSFORM, min_support::Int=CLIM_MIN_SUPPORT,
    smooth_window::Int=CLIM_SMOOTH_WINDOW, period::Int=SEASON_PERIOD,
)
    h = hist[hist.origin_date .< forecast_origin, :]
    isempty(h) && return zeros(period)
    x = to_scale.(h.wili, transform)
    locs = h.location
    woys = [mod1(week_of_season(d), period) for d in h.origin_date]

    levels = Dict(loc => mean(x[locs .== loc]) for loc in unique(locs))
    dev = [x[i] - levels[locs[i]] for i in eachindex(x)]

    raw_bins = [Float64[] for _ in 1:period]
    for i in eachindex(dev)
        push!(raw_bins[woys[i]], dev[i])
    end
    means = [length(v) >= min_support ? mean(v) : 0.0 for v in raw_bins]

    half = div(smooth_window - 1, 2)
    smoothed = similar(means)
    for w in 1:period
        idxs = [mod1(w + off, period) for off in (-half):half]
        smoothed[w] = mean(means[idxs])
    end
    smoothed .-= mean(smoothed)
    return smoothed
end

# ---------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------

"""
    amplitude_so_far(y, season, t) -> Float64

Current-season peak-to-date, relative to the location's own running
median: `maximum(y[i] for i in 1:t if season[i] == season[t]) -
median(y[1:t])`. Uses only `y[1:t]` (no look-ahead).
"""
function amplitude_so_far(
    y::AbstractVector{Float64}, season::AbstractVector{Int}, t::Int,
)
    sidx = season[t]
    season_vals = [y[i] for i in 1:t if season[i] == sidx]
    baseline = median(@view y[1:t])
    return maximum(season_vals) - baseline
end

"""
    build_feature_row(y, y_nat, woy, season, dates, clim, t, h;
                       lag_order, trend_window, period) -> Vector{Float64}

One row of `FEATURE_NAMES`, built from data known at time `t` (index
into `y`/`dates`), for a target `h` weeks ahead. Used both to build
TRAINING rows (`t + h <= length(y)`, target `y[t+h]` known) and the
final prediction row (`t` = the split's last index, i.e. the forecast
origin; the target date `dates[t] + 7h` days is then in the future --
only `week_of_season` of that date is needed, not the value itself).
"""
function build_feature_row(
    y::AbstractVector{Float64}, y_nat::AbstractVector{Float64},
    woy::AbstractVector{Int}, season::AbstractVector{Int},
    dates::AbstractVector{Date}, clim::Vector{Float64}, t::Int, h::Int;
    lag_order::Int=LAG_ORDER, trend_window::Int=TREND_WINDOW,
    period::Int=SEASON_PERIOD,
)
    f = Vector{Float64}(undef, lag_order + 6)
    for i in 1:lag_order
        f[i] = y[t - i + 1]
    end
    f[lag_order + 1] = (y[t] - y[t - trend_window + 1]) / (trend_window - 1)
    target_date = dates[t] + Day(7 * h)
    target_woy = mod1(week_of_season(target_date), period)
    f[lag_order + 2] = clim[target_woy]
    f[lag_order + 3] = amplitude_so_far(y, season, t)
    f[lag_order + 4] = y_nat[t]
    phase = 2 * pi * woy[t] / period
    f[lag_order + 5] = sin(phase)
    f[lag_order + 6] = cos(phase)
    return f
end

"""
    build_xy(y, y_nat, woy, season, dates, clim, h; lag_order,
             trend_window) -> (X, ytrain)

Supervised design matrix for one location/horizon/split: one row per
valid origin index `t` (`max(lag_order, trend_window):(T - h)`),
target `y[t + h]`.
"""
function build_xy(
    y::AbstractVector{Float64}, y_nat::AbstractVector{Float64},
    woy::AbstractVector{Int}, season::AbstractVector{Int},
    dates::AbstractVector{Date}, clim::Vector{Float64}, h::Int;
    lag_order::Int=LAG_ORDER, trend_window::Int=TREND_WINDOW,
)
    T = length(y)
    start_t = max(lag_order, trend_window)
    ts = start_t:(T - h)
    n = length(ts)
    n >= 2 * NFEAT ||
        error("too few training rows for ridge: n=$n, features=$NFEAT")
    X = Matrix{Float64}(undef, n, NFEAT)
    ytrain = Vector{Float64}(undef, n)
    for (row, t) in enumerate(ts)
        X[row, :] = build_feature_row(
            y, y_nat, woy, season, dates, clim, t, h;
            lag_order=lag_order, trend_window=trend_window,
        )
        ytrain[row] = y[t + h]
    end
    return X, ytrain
end

# ---------------------------------------------------------------------
# Ridge regression (standardised features, unpenalised intercept)
# ---------------------------------------------------------------------

"""
    fit_ridge(X, y, lambda) -> (coef, mu, sd, resid_sd)

Ridge fit of `y` on `X` (n x p): columns of `X` are standardised
(mean 0, sd 1) before fitting so a single `lambda` penalises every
feature comparably regardless of its raw scale; `coef[1]` is the
(unpenalised) intercept on the standardised scale, `coef[2:end]` the
`p` standardised-feature coefficients. `resid_sd` is the in-sample
residual standard deviation (dof-adjusted for `p + 1` fitted
coefficients).
"""
function fit_ridge(X::Matrix{Float64}, y::Vector{Float64}, lambda::Float64)
    n, p = size(X)
    mu = vec(mean(X; dims=1))
    sd = vec(std(X; dims=1))
    sd[sd .< 1e-8] .= 1.0
    Z = (X .- mu') ./ sd'
    Xfull = hcat(ones(n), Z)
    penalty = Diagonal(vcat(0.0, fill(lambda, p)))
    coef = (Xfull' * Xfull + penalty) \ (Xfull' * y)
    resid = y - Xfull * coef
    dof = max(n - (p + 1), 1)
    resid_sd = sqrt(sum(abs2, resid) / dof)
    return (coef=coef, mu=mu, sd=sd, resid_sd=resid_sd)
end

"""Predict one new row (raw feature scale) from a `fit_ridge` model."""
function predict_ridge(model, xnew::Vector{Float64})
    z = (xnew .- model.mu) ./ model.sd
    return dot(vcat(1.0, z), model.coef)
end

# ---------------------------------------------------------------------
# Forecast table builder
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, bf_profile, versions_full, hist;
        lambda, scale, model_id, importance=nothing) -> DataFrame

Fit and forecast the feature-ridge model for every cross-validation
split of every season in `seasons`. Training discipline as every other
simple-round driver: `build_model_data` caps each split at its own
forecast origin and `window_weeks=WINDOW_WEEKS`; the pooled
climatology is separately recomputed per split, also capped at that
split's own forecast origin (`build_pooled_climatology`). `season` in
`TEST_SEASONS` is only ever fetched here with `allow_test_season=true`
when the CALLER passes such a season in -- this file's own `main` only
does so for the final hub-format write, never during the validation
sweep.

If `importance` is a `Dict{String,Vector{Float64}}` (keyed by
`FEATURE_NAMES`), the absolute standardised ridge coefficient of every
(location, horizon, split) fit is appended to it (diagnostic only; not
used to build `value` predictions).
"""
function build_forecast_table(
    seasons, bf_profile, versions_full, hist;
    lambda::Float64, scale::Float64, model_id::String,
    importance::Union{Nothing,Dict{String,Vector{Float64}}}=nothing,
)
    rows = DataFrame(
        model_id=String[], location=String[], origin_date=Date[],
        horizon=Int[], target_end_date=Date[], target=String[],
        output_type=String[], output_type_id=Float64[], value=Float64[],
    )
    national_li = findfirst(==("US National"), LOCATIONS)
    for season in seasons
        splits = training_splits(
            season; allow_test_season=(season in TEST_SEASONS),
        )
        for split in splits
            data = build_model_data(
                split; Dmax=DMAX, transform=TRANSFORM,
                window_weeks=WINDOW_WEEKS, versions=versions_full,
            )
            apply_backfill_correction!(data, bf_profile)
            origin = data.origin_date
            clim = build_pooled_climatology(hist, origin)
            y_nat = Float64.(data.Y[:, national_li])
            for (li, loc) in enumerate(LOCATIONS)
                y = Float64.(data.Y[:, li])
                for h in HORIZONS
                    X, ytrain = build_xy(
                        y, y_nat, data.woy, data.season, data.dates,
                        clim, h,
                    )
                    model = fit_ridge(X, ytrain, lambda)
                    if importance !== nothing
                        for (i, name) in enumerate(FEATURE_NAMES)
                            push!(
                                importance[name], abs(model.coef[i + 1]),
                            )
                        end
                    end
                    xnew = build_feature_row(
                        y, y_nat, data.woy, data.season, data.dates,
                        clim, data.T, h,
                    )
                    point = predict_ridge(model, xnew)
                    target_end = origin + Day(7 * h)
                    for q in QUANTILE_LEVELS
                        z = quantile(Normal(), q)
                        qval = point + z * model.resid_sd * scale
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
# Sweep: ridge lambda x residual-inflation scale, validation only
# ---------------------------------------------------------------------

const LAMBDAS = [1.0, 3.0, 10.0, 30.0, 100.0]
const SCALES = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

function main()
    t0 = time()
    hub_path = length(ARGS) >= 1 ? ARGS[1] : nothing

    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    bf_profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=DELAY_CUTOFF,
        min_support=BF_MIN_SUPPORT,
    )
    hist = load_series("flu_data_hhs")
    truth = load_oracle(HUB_PATH)

    println("baseline reference: seasoncombo-core (round-1 winner) " *
            "= 0.2781 mean WIS, validation seasons")

    results = NamedTuple[]
    for lambda in LAMBDAS, scale in SCALES
        forecast = build_forecast_table(
            VALIDATION_ONLY, bf_profile, versions_full, hist;
            lambda=lambda, scale=scale, model_id="sweep-tmp",
        )
        scored = score_forecasts(forecast, truth; scale=:natural)
        summ = wis_summary(scored)
        push!(results, (
            lambda=lambda, scale=scale,
            mean_wis=summ.mean_wis[1], sd_wis=summ.sd_wis[1],
        ))
        println("lambda=$lambda scale=$scale -> " *
                "mean_wis=$(round(summ.mean_wis[1]; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis[1]; digits=4)) " *
                "($(round(time() - t0; digits=1))s elapsed)")
    end
    sort!(results; by=r -> r.mean_wis)
    best = results[1]
    println("\nbest: lambda=$(best.lambda) scale=$(best.scale) " *
            "mean_wis=$(round(best.mean_wis; digits=4)) " *
            "sd_wis=$(round(best.sd_wis; digits=4))")

    # Re-run the winner to get its scored table (breakdown) and
    # feature-importance diagnostics.
    importance = Dict(name => Float64[] for name in FEATURE_NAMES)
    best_forecast = build_forecast_table(
        VALIDATION_ONLY, bf_profile, versions_full, hist;
        lambda=best.lambda, scale=best.scale, model_id=MODEL_ID,
        importance=importance,
    )
    best_scored = score_forecasts(best_forecast, truth; scale=:natural)
    best_summ = wis_summary(best_scored)

    by_loc = combine(groupby(best_scored, :location),
        :wis => mean => :mean_wis, nrow => :n)
    sort!(by_loc, :mean_wis)

    by_h = combine(groupby(best_scored, :horizon),
        :wis => mean => :mean_wis, nrow => :n)
    sort!(by_h, :horizon)

    best_scored.season_year = season_year.(best_scored.origin_date)
    by_season = combine(groupby(best_scored, :season_year),
        :wis => mean => :mean_wis, nrow => :n)
    sort!(by_season, :season_year)

    importance_ranked = sort(
        [(name, mean(vals)) for (name, vals) in importance];
        by=x -> -x[2],
    )

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "simple-round / features (FEATURE-BASED RIDGE " *
                     "REGRESSION family)")
        println(io, "Scored on VALIDATION seasons only (1, 2), " *
                     "natural scale, per docs/contracts.md " *
                     "experimental integrity.")
        println(io, "runtime so far: " *
                     "$(round(time() - t0; digits=1))s")
        println(io)
        println(io, "Baselines quoted from other families' own " *
                     "reproductions of the same validation splits:")
        println(io, "  ar6-baseline (nfidd-ar6, no season/backfill): " *
                     "0.3684")
        println(io, "  climatology-backfill (per-location, " *
                     "experiments/simple-round/season): 0.3004")
        println(io, "  seasoncombo-core (pooled-season + AR6 + " *
                     "backfill, ROUND-1 WINNER, the target to beat): " *
                     "0.2781")
        println(io)
        println(io, "Design: per-horizon ridge regression, DIRECT " *
                     "(not iterated) h-step-ahead fit -- see " *
                     "generate.jl header for the full feature list " *
                     "and rationale.")
        println(io)
        println(io, "Sweep: lambda (ridge penalty) x scale " *
                     "(residual-inflation multiplier on the Normal " *
                     "predictive quantiles), sorted by mean_wis:")
        println(io, rpad("lambda", 10) * rpad("scale", 8) *
                     rpad("mean_wis", 12) * "sd_wis")
        for r in results
            println(io,
                rpad(string(r.lambda), 10) * rpad(string(r.scale), 8) *
                rpad(string(round(r.mean_wis; digits=4)), 12) *
                string(round(r.sd_wis; digits=4)),
            )
        end
        println(io)
        println(io, "=== best variant ===")
        println(io, "lambda=$(best.lambda) scale=$(best.scale)")
        println(io, "mean_wis=$(round(best_summ.mean_wis[1]; digits=4)) " *
                     "sd_wis=$(round(best_summ.sd_wis[1]; digits=4)) " *
                     "n_tasks=$(best_summ.n_tasks[1])")
        vs_target = 0.2781 - best_summ.mean_wis[1]
        vs_pct = 100 * vs_target / 0.2781
        println(io, "vs seasoncombo-core (0.2781): " *
                     "$(round(vs_target; digits=4)) " *
                     "($(round(vs_pct; digits=2))%)")
        println(io)
        println(io, "-- breakdown by location (best variant) --")
        for r in eachrow(by_loc)
            println(io, "$(r.location)\tmean_wis=" *
                         "$(round(r.mean_wis; digits=4))\tn=$(r.n)")
        end
        println(io)
        println(io, "-- breakdown by horizon (best variant) --")
        for r in eachrow(by_h)
            println(io, "h=$(r.horizon)\tmean_wis=" *
                         "$(round(r.mean_wis; digits=4))\tn=$(r.n)")
        end
        println(io)
        println(io, "-- breakdown by season (best variant) --")
        for r in eachrow(by_season)
            println(io, "season $(r.season_year)\tmean_wis=" *
                         "$(round(r.mean_wis; digits=4))\tn=$(r.n)")
        end
        println(io)
        println(io, "-- feature importance (best variant): mean |" *
                     "standardised ridge coefficient| across every " *
                     "(location, horizon, split) fit, so magnitudes " *
                     "are directly comparable across features --")
        for (name, val) in importance_ranked
            println(io, rpad(name, 18) *
                         string(round(val; digits=4)))
        end
        println(io)
        println(io, "total time: $(round(time() - t0; digits=1))s")
    end
    println("wrote $(joinpath(HERE, "score.txt"))")

    if hub_path !== nothing
        forecast_full = build_forecast_table(
            (1, 2, 3, 4, 5), bf_profile, versions_full, hist;
            lambda=best.lambda, scale=best.scale, model_id=MODEL_ID,
        )
        dt = round(time() - t0; digits=2)
        n_origins = length(unique(forecast_full.origin_date))
        println("built $(nrow(forecast_full)) rows across " *
                "$(n_origins) origin date(s) in $(dt)s")
        write_submission(forecast_full, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="seabbs_bot", model_abbr="features", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return best_forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
