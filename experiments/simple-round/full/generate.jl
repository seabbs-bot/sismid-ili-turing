#!/usr/bin/env julia
# generate.jl -- FULL analytic model, simple-round: SEASONALITY +
# BACKFILL + PARTIAL POOLING, all three stacked.
#
# Each lever is taken from the best-scoring variant found by the
# corresponding single-lever experiment in this same
# experiments/simple-round/ tree, and applied together on top of a
# per-location AR(order) model (order swept over {6, 8, 12}, the
# ar-order sweep's candidate range):
#
#   1. SEASONALITY (experiments/simple-round/seasonpool/generate.jl):
#      one POOLED week-of-season shape, a `N_HARMONICS`-harmonic
#      Fourier curve fit once across all 11 locations' fourth-root
#      series (centred per-location first) from PRE-2015 history
#      (season_year <= 2014) -- deliberately not a per-location Fourier
#      fit, which overfits (nfidd-ar6 + per-location Fourier(3) scored
#      0.412, worse than plain AR(6)). Per split/location, a 2-parameter
#      (intercept, amplitude) regression adapts the pooled shape to
#      that location's own level, using only that split's own window.
#   2. BACKFILL correction (experiments/simple-round/backfill/
#      generate.jl sweep): the best-scoring variant found there --
#      MULTIPLICATIVE, per-location, MEDIAN revision, window 6
#      (mean_wis 0.3586 on AR(6) alone, vs 0.359 for ar6bf's original
#      additive/window-8 choice) -- applied to the vintage series
#      before anything else, exactly as in that sweep.
#   3. Per-location AR(order) on the DESEASONALISED residual (after
#      backfill), with PARTIAL POOLING of the AR coefficients toward a
#      single :fullpool anchor (one OLS fit across all 11 locations'
#      residual design rows stacked together) at weight 0.5 -- the best
#      pooling variant found by experiments/simple-round/pool/
#      generate.jl's sweep (mean_wis 0.3643 vs 0.3684 unpooled, on
#      AR(6), no season/backfill).
#
# Forecast = per-location seasonal term at the (known) future week-of-
# season + simulated pooled-AR residual paths, backfill baked into the
# nowcast the AR is fit on. Only the AR component propagates simulated
# uncertainty forward, as in seasonpool.
#
# This script both (a) sweeps AR order in {6, 8, 12} for the full
# (all-three) model, and (b) runs a one-at-a-time ablation at the best
# order found in (a): full model, and full-minus-each-component, so we
# can see which lever is carrying the weight when all three are
# stacked together (a component's solo score, e.g. seasonpool's 0.3049,
# does not by itself say how much it still contributes once the other
# two are also present).
#
# Scope: VALIDATION SEASONS (1, 2) ONLY, scored against the local hub
# clone's oracle (docs/contracts.md experimental integrity) -- a tuning
# experiment, not a submission driver. The backfill profile is built
# only from origin dates with season_year <= 2016 (matches ar6bf /
# the backfill sweep); the pooled seasonal shape only from season_year
# <= 2014 (pre-2015 history, disjoint from both validation seasons).
#
# LIGHT + ANALYTIC: CSV/DataFrames/Statistics/LinearAlgebra only, no
# Turing.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl
# writes score.txt alongside this file; does not write a hub
# submission (no hub_path argument -- exploratory, not a
# `submissions/` candidate).

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
const AR_ORDERS = (6, 8, 12)
const N_HARMONICS = 3
const SEASON_PERIOD = 52.0
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12
const BACKFILL_WINDOW = 6        # best window from the backfill sweep
const MIN_SUPPORT = 5
const CLIMATOLOGY_YEAR = 2014    # pooled shape uses season_year <= this
const POOL_WEIGHT = 0.5          # best fullpool weight from the pool sweep
const VALIDATION_ONLY = (1, 2)
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")

# ---------------------------------------------------------------------
# Backfill correction profile -- multiplicative / per-location / median
# / window 6, the best variant from experiments/simple-round/backfill's
# sweep (generalised there over mode/window/pooling/stat; fixed here).
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical per-location revision profile on the `transform` scale:
for each `(location, delay)` with at least `min_support` observations,
the MEDIAN of `settled / vintage` (both on `transform` scale) across
matching `(location, origin_date)` groups -- the multiplicative variant
that scored best in the backfill sweep. `versions` must already be
filtered by the caller to the desired origin dates (here: training set
only, no test-season data).
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
            abs(vintage) < 1e-6 && continue
            key = (loc, delay)
            push!(get!(raw, key, Float64[]), settled / vintage)
        end
    end
    profile = Dict{Tuple{String,Int},Float64}()
    for (key, vals) in raw
        length(vals) >= min_support && (profile[key] = median(vals))
    end
    return profile
end

"""
    apply_backfill_correction!(data, profile; delay_cutoff)

Multiply `data.Y` in place, at every `(t, l)` with `0 <= data.delay[t,
l] <= delay_cutoff` and a matching `profile` entry, by the profile's
per-location/delay correction. Missing entries and delays outside the
profile's support are left untouched. `profile` may be empty (no-
backfill ablation), in which case this is a no-op.
"""
function apply_backfill_correction!(
    data::ModelData, profile::Dict{Tuple{String,Int},Float64};
    delay_cutoff::Int,
)
    for l in 1:data.L, t in 1:data.T
        d = data.delay[t, l]
        (d < 0 || d > delay_cutoff) && continue
        ismissing(data.Y[t, l]) && continue
        key = (LOCATIONS[l], d)
        haskey(profile, key) || continue
        data.Y[t, l] *= profile[key]
    end
    return data
end

# ---------------------------------------------------------------------
# Pooled seasonal climatology (identical to seasonpool/generate.jl)
# ---------------------------------------------------------------------

"""
    fourier_features(woy, K, period) -> Vector{Float64}

`2K` Fourier features for `K` harmonics of week-of-season `woy` at the
given `period` (weeks).
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

Fit ONE shared `K`-harmonic week-of-season shape, pooling all 11
locations, from `history` restricted to `season_year(origin_date) <=
cutoff_year`. Each location's `transform`-scale series is centred on
its own mean over this window first, then a no-intercept OLS
regression of the pooled centred values on `fourier_features` gives
the shared shape's `2K` coefficients.
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

Shared pooled seasonal shape (deviation from a location's own mean, on
the `transform` scale) at week-of-season `woy`.
"""
function shape_value(woy::Real, shape_coef::Vector{Float64}, K::Int,
        period::Float64)
    return dot(fourier_features(woy, K, period), shape_coef)
end

"""
    fit_seasonal_level(y, woy_vec, shape_coef, K, period) -> (alpha, beta)

Per-location OLS fit of `y_t = alpha + beta * shape(woy_t) + resid`,
fit on that split's own training window only.
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
# AR(order) fit + partial pooling (identical to pool/generate.jl),
# applied to the deseasonalised, backfill-corrected residual.
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
    resid_sd_for(X, yresp, coef, order) -> Float64

Residual SD of `coef` (not necessarily the OLS solution for `X`,
`yresp`) evaluated on this design.
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
    fit_ar(y, order) -> (coef, X, yresp)

OLS fit of an AR(`order`) model with intercept to `y`.
"""
function fit_ar(y::AbstractVector{Float64}, order::Int)
    X, yresp = ar_design(y, order)
    coef = X \ yresp
    return coef, X, yresp
end

"""
    fit_ar_pooled(ys, order) -> coef

One OLS AR(`order`) fit on the design rows of every series in `ys`
stacked together -- the `:fullpool` anchor.
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

Simulate `npaths` Gaussian-innovation AR(`order`) sample paths forward
from the end of `y` (deseasonalised residual scale), for each horizon
in `horizons`.
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
# Forecast table builder, parameterised on which of the three levers
# (season, backfill, pool) are switched on, and on AR order.
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, versions_full, shape_coef, profile;
        order, use_season, use_backfill, use_pool, model_id) -> DataFrame

Fit and forecast for every cross-validation split of every season in
`seasons`, stacking whichever of the three levers are switched on:
backfill correction (multiplicative/per-location/median, window
`BACKFILL_WINDOW`) applied to the vintage series before anything else;
pooled seasonal deseasonalisation before the AR fit; partial pooling of
the per-location AR(`order`) coefficients toward the `:fullpool` anchor
at `POOL_WEIGHT`. With all three off/false this reproduces the plain
AR(`order`) baseline; with all three on it is the full model.
"""
function build_forecast_table(
    seasons, versions_full, shape_coef, profile; order::Int,
    use_season::Bool, use_backfill::Bool, use_pool::Bool,
    model_id::String,
)
    rng = MersenneTwister(SEED)
    rows = DataFrame(
        model_id=String[], location=String[], origin_date=Date[],
        horizon=Int[], target_end_date=Date[], target=String[],
        output_type=String[], output_type_id=Float64[], value=Float64[],
    )
    empty_profile = Dict{Tuple{String,Int},Float64}()
    for season in seasons
        splits = training_splits(season)
        for split in splits
            data = build_model_data(
                split; Dmax=DMAX, transform=TRANSFORM, window_weeks=104,
                versions=versions_full,
            )
            apply_backfill_correction!(
                data, use_backfill ? profile : empty_profile;
                delay_cutoff=BACKFILL_WINDOW,
            )
            origin = data.origin_date
            L = length(LOCATIONS)

            alphas = zeros(L)
            betas = zeros(L)
            resids = Vector{Vector{Float64}}(undef, L)
            for li in 1:L
                y = Float64.(data.Y[:, li])
                if use_season
                    alpha, beta = fit_seasonal_level(
                        y, data.woy, shape_coef, N_HARMONICS, SEASON_PERIOD,
                    )
                    alphas[li] = alpha
                    betas[li] = beta
                    seasonal_now = [
                        alpha + beta * shape_value(
                            w, shape_coef, N_HARMONICS, SEASON_PERIOD,
                        ) for w in data.woy
                    ]
                    resids[li] = y .- seasonal_now
                else
                    resids[li] = y
                end
            end

            fits = [fit_ar(resids[li], order) for li in 1:L]
            coefs = [f[1] for f in fits]
            Xs = [f[2] for f in fits]
            yresps = [f[3] for f in fits]
            blended = if use_pool
                anchor = fit_ar_pooled(resids, order)
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
                resid_sd = resid_sd_for(Xs[li], yresps[li], coef, order)
                paths = simulate_paths(
                    resids[li], coef, resid_sd, order, HORIZONS, NPATHS;
                    rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    seasonal_h = use_season ? alphas[li] + betas[li] *
                        shape_value(
                            week_of_season(target_end), shape_coef,
                            N_HARMONICS, SEASON_PERIOD,
                        ) : 0.0
                    vals = paths[h] .+ seasonal_h
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

function run_variant(versions_full, shape_coef, profile, truth, order;
        use_season, use_backfill, use_pool, model_id)
    forecast = build_forecast_table(
        VALIDATION_ONLY, versions_full, shape_coef, profile; order=order,
        use_season=use_season, use_backfill=use_backfill,
        use_pool=use_pool, model_id=model_id,
    )
    scored = score_forecasts(forecast, truth; scale=:natural)
    summ = wis_summary(scored)
    return (summary=summ[1, :], scored=scored, forecast=forecast)
end

function main()
    t0 = time()

    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BACKFILL_WINDOW,
        min_support=MIN_SUPPORT,
    )
    println("backfill profile: $(length(profile)) (location, delay) " *
            "entries with >= $(MIN_SUPPORT) observations")

    history = load_series("flu_data_hhs")
    shape_coef = fit_pooled_shape(
        history; transform=TRANSFORM, K=N_HARMONICS, period=SEASON_PERIOD,
        cutoff_year=CLIMATOLOGY_YEAR,
    )
    println("pooled shape ($(N_HARMONICS) harmonics, season_year <= " *
            "$(CLIMATOLOGY_YEAR)): coef=$(round.(shape_coef; digits=4))")

    truth = load_oracle(HUB_PATH)

    # -------------------------------------------------------------
    # 1. Full model (season + backfill + pool) across AR order
    # -------------------------------------------------------------
    println("\n=== full model (season + backfill + pool), AR order " *
            "sweep ===")
    order_results = NamedTuple[]
    for order in AR_ORDERS
        r = run_variant(
            versions_full, shape_coef, profile, truth, order;
            use_season=true, use_backfill=true, use_pool=true,
            model_id="full-order$(order)",
        )
        push!(order_results, (
            order=order, mean_wis=r.summary.mean_wis,
            sd_wis=r.summary.sd_wis, n_tasks=r.summary.n_tasks,
        ))
        println("order=$(order): mean_wis=" *
                "$(round(r.summary.mean_wis; digits=4)) sd_wis=" *
                "$(round(r.summary.sd_wis; digits=4)) " *
                "($(round(time() - t0; digits=1))s elapsed)")
    end
    sort!(order_results; by=r -> r.mean_wis)
    best_order = order_results[1].order
    println("best AR order for the full model: $(best_order) " *
            "(mean_wis=$(round(order_results[1].mean_wis; digits=4)))")

    full_run = run_variant(
        versions_full, shape_coef, profile, truth, best_order;
        use_season=true, use_backfill=true, use_pool=true,
        model_id="full",
    )

    # -------------------------------------------------------------
    # 2. Ablation at the best order: drop each lever in turn
    # -------------------------------------------------------------
    println("\n=== ablation at order=$(best_order) ===")
    ablations = [
        (label="full (all three)", use_season=true, use_backfill=true,
            use_pool=true),
        (label="no season", use_season=false, use_backfill=true,
            use_pool=true),
        (label="no backfill", use_season=true, use_backfill=false,
            use_pool=true),
        (label="no pool", use_season=true, use_backfill=true,
            use_pool=false),
        (label="AR only (none of the three)", use_season=false,
            use_backfill=false, use_pool=false),
    ]
    ablation_results = NamedTuple[]
    for a in ablations
        r = run_variant(
            versions_full, shape_coef, profile, truth, best_order;
            use_season=a.use_season, use_backfill=a.use_backfill,
            use_pool=a.use_pool, model_id="ablation-$(a.label)",
        )
        push!(ablation_results, (
            label=a.label, mean_wis=r.summary.mean_wis,
            sd_wis=r.summary.sd_wis,
        ))
        println("$(a.label): mean_wis=" *
                "$(round(r.summary.mean_wis; digits=4)) sd_wis=" *
                "$(round(r.summary.sd_wis; digits=4)) " *
                "($(round(time() - t0; digits=1))s elapsed)")
    end

    # -------------------------------------------------------------
    # Persist forecast + score.txt
    # -------------------------------------------------------------
    full_forecast = full_run.forecast
    full_forecast.model_id .= "full"
    CSV.write(joinpath(HERE, "forecast.csv"), full_forecast)

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "full model (seasonality + backfill + pooling) -- " *
                "simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "components (each fixed at its own single-lever " *
                "sweep's best variant):")
        println(io, "  season:   pooled $(N_HARMONICS)-harmonic " *
                "climatology, season_year <= $(CLIMATOLOGY_YEAR)")
        println(io, "  backfill: multiplicative, per-location, median, " *
                "window $(BACKFILL_WINDOW)")
        println(io, "  pool:     fullpool anchor, weight $(POOL_WEIGHT)")
        println(io)
        println(io, "reference points:")
        println(io, "  nfidd-ar6 (plain AR(6))                 = 0.368")
        println(io, "  seabbs_bot-ar6bf (AR(6)+backfill)        = 0.359")
        println(io, "  ar-order sweep: AR(12)+backfill          = 0.3518")
        println(io, "  seasonpool (AR(6)+backfill+pooled season)" *
                " = 0.3049")
        println(io, "  pool sweep: AR(6)+fullpool(w=0.5)        = 0.3643")
        println(io)
        println(io, "-- full model, AR order sweep --")
        for r in order_results
            println(io, "  order=$(r.order): mean_wis=" *
                    "$(round(r.mean_wis; digits=4)) sd_wis=" *
                    "$(round(r.sd_wis; digits=4)) n_tasks=$(r.n_tasks)")
        end
        println(io)
        println(io, "best order: $(best_order), mean_wis=" *
                "$(round(full_run.summary.mean_wis; digits=4)) sd_wis=" *
                "$(round(full_run.summary.sd_wis; digits=4)) n_tasks=" *
                "$(full_run.summary.n_tasks)")
        vs_arorder_bf = 0.3518 - full_run.summary.mean_wis
        println(io, "vs ar-order sweep's AR(12)+backfill (0.3518): " *
                "$(round(vs_arorder_bf; digits=4)) " *
                "($(round(100 * vs_arorder_bf / 0.3518; digits=2))%)")
        vs_seasonpool = 0.3049 - full_run.summary.mean_wis
        println(io, "vs seasonpool alone (0.3049): " *
                "$(round(vs_seasonpool; digits=4)) " *
                "($(round(100 * vs_seasonpool / 0.3049; digits=2))%)")
        println(io)
        println(io, "-- ablation at order=$(best_order) --")
        for r in ablation_results
            println(io, "  $(rpad(r.label, 30)) mean_wis=" *
                    "$(round(r.mean_wis; digits=4)) sd_wis=" *
                    "$(round(r.sd_wis; digits=4))")
        end
        println(io)
        full_wis = ablation_results[1].mean_wis
        println(io, "component contributions (full model's mean_wis " *
                "minus dropping that one component; positive = the " *
                "component was helping):")
        for (idx, comp) in ((2, "season"), (3, "backfill"), (4, "pool"))
            drop_wis = ablation_results[idx].mean_wis
            contrib = drop_wis - full_wis
            println(io, "  $(comp): dropping it changes mean_wis by " *
                    "$(round(contrib; digits=4)) " *
                    "($(round(100 * contrib / full_wis; digits=2))% " *
                    "relative to the full model)")
        end

        println(io)
        println(io, "-- breakdown by location (full model) --")
        by_loc = combine(groupby(full_run.scored, :location),
            :wis => mean => :mean_wis, nrow => :n)
        sort!(by_loc, :mean_wis)
        for row in eachrow(by_loc)
            println(io, "  $(rpad(row.location, 16)) mean_wis=" *
                    "$(round(row.mean_wis; digits=4)) n=$(row.n)")
        end

        println(io)
        println(io, "-- breakdown by season (full model) --")
        full_run.scored.season_year = season_year.(
            full_run.scored.origin_date,
        )
        by_season = combine(groupby(full_run.scored, :season_year),
            :wis => mean => :mean_wis, nrow => :n)
        for row in eachrow(by_season)
            println(io, "  season $(row.season_year): mean_wis=" *
                    "$(round(row.mean_wis; digits=4)) n=$(row.n)")
        end

        println(io)
        println(io, "-- breakdown by horizon (full model) --")
        by_h = combine(groupby(full_run.scored, :horizon),
            :wis => mean => :mean_wis, nrow => :n)
        sort!(by_h, :horizon)
        for row in eachrow(by_h)
            println(io, "  h=$(row.horizon): mean_wis=" *
                    "$(round(row.mean_wis; digits=4)) n=$(row.n)")
        end
    end

    dt = round(time() - t0; digits=1)
    println("\nwrote forecast.csv + score.txt in $(dt)s total")
    return (order_results=order_results, ablation_results=ablation_results,
        full=full_run)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
