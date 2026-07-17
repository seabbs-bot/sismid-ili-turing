#!/usr/bin/env julia
# horizon-specific DIRECT forecasting -- simple-round, HORIZON family.
#
# Starting point: the two winners locked so far in the simple-model
# search --
#   - AR order: AR(12) beats AR(6) (experiments/simple-round/ar-order,
#     order=12+backfill = 0.3518 mean WIS on validation, vs 0.359 for
#     seabbs_bot-ar6bf's AR(6)+backfill).
#   - backfill correction: additive, per-location, median revision
#     profile on the most recent 8 weeks (seabbs_bot-ar6bf), unchanged
#     here.
#
# Every model tried so far -- including both of the above -- is
# ITERATED: fit one one-step-ahead AR(p), then simulate forward h=1..4
# steps, feeding each simulated value back in as a lag for the next
# step. WIS is summed over h=1..4, so an iterated model that is tuned
# (implicitly, via its single set of AR coefficients) to minimise
# one-step error is not necessarily minimising the 4-step-ahead error
# too -- forecast error compounds forward through the feedback loop,
# and a single AR(p) may just be the wrong shape for the h=3/h=4
# relationship.
#
# This experiment asks whether a DIRECT model per horizon -- regressing
# the h-step-ahead value on today's lags in one shot, a separate OLS fit
# for each h in 1:4, no forward simulation or feedback at all -- beats
# the iterated approach, especially at the longer horizons where
# iterated compounding should hurt most. It also tries horizon-specific
# interval widening: since the direct fit gives an independent
# (point, residual SD) pair per horizon already, calibrating a per-
# horizon width multiplier on the residual SD is a one-line addition,
# and there is no reason the same interval width should be optimal at
# h=1 and h=4.
#
# Both the AR(12)+backfill winner and this experiment's own season-
# ablation below are run WITHOUT the pooled seasonal term as an
# apples-to-apples reference to the locked 0.3518 number. The MAIN
# candidates additionally stack the pooled seasonal climatology
# (experiments/simple-round/seasonpool: one shared week-of-season
# Fourier(3) shape, pooled across all 11 locations, adapted to each
# location by a 2-parameter intercept+amplitude fit; AR(12)/direct is
# then fit to the DESEASONALISED residual, and the seasonal term for
# the known future week is added back at forecast time) -- the
# combination the team lead asked this branch to try, not the AR
# order/backfill combination alone.
#
# Four variants are compared, all on the SAME backfill profile and
# pooled seasonal shape, all sharing AR_ORDER=12 where an AR order
# applies:
#   A. iterated-noseasn : iterated AR(12) + backfill, NO season
#                          (reproduces the locked ar-order winner,
#                          sanity-checks this script's own plumbing
#                          against the 0.3518 reference)
#   B. iterated-season   : iterated AR(12) + backfill + pooled season
#                          (isolates the season contribution on top of
#                          the iterated AR(12) winner)
#   C. direct-season      : DIRECT per-horizon AR(12) + backfill +
#                          pooled season, unwidened (width=1 at every
#                          horizon) -- the main "does direct beat
#                          iterated" comparison, holding backfill +
#                          season fixed relative to B
#   D. direct-season-wide : C plus a per-horizon interval-width
#                          multiplier on the residual SD, calibrated by
#                          grid search independently at each horizon
#                          (WIS at horizon h depends only on that
#                          horizon's own predictive distribution, so
#                          this is a per-horizon 1-D search, not a 4-D
#                          one)
#
# Scope: VALIDATION SEASONS (1, 2) ONLY, scored against the local hub
# clone's oracle (docs/contracts.md experimental integrity) -- this is
# a tuning experiment, not a submission driver, and the width grid
# search is itself tuned only on these seasons. LIGHT + ANALYTIC:
# CSV/DataFrames/Statistics/Random/LinearAlgebra only, no Turing.
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
const AR_ORDER = 12           # matches the locked ar-order winner
const N_HARMONICS = 3         # matches seasonpool
const SEASON_PERIOD = 52.0
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12                # matches ar6bf/ar-order's build_model_data
const DELAY_CUTOFF = 8         # backfill window; matches ar6bf/ar-order
const MIN_SUPPORT = 5          # min sample size per (location, delay)
const CLIMATOLOGY_YEAR = 2014  # pooled shape uses season_year <= this
const VALIDATION_ONLY = (1, 2)
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")
const WIDTH_GRID = collect(0.6:0.1:1.6)  # per-horizon width candidates

# ---------------------------------------------------------------------
# Backfill correction profile -- identical to seabbs_bot-ar6bf /
# ar-order / seasonpool
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile on the `transform` scale:
for each `(location, delay)` with at least `min_support` recorded
versions at that delay, the median of `settled - vintage` (both on
`transform` scale) across matching `(location, origin_date)` groups.
`versions` must already be filtered by the caller (here: `season_year
<= 2016`, no test-season data).
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

Nudge `data.Y` in place, at every `(t, l)` with `0 <= data.delay[t, l]
<= DELAY_CUTOFF` and a matching `profile` entry, by adding the
profile's location/delay correction. Missing entries and delays
outside the profile's support are left untouched.
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
# Pooled seasonal climatology -- identical to seasonpool
# ---------------------------------------------------------------------

"""
    fourier_features(woy, K, period) -> Vector{Float64}

`2K` Fourier features `[sin(2*pi*1*woy/period), cos(2*pi*1*woy/period),
...]` for `K` harmonics of week-of-season `woy` at the given `period`
(weeks).
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
locations, from `history` (the `flu_data_hhs.csv` schema: `location,
origin_date, wili`) restricted to `season_year(origin_date) <=
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

Per-location OLS fit of `y_t = alpha + beta * shape(woy_t) + resid`:
the per-location amplitude/level scaling of the shared shape. Fit on
that split's own training window only.
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
# Iterated AR(p) fit + forecast -- identical in form to nfidd-ar6 /
# ar6bf / ar-order / seasonpool
# ---------------------------------------------------------------------

"""
    fit_ar(y, order) -> (coef, resid_sd)

OLS fit of a ONE-STEP-AHEAD AR(`order`) model with intercept to `y`
(ascending in time, no missing values). `coef = [c, phi_1, ...,
phi_order]`, `phi_1` multiplying the most recent lag. `resid_sd` is the
in-sample residual standard deviation.
"""
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

"""
    simulate_paths(y, coef, resid_sd, order, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Simulate `npaths` Gaussian-innovation AR(`order`) sample paths forward
from the end of `y`, for each horizon in `horizons`, feeding each
simulated value back in as a lag for later horizons -- the ITERATED
approach every other model in this repo uses.
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

"""
    build_iterated_forecast_table(seasons, profile, versions_full,
                                   shape_coef; use_season, model_id)
        -> DataFrame

Fit and forecast the ITERATED AR(`AR_ORDER`)+backfill(+season, if
`use_season`) model for every cross-validation split of every season in
`seasons`, returning the combined hub quantile table. Variants A/B.
"""
function build_iterated_forecast_table(
    seasons, profile, versions_full, shape_coef; use_season::Bool,
    model_id::String,
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
                split; Dmax=DMAX, transform=TRANSFORM, window_weeks=104,
                versions=versions_full,
            )
            apply_backfill_correction!(data, profile)
            origin = data.origin_date
            for (li, loc) in enumerate(LOCATIONS)
                y = Float64.(data.Y[:, li])
                alpha, beta = 0.0, 0.0
                resid = y
                if use_season
                    alpha, beta = fit_seasonal_level(
                        y, data.woy, shape_coef, N_HARMONICS,
                        SEASON_PERIOD,
                    )
                    seasonal_now = [
                        alpha + beta * shape_value(
                            w, shape_coef, N_HARMONICS, SEASON_PERIOD,
                        ) for w in data.woy
                    ]
                    resid = y .- seasonal_now
                end
                coef, resid_sd = fit_ar(resid, AR_ORDER)
                paths = simulate_paths(
                    resid, coef, resid_sd, AR_ORDER, HORIZONS, NPATHS;
                    rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    seasonal_h = use_season ? alpha + beta * shape_value(
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

# ---------------------------------------------------------------------
# DIRECT per-horizon AR(p) fit -- the HORIZON-SPECIFIC candidate
# ---------------------------------------------------------------------

"""
    fit_ar_direct(resid, order, horizon) -> (coef, resid_sd)

OLS fit of a DIRECT `horizon`-step-ahead regression: `resid[t+horizon]`
on an intercept and the `order` most recent lags `resid[t], resid[t-1],
..., resid[t-order+1]`, for every valid `t` (ascending in time, no
missing values). Unlike [`fit_ar`](@ref)'s one-step model iterated
forward through [`simulate_paths`](@ref), this fits the `horizon`-step
relationship in one shot -- a separate regression per horizon, so
forecast error at h=4 cannot compound through h=1..3's simulated
values. `coef = [c, phi_1, ..., phi_order]`, same lag convention as
`fit_ar`. `resid_sd` is this `horizon`-specific regression's in-sample
residual standard deviation.
"""
function fit_ar_direct(resid::AbstractVector{Float64}, order::Int,
        horizon::Int)
    n = length(resid)
    nobs = n - order - horizon + 1
    nobs >= order + 2 || error(
        "series too short for direct AR($order, h=$horizon): " *
        "n=$n, nobs=$nobs",
    )
    X = ones(nobs, order + 1)
    yresp = Vector{Float64}(undef, nobs)
    for (row, t) in enumerate(order:(n - horizon))
        yresp[row] = resid[t + horizon]
        for lag in 1:order
            X[row, lag + 1] = resid[t - lag + 1]
        end
    end
    coef = X \ yresp
    resid_err = yresp .- X * coef
    dof = max(nobs - (order + 1), 1)
    resid_sd = sqrt(sum(abs2, resid_err) / dof)
    return coef, resid_sd
end

"""
    predict_direct(tail, coef) -> Float64

Point forecast from a direct-horizon regression's `coef` (as returned
by [`fit_ar_direct`](@ref)), given `tail`, the most recent `order`
values of the series (ascending; `tail[end]` is the latest
observation).
"""
function predict_direct(tail::AbstractVector{Float64},
        coef::Vector{Float64})
    order = length(coef) - 1
    pred = coef[1]
    for lag in 1:order
        pred += coef[lag + 1] * tail[end - lag + 1]
    end
    return pred
end

"""
    compute_direct_components(seasons, profile, versions_full,
                               shape_coef; rng) -> DataFrame

Fit backfill-corrected, deseasonalised DIRECT per-horizon AR(`AR_ORDER`)
regressions for every (split, location, horizon) in `seasons`, and cache
each task's point-forecast COMPONENTS rather than a finished quantile
table: `pred` (the direct regression's `horizon`-step-ahead point
forecast on the deseasonalised residual scale), `resid_sd` (that
regression's in-sample residual SD), `seasonal` (the pooled-shape
seasonal term at the target week, scaled to this location by
`fit_seasonal_level`'s alpha/beta), and `z` (`NPATHS` iid standard-
normal draws generated once per task with `rng`). Downstream, a
forecast quantile table at ANY per-horizon interval width is just
`pred + seasonal + width[horizon] * resid_sd .* z`
([`forecast_from_components`](@ref)): caching `z` once and reusing it
across width candidates removes simulation noise from the width
comparison, isolating the effect of the width scalar itself.
"""
function compute_direct_components(
    seasons, profile, versions_full, shape_coef; rng::Random.AbstractRNG,
)
    rows = DataFrame(
        location=String[], origin_date=Date[], horizon=Int[],
        target_end_date=Date[], pred=Float64[], resid_sd=Float64[],
        seasonal=Float64[], z=Vector{Float64}[],
    )
    for season in seasons
        splits = training_splits(season)
        for split in splits
            data = build_model_data(
                split; Dmax=DMAX, transform=TRANSFORM, window_weeks=104,
                versions=versions_full,
            )
            apply_backfill_correction!(data, profile)
            origin = data.origin_date
            for (li, loc) in enumerate(LOCATIONS)
                y = Float64.(data.Y[:, li])
                alpha, beta = fit_seasonal_level(
                    y, data.woy, shape_coef, N_HARMONICS, SEASON_PERIOD,
                )
                seasonal_now = [
                    alpha + beta * shape_value(
                        w, shape_coef, N_HARMONICS, SEASON_PERIOD,
                    ) for w in data.woy
                ]
                resid = y .- seasonal_now
                for h in HORIZONS
                    coef, resid_sd = fit_ar_direct(resid, AR_ORDER, h)
                    tail = resid[(end - AR_ORDER + 1):end]
                    pred = predict_direct(tail, coef)
                    target_end = origin + Day(7 * h)
                    seasonal_h = alpha + beta * shape_value(
                        week_of_season(target_end), shape_coef,
                        N_HARMONICS, SEASON_PERIOD,
                    )
                    z = randn(rng, NPATHS)
                    push!(rows, (
                        loc, origin, h, target_end, pred, resid_sd,
                        seasonal_h, z,
                    ))
                end
            end
        end
    end
    return rows
end

"""
    forecast_from_components(components, widths, model_id) -> DataFrame

Hub quantile table (docs/contracts.md schema) built from cached direct-
model `components` (as returned by [`compute_direct_components`](@ref)):
for each task, draws are `pred + seasonal + widths[horizon] * resid_sd
.* z` (the same cached `z`, just rescaled), quantiled at the 23 hub
levels and back-transformed with `from_scale`, clamped at 0. `widths`
maps horizon to its interval-width multiplier (`1.0` reproduces the
unwidened direct model).
"""
function forecast_from_components(
    components::DataFrame, widths::Dict{Int,Float64}, model_id::String,
)
    rows = DataFrame(
        model_id=String[], location=String[], origin_date=Date[],
        horizon=Int[], target_end_date=Date[], target=String[],
        output_type=String[], output_type_id=Float64[], value=Float64[],
    )
    for row in eachrow(components)
        w = widths[row.horizon]
        draws = row.pred .+ row.seasonal .+ w * row.resid_sd .* row.z
        for q in QUANTILE_LEVELS
            qval = quantile(draws, q)
            nat = max(from_scale(qval, TRANSFORM), 0.0)
            push!(rows, (
                model_id, row.location, row.origin_date, row.horizon,
                row.target_end_date, TARGET, "quantile", q, nat,
            ))
        end
    end
    return rows
end

"""
    calibrate_widths(components, truth) -> (best, detail)

Grid-search a per-horizon interval-width multiplier over `WIDTH_GRID`,
independently at each horizon: WIS at horizon `h` depends only on that
horizon's own predictive distribution, so this is four independent 1-D
searches, not a joint 4-D one. `best` maps horizon to its WIS-
minimising width; `detail` is the full grid (`horizon, width,
mean_wis`) for reporting.
"""
function calibrate_widths(components::DataFrame, truth::DataFrame)
    best = Dict{Int,Float64}()
    detail = NamedTuple[]
    for h in HORIZONS
        comp_h = components[components.horizon .== h, :]
        best_w = 1.0
        best_wis = Inf
        for w in WIDTH_GRID
            forecast_h = forecast_from_components(
                comp_h, Dict(h => w), "calibration",
            )
            scored_h = score_forecasts(forecast_h, truth; scale=:natural)
            mwis = mean(scored_h.wis)
            push!(detail, (horizon=h, width=w, mean_wis=mwis))
            if mwis < best_wis
                best_wis = mwis
                best_w = w
            end
        end
        best[h] = best_w
    end
    return best, DataFrame(detail)
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
# Reporting helpers
# ---------------------------------------------------------------------

"""Per-horizon mean/SD WIS table for one scored forecast table."""
function by_horizon(scored::DataFrame)
    combine(groupby(scored, :horizon),
        :wis => mean => :mean_wis, :wis => std => :sd_wis, nrow => :n)
end

function main()
    t0 = time()

    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=DELAY_CUTOFF,
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

    println("\n=== A: iterated AR(12) + backfill, no season " *
            "(reproduces ar-order winner) ===")
    forecast_a = build_iterated_forecast_table(
        VALIDATION_ONLY, profile, versions_full, shape_coef;
        use_season=false, model_id="A-iterated-noseason",
    )
    scored_a = score_forecasts(forecast_a, truth; scale=:natural)
    summ_a = wis_summary(scored_a)[1, :]
    println("mean_wis=$(round(summ_a.mean_wis; digits=4)) " *
            "sd_wis=$(round(summ_a.sd_wis; digits=4)) " *
            "n_tasks=$(summ_a.n_tasks) " *
            "(reference: ar-order/score.txt reports 0.3518)")

    println("\n=== B: iterated AR(12) + backfill + pooled season ===")
    forecast_b = build_iterated_forecast_table(
        VALIDATION_ONLY, profile, versions_full, shape_coef;
        use_season=true, model_id="B-iterated-season",
    )
    scored_b = score_forecasts(forecast_b, truth; scale=:natural)
    summ_b = wis_summary(scored_b)[1, :]
    println("mean_wis=$(round(summ_b.mean_wis; digits=4)) " *
            "sd_wis=$(round(summ_b.sd_wis; digits=4)) " *
            "n_tasks=$(summ_b.n_tasks)")

    println("\n=== C: DIRECT per-horizon AR(12) + backfill + season, " *
            "unwidened ===")
    rng_c = MersenneTwister(SEED)
    components = compute_direct_components(
        VALIDATION_ONLY, profile, versions_full, shape_coef; rng=rng_c,
    )
    unit_widths = Dict(h => 1.0 for h in HORIZONS)
    forecast_c = forecast_from_components(
        components, unit_widths, "C-direct-season",
    )
    scored_c = score_forecasts(forecast_c, truth; scale=:natural)
    summ_c = wis_summary(scored_c)[1, :]
    println("mean_wis=$(round(summ_c.mean_wis; digits=4)) " *
            "sd_wis=$(round(summ_c.sd_wis; digits=4)) " *
            "n_tasks=$(summ_c.n_tasks)")

    println("\n=== D: DIRECT per-horizon AR(12) + backfill + season, " *
            "calibrated per-horizon width ===")
    best_widths, width_detail = calibrate_widths(components, truth)
    forecast_d = forecast_from_components(
        components, best_widths, "D-direct-season-wide",
    )
    scored_d = score_forecasts(forecast_d, truth; scale=:natural)
    summ_d = wis_summary(scored_d)[1, :]
    println("chosen widths: $(best_widths)")
    println("mean_wis=$(round(summ_d.mean_wis; digits=4)) " *
            "sd_wis=$(round(summ_d.sd_wis; digits=4)) " *
            "n_tasks=$(summ_d.n_tasks)")

    hz_a = by_horizon(scored_a)
    hz_b = by_horizon(scored_b)
    hz_c = by_horizon(scored_c)
    hz_d = by_horizon(scored_d)

    dt = round(time() - t0; digits=1)
    println("\nwrote score.txt in $(dt)s total")

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "horizon-specific DIRECT forecasting -- " *
                     "simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(dt)s")
        println(io)
        println(io, "Family: LIGHT + ANALYTIC (OLS, no Turing), " *
                     "fourth-root scale, AR($(AR_ORDER)) on top of " *
                     "the seabbs_bot-ar6bf backfill correction " *
                     "(unchanged) and, where noted, the seasonpool " *
                     "pooled seasonal climatology (unchanged). Only " *
                     "the ITERATED-vs-DIRECT AR fitting method and " *
                     "the per-horizon interval width are varied here.")
        println(io)
        println(io, "reference points:")
        println(io, "  ar-order winner (AR12+backfill, no season, " *
                     "iterated) = 0.3518")
        println(io, "  seabbs_bot-ar6bf (AR6+backfill, no season)  " *
                     "= 0.359")
        println(io, "  seasonpool (AR6+backfill+season, iterated)  " *
                     "= 0.3049")
        println(io)
        println(io, "=== overall summary (mean +/- SD WIS) ===")
        println(io, rpad("variant", 26) * rpad("mean_wis", 12) *
                     rpad("sd_wis", 10) * "n_tasks")
        for (label, s) in (
            ("A iterated, no season", summ_a),
            ("B iterated + season", summ_b),
            ("C direct + season", summ_c),
            ("D direct + season + width", summ_d),
        )
            println(io, rpad(label, 26) *
                         rpad(string(round(s.mean_wis; digits=4)), 12) *
                         rpad(string(round(s.sd_wis; digits=4)), 10) *
                         string(s.n_tasks))
        end
        println(io)
        vs_locked = summ_c.mean_wis - 0.3518
        vs_locked_pct = round(100 * vs_locked / 0.3518; digits=2)
        println(io, "C (direct+season) vs locked ar-order winner " *
                     "(0.3518): $(round(vs_locked; digits=4)) " *
                     "($(vs_locked_pct)%)")
        vs_iter_season = summ_c.mean_wis - summ_b.mean_wis
        vs_iter_season_pct = round(
            100 * vs_iter_season / summ_b.mean_wis; digits=2,
        )
        println(io, "C (direct) vs B (iterated), SAME backfill+" *
                     "season base: " *
                     "$(round(vs_iter_season; digits=4)) " *
                     "($(vs_iter_season_pct)%)")
        vs_widened = summ_d.mean_wis - summ_c.mean_wis
        vs_widened_pct = round(100 * vs_widened / summ_c.mean_wis; digits=2)
        println(io, "D (direct+width) vs C (direct, unwidened): " *
                     "$(round(vs_widened; digits=4)) " *
                     "($(vs_widened_pct)%)")
        println(io)
        println(io, "=== by-horizon mean +/- SD WIS ===")
        println(io, rpad("h", 4) *
                     rpad("A iter/noseason", 22) *
                     rpad("B iter/season", 22) *
                     rpad("C direct", 22) * "D direct+width")
        for h in HORIZONS
            ra = hz_a[hz_a.horizon .== h, :][1, :]
            rb = hz_b[hz_b.horizon .== h, :][1, :]
            rc = hz_c[hz_c.horizon .== h, :][1, :]
            rd = hz_d[hz_d.horizon .== h, :][1, :]
            fmt(r) = "$(round(r.mean_wis; digits=4)) " *
                     "(sd=$(round(r.sd_wis; digits=4)))"
            println(io, rpad(string(h), 4) *
                         rpad(fmt(ra), 22) * rpad(fmt(rb), 22) *
                         rpad(fmt(rc), 22) * fmt(rd))
        end
        println(io)
        println(io, "by-horizon delta, C (direct) vs B (iterated), " *
                     "same backfill+season base (negative = direct " *
                     "wins):")
        for h in HORIZONS
            rb = hz_b[hz_b.horizon .== h, :][1, :]
            rc = hz_c[hz_c.horizon .== h, :][1, :]
            delta = rc.mean_wis - rb.mean_wis
            pct = 100 * delta / rb.mean_wis
            println(io, "  h=$h: $(round(delta; digits=4)) " *
                         "($(round(pct; digits=2))%)")
        end
        println(io)
        println(io, "=== per-horizon width calibration " *
                     "(WIDTH_GRID=$(WIDTH_GRID)) ===")
        println(io, "chosen widths: $(best_widths)")
        for h in HORIZONS
            grid_h = width_detail[width_detail.horizon .== h, :]
            sort!(grid_h, :width)
            println(io, "  h=$h:")
            for row in eachrow(grid_h)
                marker = row.width == best_widths[h] ? "  <- chosen" : ""
                row_wis = round(row.mean_wis; digits=4)
                println(io, "    width=$(row.width) " *
                             "mean_wis=$(row_wis)$(marker)")
            end
        end
    end
    return (a=scored_a, b=scored_b, c=scored_c, d=scored_d)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
