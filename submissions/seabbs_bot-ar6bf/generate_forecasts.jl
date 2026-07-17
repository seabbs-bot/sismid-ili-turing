#!/usr/bin/env julia
# ar6bf_baseline.jl -- AR(6) per-location baseline (as nfidd-ar6) with an
# added BACKFILL CORRECTION step: recent, still-revisable vintage weeks
# are nudged toward their expected settled value before fitting, using
# an empirical location x delay revision profile estimated from
# `data/flu_data_hhs_versions.csv` (docs/eda/02-backfill.md).
#
# Everything else is identical to nfidd-ar6
# (submissions/nfidd-ar6/generate_forecasts.jl): independent AR(6) per
# location, fit by OLS on the fourth-root-transformed vintage series, no
# hierarchy, no seasonality term. This isolates the effect of the
# backfill correction in the nfidd-ar6-vs-nfidd-ar6bf WIS comparison.
#
# Backfill correction
# --------------------
# For each (location, delay) with delay = weeks(as_of - origin_date),
# the additive correction (on the fourth-root modelling scale) is the
# median of `fourthroot(settled) - fourthroot(vintage)` across every
# (location, origin_date) in `flu_data_hhs_versions.csv` with a recorded
# version at that delay, where `settled` is the value at that series'
# largest tracked `as_of` (proxy for the finalised value, following
# docs/eda/02-backfill.md's own methodology) and `vintage` is the value
# recorded at that delay. docs/eda/02-backfill.md's headline finding is
# that revisions are NOT a monotonic reporting-CDF completion (they
# change sign across delay and across location), so this is an
# empirical location x delay median profile, not a smooth monotonic
# completion factor.
#
# The profile is estimated ONLY from origin dates with
# `season_year <= 2016` (pre-2015 history plus the two validation
# seasons; no test-season data used anywhere -- docs/contracts.md
# experimental integrity). It is then applied, unchanged, to correct
# the most recent DELAY_CUTOFF weeks of every split's vintage series
# (both validation seasons) before the AR(6) fit. Because the profile
# is estimated from the same two validation seasons it is later scored
# on (there is no other tracked revision history -- see
# docs/eda/02-backfill.md's last section), this comparison should be
# read as "does correcting towards a known revision profile help",
# not as an unbiased estimate of out-of-sample gain; the README notes
# this explicitly.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate_forecasts.jl <hub_path>

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))
include(joinpath(PKG_DIR, "src", "hubio.jl"))

const MODEL_ID = "nfidd-ar6bf"
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12          # matches nfidd-ar6's build_model_data Dmax
const DELAY_CUTOFF = 8   # weeks; profile is ~0 beyond this, docs/eda/02
const MIN_SUPPORT = 5    # min sample size per (location, delay) to trust

# ---------------------------------------------------------------------
# Backfill correction profile
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile on the `transform` scale.
For each `(location, delay)` with at least `min_support` recorded
versions at that delay (`delay = weeks(as_of - origin_date)`), the
median of `to_scale(settled, transform) - to_scale(vintage, transform)`
across all matching `(location, origin_date)` groups, where `settled`
is the value at that group's largest tracked `as_of`
(docs/eda/02-backfill.md's settled-value proxy). `versions` must
already be filtered by the caller to the desired origin dates (here:
training set only, no test seasons).
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
profile's location/delay correction. Missing entries and delays outside
the profile's support are left untouched.
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
# AR(6) fit + forecast (identical to nfidd-ar6)
# ---------------------------------------------------------------------

"""
    fit_ar(y, order) -> (coef, resid_sd)

OLS fit of an AR(`order`) model with intercept to `y` (ascending in
time, no missing values). `coef = [c, phi_1, ..., phi_order]`, where
`phi_1` multiplies the most recent lag (`y[t-1]`) and `phi_order` the
most distant (`y[t-order]`). `resid_sd` is the in-sample residual
standard deviation with `nobs - (order + 1)` degrees of freedom.
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
from the end of `y` (modelling scale), for each horizon in `horizons`.
Each path draws one fresh Normal(0, resid_sd) innovation per step and
feeds simulated values back in as lags for later horizons (proper
forward propagation of forecast uncertainty).
"""
function simulate_paths(
    y::AbstractVector{Float64}, coef::Vector{Float64}, resid_sd::Float64,
    order::Int, horizons, npaths::Int; rng::Random.AbstractRNG,
)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    tail0 = y[(end - order + 1):end]  # most recent `order` obs, ascending
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
    build_forecast_table(seasons, profile) -> DataFrame

Fit and forecast the AR(6)+backfill-correction baseline for every
cross-validation split of every season in `seasons`, returning the
combined hub quantile table (docs/contracts.md schema). Training
discipline: `build_model_data` caps each split's data at its own
forecast origin (never carries future/finalised values), and
`window_weeks=104` further caps history to the most recent two
seasons. `versions` (full, unfiltered) is passed to `build_model_data`
so the true `as_of`-based delay is used wherever available; this only
ever looks at `as_of <= forecast_origin` (enforced inside
`build_model_data`), so it never leaks future revisions into a split's
own delay index.
"""
function build_forecast_table(seasons, profile, versions_full)
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
                coef, resid_sd = fit_ar(y, AR_ORDER)
                paths = simulate_paths(
                    y, coef, resid_sd, AR_ORDER, HORIZONS, NPATHS;
                    rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    vals = paths[h]
                    for q in QUANTILE_LEVELS
                        qval = quantile(vals, q)
                        nat = max(from_scale(qval, TRANSFORM), 0.0)
                        push!(rows, (
                            MODEL_ID, loc, origin, h, target_end,
                            TARGET, "quantile", q, nat,
                        ))
                    end
                end
            end
        end
    end
    return rows
end

function main()
    hub_path = length(ARGS) >= 1 ? ARGS[1] : nothing
    t0 = time()

    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=DELAY_CUTOFF,
        min_support=MIN_SUPPORT,
    )
    println("revision profile: $(length(profile)) (location, delay) " *
            "entries with >= $(MIN_SUPPORT) observations")

    forecast = build_forecast_table((1, 2), profile, versions_full)
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    if hub_path !== nothing
        write_submission(forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="nfidd", model_abbr="ar6bf", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
