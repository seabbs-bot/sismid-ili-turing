#!/usr/bin/env julia
# generate.jl -- climatology + backfill (the SEASONALITY family winner,
# experiments/simple-round/season/generate.jl, MODEL_ID
# "seabbs_bot-season", 0.3004 validation / 0.2781 full round-1 WIS),
# extended with a SHARED, POOLED SEASON-SEVERITY scaling term.
#
# Motivation (docs/eda/04-cross-location.md, "Correlation of seasonal
# amplitude: a shared severity-year effect"): per-season amplitude is
# correlated 0.68 on average across the 11 locations -- a season that
# runs severe tends to run severe everywhere, not independently per
# location. The climatology term in `season/generate.jl` is a FIXED
# curve (one smoothed median-by-week-of-season shape per location,
# built from all history strictly before each split's forecast
# origin); it has no way to know that THIS season, so far, is running
# hotter or colder than that fixed shape. The idea tested here: at
# each forecast origin, estimate how severe the current season is
# running (observed level vs. that location's own climatology, over
# the weeks already seen this season), POOL that estimate across all
# 11 locations (justified by the r=0.68 shared-year effect), and scale
# the climatology's contribution to the FORWARD forecast by the
# pooled severity factor -- i.e. adapt the fixed climatology shape to
# this season's apparent severity, without touching the fitted AR(6)+
# climatology regression itself.
#
# RESULT: this does NOT help. Every configuration tried made
# validation WIS worse than plain climatology-backfill (0.3004), and
# the degradation was MONOTONIC in the adjustment strength (see
# score.txt for the full scan): multiplicative and additive forms,
# mean/median pooling across locations, mean/median centering of the
# per-location estimate, and whole-season-to-date vs. recent-6-week vs.
# recent-4-week estimation windows all point the same way -- more
# severity adjustment is uniformly worse, at every horizon (including
# h=4, where the "helps more with more lead time" hypothesis predicted
# the opposite) and in most locations. The apparatus below is kept
# in full (not deleted) because it is the evidence for that negative
# result: `LAMBDA_SEVERITY = 0.0` is the empirically-best setting, so
# this file's OUTPUT is byte-for-byte the same model as
# `seabbs_bot-season` (climatology + backfill, no severity term) --
# flip `LAMBDA_SEVERITY` above 0 to reproduce the degradation directly.
#
# Likely mechanism (see score.txt discussion): the AR(6) lags already
# see the current season's actual recent values directly, so by the
# time enough in-season weeks exist to estimate a severity factor at
# all (MIN_SEVERITY_WEEKS), the AR component has already adapted to
# the season running hot or cold -- the pooled severity term is mostly
# redundant with information the model already has, and what it adds
# on top is estimation noise (worse still with the narrower recency
# windows, which have fewer weeks of support).
#
# Everything else (AR order, transform, climatology construction,
# backfill correction, path simulation, quantile levels, seed) is
# identical to `season/generate.jl`. LIGHT + ANALYTIC: CSV/DataFrames/
# Statistics/LinearAlgebra only, no Turing.
#
# Scope: tuned and scored on VALIDATION SEASONS (1, 2) ONLY
# (docs/contracts.md experimental integrity); see score.txt for the
# full scan. `main()` still generates forecasts for all five seasons
# when writing a hub submission, each split a per-origin vintage fit
# capped at its own forecast origin, matching every other
# `simple-round` driver.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl <hub_path>

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

const MODEL_ID = "seabbs_bot-severity2"
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12            # matches nfidd-ar6's build_model_data Dmax
const WINDOW_WEEKS = 104   # matches nfidd-ar6: caps AR history at 2 seasons
const SEASON_PERIOD = 52   # canonical annual cycle length for the climatology
const DELAY_CUTOFF = 8     # weeks; backfill profile is ~0 beyond this
const MIN_SUPPORT = 5      # min sample size per (location, delay) to trust

# -- season-severity tuning constants (see score.txt for the scan) --
const SEVERITY_FORM = :multiplicative  # :multiplicative or :additive
const SEVERITY_POOL_STAT = :median     # cross-location pooling statistic
const SEVERITY_CENTER = :mean          # per-location estimate centering
const SEVERITY_RECENT_WEEKS = nothing  # `nothing` = whole season-to-date
const MIN_SEVERITY_WEEKS = 3   # min in-season observed weeks to trust
const MIN_CLIM_LEVEL = 0.15    # min climatology value (fourthroot scale)
                                # to include a week in a severity estimate
const LAMBDA_SEVERITY = 0.0    # shrinkage of the pooled estimate toward
                                # "no adjustment" -- EMPIRICALLY BEST is
                                # 0.0 (disabled); see header and score.txt

# ---------------------------------------------------------------------
# Backfill correction (identical to seabbs_bot-ar6bf / season)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile on the `transform` scale.
Identical to `season/generate.jl`; see there for the full derivation.
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
matching `profile` entry exists. Identical to season/generate.jl.
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
# Climatology term (identical to season/generate.jl)
# ---------------------------------------------------------------------

"""
    build_climatology(loc_hist, forecast_origin;
                       period=SEASON_PERIOD, smooth_window=5)
        -> Vector{Float64}

Smoothed circular week-of-season climatology curve for one location,
on the `TRANSFORM` scale, length `period`. Built ONLY from `loc_hist`
rows strictly before `forecast_origin`. Identical to
`season/generate.jl`; see there for the full derivation.
"""
function build_climatology(
    loc_hist::DataFrame, forecast_origin::Date;
    period::Int=SEASON_PERIOD, smooth_window::Int=5,
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

# ---------------------------------------------------------------------
# Season-severity estimate (NEW; see header for the negative result)
# ---------------------------------------------------------------------

"""
    location_severity(y, woy, season, clim) -> Union{Missing,Float64}

Estimate one location's current-season severity relative to its
climatology: the `SEVERITY_CENTER` (mean or median) of, over weeks in
the LAST season present in `y` with an observation and a climatology
value `>= MIN_CLIM_LEVEL` (avoids noise/blow-up dividing by near-zero
off-season values) -- optionally restricted to the most recent
`SEVERITY_RECENT_WEEKS` of those -- either the ratio `y / clim`
(`SEVERITY_FORM = :multiplicative`) or the difference `y - clim`
(`:additive`). `missing` if fewer than `MIN_SEVERITY_WEEKS` qualifying
weeks exist yet (e.g. very early in a season).
"""
function location_severity(
    y::AbstractVector{Float64}, woy::Vector{Int}, season::Vector{Int},
    clim::Vector{Float64},
)
    cur = maximum(season)
    idxs = [t for t in eachindex(y) if season[t] == cur]
    if SEVERITY_RECENT_WEEKS !== nothing &&
            length(idxs) > SEVERITY_RECENT_WEEKS
        idxs = idxs[(end - SEVERITY_RECENT_WEEKS + 1):end]
    end
    vals = Float64[]
    for t in idxs
        ismissing(y[t]) && continue
        c = clim[mod1(woy[t], SEASON_PERIOD)]
        c >= MIN_CLIM_LEVEL || continue
        push!(vals, SEVERITY_FORM == :multiplicative ? y[t] / c : y[t] - c)
    end
    length(vals) >= MIN_SEVERITY_WEEKS || return missing
    return SEVERITY_CENTER == :mean ? mean(vals) : median(vals)
end

"""
    pooled_severity(loc_severities) -> Float64

Pool the per-location severity estimates (some possibly `missing`)
into ONE SHARED scalar via `SEVERITY_POOL_STAT`, justified by the
r=0.68 cross-location amplitude correlation (docs/eda/04-cross-
location.md) -- a whole season tends to be uniformly severe or mild,
so a single pooled estimate borrows strength across all 11 locations
rather than trusting each location's own noisy, still-partial-season
estimate alone. Falls back to "no adjustment" (1.0 multiplicative /
0.0 additive) if fewer than 3 locations have a usable estimate.
"""
function pooled_severity(loc_severities::Vector{Union{Missing,Float64}})
    valid = collect(skipmissing(loc_severities))
    neutral = SEVERITY_FORM == :multiplicative ? 1.0 : 0.0
    length(valid) >= 3 || return neutral
    return SEVERITY_POOL_STAT == :median ? median(valid) : mean(valid)
end

# ---------------------------------------------------------------------
# AR(6) + climatology fit and forward simulation
# ---------------------------------------------------------------------

"""
    fit_ar_clim(y, woy, order, clim) -> (coef, resid_sd)

Identical to season/generate.jl: OLS fit of an AR(`order`) model with
intercept and one extra regressor, the climatology value at each
response time. The season-severity term is NOT part of the fit (it
only adjusts the FORWARD simulation below) -- the fitted `gamma`
coefficient stays exactly as season/generate.jl's.
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
        X[row, ncols] = clim[mod1(woy[t], SEASON_PERIOD)]
    end
    coef = X \ yresp
    resid = yresp .- X * coef
    dof = max(nobs - ncols, 1)
    resid_sd = sqrt(sum(abs2, resid) / dof)
    return coef, resid_sd
end

"""
    simulate_paths_severity(y, future_woy, coef, resid_sd, order, clim,
                             severity, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Identical forward simulation to season/generate.jl's
`simulate_paths_clim`, except the FUTURE climatology value fed into
each step is adjusted by the pooled `severity` estimate before being
multiplied by the fitted climatology coefficient:
`clim_eff = SEVERITY_FORM == :multiplicative ? clim * used : clim + used`
where `used = neutral + LAMBDA_SEVERITY * (severity - neutral)`, i.e.
`LAMBDA_SEVERITY` shrinks the raw pooled estimate toward "no
adjustment" (`LAMBDA_SEVERITY = 0.0`, the shipped setting, makes
`used == neutral` and this function numerically identical to
`simulate_paths_clim`).
"""
function simulate_paths_severity(
    y::AbstractVector{Float64}, future_woy::Vector{Int},
    coef::Vector{Float64}, resid_sd::Float64, order::Int,
    clim::Vector{Float64}, severity::Float64, horizons, npaths::Int;
    rng::Random.AbstractRNG,
)
    neutral = SEVERITY_FORM == :multiplicative ? 1.0 : 0.0
    used = neutral + LAMBDA_SEVERITY * (severity - neutral)
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
            c = clim[mod1(future_woy[h], SEASON_PERIOD)]
            c_eff = SEVERITY_FORM == :multiplicative ? c * used : c + used
            pred += coef[order + 2] * c_eff
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
    build_forecast_table(seasons, profile, versions_full, hist_by_loc)
        -> DataFrame

Fit and forecast the AR(6)+climatology+severity+backfill model for
every cross-validation split of every season in `seasons`. Per split,
the pooled severity estimate is computed ONCE from all 11 locations'
in-season-so-far trajectories, then applied identically to every
location's forward simulation (the "shared severity-year effect"
structure).
"""
function build_forecast_table(seasons, profile, versions_full, hist_by_loc)
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
            apply_backfill_correction!(data, profile)
            origin = data.origin_date
            future_woy = [
                week_of_season(origin + Day(7 * h)) for h in HORIZONS
            ]

            # Per-location climatology + severity estimate, then POOL
            # severity across all 11 locations into one shared scalar
            # for this split, before any per-location forecasting.
            clims = Vector{Vector{Float64}}(undef, data.L)
            loc_sev = Vector{Union{Missing,Float64}}(undef, data.L)
            for (li, loc) in enumerate(LOCATIONS)
                clims[li] = build_climatology(hist_by_loc[loc], origin)
                y = Float64.(data.Y[:, li])
                loc_sev[li] = location_severity(
                    y, data.woy, data.season, clims[li],
                )
            end
            severity = pooled_severity(loc_sev)

            for (li, loc) in enumerate(LOCATIONS)
                y = Float64.(data.Y[:, li])
                clim = clims[li]
                coef, resid_sd = fit_ar_clim(y, data.woy, AR_ORDER, clim)
                paths = simulate_paths_severity(
                    y, future_woy, coef, resid_sd, AR_ORDER, clim,
                    severity, HORIZONS, NPATHS; rng=rng,
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

    hist_full = load_series("flu_data_hhs")
    hist_by_loc = Dict(
        loc => hist_full[hist_full.location .== loc, [:origin_date, :wili]]
        for loc in LOCATIONS
    )

    forecast = build_forecast_table(
        (1, 2, 3, 4, 5), profile, versions_full, hist_by_loc,
    )
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    if hub_path !== nothing
        write_submission(forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="seabbs_bot", model_abbr="severity2", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
