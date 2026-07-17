#!/usr/bin/env julia
# simple-round candidate "nowcast" -- DEEP NOWCAST of the reporting
# triangle, stacked on the ar-order sweep's winner (AR(12) + backfill,
# mean WIS 0.3518, experiments/simple-round/ar-order/score.txt) and
# combo's pooled seasonal term (experiments/simple-round/combo,
# 2 harmonics, ridge lambda = 0.3 * nobs on the seasonal columns only).
#
# Every prior backfill variant (backfill/, ar-order/, combo/) corrects
# the reporting triangle with ONE number per (location, delay): the
# median of `settled - vintage`, applied identically to every
# simulated path. That throws away two things a real nowcast would
# keep:
#   1. the SPREAD of the revision, not just its centre -- some
#      (location, delay) cells revise by a little, some by a lot, and
#      that variability is exactly the nowcast uncertainty a forecast
#      should propagate forward, not collapse to a point;
#   2. that revisions are not stationary across the season -- early-
#      season/rising weeks and late-season/falling weeks are
#      under/over-reported differently (docs/eda/02-backfill.md), so a
#      single delay-indexed number pools two different regimes.
#
# This file addresses both. Everything else (AR order, transform,
# window, quantile levels, seed) is unchanged from ar-order/generate.jl.
# LIGHT + ANALYTIC: CSV/DataFrames/Statistics/Random/LinearAlgebra
# only, no Turing.
#
# -- Revision profile (build_revision_profile) --
# For every historical revision event (a (location, origin_date) pair
# with more than one vintage), the event is labelled :rising or
# :falling by `revision_phase`: compares the FRESHEST report seen for
# that reference week against the freshest report seen 3 weeks
# earlier (both from `build_earliest_lookup` -- real-time information
# only, never a settled value, so this never leaks future data). The
# full vector of `settled - vintage` values (not just its median) is
# kept per key, at three tiers of specificity so every (location,
# delay) pair still gets SOME distribution even where a phase split
# would leave too few observations to trust:
#   1. (location, delay, phase)   -- richest
#   2. (location, delay)          -- phase pooled
#   3. delay alone                -- location AND phase pooled
# `min_support=5` (unchanged from every earlier backfill variant)
# gates each tier.
#
# -- Point correction vs propagated nowcast uncertainty --
# The AR+seasonal fit itself still needs ONE deterministic series, so
# recent (delay <= DELAY_CUTOFF) weeks are point-corrected by the
# MEDIAN of the resolved distribution, same as before -- this is the
# `use_phase` toggle (median of the phase-resolved distribution vs the
# old phase-blind per-(location,delay) median).
#
# The new lever is `propagate_nowcast`: at simulation time, instead of
# feeding every one of the 1000 paths the SAME point-corrected tail,
# each path independently BOOTSTRAPS a fresh draw from the resolved
# revision distribution for every still-revising lag in its own
# starting tail (`y_raw[t] + draw`, draw ~ Uniform sample from
# `dists[(location, delay, phase)]`). Those draws differ per path, and
# because they sit in the lags the AR(12) recursion conditions on,
# the disagreement between paths propagates forward through every
# horizon -- this is what makes the nowcast's uncertainty, not just
# its point estimate, show up in the forecast intervals.
#
# -- Ablation (see score.txt for the full numbers) --
#   A. ar12-bf-median      : ar-order's design, reproduced here for a
#                             same-code sanity check against its 0.3518.
#   B. ar12-bf-phase       : + phase-conditioned point correction only
#                             (still one deterministic tail per split).
#   C. ar12-bf-phase-nowcast : + propagated nowcast uncertainty (the
#                             stochastic tail).
#   D. ar12-bf-phase-nowcast-season : + combo's pooled seasonal term.
# D is the model this file recommends; see score.txt's "recommendation"
# section for whether B and C's levers actually paid off and where.
#
# Scope: VALIDATION SEASONS (1, 2) ONLY (docs/contracts.md
# experimental integrity); `training_splits` refuses seasons 3-5
# unless `allow_test_season=true`, which this script never passes.
# The revision profile is built only from origin dates with
# `season_year <= 2016` (matches every earlier backfill variant).
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl
# writes score.txt alongside this file; no hub submission (exploratory
# tuning script, like backfill/generate.jl and combo/search_grid.jl).

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
const AR_ORDER = 12       # ar-order sweep's winner
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12
const DELAY_CUTOFF = 8    # weeks; matches ar-order/combo's backfill window
const MIN_SUPPORT = 5     # min sample size per profile key to trust
const PHASE_LAG = 3       # weeks back compared, for rising/falling label
const N_HARMONICS = 2     # combo's pooled seasonal term
const PERIOD = 52.0
const LAMBDA_FRAC = 0.3   # ridge strength on seasonal cols, as a
                          # fraction of nobs (matches combo)
const VALIDATION_ONLY = (1, 2)
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")

# ---------------------------------------------------------------------
# Real-time season phase (rising / falling), from freshest reports only
# ---------------------------------------------------------------------

"""
    build_earliest_lookup(versions) -> Dict{Tuple{String,Date},Float64}

Earliest (smallest `as_of`) reported wILI for each `(location,
origin_date)`, on the `TRANSFORM` scale: the freshest number a
real-time forecaster would have seen for that reference week, before
any revisions arrived. Used only to label each historical revision
event's local season phase (rising vs falling) -- never a settled
value, so no future information leaks into the label.
"""
function build_earliest_lookup(versions::DataFrame)
    lookup = Dict{Tuple{String,Date},Float64}()
    earliest_as_of = Dict{Tuple{String,Date},Date}()
    for row in eachrow(versions)
        key = (row.location, row.origin_date)
        if !haskey(earliest_as_of, key) || row.as_of < earliest_as_of[key]
            earliest_as_of[key] = row.as_of
            lookup[key] = to_scale(row.wili, TRANSFORM)
        end
    end
    return lookup
end

"""
    revision_phase(lookup, loc, date; lag_weeks=PHASE_LAG) -> Symbol

`:rising` if the freshest report for `date` exceeds the freshest
report `lag_weeks` earlier (both from [`build_earliest_lookup`](@ref));
`:falling` if not; `:unknown` if either side has no freshest report on
record. Used to label historical revision events for
[`build_revision_profile`](@ref).
"""
function revision_phase(
    lookup::Dict{Tuple{String,Date},Float64}, loc::AbstractString,
    date::Date; lag_weeks::Int=PHASE_LAG,
)
    a = get(lookup, (loc, date), nothing)
    b = get(lookup, (loc, date - Day(7 * lag_weeks)), nothing)
    (a === nothing || b === nothing) && return :unknown
    return a > b ? :rising : :falling
end

"""
    current_phase(y, t; lag_weeks=PHASE_LAG) -> Symbol

Same rising/falling label as [`revision_phase`](@ref), applied at
forecast time to one location's own (uncorrected) `ModelData.Y` column
`y` (ascending in time) at row `t` (the forecast origin): compares
`y[t]` (the freshest available report for the most recent week,
typically still under-reported) against `y[t - lag_weeks]`. Uses only
observations already in `y` -- no lookahead.
"""
function current_phase(
    y::AbstractVector{Float64}, t::Int; lag_weeks::Int=PHASE_LAG,
)
    t - lag_weeks < 1 && return :unknown
    return y[t] > y[t - lag_weeks] ? :rising : :falling
end

"""Fold `:unknown` (only possible at the very start of a very short
series; never hit at `window_weeks=104`) onto `:rising` so lookups
always have a definite key."""
resolve_phase(phase::Symbol) = phase == :unknown ? :rising : phase

# ---------------------------------------------------------------------
# Revision profile: full distributions, tiered by (location, delay,
# phase) -> (location, delay) -> delay
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> (dists, tiers, by_loc_delay)

Empirical revision DISTRIBUTIONS (full vectors of `settled - vintage`
on `transform` scale, additive -- the backfill sweep's winning mode,
experiments/simple-round/backfill/score.txt), tiered by specificity:

  1. `(location, delay, phase)` -- richest, kept when it has
     `>= min_support` observations,
  2. `(location, delay)` (phase pooled) -- fallback,
  3. `delay` alone (location AND phase pooled) -- final fallback.

`phase` for a revision event is [`revision_phase`](@ref) at the
origin_date whose value is being revised. `versions` must already be
filtered by the caller to training-set origin dates only (no test
seasons).

Returns `dists` (`Dict{Tuple{String,Int,Symbol},Vector{Float64}}`,
always keyed `(location, delay, phase)` however it was resolved),
`tiers` (same keys, which tier supplied it: `:phase`, `:location`, or
`:pooled`, for reporting), and `by_loc_delay`
(`Dict{Tuple{String,Int},Vector{Float64}}`, phase pooled -- the
phase-blind profile every earlier backfill variant used, kept around
so this file can reproduce that baseline for comparison).
"""
function build_revision_profile(
    versions::DataFrame; transform::Symbol, max_delay::Int, min_support::Int,
)
    lookup = build_earliest_lookup(versions)

    by_phase = Dict{Tuple{String,Int,Symbol},Vector{Float64}}()
    by_loc_delay = Dict{Tuple{String,Int},Vector{Float64}}()
    by_delay = Dict{Int,Vector{Float64}}()

    for g in groupby(versions, [:location, :origin_date])
        settled_idx = argmax(g.as_of)
        settled = to_scale(g.wili[settled_idx], transform)
        settled_as_of = g.as_of[settled_idx]
        loc = g.location[1]
        od = g.origin_date[1]
        phase = revision_phase(lookup, loc, od)
        for row in eachrow(g)
            row.as_of == settled_as_of && continue
            delay = div(Dates.value(row.as_of - row.origin_date), 7)
            (delay < 0 || delay > max_delay) && continue
            vintage = to_scale(row.wili, transform)
            val = settled - vintage
            push!(get!(by_delay, delay, Float64[]), val)
            push!(get!(by_loc_delay, (loc, delay), Float64[]), val)
            if phase != :unknown
                push!(get!(by_phase, (loc, delay, phase), Float64[]), val)
            end
        end
    end

    dists = Dict{Tuple{String,Int,Symbol},Vector{Float64}}()
    tiers = Dict{Tuple{String,Int,Symbol},Symbol}()
    for loc in LOCATIONS, delay in 0:max_delay, phase in (:rising, :falling)
        key = (loc, delay, phase)
        if haskey(by_phase, key) && length(by_phase[key]) >= min_support
            dists[key] = by_phase[key]
            tiers[key] = :phase
        elseif haskey(by_loc_delay, (loc, delay)) &&
               length(by_loc_delay[(loc, delay)]) >= min_support
            dists[key] = by_loc_delay[(loc, delay)]
            tiers[key] = :location
        elseif haskey(by_delay, delay) && length(by_delay[delay]) >= min_support
            dists[key] = by_delay[delay]
            tiers[key] = :pooled
        end
        # else: no entry at all -- this (location, delay) has no
        # support anywhere, even pooled; left untouched at correction
        # time, same as every earlier backfill variant's `haskey`
        # guard.
    end
    return dists, tiers, by_loc_delay
end

# ---------------------------------------------------------------------
# AR(p) + optional ridge-regularised seasonal term (combo's design)
# ---------------------------------------------------------------------

"""
    fit_ar_seasonal(y, order; woy, n_harmonics, lambda_season, period)
        -> (coef, resid_sd)

OLS/ridge fit of an AR(`order`) model with intercept and, when
`n_harmonics > 0`, `n_harmonics` sin/cos pairs of a `period`-week
seasonal cycle evaluated at each row's `woy`. Identical to
`experiments/simple-round/combo/search_grid.jl`'s function of the same
name -- see that file for the full derivation of the ridge choice.
"""
function fit_ar_seasonal(
    y::AbstractVector{Float64}, order::Int;
    woy::Union{Nothing,AbstractVector{Int}}=nothing,
    n_harmonics::Int=0, lambda_season::Float64=0.0, period::Float64=52.0,
)
    n = length(y)
    nobs = n - order
    nobs >= order + 2 ||
        error("series too short for AR($order): n=$n, nobs=$nobs")
    n_season_cols = 2 * n_harmonics
    ncols = order + 1 + n_season_cols
    X = zeros(nobs, ncols)
    yresp = Vector{Float64}(undef, nobs)
    for (row, t) in enumerate((order + 1):n)
        yresp[row] = y[t]
        X[row, 1] = 1.0
        for lag in 1:order
            X[row, lag + 1] = y[t - lag]
        end
        if n_harmonics > 0
            wt = woy[t]
            for k in 1:n_harmonics
                ang = 2pi * k * wt / period
                X[row, order + 2k] = sin(ang)
                X[row, order + 2k + 1] = cos(ang)
            end
        end
    end
    coef = if lambda_season > 0 && n_harmonics > 0
        penalty = zeros(ncols)
        penalty[(order + 2):end] .= lambda_season
        (X' * X + Diagonal(penalty)) \ (X' * yresp)
    else
        X \ yresp
    end
    resid = yresp .- X * coef
    dof = max(nobs - ncols, 1)
    resid_sd = sqrt(sum(abs2, resid) / dof)
    return coef, resid_sd
end

"""
    season_effect(woy_val, coef, order, n_harmonics, period) -> Float64

Fitted seasonal offset at week-of-season `woy_val`. Returns 0.0 when
`n_harmonics == 0`. Identical to combo/search_grid.jl.
"""
function season_effect(
    woy_val::Int, coef::Vector{Float64}, order::Int, n_harmonics::Int,
    period::Float64,
)
    n_harmonics == 0 && return 0.0
    s = 0.0
    for k in 1:n_harmonics
        ang = 2pi * k * woy_val / period
        s += coef[order + 2k] * sin(ang) + coef[order + 2k + 1] * cos(ang)
    end
    return s
end

# ---------------------------------------------------------------------
# Simulation: deterministic tail (baseline) vs propagated nowcast
# uncertainty (bootstrapped tail, redrawn per path)
# ---------------------------------------------------------------------

"""
    simulate_paths(y_point, y_raw, delay_col, loc, dists, coef,
                   resid_sd, order, horizons, npaths, season_offsets;
                   delay_cutoff, phase, propagate_nowcast, rng)
        -> Dict{Int,Vector{Float64}}

Simulate `npaths` Gaussian-innovation sample paths forward, as the
plain AR(p)(+seasonal) simulator, except the STARTING tail (the
`order` most recent lags the recursion conditions on) is handled two
ways:

  - `propagate_nowcast=false`: every path starts from the same fixed,
    point-corrected `y_point` tail (this file's variants A/B, and
    every earlier backfill variant).
  - `propagate_nowcast=true`: at every lag position with `0 <= delay
    <= delay_cutoff` (a still-revising recent week), EACH path
    independently redraws that lag as `y_raw[t] + bootstrap draw from
    dists[(loc, delay, phase)]`, so the AR(p) recursion inherits a
    different plausible settled value for those weeks on every path.
    Because those lags feed directly into the recursion, the
    disagreement between paths propagates forward into every horizon
    -- this is the nowcast-uncertainty lever (variants C/D).
"""
function simulate_paths(
    y_point::AbstractVector{Float64}, y_raw::AbstractVector{Float64},
    delay_col::AbstractVector{Int}, loc::AbstractString, dists::Dict,
    coef::Vector{Float64}, resid_sd::Float64, order::Int, horizons,
    npaths::Int, season_offsets::Vector{Float64};
    delay_cutoff::Int, phase::Symbol, propagate_nowcast::Bool,
    rng::Random.AbstractRNG,
)
    hmax = maximum(horizons)
    T = length(y_point)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    fixed_tail = y_point[(T - order + 1):end]
    for s in 1:npaths
        tail = copy(fixed_tail)
        if propagate_nowcast
            for (i, t) in enumerate((T - order + 1):T)
                d = delay_col[t]
                (d < 0 || d > delay_cutoff) && continue
                key = (loc, d, phase)
                haskey(dists, key) || continue
                tail[i] = y_raw[t] + rand(rng, dists[key])
            end
        end
        for h in 1:hmax
            pred = coef[1]
            for lag in 1:order
                pred += coef[lag + 1] * tail[end - lag + 1]
            end
            pred += season_offsets[h]
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
# Forecast table for one ablation cell
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, versions_full, dists, by_loc_delay;
        use_phase, propagate_nowcast, use_seasonal, model_id)

One ablation cell's hub quantile table across every split of `seasons`
(validation seasons only). `use_phase` selects whether the
deterministic point correction (used to fit the AR+seasonal
coefficients) reads the phase-resolved `dists` median or the
phase-blind `by_loc_delay` median; `propagate_nowcast` selects the
[`simulate_paths`](@ref) tail scheme (always reads `dists`, since
that's the only place per-path bootstrap draws come from);
`use_seasonal` toggles the pooled ridge seasonal term.
"""
function build_forecast_table(
    seasons, versions_full, dists, by_loc_delay;
    use_phase::Bool, propagate_nowcast::Bool, use_seasonal::Bool,
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
            origin = data.origin_date
            for (li, loc) in enumerate(LOCATIONS)
                y_raw = Float64.(data.Y[:, li])
                delay_col = data.delay[:, li]
                phase = resolve_phase(current_phase(y_raw, data.T))

                y_point = copy(y_raw)
                for t in 1:data.T
                    d = delay_col[t]
                    (d < 0 || d > DELAY_CUTOFF) && continue
                    c = if use_phase
                        key = (loc, d, phase)
                        haskey(dists, key) ? median(dists[key]) : nothing
                    else
                        haskey(by_loc_delay, (loc, d)) ?
                            median(by_loc_delay[(loc, d)]) : nothing
                    end
                    c === nothing && continue
                    y_point[t] += c
                end

                n_harm = use_seasonal ? N_HARMONICS : 0
                lambda_season = use_seasonal ?
                    LAMBDA_FRAC * (data.T - AR_ORDER) : 0.0
                coef, resid_sd = fit_ar_seasonal(
                    y_point, AR_ORDER; woy=data.woy, n_harmonics=n_harm,
                    lambda_season=lambda_season, period=PERIOD,
                )
                season_offsets = [
                    season_effect(
                        week_of_season(origin + Day(7 * step)), coef,
                        AR_ORDER, n_harm, PERIOD,
                    )
                    for step in 1:maximum(HORIZONS)
                ]
                paths = simulate_paths(
                    y_point, y_raw, delay_col, loc, dists, coef, resid_sd,
                    AR_ORDER, HORIZONS, NPATHS, season_offsets;
                    delay_cutoff=DELAY_CUTOFF, phase=phase,
                    propagate_nowcast=propagate_nowcast, rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    vals = paths[h]
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
# Scoring helpers
# ---------------------------------------------------------------------

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

"""
    coverage_summary(forecast_df, truth_df) -> NamedTuple

Empirical 50%/90% central-interval coverage of a (single-model) hub
quantile forecast table against `truth_df`: for each forecast task,
checks whether the truth falls inside the `[q0.25, q0.75]` and
`[q0.05, q0.95]` predicted quantiles. Returns `(cov50, cov90, n,
by_horizon)`, `by_horizon` one row per horizon.
"""
function coverage_summary(forecast_df::DataFrame, truth_df::DataFrame)
    joined = innerjoin(forecast_df, truth_df,
        on=[:location, :target_end_date], renamecols="" => "_truth")
    task_cols = [:model_id, :location, :origin_date, :horizon,
                 :target_end_date]
    per_task = combine(groupby(joined, task_cols)) do sdf
        levels = sdf.output_type_id
        values = sdf.value
        obs = sdf.value_truth[1]
        lo50 = values[findfirst(==(0.25), levels)]
        hi50 = values[findfirst(==(0.75), levels)]
        lo90 = values[findfirst(==(0.05), levels)]
        hi90 = values[findfirst(==(0.95), levels)]
        (in50=lo50 <= obs <= hi50, in90=lo90 <= obs <= hi90)
    end
    by_horizon = combine(groupby(per_task, :horizon),
        :in50 => mean => :cov50, :in90 => mean => :cov90, nrow => :n)
    sort!(by_horizon, :horizon)
    return (cov50=mean(per_task.in50), cov90=mean(per_task.in90),
            n=nrow(per_task), by_horizon=by_horizon)
end

"""Score `forecast` against `truth` and return `(mean_wis, sd_wis,
n_tasks, cov50, cov90, scored, cov)`."""
function evaluate(forecast, truth)
    scored = score_forecasts(forecast, truth; scale=:natural)
    summ = wis_summary(scored)
    cov = coverage_summary(forecast, truth)
    return (mean_wis=summ.mean_wis[1], sd_wis=summ.sd_wis[1],
            n_tasks=summ.n_tasks[1], cov50=cov.cov50, cov90=cov.cov90,
            scored=scored, cov=cov)
end

# ---------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------

const VARIANTS = (
    (id="ar12-bf-median", use_phase=false, propagate_nowcast=false,
     use_seasonal=false,
     label="A: ar-order's design (phase-blind median, one tail)"),
    (id="ar12-bf-phase", use_phase=true, propagate_nowcast=false,
     use_seasonal=false,
     label="B: + phase-conditioned point correction"),
    (id="ar12-bf-phase-nowcast", use_phase=true, propagate_nowcast=true,
     use_seasonal=false,
     label="C: + propagated nowcast uncertainty (stochastic tail)"),
    (id="ar12-bf-phase-nowcast-season", use_phase=true,
     propagate_nowcast=true, use_seasonal=true,
     label="D: + pooled seasonal term (recommended)"),
)

function by_group(scored, col)
    combine(groupby(scored, col), :wis => mean => :mean_wis, nrow => :n)
end

function main()
    t0 = time()
    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    truth = load_oracle(HUB_PATH)

    dists, tiers, by_loc_delay = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=DELAY_CUTOFF,
        min_support=MIN_SUPPORT,
    )
    tier_counts = countmap_tiers(tiers)
    println("revision profile: $(length(dists)) (location, delay, " *
            "phase) keys resolved -- $(tier_counts)")

    results = NamedTuple[]
    tables = Dict{String,DataFrame}()
    for v in VARIANTS
        forecast = build_forecast_table(
            VALIDATION_ONLY, versions_full, dists, by_loc_delay;
            use_phase=v.use_phase, propagate_nowcast=v.propagate_nowcast,
            use_seasonal=v.use_seasonal, model_id=v.id,
        )
        r = evaluate(forecast, truth)
        tables[v.id] = forecast
        push!(results, (id=v.id, label=v.label, r...))
        println("$(v.id): mean_wis=$(round(r.mean_wis; digits=4)) " *
                "sd_wis=$(round(r.sd_wis; digits=4)) " *
                "cov50=$(round(r.cov50; digits=3)) " *
                "cov90=$(round(r.cov90; digits=3))")
    end

    by_id = Dict(r.id => r for r in results)
    a, b, c, d = (by_id[v.id] for v in VARIANTS)

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "deep nowcast -- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "reference: ar-order sweep's AR(12)+backfill = 0.3518")
        println(io, "           (experiments/simple-round/ar-order/score.txt)")
        println(io)
        println(io, "revision profile tiers (of $(length(dists)) resolved " *
                     "(location, delay, phase) keys):")
        for (tier, n) in sort(collect(tier_counts); by=first)
            println(io, "  $(tier): $(n)")
        end
        println(io)
        println(io, "ablation (sorted by mean_wis, ascending):")
        println(io, rpad("id", 32) * rpad("mean_wis", 11) *
                     rpad("sd_wis", 10) * rpad("cov50", 9) *
                     rpad("cov90", 9) * "n_tasks")
        sorted_results = sort(results; by=r -> r.mean_wis)
        for r in sorted_results
            println(io,
                rpad(r.id, 32) * rpad(string(round(r.mean_wis; digits=4)), 11) *
                rpad(string(round(r.sd_wis; digits=4)), 10) *
                rpad(string(round(r.cov50; digits=3)), 9) *
                rpad(string(round(r.cov90; digits=3)), 9) *
                string(r.n_tasks),
            )
        end
        println(io)
        for r in results
            println(io, "$(r.label)")
        end
        println(io)

        println(io, "-- lever-by-lever deltas (mean WIS, negative = better) --")
        println(io, "B - A (phase-conditioned point correction): " *
                     "$(round(b.mean_wis - a.mean_wis; digits=4))")
        println(io, "C - B (propagated nowcast uncertainty):     " *
                     "$(round(c.mean_wis - b.mean_wis; digits=4))")
        println(io, "D - C (pooled seasonal term):                " *
                     "$(round(d.mean_wis - c.mean_wis; digits=4))")
        println(io)
        println(io, "-- lever-by-lever deltas (coverage, nominal " *
                     "0.50 / 0.90) --")
        println(io, "A: cov50=$(round(a.cov50; digits=3)) " *
                     "cov90=$(round(a.cov90; digits=3))")
        println(io, "B: cov50=$(round(b.cov50; digits=3)) " *
                     "cov90=$(round(b.cov90; digits=3))")
        println(io, "C: cov50=$(round(c.cov50; digits=3)) " *
                     "cov90=$(round(c.cov90; digits=3))")
        println(io, "D: cov50=$(round(d.cov50; digits=3)) " *
                     "cov90=$(round(d.cov90; digits=3))")

        println(io)
        println(io, "-- coverage by horizon: C (no nowcast-uncertainty) " *
                     "vs D (with it) --")
        c_by_h = c.cov.by_horizon
        d_by_h = d.cov.by_horizon
        for h in sort(c_by_h.horizon)
            crow = c_by_h[c_by_h.horizon .== h, :][1, :]
            drow = d_by_h[d_by_h.horizon .== h, :][1, :]
            println(io, "h=$(h): C cov50=$(round(crow.cov50; digits=3)) " *
                         "cov90=$(round(crow.cov90; digits=3))  |  " *
                         "D cov50=$(round(drow.cov50; digits=3)) " *
                         "cov90=$(round(drow.cov90; digits=3))")
        end

        println(io)
        println(io, "-- breakdown by location (winner: D) --")
        by_loc_d = by_group(d.scored, :location)
        by_loc_a = by_group(a.scored, :location)
        merged_loc = innerjoin(
            by_loc_d, by_loc_a; on=:location, renamecols="_d" => "_a",
        )
        merged_loc.improvement = merged_loc.mean_wis_a .- merged_loc.mean_wis_d
        sort!(merged_loc, :improvement; rev=true)
        for row in eachrow(merged_loc)
            println(io, rpad(row.location, 16) *
                         "D=$(round(row.mean_wis_d; digits=4)) " *
                         "A=$(round(row.mean_wis_a; digits=4)) " *
                         "improvement=$(round(row.improvement; digits=4))")
        end

        println(io)
        println(io, "-- breakdown by season (winner: D) --")
        d.scored.season_year = season_year.(d.scored.origin_date)
        a.scored.season_year = season_year.(a.scored.origin_date)
        by_season_d = by_group(d.scored, :season_year)
        by_season_a = by_group(a.scored, :season_year)
        merged_season = innerjoin(
            by_season_d, by_season_a; on=:season_year,
            renamecols="_d" => "_a",
        )
        merged_season.improvement = merged_season.mean_wis_a .-
                                     merged_season.mean_wis_d
        for row in eachrow(merged_season)
            println(io, "season $(row.season_year): " *
                         "D=$(round(row.mean_wis_d; digits=4)) " *
                         "A=$(round(row.mean_wis_a; digits=4)) " *
                         "improvement=$(round(row.improvement; digits=4))")
        end

        println(io)
        println(io, "-- breakdown by horizon (winner: D) --")
        by_h_d = by_group(d.scored, :horizon)
        by_h_a = by_group(a.scored, :horizon)
        merged_h = innerjoin(
            by_h_d, by_h_a; on=:horizon, renamecols="_d" => "_a",
        )
        merged_h.improvement = merged_h.mean_wis_a .- merged_h.mean_wis_d
        sort!(merged_h, :horizon)
        for row in eachrow(merged_h)
            println(io, "h=$(row.horizon): " *
                         "D=$(round(row.mean_wis_d; digits=4)) " *
                         "A=$(round(row.mean_wis_a; digits=4)) " *
                         "improvement=$(round(row.improvement; digits=4))")
        end

        println(io)
        best = sorted_results[1]
        println(io, "=== best variant: $(best.id) ===")
        println(io, "mean_wis=$(round(best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(best.sd_wis; digits=4)) " *
                     "n_tasks=$(best.n_tasks)")
        vs_ref = 0.3518 - best.mean_wis
        println(io, "vs ar-order sweep's AR(12)+backfill (0.3518): " *
                     "$(round(vs_ref; digits=4)) " *
                     "($(round(100 * vs_ref / 0.3518; digits=2))%)")
    end

    dt = round(time() - t0; digits=1)
    println("\nwrote score.txt in $(dt)s total")
    return results, tables
end

"""Tally how many resolved `dists` keys came from each tier (`:phase`,
`:location`, `:pooled`), for the score.txt report."""
function countmap_tiers(tiers::Dict)
    counts = Dict{Symbol,Int}()
    for (_, tier) in tiers
        counts[tier] = get(counts, tier, 0) + 1
    end
    return counts
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
