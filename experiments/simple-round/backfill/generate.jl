#!/usr/bin/env julia
# backfill correction sweep -- simple-round, BACKFILL CORRECTION family.
#
# Starting point: `seabbs_bot-ar6bf`
# (submissions/seabbs_bot-ar6bf/generate_forecasts.jl), an AR(6)
# per-location baseline (as `nfidd-ar6`) with an ADDITIVE, PER-LOCATION,
# MEDIAN backfill correction on the most recent 8 weeks. That variant
# scored 0.359 mean WIS on validation vs 0.368 for the uncorrected
# AR(6) baseline (submissions/seabbs_bot-ar6bf/README.md), with the gain
# concentrated in HHS Region 2 and Region 9.
#
# This sweep generalises the correction along the four axes the ar6bf
# design fixed by choice, to check whether a different setting on any
# of them does better:
#   - MODE:    additive (settled - vintage) vs multiplicative
#              (settled / vintage), both on the fourth-root modelling
#              scale
#   - WINDOW:  how many of the most recent, still-revisable weeks get
#              corrected before the delay index runs out of support --
#              {4, 6, 8, 10, 12}; ar6bf used 8
#   - POOLING: per-location profile (ar6bf's choice) vs a single
#              profile pooled across all locations at each delay
#   - STAT:    median (ar6bf's choice) vs mean, as the point summary of
#              the per-(location,delay) (or per-delay, if pooled) set
#              of revisions
#
# Everything else -- AR order, transform, path simulation, quantile
# levels, seed -- is identical to `nfidd-ar6` /
# `seabbs_bot-ar6bf`. LIGHT + ANALYTIC: CSV/DataFrames/Statistics/
# LinearAlgebra only, no Turing.
#
# Scope: VALIDATION SEASONS (1, 2) ONLY, scored against the local hub
# clone's oracle (docs/contracts.md experimental integrity) -- this is
# a tuning sweep, not a submission driver. The revision profile itself
# is built only from origin dates with `season_year <= 2016`
# (pre-2015 history plus the two validation seasons; matches ar6bf),
# so the same "no test-season data anywhere" discipline applies.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl
# writes score.txt alongside this file; does not write a hub submission
# (no hub_path argument -- this is exploratory, not a candidate for
# `submissions/`).

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
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12          # matches ar6bf's build_model_data Dmax
const MIN_SUPPORT = 5    # min sample size per profile key to trust
const VALIDATION_ONLY = (1, 2)
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")

# ---------------------------------------------------------------------
# Backfill correction profile (generalised over mode/pooling/stat)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support,
                            mode, pooled, stat) -> Dict

Empirical revision profile on the `transform` scale, generalising
`seabbs_bot-ar6bf`'s (additive, per-location, median) design along
three axes:

  - `mode`: `:additive` stores `settled - vintage`; `:multiplicative`
    stores `settled / vintage` (rows with `|vintage| < 1e-6` are
    skipped for `:multiplicative` to avoid a near-zero-division blow
    up).
  - `pooled`: if `true`, keys are `delay::Int` (one shared profile
    across all locations); if `false`, keys are
    `(location::String, delay::Int)` (ar6bf's per-location design).
  - `stat`: `:median` or `:mean` of the per-key set of values.

`delay = weeks(as_of - origin_date)`, `settled` is the value at each
`(location, origin_date)` group's largest tracked `as_of`
(docs/eda/02-backfill.md's settled-value proxy). `versions` must
already be filtered by the caller to the desired origin dates (here:
training set only, no test seasons). Keys with fewer than
`min_support` observations are dropped.
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

Nudge `data.Y` in place, at every `(t, l)` with `0 <= data.delay[t, l]
<= delay_cutoff` and a matching `profile` entry, by adding
(`mode = :additive`) or multiplying by (`mode = :multiplicative`) the
profile's correction. `pooled` selects whether the lookup key is
`delay` alone or `(location, delay)`. Missing entries and delays
outside the profile's support are left untouched.
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
# AR(6) fit + forecast (identical to nfidd-ar6 / seabbs_bot-ar6bf)
# ---------------------------------------------------------------------

"""
    fit_ar(y, order) -> (coef, resid_sd)

OLS fit of an AR(`order`) model with intercept to `y` (ascending in
time, no missing values). `coef = [c, phi_1, ..., phi_order]`,
`resid_sd` the in-sample residual standard deviation.
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
    build_forecast_table(seasons, profile; mode, pooled, delay_cutoff,
                          versions_full) -> DataFrame

Fit and forecast the AR(6)+backfill-correction model for every
cross-validation split of every season in `seasons` (here always the
validation seasons), returning the combined hub quantile table
(docs/contracts.md schema).
"""
function build_forecast_table(
    seasons, profile, versions_full; mode::Symbol, pooled::Bool,
    delay_cutoff::Int, model_id::String,
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
            apply_backfill_correction!(
                data, profile; mode=mode, pooled=pooled,
                delay_cutoff=delay_cutoff,
            )
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
# Sweep
# ---------------------------------------------------------------------

const MODES = (:additive, :multiplicative)
const WINDOWS = (4, 6, 8, 10, 12)
const POOLINGS = (false, true)   # false = per-location, true = pooled
const STATS = (:median, :mean)

function run_variant(versions_full, training_versions, truth, mode, window,
        pooled, stat)
    profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=window,
        min_support=MIN_SUPPORT, mode=mode, pooled=pooled, stat=stat,
    )
    model_id = "sweep-tmp"
    forecast = build_forecast_table(
        VALIDATION_ONLY, profile, versions_full; mode=mode, pooled=pooled,
        delay_cutoff=window, model_id=model_id,
    )
    scored = score_forecasts(forecast, truth; scale=:natural)
    summ = wis_summary(scored)
    return (summary=summ[1, :], scored=scored, profile=profile)
end

function run_baseline(versions_full, truth)
    # No correction at all -- reproduces nfidd-ar6 on seasons 1,2.
    empty_profile = Dict{Any,Float64}()
    forecast = build_forecast_table(
        VALIDATION_ONLY, empty_profile, versions_full; mode=:additive,
        pooled=false, delay_cutoff=0, model_id="baseline-ar6",
    )
    scored = score_forecasts(forecast, truth; scale=:natural)
    summ = wis_summary(scored)
    return (summary=summ[1, :], scored=scored)
end

function main()
    t0 = time()
    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    truth = load_oracle(HUB_PATH)

    println("=== baseline: plain AR(6), no backfill correction ===")
    base = run_baseline(versions_full, truth)
    println("mean_wis=$(round(base.summary.mean_wis; digits=4)) " *
            "sd_wis=$(round(base.summary.sd_wis; digits=4)) " *
            "n_tasks=$(base.summary.n_tasks)")

    results = NamedTuple[]
    for mode in MODES, window in WINDOWS, pooled in POOLINGS, stat in STATS
        r = run_variant(
            versions_full, training_versions, truth, mode, window, pooled,
            stat,
        )
        push!(results, (
            mode=mode, window=window, pooled=pooled, stat=stat,
            mean_wis=r.summary.mean_wis, sd_wis=r.summary.sd_wis,
            n_tasks=r.summary.n_tasks,
        ))
        println("mode=$mode window=$window " *
                "pooled=$pooled stat=$stat -> " *
                "mean_wis=$(round(r.summary.mean_wis; digits=4)) " *
                "sd_wis=$(round(r.summary.sd_wis; digits=4))")
    end

    sort!(results; by=r -> r.mean_wis)
    best = results[1]
    println("\n=== best variant ===")
    println(best)

    # Re-run best variant to get its scored table for the breakdown.
    best_run = run_variant(
        versions_full, training_versions, truth, best.mode, best.window,
        best.pooled, best.stat,
    )

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "backfill correction sweep -- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "baseline (plain AR(6), no correction):")
        println(io, "  mean_wis=$(round(base.summary.mean_wis; digits=4)) " *
                     "sd_wis=$(round(base.summary.sd_wis; digits=4)) " *
                     "n_tasks=$(base.summary.n_tasks)")
        println(io, "reference points from " *
                     "submissions/seabbs_bot-ar6bf/README.md: " *
                     "nfidd-ar6=0.368 (sd 0.471), " *
                     "seabbs_bot-ar6bf (additive/per-loc/median/w=8)" *
                     "=0.359 (sd 0.452)")
        println(io)
        println(io, "full sweep (sorted by mean_wis, ascending):")
        println(io, rpad("mode", 15) * rpad("window", 8) *
                     rpad("pooled", 8) * rpad("stat", 8) *
                     rpad("mean_wis", 12) * "sd_wis")
        for r in results
            println(io,
                rpad(String(r.mode), 15) * rpad(string(r.window), 8) *
                rpad(r.pooled ? "pooled" : "per-loc", 8) *
                rpad(String(r.stat), 8) *
                rpad(string(round(r.mean_wis; digits=4)), 12) *
                string(round(r.sd_wis; digits=4)),
            )
        end
        println(io)
        println(io, "=== best variant ===")
        println(io, "mode=$(best.mode) window=$(best.window) " *
                     "pooled=$(best.pooled) stat=$(best.stat)")
        println(io, "mean_wis=$(round(best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(best.sd_wis; digits=4)) " *
                     "n_tasks=$(best.n_tasks)")
        vs_corrected = base.summary.mean_wis - best.mean_wis
        vs_pct = 100 * vs_corrected / base.summary.mean_wis
        println(io, "improvement vs uncorrected AR(6) baseline: " *
                     "$(round(vs_corrected; digits=4)) " *
                     "($(round(vs_pct; digits=2))%)")

        println(io)
        println(io, "-- breakdown by location (best variant) --")
        by_loc = combine(groupby(best_run.scored, :location),
            :wis => mean => :mean_wis, nrow => :n)
        base_by_loc = combine(groupby(base.scored, :location),
            :wis => mean => :mean_wis)
        merged_loc = innerjoin(
            by_loc, base_by_loc; on=:location, renamecols="_best" => "_base",
        )
        merged_loc.improvement = merged_loc.mean_wis_base .-
                                  merged_loc.mean_wis_best
        sort!(merged_loc, :improvement; rev=true)
        for row in eachrow(merged_loc)
            println(io, rpad(row.location, 16) *
                         "best=$(round(row.mean_wis_best; digits=4)) " *
                         "base=$(round(row.mean_wis_base; digits=4)) " *
                         "improvement=$(round(row.improvement; digits=4))")
        end

        println(io)
        println(io, "-- breakdown by season (best variant) --")
        best_run.scored.season_year = season_year.(
            best_run.scored.origin_date,
        )
        base.scored.season_year = season_year.(base.scored.origin_date)
        by_season = combine(groupby(best_run.scored, :season_year),
            :wis => mean => :mean_wis, nrow => :n)
        base_by_season = combine(groupby(base.scored, :season_year),
            :wis => mean => :mean_wis)
        merged_season = innerjoin(
            by_season, base_by_season; on=:season_year,
            renamecols="_best" => "_base",
        )
        merged_season.improvement = merged_season.mean_wis_base .-
                                     merged_season.mean_wis_best
        for row in eachrow(merged_season)
            println(io, "season $(row.season_year): " *
                         "best=$(round(row.mean_wis_best; digits=4)) " *
                         "base=$(round(row.mean_wis_base; digits=4)) " *
                         "improvement=$(round(row.improvement; digits=4))")
        end

        println(io)
        println(io, "-- breakdown by horizon (best variant) --")
        by_h = combine(groupby(best_run.scored, :horizon),
            :wis => mean => :mean_wis, nrow => :n)
        base_by_h = combine(groupby(base.scored, :horizon),
            :wis => mean => :mean_wis)
        merged_h = innerjoin(
            by_h, base_by_h; on=:horizon, renamecols="_best" => "_base",
        )
        merged_h.improvement = merged_h.mean_wis_base .-
                                merged_h.mean_wis_best
        sort!(merged_h, :horizon)
        for row in eachrow(merged_h)
            println(io, "h=$(row.horizon): " *
                         "best=$(round(row.mean_wis_best; digits=4)) " *
                         "base=$(round(row.mean_wis_base; digits=4)) " *
                         "improvement=$(round(row.improvement; digits=4))")
        end
    end

    dt = round(time() - t0; digits=1)
    println("\nwrote score.txt in $(dt)s total")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
