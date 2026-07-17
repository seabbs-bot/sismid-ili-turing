#!/usr/bin/env julia
# sweep.jl -- exploration script for the TRANSFORM family, simple-model
# wide round (sismid-ili-turing). Not a submission driver; its job is to
# decide, on validation-season WIS alone, which per-location modelling
# scale generate.jl should use for the analytic AR(6) baseline
# (submissions/nfidd-ar6/generate_forecasts.jl currently uses fourth-root
# globally).
#
# Deliberately avoids `using SismidILITuring` / Turing for the same
# reason nfidd-ar6/generate_forecasts.jl does (box shared with other
# agents' fits this round): only CSV/DataFrames/Dates/Statistics/Random/
# LinearAlgebra. Scoring is done with a small self-contained WIS
# implementation (same formula as src/scoring.jl's `wis`, no ScoringRules
# dependency) so this stays a fully standalone, few-second script.
#
# Candidates (docs/eda/01-series-overview.md's transform-comparison
# table plus two not in `src/core.jl`'s TRANSFORMS: identity, sqrt):
#   identity, sqrt, fourth-root (current baseline), log (EPS-offset),
#   log1p, a pooled fitted Box-Cox power (EDA: lambda ~ 0.13), and a
#   PER-LOCATION fitted Box-Cox power (one power per location, fit on
#   this script's own quick grid-search MLE -- see `fit_boxcox_lambda`).
#
# All candidates are scored on validation seasons (1, 2) ONLY, natural
# scale, against the hub oracle -- see docs/contracts.md / docs/brief.md
# experimental integrity: seasons 3-5 are held out and never touched here.

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))

# `build_model_data` always routes through the global `TRANSFORMS` dict
# (`src/core.jl`) via `to_scale`. This script always builds data at
# `:identity` (raw wILI) and applies its own per-location transform
# column-wise afterwards (see `build_forecast_table`), so `:identity`
# needs to exist as a no-op entry. This mutates only this process's
# in-memory `TRANSFORMS` Dict, not the `src/core.jl` file on disk -- safe
# alongside other agents editing the repo concurrently this round.
TRANSFORMS[:identity] = (w -> Float64(w), x -> x)

const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717

# ---------------------------------------------------------------------
# Transform family: simple (unshifted) power transforms w -> w^p, with
# p=0 meaning log(max(w, EPS)) (the existing `:log` convention in
# `src/core.jl`). Built locally rather than by mutating the shared
# `TRANSFORMS` dict in src/core.jl, since several agents are editing
# this repo concurrently this round and that file is common ground.
# `fourthroot` (p=0.25) and `log`/`log1p` below are lifted straight from
# `TRANSFORMS` for consistency with the rest of the package.
# ---------------------------------------------------------------------

"""
    power_pair(p) -> (fwd, inv)

Forward/inverse pair for the power transform `w -> max(w, 0)^p`
(`p == 0` falls back to `log(max(w, EPS))`, matching `:log` in
`src/core.jl`). The inverse is sign-preserving for non-integer `1/p`
so a negative latent draw never raises a `DomainError`; callers already
clamp the final natural-scale value at 0 (as `generate_forecasts.jl`
does), so the sign only matters for avoiding that error, not for the
output.
"""
function power_pair(p::Float64)
    p == 0.0 && return (w -> log(max(w, EPS)), x -> exp(x))
    fwd = w -> max(w, 0.0)^p
    inv = x -> x >= 0 ? x^(1 / p) : -((-x)^(1 / p))
    return fwd, inv
end

const CANDIDATES = Dict{Symbol,Tuple{Function,Function}}(
    :identity    => power_pair(1.0),
    :sqrt        => power_pair(0.5),
    :fourthroot  => TRANSFORMS[:fourthroot],
    :log         => TRANSFORMS[:log],
    :log1p       => TRANSFORMS[:log1p],
    :boxcox_p013 => power_pair(0.13),  # pooled Box-Cox MLE, EDA report 01
)

# ---------------------------------------------------------------------
# Per-location Box-Cox power: quick grid-search MLE on the finalized
# series (data/flu_data_hhs.csv), filtered to season_year <= 2016 (the
# validation+history window used throughout docs/eda/ -- see
# docs/eda/scripts/common.jl), matching the exponent convention used
# for :boxcox_p013 above (`w -> w^p`, not the classical shifted
# `(w^lambda - 1)/lambda` Box-Cox statistic, so the fitted power plugs
# straight into `power_pair`).
# ---------------------------------------------------------------------

eda_season_year(dt::Date) = dayofyear(dt) >= 205 ? year(dt) : year(dt) - 1

"""
    fit_boxcox_power(y) -> Float64

Grid-search Box-Cox MLE power `p` (in `-0.6:0.02:1.0`) maximising the
classical profile log-likelihood on strictly positive `y`:
`(p - 1) * sum(log(y)) - (n / 2) * log(var((y.^p .- 1) ./ p))`, with the
`p == 0` branch using `log.(y)` in place of `(y.^p .- 1) ./ p` (the
standard limiting case). Returns the maximising `p`.
"""
function fit_boxcox_power(y::AbstractVector{Float64})
    y = filter(>(0), y)
    n = length(y)
    logsum = sum(log, y)
    grid = -0.6:0.02:1.0
    best_p, best_ll = 0.0, -Inf
    for p in grid
        bc = p == 0 ? log.(y) : (y .^ p .- 1) ./ p
        ll = (p - 1) * logsum - (n / 2) * log(var(bc))
        if ll > best_ll
            best_ll = ll
            best_p = p
        end
    end
    return best_p
end

function per_location_powers()
    d = CSV.read(joinpath(PKG_DIR, "data", "flu_data_hhs.csv"), DataFrame)
    d.origin_date = Date.(d.origin_date)
    d.season_year = eda_season_year.(d.origin_date)
    d = filter(row -> row.season_year <= 2016, d)
    powers = Dict{String,Float64}()
    for loc in LOCATIONS
        y = Float64.(d.wili[d.location .== loc])
        powers[loc] = fit_boxcox_power(y)
    end
    return powers
end

# ---------------------------------------------------------------------
# AR(6) fit/simulate, copied unchanged from
# submissions/nfidd-ar6/generate_forecasts.jl (the model this round is
# tuning is the transform only, not the AR mechanics).
# ---------------------------------------------------------------------

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
    build_forecast_table(seasons, fwd_of, inv_of) -> DataFrame

Same shape/discipline as nfidd-ar6's `build_forecast_table`, but the
transform is per-location: `fwd_of(loc)`/`inv_of(loc)` return the
forward/inverse functions for location `loc`. A single global transform
(the common case) just returns the same pair for every location. Data
is always built at `transform=:identity` (raw wILI, no scale change) so
the per-location transform can be applied column-wise here.
"""
function build_forecast_table(seasons, fwd_of, inv_of)
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
                split; Dmax=12, transform=:identity, window_weeks=104,
            )
            origin = data.origin_date
            for (li, loc) in enumerate(LOCATIONS)
                fwd, inv = fwd_of(loc), inv_of(loc)
                y = fwd.(Float64.(data.Y[:, li]))
                coef, resid_sd = fit_ar(y, AR_ORDER)
                paths = simulate_paths(
                    y, coef, resid_sd, AR_ORDER, HORIZONS, NPATHS; rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    vals = paths[h]
                    for q in QUANTILE_LEVELS
                        qval = quantile(vals, q)
                        nat = max(inv(qval), 0.0)
                        push!(rows, (
                            "sweep", loc, origin, h, target_end,
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
# Self-contained WIS (same formula as src/scoring.jl's `wis`, no
# ScoringRules dependency -- see file header).
# ---------------------------------------------------------------------

function wis(observation::Float64, values::AbstractVector{Float64},
        levels::AbstractVector{Float64})
    tol = 1e-8
    median_idx = findfirst(a -> abs(a - 0.5) < tol, levels)
    median = values[median_idx]
    lower_levels = filter(a -> a < 0.5 - tol, levels)
    K = length(lower_levels)
    is_sum = 0.0
    for a in lower_levels
        lower = values[findfirst(x -> abs(x - a) < tol, levels)]
        upper = values[findfirst(x -> abs(x - (1 - a)) < tol, levels)]
        alpha_k = 2 * a
        is_k = (upper - lower) +
               (2 / alpha_k) * max(lower - observation, 0.0) +
               (2 / alpha_k) * max(observation - upper, 0.0)
        is_sum += (alpha_k / 2) * is_k
    end
    denom = K + 0.5
    return (0.5 * abs(observation - median) + is_sum) / denom
end

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

"""Score `forecast` (hub quantile table) against `truth`; returns a
per-task DataFrame with one `wis` row per (location, origin_date,
horizon)."""
function score(forecast::DataFrame, truth::DataFrame)
    joined = innerjoin(forecast, truth,
        on=[:location, :target_end_date], renamecols="" => "_truth")
    task_cols = [:location, :origin_date, :horizon, :target_end_date]
    combine(groupby(joined, task_cols)) do sdf
        (wis=wis(sdf.value_truth[1], sdf.value, sdf.output_type_id),)
    end
end

# ---------------------------------------------------------------------
# Run the sweep
# ---------------------------------------------------------------------

function main()
    t0 = time()
    truth = load_oracle(HUB_PATH)
    powers = per_location_powers()
    println("per-location fitted Box-Cox power (grid MLE, season_year <= " *
            "2016):")
    for loc in LOCATIONS
        println("  ", rpad(loc, 14), round(powers[loc]; digits=3))
    end

    results = DataFrame(
        candidate=String[], mean_wis=Float64[], sd_wis=Float64[],
        n_tasks=Int[],
    )
    scored_by_candidate = Dict{String,DataFrame}()

    for (name, (fwd, inv)) in CANDIDATES
        fwd_of = _ -> fwd
        inv_of = _ -> inv
        forecast = build_forecast_table((1, 2), fwd_of, inv_of)
        scored = score(forecast, truth)
        scored_by_candidate[String(name)] = scored
        push!(results, (
            String(name), mean(scored.wis), std(scored.wis), nrow(scored),
        ))
    end

    # Per-location optimal: each location uses its own fitted Box-Cox
    # power instead of one shared choice.
    fwd_of_loc = loc -> power_pair(powers[loc])[1]
    inv_of_loc = loc -> power_pair(powers[loc])[2]
    forecast_perloc = build_forecast_table((1, 2), fwd_of_loc, inv_of_loc)
    scored_perloc = score(forecast_perloc, truth)
    scored_by_candidate["per_location_boxcox"] = scored_perloc
    push!(results, (
        "per_location_boxcox", mean(scored_perloc.wis), std(scored_perloc.wis),
        nrow(scored_perloc),
    ))

    # Targeted mixed candidates: `log` won the global sweep above, and
    # the naive fully-per-location Box-Cox candidate above blew up
    # because several locations' fitted powers are negative (unshifted
    # `w^p` diverges near the many near-zero wILI weeks when p < 0).
    # Region 9 is the one location the EDA flags as genuinely wanting a
    # HIGHER (more sqrt-like, less log-like) power
    # (docs/eda/07-region9-deepdive.md; fitted here at p=$(round(
    # powers["HHS Region 9"]; digits=2)), positive and so numerically
    # safe) -- test overriding just that one location against the
    # otherwise-global `log` winner, both with its own fitted power and
    # with plain `sqrt` (the team's stated "Poisson-like" heuristic).
    for (mix_name, r9_pair) in (
        ("log_r9_boxcox", power_pair(powers["HHS Region 9"])),
        ("log_r9_sqrt", power_pair(0.5)),
    )
        fwd_of_mix = loc -> loc == "HHS Region 9" ? r9_pair[1] : TRANSFORMS[:log][1]
        inv_of_mix = loc -> loc == "HHS Region 9" ? r9_pair[2] : TRANSFORMS[:log][2]
        forecast_mix = build_forecast_table((1, 2), fwd_of_mix, inv_of_mix)
        scored_mix = score(forecast_mix, truth)
        scored_by_candidate[mix_name] = scored_mix
        push!(results, (
            mix_name, mean(scored_mix.wis), std(scored_mix.wis),
            nrow(scored_mix),
        ))
    end

    sort!(results, :mean_wis)
    println("\n=== validation WIS by transform candidate (natural scale) ===")
    println(results)

    best = results.candidate[1]
    println("\nbest candidate: $(best), mean_wis=" *
            "$(round(results.mean_wis[1]; digits=4)), sd_wis=" *
            "$(round(results.sd_wis[1]; digits=4))")

    println("\n=== per-location mean WIS: fourthroot vs log vs " *
            "per_location_boxcox vs log_r9_boxcox ===")
    for cand in ("fourthroot", "log", "per_location_boxcox", "log_r9_boxcox")
        sdf = scored_by_candidate[cand]
        byloc = combine(groupby(sdf, :location), :wis => mean => :mean_wis)
        sort!(byloc, :location)
        println("-- $(cand) --")
        println(byloc)
    end

    println("\n=== per-season mean WIS (best candidate: $(best)) ===")
    sdf = scored_by_candidate[best]
    sdf.season = [row.origin_date <= Date(2016, 8, 1) ? 1 : 2 for
                  row in eachrow(sdf)]
    byseason = combine(groupby(sdf, :season), :wis => mean => :mean_wis,
        :wis => std => :sd_wis, nrow => :n)
    println(byseason)

    println("\ntotal sweep time: $(round(time() - t0; digits=1))s")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
