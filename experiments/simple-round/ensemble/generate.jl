#!/usr/bin/env julia
# ensemble of simple analytic models -- simple-round, ENSEMBLE family.
#
# LIGHT + ANALYTIC: CSV/DataFrames/Statistics/LinearAlgebra only, no
# Turing. Four cheap, independent per-location MEMBERS:
#
#   ar6         -- plain AR(6), OLS + Gaussian-innovation path
#                  simulation (`submissions/nfidd-ar6/generate_forecasts.jl`)
#   ar6bf       -- AR(6) + the additive, per-location, median backfill
#                  correction from `submissions/seabbs_bot-ar6bf/`
#                  (scored 0.359 mean WIS alone, vs 0.368 for ar6)
#   climatology -- seasonal-naive: empirical quantiles of history at
#                  the same week-of-season (+/- a small band), no
#                  fitting at all
#   ses         -- simple exponential smoothing (level only, no
#                  trend), Gaussian-innovation path simulation with
#                  variance growing with horizon
#
# and three ways to COMBINE their quantile forecasts pointwise, per
# (location, origin_date, horizon, quantile level):
#
#   mean          -- simple average across members
#   median        -- pointwise median across members
#   wis_weighted  -- weighted average, weights = 1/mean_wis per member
#                    (tuned on validation only, see `main`)
#
# All three preserve monotonicity in the quantile level automatically:
# each member's own quantile function is non-decreasing in the level,
# and any order statistic (median) or convex combination (mean,
# weighted average) of several componentwise-ordered vectors is itself
# non-decreasing -- no post-hoc sorting needed.
#
# Scope: VALIDATION SEASONS (1, 2) ONLY, scored against the local hub
# clone's oracle (docs/contracts.md experimental integrity) -- this is
# a tuning/selection sweep, not a submission driver.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl
# writes score.txt alongside this file; does not write a hub
# submission (no hub_path argument -- exploratory, not a candidate for
# `submissions/`).

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra
using Printf

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
const DMAX = 12              # matches ar6bf's build_model_data Dmax
const DELAY_CUTOFF = 8       # ar6bf's chosen backfill window
const MIN_SUPPORT = 5        # min sample size per profile key to trust
const CLIMO_BAND = 2         # +/- weeks-of-season pooled for climatology
const CLIMO_MIN_N = 8        # min pooled n before falling back to full history
const SES_ALPHAS = 0.1:0.1:0.9
const VALIDATION_ONLY = (1, 2)
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")
const MEMBER_NAMES = ("ar6", "ar6bf", "climatology", "ses")

# ---------------------------------------------------------------------
# ar6 / ar6bf members: AR(6) OLS fit + Gaussian path simulation, and
# the ar6bf backfill correction (both identical to their submissions).
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
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile on the `transform` scale,
identical to `submissions/seabbs_bot-ar6bf/generate_forecasts.jl`: for
each `(location, delay)` with at least `min_support` recorded
versions at that delay, the median of `to_scale(settled, transform) -
to_scale(vintage, transform)`, where `settled` is the value at each
`(location, origin_date)` group's largest tracked `as_of`. `versions`
must already be filtered by the caller to the desired origin dates.
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
profile's location/delay correction.
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
# ses member: simple exponential smoothing (level only, no trend).
# ---------------------------------------------------------------------

"""
    fit_ses(y, alphas) -> (level, resid_sd, alpha)

Simple exponential smoothing (level only) on `y` (ascending in time),
fit by grid search over `alphas`: picks the value minimising in-sample
one-step-ahead SSE. `level` is the final smoothed level; `resid_sd` is
the in-sample one-step residual standard deviation at the chosen
`alpha`.
"""
function fit_ses(y::AbstractVector{Float64}, alphas)
    best_alpha = first(alphas)
    best_sse = Inf
    best_level = y[1]
    best_resid = Float64[]
    for alpha in alphas
        level = y[1]
        resid = Vector{Float64}(undef, length(y) - 1)
        for t in 2:length(y)
            resid[t - 1] = y[t] - level
            level = alpha * y[t] + (1 - alpha) * level
        end
        sse = sum(abs2, resid)
        if sse < best_sse
            best_sse = sse
            best_alpha = alpha
            best_level = level
            best_resid = resid
        end
    end
    dof = max(length(best_resid) - 1, 1)
    resid_sd = sqrt(sum(abs2, best_resid) / dof)
    return best_level, resid_sd, best_alpha
end

"""
    simulate_ses_paths(level, resid_sd, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Simulate `npaths` Gaussian sample paths for a level-only forecast:
`level + resid_sd * sqrt(h) * randn()` per horizon `h`. The `sqrt(h)`
growth is a simplifying random-walk-like heuristic for how one-step
forecast variance accumulates over the horizon (SES has no explicit
multi-step formula the way an AR model's recursion does).
"""
function simulate_ses_paths(
    level::Float64, resid_sd::Float64, horizons, npaths::Int;
    rng::Random.AbstractRNG,
)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    for h in horizons, s in 1:npaths
        out[h][s] = level + resid_sd * sqrt(h) * randn(rng)
    end
    return out
end

# ---------------------------------------------------------------------
# climatology member: seasonal-naive empirical quantiles, no fitting.
# ---------------------------------------------------------------------

"""
    climatology_quantiles(split_df, loc, target_end_date, transform;
                          band, min_n) -> Vector{Float64}

Seasonal-naive/climatology forecast: pools every historical `wili`
observation at `loc` in `split_df` (already capped at the split's own
forecast origin -- no future leakage) whose week-of-season is within
`band` weeks of `target_end_date`'s week-of-season (a circular
distance on the ~52-week season cycle), transforms the pool to
`transform` scale, and returns the 23 `QUANTILE_LEVELS` of that pooled
empirical distribution, back-transformed to the natural scale. Falls
back to the location's full history if fewer than `min_n`
observations are pooled at that band.
"""
function climatology_quantiles(
    split_df::DataFrame, loc::AbstractString, target_end_date::Date,
    transform::Symbol; band::Int, min_n::Int,
)
    target_woy = week_of_season(target_end_date)
    loc_df = split_df[split_df.location .== loc, :]
    isempty(loc_df) && error("no history for location $loc")
    woy = week_of_season.(loc_df.origin_date)
    dist = [min(abs(w - target_woy), 52 - abs(w - target_woy)) for w in woy]
    pooled = loc_df.wili[dist .<= band]
    length(pooled) < min_n && (pooled = loc_df.wili)
    vals = to_scale.(pooled, transform)
    return [max(from_scale(quantile(vals, q), transform), 0.0)
            for q in QUANTILE_LEVELS]
end

# ---------------------------------------------------------------------
# Build all four members' forecast tables in one pass over the splits.
# ---------------------------------------------------------------------

_empty_rows() = DataFrame(
    model_id=String[], location=String[], origin_date=Date[],
    horizon=Int[], target_end_date=Date[], target=String[],
    output_type=String[], output_type_id=Float64[], value=Float64[],
)

"""
    build_member_forecasts(seasons, profile, versions_full)
        -> Dict{String,DataFrame}

Fit and forecast all four members for every cross-validation split of
every season in `seasons`, sharing one `ModelData` build per split
across the `ar6`/`ar6bf`/`ses` members (the ar6bf copy gets the
backfill correction applied). Returns one hub quantile table
(docs/contracts.md schema) per member name in `MEMBER_NAMES`, each
tagged with `model_id = <member name>`.
"""
function build_member_forecasts(seasons, profile, versions_full)
    rng = MersenneTwister(SEED)
    rows = Dict(m => _empty_rows() for m in MEMBER_NAMES)
    for season in seasons
        splits = training_splits(season)
        for split in splits
            data = build_model_data(
                split; Dmax=DMAX, transform=TRANSFORM, window_weeks=104,
                versions=versions_full,
            )
            data_bf = deepcopy(data)
            apply_backfill_correction!(data_bf, profile)
            origin = data.origin_date
            for (li, loc) in enumerate(LOCATIONS)
                y = Float64.(data.Y[:, li])
                y_bf = Float64.(data_bf.Y[:, li])

                coef, resid_sd = fit_ar(y, AR_ORDER)
                paths_ar6 = simulate_paths(
                    y, coef, resid_sd, AR_ORDER, HORIZONS, NPATHS; rng=rng,
                )
                coef_bf, resid_sd_bf = fit_ar(y_bf, AR_ORDER)
                paths_ar6bf = simulate_paths(
                    y_bf, coef_bf, resid_sd_bf, AR_ORDER, HORIZONS, NPATHS;
                    rng=rng,
                )
                level, resid_sd_ses, _ = fit_ses(y, SES_ALPHAS)
                paths_ses = simulate_ses_paths(
                    level, resid_sd_ses, HORIZONS, NPATHS; rng=rng,
                )

                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    climo_vals = climatology_quantiles(
                        split, loc, target_end, TRANSFORM; band=CLIMO_BAND,
                        min_n=CLIMO_MIN_N,
                    )
                    for (qi, q) in enumerate(QUANTILE_LEVELS)
                        nat_ar6 = max(
                            from_scale(quantile(paths_ar6[h], q), TRANSFORM),
                            0.0,
                        )
                        nat_ar6bf = max(
                            from_scale(quantile(paths_ar6bf[h], q), TRANSFORM),
                            0.0,
                        )
                        nat_ses = max(
                            from_scale(quantile(paths_ses[h], q), TRANSFORM),
                            0.0,
                        )
                        push!(rows["ar6"], ("ar6", loc, origin, h,
                            target_end, TARGET, "quantile", q, nat_ar6))
                        push!(rows["ar6bf"], ("ar6bf", loc, origin, h,
                            target_end, TARGET, "quantile", q, nat_ar6bf))
                        push!(rows["ses"], ("ses", loc, origin, h,
                            target_end, TARGET, "quantile", q, nat_ses))
                        push!(rows["climatology"], ("climatology", loc,
                            origin, h, target_end, TARGET, "quantile", q,
                            climo_vals[qi]))
                    end
                end
            end
        end
    end
    return rows
end

# ---------------------------------------------------------------------
# Quantile-level combination across members.
# ---------------------------------------------------------------------

"""
    combine_members(all_df, method; weights=nothing, model_id) -> DataFrame

Combine member quantile forecasts into one ensemble forecast table,
pointwise per (location, origin_date, horizon, target_end_date,
output_type_id). `method` is `:mean` (simple average), `:median`
(pointwise median), or `:weighted` (`weights[model_id]`-weighted
average; required for `:weighted`). See the file header for why this
preserves monotonicity in the quantile level with no post-hoc sorting.
"""
function combine_members(
    all_df::DataFrame, method::Symbol;
    weights::Union{Nothing,Dict{String,Float64}}=nothing,
    model_id::String,
)
    group_cols = [:location, :origin_date, :horizon, :target_end_date,
                  :target, :output_type, :output_type_id]
    combined = combine(groupby(all_df, group_cols)) do sdf
        value = if method == :mean
            mean(sdf.value)
        elseif method == :median
            median(sdf.value)
        elseif method == :weighted
            weights === nothing && error("`:weighted` needs `weights`")
            w = [weights[m] for m in sdf.model_id]
            sum(w .* sdf.value) / sum(w)
        else
            error("unknown method $method")
        end
        (value=value,)
    end
    insertcols!(combined, 1, :model_id => model_id)
    return combined[:, [:model_id, :location, :origin_date, :horizon,
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
    truth = load_oracle(HUB_PATH)

    member_rows = build_member_forecasts(VALIDATION_ONLY, profile, versions_full)
    all_members = vcat((member_rows[m] for m in MEMBER_NAMES)...)

    member_summary = Dict{String,NamedTuple}()
    member_scored = Dict{String,DataFrame}()
    for m in MEMBER_NAMES
        scored = score_forecasts(member_rows[m], truth)
        summ = wis_summary(scored)[1, :]
        member_summary[m] = (mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
                              n_tasks=summ.n_tasks)
        member_scored[m] = scored
    end

    # Validation-WIS-weighted combination: weight = 1/mean_wis per
    # member, normalised to sum to 1 (tuned on validation only -- this
    # IS the validation round, so no further held-out split is
    # available; see docs/contracts.md experimental integrity for why
    # the test seasons are untouched throughout).
    weights = Dict(m => 1 / member_summary[m].mean_wis for m in MEMBER_NAMES)
    wsum = sum(values(weights))
    weights = Dict(m => w / wsum for (m, w) in weights)

    combos = Dict(
        "ens-mean" => combine_members(all_members, :mean; model_id="ens-mean"),
        "ens-median" => combine_members(
            all_members, :median; model_id="ens-median",
        ),
        "ens-wis-weighted" => combine_members(
            all_members, :weighted; weights=weights,
            model_id="ens-wis-weighted",
        ),
    )
    combo_summary = Dict{String,NamedTuple}()
    combo_scored = Dict{String,DataFrame}()
    for (name, df_) in combos
        scored = score_forecasts(df_, truth)
        summ = wis_summary(scored)[1, :]
        combo_summary[name] = (mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
                                n_tasks=summ.n_tasks)
        combo_scored[name] = scored
    end

    ranking = sort(
        vcat(
            [(model=m, member_summary[m]...) for m in MEMBER_NAMES],
            [(model=n, combo_summary[n]...) for n in keys(combo_summary)],
        );
        by=r -> r.mean_wis,
    )
    best = ranking[1]
    best_scored = best.model in MEMBER_NAMES ?
        member_scored[best.model] : combo_scored[best.model]

    println("=== simple ensemble round (validation seasons 1, 2 only) ===")
    for r in ranking
        @printf("%-18s mean_wis=%.4f  sd_wis=%.4f  n=%d\n",
                r.model, r.mean_wis, r.sd_wis, r.n_tasks)
    end
    println("weights (1/mean_wis, normalised): ", weights)

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "ensemble of simple analytic models -- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "members:")
        println(io, "  ar6         -- plain AR(6), OLS + Gaussian paths")
        println(io, "  ar6bf       -- AR(6) + ar6bf backfill correction")
        println(io, "  climatology -- seasonal-naive empirical quantiles " *
                     "(+/-$(CLIMO_BAND) wk band)")
        println(io, "  ses         -- simple exponential smoothing " *
                     "(level only, grid-searched alpha)")
        println(io)
        println(io, "combination methods:")
        println(io, "  ens-mean         -- simple average of quantile " *
                     "values across members")
        println(io, "  ens-median       -- pointwise median across members")
        println(io, "  ens-wis-weighted -- weights = 1/mean_wis per " *
                     "member, normalised (tuned on validation only)")
        println(io)
        println(io, "reference points from " *
                     "submissions/seabbs_bot-ar6bf/README.md: " *
                     "nfidd-ar6=0.368 (sd 0.471), " *
                     "seabbs_bot-ar6bf=0.359 (sd 0.452)")
        println(io)
        println(io, "ranking (sorted by mean_wis, ascending):")
        println(io, rpad("model", 20) * rpad("mean_wis", 12) *
                     rpad("sd_wis", 12) * "n_tasks")
        for r in ranking
            println(io, rpad(r.model, 20) *
                         rpad(string(round(r.mean_wis; digits=4)), 12) *
                         rpad(string(round(r.sd_wis; digits=4)), 12) *
                         string(r.n_tasks))
        end
        println(io)
        println(io, "weights (1/mean_wis, normalised): $(weights)")
        println(io)
        println(io, "=== best: $(best.model) ===")
        println(io, "mean_wis=$(round(best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(best.sd_wis; digits=4)) " *
                     "n_tasks=$(best.n_tasks)")
        vs_baseline = 0.359 - best.mean_wis
        vs_pct = 100 * vs_baseline / 0.359
        println(io, "vs seabbs_bot-ar6bf (0.359): " *
                     "$(round(vs_baseline; digits=4)) " *
                     "($(round(vs_pct; digits=2))%)")

        println(io)
        println(io, "-- breakdown by location (best) --")
        by_loc = combine(groupby(best_scored, :location),
            :wis => mean => :mean_wis, nrow => :n)
        sort!(by_loc, :mean_wis)
        for row in eachrow(by_loc)
            println(io, rpad(row.location, 16) *
                         "mean_wis=$(round(row.mean_wis; digits=4)) " *
                         "n=$(row.n)")
        end

        println(io)
        println(io, "-- breakdown by horizon (best) --")
        by_h = combine(groupby(best_scored, :horizon),
            :wis => mean => :mean_wis, nrow => :n)
        sort!(by_h, :horizon)
        for row in eachrow(by_h)
            println(io, "h=$(row.horizon): " *
                         "mean_wis=$(round(row.mean_wis; digits=4)) " *
                         "n=$(row.n)")
        end

        println(io)
        println(io, "-- breakdown by season (best) --")
        best_scored.season_year = season_year.(best_scored.origin_date)
        by_season = combine(groupby(best_scored, :season_year),
            :wis => mean => :mean_wis, nrow => :n)
        sort!(by_season, :season_year)
        for row in eachrow(by_season)
            println(io, "season $(row.season_year): " *
                         "mean_wis=$(round(row.mean_wis; digits=4)) " *
                         "n=$(row.n)")
        end
    end

    dt = round(time() - t0; digits=1)
    println("\nwrote score.txt in $(dt)s total")
    return ranking
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
