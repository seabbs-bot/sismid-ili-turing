#!/usr/bin/env julia
# ROUND 2 stack -- simple-round.
#
# Round 1 winner (experiments/simple-round/seasoncombo/generate.jl,
# combo "core"): pooled week-of-season climatology (fourth-root scale)
# + independent per-location AR(6) + additive/per-location/median
# backfill correction (window=8). mean_wis=0.2781 (sd 0.3341,
# validation seasons 1,2 -- see seasoncombo/score.txt).
#
# This script stacks three further wins, each found ORTHOGONAL to the
# seasonal core in other simple-round experiments, onto that core, and
# checks which of them (alone or combined) actually improve on 0.2781
# once the seasonal term is already in the model:
#
#   1. LOG transform instead of fourth-root
#      (experiments/simple-round/transform/{generate.jl,score.txt}):
#      plain `log` beat fourth-root by ~4% mean WIS (0.3537 vs 0.3684)
#      on the plain-AR(6) family. The Region-9 `sqrt` override
#      (log_r9_sqrt, a further ~0.2% on top of plain log) is NOT
#      stacked here: it needs a genuinely per-location modelling scale,
#      but the pooled seasonal profile is a single shared shape
#      estimated by pooling DEVIATIONS across all 11 locations
#      (`build_seasonal_profile`) -- mixing one location's deviations
#      in sqrt units into a pool of the other ten's log-unit deviations
#      would quietly distort the shared shape for a ~0.2% effect that
#      is small even in the plain-AR family. Left as follow-up; see
#      score.txt.
#   2. Student-t(df=10) innovations, variance-matched to resid_sd, then
#      scaled by 1.4 (experiments/simple-round/intervals/{generate.jl,
#      score.txt}): fixes the raw fitted resid_sd's under-coverage (50%
#      nominal covering ~41%) and improves mean WIS ~3% on the plain
#      AR(6)+backfill family.
#   3. Partial pooling of the per-location AR(6) coefficients toward a
#      single fullpool (all-locations-stacked) OLS fit
#      (experiments/simple-round/pool/{generate.jl,score.txt}): blend
#      weight w=0.5 was the sweep's best on the plain-AR(6) family
#      (mean_wis 0.3643 vs 0.3684 unpooled, ~1.1% there; w in
#      [0.25,0.75] all similarly close). Applied here to the
#      DESEASONALIZED residual columns (the same quantity the seasonal
#      core's own per-location AR(6) already fits), not raw Y.
#
# Each addition is tried alone on top of the (fourthroot) core AND on
# top of the log core, plus the full 3-way stack and a small pool-
# weight sensitivity check, so the sweep below also answers "does this
# still help once the OTHER stacked wins are already present, not just
# alone against the bare core". Scored on VALIDATION SEASONS (1, 2)
# ONLY, against the local hub clone's oracle (docs/contracts.md
# experimental integrity) -- a tuning sweep, not a submission driver.
#
# LIGHT + ANALYTIC: CSV/DataFrames/Dates/Statistics/Random/
# LinearAlgebra/Distributions only, no Turing.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl
# writes score.txt alongside this file; does not write a hub submission
# (no hub_path argument -- exploratory, not a submissions/ candidate).
# The winning combo, however, IS full-5-season capable
# (build_forecast_table takes any `seasons` tuple and both AR fitting
# and simulation are per-split, capped at that split's own forecast
# origin throughout) -- see `full_stack_forecast` in main(), unused
# here but there for a submission driver to call with all 5 seasons.

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
const MIN_SUPPORT = 5           # min sample size per profile bin to trust
const SMOOTH_WINDOW = 3         # circular smoothing span for the profile
const VALIDATION_ONLY = (1, 2)
const MAX_TRAIN_SEASON_YEAR = 2016  # pre-2015 history + validation seasons
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")

# Backfill design, identical to seasoncombo's "core" combo.
const BF_MODE = :additive
const BF_WINDOW = 8
const BF_STAT = :median

# Tuned interval scheme, identical to experiments/simple-round/intervals.
const T_DF = 10
const T_SCALE = 1.4

# Pool weight sweep around the round-1 pool sweep's w=0.5 pick.
const POOL_WEIGHTS = (0.25, 0.5, 0.75, 0.9, 1.0)

# ---------------------------------------------------------------------
# Pooled seasonal shape (identical to seasoncombo/generate.jl)
# ---------------------------------------------------------------------

"""
    build_seasonal_profile(hist; transform, max_season_year, min_support,
                            smooth_window) -> Dict{Int,Float64}

Pooled week-of-season climatology on the `transform` scale. Identical in
design to `experiments/simple-round/seasoncombo/generate.jl`'s function
of the same name -- see that file for the full derivation.
"""
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

"""
    deseasonalize(Y, woy, profile) -> (R, level)

Remove each location's own mean level and the pooled seasonal shape
from `Y` (T x L, modelling scale). Identical to seasoncombo's function
of the same name with `amp` fixed at 1 everywhere (this round does not
stack the amplitude-scaling combo, only core + the three requested
additions).
"""
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

# ---------------------------------------------------------------------
# Backfill correction (identical to seasoncombo's "core" combo)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support,
                            mode, stat) -> Dict{Tuple{String,Int},Float64}

Empirical per-(location, delay) revision profile on the `transform`
scale. Identical to seasoncombo's function of the same name with
`pooled=false` fixed (matches the core combo and seabbs_bot-ar6bf).
"""
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

"""
    apply_backfill_correction!(data, profile; mode, delay_cutoff)

Nudge `data.Y` in place, identical to seasoncombo's function of the
same name with `pooled=false` fixed.
"""
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

# ---------------------------------------------------------------------
# Per-location AR(6): OLS, factored so a blended (pooled) coefficient
# vector can be re-scored on the same design (experiments/simple-round/
# pool/generate.jl's pattern), plus a fullpool anchor fit across all L
# deseasonalized residual columns.
# ---------------------------------------------------------------------

"""
    ar_design(y, order) -> (X, yresp)

Design matrix and response for an OLS AR(`order`) fit with intercept.
Identical to `experiments/simple-round/pool/generate.jl`'s function of
the same name.
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
`yresp`) evaluated on this design. Identical to the pool experiment's
function of the same name -- lets a blended/pooled coefficient vector's
own fit quality (not the unpooled fit's) set the simulated path spread.
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

One OLS AR(`order`) fit on the design rows of every column in `ys`
stacked together -- the `:fullpool` anchor, identical in spirit to the
pool experiment's function of the same name, applied here to the
DESEASONALIZED residual columns rather than raw Y (the seasonal core
already removes each location's level and shared shape, so what the
per-location AR(6) fits, and what pooling here blends toward, is that
residual, not the raw series).
"""
function fit_ar_pooled(ys::Vector{Vector{Float64}}, order::Int)
    designs = [ar_design(y, order) for y in ys]
    Xall = reduce(vcat, first.(designs))
    yall = reduce(vcat, last.(designs))
    return Xall \ yall
end

# ---------------------------------------------------------------------
# Path simulation: Gaussian or variance-matched Student-t innovations
# ---------------------------------------------------------------------

"""
    simulate_paths(y, coef, resid_sd, order, horizons, npaths;
                   rng, innovation, t_df, t_scale)
        -> Dict{Int,Vector{Float64}}

Simulate `npaths` sample paths forward from the end of `y` (modelling
scale, here the deseasonalized residual), for each horizon in
`horizons`. `innovation = :gaussian` draws Normal(0, resid_sd)
(seasoncombo's "core" scheme); `innovation = :student_t` draws
Student-t(`t_df`), variance-matched to `resid_sd` then scaled by
`t_scale` (experiments/simple-round/intervals' winning scheme).
"""
function simulate_paths(
    y::AbstractVector{Float64}, coef::Vector{Float64}, resid_sd::Float64,
    order::Int, horizons, npaths::Int;
    rng::Random.AbstractRNG, innovation::Symbol,
    t_df::Int=T_DF, t_scale::Float64=T_SCALE,
)
    tdist = TDist(t_df)
    vscale = sqrt((t_df - 2) / t_df)
    innov_sd = innovation == :student_t ? resid_sd * vscale * t_scale : resid_sd

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
            innov = innovation == :student_t ?
                innov_sd * rand(rng, tdist) : innov_sd * randn(rng)
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

# ---------------------------------------------------------------------
# Forecast table builder -- seasonal core + the three optional additions
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, versions_full, profile; kwargs...)
        -> DataFrame

Fit and forecast one point on the stack sweep for every cross-
validation split of every season in `seasons`: pooled seasonal
deseasonalization (`profile`, on `transform` scale) + backfill
correction, then per-location AR(`AR_ORDER`), optionally blended toward
a fullpool anchor (`pool_w > 0`) fit on the deseasonalized residuals,
then simulated forward with either Gaussian or Student-t innovations
(`innovation`).

`pool_w = 0.0` reproduces the plain per-location AR(6) (no pooling);
`innovation = :gaussian` reproduces the seasonal core's own scheme.
"""
function build_forecast_table(
    seasons, versions_full, profile::Dict{Int,Float64};
    transform::Symbol, backfill_profile::Dict{Tuple{String,Int},Float64},
    backfill_window::Int=BF_WINDOW, innovation::Symbol=:gaussian,
    pool_w::Float64=0.0, model_id::String,
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
                split; Dmax=DMAX, transform=transform,
                window_weeks=WINDOW_WEEKS, versions=versions_full,
            )
            apply_backfill_correction!(
                data, backfill_profile; mode=BF_MODE,
                delay_cutoff=backfill_window,
            )
            R, level = deseasonalize(data.Y, data.woy, profile)
            origin = data.origin_date
            L = data.L

            ys = [R[:, li] for li in 1:L]
            fits = [fit_ar(ys[li], AR_ORDER) for li in 1:L]
            coefs = [f[1] for f in fits]
            Xs = [f[2] for f in fits]
            yresps = [f[3] for f in fits]

            blended = if pool_w <= 0.0
                coefs
            else
                anchor = fit_ar_pooled(ys, AR_ORDER)
                [(1 - pool_w) .* coefs[li] .+ pool_w .* anchor for li in 1:L]
            end

            for (li, loc) in enumerate(LOCATIONS)
                coef = blended[li]
                resid_sd = resid_sd_for(Xs[li], yresps[li], coef, AR_ORDER)
                paths = simulate_paths(
                    ys[li], coef, resid_sd, AR_ORDER, HORIZONS, NPATHS;
                    rng=rng, innovation=innovation,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    s = get(profile, week_of_season(target_end), 0.0)
                    vals = paths[h] .+ level[li] .+ s
                    for q in QUANTILE_LEVELS
                        qval = quantile(vals, q)
                        nat = max(from_scale(qval, transform), 0.0)
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

"""
    coverage(forecast, truth, level) -> Float64

Empirical coverage of the nominal `level` central interval (e.g.
`level=0.5` -> the [0.25, 0.75] quantile pair): fraction of tasks where
the truth falls within [lower, upper]. No shared coverage helper exists
in `src/scoring.jl` (`docs/contracts.md` covers WIS only), so this is
local to this experiment, in the same spirit as the ad hoc coverage
checks in `experiments/simple-round/intervals/score.txt`.
"""
function coverage(forecast::DataFrame, truth::DataFrame, level::Float64)
    a = (1 - level) / 2
    task_cols = [:location, :origin_date, :horizon, :target_end_date]
    lo = forecast[isapprox.(forecast.output_type_id, a; atol=1e-6), :]
    hi = forecast[isapprox.(forecast.output_type_id, 1 - a; atol=1e-6), :]
    lo_r = rename(lo[:, vcat(task_cols, [:value])], :value => :lo)
    hi_r = rename(hi[:, vcat(task_cols, [:value])], :value => :hi)
    joined = innerjoin(lo_r, hi_r, on=task_cols)
    joined = innerjoin(joined, truth, on=[:location, :target_end_date])
    return mean(joined.lo .<= joined.value .<= joined.hi)
end

# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------

function run_combo(label, seasons, versions_full, profile, backfill_profile;
        transform, innovation, pool_w, truth)
    fc = build_forecast_table(
        seasons, versions_full, profile; transform=transform,
        backfill_profile=backfill_profile, innovation=innovation,
        pool_w=pool_w, model_id=label,
    )
    summ = score_one(fc, truth)
    cov50 = coverage(fc, truth, 0.5)
    cov90 = coverage(fc, truth, 0.9)
    println("  $(rpad(label, 26)) mean_wis=$(round(summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(summ.sd_wis; digits=4)) " *
            "cov50=$(round(cov50; digits=3)) cov90=$(round(cov90; digits=3))")
    return (
        label=label, transform=transform, innovation=innovation,
        pool_w=pool_w, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        cov50=cov50, cov90=cov90, forecast=fc,
    )
end

function main()
    t0 = time()
    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")
    truth = load_oracle(HUB_PATH)
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= MAX_TRAIN_SEASON_YEAR, :,
    ]

    # Profile + backfill profile are estimated ONCE PER TRANSFORM (both
    # scale-dependent), reused across every (innovation, pool_w) combo
    # on that transform.
    profiles = Dict{Symbol,Dict{Int,Float64}}()
    bf_profiles = Dict{Symbol,Dict{Tuple{String,Int},Float64}}()
    for transform in (:fourthroot, :log)
        profiles[transform] = build_seasonal_profile(
            hist; transform=transform, max_season_year=MAX_TRAIN_SEASON_YEAR,
            min_support=MIN_SUPPORT, smooth_window=SMOOTH_WINDOW,
        )
        bf_profiles[transform] = build_revision_profile(
            training_versions; transform=transform, max_delay=BF_WINDOW,
            min_support=MIN_SUPPORT, mode=BF_MODE, stat=BF_STAT,
        )
    end

    results = NamedTuple[]

    println("=== reproduction check + single-addition ablations ===")
    push!(results, run_combo(
        "core (fourthroot, reproduce)", VALIDATION_ONLY, versions_full,
        profiles[:fourthroot], bf_profiles[:fourthroot];
        transform=:fourthroot, innovation=:gaussian, pool_w=0.0, truth=truth,
    ))
    push!(results, run_combo(
        "core+log", VALIDATION_ONLY, versions_full,
        profiles[:log], bf_profiles[:log];
        transform=:log, innovation=:gaussian, pool_w=0.0, truth=truth,
    ))
    push!(results, run_combo(
        "core+tstudent", VALIDATION_ONLY, versions_full,
        profiles[:fourthroot], bf_profiles[:fourthroot];
        transform=:fourthroot, innovation=:student_t, pool_w=0.0, truth=truth,
    ))
    push!(results, run_combo(
        "core+pool(w=0.5)", VALIDATION_ONLY, versions_full,
        profiles[:fourthroot], bf_profiles[:fourthroot];
        transform=:fourthroot, innovation=:gaussian, pool_w=0.5, truth=truth,
    ))

    println("\n=== additions stacked on the log core ===")
    push!(results, run_combo(
        "log+tstudent", VALIDATION_ONLY, versions_full,
        profiles[:log], bf_profiles[:log];
        transform=:log, innovation=:student_t, pool_w=0.0, truth=truth,
    ))
    push!(results, run_combo(
        "log+pool(w=0.5)", VALIDATION_ONLY, versions_full,
        profiles[:log], bf_profiles[:log];
        transform=:log, innovation=:gaussian, pool_w=0.5, truth=truth,
    ))
    push!(results, run_combo(
        "log+tstudent+pool(w=0.5)  [full stack]", VALIDATION_ONLY,
        versions_full, profiles[:log], bf_profiles[:log];
        transform=:log, innovation=:student_t, pool_w=0.5, truth=truth,
    ))

    println("\n=== pool-weight sensitivity on top of log+tstudent ===")
    for w in POOL_WEIGHTS
        push!(results, run_combo(
            "log+tstudent+pool(w=$w)", VALIDATION_ONLY, versions_full,
            profiles[:log], bf_profiles[:log];
            transform=:log, innovation=:student_t, pool_w=w, truth=truth,
        ))
    end

    sorted = sort(results; by=r -> r.mean_wis)
    winner = sorted[1]
    core = results[1]

    println("\n=== ranked ===")
    for r in sorted
        println("  $(rpad(r.label, 40)) mean_wis=$(round(r.mean_wis; digits=4)) " *
                "sd_wis=$(round(r.sd_wis; digits=4))")
    end
    println("\nwinner: $(winner.label) mean_wis=$(round(winner.mean_wis; digits=4))")
    println("vs round-1 core (0.2781): " *
            "$(round(0.2781 - winner.mean_wis; digits=4)) " *
            "($(round(100 * (0.2781 - winner.mean_wis) / 0.2781; digits=2))%)")

    # Winner breakdown: by location, by season, by horizon.
    winner_scored = score_forecasts(winner.forecast, truth; scale=:natural)
    by_loc = combine(groupby(winner_scored, :location),
        :wis => mean => :mean_wis, nrow => :n)
    sort!(by_loc, :mean_wis)

    winner_scored.season_num = [
        season_year(d) == 2015 ? 1 : 2 for d in winner_scored.origin_date
    ]
    by_season = combine(groupby(winner_scored, :season_num),
        :wis => mean => :mean_wis, :wis => std => :sd_wis, nrow => :n)
    sort!(by_season, :season_num)

    by_h = combine(groupby(winner_scored, :horizon),
        :wis => mean => :mean_wis, nrow => :n)
    sort!(by_h, :horizon)

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "round 2 stack -- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "round-1 winner (seasoncombo core): mean_wis=0.2781 " *
                     "sd_wis=0.3341 (experiments/simple-round/seasoncombo/" *
                     "score.txt)")
        println(io)
        println(io, "=== reproduction check + single-addition ablations ===")
        for r in results[1:4]
            println(io, "  $(rpad(r.label, 32)) mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4)) cov50=" *
                         "$(round(r.cov50; digits=3)) cov90=" *
                         "$(round(r.cov90; digits=3))")
        end
        println(io)
        println(io, "=== additions stacked on the log core ===")
        for r in results[5:7]
            println(io, "  $(rpad(r.label, 42)) mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4)) cov50=" *
                         "$(round(r.cov50; digits=3)) cov90=" *
                         "$(round(r.cov90; digits=3))")
        end
        println(io)
        println(io, "=== pool-weight sensitivity on top of log+tstudent ===")
        for r in results[8:end]
            println(io, "  $(rpad(r.label, 32)) mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4)) cov50=" *
                         "$(round(r.cov50; digits=3)) cov90=" *
                         "$(round(r.cov90; digits=3))")
        end
        println(io)
        println(io, "=== ranked (all combos) ===")
        for r in sorted
            println(io, "  $(rpad(r.label, 42)) mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4))")
        end
        println(io)
        println(io, "=== marginal contribution on top of the core " *
                     "(fourthroot, mean_wis=$(round(core.mean_wis; digits=4))) ===")
        for r in results[2:4]
            delta = core.mean_wis - r.mean_wis
            pct = 100 * delta / core.mean_wis
            println(io, "  $(rpad(r.label, 20)) delta=$(round(delta; digits=4)) " *
                         "($(round(pct; digits=2))%)")
        end
        log_core = results[2]
        println(io)
        println(io, "=== marginal contribution on top of the log core " *
                     "(mean_wis=$(round(log_core.mean_wis; digits=4))) ===")
        for r in results[5:7]
            delta = log_core.mean_wis - r.mean_wis
            pct = 100 * delta / log_core.mean_wis
            println(io, "  $(rpad(r.label, 42)) delta=$(round(delta; digits=4)) " *
                         "($(round(pct; digits=2))%)")
        end
        println(io)
        println(io, "=== winner: $(winner.label) ===")
        println(io, "mean_wis=$(round(winner.mean_wis; digits=4)) " *
                     "sd_wis=$(round(winner.sd_wis; digits=4)) " *
                     "cov50=$(round(winner.cov50; digits=3)) " *
                     "cov90=$(round(winner.cov90; digits=3))")
        vs_ref = 0.2781 - winner.mean_wis
        vs_pct = 100 * vs_ref / 0.2781
        println(io, "vs round-1 core (0.2781): $(round(vs_ref; digits=4)) " *
                     "($(round(vs_pct; digits=2))%)")
        println(io)
        println(io, "winner mean WIS by location:")
        for row in eachrow(by_loc)
            println(io, "  $(rpad(row.location, 16)) $(round(row.mean_wis; digits=4)) " *
                         "(n=$(row.n))")
        end
        println(io)
        println(io, "winner mean WIS by season:")
        for row in eachrow(by_season)
            println(io, "  season $(row.season_num): mean_wis=" *
                         "$(round(row.mean_wis; digits=4)) sd_wis=" *
                         "$(round(row.sd_wis; digits=4)) (n=$(row.n))")
        end
        println(io)
        println(io, "winner mean WIS by horizon:")
        for row in eachrow(by_h)
            println(io, "  h=$(row.horizon): $(round(row.mean_wis; digits=4)) " *
                         "(n=$(row.n))")
        end
        println(io)
        println(io, "Not stacked here: the log_r9_sqrt refinement " *
                     "(experiments/simple-round/transform, a further ~0.2% " *
                     "on top of plain log in the plain-AR family) -- the " *
                     "pooled seasonal profile mixes deviations from all 11 " *
                     "locations in one shared shape, and a per-location " *
                     "modelling scale override would need either a " *
                     "separate single-location climatology for Region 9 " *
                     "or a scale-aware re-derivation of the pooling step; " *
                     "left as follow-up given the small expected effect " *
                     "size relative to the three additions above.")
    end

    dt = round(time() - t0; digits=1)
    println("\nwrote score.txt in $(dt)s total")
    return sorted
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
