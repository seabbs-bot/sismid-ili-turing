#!/usr/bin/env julia
# smoother -- simple-round, CLOSE-VARIANT family.
#
# Round-1 winner (experiments/simple-round/seasoncombo/generate.jl,
# combo 1 "core"): a POOLED seasonal shape (one week-of-season shape
# shared across all 11 locations + the full training history, see
# `build_seasonal_profile` below -- copied verbatim from seasoncombo)
# plus the `seabbs_bot-ar6bf` backfill correction (additive/per-
# location/median, window=8) plus a per-location AR(6) fit on the
# deseasonalized+backfilled residual. That combination scores
# mean_wis=0.2781 on the validation seasons (seasoncombo/score.txt).
#
# This driver keeps the pooled seasonal shape and the backfill
# correction EXACTLY as in that winner (same functions, same
# constants, same additive/per-location/median/window=8 profile) and
# swaps ONLY the residual mechanism: instead of a per-location AR(6)
# with Gaussian innovations, the deseasonalized+backfilled residual is
# forecast with a NON-AR, non-parametric KERNEL-WEIGHTED ANALOGUE
# method (a "poor-man's" combination of a Nadaraya-Watson kernel
# extrapolation and a nearest-neighbour analogue forecaster):
#
#   1. Take the most recent `seg_len` residuals as a "query" segment.
#   2. Compare it (Euclidean/RMSE distance) against every other
#      `seg_len`-long segment earlier in the SAME training window that
#      still has `hmax` weeks of continuation after it (an
#      "analogue").
#   3. Weight each analogue by a Gaussian kernel on that distance
#      (bandwidth = the median pairwise distance among candidates,
#      scaled by `BW_MULT`) -- this is the Nadaraya-Watson step: closer
#      analogues count more.
#   4. The forecast at horizon h is the CURRENT residual level plus a
#      resampling bootstrap draw of one analogue's own h-step-ahead
#      CHANGE (`R[t+h] - R[t]`), drawn with probability proportional to
#      its kernel weight, plus a small amount of Gaussian jitter (scale
#      = `JITTER_FRAC` times the weighted spread of that horizon's
#      analogue changes) to smooth the otherwise-discrete bootstrap
#      distribution into continuous quantiles.
#
# Because the same drawn analogue index is reused across all four
# horizons within one simulated path, the simulated paths inherit
# whatever autocorrelation/shape the real historical continuations
# had -- not the flat Gaussian-innovation shape an AR(6) model
# imposes. No OLS, no AR coefficients, no distributional assumption on
# the innovations beyond the jitter smoothing term.
#
# LIGHT + ANALYTIC: CSV/DataFrames/Statistics/LinearAlgebra/Random
# only, no Turing.
#
# Scope: VALIDATION SEASONS (1, 2) ONLY, scored against the local hub
# clone's oracle (docs/contracts.md experimental integrity) -- this is
# a tuning/comparison driver, not a submission driver (no hub_path
# argument, matches experiments/simple-round/seasoncombo/generate.jl).
# The pooled seasonal shape and the backfill profile are both
# estimated only from `season_year <= 2016` (pre-2015 history plus the
# two validation seasons), same discipline as seasoncombo.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl
# writes score.txt alongside this file.

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
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12                     # matches ar6bf's build_model_data Dmax
const WINDOW_WEEKS = 104
const MIN_SUPPORT = 5               # min pooled obs per profile bin to trust
const SMOOTH_WINDOW = 3             # circular smoothing span for the profile
const VALIDATION_ONLY = (1, 2)
const MAX_TRAIN_SEASON_YEAR = 2016  # pre-2015 history + validation seasons
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")

# Reference backfill design (`seabbs_bot-ar6bf`, reused unchanged from
# `experiments/simple-round/seasoncombo/generate.jl`'s "core" combo --
# the 0.2781 reference this driver is a close variant of).
const BF_MODE = :additive
const BF_WINDOW = 8
const BF_STAT = :median

# ---------------------------------------------------------------------
# Pooled seasonal shape (verbatim from seasoncombo/generate.jl)
# ---------------------------------------------------------------------

"""
    build_seasonal_profile(hist; transform, max_season_year, min_support,
                            smooth_window) -> Dict{Int,Float64}

Pooled week-of-season climatology on the `transform` scale, identical
to `experiments/simple-round/seasoncombo/generate.jl`'s function of the
same name -- see that file for the full derivation. Reused unchanged:
this driver keeps the round-1 winner's seasonal term EXACTLY, and only
swaps the residual mechanism.
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
    deseasonalize(Y, woy, profile, amp) -> (R, level)

Identical to seasoncombo's function of the same name: remove each
location's own mean level and the (`amp`-scaled) pooled seasonal shape
from `Y` (T x L, modelling scale). This driver always calls it with
`amp = ones(L)` (no per-location amplitude scaling), matching the
0.2781 "core" combo exactly.
"""
function deseasonalize(
    Y::AbstractMatrix, woy::Vector{Int}, profile::Dict{Int,Float64},
    amp::Vector{Float64},
)
    T, L = size(Y)
    level = zeros(L)
    for l in 1:L
        level[l] = mean(Float64.(Y[:, l]))
    end
    R = Matrix{Float64}(undef, T, L)
    for l in 1:L, t in 1:T
        s = get(profile, woy[t], 0.0)
        R[t, l] = Y[t, l] - level[l] - amp[l] * s
    end
    return R, level
end

# ---------------------------------------------------------------------
# Backfill correction (identical to seasoncombo / seabbs_bot-ar6bf;
# used with BF_MODE=:additive, pooled=false, BF_STAT=:median, exactly
# as the 0.2781 "core" combo)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support,
                            mode, pooled, stat) -> Dict

Identical to seasoncombo/generate.jl's function of the same name.
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

Identical to seasoncombo/generate.jl's function of the same name.
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
# Kernel-weighted analogue residual forecaster (the ONE new piece)
# ---------------------------------------------------------------------

"""
    fit_analogue(y, seg_len, hmax) -> (candidates, weights, bandwidth)

Build the analogue library for one location's deseasonalized+backfilled
residual series `y` (length T, most recent value last). The QUERY is
`y`'s last `seg_len` values. A CANDIDATE is every earlier index `t`
(`seg_len <= t <= T - hmax`, so it both has a full `seg_len`-long
segment before it and a full `hmax`-step continuation after it) --
these are analogue "anchors", not future information: every one of
them lies strictly inside the training window already capped at this
split's own forecast origin by `build_model_data`.

`weights[i]` is a Gaussian kernel value on the RMSE distance between
candidate `i`'s own trailing `seg_len`-long segment and the query,
`exp(-0.5 * (dist / bandwidth)^2)`, normalised to sum to 1 --
Nadaraya-Watson style: closer analogues (segments that looked like the
current recent trajectory) count for more. `bandwidth` is the median
pairwise distance among all candidates, scaled by `BW_MULT` (module
global, set per sweep entry) -- an adaptive, scale-free choice that
does not need a fixed distance unit tuned by hand.
"""
function fit_analogue(
    y::AbstractVector{Float64}, seg_len::Int, hmax::Int; bw_mult::Float64,
)
    T = length(y)
    max_t = T - hmax
    max_t >= seg_len + 3 || error(
        "series too short for analogue: T=$T, seg_len=$seg_len, hmax=$hmax",
    )
    query = y[(T - seg_len + 1):T]
    candidates = collect(seg_len:max_t)
    dists = Vector{Float64}(undef, length(candidates))
    for (i, t) in enumerate(candidates)
        seg = y[(t - seg_len + 1):t]
        dists[i] = sqrt(mean(abs2, seg .- query))
    end
    bandwidth = max(median(dists), 1e-3) * bw_mult
    weights = exp.(-0.5 .* (dists ./ bandwidth) .^ 2)
    weights ./= sum(weights)
    return candidates, weights, bandwidth
end

"""
    simulate_paths_analogue(y, candidates, weights, hmax, horizons,
                             npaths; jitter_frac, rng)
        -> Dict{Int,Vector{Float64}}

Simulate `npaths` sample paths of the analogue forecaster forward from
the end of `y`, for each horizon in `horizons`. For each simulated
path, ONE candidate anchor `t` is drawn with probability proportional
to `weights` (a weighted bootstrap of the analogue library), and the
path's value at horizon `h` is `y[end] + (y[t + h] - y[t]) +
jitter_sd[h] * randn()`: the current level, plus that analogue's own
realised h-step change, plus a small Gaussian jitter that smooths the
otherwise-discrete bootstrap distribution into continuous quantiles.
`jitter_sd[h]` is `jitter_frac` times the weighted standard deviation
of the horizon-`h` changes across the whole analogue library, so the
smoothing scales with how much the analogues themselves disagree at
that horizon (naturally growing with `h`, since further-out
continuations diverge more).

The SAME drawn analogue is reused for every horizon within one path,
so a path's h=1..4 values inherit whatever real historical
autocorrelation/shape that analogue's continuation had -- unlike an
AR(6) model's flat Gaussian-innovation paths.
"""
function simulate_paths_analogue(
    y::AbstractVector{Float64}, candidates::Vector{Int},
    weights::Vector{Float64}, hmax::Int, horizons, npaths::Int;
    jitter_frac::Float64, rng::Random.AbstractRNG,
)
    ncand = length(candidates)
    deltas = Matrix{Float64}(undef, ncand, hmax)
    for (i, t) in enumerate(candidates), h in 1:hmax
        deltas[i, h] = y[t + h] - y[t]
    end
    jitter_sd = Vector{Float64}(undef, hmax)
    for h in 1:hmax
        m = sum(weights .* deltas[:, h])
        v = sum(weights .* (deltas[:, h] .- m) .^ 2)
        jitter_sd[h] = sqrt(max(v, 0.0)) * jitter_frac
    end

    anchor = y[end]
    cw = cumsum(weights)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    for s in 1:npaths
        r = rand(rng)
        idx = searchsortedfirst(cw, r)
        idx = min(idx, ncand)
        for h in horizons
            val = anchor + deltas[idx, h]
            jitter_sd[h] > 0 && (val += jitter_sd[h] * randn(rng))
            out[h][s] = val
        end
    end
    return out
end

# ---------------------------------------------------------------------
# Forecast table builder
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, versions_full, profile, backfill_profile;
                          seg_len, bw_mult, jitter_frac, model_id)
        -> DataFrame

Fit and forecast the pooled-seasonal + backfill + analogue-residual
model for every cross-validation split of every season in `seasons`.
Training discipline identical to seasoncombo's "core" combo:
`build_model_data` caps each split at its own forecast origin,
`window_weeks=104` further caps history to 2 seasons, the backfill
correction is applied to `data.Y` before deseasonalizing, and `profile`
(the pooled seasonal shape) is looked up with `amp = 1` everywhere (no
per-location scaling).
"""
function build_forecast_table(
    seasons, versions_full, profile::Dict{Int,Float64}, backfill_profile::Dict;
    seg_len::Int, bw_mult::Float64, jitter_frac::Float64, model_id::String,
)
    rng = MersenneTwister(SEED)
    ones_amp = ones(length(LOCATIONS))
    hmax = maximum(HORIZONS)
    rows = DataFrame(
        model_id=String[], location=String[], origin_date=Date[],
        horizon=Int[], target_end_date=Date[], target=String[],
        output_type=String[], output_type_id=Float64[], value=Float64[],
    )
    for season in seasons
        splits = training_splits(season)
        for split in splits
            data = build_model_data(
                split; Dmax=DMAX, transform=TRANSFORM,
                window_weeks=WINDOW_WEEKS, versions=versions_full,
            )
            apply_backfill_correction!(
                data, backfill_profile; mode=BF_MODE, pooled=false,
                delay_cutoff=BF_WINDOW,
            )
            R, level = deseasonalize(data.Y, data.woy, profile, ones_amp)
            origin = data.origin_date

            for (li, loc) in enumerate(LOCATIONS)
                y = R[:, li]
                candidates, weights, _ = fit_analogue(
                    y, seg_len, hmax; bw_mult=bw_mult,
                )
                paths = simulate_paths_analogue(
                    y, candidates, weights, hmax, HORIZONS, NPATHS;
                    jitter_frac=jitter_frac, rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    s = get(profile, week_of_season(target_end), 0.0)
                    vals = paths[h] .+ level[li] .+ s
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

score_one(forecast, truth) = wis_summary(score_forecasts(
    forecast, truth; scale=:natural,
))[1, :]

# ---------------------------------------------------------------------
# Sweep: segment length, kernel bandwidth multiplier, jitter fraction
# ---------------------------------------------------------------------

const SEG_LENS = (4, 6, 8)
const BW_MULTS = (0.75, 1.0, 1.5)
const JITTER_FRACS = (0.0, 0.3)

const CORE_REFERENCE = 0.2781  # seasoncombo combo 1, this variant's baseline

function main()
    t0 = time()
    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")
    truth = load_oracle(HUB_PATH)

    profile = build_seasonal_profile(
        hist; transform=TRANSFORM, max_season_year=MAX_TRAIN_SEASON_YEAR,
        min_support=MIN_SUPPORT, smooth_window=SMOOTH_WINDOW,
    )
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= MAX_TRAIN_SEASON_YEAR, :,
    ]
    backfill_profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_WINDOW,
        min_support=5, mode=BF_MODE, pooled=false, stat=BF_STAT,
    )

    results = NamedTuple[]
    for seg_len in SEG_LENS, bw_mult in BW_MULTS, jitter_frac in JITTER_FRACS
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, backfill_profile;
            seg_len=seg_len, bw_mult=bw_mult, jitter_frac=jitter_frac,
            model_id="smoother",
        )
        summ = score_one(fc, truth)
        push!(results, (
            seg_len=seg_len, bw_mult=bw_mult, jitter_frac=jitter_frac,
            mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
            n_tasks=summ.n_tasks,
        ))
        println("seg_len=$seg_len bw_mult=$bw_mult jitter_frac=$jitter_frac " *
                "-> mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(results; by=r -> r.mean_wis)
    best = results[1]

    # Rebuild + score the winning config's forecast table once more so
    # its region/time breakdown can be reported alongside the sweep.
    best_fc = build_forecast_table(
        VALIDATION_ONLY, versions_full, profile, backfill_profile;
        seg_len=best.seg_len, bw_mult=best.bw_mult,
        jitter_frac=best.jitter_frac, model_id="smoother-best",
    )
    scored = score_forecasts(best_fc, truth; scale=:natural)

    by_loc = combine(groupby(scored, :location), :wis => mean => :mean_wis)
    sort!(by_loc, :mean_wis; rev=true)
    by_h = combine(groupby(scored, :horizon), :wis => mean => :mean_wis)
    sort!(by_h, :horizon)
    scored.season_year = season_year.(scored.origin_date)
    by_season = combine(
        groupby(scored, :season_year), :wis => mean => :mean_wis,
    )
    sort!(by_season, :season_year)

    dt = round(time() - t0; digits=1)
    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "smoother -- simple-round, CLOSE-VARIANT family")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(dt)s")
        println(io)
        println(io, "reference (this variant is a CLOSE variant of):")
        println(io, "  seasoncombo core (pooled-season + AR(6) + backfill) " *
                     "= $(CORE_REFERENCE)")
        println(io, "  seabbs_bot-ar6bf (AR(6) + backfill, no season)      " *
                     "= 0.359")
        println(io, "  nfidd-ar6 (plain AR(6), no season)                 " *
                     "= 0.368")
        println(io)
        println(io, "swap under test: same pooled seasonal shape + same " *
                     "backfill correction as seasoncombo core, but the " *
                     "AR(6) residual model is replaced by a NON-AR " *
                     "kernel-weighted analogue/nearest-neighbour " *
                     "bootstrap forecaster (see docstrings above).")
        println(io)
        println(io, "sweep: seg_len x bw_mult x jitter_frac " *
                     "($(length(SEG_LENS))x$(length(BW_MULTS))x" *
                     "$(length(JITTER_FRACS)) = $(length(results)) runs)")
        for r in results
            println(io, "  seg_len=$(r.seg_len) bw_mult=$(r.bw_mult) " *
                         "jitter_frac=$(r.jitter_frac) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io)
        println(io, "=== best: seg_len=$(best.seg_len) " *
                     "bw_mult=$(best.bw_mult) " *
                     "jitter_frac=$(best.jitter_frac) ===")
        println(io, "mean_wis=$(round(best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(best.sd_wis; digits=4)) " *
                     "n_tasks=$(best.n_tasks)")
        vs_core = CORE_REFERENCE - best.mean_wis
        vs_core_pct = 100 * vs_core / CORE_REFERENCE
        vs_core_r = round(vs_core; digits=4)
        vs_core_pct_r = round(vs_core_pct; digits=2)
        println(io, "vs seasoncombo core reference ($(CORE_REFERENCE)): " *
                     "$(vs_core_r) ($(vs_core_pct_r)%)")
        println(io)
        println(io, "-- breakdown by location (best config, mean WIS) --")
        for row in eachrow(by_loc)
            println(io, "$(row.location)  $(round(row.mean_wis; digits=4))")
        end
        println(io)
        println(io, "-- breakdown by horizon (best config, mean WIS) --")
        for row in eachrow(by_h)
            println(io, "h=$(row.horizon): $(round(row.mean_wis; digits=4))")
        end
        println(io)
        println(io, "-- breakdown by validation season (best config, " *
                     "mean WIS) --")
        for row in eachrow(by_season)
            println(io, "season $(row.season_year): " *
                         "$(round(row.mean_wis; digits=4))")
        end
        println(io)
        if best.mean_wis < CORE_REFERENCE
            println(io, "CONCLUSION: the analogue/kernel residual model " *
                         "BEATS the AR(6)-residual core reference while " *
                         "keeping the same seasonal + backfill terms.")
        else
            println(io, "CONCLUSION: the analogue/kernel residual model " *
                         "does NOT beat the AR(6)-residual core reference " *
                         "here -- reported anyway as a structurally " *
                         "DIVERSE (non-AR, non-parametric) forecaster, " *
                         "potentially useful for ensembling even if not " *
                         "individually the best.")
        end
    end

    println("\nbest: seg_len=$(best.seg_len) bw_mult=$(best.bw_mult) " *
            "jitter_frac=$(best.jitter_frac) " *
            "mean_wis=$(round(best.mean_wis; digits=4)) " *
            "sd_wis=$(round(best.sd_wis; digits=4))")
    println("wrote score.txt in $(dt)s total")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
