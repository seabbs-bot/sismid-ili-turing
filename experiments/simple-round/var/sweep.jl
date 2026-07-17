#!/usr/bin/env julia
# sweep.jl -- multi-location AR/VAR family, simple-model wide round.
#
# Family: submissions/nfidd-ar6/generate_forecasts.jl (plain per-
# location AR(p), OLS on fourthroot-transformed vintage series, 1000
# simulated Gaussian-innovation paths -- CSV/DataFrames/Statistics/
# LinearAlgebra only, no `SismidILITuring`/Turing). This sweep asks
# whether OTHER locations' recent values, added as extra predictors,
# improve on the plain per-location AR(6) baseline.
#
# docs/eda/04-cross-location.md: cross-location coupling in the
# differenced series is real but MODERATE (mean r=0.24) and
# CONTEMPORANEOUS (every region's cross-correlation with the national
# series peaks at lag 0, not lag >= 1) -- so a lagged VAR is not
# expected to recover the full correlation (that lives in the same-
# week innovations, not the lagged levels), and a dense, unregularised
# VAR(p) over all 11 locations is expected to overfit given the
# window_weeks=104 training history. Both expectations are tested
# directly below (`var1_ols` vs the ridge-penalised variants).
#
# Candidates:
#   ar6_baseline        -- plain per-location AR(6), no cross terms
#   var1_*               -- own AR(6) + ALL other locations' lag-1
#                            value, ridge-penalised (lambda on the
#                            cross-location coefficients only; own
#                            AR(6) lags are never penalised)
#   var2_*               -- as var1, plus each other location's lag-2
#                            value too (a full VAR(2))
#   ar_nat_ols            -- own AR(6) + `US National`'s lag-1 value
#                            only (single common-factor predictor,
#                            reduced-rank in spirit), OLS
#   ar_neighbor_ols        -- own AR(6) + the single most-correlated
#                            OTHER location's lag-1 value (computed
#                            from the training window itself, not
#                            hand-coded geography), OLS
#   ar_nat_neighbor_ols    -- both of the above combined, OLS
#
# All candidates simulate `NPATHS` Gaussian-innovation paths forward
# h=1..4 JOINTLY across all 11 locations: because the cross-location
# predictors are LAGGED (lag 1, never lag 0), simulating h=2 for one
# location genuinely depends on every location's simulated h=1 value,
# not just its own -- true VAR forward propagation, not 11 independent
# per-location simulations bolted together.
#
# Scored on VALIDATION SEASONS ONLY (1, 2 = 2015/16, 2016/17) against
# target-data/oracle-output.csv in the local hub clone
# (~/code/external/sismid-ili-forecasting-sandbox), WIS computed
# in-line (same formula as src/scoring.jl's `wis`, reimplemented here
# to avoid the ScoringRules dependency and stay on the light stack).
# Test seasons (3-5) are never generated, scored, or looked at.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> sweep.jl

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))

const HUB_PATH = joinpath(homedir(), "code", "external",
                           "sismid-ili-forecasting-sandbox")
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const NAT_IDX = findfirst(==("US National"), LOCATIONS)
const L = length(LOCATIONS)

# ---------------------------------------------------------------------
# Predictor spec: own AR(own_order), plus lag-1..cross_lags of each
# OTHER location listed in cross_idx[l]. Ridge lambda applies
# separately to the own-AR block (`lambda_own`, always 0 here -- the
# per-location AR(6) baseline is not the overfitting risk) and the
# cross-location block (`lambda_cross`, the actual regulariser).
# ---------------------------------------------------------------------

struct Spec
    own_order::Int
    cross_idx::Vector{Vector{Int}}
    cross_lags::Int
    lambda_own::Float64
    lambda_cross::Float64
end

"""
    build_row(tail, l, spec) -> Vector{Float64}

One predictor row for location `l`: intercept, `spec.own_order` own
lags (most recent first), then `spec.cross_lags` lags of each other
location in `spec.cross_idx[l]`. `tail` holds the most recent
`max(own_order, cross_lags)` rows (all locations), most recent LAST.
"""
function build_row(tail::AbstractMatrix{Float64}, l::Int, spec::Spec)
    x = Float64[1.0]
    for k in 1:spec.own_order
        push!(x, tail[end - k + 1, l])
    end
    for j in spec.cross_idx[l], k in 1:spec.cross_lags
        push!(x, tail[end - k + 1, j])
    end
    return x
end

"""
    fit_loc(Y, l, spec) -> (coef, resid_sd)

Ridge fit of location `l`'s predictor row (`build_row`) against
`Y[:, l]`, over every time step with a full lag history available.
Equivalent to OLS when both lambdas are 0.
"""
function fit_loc(Y::Matrix{Float64}, l::Int, spec::Spec)
    T = size(Y, 1)
    maxlag = max(spec.own_order, spec.cross_lags)
    nobs = T - maxlag
    ncol = 1 + spec.own_order + spec.cross_lags * length(spec.cross_idx[l])
    X = Matrix{Float64}(undef, nobs, ncol)
    yresp = Vector{Float64}(undef, nobs)
    for (row, t) in enumerate((maxlag + 1):T)
        X[row, :] = build_row((@view Y[(t - maxlag):(t - 1), :]), l, spec)
        yresp[row] = Y[t, l]
    end
    lam = fill(spec.lambda_cross, ncol)
    lam[1] = 0.0
    lam[2:(1 + spec.own_order)] .= spec.lambda_own
    coef = (X'X + Diagonal(lam)) \ (X'yresp)
    resid = yresp .- X * coef
    dof = max(nobs - ncol, 1)
    resid_sd = sqrt(sum(abs2, resid) / dof)
    return coef, resid_sd
end

"""
    simulate_joint(Y, coefs, resid_sds, spec, horizons, npaths; rng)
        -> Dict{Int,Vector{Vector{Float64}}}

Simulate `npaths` Gaussian-innovation sample paths forward from the
end of `Y`, JOINTLY across all `L` locations: every location's step
uses the SAME simulated tail, so lagged cross-location feedback
propagates correctly into later horizons.
"""
function simulate_joint(
    Y::Matrix{Float64}, coefs::Vector{Vector{Float64}},
    resid_sds::Vector{Float64}, spec::Spec, horizons, npaths::Int;
    rng::Random.AbstractRNG,
)
    L_ = size(Y, 2)
    maxlag = max(spec.own_order, spec.cross_lags)
    hmax = maximum(horizons)
    out = Dict(h => [Vector{Float64}(undef, npaths) for _ in 1:L_]
               for h in horizons)
    tail0 = Y[(end - maxlag + 1):end, :]
    for s in 1:npaths
        tail = copy(tail0)
        for h in 1:hmax
            newrow = Vector{Float64}(undef, L_)
            for l in 1:L_
                pred = dot(coefs[l], build_row(tail, l, spec))
                newrow[l] = pred + resid_sds[l] * randn(rng)
            end
            if h in horizons
                for l in 1:L_
                    out[h][l][s] = newrow[l]
                end
            end
            tail = vcat(tail[2:end, :], reshape(newrow, 1, L_))
        end
    end
    return out
end

"""
    nearest_neighbour(Y) -> Vector{Int}

For each location column, the index of the OTHER location most
correlated with it over the training window `Y` (levels, on the
modelling scale). Computed fresh per split -- no hand-coded geography.
"""
function nearest_neighbour(Y::Matrix{Float64})
    C = cor(Y)
    nn = Vector{Int}(undef, L)
    for l in 1:L
        best, bestj = -Inf, l
        for j in 1:L
            j == l && continue
            if C[l, j] > best
                best, bestj = C[l, j], j
            end
        end
        nn[l] = bestj
    end
    return nn
end

# ---------------------------------------------------------------------
# WIS, reimplemented in-line (formula matches src/scoring.jl's `wis`;
# avoids the ScoringRules dependency so this stays on the light stack).
# ---------------------------------------------------------------------

function wis_local(observation::Float64, values::Vector{Float64},
        levels::Vector{Float64})
    median_idx = findfirst(a -> abs(a - 0.5) < 1e-8, levels)
    median = values[median_idx]
    lower_levels = filter(a -> a < 0.5 - 1e-8, levels)
    K = length(lower_levels)
    is_sum = 0.0
    for a in lower_levels
        li = findfirst(x -> abs(x - a) < 1e-8, levels)
        ui = findfirst(x -> abs(x - (1 - a)) < 1e-8, levels)
        lo, hi = values[li], values[ui]
        alpha = 2a
        is_k = (hi - lo) + (2 / alpha) * max(lo - observation, 0.0) +
               (2 / alpha) * max(observation - hi, 0.0)
        is_sum += (alpha / 2) * is_k
    end
    return (0.5 * abs(observation - median) + is_sum) / (K + 0.5)
end

"""Hub oracle as a `(location, target_end_date) => value` lookup."""
function load_oracle_lookup(hub_path)
    df = CSV.read(joinpath(hub_path, "target-data", "oracle-output.csv"),
                  DataFrame)
    truth = DataFrame(location=String.(df.location),
                      target_end_date=Date.(df.target_end_date),
                      value=Float64.(df.oracle_value))
    dropmissing!(truth)
    return Dict((r.location, r.target_end_date) => r.value
                for r in eachrow(truth))
end

const ORACLE = load_oracle_lookup(HUB_PATH)

"""
    score_spec(spec_builder, seasons) -> (mean_wis, sd_wis, n)

Fit + jointly simulate `spec_builder(Y)` on every split of `seasons`,
score every (location, horizon) task against `ORACLE`.
`spec_builder` takes the split's `Y` matrix so that data-dependent
specs (e.g. the nearest-neighbour candidates) are rebuilt per split,
with no leakage across splits.
"""
function score_spec(spec_builder, seasons)
    rng = MersenneTwister(SEED)
    wis_vals = Float64[]
    for season in seasons
        for split in training_splits(season)
            data = build_model_data(
                split; Dmax=12, transform=TRANSFORM, window_weeks=104,
            )
            Y = Float64.(data.Y)
            spec = spec_builder(Y)
            coefs = Vector{Vector{Float64}}(undef, data.L)
            resid_sds = Vector{Float64}(undef, data.L)
            for l in 1:data.L
                coefs[l], resid_sds[l] = fit_loc(Y, l, spec)
            end
            paths = simulate_joint(
                Y, coefs, resid_sds, spec, HORIZONS, NPATHS; rng=rng,
            )
            for h in HORIZONS, (li, loc) in enumerate(LOCATIONS)
                key = (loc, data.origin_date + Day(7 * h))
                haskey(ORACLE, key) || continue
                obs = ORACLE[key]
                vals = [max(from_scale(quantile(paths[h][li], q),
                                       TRANSFORM), 0.0)
                        for q in QUANTILE_LEVELS]
                push!(wis_vals, wis_local(obs, vals, QUANTILE_LEVELS))
            end
        end
    end
    return mean(wis_vals), std(wis_vals), length(wis_vals)
end

none_idx = [Int[] for _ in 1:L]
all_other_idx = [[j for j in 1:L if j != l] for l in 1:L]
nat_only_idx = [l == NAT_IDX ? Int[] : [NAT_IDX] for l in 1:L]
neighbour_only_idx(Y) = [[nn] for nn in nearest_neighbour(Y)]
function nat_neighbour_idx(Y)
    nn = nearest_neighbour(Y)
    return [l == NAT_IDX ? [nn[l]] : unique([NAT_IDX, nn[l]]) for l in 1:L]
end

candidates = [
    ("ar6_baseline",       Y -> Spec(AR_ORDER, none_idx, 1, 0.0, 0.0)),
    ("var1_ols",           Y -> Spec(AR_ORDER, all_other_idx, 1, 0.0, 0.0)),
    ("var1_ridge0.5",      Y -> Spec(AR_ORDER, all_other_idx, 1, 0.0, 0.5)),
    ("var1_ridge1",        Y -> Spec(AR_ORDER, all_other_idx, 1, 0.0, 1.0)),
    ("var1_ridge1.5",      Y -> Spec(AR_ORDER, all_other_idx, 1, 0.0, 1.5)),
    ("var1_ridge2",        Y -> Spec(AR_ORDER, all_other_idx, 1, 0.0, 2.0)),
    ("var1_ridge3",        Y -> Spec(AR_ORDER, all_other_idx, 1, 0.0, 3.0)),
    ("var1_ridge5",        Y -> Spec(AR_ORDER, all_other_idx, 1, 0.0, 5.0)),
    ("var2_ridge5",        Y -> Spec(AR_ORDER, all_other_idx, 2, 0.0, 5.0)),
    ("var2_ridge50",       Y -> Spec(AR_ORDER, all_other_idx, 2, 0.0, 50.0)),
    ("ar_nat_ols",         Y -> Spec(AR_ORDER, nat_only_idx, 1, 0.0, 0.0)),
    ("ar_neighbor_ols",
        Y -> Spec(AR_ORDER, neighbour_only_idx(Y), 1, 0.0, 0.0)),
    ("ar_nat_neighbor_ols",
        Y -> Spec(AR_ORDER, nat_neighbour_idx(Y), 1, 0.0, 0.0)),
]

t0 = time()
println("=== validation seasons (1,2) WIS by candidate ===")
results = NamedTuple[]
for (name, spec_builder) in candidates
    m, s, n = score_spec(spec_builder, (1, 2))
    println(rpad(name, 20), "mean_wis=", round(m; digits=4),
            " sd_wis=", round(s; digits=4), " n=", n)
    push!(results, (; name, mean_wis=m, sd_wis=s, n))
end
sort!(results; by=r -> r.mean_wis)
println("\n=== ranked ===")
for r in results
    println(rpad(r.name, 20), round(r.mean_wis; digits=4))
end
println("\ntotal sweep runtime: ", round(time() - t0; digits=1), "s")

# --- breakdown of the winner vs the plain AR(6) baseline ---

winner_idx = findfirst(c -> c[1] == results[1].name, candidates)
winner_builder = candidates[winner_idx][2]
println("\n=== breakdown: $(results[1].name) (winner) vs ar6_baseline ===")

function score_spec_tasks(spec_builder, seasons)
    rng = MersenneTwister(SEED)
    recs = NamedTuple[]
    for season in seasons
        for split in training_splits(season)
            data = build_model_data(
                split; Dmax=12, transform=TRANSFORM, window_weeks=104,
            )
            Y = Float64.(data.Y)
            spec = spec_builder(Y)
            coefs = Vector{Vector{Float64}}(undef, data.L)
            resid_sds = Vector{Float64}(undef, data.L)
            for l in 1:data.L
                coefs[l], resid_sds[l] = fit_loc(Y, l, spec)
            end
            paths = simulate_joint(
                Y, coefs, resid_sds, spec, HORIZONS, NPATHS; rng=rng,
            )
            syear = season_year(data.origin_date)
            for h in HORIZONS, (li, loc) in enumerate(LOCATIONS)
                key = (loc, data.origin_date + Day(7 * h))
                haskey(ORACLE, key) || continue
                obs = ORACLE[key]
                vals = [max(from_scale(quantile(paths[h][li], q),
                                       TRANSFORM), 0.0)
                        for q in QUANTILE_LEVELS]
                w = wis_local(obs, vals, QUANTILE_LEVELS)
                push!(recs, (; loc, h, syear, w))
            end
        end
    end
    return recs
end

recs = score_spec_tasks(winner_builder, (1, 2))
recs_base = score_spec_tasks(candidates[1][2], (1, 2))

by_loc = sort(combine(groupby(DataFrame(recs), :loc),
                       :w => mean => :mean_wis), :mean_wis)
println("-- by location --")
foreach(r -> println(rpad(r.loc, 16), round(r.mean_wis; digits=4)),
        eachrow(by_loc))

by_h = sort(combine(groupby(DataFrame(recs), :h),
                     :w => mean => :mean_wis), :h)
println("-- by horizon --")
foreach(r -> println("h=$(r.h): ", round(r.mean_wis; digits=4)),
        eachrow(by_h))

by_s = sort(combine(groupby(DataFrame(recs), :syear),
                     :w => mean => :mean_wis), :syear)
println("-- by season --")
foreach(r -> println("season $(r.syear): ", round(r.mean_wis; digits=4)),
        eachrow(by_s))

n_improved = count(a.w < b.w for (a, b) in zip(recs, recs_base))
println("\ntask-level improvements vs ar6 baseline: $(n_improved) / ",
        "$(length(recs)) ",
        "($(round(100 * n_improved / length(recs); digits=1))%)")
