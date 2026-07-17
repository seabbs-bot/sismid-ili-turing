#!/usr/bin/env julia
# generate.jl -- ANALYTIC PARTIAL POOLING for the simple-model wide round.
#
# Same family as submissions/nfidd-ar6/generate_forecasts.jl (plain
# per-location AR(6), OLS, fourthroot, 1000 simulated paths): the only
# thing varied here is whether, and how much, each location's AR(6)
# coefficient vector is shrunk toward a pooled estimate before
# simulating. This is the poor-man's version of the Turing model's
# hierarchical partial pooling -- no Turing, just linear algebra -- and
# it exists to answer one question directly: does borrowing strength
# across locations help on this data at all? If analytic shrinkage
# barely moves the needle, that is evidence the Turing model's much
# more expensive hierarchical pooling may not be earning its cost
# either; if it moves the needle a lot, it is.
#
# Two pool anchors, each swept over a shrinkage weight w in [0, 1]:
#   :fullpool -- coefficients from one OLS fit on all 11 locations'
#                AR(6) design rows stacked together (a common-dynamics
#                estimate across the whole system).
#   :national -- the "US National" location's own AR(6) OLS fit, used
#                as the pooling anchor for the 10 HHS regions (w has no
#                effect on the national series itself: its own OLS fit
#                is its own anchor).
# w=0 recovers the unpooled nfidd-ar6 baseline exactly; w=1 forecasts
# every location from the anchor's coefficients alone (full pooling).
#
# Also included: a one-shot James-Stein / empirical-Bayes scheme with
# no weight to sweep -- per AR(6) coefficient, the shrinkage weight
# toward the cross-location mean is set automatically from the ratio
# of between-location coefficient variance to average within-location
# sampling variance (more shrinkage where locations agree with each
# other and the OLS fit is noisy; less where locations genuinely
# differ or data is plentiful).
#
# In every scheme, `resid_sd` is recomputed from the *blended*
# coefficients' own residuals on that location's data (not the
# unpooled fit's residual SD) so simulated path spread reflects the
# pooled model actually being forecast from.
#
# Deliberately avoids `using SismidILITuring` (Turing/Mooncake/
# Pathfinder weight, see nfidd-ar6's own header comment) --
# CSV/DataFrames/Dates/Statistics/LinearAlgebra/ScoringRules only.
#
# Scores on the VALIDATION seasons only (1, 2 -- docs/contracts.md
# experimental integrity) against the hub oracle, and writes the best
# variant's forecast table + a score summary alongside this script.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra
using ScoringRules

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))
include(joinpath(PKG_DIR, "src", "hubio.jl"))
include(joinpath(PKG_DIR, "src", "scoring.jl"))

const HUB_PATH = joinpath(
    homedir(), "code", "external", "sismid-ili-forecasting-sandbox",
)
const OUT_DIR = @__DIR__
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12
const VALIDATION = (1, 2)
const WEIGHTS = [0.1, 0.25, 0.5, 0.75, 1.0]
const NATIONAL_IDX = findfirst(==("US National"), LOCATIONS)

# ---------------------------------------------------------------------
# AR(6) fit + forecast -- identical in form to nfidd-ar6, factored so
# residual SD can be recomputed for a coefficient vector other than the
# one the design matrix was fit on (i.e. a shrunk/blended coefficient).
# ---------------------------------------------------------------------

"""
    ar_design(y, order) -> (X, yresp)

Design matrix and response for an OLS AR(`order`) fit with intercept:
`X = [1 y[t-1] ... y[t-order]]`, `yresp[row] = y[t]`, ascending `t`.
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
`yresp`) evaluated on this design, with `nobs - (order + 1)` degrees of
freedom -- how well a (possibly shrunk) coefficient vector fits this
location's own data.
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

OLS fit of an AR(`order`) model with intercept to `y`. Returns the
coefficient vector plus the design so callers can also evaluate other
(shrunk) coefficient vectors on the same data via `resid_sd_for`.
"""
function fit_ar(y::AbstractVector{Float64}, order::Int)
    X, yresp = ar_design(y, order)
    coef = X \ yresp
    return coef, X, yresp
end

"""
    fit_ar_pooled(ys, order) -> coef

One OLS AR(`order`) fit on the design rows of every series in `ys`
stacked together -- a single common-dynamics coefficient vector across
all locations, the `:fullpool` anchor.
"""
function fit_ar_pooled(ys::Vector{Vector{Float64}}, order::Int)
    designs = [ar_design(y, order) for y in ys]
    Xall = reduce(vcat, first.(designs))
    yall = reduce(vcat, last.(designs))
    return Xall \ yall
end

"""
    js_shrink(coefs, resid_sds, nobs) -> Vector{Vector{Float64}}

Empirical-Bayes ("James-Stein flavoured") shrinkage of each location's
AR coefficient vector toward the cross-location mean, one shrinkage
weight per coefficient position. For coefficient `j`, the
between-location variance `tau2_j = var(coefs[:][j]) - vbar` (method-
of-moments, floored above zero) is compared with the average
within-location sampling variance `vbar = mean(resid_sds.^2 ./ nobs)`;
the weight on the location's own estimate is `tau2_j / (tau2_j +
vbar)`, i.e. more shrinkage toward the pool where locations agree with
each other and/or fits are noisy, less where locations genuinely
differ or data is plentiful.
"""
function js_shrink(
    coefs::Vector{Vector{Float64}}, resid_sds::Vector{Float64},
    nobs::Vector{Int},
)
    L = length(coefs)
    p = length(coefs[1])
    cbar = [mean(c[j] for c in coefs) for j in 1:p]
    vbar = mean(resid_sds[l]^2 / nobs[l] for l in 1:L)
    blended = Vector{Vector{Float64}}(undef, L)
    for l in 1:L
        newc = similar(coefs[l])
        for j in 1:p
            tau2 = max(var(c[j] for c in coefs) - vbar, 1e-8)
            wj = tau2 / (tau2 + vbar)
            newc[j] = wj * coefs[l][j] + (1 - wj) * cbar[j]
        end
        blended[l] = newc
    end
    return blended
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

# ---------------------------------------------------------------------
# Forecast table builder, parameterised on pooling scheme
# ---------------------------------------------------------------------

"""
    build_forecast_table(model_id, seasons, order, scheme) -> DataFrame

Fit and forecast an AR(`order`) model for every split of every season
in `seasons`, blending each location's OLS coefficients according to
`scheme`:

  - `(:none,)`             -- no pooling (nfidd-ar6 baseline).
  - `(:fixed, :fullpool, w)` / `(:fixed, :national, w)`
                            -- blend `(1 - w) * own + w * anchor`.
  - `(:js,)`                -- empirical-Bayes per-coefficient weight.
"""
function build_forecast_table(model_id, seasons, order::Int, scheme)
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
            )
            origin = data.origin_date
            L = length(LOCATIONS)

            ys = [Float64.(data.Y[:, li]) for li in 1:L]
            fits = [fit_ar(ys[li], order) for li in 1:L]
            coefs = [f[1] for f in fits]
            Xs = [f[2] for f in fits]
            yresps = [f[3] for f in fits]
            nobs = [size(Xs[li], 1) for li in 1:L]
            own_resid_sd = [
                resid_sd_for(Xs[li], yresps[li], coefs[li], order)
                for li in 1:L
            ]

            blended = if scheme[1] == :none
                coefs
            elseif scheme[1] == :fixed
                _, anchor_kind, w = scheme
                anchor = anchor_kind == :fullpool ?
                    fit_ar_pooled(ys, order) : coefs[NATIONAL_IDX]
                [(1 - w) .* coefs[li] .+ w .* anchor for li in 1:L]
            elseif scheme[1] == :js
                js_shrink(coefs, own_resid_sd, nobs)
            else
                error("unknown scheme: $(scheme)")
            end

            for li in 1:L
                loc = LOCATIONS[li]
                coef = blended[li]
                resid_sd = resid_sd_for(Xs[li], yresps[li], coef, order)
                paths = simulate_paths(
                    ys[li], coef, resid_sd, order, HORIZONS, NPATHS;
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

"""Hub oracle (`target-data/oracle-output.csv`) as a scoring truth table."""
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
    truth = load_oracle(HUB_PATH)

    schemes = Tuple[(:none,)]
    for w in WEIGHTS
        push!(schemes, (:fixed, :fullpool, w))
    end
    for w in WEIGHTS
        push!(schemes, (:fixed, :national, w))
    end
    push!(schemes, (:js,))

    scheme_label(s) = s[1] == :none ? "none" :
        s[1] == :js ? "js" :
        "$(s[2])-w$(s[3])"

    results = DataFrame(
        scheme=String[], mean_wis=Float64[], sd_wis=Float64[],
    )
    scored_by_scheme = Dict{String,DataFrame}()
    forecast_by_scheme = Dict{String,DataFrame}()

    for scheme in schemes
        label = scheme_label(scheme)
        model_id = "pool-$(label)"
        forecast = build_forecast_table(
            model_id, VALIDATION, AR_ORDER, scheme,
        )
        scored = score_forecasts(forecast, truth; scale=:natural)
        summ = wis_summary(scored)
        push!(results, (label, summ.mean_wis[1], summ.sd_wis[1]))
        scored_by_scheme[label] = scored
        forecast_by_scheme[label] = forecast
        println("scheme=$(label): mean_wis=$(round(summ.mean_wis[1]; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis[1]; digits=4)) " *
                "($(round(time() - t0; digits=1))s elapsed)")
    end

    sort!(results, :mean_wis)
    println("\n=== ranked (validation seasons $(VALIDATION)) ===")
    println(results)

    best = results[1, :]
    none_wis = results[results.scheme .== "none", :mean_wis][1]
    println("\nbest: scheme=$(best.scheme) " *
            "mean_wis=$(round(best.mean_wis; digits=4)) " *
            "sd_wis=$(round(best.sd_wis; digits=4))")
    println("unpooled (none, == nfidd-ar6) mean_wis = " *
            "$(round(none_wis; digits=4))")
    println("baselines: nfidd-ar6 (order=6, no backfill) = 0.368; " *
            "seabbs_bot-ar6bf (order=6, backfill) = 0.359")

    best_scored = scored_by_scheme[best.scheme]
    by_loc = combine(groupby(best_scored, :location),
        :wis => mean => :mean_wis, nrow => :n)
    sort!(by_loc, :mean_wis)
    println("\n=== best scheme: mean WIS by location ===")
    println(by_loc)

    by_h = combine(groupby(best_scored, :horizon),
        :wis => mean => :mean_wis, nrow => :n)
    sort!(by_h, :horizon)
    println("\n=== best scheme: mean WIS by horizon ===")
    println(by_h)

    println("\ntotal time: $(round(time() - t0; digits=1))s")

    # Persist the best variant's forecast table + a score summary
    # alongside this script.
    best_forecast = forecast_by_scheme[best.scheme]
    best_forecast.model_id .= "pool"
    CSV.write(joinpath(OUT_DIR, "forecast.csv"), best_forecast)

    open(joinpath(OUT_DIR, "score.txt"), "w") do io
        println(io, "analytic partial pooling -- validation seasons " *
                "$(VALIDATION), AR($(AR_ORDER)), $(TRANSFORM) scale")
        println(io, "")
        println(io, "ranked schemes:")
        for row in eachrow(results)
            println(io, "  $(row.scheme): mean_wis=" *
                    "$(round(row.mean_wis; digits=4)) sd_wis=" *
                    "$(round(row.sd_wis; digits=4))")
        end
        println(io, "")
        println(io, "best: $(best.scheme) mean_wis=" *
                "$(round(best.mean_wis; digits=4)) sd_wis=" *
                "$(round(best.sd_wis; digits=4))")
        println(io, "unpooled (none) mean_wis=$(round(none_wis; digits=4))")
        println(io, "baselines: nfidd-ar6 = 0.368; seabbs_bot-ar6bf = 0.359")
    end
    println("\nwrote forecast.csv + score.txt to $(OUT_DIR)")

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
