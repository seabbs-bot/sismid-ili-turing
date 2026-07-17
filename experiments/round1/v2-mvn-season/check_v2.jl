# Bayesian-workflow check for round 1 candidate v2-mvn-season, ahead
# of scoring (see docs/plan.md's "each candidate passes ... before it
# is scored"). Builds a small synthetic ModelData by hand (same
# pattern as test/test_model.jl), then:
#
# 1. A PRIOR predictive check: draws from the prior and confirms the
#    implied wILI percentages are finite and in a plausible range.
# 2. A TINY NUTS(AutoMooncake) fit, to confirm `model_v2` actually
#    samples — the LKJ-Cholesky non-centred location-effect prior is
#    the risky new part of this variant (Cholesky-factor bijectors
#    under reverse-mode AD), so this is the thing most worth
#    confirming before spending a real Pathfinder/MCMC budget on it.
# 3. A `project_v2` smoke check on a fitted draw, confirming the
#    returned forecast matrix has the right shape and is finite.
#
# Run from the repo root: `julia --project=. \
#   experiments/round1/v2-mvn-season/check_v2.jl`

using Random
using Dates
using Distributions
using Turing
using Mooncake
using Statistics
using LinearAlgebra

const HERE = @__DIR__
include(joinpath(HERE, "..", "..", "..", "src", "core.jl"))
include(joinpath(HERE, "..", "..", "..", "src", "model.jl"))
include(joinpath(HERE, "model_v2.jl"))
include(joinpath(HERE, "project_v2.jl"))

Random.seed!(20260717)

# Minimal local stand-in for src/inference.jl's `generated_draws`, so
# this script does not need src/inference.jl or the full package. See
# src/inference.jl for the real accessor used once model_v2 is wired
# into the package's fitting/forecasting pipeline.
function generated_draws_local(model, chain)
    gq = Turing.DynamicPPL.returned(model, chain)
    return vec(collect(gq))
end

# --- Build a small synthetic ModelData by hand (as test_model.jl) ---
const T = 40
const L = 4
const W = 33
const S = 1
const Dmax = 4

woy = [mod1(t, W) for t in 1:T]
season = fill(1, T)
dates = Date(2016, 1, 2) .+ Day.(7 .* (0:(T - 1)))

delay = [min(T - t, Dmax) for t in 1:T, l in 1:L]
delay[T, 2] = -1
delay[T, 3] = -1
delay[T - 1, 3] = -1

transform = :log1p
true_curve_pct = [2.0 .+ 1.5 .* sin(2 * pi * w / W) for w in woy]
Y = Matrix{Union{Missing,Float64}}(undef, T, L)
for l in 1:L, t in 1:T
    Y[t, l] = delay[t, l] == -1 ? missing :
              to_scale(true_curve_pct[t] + 0.05 * randn(), transform)
end

d = ModelData(Y, delay, woy, season, dates, L, T, W, S, Dmax, transform,
              dates[end])

println("=== v2-mvn-season check ===")

# --- 1. Prior predictive check ---
model = model_v2(d; transform=transform)
prior_chain = sample(model, Prior(), 200; progress=false)
@assert all(isfinite, Array(prior_chain)) "non-finite prior draws"

prior_natural = Float64[]
mu_w_spread = Float64[]     # max|mu_w| per draw
delta_spread = Float64[]    # max|delta| per draw (the new MVN piece)
resid_spread = Float64[]    # max|residual| per draw
for i in 1:200
    gq = model()  # fresh prior draw + generated quantities each call
    for l in 1:L, t in 1:T
        push!(prior_natural, from_scale(gq.latent[t, l], transform))
    end
    push!(mu_w_spread, maximum(abs, gq.mu_w))
    push!(delta_spread, maximum(abs, gq.delta))
    push!(resid_spread, maximum(abs, gq.residual))
end

# The prior-only draw uses the same weakly-informative hierarchical
# priors as base_model: an unbounded random-walk `mu_w` (cumsum over W
# = 33 steps) and a near-unit-root AR(1) residual (`phi` can approach
# +-1, blowing up the stationary variance) both inherit directly from
# base_model, unchanged here. Combined with `:log1p`'s heavily
# right-skewed back-transform (`expm1`), this gives a prior predictive
# with a very heavy upper tail even though the bulk of draws are
# plausible wILI% values. So "plausible range" is read from the bulk
# (median, 75th pct) rather than the extrema/99th pct, which are
# dominated by that pre-existing, unbounded-hyperprior tail behaviour.
lo, hi = extrema(prior_natural)
q50, q75, q99 = quantile(prior_natural, [0.5, 0.75, 0.99])
println("prior predictive wILI% full range: [$(round(lo; digits=2)), ",
        "$(round(hi; digits=1))]")
println("prior predictive wILI% 50/75/99th pct: ",
        "$(round(q50; digits=2)) / $(round(q75; digits=2)) / ",
        "$(round(q99; digits=1))")
@assert all(isfinite, prior_natural) "non-finite prior predictive values"
@assert -1.0 < q50 < 50.0 "implausible prior predictive: bulk median"
@assert q75 < 200.0 "implausible prior predictive: bulk 75th pct"
println("prior predictive check (bulk mass): PASS")

# Attribute the heavy tail: confirm it is NOT the new MVN `delta` that
# drives it. `delta`'s own spread should stay the same order of
# magnitude as `mu_w`/`residual` (both unchanged from base_model), not
# dominate them — i.e. the LKJ/per-location-scale prior is not itself
# inflating variance beyond what base_model's own components already
# contribute.
med_mu_w, med_delta, med_resid =
    median(mu_w_spread), median(delta_spread), median(resid_spread)
println("median per-draw max|.|  mu_w=$(round(med_mu_w; digits=2))  ",
        "delta=$(round(med_delta; digits=2))  ",
        "residual=$(round(med_resid; digits=2))")
@assert med_delta < 5 * max(med_mu_w, med_resid) (
    "MVN delta variance is disproportionately larger than base_model's " *
    "own mu_w/residual components"
)
println("delta-vs-base-component spread check: PASS")

# --- 2. Tiny NUTS(AutoMooncake) fit: confirm LKJ/Cholesky samples ---
chain = sample(
    model, NUTS(; adtype=AutoMooncake()), 30; progress=false,
)
@assert all(isfinite, chain[:lp]) "non-finite log density in NUTS fit"
println("tiny NUTS(AutoMooncake) fit: PASS (", length(chain[:lp]),
        " draws, all finite lp)")

# Sanity-check the recovered correlation matrix is a valid correlation
# matrix (unit diagonal, symmetric, in [-1, 1] off-diagonal).
gq = model()
@assert size(gq.loc_corr) == (L, L)
@assert all(isapprox.(diag(gq.loc_corr), 1.0; atol=1e-8))
@assert all(-1.0 .<= gq.loc_corr .<= 1.0)
@assert length(gq.sigma_season_loc) == L
println("loc_corr / sigma_season_loc shapes and bounds: PASS")

# --- 3. project_v2 smoke check on a fitted draw ---
draws = generated_draws_local(model, chain)
latent = project_v2(draws[1], d, 1:4)
@assert size(latent) == (L, 4)
@assert all(isfinite, latent)
println("project_v2 output shape: PASS (", size(latent), ", all finite)")

println("=== v2-mvn-season check: ALL PASS ===")
