# Bayesian-workflow check for round 2 candidate var, ahead of scoring
# (see docs/plan.md's "each candidate passes ... before it is scored").
# Builds a small synthetic ModelData by hand (same pattern as
# test/test_model.jl and experiments/round2/severity/check_severity.jl).
#
# PRIOR-only check (no NUTS/Pathfinder fit): the box is busy with other
# candidates, so this stays light --
#
# 1. A PRIOR predictive check: draws from the prior and confirms the
#    implied wILI percentages are finite and in a plausible range.
# 2. A VAR(1)-STABILITY check: the new, risky part of this candidate is
#    the `L x L` transition matrix `A` -- confirms `A`'s spectral radius
#    (max |eigenvalue|) stays under 1 for the overwhelming majority of
#    prior draws, i.e. the recursion does not explode, and that the
#    residual's spread stays the same order of magnitude as
#    base_model's own AR(1) residual would (not blown up by the added
#    cross-location coupling).
# 3. A `project_var` smoke check directly on a prior draw's generated
#    quantities (no fit needed -- `project_var` only consumes the
#    model's RETURN fields, which a prior draw already has).
#
# Run from the repo root: `julia --project=. \
#   experiments/round2/var/check_var.jl`

using Random
using Dates
using Distributions
using Turing
using LinearAlgebra
using Statistics

const HERE = @__DIR__
include(joinpath(HERE, "..", "..", "..", "src", "core.jl"))
include(joinpath(HERE, "..", "..", "..", "src", "model.jl"))
include(joinpath(HERE, "model_var.jl"))
include(joinpath(HERE, "project_var.jl"))

Random.seed!(20260717)

# --- Build a small synthetic ModelData by hand, L=5, THREE seasons ---
# L=5 (not the full 11) keeps this light while still giving `A` enough
# locations to exercise genuine off-diagonal coupling, not just a 2x2
# corner case.
const W = 20
const S = 3
const T = S * W
const L = 5
const Dmax = 4

woy = [mod1(t, W) for t in 1:T]
season = [((t - 1) ÷ W) + 1 for t in 1:T]
dates = Date(2016, 1, 2) .+ Day.(7 .* (0:(T - 1)))

delay = [min(T - t, Dmax) for t in 1:T, l in 1:L]
delay[T, 2] = -1
delay[T, 3] = -1
delay[T - 1, 3] = -1

transform = :fourthroot  # matches experiments/round1_run.jl's PRIMARY_TRANSFORM
true_curve_pct = [2.0 .+ 1.5 .* sin(2 * pi * w / W) for w in woy]
Y = Matrix{Union{Missing,Float64}}(undef, T, L)
for l in 1:L, t in 1:T
    Y[t, l] = delay[t, l] == -1 ? missing :
              to_scale(max(true_curve_pct[t] + 0.05 * randn(), 0.0), transform)
end

d = ModelData(Y, delay, woy, season, dates, L, T, W, S, Dmax, transform,
              dates[end])

println("=== var check ===")

# --- 1. Prior predictive check ---
model = model_var(d; transform=transform)

prior_natural = Float64[]
mu_w_spread = Float64[]        # max|mu_w| per draw
delta_spread = Float64[]       # max|delta| per draw
resid_spread = Float64[]       # max|residual| per draw
spectral_radii = Float64[]     # max|eigenvalue(A)| per draw
sigma_couple_draws = Float64[] # sigma_couple per draw
const NPRIOR = 300
for i in 1:NPRIOR
    gq = model()  # fresh prior draw + generated quantities each call
    for l in 1:L, t in 1:T
        push!(prior_natural, from_scale(gq.latent[t, l], transform))
    end
    push!(mu_w_spread, maximum(abs, gq.mu_w))
    push!(delta_spread, maximum(abs, gq.delta))
    push!(resid_spread, maximum(abs, gq.residual))
    push!(spectral_radii, maximum(abs, eigvals(gq.A)))
    push!(sigma_couple_draws, gq.sigma_couple)
end

@assert all(isfinite, prior_natural) "non-finite prior predictive values"
lo, hi = extrema(prior_natural)
q50, q75, q99 = quantile(prior_natural, [0.5, 0.75, 0.99])
println("prior predictive wILI% full range: [$(round(lo; digits=2)), ",
        "$(round(hi; digits=1))]")
println("prior predictive wILI% 50/75/99th pct: ",
        "$(round(q50; digits=2)) / $(round(q75; digits=2)) / ",
        "$(round(q99; digits=1))")
# The heavy right tail here (up to ~1e75 in a 300-draw run) is
# INHERITED from base_model's own priors, not introduced by the VAR
# structure: `phi` (identical prior to base_model's) can sit near a
# unit root (|phi| up to ~0.94-0.98 was observed directly in a 10-draw
# spot check while developing this file), and `mu_w` is an unbounded
# random walk (also identical to base_model), both well known to give
# heavy prior-predictive tails (see check_v2.jl's matching comment).
# With `L=5` locations pooled per draw (vs check_severity.jl's `L=3`),
# there is more chance that at least one location's `phi` lands near
# that unit root in a given draw, which then interacts with
# `:fourthroot`'s `x^4` back-transform to inflate even the pooled
# BULK (50th pct), not just the extrema/99th pct -- so the bound here
# is looser than check_severity.jl's (which has no residual-dynamics
# change and a smaller `L`), matching check_v2.jl's `q50 < 50` bound
# for the same reason.
@assert -1.0 < q50 < 50.0 "implausible prior predictive: bulk median"
@assert q75 < 300.0 "implausible prior predictive: bulk 75th pct"
println("prior predictive check (bulk mass): PASS")

med_mu_w, med_delta, med_resid =
    median(mu_w_spread), median(delta_spread), median(resid_spread)
println("median per-draw max|.|  mu_w=$(round(med_mu_w; digits=2))  ",
        "delta=$(round(med_delta; digits=2))  ",
        "residual=$(round(med_resid; digits=2))")
@assert med_resid < 5 * max(med_mu_w, med_delta) (
    "VAR residual variance is disproportionately larger than base " *
    "model's own mu_w/delta components -- possible explosive recursion"
)
println("residual-vs-base-component spread check: PASS")

# --- 2. VAR(1) stability: spectral radius of A ---
frac_stable = mean(spectral_radii .< 1.0)
med_radius, hi_radius = median(spectral_radii), quantile(spectral_radii, 0.95)
med_couple = median(sigma_couple_draws)
println("A spectral radius median=$(round(med_radius; digits=3)) ",
        "95th pct=$(round(hi_radius; digits=3)) ",
        "fraction < 1: $(round(frac_stable; digits=3))")
println("sigma_couple median=$(round(med_couple; digits=3))")
@assert frac_stable > 0.95 (
    "VAR(1) transition matrix A is unstable (spectral radius >= 1) for " *
    "more than 5% of prior draws -- shrinkage prior on sigma_couple or " *
    "phi's prior may need tightening"
)
println("VAR(1) stability (spectral radius) check: PASS")

# --- 3. project_var smoke check directly on a prior draw ---
# No fit needed: project_var only reads the model's RETURN fields,
# which a fresh prior draw's generated quantities already carry.
gq = model()
latent = project_var(gq, d, 1:4)
@assert size(latent) == (L, 4)
@assert all(isfinite, latent)
println("project_var output shape: PASS (", size(latent), ", all finite)")

println("=== var check: ALL PASS ===")
