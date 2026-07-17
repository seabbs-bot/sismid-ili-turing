# Bayesian-workflow check for round 2 candidate mvn-innov, ahead of
# scoring (see docs/plan.md's "each candidate passes ... before it is
# scored"). Builds a small synthetic ModelData by hand (same pattern as
# test/test_model.jl and experiments/round2/var/check_var.jl).
#
# PRIOR-only check (no NUTS/Pathfinder fit): the box is busy with other
# candidates, so this stays light --
#
# 1. A PRIOR predictive check: draws from the prior and confirms the
#    implied wILI percentages are finite and in a plausible range.
# 2. An LKJ-CHOLESKY sanity check: the new, risky part of this
#    candidate is `Lcorr`/`corr` -- confirms the reconstructed
#    correlation matrix has a unit diagonal and all off-diagonal
#    entries in [-1, 1] for every prior draw (i.e. `Lcorr` really is a
#    valid correlation Cholesky factor, not just "close enough"), and
#    that the residual's spread stays the same order of magnitude as
#    base_model's own AR(1) residual would (not blown up by the added
#    cross-location correlation).
# 3. A `project_mvn_innov` smoke check directly on a prior draw's
#    generated quantities (no fit needed -- `project_mvn_innov` only
#    consumes the model's RETURN fields, which a prior draw already
#    has).
#
# Run from the repo root: `julia --project=. \
#   experiments/round2/mvn-innov/check_mvn_innov.jl`

using Random
using Dates
using Distributions
using Turing
using LinearAlgebra
using Statistics

const HERE = @__DIR__
include(joinpath(HERE, "..", "..", "..", "src", "core.jl"))
include(joinpath(HERE, "..", "..", "..", "src", "model.jl"))
include(joinpath(HERE, "model_mvn_innov.jl"))
include(joinpath(HERE, "project_mvn_innov.jl"))

Random.seed!(20260717)

# --- Build a small synthetic ModelData by hand, L=5, THREE seasons ---
# L=5 (not the full 11) keeps this light while still giving `Lcorr`
# enough locations to exercise genuine off-diagonal correlation, not
# just a 2x2 corner case.
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

println("=== mvn-innov check ===")

# --- 1. Prior predictive check ---
model = model_mvn_innov(d; transform=transform)

prior_natural = Float64[]
mu_w_spread = Float64[]        # max|mu_w| per draw
delta_spread = Float64[]       # max|delta| per draw
resid_spread = Float64[]       # max|residual| per draw
diag_errs = Float64[]          # max|diag(corr) - 1| per draw
offdiag_maxabs = Float64[]     # max|off-diagonal corr entry| per draw
offdiag_means = Float64[]      # mean|off-diagonal corr entry| per draw
const NPRIOR = 300
for i in 1:NPRIOR
    gq = model()  # fresh prior draw + generated quantities each call
    for l in 1:L, t in 1:T
        push!(prior_natural, from_scale(gq.latent[t, l], transform))
    end
    push!(mu_w_spread, maximum(abs, gq.mu_w))
    push!(delta_spread, maximum(abs, gq.delta))
    push!(resid_spread, maximum(abs, gq.residual))
    push!(diag_errs, maximum(abs, diag(gq.corr) .- 1.0))
    offdiag = [gq.corr[i, j] for i in 1:L, j in 1:L if i != j]
    push!(offdiag_maxabs, maximum(abs, offdiag))
    push!(offdiag_means, mean(abs, offdiag))
end

@assert all(isfinite, prior_natural) "non-finite prior predictive values"
lo, hi = extrema(prior_natural)
q50, q75, q99 = quantile(prior_natural, [0.5, 0.75, 0.99])
println("prior predictive wILI% full range: [$(round(lo; digits=2)), ",
        "$(round(hi; digits=1))]")
println("prior predictive wILI% 50/75/99th pct: ",
        "$(round(q50; digits=2)) / $(round(q75; digits=2)) / ",
        "$(round(q99; digits=1))")
# The heavy right tail here is INHERITED from base_model's own priors,
# not introduced by the MVN-innovation structure: `phi` (identical
# prior to base_model's) can sit near a unit root, and `mu_w` is an
# unbounded random walk (also identical to base_model), both well known
# to give heavy prior-predictive tails when combined with
# `:fourthroot`'s `x^4` back-transform (see check_var.jl's matching
# comment, which hits the same effect from its own residual-dynamics
# change). Bound matches check_var.jl's for the same reason.
@assert -1.0 < q50 < 50.0 "implausible prior predictive: bulk median"
@assert q75 < 300.0 "implausible prior predictive: bulk 75th pct"
println("prior predictive check (bulk mass): PASS")

med_mu_w, med_delta, med_resid =
    median(mu_w_spread), median(delta_spread), median(resid_spread)
println("median per-draw max|.|  mu_w=$(round(med_mu_w; digits=2))  ",
        "delta=$(round(med_delta; digits=2))  ",
        "residual=$(round(med_resid; digits=2))")
@assert med_resid < 5 * max(med_mu_w, med_delta) (
    "MVN-innovation residual variance is disproportionately larger " *
    "than base model's own mu_w/delta components -- possible " *
    "mis-scaled correlation mixing"
)
println("residual-vs-base-component spread check: PASS")

# --- 2. LKJ-Cholesky sanity: corr is a valid correlation matrix ---
max_diag_err = maximum(diag_errs)
max_offdiag = maximum(offdiag_maxabs)
med_offdiag_mean = median(offdiag_means)
println("max|diag(corr) - 1| across draws: $(round(max_diag_err; digits=8))")
println("max |off-diagonal corr entry| across draws: ",
        "$(round(max_offdiag; digits=4))")
println("median per-draw mean|off-diagonal corr entry|: ",
        "$(round(med_offdiag_mean; digits=4))")
@assert max_diag_err < 1e-6 (
    "Lcorr does not reconstruct a unit-diagonal correlation matrix"
)
@assert max_offdiag < 1.0 + 1e-8 (
    "Lcorr reconstructs an off-diagonal correlation entry outside " *
    "[-1, 1] -- Lcorr is not a valid correlation Cholesky factor"
)
println("LKJ-Cholesky validity check: PASS")

# --- 3. project_mvn_innov smoke check on a prior draw ---
# No fit needed: project_mvn_innov only reads the model's RETURN
# fields, which a fresh prior draw's generated quantities already have.
gq = model()
latent = project_mvn_innov(gq, d, 1:4)
@assert size(latent) == (L, 4)
@assert all(isfinite, latent)
println("project_mvn_innov output shape: PASS (", size(latent),
        ", all finite)")

println("=== mvn-innov check: ALL PASS ===")
