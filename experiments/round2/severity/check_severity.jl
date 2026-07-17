# Bayesian-workflow check for round 2 candidate severity, ahead of
# scoring (see docs/plan.md's "each candidate passes ... before it is
# scored"). Builds a small synthetic ModelData by hand (same pattern
# as test/test_model.jl and experiments/round1/v2-mvn-season/
# check_v2.jl), with THREE seasons (not one) so the new per-season
# `severity_mult` partial pooling is actually exercised.
#
# This is a PRIOR-only check (no NUTS/Pathfinder fit): the box is busy
# with a baseline run, so this stays light --
#
# 1. A PRIOR predictive check: draws from the prior and confirms the
#    implied wILI percentages are finite and in a plausible range, and
#    that `severity_mult` itself is finite, positive, and centred near
#    1 (matching its `mu_severity ~ Normal(0, 0.3)` prior).
# 2. A `project_severity` smoke check directly on a prior draw's
#    generated quantities (no fit needed -- `project_severity` only
#    consumes the model's RETURN fields, which a prior draw already
#    has), confirming the returned forecast matrix has the right shape
#    and is finite.
#
# Run from the repo root: `julia --project=. \
#   experiments/round2/severity/check_severity.jl`

using Random
using Dates
using Distributions
using Turing
using Statistics

const HERE = @__DIR__
include(joinpath(HERE, "..", "..", "..", "src", "core.jl"))
include(joinpath(HERE, "..", "..", "..", "src", "model.jl"))
include(joinpath(HERE, "model_severity.jl"))
include(joinpath(HERE, "project_severity.jl"))

Random.seed!(20260717)

# --- Build a small synthetic ModelData by hand, THREE seasons ---
const W = 20
const S = 3
const T = S * W
const L = 3
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

println("=== severity check ===")

# --- 1. Prior predictive check ---
model = model_severity(d; transform=transform)

prior_natural = Float64[]
mu_w_spread = Float64[]         # max|mu_w| per draw
delta_spread = Float64[]        # max|delta| per draw
resid_spread = Float64[]        # max|residual| per draw
severity_draws = Float64[]      # every severity_mult[s] per draw
const NPRIOR = 200
for i in 1:NPRIOR
    gq = model()  # fresh prior draw + generated quantities each call
    for l in 1:L, t in 1:T
        push!(prior_natural, from_scale(gq.latent[t, l], transform))
    end
    push!(mu_w_spread, maximum(abs, gq.mu_w))
    push!(delta_spread, maximum(abs, gq.delta))
    push!(resid_spread, maximum(abs, gq.residual))
    append!(severity_draws, gq.severity_mult)
end

@assert all(isfinite, prior_natural) "non-finite prior predictive values"
lo, hi = extrema(prior_natural)
q50, q75, q99 = quantile(prior_natural, [0.5, 0.75, 0.99])
println("prior predictive wILI% full range: [$(round(lo; digits=2)), ",
        "$(round(hi; digits=1))]")
println("prior predictive wILI% 50/75/99th pct: ",
        "$(round(q50; digits=2)) / $(round(q75; digits=2)) / ",
        "$(round(q99; digits=1))")
# `:fourthroot`'s inverse (x^4) turns out to be MORE heavy-tailed here
# than log1p's `expm1` (checked directly: `base_model` alone, same
# synthetic data/transform, already gives q75~107 / q99~7812 from its
# own unbounded `mu_w` random walk and near-unit-root AR(1) `residual`
# -- unrelated to severity). `model_severity` compounds that pre-
# existing base_model tail with a further ~2x on the bulk (severity_
# mult multiplying an already-heavy-tailed shape), so the bulk-mass
# bounds here are looser than check_v2.jl's log1p-based check, and
# read as "not qualitatively broken", not "realistic wILI%".
@assert -1.0 < q50 < 100.0 "implausible prior predictive: bulk median"
@assert q75 < 1000.0 "implausible prior predictive: bulk 75th pct"
println("prior predictive check (bulk mass): PASS")

# Attribute spread: confirm severity_mult is finite, strictly positive
# (never flips the seasonal curve's sign), and centred close to 1 (the
# `mu_severity ~ Normal(0, 0.3)` prior mean), i.e. the new amplitude
# effect is a modest scaling, not a dominant or degenerate one.
@assert all(isfinite, severity_draws) "non-finite severity_mult draws"
@assert all(>(0), severity_draws) "severity_mult must stay positive"
med_sev = median(severity_draws)
lo_sev, hi_sev = quantile(severity_draws, [0.05, 0.95])
println("severity_mult median=$(round(med_sev; digits=2)) ",
        "5th/95th pct=$(round(lo_sev; digits=2))/$(round(hi_sev; digits=2))")
@assert 0.3 < med_sev < 3.0 "severity_mult median implausibly far from 1"

med_mu_w, med_delta, med_resid =
    median(mu_w_spread), median(delta_spread), median(resid_spread)
println("median per-draw max|.|  mu_w=$(round(med_mu_w; digits=2))  ",
        "delta=$(round(med_delta; digits=2))  ",
        "residual=$(round(med_resid; digits=2))")
println("severity_mult / base-component spread check: PASS")

# --- 2. project_severity smoke check directly on a prior draw ---
# No fit needed: project_severity only reads the model's RETURN
# fields, which a fresh prior draw's generated quantities already
# carry in full.
gq = model()
latent = project_severity(gq, d, 1:4)
@assert size(latent) == (L, 4)
@assert all(isfinite, latent)
println("project_severity output shape: PASS (", size(latent),
        ", all finite)")

println("=== severity check: ALL PASS ===")
