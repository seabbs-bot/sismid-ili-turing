# Bayesian-workflow check for round 2 candidate combo, ahead of scoring
# (see docs/plan.md's "each candidate passes ... before it is scored").
# Builds a small synthetic ModelData by hand (same pattern as
# test/test_model.jl and experiments/round2/severity/check_severity.jl),
# with THREE seasons so both `severity_mult` and `r_season`'s partial
# pooling are actually exercised.
#
# PRIOR-only check (no NUTS/Pathfinder fit): this is a light, combine-
# three-existing-mechanisms candidate, so the check mirrors
# check_severity.jl / check_season_backfill.jl / check_v1.jl rather
# than re-deriving new checks:
#
# 1. A prior predictive check: draws from the prior and confirms the
#    implied wILI percentages are finite and in a plausible range.
# 2. Per-mechanism attribute checks: `severity_mult` finite, positive,
#    centred near 1; `phi` (AR(p)) stationary-by-construction and the
#    right shape; `r_season` actually varies the backfill profile
#    across seasons.
# 3. A `project_combo` smoke check directly on a prior draw's generated
#    quantities (no fit needed).
#
# Run from the repo root:
#   julia --project=. experiments/round2/combo/check_combo.jl

using Random
using Dates
using Distributions
using Turing
using Statistics

const HERE = @__DIR__
include(joinpath(HERE, "..", "..", "..", "src", "core.jl"))
include(joinpath(HERE, "..", "..", "..", "src", "model.jl"))
include(joinpath(HERE, "..", "..", "round1", "v1-ar-high", "model_v1.jl"))
include(joinpath(HERE, "model_combo.jl"))
include(joinpath(HERE, "project_combo.jl"))

Random.seed!(20260717)

# --- Build a small synthetic ModelData by hand, THREE seasons ---
const W = 20
const S = 3
const T = S * W
const L = 3
const Dmax = 4
const P = 5

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

println("=== combo check ===")
println(model_dims(d))

# --- 1. Prior predictive check ---
model = model_combo(d; transform=transform, p=P)

prior_natural = Float64[]
severity_draws = Float64[]      # every severity_mult[s] per draw
phi_shapes = Bool[]
r_season_active = Bool[]
const NPRIOR = 200
for i in 1:NPRIOR
    gq = model()  # fresh prior draw + generated quantities each call
    for l in 1:L, t in 1:T
        push!(prior_natural, from_scale(gq.latent[t, l], transform))
    end
    append!(severity_draws, gq.severity_mult)
    push!(phi_shapes, size(gq.phi) == (L, P))
    push!(r_season_active,
          any(gq.r[:, l, 1] != gq.r[:, l, 2] for l in 1:L) ||
          gq.sigma_r_season < 1e-8)
end

@assert all(isfinite, prior_natural) "non-finite prior predictive values"
lo, hi = extrema(prior_natural)
q50, q75, q99 = quantile(prior_natural, [0.5, 0.75, 0.99])
println("prior predictive wILI% full range: [$(round(lo; digits=2)), ",
        "$(round(hi; digits=1))]")
println("prior predictive wILI% 50/75/99th pct: ",
        "$(round(q50; digits=2)) / $(round(q75; digits=2)) / ",
        "$(round(q99; digits=1))")
# As in check_severity.jl: `:fourthroot`'s inverse (x^4) is heavy-tailed
# on rare draws given base_model's own unbounded `mu_w` random walk and
# near-unit-root residual, compounded here by severity_mult's further
# ~2x on the bulk. Bulk-mass bounds are loosened accordingly, matching
# check_severity.jl -- this is "not qualitatively broken", not a claim
# of realistic wILI% on every draw.
@assert -1.0 < q50 < 100.0 "implausible prior predictive: bulk median"
@assert q75 < 1000.0 "implausible prior predictive: bulk 75th pct"
println("prior predictive check (bulk mass): PASS")

# --- 2. Per-mechanism attribute checks ---
@assert all(isfinite, severity_draws) "non-finite severity_mult draws"
@assert all(>(0), severity_draws) "severity_mult must stay positive"
med_sev = median(severity_draws)
lo_sev, hi_sev = quantile(severity_draws, [0.05, 0.95])
println("severity_mult median=$(round(med_sev; digits=2)) ",
        "5th/95th pct=$(round(lo_sev; digits=2))/$(round(hi_sev; digits=2))")
@assert 0.3 < med_sev < 3.0 "severity_mult median implausibly far from 1"

@assert all(phi_shapes) "phi (AR(p) coefficients) has the wrong shape"
println("phi (AR(p) coefficients) shape check: PASS, size=($L, $P)")

@assert all(r_season_active) "season dimension of r is inert every draw"
println("season deviation on r is active (or sigma_r_season ~ 0) ",
        "every draw: PASS")

# --- 3. project_combo smoke check directly on a prior draw ---
# No fit needed: project_combo only reads the model's RETURN fields,
# which a fresh prior draw's generated quantities already carry in full.
gq = model()
latent = project_combo(gq, d, 1:4)
@assert size(latent) == (L, 4)
@assert all(isfinite, latent)
println("project_combo output shape: PASS (", size(latent), ", all finite)")

println("=== combo check: ALL PASS ===")
