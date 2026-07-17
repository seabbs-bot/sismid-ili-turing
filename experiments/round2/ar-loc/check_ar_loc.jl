# Prior predictive check for the round-2 candidate ar-loc. PRIOR ONLY
# — no MCMC/Pathfinder fit here (box busy; see docs/lessons.md #4 on
# not over-parallelising fits). Run with:
#     julia --project=. experiments/round2/ar-loc/check_ar_loc.jl

include("../../../src/core.jl")
include("../../../src/model.jl")
include("model_ar_loc.jl")
include("project_ar_loc.jl")

using Dates
using Random
using Statistics

Random.seed!(20260717)

# --- Synthetic ModelData -------------------------------------------------
# Small enough to fit fast, but T well above Pmax so the ar_p burn-in
# approximation (see model_ar_loc.jl) is negligible, and with both
# missing cells and non-zero delay to exercise the observation model's
# full behaviour.

T, L, W, S, Dmax = 60, 3, 52, 2, 6
woy = [mod1(t, W) for t in 1:T]
season = [t <= T ÷ 2 ? 1 : 2 for t in 1:T]
dates = [Date(2015, 1, 3) + Day(7 * (t - 1)) for t in 1:T]

transform = :log1p
Y = Matrix{Union{Missing,Float64}}(undef, T, L)
delay = fill(-1, T, L)
for l in 1:L, t in 1:T
    wili_pct = 1.5 + 1.0 * sin(2pi * woy[t] / W) + 0.2 * randn()
    wili_pct = max(wili_pct, 0.05)
    if t > T - 3 && l == 2
        # A couple of recent, still-partial cells for one location, to
        # exercise the missing-data branch of the observation model.
        Y[t, l] = missing
    else
        Y[t, l] = to_scale(wili_pct, transform)
        delay[t, l] = min(rand(0:3), Dmax)
    end
end

d = ModelData(Y, delay, woy, season, dates, L, T, W, S, Dmax,
              transform, dates[end])

# --- Prior predictive check -----------------------------------------------
# latent = seasonal + residual is a deterministic function of the
# sampled parameters only (it does not depend on d.Y), so calling the
# model directly draws one full forward (prior) sample each time.
#
# Three checks, kept separate deliberately:
#
# 1. `rho` (the per-location PACF decay rate this branch introduces)
#    should land in (0, 1) for every location and every draw — a basic
#    sanity check on the invlogit transform, not a claim about where
#    in (0, 1) it concentrates (that is a posterior question).
# 2. `residual` (the part this branch actually changed relative to
#    v1-ar-high: a SHARED decay -> a per-location decay) should sit
#    tightly around 0 (it's a mean-zero process by construction),
#    regardless of the seasonal component.
# 3. Full back-transformed wILI% has a heavy right tail under the
#    weakly-informative prior, exactly as in v1-ar-high's check (the
#    seasonal component is untouched here) — a central-tendency / bulk
#    check, not a hard bound on every sample.

Pmax = 10
n_prior = 60
all_wili = Float64[]
all_resid = Float64[]
all_rho = Float64[]
for _ in 1:n_prior
    draw = model_ar_loc(d; transform=transform, Pmax=Pmax)()
    append!(all_wili, from_scale.(vec(draw.latent), transform))
    append!(all_resid, vec(draw.residual))
    append!(all_rho, draw.rho)
    @assert size(draw.phi) == (L, Pmax) "phi shape: $(size(draw.phi))"
    @assert length(draw.rho) == L "rho length: $(length(draw.rho))"
end

@assert all(0 .< all_rho .< 1) "rho outside (0, 1) somewhere"
rho_med = median(all_rho)
println("Prior predictive rho (per-location decay rate): median=",
        round(rho_med; digits=3),
        ", range=(", round(minimum(all_rho); digits=3), ", ",
        round(maximum(all_rho); digits=3), ")")

resid_med = median(all_resid)
println("Prior predictive residual (AR(Pmax) component): median=",
        round(resid_med; digits=3))
@assert abs(resid_med) < 1.0 "AR(Pmax) residual median far from 0: $resid_med"

med = median(all_wili)
frac_plausible = count(x -> -1 <= x <= 15, all_wili) / length(all_wili)
lo, hi = quantile(all_wili, [0.25, 0.75])
println("Prior predictive wILI%: median=$(round(med; digits=2)), ",
        "IQR=($(round(lo; digits=2)), $(round(hi; digits=2))), ",
        "fraction in [-1, 15]=$(round(frac_plausible; digits=2))")
@assert -1 <= med <= 15 "prior predictive median wILI% implausible: $med"
@assert frac_plausible > 0.5 "prior predictive bulk not in plausible range"

# --- project_ar_loc shape check (no fit; uses one prior draw) ------------
# Confirms the projection consumes model_ar_loc's return fields with no
# separate order argument, reading Pmax from size(phi, 2) as documented.

draw = model_ar_loc(d; transform=transform, Pmax=Pmax)()
latent_fc = project_ar_loc(draw, d, 1:4)
@assert size(latent_fc) == (L, 4) "project_ar_loc output size: $(size(latent_fc))"
println("project_ar_loc output size: ", size(latent_fc), " (expect (", L, ", 4))")

println("check_ar_loc.jl: prior predictive checks passed (no fit run)")
