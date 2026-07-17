# Prior predictive + tiny-fit smoke check for the v1-ar-high round-1
# candidate. Run with:
#     julia --project=. experiments/round1/v1-ar-high/check_v1.jl

include("../../../src/core.jl")
include("../../../src/model.jl")
include("../../../src/inference.jl")
include("model_v1.jl")
include("project_v1.jl")

using Dates
using Random
using Statistics

Random.seed!(20260717)

# --- Synthetic ModelData -------------------------------------------------
# Small enough to fit fast, but T well above the AR order p so the
# ar_p burn-in approximation (see model_v1.jl) is negligible, and with
# both missing cells and non-zero delay to exercise the observation
# loop's full behaviour.

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
        # exercise the missing-data branch of the observation loop.
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
# Two checks, kept separate deliberately:
#
# 1. `residual` is the part this branch actually changed (AR(1) ->
#    AR(p)); it should sit tightly around 0 (it's a mean-zero process
#    by construction), regardless of the seasonal component.
# 2. Full back-transformed wILI% has a heavy right tail under the
#    weakly-informative prior: `mu_w` is a non-centred random walk
#    over W=52 weeks (unchanged from base_model), so occasional prior
#    draws of `sigma_season_pop` give a large cumulative `mu_w` value
#    that `expm1` blows up. This is a pre-existing base_model
#    characteristic (the seasonal component is untouched here), not
#    something the AR(p) change introduces, so the check on it is a
#    central-tendency / bulk-of-mass check, not a hard bound on every
#    sample.

p = 5
n_prior = 60
all_wili = Float64[]
all_resid = Float64[]
for _ in 1:n_prior
    draw = model_v1(d; transform=transform, p=p)()
    append!(all_wili, from_scale.(vec(draw.latent), transform))
    append!(all_resid, vec(draw.residual))
end

resid_med = median(all_resid)
println("Prior predictive residual (AR(p) component): median=",
        round(resid_med; digits=3))
@assert abs(resid_med) < 1.0 "AR(p) residual median far from 0: $resid_med"

med = median(all_wili)
frac_plausible = count(x -> -1 <= x <= 15, all_wili) / length(all_wili)
lo, hi = quantile(all_wili, [0.25, 0.75])
println("Prior predictive wILI%: median=$(round(med; digits=2)), ",
        "IQR=($(round(lo; digits=2)), $(round(hi; digits=2))), ",
        "fraction in [-1, 15]=$(round(frac_plausible; digits=2))")
@assert -1 <= med <= 15 "prior predictive median wILI% implausible: $med"
@assert frac_plausible > 0.5 "prior predictive bulk not in plausible range"
println("Prior predictive check passed.")

# --- Tiny fit smoke test ---------------------------------------------------

fit_model = model_v1(d; transform=transform, p=p)
chain = fit_mcmc(fit_model; nsamples=30, nchains=1,
                  adtype=Turing.AutoMooncake())
println("NUTS smoke test completed: chain size = ", size(chain))

draws = generated_draws(fit_model, chain)
@assert length(draws) == 30
@assert size(draws[1].phi) == (L, p)
latent_fc = project_v1(draws[1], d, 1:4)
@assert size(latent_fc) == (L, 4)
println("project_v1 output size: ", size(latent_fc), " (expect (", L, ", 4))")

println("check_v1.jl: all checks passed")
