# Prior predictive check for the round-2 candidate loc-obs. PRIOR ONLY
# -- no MCMC/Pathfinder fit here (light check as requested; see
# docs/lessons.md #4 on not over-parallelising fits). Run with:
#     julia --project=. experiments/round2/loc-obs/check_loc_obs.jl

include("../../../src/core.jl")
include("../../../src/model.jl")
include("../../../src/forecast.jl")
include("model_loc_obs.jl")
include("project_loc_obs.jl")

using Dates
using Random
using Statistics

Random.seed!(20260717)

# --- Synthetic ModelData -------------------------------------------------
# Small enough to fit fast, with both missing cells and non-zero delay
# to exercise the observation model's full behaviour, and L > 1 so the
# per-location sigma_obs vector this candidate introduces is genuinely
# exercised (not just a length-1 degenerate case).

T, L, W, S, Dmax = 60, 4, 52, 2, 6
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
# `latent` (seasonal + residual) does not depend on `d.Y`, so calling
# the model directly draws one full forward (prior) sample each time.
# `sigma_obs` DOES only enter the likelihood, but it is still a plain
# sampled/derived quantity in the model's return value, so it is
# available on every call with no fit needed.
#
# Three checks, kept separate deliberately:
#
# 1. `sigma_obs` (the length-L vector this branch introduces in place
#    of base_model's scalar) should be strictly positive for every
#    location and every draw, and should show genuine across-location
#    spread under the prior -- not collapse back to a single shared
#    value -- confirming the partial-pooling hierarchy actually lets
#    locations differ rather than being pooled away to nothing.
# 2. `residual` (untouched by this candidate) should sit tightly around
#    0, exactly as in base_model/ar-loc's checks.
# 3. Full back-transformed wILI% has a heavy right tail under the
#    weakly-informative prior, exactly as in base_model's check (the
#    seasonal/AR blocks are untouched here) -- a bulk check, not a hard
#    bound on every sample.

n_prior = 60
all_wili = Float64[]
all_resid = Float64[]
sigma_obs_draws = Matrix{Float64}(undef, n_prior, L)
for i in 1:n_prior
    draw = model_loc_obs(d; transform=transform)()
    append!(all_wili, from_scale.(vec(draw.latent), transform))
    append!(all_resid, vec(draw.residual))
    n_sigma_obs = length(draw.sigma_obs)
    @assert n_sigma_obs == L "sigma_obs length: $n_sigma_obs"
    sigma_obs_draws[i, :] = draw.sigma_obs
end

@assert all(sigma_obs_draws .> 0) "sigma_obs non-positive somewhere"
per_loc_median = [median(sigma_obs_draws[:, l]) for l in 1:L]
println("Prior predictive sigma_obs per-location median: ",
        round.(per_loc_median; digits=3))
# Across-draw, within-location sd of log(sigma_obs) should be > 0 for
# every location (each location's own prior is not a point mass), and
# the per-location MEDIANS should not all collapse to (near) the same
# value -- confirming genuine partial pooling, not pooling-to-a-point.
within_loc_sd = std(log.(sigma_obs_draws); dims=1)
@assert all(within_loc_sd .> 0.05) "sigma_obs prior collapsed within a location"
spread_across_loc = maximum(per_loc_median) - minimum(per_loc_median)
println("Prior predictive sigma_obs across-location median spread: ",
        round(spread_across_loc; digits=3))
@assert spread_across_loc > 0 "sigma_obs medians identical across locations"

resid_med = median(all_resid)
println("Prior predictive residual (AR(1)/diff component): median=",
        round(resid_med; digits=3))
@assert abs(resid_med) < 1.0 "AR residual median far from 0: $resid_med"

med = median(all_wili)
frac_plausible = count(x -> -1 <= x <= 15, all_wili) / length(all_wili)
lo, hi = quantile(all_wili, [0.25, 0.75])
println("Prior predictive wILI%: median=$(round(med; digits=2)), ",
        "IQR=($(round(lo; digits=2)), $(round(hi; digits=2))), ",
        "fraction in [-1, 15]=$(round(frac_plausible; digits=2))")
@assert -1 <= med <= 15 "prior predictive median wILI% implausible: $med"
@assert frac_plausible > 0.5 "prior predictive bulk not in plausible range"

# --- project_loc_obs shape check (no fit; uses one prior draw) -----------
# Confirms base_project (aliased as project_loc_obs) consumes this
# model's draws with no changes needed, per project_loc_obs.jl's header.

draw = model_loc_obs(d; transform=transform)()
latent_fc = project_loc_obs(draw, d, 1:4)
fc_size = size(latent_fc)
@assert fc_size == (L, 4) "project_loc_obs output size: $fc_size"
println("project_loc_obs output size: ", fc_size, " (expect (", L, ", 4))")

println("check_loc_obs.jl: prior predictive checks passed (no fit run)")
