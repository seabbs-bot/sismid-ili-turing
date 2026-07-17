# Round 2 candidate loc-obs: LOCATION-VARYING observation noise scale.
#
# MOTIVATION: `docs/eda/06-regional-heterogeneity.md` finds off-season
# baseline anti-correlates with differenced-series volatility across
# the 11 locations (r=-0.63): smaller/lower-baseline locations (Region
# 7, Region 8, Region 10) are proportionally noisier week to week, not
# scaled-down versions of the bigger ones, and explicitly recommends a
# location-varying observation/process noise scale rather than one
# shared value fitted mostly to the large locations (US National,
# Region 6). `docs/eda/07-region9-deepdive.md` adds Region 9
# specifically: its Taylor's-law/transform-power fit is the most
# Poisson-like of all 11 locations (lambda=0.91 vs a pooled 1.77),
# i.e. its variance-to-mean relationship does not match the shared
# fourthroot transform as well as most other locations do, which shows
# up as extra unexplained scatter around `latent` for exactly the
# locations the shared `sigma_obs` under-serves.
#
# WHAT CHANGED relative to `base_model` (src/model.jl): the single
# scalar `sigma_obs ~ truncated(Normal(0, 1); lower=0)` is replaced by
# a length-L vector `sigma_obs[l]`, partially pooled across locations
# via the SAME non-centred log-normal hierarchy `base_model` already
# uses for the per-location AR innovation sd `sigma_ar` (see that
# file's "Post-seasonal residual" block) -- reusing an already-reviewed
# pooling pattern rather than inventing a new one for this candidate.
# Every other block (seasonality, AR(1)/difference residual, backfill)
# is copied verbatim from `base_model`. This is the cleanest general
# form of the EDA finding: it lets the DATA decide how much noisier
# each location is (Region 9, Region 7, Region 8, Region 10 included)
# rather than hard-coding a Region-9-specific fix, and it changes
# nothing about the latent dynamics or the forecast projection (see
# project_loc_obs.jl).
#
# This file assumes `src/core.jl` (for `ModelData`) and `src/model.jl`
# (for `model_dims`, `ar_or_diff`, `backfill_profile`,
# `observation_index`) are already `include`d into scope, exactly as
# `base_model` itself assumes -- see check_loc_obs.jl.

using Turing
using Distributions
using Statistics

"""
    model_loc_obs(d::ModelData; transform=:log, difference=false,
                  obsdata=observation_index(d))

Round 2 candidate `loc-obs`: identical to `base_model` (partially-pooled
seasonality, per-location AR(1)/difference residual, non-monotonic
backfill) except the observation noise scale is LOCATION-VARYING,
partially pooled, rather than one value shared across all 11 locations.

# Location-varying observation noise

`sigma_obs` is now a length-`L` vector, built non-centred on the log
scale exactly as `base_model` already builds the per-location AR
innovation sd `sigma_ar`:

- `mu_log_sigma_obs ~ Normal(log(0.5), 0.5)`: the population log-mean
  observation noise scale. `log(0.5)` centres the population median
  noise scale a little below `base_model`'s shared
  `truncated(Normal(0, 1); lower=0)` prior mean (~0.8), leaving room for
  the partially-pooled per-location spread below to push individual
  locations (e.g. Region 9) higher without the population centre itself
  needing to move.
- `tau_log_sigma_obs ~ truncated(Normal(0, 0.3); lower=0)`: how much
  locations are allowed to differ from that population value on the
  log scale. Deliberately narrower than `sigma_ar`'s equivalent
  `tau_log_sigma_ar` prior (`truncated(Normal(0, 0.5); lower=0)`):
  observation noise is a secondary, "how noisy is the reporting"
  effect layered on top of the AR residual's own location variation,
  so it should be able to differentiate locations (per the EDA) without
  swamping the residual as the dominant source of per-location spread.
- `z_sigma_obs ~ filldist(Normal(0, 1), L)`, `sigma_obs = exp.(
  mu_log_sigma_obs .+ tau_log_sigma_obs .* z_sigma_obs)`: the usual
  non-centred parameterisation, guaranteeing `sigma_obs[l] > 0` for
  every location and every draw with no truncation boundary for
  Pathfinder/HMC to fight.

The observation model becomes `Y[t, l] ~ Normal(latent[t, l] +
r[delay[t, l], l], sigma_obs[l])`, still evaluated as a single
vectorised `arraydist` observe over all non-missing cells (see
`observation_index`): the per-cell noise scale is gathered as
`sigma_obs[loc_idx]` where `loc_idx` is each observed cell's location
column, read straight off `obsdata.obs_idx` (a `CartesianIndex{2}`
into the `T x L` `Y`/`latent` matrices, second index is location) with
no separate bookkeeping needed.

`transform` and `difference` behave exactly as in `base_model` (see
that docstring); `Dmax`/the backfill block, seasonality block, and
AR(1)/difference residual block are all copied unchanged.

Returns the same `NamedTuple` shape as `base_model` (`latent`,
`seasonal`, `residual`, `mu0`, `mu_w`, `delta`, `season_eff`, `phi`,
`sigma_ar`, `r`, `r_pop`, `sigma_obs`, `transform`) with ONE field
reshaped: `sigma_obs` is now length-`L`, not a scalar. Every field a
forecaster needs to project `latent` forward (`mu0`, `mu_w`, `delta`,
`season_eff`, `phi`, `sigma_ar`, `residual`) is unchanged in shape, so
`base_project` (src/forecast.jl) -- which never reads `sigma_obs`, the
observation noise plays no part in the latent forecast -- consumes
draws from this model exactly as it does `base_model`'s; see
project_loc_obs.jl.
"""
@model function model_loc_obs(d::ModelData; transform::Symbol=:log,
                               difference::Bool=false,
                               obsdata=observation_index(d))
    T, L, W, S, Dmax = model_dims(d)
    obs_idx, r_idx, yobs = obsdata.obs_idx, obsdata.r_idx, obsdata.yobs

    # --- Seasonality: partially-pooled week-of-season random effect ---
    # (identical to base_model)
    mu0 ~ Normal(0, 2)

    sigma_season_pop ~ truncated(Normal(0, 1); lower=0)
    mu_w_raw ~ filldist(Normal(0, 1), W)
    mu_w_uncentred = cumsum(mu_w_raw) .* sigma_season_pop
    mu_w = mu_w_uncentred .- mean(mu_w_uncentred)

    sigma_season_loc ~ truncated(Normal(0, 1); lower=0)
    delta_raw ~ filldist(Normal(0, 1), W, L)
    delta = delta_raw .* sigma_season_loc

    sigma_season_time ~ truncated(Normal(0, 1); lower=0)
    season_eff_raw ~ filldist(Normal(0, 1), S)
    season_eff = season_eff_raw .* sigma_season_time

    seasonal = mu0 .+ mu_w[d.woy] .+ delta[d.woy, :] .+ season_eff[d.season]

    # --- Post-seasonal residual: partially-pooled AR(1) or difference ---
    # (identical to base_model)
    phi_pop_mean ~ Normal(0, 1)
    phi_pop_sd ~ truncated(Normal(0, 0.5); lower=0)
    phi_raw ~ filldist(Normal(0, 1), L)
    phi = tanh.(phi_pop_mean .+ phi_pop_sd .* phi_raw)

    mu_log_sigma_ar ~ Normal(log(0.2), 1)
    tau_log_sigma_ar ~ truncated(Normal(0, 0.5); lower=0)
    z_sigma_ar ~ filldist(Normal(0, 1), L)
    sigma_ar = exp.(mu_log_sigma_ar .+ tau_log_sigma_ar .* z_sigma_ar)

    eps_raw ~ filldist(Normal(0, 1), T, L)
    residual = reduce(hcat, [
        ar_or_diff(view(eps_raw, :, l), sigma_ar[l], phi[l], difference)
        for l in 1:L
    ])

    latent = seasonal .+ residual

    # --- Backfill: non-monotonic delay-indexed revision (identical) ---
    r_pop_anchor ~ Normal(0, 0.05)
    sigma_r_pop ~ truncated(Normal(0, 0.3); lower=0)
    r_steps_raw ~ filldist(Normal(0, 1), Dmax)
    r_pop = backfill_profile(r_pop_anchor, r_steps_raw .* sigma_r_pop)

    sigma_r_loc ~ truncated(Normal(0, 0.3); lower=0)
    r_loc_raw ~ filldist(Normal(0, 1), Dmax + 1, L)
    r = r_pop .+ r_loc_raw .* sigma_r_loc

    # --- Observation noise: LOCATION-VARYING, partially pooled ---
    # (this is the change this candidate makes -- base_model has one
    # shared `sigma_obs ~ truncated(Normal(0, 1); lower=0)` here)
    mu_log_sigma_obs ~ Normal(log(0.5), 0.5)
    tau_log_sigma_obs ~ truncated(Normal(0, 0.3); lower=0)
    z_sigma_obs ~ filldist(Normal(0, 1), L)
    sigma_obs = exp.(mu_log_sigma_obs .+ tau_log_sigma_obs .* z_sigma_obs)

    loc_idx = getindex.(obs_idx, 2)
    mu_obs = latent[obs_idx] .+ r[r_idx]
    yobs ~ arraydist(Normal.(mu_obs, sigma_obs[loc_idx]))

    return (;
        latent, seasonal, residual, mu0, mu_w, delta, season_eff,
        phi, sigma_ar, r, r_pop, sigma_obs, transform,
    )
end
