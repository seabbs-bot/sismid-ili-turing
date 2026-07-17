# Time-varying-AR variant of the base joint model (Round 1, tree search
# candidate v4-tv-ar). Builds on src/model.jl's `base_model`:
# seasonality and backfill blocks are copied unchanged; only the
# residual block changes so the per-location AR(1) coefficient
# `phi[t, l]` evolves over time instead of being fixed per location.
#
# Assumes `src/core.jl` and `src/model.jl` are already `include`d (for
# `ModelData`, `model_dims`, and `backfill_profile`), matching this
# experiment's `check_v4.jl`.

using Turing
using Distributions
using LinearAlgebra
using Statistics

"""
    tv_ar_path(eps, sigma, phi_path)

Build one location's post-seasonal residual path (length `T`) from
standard-normal innovations `eps`, an innovation sd `sigma`, and a
TIME-VARYING AR coefficient path `phi_path` (length `T`, one value per
time step, each in (-1, 1)).

`residual[1]` is drawn at the stationary variance implied by the
*first* coefficient, `sigma^2 / (1 - phi_path[1]^2)`, as a local
approximation (the process is not exactly stationary once `phi`
moves). Later values follow `residual[t] = phi_path[t] *
residual[t - 1] + innovation[t]`, using the coefficient in force at
`t`.

Written as a non-mutating `accumulate` over `(phi, innovation)` pairs,
matching `src/model.jl`'s `ar_or_diff`, so it stays Mooncake-friendly.
"""
function tv_ar_path(eps::AbstractVector, sigma, phi_path::AbstractVector)
    innov = sigma .* eps
    first_val = innov[1] / sqrt(1 - phi_path[1]^2)
    steps = collect(zip(view(phi_path, 2:length(phi_path)),
                        view(innov, 2:length(innov))))
    rest = accumulate((prev, step) -> step[1] * prev + step[2], steps;
                       init=first_val)
    return vcat(first_val, rest)
end

"""
    model_v4(d::ModelData; transform=:log1p)

Time-varying-AR variant of `base_model`. Seasonality (partially-pooled
week-of-season population curve, iid location deviations, per-season
level shift) and backfill (non-monotonic delay-indexed population
profile with iid location deviations) are unchanged from `base_model`.

The residual block replaces the fixed per-location AR(1) coefficient
with a coefficient path `phi[t, l] = tanh(z[t, l])`, where `z` is a
slowly-varying random walk on the unconstrained scale, partially
pooled across locations:

- a population path `z_pop[t] = z_level + cumsum(z_pop_raw) .*
  sigma_z_pop` (non-centred), a single slow random walk shared by all
  locations;
- a per-location constant offset `z_loc_level[l]`, iid partially
  pooled (non-centred) around 0, capturing each location's typical
  persistence relative to the population;
- a per-location deviation path `z_loc[t, l] = cumsum(z_loc_raw[:,
  l]) .* sigma_z_loc` (non-centred), with the innovation sd
  `sigma_z_loc` SHARED (pooled) across locations, so individual paths
  are regularised towards the population path without being forced to
  be identical.

`z[t, l] = z_pop[t] + z_loc_level[l] + z_loc[t, l]`, and `phi[t, l] =
tanh(z[t, l])` keeps every entry in (-1, 1). The innovation sd
`sigma_ar` stays fixed per location and partially pooled exactly as in
`base_model`; only the AR coefficient itself is time-varying.

Priors on the random-walk step sds (`sigma_z_pop`, `sigma_z_loc`) are
tight (`truncated(Normal(0, 0.15); lower=0)` and `truncated(Normal(0,
0.1); lower=0)`) so `phi` moves slowly week to week rather than
behaving like independent draws.

Returns a `NamedTuple` with the same fields as `base_model` (`latent,
seasonal, residual, mu0, mu_w, delta, season_eff, phi, sigma_ar, r,
r_pop, sigma_obs, transform`), PLUS the full time-varying coefficient
path `phi_path` (`T x L`; `phi` itself is `phi_path[end, :]`, the
final-time coefficient per location, kept for interface-compatibility
with code written against `base_model`) and the two random-walk
innovation sds `sigma_z_pop`, `sigma_z_loc` that `project_v4` needs to
continue the walk forward.
"""
@model function model_v4(d::ModelData; transform::Symbol=:log1p)
    T, L, W, S, Dmax = model_dims(d)

    # --- Seasonality: identical to base_model ---
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

    # --- Residual: time-varying AR(1) coefficient, partially pooled ---
    z_level ~ Normal(0, 1)
    sigma_z_pop ~ truncated(Normal(0, 0.15); lower=0)
    z_pop_raw ~ filldist(Normal(0, 1), T)
    z_pop = z_level .+ cumsum(z_pop_raw) .* sigma_z_pop

    sigma_z_loc_level ~ truncated(Normal(0, 0.5); lower=0)
    z_loc_level_raw ~ filldist(Normal(0, 1), L)
    z_loc_level = z_loc_level_raw .* sigma_z_loc_level

    sigma_z_loc ~ truncated(Normal(0, 0.1); lower=0)
    z_loc_raw ~ filldist(Normal(0, 1), T, L)
    z_loc = reduce(hcat, [
        cumsum(view(z_loc_raw, :, l)) .* sigma_z_loc for l in 1:L
    ])

    z = z_pop .+ z_loc_level' .+ z_loc          # (T, L)
    phi_path = tanh.(z)                          # (T, L), each in (-1, 1)
    phi = phi_path[T, :]

    mu_log_sigma_ar ~ Normal(log(0.2), 1)
    tau_log_sigma_ar ~ truncated(Normal(0, 0.5); lower=0)
    z_sigma_ar ~ filldist(Normal(0, 1), L)
    sigma_ar = exp.(mu_log_sigma_ar .+ tau_log_sigma_ar .* z_sigma_ar)

    eps_raw ~ filldist(Normal(0, 1), T, L)
    residual = reduce(hcat, [
        tv_ar_path(view(eps_raw, :, l), sigma_ar[l], view(phi_path, :, l))
        for l in 1:L
    ])

    latent = seasonal .+ residual

    # --- Backfill: identical to base_model ---
    r_pop_anchor ~ Normal(0, 0.05)
    sigma_r_pop ~ truncated(Normal(0, 0.3); lower=0)
    r_steps_raw ~ filldist(Normal(0, 1), Dmax)
    r_pop = backfill_profile(r_pop_anchor, r_steps_raw .* sigma_r_pop)

    sigma_r_loc ~ truncated(Normal(0, 0.3); lower=0)
    r_loc_raw ~ filldist(Normal(0, 1), Dmax + 1, L)
    r = r_pop .+ r_loc_raw .* sigma_r_loc

    sigma_obs ~ truncated(Normal(0, 1); lower=0)

    for l in 1:L, t in 1:T
        if !ismissing(d.Y[t, l])
            mean_obs = latent[t, l] + r[d.delay[t, l] + 1, l]
            d.Y[t, l] ~ Normal(mean_obs, sigma_obs)
        end
    end

    return (;
        latent, seasonal, residual, mu0, mu_w, delta, season_eff,
        phi, phi_path, sigma_ar, sigma_z_pop, sigma_z_loc, r, r_pop,
        sigma_obs, transform,
    )
end
