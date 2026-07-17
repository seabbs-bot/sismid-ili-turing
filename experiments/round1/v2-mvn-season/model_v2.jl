# Round 1 candidate v2-mvn-season: correlated (multivariate-normal)
# location effects for the week-of-season seasonal deviation, in place
# of the committed BASE's (src/model.jl, `base_model`) iid per-location
# `delta`. Everything else is unchanged: partially-pooled AR(1)/
# difference residual, non-monotonic delay-indexed backfill.
#
# Motivation: docs/eda/04-cross-location.md finds moderate but genuine
# CONTEMPORANEOUS cross-location correlation on the differenced series
# (mean ~0.26, up to ~0.69 for closely related regions), with no
# lead-lag structure, which the EDA reads as support for a correlated
# location-effect (or AR-innovation) structure over fully independent
# locations. This is provisional inspiration, not a fixed target: EDA
# numbers are revisited across search rounds (see docs/plan.md).
#
# This file assumes `src/core.jl` (for `ModelData`) and `src/model.jl`
# (for `model_dims`, `ar_or_diff`, `backfill_profile`) are already
# `include`d into scope, exactly as `base_model` itself assumes for
# `core.jl`. It is a component file, not a package.

using Turing
using Distributions
using LinearAlgebra
using Statistics

"""
    model_v2(d::ModelData; transform=:log1p, difference=false)

Round 1 candidate `v2-mvn-season`: identical to `base_model` except the
per-location week-of-season seasonal deviations (`base_model`'s
`delta`) are drawn from a multivariate normal across the `L` locations,
with an LKJ prior on the location correlation matrix and per-location
scales, instead of `base_model`'s iid-per-cell `delta_raw .*
sigma_season_loc`.

# What changed vs `base_model`

- `sigma_season_loc` is now a length-`L` vector of per-location scales
  (`base_model` has one pooled scalar shared by every location), each
  `truncated(Normal(0, 1); lower=0)`.
- `loc_corr_chol ~ LKJCholesky(L, ETA_LKJ)` is the Cholesky factor of
  the `L x L` location correlation matrix `Omega` (`ETA_LKJ = 2.0`, a
  fixed, weakly-informative concentration constant favouring `Omega`
  mildly towards the identity, not sampled — the same "keep it simple"
  spirit as `base_model`'s fixed hyperprior scales).
- `delta_z ~ filldist(Normal(0, 1), L, W)` are raw iid standard normals
  (one length-`L` vector per week-of-season bin).
- The non-centred Cholesky transform `chol_sigma = Diagonal
  (sigma_season_loc) * loc_corr_chol.L` is the Cholesky factor of the
  full `L x L` covariance `Sigma = Diagonal(sigma_season_loc) * Omega *
  Diagonal(sigma_season_loc)`, so `delta[:, w] = chol_sigma *
  delta_z[:, w] ~ MvNormal(0, Sigma)` for every week-of-season `w`,
  independently across `w`. This is a plain matrix product (no
  in-model Cholesky decomposition), matching the non-centred style
  `base_model` already uses for `mu_w`/`phi`/`sigma_ar`/`r_pop`, and is
  the Mooncake-friendly form: only `LKJCholesky`'s own bijector needs a
  differentiable Cholesky, not any code in this model.

Everything else — `mu0`, the population `mu_w` random walk,
`season_eff`, the per-location AR(1)/difference `residual`
([`ar_or_diff`](@ref)), and the non-monotonic backfill
([`backfill_profile`](@ref)) — is copied verbatim from `base_model`.

Returns a `NamedTuple` with the same field names as `base_model`
(`latent, seasonal, residual, mu0, mu_w, delta, season_eff, phi,
sigma_ar, r, r_pop, sigma_obs, transform`), so `base_project` (or
[`project_v2`](@ref)) applies unchanged, plus two extra diagnostic
fields not present on `base_model`: `sigma_season_loc` (now a length-`L`
vector rather than a scalar) and `loc_corr` (the reconstructed `L x L`
location correlation matrix `Omega = loc_corr_chol.L *
loc_corr_chol.U`), useful for checking what cross-location correlation
the fit recovers.
"""
@model function model_v2(d::ModelData; transform::Symbol=:log1p,
                          difference::Bool=false)
    T, L, W, S, Dmax = model_dims(d)
    ETA_LKJ = 2.0

    # --- Seasonality: partially-pooled week-of-season random effect ---
    mu0 ~ Normal(0, 2)

    sigma_season_pop ~ truncated(Normal(0, 1); lower=0)
    mu_w_raw ~ filldist(Normal(0, 1), W)
    mu_w_uncentred = cumsum(mu_w_raw) .* sigma_season_pop
    mu_w = mu_w_uncentred .- mean(mu_w_uncentred)

    # Correlated (MVN/LKJ) location deviation, non-centred via the
    # Cholesky factor of the location covariance. Replaces
    # base_model's `sigma_season_loc` scalar + iid `delta_raw`.
    sigma_season_loc ~ filldist(truncated(Normal(0, 1); lower=0), L)
    loc_corr_chol ~ LKJCholesky(L, ETA_LKJ)
    delta_z ~ filldist(Normal(0, 1), L, W)
    chol_sigma = Diagonal(sigma_season_loc) * loc_corr_chol.L
    delta = permutedims(chol_sigma * delta_z)  # (W, L), Cov across L

    sigma_season_time ~ truncated(Normal(0, 1); lower=0)
    season_eff_raw ~ filldist(Normal(0, 1), S)
    season_eff = season_eff_raw .* sigma_season_time

    seasonal = mu0 .+ mu_w[d.woy] .+ delta[d.woy, :] .+ season_eff[d.season]

    # --- Post-seasonal residual: partially-pooled AR(1) or difference ---
    # (unchanged from base_model)
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

    # --- Backfill: non-monotonic delay-indexed revision ---
    # (unchanged from base_model)
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

    loc_corr = loc_corr_chol.L * loc_corr_chol.U

    return (;
        latent, seasonal, residual, mu0, mu_w, delta, season_eff,
        phi, sigma_ar, r, r_pop, sigma_obs, transform,
        sigma_season_loc, loc_corr,
    )
end
