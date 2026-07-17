# Round 2 candidate severity: a per-season "severity" random effect
# that SCALES the seasonal amplitude, in place of base_model's
# (src/model.jl, `base_model`) implicit assumption that every season
# has the same amplitude (only `season_eff`, an additive per-season
# LEVEL shift, varies by season there). Everything else is unchanged:
# partially-pooled week-of-season population curve, iid partially-
# pooled location deviation, per-location AR(1)/difference residual,
# non-monotonic delay-indexed backfill.
#
# Motivation: docs/eda/04-cross-location.md finds the per-season
# seasonal AMPLITUDE (season peak minus off-season baseline) is far
# more correlated across locations (mean 0.68, up to 0.94) than the
# week-to-week differenced series (mean 0.24) -- i.e. "how bad was
# this whole season" is a much stronger shared signal than "did this
# specific week move together". That reads as a season-level latent
# severity effect shared across locations, multiplying the seasonal
# curve's amplitude, on top of (not instead of) the existing
# per-location `delta` deviation and week-to-week AR(1) coupling. This
# is provisional inspiration, not a fixed target: EDA numbers are
# revisited across search rounds (see docs/plan.md).
#
# This file assumes `src/core.jl` (for `ModelData`) and `src/model.jl`
# (for `model_dims`, `ar_or_diff`, `backfill_profile`,
# `observation_index`) are already `include`d into scope, exactly as
# `base_model` itself assumes for `core.jl`. It is a component file,
# not a package.

using Turing
using Distributions
using LinearAlgebra
using Statistics

"""
    model_severity(d::ModelData; transform=:log, difference=false,
                   obsdata=observation_index(d))

Round 2 candidate `severity`: identical to `base_model` except the
week-of-season seasonal SHAPE (`base_model`'s `mu_w[w] + delta[w, l]`,
i.e. the curve around its own mean, `mu_w` already centred to mean
zero) is multiplied by a per-season, partially-pooled, positive
`severity_mult[s]` before the additive `season_eff[s]` level shift is
added:

```
seasonal[t, l] = mu0 + severity_mult[season[t]] *
                 (mu_w[woy[t]] + delta[woy[t], l]) + season_eff[season[t]]
```

# What changed vs `base_model`

- `mu_severity ~ Normal(0, 0.3)` and `sigma_severity ~
  truncated(Normal(0, 0.3); lower=0)` are the population mean/scale of
  a log-severity random effect, partially pooled across the `S`
  seasons (weakly informative: `sigma_severity` on this scale lets
  `severity_mult` range roughly 0.5x-2x for a typical season, without
  dominating the existing `delta`/`residual` variance).
- `severity_raw ~ filldist(Normal(0, 1), S)` are raw iid standard
  normals, one per season, non-centred like every other hierarchical
  scale in `base_model` (`mu_w`, `phi`, `sigma_ar`, `r_pop`).
- `severity_mult = exp.(mu_severity .+ sigma_severity .* severity_raw)`
  is always positive (a season can shrink or amplify the curve, never
  flip its sign), length `S`, with `severity_mult[s] == 1` at the
  population mean when `mu_severity == 0`.
- The seasonal reconstruction multiplies `severity_mult[season[t]]`
  onto `mu_w[woy[t]] + delta[woy[t], l]` only; `mu0` (the overall
  level) and `season_eff` (the existing additive per-season level
  shift) are untouched, so `severity_mult` is purely an AMPLITUDE
  effect, not a second level shift competing with `season_eff` for the
  same signal.

Everything else -- the population `mu_w` random walk, the iid
partially-pooled `delta`, the per-location AR(1)/difference `residual`
([`ar_or_diff`](@ref)), the non-monotonic backfill
([`backfill_profile`](@ref)), and the vectorised observation likelihood
over [`observation_index`](@ref)'s precomputed non-missing cells -- is
copied verbatim from `base_model`.

Returns a `NamedTuple` with the same field names as `base_model`
(`latent, seasonal, residual, mu0, mu_w, delta, season_eff, phi,
sigma_ar, r, r_pop, sigma_obs, transform`), so `base_project` still
type-checks against it, plus the fields
[`project_severity`](@ref) needs and diagnostic scalars: `severity_mult`
(length `S`, the per-season amplitude multiplier `project_severity`
reuses for in-season horizons), `mu_severity`, and `sigma_severity`.
"""
@model function model_severity(d::ModelData; transform::Symbol=:log,
                                difference::Bool=false,
                                obsdata=observation_index(d))
    T, L, W, S, Dmax = model_dims(d)
    obs_idx, r_idx, yobs = obsdata.obs_idx, obsdata.r_idx, obsdata.yobs

    # --- Seasonality: partially-pooled week-of-season random effect ---
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

    # Season-severity amplitude effect (new): a partially-pooled,
    # always-positive per-season multiplier on the seasonal SHAPE
    # (mu_w + delta), non-centred on the log scale.
    mu_severity ~ Normal(0, 0.3)
    sigma_severity ~ truncated(Normal(0, 0.3); lower=0)
    severity_raw ~ filldist(Normal(0, 1), S)
    severity_mult = exp.(mu_severity .+ sigma_severity .* severity_raw)

    shape = mu_w[d.woy] .+ delta[d.woy, :]                  # (T x L)
    seasonal = mu0 .+ severity_mult[d.season] .* shape .+ season_eff[d.season]

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

    mu_obs = latent[obs_idx] .+ r[r_idx]
    yobs ~ arraydist(Normal.(mu_obs, sigma_obs))

    return (;
        latent, seasonal, residual, mu0, mu_w, delta, season_eff,
        phi, sigma_ar, r, r_pop, sigma_obs, transform,
        severity_mult, mu_severity, sigma_severity,
    )
end
