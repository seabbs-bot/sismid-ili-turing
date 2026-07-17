# Round 2 candidate season-backfill: season-level random effect on the
# backfill revision profile. See src/model.jl for the BASE this extends
# (docs/brief.md, docs/plan.md for design rationale). This file is
# self-contained: it relies on `backfill_profile`, `ar_or_diff`, and
# `model_dims` from src/model.jl already being in scope (both are
# `include`d together with this file, see check_season_backfill.jl).
#
# CHANGE FROM BASE: docs/eda/02-backfill.md ("Revision structure by
# location and by tracked season" and the phase-crossed follow-up)
# finds that HHS Region 1 and Region 4 fully reverse the SIGN of their
# delay-1 revision bias between the two tracked training seasons
# (2015/16, 2016/17), and that this tracks the season as a whole, not
# a within-season peak-/off-season phase (ruling out the phase
# explanation used by the `v5-backfill` Round 1 candidate). This
# candidate adds a season-level deviation to the delay-indexed
# revision profile, partially pooled across seasons via `d.season`
# (the same season index `base_model` already uses for `season_eff`,
# so `S` stays small: 2-3 seasons in a `window_weeks=104` fit, not the
# full season history). It is additive with, not an interaction with,
# the existing per-location deviation: `r[delay, l, s] = r_pop[delay] +
# r_loc[delay, l] + r_season[delay, s]`, so a location's revision
# profile can be shifted the same way in every season (via `r_loc`) and
# every location can be shifted the same way in a given season (via
# `r_season`), without fitting a separate profile per (location,
# season) pair. Seasonality and residual dynamics are otherwise
# unchanged from `base_model`.

using Turing
using Distributions
using LinearAlgebra
using Statistics

"""
    model_season_backfill(d::ModelData; transform=:log, difference=false)

`base_model` with a season-varying backfill: the delay-indexed revision
profile gets a partially-pooled deviation per training season (`d.season`,
`1:S`), on top of the existing partially-pooled per-location deviation,
while remaining non-monotonic. Seasonality (the partially-pooled
week-of-season random effect) and the per-location AR(1) residual are
unchanged from `base_model`.

# Backfill

The population profile `r_pop` is built exactly as in `base_model` (an
anchor at the largest delay plus `Dmax` free steps, via
[`backfill_profile`](@ref)). Two independent deviations are added to
it, both zero-mean and partially pooled around `r_pop`:

- `r_loc[delay, l]`, iid `Normal(0, sigma_r_loc)`, unchanged from
  `base_model`.
- `r_season[delay, s]`, iid `Normal(0, sigma_r_season)`, NEW: one
  deviation per training season, shared across all locations. This is
  what lets the delay-1 revision change sign season to season (as
  Region 1 and Region 4 do in the EDA) without needing a full
  location x season interaction.

The full revision applied at a given (delay, location, season) cell is
`r[delay, l, s] = r_pop[delay] + r_loc[delay, l] + r_season[delay, s]`,
built by broadcasting the three terms into a `(Dmax+1) x L x S` array
so the observation loop is a single indexing operation, not a branch
(contrast with `v5-backfill`'s phase ternary, which only needed 2
branches; a branch per season would not scale as cleanly and the
broadcast form is exactly as fast to trace).

Relative to `base_model` at the same `Dmax`, this adds `(Dmax + 1) * S`
free parameters (`r_season_raw`) plus one scale (`sigma_r_season`) --
with the `window_weeks=104` (two-season) fits used for candidate
selection, `S` is 2-3, so this is `(Dmax + 1) * 2` to `(Dmax + 1) * 3`
extra parameters (e.g. 26-39 at `Dmax=12`), not `(Dmax + 1) * L * S`: a
deliberately modest addition, since this variant is scored on WIS AND
WIS SD and the EDA itself notes only two tracked seasons of revision
history are available to estimate the season effect from.

# Returns

A `NamedTuple` `(latent, seasonal, residual, mu0, mu_w, delta,
season_eff, phi, sigma_ar, r, r_pop, r_season, sigma_obs, transform)`.
`mu0`, `mu_w`, `delta`, `season_eff`, `phi`, `sigma_ar`, `residual`
match `base_model` exactly, field-for-field, so
[`project_season_backfill`](@ref) can mirror `base_project`'s dynamics
unchanged; no backfill revision is applied to future weeks either way.
`r` changes shape relative to `base_model`, from `(Dmax+1) x L` to
`(Dmax+1) x L x S` (delay x location x season), and `r_season` (`(Dmax+1)
x S`) is a new field not present in `base_model`'s return value.
"""
@model function model_season_backfill(
    d::ModelData; transform::Symbol=:log, difference::Bool=false,
)
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

    # --- Post-seasonal residual: identical to base_model ---
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

    # --- Backfill: non-monotonic, location- AND season-varying ---
    r_pop_anchor ~ Normal(0, 0.05)
    sigma_r_pop ~ truncated(Normal(0, 0.3); lower=0)
    r_steps_raw ~ filldist(Normal(0, 1), Dmax)
    r_pop = backfill_profile(r_pop_anchor, r_steps_raw .* sigma_r_pop)

    sigma_r_loc ~ truncated(Normal(0, 0.3); lower=0)
    r_loc_raw ~ filldist(Normal(0, 1), Dmax + 1, L)
    r_loc = r_loc_raw .* sigma_r_loc

    sigma_r_season ~ truncated(Normal(0, 0.3); lower=0)
    r_season_raw ~ filldist(Normal(0, 1), Dmax + 1, S)
    r_season = r_season_raw .* sigma_r_season

    # (Dmax+1) x L x S, additive: population + location + season.
    r = reshape(r_pop, Dmax + 1, 1, 1) .+
        reshape(r_loc, Dmax + 1, L, 1) .+
        reshape(r_season, Dmax + 1, 1, S)

    sigma_obs ~ truncated(Normal(0, 1); lower=0)

    for l in 1:L, t in 1:T
        if !ismissing(d.Y[t, l])
            mean_obs = latent[t, l] + r[d.delay[t, l] + 1, l, d.season[t]]
            d.Y[t, l] ~ Normal(mean_obs, sigma_obs)
        end
    end

    return (;
        latent, seasonal, residual, mu0, mu_w, delta, season_eff,
        phi, sigma_ar, r, r_pop, r_season, sigma_obs, transform,
    )
end
