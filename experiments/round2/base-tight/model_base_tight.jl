# Round 2 candidate base-tight: `base_model` (src/model.jl) with its
# under-regularised hyperpriors TIGHTENED, structure unchanged.
#
# MOTIVATION: several round-1 candidates' diagnostics
# (`experiments/round1/_results/*/diagnostics.txt`, column
# `prior_frac_outside`) show 20-65% of prior-predictive draws land
# outside the plausible wILI% range (`src/diagnostics.jl`'s
# `PLAUSIBLE_RANGE = (0, 15)`), and a standalone re-check of
# `base_model` itself (not run through Turing here -- see the
# `explore_priors*.jl` scratch scripts this file's priors were tuned
# against) finds a prior-predictive q99 in the THOUSANDS of percent
# wILI, occasionally reaching q99 ~ 7800%+. `base_model`'s docstring
# already flags the mechanism: `mu_w` is a non-centred cumulative
# random walk over `W ~ 52` weeks, so its entries have standard
# deviation `sigma_season_pop * sqrt(week)` -- growing to
# `sigma_season_pop * 7.2` at week 52 -- and with `sigma_season_pop`
# itself only weakly bounded (`truncated(Normal(0, 1); lower=0)`), an
# unremarkable prior draw of `sigma_season_pop` around 1-2 gives `mu_w`
# entries with sd 7-14 BEFORE the `fourthroot` back-transform (`x ->
# x^4`, `src/core.jl`) turns that into an astronomical wILI% value.
# `phi_pop_mean`/`phi_pop_sd` have the same problem on the AR side:
# `Normal(0, 1)`/`truncated(Normal(0, 0.5); lower=0)` regularly push
# the partially-pooled `phi = tanh(...)` to within a whisker of a unit
# root (tanh saturates near +-1 only for |argument| >~ 3, which this
# prior reaches routinely), at which point the AR(1) residual's
# stationary variance `sigma^2 / (1 - phi^2)` explodes. And `mu0 ~
# Normal(0, 2)` alone -- with every other hyperprior already tightened
# -- was found to be the SINGLEST largest contributor: an unremarkable
# `mu0` draw of 2-3 (well within its 2 prior sd) already back-transforms
# to `2^4`-`3^4` = 16-81% before any seasonal/AR contribution is added.
#
# WHAT CHANGED (every other line is copied verbatim from `base_model`).
# Format: prior name: base_model -> base-tight -- why.
#
# - `mu0`: `Normal(0, 2)` -> `Normal(1, 0.4)` -- dominant single driver
#   of the tail (see above); recentred near `fourthroot(1%)` ~ 1, the
#   rough population mean wILI%, with a much narrower sd.
# - `sigma_season_pop`: `trunc. Normal(0, 1)` -> `trunc. Normal(0,
#   0.07)` -- bounds the `mu_w` random walk's cumulative sd at week 52
#   to roughly `0.07 * 7.2 ~ 0.5`, plausible for a fourthroot-scale
#   seasonal amplitude.
# - `sigma_season_loc`: `trunc. Normal(0, 1)` -> `trunc. Normal(0,
#   0.2)` -- per-location seasonal deviation should be a fraction of
#   the population amplitude, not comparable to it.
# - `sigma_season_time`: `trunc. Normal(0, 1)` -> `trunc. Normal(0,
#   0.2)` -- same reasoning for the per-season level shift.
# - `phi_pop_mean`: `Normal(0, 1)` -> `Normal(0, 0.3)` -- keeps the
#   partially-pooled AR coefficient's pre-tanh location away from the
#   tanh saturation region that produces a near-unit root.
# - `phi_pop_sd`: `trunc. Normal(0, 0.5)` -> `trunc. Normal(0, 0.15)`
#   -- same, for the spread across locations.
# - `mu_log_sigma_ar`: `Normal(log(0.2), 1)` -> `Normal(log(0.12),
#   0.3)` -- narrows the AR innovation sd's population mean/spread so
#   a typical location's residual scale stays small relative to the
#   (now also tightened) seasonal amplitude.
# - `tau_log_sigma_ar`: `trunc. Normal(0, 0.5)` -> `trunc. Normal(0,
#   0.15)` -- same, for the spread across locations.
#
# `r_pop_anchor`, `sigma_r_pop`, `sigma_r_loc`, and `sigma_obs` are
# UNCHANGED from `base_model`: the diagnosed tail comes from the
# seasonal random walk and the AR coefficient/scale, not from the
# backfill block (already tightly anchored via `r_pop_anchor ~
# Normal(0, 0.05)`), so they are left alone to keep this a minimal,
# targeted fix rather than a general re-tune.
#
# This is a PRIOR-ONLY change: no sampled site is added, removed, or
# reshaped, no return field changes, and no likelihood term changes.
# `project_base_tight.jl` therefore reuses `base_project` unchanged
# (see that file), and this candidate's `build_model` is a drop-in
# replacement for `base_model` anywhere `(build_model, project)` is
# threaded through (`experiments/README.md`).
#
# This file assumes `src/core.jl` (for `ModelData`) and `src/model.jl`
# (for `model_dims`, `ar_or_diff`, `backfill_profile`,
# `observation_index`) are already `include`d into scope, exactly as
# `base_model` itself assumes -- see `check_base_tight.jl`.

using Turing
using Distributions
using LinearAlgebra
using Statistics

"""
    model_base_tight(d::ModelData; transform=:log, difference=false,
                      obsdata=observation_index(d))

`base_model` (src/model.jl) with tightened hyperpriors so the PRIOR
predictive back-transformed wILI% stays mostly in the plausible `(0,
15)` range (`src/diagnostics.jl`'s `PLAUSIBLE_RANGE`) instead of the
heavy-tailed, occasionally-thousands-of-percent draws `base_model`
produces (see this file's header comment for the tuning rationale and
the before/after table). Structure -- partially-pooled week-of-season
seasonality, per-location AR(1)/difference residual, non-monotonic
delay-indexed backfill, the vectorised observation model over
[`observation_index`](@ref) -- is IDENTICAL to `base_model`; only the
hyperprior scale/location parameters listed in the header table
differ. See `base_model`'s own docstring for the full component
description, which otherwise applies unchanged here.

Returns the same `NamedTuple` as `base_model`: `(latent, seasonal,
residual, mu0, mu_w, delta, season_eff, phi, sigma_ar, r, r_pop,
sigma_obs, transform)`, so `base_project` (src/forecast.jl) consumes
draws from this model exactly as it does `base_model`'s.
"""
@model function model_base_tight(d::ModelData; transform::Symbol=:log,
                                  difference::Bool=false,
                                  obsdata=observation_index(d))
    T, L, W, S, Dmax = model_dims(d)
    obs_idx, r_idx, yobs = obsdata.obs_idx, obsdata.r_idx, obsdata.yobs

    # --- Seasonality: partially-pooled week-of-season random effect ---
    # (tightened: mu0, sigma_season_pop, sigma_season_loc,
    # sigma_season_time -- see header table)
    mu0 ~ Normal(1, 0.4)

    sigma_season_pop ~ truncated(Normal(0, 0.07); lower=0)
    mu_w_raw ~ filldist(Normal(0, 1), W)
    mu_w_uncentred = cumsum(mu_w_raw) .* sigma_season_pop
    mu_w = mu_w_uncentred .- mean(mu_w_uncentred)

    sigma_season_loc ~ truncated(Normal(0, 0.2); lower=0)
    delta_raw ~ filldist(Normal(0, 1), W, L)
    delta = delta_raw .* sigma_season_loc

    sigma_season_time ~ truncated(Normal(0, 0.2); lower=0)
    season_eff_raw ~ filldist(Normal(0, 1), S)
    season_eff = season_eff_raw .* sigma_season_time

    seasonal = mu0 .+ mu_w[d.woy] .+ delta[d.woy, :] .+ season_eff[d.season]

    # --- Post-seasonal residual: partially-pooled AR(1) or difference ---
    # (tightened: phi_pop_mean, phi_pop_sd, mu_log_sigma_ar,
    # tau_log_sigma_ar -- see header table)
    phi_pop_mean ~ Normal(0, 0.3)
    phi_pop_sd ~ truncated(Normal(0, 0.15); lower=0)
    phi_raw ~ filldist(Normal(0, 1), L)
    phi = tanh.(phi_pop_mean .+ phi_pop_sd .* phi_raw)

    mu_log_sigma_ar ~ Normal(log(0.12), 0.3)
    tau_log_sigma_ar ~ truncated(Normal(0, 0.15); lower=0)
    z_sigma_ar ~ filldist(Normal(0, 1), L)
    sigma_ar = exp.(mu_log_sigma_ar .+ tau_log_sigma_ar .* z_sigma_ar)

    eps_raw ~ filldist(Normal(0, 1), T, L)
    residual = reduce(hcat, [
        ar_or_diff(view(eps_raw, :, l), sigma_ar[l], phi[l], difference)
        for l in 1:L
    ])

    latent = seasonal .+ residual

    # --- Backfill: non-monotonic delay-indexed revision (UNCHANGED) ---
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
    )
end
