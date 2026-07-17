# Round 1 candidate v5-backfill: richer, time-varying backfill revision.
# See src/model.jl for the BASE this extends (docs/brief.md, docs/plan.md
# for design rationale). This file is self-contained: it relies on
# `backfill_profile`, `ar_or_diff`, and `model_dims` from src/model.jl
# already being in scope (both are `include`d together with this file,
# see check_v5.jl).
#
# CHANGE FROM BASE: the backfill revision profile is (a) meant to be fit
# over a WIDER delay window (pair this variant with `ModelData` built at
# `Dmax` around 12, rather than the contract default of 6 —
# docs/eda/02-backfill.md finds revisions settle by delay ~10-15 weeks,
# so Dmax=6 misses a real tail), and (b) TIME-VARYING: the population
# revision profile differs between the rising and falling phases of the
# season (a crude, real-time-safe proxy for the EDA's peak-/off-season
# split — the EDA splits by settled value, which is not available in
# real time). Seasonality and residual dynamics are otherwise unchanged
# from `base_model`.

using Turing
using Distributions
using LinearAlgebra
using Statistics

"""
    season_phase(woy::AbstractVector{<:Integer}, W::Integer)

Crude, real-time-safe season-phase indicator: `1` (rising) for the
first half of the season's week-of-season index, `2` (falling) for the
second half. `woy` and `W` are data, not sampled parameters, so this is
computed once outside the model and used to select which population
backfill profile applies at each week. This is a coarse stand-in for
docs/eda/02-backfill.md's peak-/off-season split (which uses the
settled value, unavailable at forecast time): flu seasons here run
roughly Oct-May, so the season midpoint falls close to the typical
January/February peak.
"""
season_phase(woy::AbstractVector{<:Integer}, W::Integer) =
    [w <= cld(W, 2) ? 1 : 2 for w in woy]

"""
    model_v5(d::ModelData; transform=:log1p)

`base_model` with a richer backfill: a wider delay window (build `d`
with `Dmax` around 12) and a revision profile that varies by season
phase (rising vs falling, see [`season_phase`](@ref)), while remaining
non-monotonic and partially pooled across locations. Seasonality (the
partially-pooled week-of-season random effect) and the per-location
AR(1) residual are unchanged from `base_model`.

# Backfill

Two population profiles, `r_pop_rising` and `r_pop_falling`, are each
built by [`backfill_profile`](@ref) from a SHARED anchor
(`r_pop_anchor`) and SHARED step scale (`sigma_r_pop`), but independent
step innovations, so the two phases are partially pooled toward a
common process-noise magnitude while their shapes can differ.

Location deviations (`sigma_r_loc`, `r_loc_raw`) are shared across
phase (not duplicated per phase) to keep the added parameter count
modest: relative to a same-`Dmax` version of `base_model`, this adds
`Dmax` extra parameters (one extra set of population steps), not
`Dmax * L` — deliberately, since this variant is scored on WIS AND WIS
SD and a full phase x location interaction would risk overfitting the
short validation history.

`r_pop` is returned as a `(Dmax+1) x 2` matrix (columns: rising,
falling); `r` as a `(Dmax+1) x L x 2` array (delay x location x
phase). `mu0`, `mu_w`, `delta`, `season_eff`, `phi`, `sigma_ar`,
`residual` match `base_model` exactly, field-for-field, so
[`project_v5`](@ref) can mirror `base_project`'s dynamics unchanged; no
backfill revision is applied to future weeks either way.

Returns a `NamedTuple` `(latent, seasonal, residual, mu0, mu_w, delta,
season_eff, phi, sigma_ar, r, r_pop, sigma_obs, transform)`.
"""
@model function model_v5(d::ModelData; transform::Symbol=:log1p)
    T, L, W, S, Dmax = model_dims(d)
    phase = season_phase(d.woy, W)  # (T,), data-derived, not sampled

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

    # --- Post-seasonal residual: identical to base_model (AR(1) only) ---
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
        ar_or_diff(view(eps_raw, :, l), sigma_ar[l], phi[l], false)
        for l in 1:L
    ])

    latent = seasonal .+ residual

    # --- Backfill: non-monotonic, location- AND phase-varying ---
    r_pop_anchor ~ Normal(0, 0.05)
    sigma_r_pop ~ truncated(Normal(0, 0.3); lower=0)

    r_steps_rising_raw ~ filldist(Normal(0, 1), Dmax)
    r_steps_falling_raw ~ filldist(Normal(0, 1), Dmax)
    r_pop_rising = backfill_profile(
        r_pop_anchor, r_steps_rising_raw .* sigma_r_pop,
    )
    r_pop_falling = backfill_profile(
        r_pop_anchor, r_steps_falling_raw .* sigma_r_pop,
    )
    r_pop = hcat(r_pop_rising, r_pop_falling)  # (Dmax+1) x 2

    sigma_r_loc ~ truncated(Normal(0, 0.3); lower=0)
    r_loc_raw ~ filldist(Normal(0, 1), Dmax + 1, L)
    r_loc = r_loc_raw .* sigma_r_loc  # shared across phase

    r_rising = r_pop_rising .+ r_loc
    r_falling = r_pop_falling .+ r_loc
    r = cat(r_rising, r_falling; dims=3)  # (Dmax+1) x L x 2

    sigma_obs ~ truncated(Normal(0, 1); lower=0)

    for l in 1:L, t in 1:T
        if !ismissing(d.Y[t, l])
            rev = phase[t] == 1 ? r_rising[d.delay[t, l] + 1, l] :
                  r_falling[d.delay[t, l] + 1, l]
            mean_obs = latent[t, l] + rev
            d.Y[t, l] ~ Normal(mean_obs, sigma_obs)
        end
    end

    return (;
        latent, seasonal, residual, mu0, mu_w, delta, season_eff,
        phi, sigma_ar, r, r_pop, sigma_obs, transform,
    )
end
