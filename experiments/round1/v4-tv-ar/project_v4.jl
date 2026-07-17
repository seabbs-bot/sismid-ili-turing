# Forecasting for the time-varying-AR model (v4-tv-ar). Mirrors
# src/forecast.jl's `base_project`, but the AR coefficient is itself
# forward-simulated (continuing its own random walk) rather than held
# fixed, since `model_v4`'s `phi` moves over time by design.
#
# Assumes `src/forecast.jl` is loaded (for `_field` and `ModelData`)
# alongside `model_v4.jl`, matching this experiment's `check_v4.jl`.

"""
    project_v4(draw, data, horizons)

`project` function bridging one `model_v4` posterior draw (generated
quantities, e.g. from `generated_draws`) to the `(L x
maximum(horizons))` latent-scale forecast matrix `forecast_quantiles`
consumes.

The draw must expose `model_v4`'s return fields `mu0`, `mu_w` (`W`),
`delta` (`W x L`), `season_eff` (`S`), `phi_path` (`T x L`), `sigma_ar`
(`L`), `sigma_z_pop`, `sigma_z_loc`, and `residual` (`T x L`). All
derived per-horizon quantities (future seasonal level, the continued
AR-coefficient path, the forward residual) are recomputed here from
these raw pieces, exactly as `base_project` recomputes the future
seasonal level from `mu0`/`mu_w`/`delta`/`season_eff` rather than
reading a precomputed future value off the draw.

For each horizon `h = 1:maximum(horizons)` and location `l`:

- **Seasonality**, as in `base_project`: future week-of-season `w =
  mod1(data.woy[end] + h, data.W)`, `mu0 + mu_w[w] + delta[w, l] +
  season_eff[s]` with `s = data.season[end]` (the current season's
  level shift held for the short within-season horizons).
- **AR coefficient**: `model_v4`'s coefficient path is CONTINUED
  forward on its unconstrained scale rather than held at its last
  value, since the point of this variant is that `phi` keeps moving.
  The last in-sample value is recovered as `z = atanh(phi_path[end,
  l])`; at each future step `z += sqrt(sigma_z_pop^2 + sigma_z_loc^2) *
  randn()` (the combined population + location random-walk step sd)
  and `phi_h = tanh(z)`.
- **Residual**: `resid_h = phi_h * resid_{h - 1} + sigma_ar[l] *
  randn()`, seeded from `residual[end, l]`, exactly `base_project`'s
  AR(1) recursion but with the freshly-simulated `phi_h` in place of a
  fixed `phi[l]`.

No backfill revision is applied to future weeks (revision -> 0),
matching `base_project` and the observed-vs-forecast distinction in
docs/contracts.md. Each call draws fresh phi-path and residual
innovations, so repeated calls with the same draw give different
posterior-predictive realisations, matching what an MCMC/Pathfinder
draw represents.
"""
function project_v4(draw, data::ModelData, horizons)
    L = data.L
    H = maximum(horizons)
    mu0 = _field(draw, :mu0)
    mu_w = _field(draw, :mu_w)
    delta = _field(draw, :delta)              # (W x L)
    season_eff = _field(draw, :season_eff)
    phi_path = _field(draw, :phi_path)        # (T x L)
    sigma_ar = _field(draw, :sigma_ar)        # (L,)
    sigma_z_pop = _field(draw, :sigma_z_pop)
    sigma_z_loc = _field(draw, :sigma_z_loc)
    residual = _field(draw, :residual)        # (T x L)

    seff = season_eff[data.season[end]]
    step_sd = sqrt(sigma_z_pop^2 + sigma_z_loc^2)

    resid = Float64[residual[end, l] for l in 1:L]
    z = Float64[atanh(clamp(phi_path[end, l], -0.999999, 0.999999))
                for l in 1:L]

    latent = Matrix{Float64}(undef, L, H)
    for h in 1:H
        w = mod1(data.woy[end] + h, data.W)
        for l in 1:L
            z[l] += step_sd * randn()
            phi_h = tanh(z[l])
            resid[l] = phi_h * resid[l] + sigma_ar[l] * randn()
            latent[l, h] = mu0 + mu_w[w] + delta[w, l] + seff + resid[l]
        end
    end
    return latent
end
