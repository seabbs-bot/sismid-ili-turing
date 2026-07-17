# Forecast projection for round 1 candidate v2-mvn-season
# (`model_v2`, see model_v2.jl in this directory).
#
# `model_v2` changes only how the seasonal location deviation `delta`
# is drawn (correlated/MVN instead of iid); the AR(1)/difference
# residual dynamics and the seasonal reconstruction it feeds are
# otherwise identical to base_model. So this mirrors
# `src/forecast.jl`'s `base_project` exactly, and is included here
# standalone (rather than reused from src/forecast.jl) so the variant
# stays self-contained: nothing outside this directory is required to
# project a `model_v2` fit forward. Assumes `src/core.jl`'s
# `ModelData` is in scope.

using Statistics

# Read a field from a draw that may be a NamedTuple or a Dict{Symbol}.
# Local, differently-named copy of src/forecast.jl's `_field` so this
# file has no load-order dependency on src/forecast.jl.
_field_v2(draw, key::Symbol) =
    draw isa AbstractDict ? draw[key] : getproperty(draw, key)

"""
    project_v2(draw, data::ModelData, horizons)

`project` function bridging one `model_v2` posterior draw (a generated-
quantities `NamedTuple`, e.g. from `generated_draws` in
`src/inference.jl`) to the `(L x maximum(horizons))` latent-scale
forecast matrix that `forecast_quantiles` (`src/forecast.jl`) consumes.

The draw must expose the fields `model_v2` returns with the same names
as `base_model`: `mu0`, `mu_w` (length `W`), `delta` (`W x L`, the
correlated location deviation matrix — already realised on the
modelling scale by `model_v2`, so its cross-location correlation needs
no special handling here), `season_eff` (length `S`), `phi` (`L`),
`sigma_ar` (`L`), and `residual` (`T x L`).

For each horizon `h = 1:maximum(horizons)`, derived quantities are
recomputed fresh from these raw fields (nothing beyond `residual[end,
l]` is reused from the training period):

- **Seasonality** for future week-of-season `w = mod1(data.woy[end] +
  h, data.W)` is rebuilt as `mu0 + mu_w[w] + delta[w, l] + season_eff
  [s]`, with `s = data.season[end]` (the current season's level shift
  held for the short within-season horizons, as in `base_project`).
- **Residual** is forward-simulated as an AR(1) recursion `resid_h =
  phi * resid_{h-1} + sigma_ar * randn()`, seeded from the last
  training-week residual `residual[end, l]`. Innovations are drawn
  fresh each step (not decayed) so predictive intervals stay
  calibrated; each call gives a fresh posterior-predictive realisation
  for the draw.

No backfill revision is applied to future weeks (revision -> 0),
matching `base_project` and the observed-vs-forecast distinction in
docs/contracts.md. `model_v2`'s default residual is a stationary AR(1)
(`difference=false`); this mirrors that recursion.
"""
function project_v2(draw, data::ModelData, horizons)
    L = data.L
    H = maximum(horizons)
    mu0 = _field_v2(draw, :mu0)
    mu_w = _field_v2(draw, :mu_w)
    delta = _field_v2(draw, :delta)          # (W x L), correlated
    season_eff = _field_v2(draw, :season_eff)
    phi = _field_v2(draw, :phi)              # (L,)
    sigma_ar = _field_v2(draw, :sigma_ar)    # (L,)
    residual = _field_v2(draw, :residual)    # (T x L)
    seff = season_eff[data.season[end]]
    resid = Float64[residual[end, l] for l in 1:L]
    latent = Matrix{Float64}(undef, L, H)
    for h in 1:H
        w = mod1(data.woy[end] + h, data.W)
        for l in 1:L
            resid[l] = phi[l] * resid[l] + sigma_ar[l] * randn()
            latent[l, h] = mu0 + mu_w[w] + delta[w, l] + seff + resid[l]
        end
    end
    return latent
end
