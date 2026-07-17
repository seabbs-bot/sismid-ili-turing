# Forecast projection for round 2 candidate severity
# (`model_severity`, see model_severity.jl in this directory).
#
# `model_severity` changes only how the seasonal curve's amplitude is
# scaled (a per-season `severity_mult` multiplying `mu_w + delta`); the
# AR(1)/difference residual dynamics are otherwise identical to
# base_model. So this mirrors `src/forecast.jl`'s `base_project`
# exactly, with one extra multiplication, and is included here
# standalone (rather than reused from src/forecast.jl) so the variant
# stays self-contained: nothing outside this directory is required to
# project a `model_severity` fit forward. Assumes `src/core.jl`'s
# `ModelData` is in scope.

using Statistics

# Read a field from a draw that may be a NamedTuple or a Dict{Symbol}.
# Local, differently-named copy of src/forecast.jl's `_field` so this
# file has no load-order dependency on src/forecast.jl.
_field_severity(draw, key::Symbol) =
    draw isa AbstractDict ? draw[key] : getproperty(draw, key)

"""
    project_severity(draw, data::ModelData, horizons; difference=false)

`project` function bridging one `model_severity` posterior draw (a
generated-quantities `NamedTuple`, e.g. from `generated_draws` in
`src/inference.jl`) to the `(L x maximum(horizons))` latent-scale
forecast matrix that `forecast_quantiles` (`src/forecast.jl`) consumes.

The draw must expose the fields `model_severity` returns: `mu0`,
`mu_w` (length `W`), `delta` (`W x L`), `season_eff` (length `S`),
`severity_mult` (length `S`, the season-level amplitude multiplier),
`phi` (`L`), `sigma_ar` (`L`), and `residual` (`T x L`).

The forecast horizons here (1-4 weeks) never cross a season boundary,
so -- exactly as `base_project` holds `season_eff[data.season[end]]`
fixed across the forecast horizon -- the current season's POSTERIOR
`severity_mult[data.season[end]]` value is reused unchanged for every
forecast week; there is no future season to draw a fresh severity
effect for within this horizon.

For each horizon `h = 1:maximum(horizons)`:

- **Seasonality** for future week-of-season `w = mod1(data.woy[end] +
  h, data.W)` is rebuilt as `mu0 + severity_mult[s] * (mu_w[w] +
  delta[w, l]) + season_eff[s]`, with `s = data.season[end]`.
- **Residual** is forward-simulated as an AR(1) recursion (or, when
  `difference` is `true`, an integrated random walk), seeded from the
  last training-week residual `residual[end, l]`, exactly as
  `base_project`. Innovations are drawn fresh each step (not decayed)
  so predictive intervals stay calibrated.

No backfill revision is applied to future weeks (revision -> 0),
matching `base_project` and the observed-vs-forecast distinction in
docs/contracts.md. `difference` must match how `model_severity` was
fit (its default is `false`, matching `base_model`'s default).
"""
function project_severity(draw, data::ModelData, horizons;
                           difference::Bool=false)
    L = data.L
    H = maximum(horizons)

    mu0 = _field_severity(draw, :mu0)
    mu_w = _field_severity(draw, :mu_w)                    # (W,)
    delta = _field_severity(draw, :delta)                  # (W x L)
    season_eff = _field_severity(draw, :season_eff)        # (S,)
    severity_mult = _field_severity(draw, :severity_mult)  # (S,)
    s = data.season[end]
    seff = season_eff[s]
    sev = severity_mult[s]

    phi = _field_severity(draw, :phi)                      # (L,)
    sigma_ar = _field_severity(draw, :sigma_ar)            # (L,)
    residual = _field_severity(draw, :residual)            # (T x L)
    resid = Float64[residual[end, l] for l in 1:L]

    latent = Matrix{Float64}(undef, L, H)
    for h in 1:H
        w = mod1(data.woy[end] + h, data.W)
        for l in 1:L
            innov = sigma_ar[l] * randn()
            resid[l] = difference ? resid[l] + innov :
                       phi[l] * resid[l] + innov
            latent[l, h] = mu0 + sev * (mu_w[w] + delta[w, l]) +
                           seff + resid[l]
        end
    end
    return latent
end
