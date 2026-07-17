# Forecast projection for round 2 candidate mvn-innov (`model_mvn_innov`,
# see model_mvn_innov.jl in this directory).
#
# `model_mvn_innov` changes only how the post-seasonal AR(1) innovation
# is drawn (correlated across locations at each time step via `Lcorr`,
# in place of base_model's independent per-location innovation); the
# per-location AR(1) LEVEL recursion (`phi[l]`) and the seasonal
# reconstruction it feeds are otherwise identical to base_model. So
# this mirrors src/forecast.jl's `base_project` exactly except the
# innovation drawn at each forecast step is a correlated `L`-vector
# (via `Lcorr`/`sigma_ar`) rather than `L` independent scalars, and is
# included here standalone (rather than reused from src/forecast.jl)
# so the variant stays self-contained. Assumes src/core.jl's
# `ModelData` is in scope.

using LinearAlgebra
using Statistics

# Read a field from a draw that may be a NamedTuple or a Dict{Symbol}.
# Local, differently-named copy of src/forecast.jl's `_field` so this
# file has no load-order dependency on src/forecast.jl.
_field_mvn(draw, key::Symbol) =
    draw isa AbstractDict ? draw[key] : getproperty(draw, key)

"""
    project_mvn_innov(draw, data::ModelData, horizons; difference=false)

`project` function bridging one `model_mvn_innov` posterior draw (a
generated-quantities `NamedTuple`, e.g. from `generated_draws` in
`src/inference.jl`) to the `(L x maximum(horizons))` latent-scale
forecast matrix that `forecast_quantiles` (`src/forecast.jl`) consumes.

The draw must expose the fields `model_mvn_innov` returns: `mu0`,
`mu_w` (length `W`), `delta` (`W x L`), `season_eff` (length `S`),
`phi` (`L`, per-location AR(1) coefficient -- level dynamics stay
per-location, unlike `project_var`'s `A`), `sigma_ar` (`L`,
per-location innovation scale), `Lcorr` (`L x L` lower-triangular
correlation Cholesky factor), and `residual` (`T x L`).

For each horizon `h = 1:maximum(horizons)`:

- **Seasonality** for future week-of-season `w = mod1(data.woy[end] +
  h, data.W)` is rebuilt as `mu0 + mu_w[w] + delta[w, l] + season_eff
  [s]`, with `s = data.season[end]` (the current season's level shift
  held for the short within-season horizons, as in `base_project`).
- **Residual** is forward-simulated per location as `resid_h[l] =
  phi[l] * resid_{h-1}[l] + innov_h[l]` (or `resid_{h-1}[l] +
  innov_h[l]` when `difference` is `true`, matching `model_mvn_innov`'s
  `difference` switch), where `innov_h = (Diagonal(sigma_ar) * Lcorr) *
  randn(L)` is drawn FRESH each step as one correlated `L`-vector (not
  `L` independent draws), seeded from the last training-week residual
  vector `residual[end, :]`. Fresh (not decayed) innovations keep
  predictive intervals calibrated; each call gives a fresh
  posterior-predictive realisation for the draw.

No backfill revision is applied to future weeks (revision -> 0),
matching `base_project` and the observed-vs-forecast distinction in
docs/contracts.md.
"""
function project_mvn_innov(draw, data::ModelData, horizons;
                            difference::Bool=false)
    L = data.L
    H = maximum(horizons)
    mu0 = _field_mvn(draw, :mu0)
    mu_w = _field_mvn(draw, :mu_w)
    delta = _field_mvn(draw, :delta)          # (W x L)
    season_eff = _field_mvn(draw, :season_eff)
    phi = _field_mvn(draw, :phi)              # (L,)
    sigma_ar = _field_mvn(draw, :sigma_ar)    # (L,)
    Lcorr = _field_mvn(draw, :Lcorr)          # (L x L)
    residual = _field_mvn(draw, :residual)    # (T x L)
    seff = season_eff[data.season[end]]
    scaled_L = Diagonal(sigma_ar) * Lcorr
    resid = Float64[residual[end, l] for l in 1:L]
    latent = Matrix{Float64}(undef, L, H)
    for h in 1:H
        w = mod1(data.woy[end] + h, data.W)
        innov = scaled_L * randn(L)
        resid = difference ? resid .+ innov : phi .* resid .+ innov
        for l in 1:L
            latent[l, h] = mu0 + mu_w[w] + delta[w, l] + seff + resid[l]
        end
    end
    return latent
end
