# Forecast projection for round 2 candidate var (`model_var`, see
# model_var.jl in this directory).
#
# `model_var` changes only how the post-seasonal residual evolves (a
# VAR(1) across locations via the `L x L` matrix `A`, in place of
# base_model's independent per-location AR(1)); the seasonal
# reconstruction it feeds is otherwise identical to base_model. So this
# mirrors src/forecast.jl's `base_project` exactly except the residual
# recursion is a matrix-vector VAR(1) step instead of `L` independent
# scalar AR(1) steps, and is included here standalone (rather than
# reused from src/forecast.jl) so the variant stays self-contained.
# Assumes src/core.jl's `ModelData` is in scope.

using Statistics

# Read a field from a draw that may be a NamedTuple or a Dict{Symbol}.
# Local, differently-named copy of src/forecast.jl's `_field` so this
# file has no load-order dependency on src/forecast.jl.
_field_var(draw, key::Symbol) =
    draw isa AbstractDict ? draw[key] : getproperty(draw, key)

"""
    project_var(draw, data::ModelData, horizons)

`project` function bridging one `model_var` posterior draw (a
generated-quantities `NamedTuple`, e.g. from `generated_draws` in
`src/inference.jl`) to the `(L x maximum(horizons))` latent-scale
forecast matrix that `forecast_quantiles` (`src/forecast.jl`) consumes.

The draw must expose the fields `model_var` returns: `mu0`, `mu_w`
(length `W`), `delta` (`W x L`), `season_eff` (length `S`), `A` (the
`L x L` VAR(1) transition matrix -- NOT `phi` alone, which is only
`A`'s diagonal and would silently drop the cross-location coupling),
`sigma_ar` (`L`, per-location innovation sd, independent across
locations), and `residual` (`T x L`).

For each horizon `h = 1:maximum(horizons)`:

- **Seasonality** for future week-of-season `w = mod1(data.woy[end] +
  h, data.W)` is rebuilt as `mu0 + mu_w[w] + delta[w, l] + season_eff
  [s]`, with `s = data.season[end]` (the current season's level shift
  held for the short within-season horizons, as in `base_project`).
- **Residual** is forward-simulated as the VAR(1) recursion `resid_h =
  A * resid_{h-1} + sigma_ar .* randn(L)`, seeded from the last
  training-week residual vector `residual[end, :]`. Innovations are
  independent across locations and drawn fresh each step (not decayed)
  so predictive intervals stay calibrated; each call gives a fresh
  posterior-predictive realisation for the draw.

No backfill revision is applied to future weeks (revision -> 0),
matching `base_project` and the observed-vs-forecast distinction in
docs/contracts.md.
"""
function project_var(draw, data::ModelData, horizons)
    L = data.L
    H = maximum(horizons)
    mu0 = _field_var(draw, :mu0)
    mu_w = _field_var(draw, :mu_w)
    delta = _field_var(draw, :delta)          # (W x L)
    season_eff = _field_var(draw, :season_eff)
    A = _field_var(draw, :A)                  # (L x L)
    sigma_ar = _field_var(draw, :sigma_ar)    # (L,)
    residual = _field_var(draw, :residual)    # (T x L)
    seff = season_eff[data.season[end]]
    resid = Float64[residual[end, l] for l in 1:L]
    latent = Matrix{Float64}(undef, L, H)
    for h in 1:H
        w = mod1(data.woy[end] + h, data.W)
        resid = A * resid .+ sigma_ar .* randn(L)
        for l in 1:L
            latent[l, h] = mu0 + mu_w[w] + delta[w, l] + seff + resid[l]
        end
    end
    return latent
end
