# Forecast projection for the round-2 candidate ar-loc.
#
# Structurally identical to `project_v1` (experiments/round1/v1-ar-high/
# project_v1.jl): forward-simulates the AR(p) recursion seeded from the
# last `p` training-week residuals per location. `model_ar_loc` gives
# each location its own EFFECTIVE order via the location-specific PACF
# decay rate `rho[l]` (see model_ar_loc.jl), but that only changes how
# `phi` is built inside the model, not the shape of the fields this
# projection consumes: `phi` is still `(L x Pmax)` (all locations share
# the same ceiling order `Pmax`, with the low-order ones' unneeded
# high-lag coefficients shrunk near zero), so the same per-location AR
# forward recursion applies unchanged, with `p` read from `size(phi,
# 2)` (`Pmax` here rather than v1-ar-high's shared `p`).
#
# Plugs into `forecast_quantiles(fit, d, id; project=project_ar_loc)`.

"""
    _field(draw, key)

Read a field from a draw that may be a `NamedTuple` or a
`Dict{Symbol}`. Duplicated from `src/forecast.jl` so this file stays
loadable on its own (see check_ar_loc.jl); behaviour is identical.
"""
_field(draw, key::Symbol) =
    draw isa AbstractDict ? draw[key] : getproperty(draw, key)

"""
    project_ar_loc(draw, data, horizons)

`project` function bridging one `model_ar_loc` posterior draw (a
generated-quantities `NamedTuple` from `generated_draws`, see
src/inference.jl) to the `(L x maximum(horizons))` latent-scale
forecast matrix that [`forecast_quantiles`](@ref) (src/forecast.jl)
consumes.

The draw must expose `model_ar_loc`'s return fields `mu0`, `mu_w`
(length `W`), `delta` (`W x L`), `season_eff` (length `S`), `phi`
(`L x Pmax`, each location's AR coefficients, already reflecting that
location's effective order via its own PACF decay rate), `sigma_ar`
(`L`), and `residual` (`T x L`). The ceiling order `Pmax` is read from
`size(phi, 2)`, so this works for whatever `Pmax` `model_ar_loc` was
built with — no separate order argument needed. `rho` (the
per-location decay rate) is NOT needed here: it only shapes `phi`
inside the model, and `phi` already carries its effect forward.

For each horizon `h = 1:maximum(horizons)`:

- **Seasonality** is rebuilt exactly as in `base_project`: future
  week-of-season `w = mod1(data.woy[end] + h, data.W)`, as
  `mu0 + mu_w[w] + delta[w, l] + season_eff[s]`, with `s =
  data.season[end]` (current season's level shift held for the short
  within-season horizons).
- **Residual** is forward-simulated as an AR(Pmax) recursion per
  location, seeded from the last `Pmax` training-week residuals
  (`residual[end-Pmax+1:end, l]`), drawing fresh innovations
  `Normal(0, sigma_ar[l])` at each step (not decayed), so repeated
  calls with the same `draw` give different posterior-predictive
  realisations, matching `base_project`/`project_v1`.

No backfill revision is applied to future weeks (revision -> 0),
matching the observed-vs-forecast distinction in docs/contracts.md.
"""
function project_ar_loc(draw, data::ModelData, horizons)
    L = data.L
    H = maximum(horizons)
    mu0 = _field(draw, :mu0)
    mu_w = _field(draw, :mu_w)
    delta = _field(draw, :delta)          # (W x L)
    season_eff = _field(draw, :season_eff)
    phi = _field(draw, :phi)              # (L x Pmax)
    sigma_ar = _field(draw, :sigma_ar)    # (L,)
    residual = _field(draw, :residual)    # (T x L)
    p = size(phi, 2)
    T = size(residual, 1)
    seff = season_eff[data.season[end]]

    # Per-location sliding window of the last p residuals, oldest first.
    hist = [Float64[residual[T - p + k, l] for k in 1:p] for l in 1:L]

    latent = Matrix{Float64}(undef, L, H)
    for h in 1:H
        w = mod1(data.woy[end] + h, data.W)
        for l in 1:L
            past = hist[l]
            newval = sum(phi[l, k] * past[p - k + 1] for k in 1:p) +
                     sigma_ar[l] * randn()
            hist[l] = vcat(@view(past[2:end]), newval)
            latent[l, h] = mu0 + mu_w[w] + delta[w, l] + seff + newval
        end
    end
    return latent
end
