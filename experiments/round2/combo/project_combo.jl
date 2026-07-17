# Forecast projection for the round 2 combo candidate (`model_combo`,
# see model_combo.jl in this directory).
#
# Combines the two projection-relevant pieces `model_combo` changed
# relative to `base_model`: `project_severity`'s severity-scaled
# seasonality (experiments/round2/severity/project_severity.jl) and
# `project_v1`'s AR(p) forward simulation
# (experiments/round1/v1-ar-high/project_v1.jl). The third piece,
# season-varying backfill (`model_season_backfill`'s `r`/`r_pop`/
# `r_season`), is NOT used here: as in every other candidate in this
# search, no backfill revision is applied to future weeks (revision ->
# 0), matching the observed-vs-forecast distinction in
# docs/contracts.md -- `model_combo`'s season-varying `r` only affects
# the fit/nowcast of already-observed weeks. Self-contained: does not
# depend on src/forecast.jl or the other experiments' project files.

using Statistics

"""
    _field_combo(draw, key)

Read a field from a draw that may be a `NamedTuple` or a
`Dict{Symbol}`. Own-namespaced copy of `src/forecast.jl`'s `_field`, as
every other experiment's project file does, so this file has no
load-order dependency on `src/forecast.jl`.
"""
_field_combo(draw, key::Symbol) =
    draw isa AbstractDict ? draw[key] : getproperty(draw, key)

"""
    project_combo(draw, data::ModelData, horizons)

`project` function bridging one `model_combo` posterior draw (a
generated-quantities `NamedTuple`, see `generated_draws` in
src/inference.jl) to the `(L x maximum(horizons))` latent-scale
forecast matrix that `forecast_quantiles` (src/forecast.jl) consumes.

The draw must expose `model_combo`'s return fields `mu0`, `mu_w`
(length `W`), `delta` (`W x L`), `season_eff` (length `S`),
`severity_mult` (length `S`), `phi` (`L x p`, the AR(p) coefficients
per location), `sigma_ar` (`L`), and `residual` (`T x L`). The AR order
`p` is read from `size(phi, 2)`, exactly as `project_v1` does, so this
works for whatever `p` `model_combo` was built with.

For each horizon `h = 1:maximum(horizons)`:

- **Seasonality** for future week-of-season `w = mod1(data.woy[end] +
  h, data.W)` is rebuilt as `mu0 + severity_mult[s] * (mu_w[w] +
  delta[w, l]) + season_eff[s]`, with `s = data.season[end]` (the
  current season's posterior `severity_mult` and `season_eff` both
  held fixed across the short within-season horizon, exactly as
  `project_severity` and `base_project` do).
- **Residual** is forward-simulated as an AR(p) recursion, seeded from
  the last `p` training-week residuals per location
  (`residual[end-p+1:end, l]`), drawing fresh innovations
  `Normal(0, sigma_ar[l])` at each step (not decayed), exactly as
  `project_v1`.

No backfill revision is applied to future weeks (revision -> 0).
"""
function project_combo(draw, data::ModelData, horizons)
    L = data.L
    H = maximum(horizons)

    mu0 = _field_combo(draw, :mu0)
    mu_w = _field_combo(draw, :mu_w)                    # (W,)
    delta = _field_combo(draw, :delta)                  # (W x L)
    season_eff = _field_combo(draw, :season_eff)        # (S,)
    severity_mult = _field_combo(draw, :severity_mult)  # (S,)
    s = data.season[end]
    seff = season_eff[s]
    sev = severity_mult[s]

    phi = _field_combo(draw, :phi)                      # (L x p)
    sigma_ar = _field_combo(draw, :sigma_ar)            # (L,)
    residual = _field_combo(draw, :residual)            # (T x L)
    p = size(phi, 2)
    T = size(residual, 1)

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
            latent[l, h] = mu0 + sev * (mu_w[w] + delta[w, l]) +
                           seff + newval
        end
    end
    return latent
end
