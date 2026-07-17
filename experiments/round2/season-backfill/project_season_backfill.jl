# Round 2 candidate season-backfill: forward projection.
# Mirrors `base_project` (src/forecast.jl) exactly: model_season_
# backfill's seasonal + AR(1)/difference dynamics are unchanged from
# base_model, and the new season-varying backfill only affects the
# fit/nowcast of already-observed weeks, not future ones (revision ->
# 0 beyond the forecast origin, matching docs/contracts.md's observed-
# vs-forecast distinction). This file is self-contained: it does not
# depend on src/forecast.jl.

"""
    project_season_backfill(draw, d, horizons; difference=false)

`project` function bridging one `model_season_backfill` posterior draw
(a generated-quantities `NamedTuple`, see `generated_draws` in
src/inference.jl) to the `(L x maximum(horizons))` latent-scale
forecast matrix that `forecast_quantiles` consumes.

Identical in behaviour to `base_project`: forward-simulates the
per-location AR(1) (or differenced) residual from the last
training-week residual, and adds back the partially-pooled seasonal
curve for each future week-of-season. The draw must expose `mu0`,
`mu_w` (length `W`), `delta` (`W x L`), `season_eff` (length `S`),
`phi` (`L`), `sigma_ar` (`L`), and `residual` (`T x L`) -- exactly the
fields `model_season_backfill` returns alongside its season-varying
`r`/`r_pop`/`r_season` backfill fields, which this function does not
use: no backfill revision is applied to future weeks. `difference`
must match how the model was fit (default `false`, matching
`model_season_backfill`'s default).
"""
function project_season_backfill(
    draw, d, horizons; difference::Bool=false,
)
    L = d.L
    H = maximum(horizons)
    mu0 = _field_season(draw, :mu0)
    mu_w = _field_season(draw, :mu_w)
    delta = _field_season(draw, :delta)          # (W x L)
    season_eff = _field_season(draw, :season_eff)
    phi = _field_season(draw, :phi)              # (L,)
    sigma_ar = _field_season(draw, :sigma_ar)    # (L,)
    residual = _field_season(draw, :residual)    # (T x L)
    seff = season_eff[d.season[end]]
    resid = Float64[residual[end, l] for l in 1:L]
    latent = Matrix{Float64}(undef, L, H)
    for h in 1:H
        w = mod1(d.woy[end] + h, d.W)
        for l in 1:L
            innov = sigma_ar[l] * randn()
            resid[l] = difference ? resid[l] + innov : phi[l] * resid[l] + innov
            latent[l, h] = mu0 + mu_w[w] + delta[w, l] + seff + resid[l]
        end
    end
    return latent
end

# Read a field from a draw that may be a NamedTuple or a Dict{Symbol}.
# Duplicated from src/forecast.jl's `_field` (own-namespaced here so
# this file stays self-contained per this experiment's ownership).
_field_season(draw, key::Symbol) =
    draw isa AbstractDict ? draw[key] : getproperty(draw, key)
