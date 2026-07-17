# Round 1 candidate v3-diff: `project` function for `model_v3`
# (differencing residual). See src/forecast.jl for `default_project`
# and `base_project`, the AR(1) analogue this mirrors, and for
# `_field`, which this file reuses to read NamedTuple-or-Dict draws.
#
# Assumes `src/core.jl` and `src/forecast.jl` are already loaded (for
# `ModelData`, `_field`).

"""
    project_v3(draw, data::ModelData, horizons)

`project` function bridging one `model_v3` (differencing) posterior
draw — a generated-quantities `NamedTuple` from `generated_draws`,
see `src/inference.jl` — to the `(L × maximum(horizons))` latent-scale
forecast matrix that `forecast_quantiles` consumes.

The draw must expose `model_v3`/`base_model`'s return fields `mu0`,
`mu_w` (length `W`), `delta` (`W × L`), `season_eff` (length `S`),
`sigma_ar` (`L`), and `residual` (`T × L`) — the same raw sampled
sites `base_project` reads, recomputed forward from scratch each call
rather than read off the fixed-length `latent`/`residual` training
paths. `phi` is *not* read: under `difference=true` the residual is an
integrated random walk (see `ar_or_diff` in `src/model.jl`), so there
is no AR coefficient to apply going forward, only the innovation sd.

For each horizon `h = 1:maximum(horizons)`:

- **Seasonality** for future week-of-season `w = mod1(data.woy[end] +
  h, data.W)` is rebuilt as `mu0 + mu_w[w] + delta[w, l] +
  season_eff[s]`, with `s = data.season[end]` (current season's level
  shift held for the short within-season horizon) — identical to
  `base_project`.
- **Residual** is forward-simulated as a random walk, continuing the
  training-time recursion: `resid_h = resid_{h-1} + sigma_ar *
  randn()`, seeded from the last training-week residual
  `residual[end, l]`. There is no mean reversion (no `phi` term), so
  the walk can drift away from zero; this is the deliberate cost of
  the differencing branch (see docs/eda/05-autocorrelation.md's
  over-differencing note) and is why the prior-predictive check in
  `check_v3.jl` watches for drift blowing up the plausible wILI%
  range. Innovations are drawn fresh each call (not decayed), so
  repeated calls with the same `draw` give different
  posterior-predictive realisations, matching `base_project`.

No backfill revision is applied to future weeks (revision -> 0),
matching `base_project` and the observed-vs-forecast distinction in
docs/contracts.md.
"""
function project_v3(draw, data::ModelData, horizons)
    L = data.L
    H = maximum(horizons)
    mu0 = _field(draw, :mu0)
    mu_w = _field(draw, :mu_w)
    delta = _field(draw, :delta)          # (W x L)
    season_eff = _field(draw, :season_eff)
    sigma_ar = _field(draw, :sigma_ar)    # (L,), innovation sd
    residual = _field(draw, :residual)    # (T x L)
    seff = season_eff[data.season[end]]
    resid = Float64[residual[end, l] for l in 1:L]
    latent = Matrix{Float64}(undef, L, H)
    for h in 1:H
        w = mod1(data.woy[end] + h, data.W)
        for l in 1:L
            resid[l] = resid[l] + sigma_ar[l] * randn()
            latent[l, h] = mu0 + mu_w[w] + delta[w, l] + seff + resid[l]
        end
    end
    return latent
end
