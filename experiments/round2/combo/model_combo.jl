# Round 2 candidate combo: a "kitchen sink of what the EDA supports",
# combining the three most-promising EDA-backed axes from the search so
# far into ONE model on top of `base_model`'s (src/model.jl) partially-
# pooled seasonality:
#
#   - higher-order per-location AR(p) residual, PACF-parameterised, from
#     round 1's v1-ar-high (experiments/round1/v1-ar-high/model_v1.jl),
#     motivated by docs/eda/05-autocorrelation.md (AIC-selected AR order
#     median 5, PACF cutting off sharply after lag 1).
#   - a per-season "severity" amplitude multiplier on the seasonal
#     SHAPE, from round 2's severity
#     (experiments/round2/severity/model_severity.jl), motivated by
#     docs/eda/04-cross-location.md (season-level amplitude far more
#     correlated across locations, mean 0.68, than week-to-week moves,
#     mean 0.24).
#   - a season-level deviation on the delay-indexed backfill revision,
#     from round 2's season-backfill
#     (experiments/round2/season-backfill/model_season_backfill.jl),
#     motivated by docs/eda/02-backfill.md (Region 1 and Region 4 fully
#     reverse the sign of their delay-1 revision bias between tracked
#     seasons).
#
# This is deliberately a combination candidate to test whether the
# pieces' individual gains stack, or instead compound overfitting risk
# (this candidate is scored on WIS AND WIS SD, same as every other
# round 2 candidate -- see docs/plan.md). It does not introduce any new
# mechanism beyond the three above: nothing here is a novel idea, only
# their union.
#
# This file assumes `src/core.jl` (`ModelData`), `src/model.jl`
# (`model_dims`, `backfill_profile`), and v1-ar-high's `model_v1.jl`
# (`pacf_to_ar`, `ar_p`) are already `include`d into scope -- see
# check_combo.jl for the load order. It does NOT reuse
# `src/model.jl`'s `ar_or_diff` (superseded here by `ar_p`) or
# `observation_index` (superseded by `observation_index_seasonal`
# below, which needs a season-indexed `r`).

using Turing
using Distributions
using LinearAlgebra
using Statistics

"""
    observation_index_seasonal(d::ModelData)

Like `src/model.jl`'s `observation_index`, but for a season-varying
backfill array `r` of shape `(Dmax+1) x L x S` (delay x location x
season) rather than `(Dmax+1) x L` -- the shape `model_combo`'s
combined backfill needs (see `model_season_backfill.jl`, whose
observation loop this vectorises the same way `observation_index`
vectorises `base_model`'s).

Returns a `NamedTuple` `(obs_idx, r_idx, yobs)`:

- `obs_idx`: `CartesianIndex{2}` vector into `d.Y` (shape `T x L`) for
  every non-missing cell, exactly as `observation_index` returns.
- `r_idx`: `CartesianIndex{3}` vector into `r` (shape `Dmax+1 x L x
  S`), i.e. `(d.delay[i] + 1, i[2], d.season[i[1]])` for each `i` in
  `obs_idx` -- `i[1]` is the time index `t`, so `d.season[i[1]]` is
  that cell's training season.
- `yobs`: the gathered observed values as a concrete `Vector{Float64}`.

Evaluated once (default keyword argument), not re-derived every
log-density/gradient evaluation, exactly as `observation_index` is.
"""
function observation_index_seasonal(d::ModelData)
    obs_idx = findall(!ismissing, d.Y)
    yobs = Float64.(d.Y[obs_idx])
    r_idx = [CartesianIndex(d.delay[i] + 1, i[2], d.season[i[1]])
             for i in obs_idx]
    return (; obs_idx, r_idx, yobs)
end

"""
    model_combo(d::ModelData; transform=:log, p=5,
                obsdata=observation_index_seasonal(d))

Round 2 combination candidate: `base_model`'s partially-pooled
week-of-season seasonality, PLUS all three axes below layered on top of
it simultaneously.

# What changed vs `base_model`

- **Seasonality**: unchanged population curve (`mu_w`, non-centred
  random walk) and per-location deviation (`delta`), but the SHAPE
  `mu_w[woy] + delta[woy, l]` is now scaled by a per-season, always-
  positive `severity_mult[season]` before the additive `season_eff`
  level shift is added (identical mechanism to `model_severity`):
  `seasonal[t, l] = mu0 + severity_mult[season[t]] *
  (mu_w[woy[t]] + delta[woy[t], l]) + season_eff[season[t]]`.
- **Residual**: `base_model`'s per-location AR(1) is replaced by a
  partially-pooled AR(`p`) built from per-location, per-lag partial
  autocorrelations (identical mechanism to `model_v1`, including the
  fixed `0.6^(k-1)` lag-decay tightening the pooling sd as lag order
  grows, and the `pacf_to_ar` Durbin-Levinson map that guarantees
  stationarity for every draw). `p` is a fixed keyword argument
  (default 5, the EDA's median selected order), not sampled.
- **Backfill**: `base_model`'s non-monotonic delay-indexed revision
  profile gains a season-level deviation on top of the existing
  per-location one (identical mechanism to `model_season_backfill`):
  `r[delay, l, s] = r_pop[delay] + r_loc[delay, l] + r_season[delay,
  s]`, built by broadcasting the three terms into a `(Dmax+1) x L x S`
  array. The observation likelihood is vectorised over
  [`observation_index_seasonal`](@ref)'s precomputed non-missing cells
  (a single `arraydist` observe, not a scalar loop per cell), matching
  `base_model`'s and `model_severity`'s vectorised approach rather than
  `model_season_backfill`'s original scalar loop -- Mooncake traces one
  observe statement per gradient step either way, but this keeps the
  combo model's cost from also scaling with the manual-loop overhead
  `docs/lessons.md` #5 flags as a `base_model` follow-up.

None of the three mechanisms interact with each other beyond sharing
`d.season`/`d.woy`/`d.delay`: `severity_mult` only touches `seasonal`,
`phi`/`sigma_ar`/`eps_raw` only touch `residual`, and `r_season` only
touches the backfill `r` array -- exactly their individual behaviour in
`model_severity`/`model_v1`/`model_season_backfill` respectively,
simply summed/multiplied into `base_model`'s structure rather than
re-derived.

# Parameter count

Relative to `base_model` at the same `L`, `W`, `S`, `Dmax` (roughly
1,959 free parameters at `L=11, W~52, S=2, Dmax=12, window_weeks=104`,
per `docs/lessons.md` #5's accounting), this candidate adds:

- severity: `mu_severity` + `sigma_severity` + `severity_raw` (`S`) =
  `2 + S` (4 at `S=2`).
- AR(p) vs AR(1): `mu_pacf` (`p`) + `sigma_pacf_pop` + `pacf_raw`
  (`L*p`) replaces `phi_pop_mean` + `phi_pop_sd` + `phi_raw` (`L`), a
  net `p + 1 + L*p - 2 - L` (48 at `L=11, p=5`).
- season-backfill: `sigma_r_season` + `r_season_raw` (`(Dmax+1)*S`) =
  `1 + (Dmax+1)*S` (27 at `Dmax=12, S=2`).

Total addition ~79 parameters (~4%) at that sizing -- a modest
increase given three mechanisms are added at once, since none of them
scales with `T` or `W` (the two dominant blocks, `eps_raw` and
`delta_raw`, are untouched).

# Returns

A `NamedTuple` `(latent, seasonal, residual, mu0, mu_w, delta,
season_eff, severity_mult, mu_severity, sigma_severity, phi, sigma_ar,
r, r_pop, r_loc, r_season, sigma_obs, transform)`. `phi` is `(L x p)`
(the AR(p) coefficients, as in `model_v1`) and `r` is `(Dmax+1) x L x
S` (as in `model_season_backfill`); every other field matches
`base_model`'s field-for-field. See `project_combo.jl` for the matching
forecast projection (severity-scaled seasonality + AR(p) forward
simulation; no backfill revision applied to future weeks, as always).
"""
@model function model_combo(d::ModelData; transform::Symbol=:log,
                             p::Int=5,
                             obsdata=observation_index_seasonal(d))
    T, L, W, S, Dmax = model_dims(d)
    obs_idx, r_idx, yobs = obsdata.obs_idx, obsdata.r_idx, obsdata.yobs

    # --- Seasonality: partially-pooled week-of-season + severity amplitude ---
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

    mu_severity ~ Normal(0, 0.3)
    sigma_severity ~ truncated(Normal(0, 0.3); lower=0)
    severity_raw ~ filldist(Normal(0, 1), S)
    severity_mult = exp.(mu_severity .+ sigma_severity .* severity_raw)

    shape = mu_w[d.woy] .+ delta[d.woy, :]                  # (T x L)
    seasonal = mu0 .+ severity_mult[d.season] .* shape .+ season_eff[d.season]

    # --- Post-seasonal residual: partially-pooled AR(p) via PACF ---
    mu_pacf ~ filldist(Normal(0, 1), p)
    sigma_pacf_pop ~ truncated(Normal(0, 0.5); lower=0)
    pacf_raw ~ filldist(Normal(0, 1), L, p)
    lag_decay = [0.6^(k - 1) for k in 1:p]
    pacf_pre = mu_pacf' .+ (sigma_pacf_pop .* lag_decay') .* pacf_raw
    pacf = tanh.(pacf_pre)                              # (L x p), in (-1, 1)

    phi = permutedims(reduce(hcat,
        [pacf_to_ar(view(pacf, l, :)) for l in 1:L]))   # (L x p)

    mu_log_sigma_ar ~ Normal(log(0.2), 1)
    tau_log_sigma_ar ~ truncated(Normal(0, 0.5); lower=0)
    z_sigma_ar ~ filldist(Normal(0, 1), L)
    sigma_ar = exp.(mu_log_sigma_ar .+ tau_log_sigma_ar .* z_sigma_ar)

    eps_raw ~ filldist(Normal(0, 1), T, L)
    residual = reduce(hcat, [
        ar_p(view(eps_raw, :, l), sigma_ar[l], view(phi, l, :))
        for l in 1:L
    ])

    latent = seasonal .+ residual

    # --- Backfill: non-monotonic, location- AND season-varying ---
    r_pop_anchor ~ Normal(0, 0.05)
    sigma_r_pop ~ truncated(Normal(0, 0.3); lower=0)
    r_steps_raw ~ filldist(Normal(0, 1), Dmax)
    r_pop = backfill_profile(r_pop_anchor, r_steps_raw .* sigma_r_pop)

    sigma_r_loc ~ truncated(Normal(0, 0.3); lower=0)
    r_loc_raw ~ filldist(Normal(0, 1), Dmax + 1, L)
    r_loc = r_loc_raw .* sigma_r_loc

    sigma_r_season ~ truncated(Normal(0, 0.3); lower=0)
    r_season_raw ~ filldist(Normal(0, 1), Dmax + 1, S)
    r_season = r_season_raw .* sigma_r_season

    # (Dmax+1) x L x S, additive: population + location + season.
    r = reshape(r_pop, Dmax + 1, 1, 1) .+
        reshape(r_loc, Dmax + 1, L, 1) .+
        reshape(r_season, Dmax + 1, 1, S)

    sigma_obs ~ truncated(Normal(0, 1); lower=0)

    mu_obs = latent[obs_idx] .+ r[r_idx]
    yobs ~ arraydist(Normal.(mu_obs, sigma_obs))

    return (;
        latent, seasonal, residual, mu0, mu_w, delta, season_eff,
        severity_mult, mu_severity, sigma_severity,
        phi, sigma_ar, r, r_pop, r_loc, r_season, sigma_obs, transform,
    )
end
