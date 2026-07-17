# Round 1 candidate v3-diff: the DIFFERENCING variant of the base
# joint model. See src/model.jl for `base_model`, `ar_or_diff`, and
# `backfill_profile`, which this file reuses unchanged.
#
# The only structural change from the base model is which branch of
# `ar_or_diff` builds the post-seasonal residual: `base_model` defaults
# to a stationary AR(1) level (`difference=false`); this candidate
# fixes `difference=true`, so the residual is an integrated random
# walk and it is the residual's *first difference* that is modelled as
# white noise (`sigma_ar`-scaled innovations), not the level itself.
# Seasonality (partially-pooled week-of-season curve) and backfill
# (non-monotonic delay-indexed profile) are untouched.
#
# docs/eda/05-autocorrelation.md flags this as a provisional,
# location-varying question: differencing the deseasonalised residual
# shows a negative lag-1 ACF (an over-differencing signature) at most
# locations, but HHS Region 4 and Region 5 show a small *positive*
# lag-1 instead, so a hard-coded "AR everywhere" choice may be wrong
# for those two. This candidate takes the opposite bet from the base
# model everywhere (differencing for all locations) so it can be
# scored against `base_model` per-location; it is not claimed to beat
# AR(1) globally.
#
# Assumes `src/core.jl` and `src/model.jl` are already loaded (this
# file only adds a thin wrapper around `base_model`, it does not
# redefine `ar_or_diff`/`backfill_profile`/`model_dims`).

"""
    model_v3(d::ModelData; transform=:log1p)

Round 1 candidate v3-diff: `base_model` with `difference=true`, i.e.
the post-seasonal residual is an integrated random walk (first
difference modelled as white noise) rather than a stationary AR(1)
level. See `base_model`'s docstring in `src/model.jl` for the full
model description (seasonality, backfill, priors); this wrapper only
fixes the `difference` switch and the default `transform`.

`transform` defaults to `:log1p` here (candidate default for this
branch) rather than `base_model`'s `:log`, but is otherwise threaded
through unchanged; it is not used inside the model itself (`d.Y` is
already on the modelling scale) — see `base_model`'s docstring.

Returns the same `NamedTuple` shape as `base_model` (`latent`,
`seasonal`, `residual`, `mu0`, `mu_w`, `delta`, `season_eff`, `phi`,
`sigma_ar`, `r`, `r_pop`, `sigma_obs`, `transform`). `phi` is still
sampled (from `phi_pop_mean`/`phi_pop_sd`/`phi_raw`) and returned
as-is, but it plays no role in building `residual` under
`difference=true` — see `ar_or_diff`.
"""
model_v3(d::ModelData; transform::Symbol=:log1p) =
    base_model(d; transform=transform, difference=true)
