# Round 2 candidate var: a VAR(1) across the 11 locations for the
# post-seasonal residual, in place of base_model's (src/model.jl,
# `base_model`) INDEPENDENT per-location AR(1). Seasonality and
# backfill are unchanged: partially-pooled week-of-season population
# curve, iid partially-pooled location deviation, non-monotonic
# delay-indexed backfill.
#
# Motivation / caveat: docs/eda/04-cross-location.md finds moderate
# cross-location correlation on the differenced series (mean ~0.24, up
# to ~0.69 for closely related regions), but that coupling is
# CONTEMPORANEOUS (lag 0) with no lead-lag structure -- every region's
# cross-correlation against the national series peaks at lag 0, not
# +-1..3 weeks. The EDA's own "Implications for the model" section
# reads that as making a LAGGED VAR (this candidate) a lower-priority
# axis than a contemporaneous-correlation (MVN innovation) design, but
# flags full VAR as "a candidate to test rather than an obvious win" --
# this file is that test. If it under-performs a contemporaneous-MVN-
# innovation variant, that would be the EDA's contemporaneous-coupling
# reading confirmed empirically, not a failure of this file.
#
# This file assumes `src/core.jl` (for `ModelData`) and `src/model.jl`
# (for `model_dims`, `backfill_profile`, `observation_index`) are
# already `include`d into scope, exactly as `base_model` itself assumes
# for `core.jl`. It is a component file, not a package.

using Turing
using Distributions
using LinearAlgebra
using Statistics

"""
    model_var(d::ModelData; transform=:log, obsdata=observation_index(d))

Round 2 candidate `var`: identical to `base_model` except the
per-location post-seasonal residual is replaced by a VAR(1) across all
`L` locations: `residual[t, :] = A * residual[t-1, :] + innovation[t,
:]`, where `A` is an `L x L` transition matrix and `innovation[t, :]`
is independent across locations (no correlated-innovation term; that
is the separate, contemporaneous-coupling axis the EDA itself reads as
the higher-priority one, left to another candidate).

# `A` parameterisation: diagonal + shrunk rank-1

A full dense `L x L` transition matrix has `L^2` (121 at `L=11`) free
entries, most of them the lagged, cross-location terms the EDA reads
as the weakest-supported axis of coupling here -- too many parameters
to identify well against a training window of ~2 seasons. Instead `A`
is built as a per-location diagonal (the AR(1) persistence `base_model`
already has) plus a RANK-1, SHRUNK perturbation confined to the
off-diagonal:

- `phi` (length `L`): identical to `base_model` -- a partially-pooled,
  non-centred, `tanh`-mapped AR(1) coefficient per location.
- `u_raw`, `v_raw` (each length `L`): raw iid standard normals, UNIT-
  NORMALISED to `u_hat = u_raw / norm(u_raw)` (and likewise `v_hat`)
  before use. This normalisation is what keeps the perturbation's size
  from scaling up with `L`: an un-normalised rank-1 term `u_raw *
  v_raw'` from iid standard normals has induced 2-norm growing like
  `L` (`norm(u_raw) ~ sqrt(L)` on each factor), so at `L=11` it was
  observed EMPIRICALLY (while developing this file) to push the full
  `A`'s spectral radius past 1 for ~18% of prior draws even with a
  small `sigma_couple` -- unacceptably unstable. Normalising both
  factors to unit vectors makes the perturbation's own induced norm
  exactly `sigma_couple` regardless of `L`, which empirically brings
  instability down to ~1-2% at both `L=5` and `L=11` for the prior
  below (see `check_var.jl`'s spectral-radius check for the number on
  this file's actual dimensions).
- `sigma_couple ~ truncated(Normal(0, 0.15); lower=0)`: a single
  shrinkage scale pulling the whole rank-1 term toward zero a priori,
  so the model defaults close to `base_model`'s independent-AR(1)
  behaviour unless the data support cross-location coupling.
- `off_diag_mask` (`L x L`, `1` off-diagonal / `0` on it, a plain
  `Float64` constant built from `L`, not a sampled site) zeroes the
  rank-1 term's diagonal so it never doubles up with `phi`.
- `A = Diagonal(phi) + sigma_couple .* (u_hat * v_hat') .* off_diag_mask`.

This adds only `2L + 1` parameters over `base_model`'s `phi` block (23
at `L=11`), not `L^2`. `A`'s spectral radius is NOT algebraically
constrained to stay under 1 (unlike `phi`'s per-location `tanh` bound,
which does guarantee `|phi[l]| < 1` exactly); a handful of prior draws
do still land with spectral radius >= 1 (see the empirical rate
above), which is checked and tolerated -- not eliminated -- in
`check_var.jl`, matching the "keep it modest, not exhaustively
constrained" spirit of `base_model`'s own AR(1) and backfill priors.

# Residual recursion and initial condition

`innovation = eps_raw .* sigma_ar'` (`T x L`, independent across
locations, same partially-pooled log-normal `sigma_ar` as
`base_model`). The first row is seeded with a DIAGONAL-ONLY stationary-
variance approximation, `first_resid = innovation[1, :] ./ sqrt.(1 .-
phi.^2)` -- this ignores the off-diagonal contribution to the true
VAR(1) stationary covariance (an exact treatment needs a discrete
Lyapunov solve, `vec(Sigma0) = (I - A kron A)^-1 vec(Sigma_eps)`, which
is both expensive and an unnecessary complication for a training window
of ~100 weeks where one week's initial-variance mismatch washes out).
Subsequent rows follow the VAR(1) recursion `resid[t, :] = A *
resid[t-1, :] + innovation[t, :]`, built with `accumulate` over
explicit `view`s (matching `ar_or_diff`'s non-mutating style in
`src/model.jl`) rather than an in-place loop, so it stays Mooncake-AD-
friendly.

Everything else -- `mu0`, the population `mu_w` random walk, `delta`,
`season_eff`, and the non-monotonic backfill ([`backfill_profile`](@ref))
-- is copied verbatim from `base_model`, including its vectorised
observation likelihood over [`observation_index`](@ref)'s precomputed
non-missing cells.

Returns a `NamedTuple` with the same field names as `base_model`
(`latent, seasonal, residual, mu0, mu_w, delta, season_eff, phi,
sigma_ar, r, r_pop, sigma_obs, transform`) -- `phi` here is only the
diagonal component of `A`, kept for shape/diagnostic compatibility with
`base_project`, not the whole story -- plus the fields
[`project_var`](@ref) needs and diagnostics: `A` (the full `L x L`
transition matrix), `sigma_couple`, `u`, `v` (the unit-normalised
rank-1 factors, useful for checking which locations the fit couples
together).
"""
@model function model_var(d::ModelData; transform::Symbol=:log,
                           obsdata=observation_index(d))
    T, L, W, S, Dmax = model_dims(d)
    obs_idx, r_idx, yobs = obsdata.obs_idx, obsdata.r_idx, obsdata.yobs

    # --- Seasonality: identical to base_model ---
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

    seasonal = mu0 .+ mu_w[d.woy] .+ delta[d.woy, :] .+ season_eff[d.season]

    # --- Post-seasonal residual: VAR(1) across locations ---
    phi_pop_mean ~ Normal(0, 1)
    phi_pop_sd ~ truncated(Normal(0, 0.5); lower=0)
    phi_raw ~ filldist(Normal(0, 1), L)
    phi = tanh.(phi_pop_mean .+ phi_pop_sd .* phi_raw)

    u_raw ~ filldist(Normal(0, 1), L)
    v_raw ~ filldist(Normal(0, 1), L)
    u_hat = u_raw ./ norm(u_raw)
    v_hat = v_raw ./ norm(v_raw)
    sigma_couple ~ truncated(Normal(0, 0.15); lower=0)
    off_diag_mask = 1.0 .- Matrix{Float64}(I, L, L)
    A = Diagonal(phi) .+ sigma_couple .* (u_hat * v_hat') .* off_diag_mask

    mu_log_sigma_ar ~ Normal(log(0.2), 1)
    tau_log_sigma_ar ~ truncated(Normal(0, 0.5); lower=0)
    z_sigma_ar ~ filldist(Normal(0, 1), L)
    sigma_ar = exp.(mu_log_sigma_ar .+ tau_log_sigma_ar .* z_sigma_ar)

    eps_raw ~ filldist(Normal(0, 1), T, L)
    innov = eps_raw .* sigma_ar'                       # (T x L)
    first_resid = view(innov, 1, :) ./ sqrt.(1 .- phi .^ 2)
    later_rows = [view(innov, t, :) for t in 2:T]
    states = accumulate(
        (prev, row) -> A * prev + row, later_rows; init=first_resid,
    )
    residual = permutedims(reduce(hcat, vcat([first_resid], states)))

    latent = seasonal .+ residual

    # --- Backfill: identical to base_model ---
    r_pop_anchor ~ Normal(0, 0.05)
    sigma_r_pop ~ truncated(Normal(0, 0.3); lower=0)
    r_steps_raw ~ filldist(Normal(0, 1), Dmax)
    r_pop = backfill_profile(r_pop_anchor, r_steps_raw .* sigma_r_pop)

    sigma_r_loc ~ truncated(Normal(0, 0.3); lower=0)
    r_loc_raw ~ filldist(Normal(0, 1), Dmax + 1, L)
    r = r_pop .+ r_loc_raw .* sigma_r_loc

    sigma_obs ~ truncated(Normal(0, 1); lower=0)

    mu_obs = latent[obs_idx] .+ r[r_idx]
    yobs ~ arraydist(Normal.(mu_obs, sigma_obs))

    return (;
        latent, seasonal, residual, mu0, mu_w, delta, season_eff,
        phi, sigma_ar, r, r_pop, sigma_obs, transform,
        A, sigma_couple, u=u_hat, v=v_hat,
    )
end
