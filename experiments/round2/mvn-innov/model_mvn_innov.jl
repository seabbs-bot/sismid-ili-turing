# Round 2 candidate mvn-innov: independent-per-location AR(1) level
# dynamics (identical to base_model, src/model.jl), but the AR
# INNOVATIONS at each time step are drawn from a multivariate normal
# across the 11 locations, instead of independently. Seasonality and
# backfill are unchanged from base_model: partially-pooled
# week-of-season population curve, iid partially-pooled location
# deviation, non-monotonic delay-indexed backfill.
#
# Motivation / contrast with round2/var: docs/eda/04-cross-location.md
# finds moderate cross-location correlation on the differenced series
# (mean ~0.24, up to ~0.69 for closely related regions), and that
# coupling is CONTEMPORANEOUS (lag 0) -- every region's cross-
# correlation against the national series peaks at lag 0, with no
# lead-lag. The EDA's own "Implications for the model" section reads
# this as favouring a contemporaneous-correlation (MVN innovation)
# design over a lagged VAR, with full VAR flagged as "a candidate to
# test rather than an obvious win" (that test is
# experiments/round2/var/model_var.jl, a VAR(1) with a shrunk rank-1
# cross-location transition term). This file is the design the EDA
# itself reads as the higher-priority axis: no lagged cross-location
# term at all, only a same-timestep correlated innovation. If this
# candidate outperforms round2/var, that is the EDA's contemporaneous-
# coupling reading confirmed empirically.
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
    ar_from_innovations(innov, phi, difference)

Build one location's post-seasonal residual path (length `T`) from an
ALREADY-FORMED innovation series `innov` (i.e. `sigma * eps` has been
applied, and in this file the cross-location correlation has also
already been mixed in -- see `model_mvn_innov`), and an AR coefficient
`phi`.

This is [`ar_or_diff`](@ref) (`src/model.jl`) with the `sigma .* eps`
step removed, because in `model_mvn_innov` that scaling is not a
per-location scalar multiply: it is the matrix product that mixes
locations together (see below), so it has to happen before this
function is called, once for all locations at once, rather than inside
a per-location loop. Kept as a separate, differently-named function
(rather than calling `ar_or_diff` with `sigma=1`) so the "already
scaled" precondition is explicit at the call site, not a magic-number
argument. Written with `cumsum`/`accumulate` for the same
Mooncake-AD-friendliness reason as `ar_or_diff`.
"""
function ar_from_innovations(innov::AbstractVector, phi, difference::Bool)
    if difference
        return cumsum(innov)
    end
    first_val = innov[1] / sqrt(1 - phi^2)
    rest = accumulate((prev, x) -> phi * prev + x, @view(innov[2:end]);
                       init=first_val)
    return vcat(first_val, rest)
end

"""
    model_mvn_innov(d::ModelData; transform=:log, difference=false,
                     lkj_eta=1.0, obsdata=observation_index(d))

Round 2 candidate `mvn-innov`: identical to `base_model` except the
per-location post-seasonal AR(1) innovations are drawn from a
multivariate normal across all `L` locations at each time step, instead
of independently. The AR(1) LEVEL dynamics stay per-location
(`phi[l]`, identical prior to `base_model`'s) -- only the innovation
driving each location's recursion is correlated with the other
locations' innovations at that same time step.

# Correlation parameterisation: LKJ Cholesky + per-location scales

- `Lcorr ~ LKJCholesky(L, lkj_eta)`: the Cholesky factor of an `L x L`
  correlation matrix, `lkj_eta` fixed at `1.0` (uniform over
  correlation matrices) rather than given its own hyperprior -- a
  single shape hyperparameter over an already-modest `L=11` structure
  is not worth the extra identifiability burden, matching this
  search's general preference for parsimony over completeness (e.g.
  round2/var fixes its rank-1 coupling to a single shrinkage scale
  rather than a full free matrix).
- `sigma_ar` (length `L`): IDENTICAL partially-pooled log-normal prior
  to `base_model`'s per-location innovation sd.
- `eps_raw ~ filldist(Normal(0, 1), L, T)`: raw iid standard normals,
  one per (location, time) cell -- the same total count and
  distribution as `base_model`'s `eps_raw ~ filldist(Normal(0, 1), T,
  L)`, just transposed to `L x T` so the correlation mixing below is a
  single matrix product across locations at each time column.
- `scaled_L = Diagonal(sigma_ar) * Lcorr.L` (`L x L`, deterministic):
  combines the per-location scales with the correlation structure.
  `innov = scaled_L * eps_raw` (`L x T`) is the fully-formed,
  cross-location-correlated innovation series. Column `t`'s covariance
  is `scaled_L * scaled_L' = Diagonal(sigma_ar) * Corr *
  Diagonal(sigma_ar)`, i.e. exactly `base_model`'s per-location
  variance `sigma_ar[l]^2` on the diagonal, with `Corr`'s off-diagonal
  entries setting the cross-location covariance -- so this candidate
  nests `base_model` exactly at `Corr = I`.
- This is the NON-CENTRED pattern used throughout `base_model` (`eps_raw`,
  `mu_w_raw`, `delta_raw`, `phi_raw`, `z_sigma_ar`, `r_steps_raw`,
  `r_loc_raw` are all raw iid Normal(0, 1) sites, deterministically
  transformed): the only sampled sites here are `Lcorr` and `eps_raw`,
  and `innov` is a plain deterministic matrix product of the two, kept
  as simple algebra (no covariance-matrix inversion/decomposition at
  each gradient step) for AD-friendliness. `Lcorr` itself is sampled
  directly from `LKJCholesky` (there is no non-centred alternative
  parameterisation of a correlation matrix in general use); this file
  only needs `Lcorr` for PRIOR forward-sampling here (no NUTS/Pathfinder
  fit in this pass -- see `check_mvn_innov.jl`), so the Bijectors.jl
  correlation-Cholesky transform used during HMC is not exercised yet
  and should be checked before a full fit is attempted on this
  candidate.

Each location's residual path is then [`ar_from_innovations`](@ref)
applied to that location's column of `innov` (transposed to `T x L`
first, to match `base_model`'s residual layout) with its own `phi[l]`
-- an AR(1) recursion per location, exactly as in `base_model`, just
fed a correlated rather than independent innovation series.

Everything else -- `mu0`, the population `mu_w` random walk, `delta`,
`season_eff`, and the non-monotonic backfill ([`backfill_profile`](@ref))
-- is copied verbatim from `base_model`, including its vectorised
observation likelihood over [`observation_index`](@ref)'s precomputed
non-missing cells.

`difference` behaves exactly as in `base_model`: switches every
location's residual from AR(1) level to a first-order difference (see
`ar_or_diff`/[`ar_from_innovations`](@ref)); the correlated-innovation
structure applies identically in either case, since it only changes
how `innov` is formed, not how each location's path is built from it.

Returns a `NamedTuple` with the same field names as `base_model`
(`latent, seasonal, residual, mu0, mu_w, delta, season_eff, phi,
sigma_ar, r, r_pop, sigma_obs, transform`) plus the fields
[`project_mvn_innov`](@ref) needs and diagnostics: `Lcorr` (the `L x L`
lower-triangular correlation Cholesky factor, as a plain `Matrix`) and
`corr` (the reconstructed `L x L` correlation matrix, for
inspecting/checking which locations the fit couples together).
"""
@model function model_mvn_innov(d::ModelData; transform::Symbol=:log,
                                 difference::Bool=false,
                                 lkj_eta::Real=1.0,
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

    # --- Post-seasonal residual: per-location AR(1), MVN innovations ---
    phi_pop_mean ~ Normal(0, 1)
    phi_pop_sd ~ truncated(Normal(0, 0.5); lower=0)
    phi_raw ~ filldist(Normal(0, 1), L)
    phi = tanh.(phi_pop_mean .+ phi_pop_sd .* phi_raw)

    mu_log_sigma_ar ~ Normal(log(0.2), 1)
    tau_log_sigma_ar ~ truncated(Normal(0, 0.5); lower=0)
    z_sigma_ar ~ filldist(Normal(0, 1), L)
    sigma_ar = exp.(mu_log_sigma_ar .+ tau_log_sigma_ar .* z_sigma_ar)

    Lcorr ~ LKJCholesky(L, lkj_eta)
    scaled_L = Diagonal(sigma_ar) * Lcorr.L

    eps_raw ~ filldist(Normal(0, 1), L, T)
    innov = scaled_L * eps_raw            # (L x T), correlated across L
    innov_TL = permutedims(innov)         # (T x L), matches residual layout

    residual = reduce(hcat, [
        ar_from_innovations(view(innov_TL, :, l), phi[l], difference)
        for l in 1:L
    ])

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
        Lcorr=Matrix(Lcorr.L), corr=Matrix(Lcorr),
    )
end
