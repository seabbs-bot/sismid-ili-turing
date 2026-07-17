# Round 2 candidate ar-loc: HETEROGENEOUS per-location AR order.
#
# `v1-ar-high` (experiments/round1/v1-ar-high/model_v1.jl) raised the
# post-seasonal residual from AR(1) to a partially-pooled AR(p) with a
# single SHARED order `p` and a fixed lag-decay `0.6^(k-1)` applied
# identically to every location. `docs/eda/05-autocorrelation.md`'s
# per-location AIC order search (median 5, range 4-10 across the 11
# locations, a more-than-2-fold spread) and its per-season stability
# check (persistence is stable location to location and season to
# season; the AR-vs-differencing sign is not, but that is a separate
# axis, not this one) argue that a single shared order is too rigid:
# some locations plausibly need order ~4, others ~9-10.
#
# This file keeps v1's PACF (partial-autocorrelation) parameterisation
# — it is the right tool, since PACF unconstrained values map to a
# guaranteed-stationary AR(p) via Durbin-Levinson regardless of order —
# but replaces the single fixed, shared lag-decay with a per-location
# decay rate `rho[l]`, partially pooled across locations on the logit
# scale. A location with a small `rho[l]` shrinks its higher-lag PACF
# coefficients to (near) zero fast, giving it an effectively LOW order;
# a location with `rho[l]` close to 1 barely shrinks them at all, up to
# the shared ceiling `Pmax`, giving it an effectively HIGH order. This
# is the continuous, differentiable relaxation of "let each location
# pick its own AR order": no discrete order variable (which HMC/
# Pathfinder cannot handle) is introduced; instead each location gets
# its own smooth shrinkage profile, partially pooled toward a shared
# population profile, so the data can pull `rho[l]` high or low per
# location while still borrowing strength across locations.
#
# Seasonality and backfill are copied unchanged from `base_model`
# (src/model.jl); this file relies on `model_dims`, `backfill_profile`,
# and `observation_index` already being in scope (`src/model.jl` is
# included before this file — see check_ar_loc.jl).

using Turing
using Distributions
using Statistics

"""
    pacf_to_ar(pacf)

Durbin-Levinson recursion mapping `p` partial autocorrelations
`pacf[k] in (-1, 1)` to AR(p) coefficients guaranteed to lie in the
stationary region for any such `pacf`. Duplicated verbatim from
`experiments/round1/v1-ar-high/model_v1.jl` so this file stays loadable
on its own (see check_ar_loc.jl); behaviour is identical.
"""
function pacf_to_ar(pacf::AbstractVector)
    p = length(pacf)
    phi = [pacf[1]]
    for k in 2:p
        kk = pacf[k]
        prev = phi
        phi = vcat([prev[j] - kk * prev[k - j] for j in 1:(k - 1)], kk)
    end
    return phi
end

"""
    ar_p(eps, sigma, phi)

Build one location's post-seasonal residual path (length `T`) from
standard-normal innovations `eps`, an innovation sd `sigma`, and a
length-`p` AR coefficient vector `phi` (already passed through
[`pacf_to_ar`](@ref)). Duplicated verbatim from
`experiments/round1/v1-ar-high/model_v1.jl` (see that file's docstring
for the burn-in-approximation rationale for the first `p` values); kept
identical here so the per-location order change is isolated to how
`phi` is built, not how the recursion runs.
"""
function ar_p(eps::AbstractVector, sigma, phi::AbstractVector)
    p = length(phi)
    T = length(eps)
    innov = sigma .* eps
    init = innov[1:p]
    nsteps = T - p
    nsteps <= 0 && return init[1:T]
    states = accumulate(1:nsteps; init=init) do state, i
        t = p + i
        newval = sum(phi[k] * state[p - k + 1] for k in 1:p) + innov[t]
        vcat(@view(state[2:end]), newval)
    end
    tail = [s[end] for s in states]
    return vcat(init, tail)
end

"""
    model_ar_loc(d::ModelData; transform=:log1p, Pmax=10)

Round 2 candidate `ar-loc`: identical to `base_model` (partially-pooled
seasonality, non-monotonic backfill) except the per-location
post-seasonal residual is a partially-pooled AR(p) with a
LOCATION-SPECIFIC effective order, up to a shared ceiling `Pmax`.

# Per-location AR order via a location-specific PACF decay rate

`Pmax` is fixed (a keyword argument, not sampled) at the ceiling order
every location's AR(p) can reach; default 10, the largest per-location
order in `docs/eda/05-autocorrelation.md`'s AIC search (Region 7).
Locations that need a lower order shrink their unneeded high-lag
coefficients toward zero rather than being truncated to a smaller `p`.

For each location `l` and lag `k in 1:Pmax`:

- `rho[l] in (0, 1)` is that location's own decay rate, built
  non-centred on the logit scale and partially pooled across locations
  (`rho_pop_mean_logit`, `rho_pop_sd_logit`): `rho[l] =
  invlogit(rho_pop_mean_logit + rho_pop_sd_logit * rho_raw[l])`.
  `rho_pop_mean_logit`'s prior is centred at `logit(0.6)`, matching
  `v1-ar-high`'s fixed shared decay as the population starting point.
- `lag_decay[l, k] = rho[l]^(k - 1)`: small `rho[l]` decays fast (a
  location that only needs a low order), `rho[l]` near 1 decays slowly
  (a location that needs most of `Pmax`).
- `mu_pacf[k]` is a shared (across locations) population PACF profile,
  and `pacf_raw[l, k]` is each location's non-centred deviation from
  it, scaled by a shared `sigma_pacf_pop`. Critically, `lag_decay`
  multiplies the WHOLE pre-tanh value
  `mu_pacf[k] + sigma_pacf_pop * pacf_raw[l, k]`, not just the
  deviation, so a low-`rho[l]` location's high-lag PACF genuinely goes
  to zero (an effectively lower order) rather than merely shrinking
  toward the shared population profile.
- `pacf = tanh.(pacf_pre)` keeps every entry in (-1, 1); `phi[l, :] =
  pacf_to_ar(pacf[l, :])` (Durbin-Levinson) then guarantees a
  stationary AR(`Pmax`) for every location and every posterior draw.

`sigma_ar` (per-location innovation sd) keeps the same partially-pooled
log-normal structure as `base_model`/`v1-ar-high`. The observation
model reuses `base_model`'s vectorised `observation_index` approach
(one `arraydist` observe over all non-missing cells) rather than
`v1-ar-high`'s scalar loop, per `docs/lessons.md`'s note that the
scalar-loop observation model is a known overhead source.

Returns the same fields as `base_model`/`v1-ar-high` (`latent`,
`seasonal`, `residual`, `mu0`, `mu_w`, `delta`, `season_eff`, `phi`,
`sigma_ar`, `r`, `r_pop`, `sigma_obs`, `transform`), plus `rho`
(length-`L`, each location's decay rate — a diagnostic on the model's
effective per-location order, not needed by the projection but cheap
to carry). `phi` is `(L x Pmax)`, matching `v1-ar-high`'s shape so
`project_ar_loc.jl` (which reads the order from `size(phi, 2)`) is
usable unchanged for any `Pmax` this model is built with.
"""
@model function model_ar_loc(d::ModelData; transform::Symbol=:log1p,
                              Pmax::Int=10,
                              obsdata=observation_index(d))
    T, L, W, S, Dmax = model_dims(d)

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

    # --- Post-seasonal residual: per-location AR order via PACF decay ---
    rho_pop_mean_logit ~ Normal(logit(0.6), 1)
    rho_pop_sd_logit ~ truncated(Normal(0, 1); lower=0)
    rho_raw ~ filldist(Normal(0, 1), L)
    rho = invlogit.(rho_pop_mean_logit .+ rho_pop_sd_logit .* rho_raw)  # (L,)

    exps = 0:(Pmax - 1)
    lag_decay = rho .^ exps'                             # (L x Pmax)

    mu_pacf ~ filldist(Normal(0, 1), Pmax)
    sigma_pacf_pop ~ truncated(Normal(0, 0.5); lower=0)
    pacf_raw ~ filldist(Normal(0, 1), L, Pmax)
    pacf_pre = lag_decay .* (mu_pacf' .+ sigma_pacf_pop .* pacf_raw)
    pacf = tanh.(pacf_pre)                                # (L x Pmax), (-1,1)

    phi = permutedims(reduce(hcat,
        [pacf_to_ar(view(pacf, l, :)) for l in 1:L]))     # (L x Pmax)

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

    # --- Backfill: identical to base_model ---
    r_pop_anchor ~ Normal(0, 0.05)
    sigma_r_pop ~ truncated(Normal(0, 0.3); lower=0)
    r_steps_raw ~ filldist(Normal(0, 1), Dmax)
    r_pop = backfill_profile(r_pop_anchor, r_steps_raw .* sigma_r_pop)

    sigma_r_loc ~ truncated(Normal(0, 0.3); lower=0)
    r_loc_raw ~ filldist(Normal(0, 1), Dmax + 1, L)
    r = r_pop .+ r_loc_raw .* sigma_r_loc

    sigma_obs ~ truncated(Normal(0, 1); lower=0)

    mu_obs = latent[obsdata.obs_idx] .+ r[obsdata.r_idx]
    obsdata.yobs ~ arraydist(Normal.(mu_obs, sigma_obs))

    return (;
        latent, seasonal, residual, mu0, mu_w, delta, season_eff,
        phi, sigma_ar, rho, r, r_pop, sigma_obs, transform,
    )
end
