# Base joint Turing model for ILI forecasting.
# See docs/brief.md and docs/plan.md for the design rationale, and
# docs/contracts.md for `ModelData` and the module-scope constants this
# file relies on (it is `include`d into `SismidILITuring`'s scope).
#
# This is the tractable BASE of a search that will grow branches later
# (correlated location effects, VAR residuals, time-varying backfill,
# etc — see docs/plan.md). It is kept simple and correct rather than
# maximal: independent locations throughout, a random-walk-in-week
# population seasonal curve (not circular), iid partially-pooled
# location deviations, and a single shared observation noise term.

using Turing
using Distributions
using LinearAlgebra
using Statistics

"""
    model_dims(d::ModelData)

Return the model dimensions of `d` as a `NamedTuple`
`(T, L, W, S, Dmax)`, for use when sizing draws or projecting a fit
forward without re-deriving sizes from the data by hand.
"""
model_dims(d::ModelData) = (T=d.T, L=d.L, W=d.W, S=d.S, Dmax=d.Dmax)

"""
    ar_or_diff(eps, sigma, phi, difference)

Build one location's post-seasonal residual path (length `T`) from
standard-normal innovations `eps`, an innovation sd `sigma`, and an AR
coefficient `phi`.

If `difference` is `false` the path is a stationary AR(1): the first
value is drawn at the stationary variance `sigma^2 / (1 - phi^2)` and
later values follow `residual[t] = phi * residual[t-1] + innovation`.
If `difference` is `true` the AR coefficient is ignored and the path is
an integrated random walk (its first difference is white noise), i.e.
`residual[t] = residual[t-1] + innovation`, so `diff(residual)` is the
part being modelled.

Written as a non-mutating accumulation (`cumsum` / `accumulate`) rather
than an in-place loop so it stays friendly to Mooncake's AD.
"""
function ar_or_diff(eps::AbstractVector, sigma, phi, difference::Bool)
    innov = sigma .* eps
    if difference
        return cumsum(innov)
    end
    first_val = innov[1] / sqrt(1 - phi^2)
    rest = accumulate((prev, x) -> phi * prev + x, @view(innov[2:end]);
                       init=first_val)
    return vcat(first_val, rest)
end

"""
    backfill_profile(anchor, steps)

Build the population backfill profile `r_pop[1:Dmax+1]` (indexed
delay 0..Dmax) from an anchor value at the largest delay and `Dmax`
free steps, via a random walk built backward from the anchor. Each
step is unconstrained in sign, so the profile is non-monotonic; the
anchor's prior (set tight in `base_model`) is what pulls the profile
towards zero at the largest delay, not a hard constraint.
"""
function backfill_profile(anchor, steps::AbstractVector)
    offsets = cumsum(steps)               # delay Dmax-1, Dmax-2, ..., 0
    descending = anchor .+ offsets
    return vcat(reverse(descending), anchor)  # delay 0, 1, ..., Dmax
end

"""
    observation_index(d::ModelData)

Precompute the flat set of non-missing `(t, l)` observation cells in
`d.Y`, once, from data alone (no sampled parameters involved). Returns
a `NamedTuple` `(obs_idx, r_idx, yobs)`:

- `obs_idx`: `CartesianIndex{2}` vector into `d.Y` / `latent` (shape
  `T x L`) for every non-missing cell, in the same order the old
  `for l in 1:L, t in 1:T` loop visited them (column-major, so `l`
  varies slowest) — irrelevant to the log-density since summation is
  commutative, but kept for a stable, checkable ordering.
- `r_idx`: the matching `CartesianIndex{2}` vector into `r` (shape
  `Dmax+1 x L`), i.e. `(d.delay[i] + 1, i[2])` for each `i` in
  `obs_idx`.
- `yobs`: the gathered observed values as a concrete `Vector{Float64}`.

`base_model` takes this as a keyword argument defaulting to
`observation_index(d)`, so it is evaluated once when the `Model` is
constructed (a plain Julia default-argument evaluation), not
re-derived on every log-density/gradient evaluation during sampling.
"""
function observation_index(d::ModelData)
    obs_idx = findall(!ismissing, d.Y)
    yobs = Float64.(d.Y[obs_idx])
    r_idx = [CartesianIndex(d.delay[i] + 1, i[2]) for i in obs_idx]
    return (; obs_idx, r_idx, yobs)
end

"""
    base_model(d::ModelData; transform=:log, difference=false)

The base joint Turing model: partially-pooled week-of-season
seasonality, per-location AR(1) (or differenced) residual noise, and a
non-monotonic delay-indexed backfill revision, all with weakly
informative priors and hierarchical, partially-pooled variances.

# Components

- **Seasonality**: `seasonal[t, l] = mu0 + mu_w[woy[t]] + delta[woy[t],
  l] + season_eff[season[t]]`. `mu_w` is a non-centred random walk over
  week-of-season (population curve, not Fourier, not enforced circular
  in this base version), centred so it does not trade off against
  `mu0`. `delta` is an iid partially-pooled location deviation from the
  population curve. `season_eff` is a simple partially-pooled per-season
  level shift shared across locations.
- **Residual**: `latent = seasonal + residual`, with `residual` built by
  [`ar_or_diff`](@ref) per location. The AR coefficient `phi` and
  innovation sd `sigma_ar` are both partially pooled across locations
  (non-centred), `phi` mapped through `tanh` to stay in (-1, 1).
- **Backfill**: `r[delay, l] = r_pop[delay] + location deviation`,
  with the population profile `r_pop` built by
  [`backfill_profile`](@ref) (a random walk anchored, with a tight
  prior, at the largest delay) and location deviations iid
  partially-pooled around it. The observation model is
  `Y[t, l] ~ Normal(latent[t, l] + r[delay[t, l], l], sigma_obs)` for
  every non-missing cell; missing cells contribute no likelihood term.
  This is evaluated as a single vectorised observe over all non-missing
  cells at once (see [`observation_index`](@ref)), not a scalar loop,
  so Mooncake traces one observe statement per gradient step instead
  of one per non-missing cell.

`transform` is not used inside the model (`d.Y` is already on the
modelling scale), but is threaded through to the returned `NamedTuple`
so a forecaster can back-transform without having to also thread
`d.transform` through separately. `difference` switches the residual
from AR(1) level to a first-order difference (see [`ar_or_diff`](@ref)).

Returns a `NamedTuple` of the modelled quantities (`latent`,
`seasonal`, `residual`, `mu0`, `mu_w`, `delta`, `season_eff`, `phi`,
`sigma_ar`, `r`, `r_pop`, `sigma_obs`, `transform`) so a forecaster can
project `latent` forward (continue the AR/difference recursion with
`phi`/`sigma_ar`, and the seasonal curve via `mu_w`/`season_eff`) and
apply `r`/`r_pop` to nowcast the most recent, partially-observed weeks.
"""
@model function base_model(d::ModelData; transform::Symbol=:log,
                            difference::Bool=false,
                            obsdata=observation_index(d))
    T, L, W, S, Dmax = model_dims(d)
    obs_idx, r_idx, yobs = obsdata.obs_idx, obsdata.r_idx, obsdata.yobs

    # --- Seasonality: partially-pooled week-of-season random effect ---
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

    # --- Post-seasonal residual: partially-pooled AR(1) or difference ---
    phi_pop_mean ~ Normal(0, 1)
    phi_pop_sd ~ truncated(Normal(0, 0.5); lower=0)
    phi_raw ~ filldist(Normal(0, 1), L)
    phi = tanh.(phi_pop_mean .+ phi_pop_sd .* phi_raw)

    mu_log_sigma_ar ~ Normal(log(0.2), 1)
    tau_log_sigma_ar ~ truncated(Normal(0, 0.5); lower=0)
    z_sigma_ar ~ filldist(Normal(0, 1), L)
    sigma_ar = exp.(mu_log_sigma_ar .+ tau_log_sigma_ar .* z_sigma_ar)

    eps_raw ~ filldist(Normal(0, 1), T, L)
    residual = reduce(hcat, [
        ar_or_diff(view(eps_raw, :, l), sigma_ar[l], phi[l], difference)
        for l in 1:L
    ])

    latent = seasonal .+ residual

    # --- Backfill: non-monotonic delay-indexed revision ---
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
    )
end
