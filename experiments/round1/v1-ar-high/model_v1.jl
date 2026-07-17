# Round 1 candidate v1-ar-high: raise the post-seasonal residual from
# AR(1) to a partially-pooled AR(p), motivated by
# docs/eda/05-autocorrelation.md (AIC-selected AR order on the
# deseasonalised residual has median 5, range 4-10 across locations,
# while the PACF drops sharply after lag 1).
#
# Everything else (seasonality, backfill) is copied unchanged from
# `base_model` in src/model.jl. This file relies on `model_dims` and
# `backfill_profile` already being in scope (`src/model.jl` is included
# before this file — see check_v1.jl); it does NOT reuse
# `ar_or_diff`, since AR(1)'s exact stationary-variance formula does not
# generalise to AR(p) — see `ar_p` below.

using Turing
using Distributions
using Statistics

"""
    pacf_to_ar(pacf)

Durbin-Levinson recursion mapping `p` partial autocorrelations
`pacf[k] in (-1, 1)` to AR(p) coefficients that are guaranteed to lie
in the stationary region for any such `pacf`. This is exactly why PACF
is a convenient unconstrained-to-stationary parameterisation for AR(p),
and it is a natural fit here: `docs/eda/05-autocorrelation.md` reports
the PACF on the deseasonalised residual cuts off sharply after lag 1
(lag-1 partial 0.78-0.96, lag-2 mostly -0.13 to +0.3, lag 3+ mostly
inside +-0.13), while the ACF decays gradually — the classic AR(p)
signature this recursion is designed for.

Builds the coefficient vector up from length 1 to length `p` via a
plain (non-Turing) loop; each step replaces the whole vector rather
than mutating it in place, so it stays Mooncake-AD-friendly like
`ar_or_diff` in the base model.
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
[`pacf_to_ar`](@ref), so guaranteed stationary).

`residual[t] = sum_{k=1}^p phi[k] * residual[t-k] + innovation[t]` for
`t > p`. The first `p` values are seeded directly from scaled
innovations (`residual[1:p] = sigma .* eps[1:p]`) rather than the exact
AR(p) stationary variance, which needs a `p x p` Yule-Walker solve;
this is a burn-in approximation, and is considered negligible given
`T` (~100+ weeks of training history in practice) is much larger than
`p` (<=10 per the EDA's order search).

Written with `accumulate` over a sliding `p`-length state (oldest to
newest), rather than an in-place loop, so it stays Mooncake-AD-friendly
like `ar_or_diff` in the base model.
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
    model_v1(d::ModelData; transform=:log1p, p=5)

Round 1 candidate `v1-ar-high`: identical to `base_model` (partially-
pooled seasonality, non-monotonic backfill) except the per-location
post-seasonal residual is a partially-pooled AR(`p`) instead of AR(1).

# AR(p) parameterisation

`p` is fixed (a keyword argument, not sampled) at a single shared order
across all locations, defaulting to 5 (the EDA's median selected
order). Per-location, per-lag partial autocorrelations `pacf[l, k]` are
built non-centred and partially pooled across locations, with a fixed
(not sampled) lag-decay multiplier `0.6^(k - 1)` tightening the pooling
sd as lag order grows — this directly encodes the EDA finding that the
PACF is large and variable at lag 1 but small and tightly clustered
around 0 from lag 2 on, rather than fixing one hard AR order or
treating every lag as equally informative. Each `pacf[l, k]` is mapped
through `tanh` to stay in (-1, 1), then converted to AR coefficients
`phi[l, :]` via [`pacf_to_ar`](@ref) (Durbin-Levinson), which
guarantees a stationary AR(p) for every posterior draw and every
location without a hard constraint. A single shared order `p` (rather
than a location-varying order, which the EDA's 4-10 range might
suggest) is the tractable choice for this branch: partial pooling on
the PACF already lets locations with a truly lower order shrink their
higher-lag coefficients close to zero, which is a softer version of
the same idea.

`sigma_ar` (per-location innovation sd) keeps the same partially-pooled
log-normal structure as `base_model`.

Returns the same fields as `base_model` (`latent`, `seasonal`,
`residual`, `mu0`, `mu_w`, `delta`, `season_eff`, `sigma_ar`, `r`,
`r_pop`, `sigma_obs`, `transform`), with `phi` now an `(L x p)` matrix
(one AR(p) coefficient vector per location) rather than a length-`L`
vector of AR(1) coefficients. See `project_v1.jl` for the matching
forecast projection.
"""
@model function model_v1(d::ModelData; transform::Symbol=:log1p, p::Int=5)
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

    # --- Backfill: identical to base_model ---
    r_pop_anchor ~ Normal(0, 0.05)
    sigma_r_pop ~ truncated(Normal(0, 0.3); lower=0)
    r_steps_raw ~ filldist(Normal(0, 1), Dmax)
    r_pop = backfill_profile(r_pop_anchor, r_steps_raw .* sigma_r_pop)

    sigma_r_loc ~ truncated(Normal(0, 0.3); lower=0)
    r_loc_raw ~ filldist(Normal(0, 1), Dmax + 1, L)
    r = r_pop .+ r_loc_raw .* sigma_r_loc

    sigma_obs ~ truncated(Normal(0, 1); lower=0)

    for l in 1:L, t in 1:T
        if !ismissing(d.Y[t, l])
            mean_obs = latent[t, l] + r[d.delay[t, l] + 1, l]
            d.Y[t, l] ~ Normal(mean_obs, sigma_obs)
        end
    end

    return (;
        latent, seasonal, residual, mu0, mu_w, delta, season_eff,
        phi, sigma_ar, r, r_pop, sigma_obs, transform,
    )
end
