# Inference wrappers shared by all Turing models in this package, plus a
# common posterior-draws accessor. See docs/contracts.md.
#
# This file is standalone: it can be `include`d after src/core.jl without
# loading the whole package (e.g. from a test file).

using Turing
using Pathfinder
using Mooncake
using Random
using Statistics

"""
    fit_pathfinder(model; ndraws=1000, nruns=1, rng=Random.default_rng())

Fast variational screening pass using Pathfinder.jl. This is the DEFAULT
fitting path for any Turing `model` built in this package, since MCMC
(`fit_mcmc`) is slow.

With `nruns == 1` (the default) this runs single-path Pathfinder
(`Pathfinder.pathfinder`). With `nruns > 1` it runs multi-path Pathfinder
(`Pathfinder.multipathfinder`), fitting `nruns` independent paths and mixing
them via Pareto-smoothed importance resampling; use this when the posterior
may be multimodal or a single path looks unreliable.

`ndraws` is the number of posterior draws taken from the fitted
approximation.

Returns the raw `Pathfinder.PathfinderResult` (or
`Pathfinder.MultiPathfinderResult` for `nruns > 1`). Pass the result to
[`posterior_draws`](@ref) to get the common draws representation.
"""
function fit_pathfinder(
    model;
    ndraws::Int=1000,
    nruns::Int=1,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    if nruns == 1
        return Pathfinder.pathfinder(model; ndraws=ndraws, rng=rng)
    end
    return Pathfinder.multipathfinder(model, ndraws; nruns=nruns, rng=rng)
end

"""
    fit_mcmc(model; nsamples=1000, nchains=2, adtype=Turing.AutoMooncake(),
             callback=nothing, rng=Random.default_rng())

Fit `model` by NUTS via Turing. Mooncake (`Turing.AutoMooncake()`) is the
required AD backend.

Chains are sampled serially (`Turing.MCMCSerial()`) rather than with
`MCMCThreads()`. This is deliberate: it keeps a stateful `callback` (e.g.
[`progress_callback`](@ref)) free of data races between chains, at the cost
of not parallelising chains across threads.

Returns the sampled chain (a `FlexiChains.VNChain` with this Turing
version). Pass it to [`posterior_draws`](@ref) to get the common draws
representation.
"""
function fit_mcmc(
    model;
    nsamples::Int=1000,
    nchains::Int=2,
    adtype=Turing.AutoMooncake(),
    callback=nothing,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    return Turing.sample(
        rng,
        model,
        Turing.NUTS(),
        Turing.MCMCSerial(),
        nsamples,
        nchains;
        adtype=adtype,
        callback=callback,
        progress=false,
    )
end

"""
    progress_callback(; every=100)

Build a Turing/AbstractMCMC sampling callback for live monitoring of
`fit_mcmc` runs.

Every `every` iterations it logs (via `@info`) the iteration number and, if
the sampler's transition exposes them, the mean acceptance rate and mean
step size accumulated since the last log. NUTS transitions expose these as
`stats.acceptance_rate` / `stats.step_size`; samplers that do not are logged
with the iteration number only.

The returned closure matches the `AbstractMCMC` callback signature
`(rng, model, sampler, sample, state, iteration; kwargs...)`, so it can be
passed straight to `fit_mcmc`'s `callback` keyword.
"""
function progress_callback(; every::Int=100)
    acceptance_total = Ref(0.0)
    step_size_total = Ref(0.0)
    nstats = Ref(0)
    return function (rng, model, sampler, sample, state, iteration; kwargs...)
        stats = hasproperty(sample, :stats) ? sample.stats : nothing
        if stats !== nothing
            hasproperty(stats, :acceptance_rate) &&
                (acceptance_total[] += stats.acceptance_rate)
            hasproperty(stats, :step_size) &&
                (step_size_total[] += stats.step_size)
            nstats[] += 1
        end
        if iteration % every == 0
            if nstats[] > 0
                @info "MCMC progress" iteration mean_acceptance =
                    acceptance_total[] / nstats[] mean_step_size =
                    step_size_total[] / nstats[]
                acceptance_total[] = 0.0
                step_size_total[] = 0.0
                nstats[] = 0
            else
                @info "MCMC progress" iteration
            end
        end
        return nothing
    end
end

# Underlying chain object for a fit: `fit_mcmc` returns one directly;
# Pathfinder results carry theirs in `draws_transformed`.
_chain(fit) = fit
_chain(fit::Pathfinder.PathfinderResult) = fit.draws_transformed
_chain(fit::Pathfinder.MultiPathfinderResult) = fit.draws_transformed

"""
    posterior_draws(fit)::Vector{<:NamedTuple}

Common accessor turning any fit produced by [`fit_pathfinder`](@ref) or
[`fit_mcmc`](@ref) into a flat `Vector` of per-draw `NamedTuple`s, keyed by
the sampled model's parameter names (vector/matrix-valued parameters keep
their array shape as a single field, e.g. `draw.mu isa Vector{Float64}`).
This is the interface `src/forecast.jl` consumes.

For `fit_mcmc` results with more than one chain, draws from all chains are
concatenated into one vector (order: chain-major, i.e. all draws of chain 1
first). Callers that need per-chain diagnostics should use the chain object
directly rather than this accessor.

Internally this reads the fit's underlying chain object (a
`FlexiChains.VNChain`, exposed via `Turing.FlexiChains` since FlexiChains is
not a direct dependency of this package) and asks it to reconstruct one
`NamedTuple` per (iteration, chain) pair.
"""
function posterior_draws(fit)::Vector{<:NamedTuple}
    mat = Turing.FlexiChains.parameters_at(_chain(fit), NamedTuple)
    return vec(collect(mat))
end

"""
    generated_draws(model, fit)::Vector{<:NamedTuple}

Evaluate `model`'s return value (its generated quantities) for every
posterior draw in `fit`, returning a flat `Vector` of `NamedTuple`s. For
`base_model` these carry the derived quantities a forecaster needs
(`latent`, `seasonal`, `residual`, `mu0`, `mu_w`, `delta`, `season_eff`,
`phi`, `sigma_ar`, `r`, `r_pop`, ...), which the sampled-site view from
[`posterior_draws`](@ref) does not expose. This is the interface
`src/forecast.jl`'s `base_project` consumes.

Draws from all chains are flattened into one vector (as for
[`posterior_draws`](@ref)). Works for both `fit_pathfinder` and
`fit_mcmc` results via their underlying `FlexiChains.VNChain`.
"""
function generated_draws(model, fit)::Vector{<:NamedTuple}
    gq = Turing.DynamicPPL.returned(model, _chain(fit))
    return vec(collect(gq))
end
