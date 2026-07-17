# Tests for src/inference.jl.
# Uses a trivial Turing model (a single Normal mean, plus a doubled-mu
# "generated quantity" in its return value) so the checks are about the
# wrappers, not about any particular SismidILITuring model. Checks:
# fit_pathfinder runs and posterior_draws returns a non-empty Vector of
# NamedTuples with a `mu` field; fit_mcmc with AutoMooncake and the progress
# callback runs for a tiny number of samples and posterior_draws works on
# its chain too; generated_draws recovers the model's returned `mu_doubled`
# for both fit_pathfinder and fit_mcmc results.

using Random
using Turing
using Mooncake
using Statistics
using Test

include(joinpath(@__DIR__, "..", "src", "core.jl"))
include(joinpath(@__DIR__, "..", "src", "inference.jl"))

Random.seed!(20240717)

@model function trivial_model(y)
    mu ~ Normal()
    y .~ Normal(mu, 1)
    return (mu_doubled=2 * mu,)
end

const Y_TRIVIAL = [0.1, 0.2, -0.1, 0.3, 0.05]

@testset "fit_pathfinder / posterior_draws" begin
    model = trivial_model(Y_TRIVIAL)

    fit = fit_pathfinder(model; ndraws=200)
    draws = posterior_draws(fit)
    @test draws isa Vector{<:NamedTuple}
    @test !isempty(draws)
    @test all(d -> haskey(d, :mu), draws)
    @test all(d -> isfinite(d.mu), draws)

    # Multi-path Pathfinder should also work through the same interface.
    fit_multi = fit_pathfinder(model; ndraws=200, nruns=4)
    draws_multi = posterior_draws(fit_multi)
    @test draws_multi isa Vector{<:NamedTuple}
    @test !isempty(draws_multi)
    @test all(d -> haskey(d, :mu), draws_multi)

    # generated_draws exposes the model's returned generated quantities
    # (here `mu_doubled`), not just the sampled sites.
    gdraws = generated_draws(model, fit)
    @test gdraws isa Vector{<:NamedTuple}
    @test length(gdraws) == length(draws)
    @test all(d -> haskey(d, :mu_doubled), gdraws)
    @test all(
        gr -> isapprox(gr[1].mu_doubled, 2 * gr[2].mu),
        zip(gdraws, draws),
    )
end

@testset "fit_mcmc / progress_callback / posterior_draws" begin
    model = trivial_model(Y_TRIVIAL)

    logged_iterations = Int[]
    cb = progress_callback(; every=5)
    wrapped_cb = function (rng, m, sampler, sample, state, iteration; kwargs...)
        iteration % 5 == 0 && push!(logged_iterations, iteration)
        return cb(rng, m, sampler, sample, state, iteration; kwargs...)
    end

    chain = fit_mcmc(
        model;
        nsamples=20,
        nchains=2,
        adtype=AutoMooncake(),
        callback=wrapped_cb,
    )

    @test !isempty(logged_iterations)

    draws = posterior_draws(chain)
    @test draws isa Vector{<:NamedTuple}
    @test length(draws) == 20 * 2
    @test all(d -> haskey(d, :mu), draws)
    @test all(d -> isfinite(d.mu), draws)

    # generated_draws also works on a bare fit_mcmc chain, given the model.
    gdraws = generated_draws(model, chain)
    @test gdraws isa Vector{<:NamedTuple}
    @test length(gdraws) == length(draws)
    @test all(d -> haskey(d, :mu_doubled), gdraws)
    @test all(
        gr -> isapprox(gr[1].mu_doubled, 2 * gr[2].mu),
        zip(gdraws, draws),
    )
end

println("test_inference.jl passed")
