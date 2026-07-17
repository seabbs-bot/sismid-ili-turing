# Tests for src/model.jl.
# src/data.jl does not exist yet, so this builds a small synthetic
# ModelData by hand instead of loading it. Checks: base_model builds
# for both `difference` settings, a prior predictive draw runs and is
# finite, the returned NamedTuple exposes the named components at the
# right sizes, and a tiny NUTS fit with the Mooncake AD backend runs
# without error and returns a finite log-density.

using Random
using Dates
using Distributions
using Turing
using Mooncake
using Statistics
using Test

include(joinpath(@__DIR__, "..", "src", "core.jl"))
include(joinpath(@__DIR__, "..", "src", "model.jl"))

Random.seed!(20240717)

# --- Build a small synthetic ModelData by hand ---
const T = 40
const L = 3
const W = 33
const S = 1
const Dmax = 4

woy = [mod1(t, W) for t in 1:T]
season = fill(1, T)
dates = Date(2016, 1, 2) .+ Day.(7 .* (0:(T - 1)))

delay = [min(T - t, Dmax) for t in 1:T, l in 1:L]
# Mark a few of the most recent cells as not-yet-reported (missing).
delay[T, 2] = -1
delay[T, 3] = -1
delay[T - 1, 3] = -1

true_curve = [2.0 + 0.5 * sin(2 * pi * w / W) for w in woy]
Y = Matrix{Union{Missing,Float64}}(undef, T, L)
for l in 1:L, t in 1:T
    Y[t, l] = delay[t, l] == -1 ? missing : true_curve[t] + 0.05 * randn()
end

d = ModelData(Y, delay, woy, season, dates, L, T, W, S, Dmax, :log, dates[end])

@testset "base_model" begin
    dims = model_dims(d)
    @test dims == (T=T, L=L, W=W, S=S, Dmax=Dmax)

    for difference in (false, true)
        model = base_model(d; transform=:log, difference=difference)

        # Prior predictive draw: should run and be finite throughout.
        prior_chain = sample(model, Prior(), 5; progress=false)
        @test all(isfinite, Array(prior_chain))

        # Calling the model directly draws from the prior and returns
        # the generated quantities; check the named components.
        gq = model()
        @test size(gq.latent) == (T, L)
        @test size(gq.seasonal) == (T, L)
        @test size(gq.residual) == (T, L)
        @test size(gq.r) == (Dmax + 1, L)
        @test length(gq.r_pop) == Dmax + 1
        @test length(gq.mu_w) == W
        @test length(gq.phi) == L
        @test length(gq.sigma_ar) == L
        @test gq.transform == :log

        # Tiny MCMC fit, Mooncake AD backend as required by the brief.
        chain = sample(
            model, NUTS(; adtype=AutoMooncake()), 30; progress=false,
        )
        @test all(isfinite, chain[:lp])
    end
end

println("test_model.jl passed")
