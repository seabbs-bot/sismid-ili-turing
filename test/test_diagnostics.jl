# Tests for src/diagnostics.jl against a tiny synthetic ModelData, a
# synthetic draws vector, and a trivial predict closure, so this exercises
# the Bayesian-workflow checks without depending on src/model.jl or
# src/inference.jl (developed in parallel; see docs/contracts.md).

using Test
using Dates
using Turing

include(joinpath(@__DIR__, "..", "src", "core.jl"))
include(joinpath(@__DIR__, "..", "src", "diagnostics.jl"))

# -- synthetic ModelData: T=6 weeks, L=2 locations, one season -------------

natural = [
    3.0 5.0
    4.0 6.0
    5.0 5.0
    4.0 4.0
    3.0 5.0
    4.0 6.0
]
Y = Matrix{Union{Missing,Float64}}(log.(natural))
Y[1, 1] = missing
Y[6, 2] = missing

delay = fill(0, 6, 2)
delay[1, 1] = -1
delay[6, 2] = -1

data = ModelData(
    Y,
    delay,
    collect(1:6),           # woy
    fill(1, 6),             # season
    Date(2016, 1, 2) .+ Week.(0:5),
    2, 6, 6, 1, 1,
    :log,
    Date(2016, 1, 2) + Week(5),
)

# -- trivial predict closure: constant per draw, no model internals -------

trivial_predict(draw, d::ModelData) = fill(draw.level, d.T, d.L)

# -- synthetic draws vector (stands in for a real fit's posterior draws) --

draws = [(level=4.0 + 0.4 * sin(i),) for i in 1:30]

# -- tiny Turing model for the prior predictive default draw path ---------

@model function toy_model()
    level ~ Normal(4.0, 1.0)
    return (level=level,)
end
model = toy_model()

@testset "prior_predictive" begin
    result = prior_predictive(model, data; ndraws=20, predict=trivial_predict)
    s = result.summary
    @test length(result.simulated) == 20
    @test all(size(m) == (data.T, data.L) for m in result.simulated)
    @test isfinite(s.mean)
    @test isfinite(s.sd)
    @test s.min <= s.q50 <= s.max
    @test 0.0 <= s.frac_outside_plausible_range <= 1.0
    @test 0.0 <= s.frac_nonfinite <= 1.0
end

@testset "posterior_predictive" begin
    result = posterior_predictive(
        draws, model, data; ndraws=20, predict=trivial_predict,
    )
    @test result.per_observation isa DataFrame
    expected_obs = count(!ismissing, data.Y)
    @test nrow(result.per_observation) == expected_obs
    @test all(0.0 .<= result.per_observation.pred_mean)
    @test 0.0 <= result.calibration.coverage50 <= 1.0
    @test 0.0 <= result.calibration.coverage90 <= 1.0
    @test result.calibration.n == expected_obs
end

@testset "residual_summary" begin
    result = residual_summary(
        draws, model, data; ndraws=20, predict=trivial_predict,
    )
    @test result.by_location isa DataFrame
    @test nrow(result.by_location) == data.L
    @test result.by_week_of_season isa DataFrame
    @test result.autocorrelation isa DataFrame
    @test size(result.residuals) == (data.T, data.L)
    @test count(!ismissing, result.residuals) == count(!ismissing, data.Y)
end

@testset "bayesian_checks" begin
    result = bayesian_checks(
        draws, model, data; ndraws=20, predict=trivial_predict,
    )
    @test 0.0 <= result.posterior.calibration.coverage50 <= 1.0
    @test 0.0 <= result.posterior.calibration.coverage90 <= 1.0
    @test isfinite(result.prior.summary.mean)
    @test nrow(result.residuals.by_location) == data.L
end
