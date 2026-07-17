# Tests for src/scoring.jl (WIS scoring). Standalone: includes core.jl and
# scoring.jl directly rather than loading the whole package.

using Test
using Dates
using DataFrames
using Distributions
using Statistics

include("../src/core.jl")
include("../src/scoring.jl")

@testset "wis: hand-computed example" begin
    # levels/values chosen so the arithmetic is easy to check by hand.
    # One central interval (0.25, 0.75) plus the median (0.5).
    # K = 1, alpha_1 = 0.5, coverage = 0.5.
    # IS_0.5(1.0, 4.0; 3.0) = (4-1) + 0 + 0 = 3   (y = 3 lies inside [1, 4])
    # WIS = (1/1.5) * (0.5*|3-2| + (0.5/2)*3) = (1/1.5) * (0.5 + 0.75) = 5/6
    # dispersion       = (0.25*3) / 1.5 = 0.5
    # overprediction   = 0 / 1.5 = 0
    # underprediction  = (0 + 0.5*max(3-2,0)) / 1.5 = 1/3
    levels = [0.25, 0.5, 0.75]
    values = [1.0, 2.0, 4.0]
    y = 3.0
    result = wis(y, values, levels)

    @test isapprox(result.wis, 5 / 6; atol = 1e-10)
    @test isapprox(result.dispersion, 0.5; atol = 1e-10)
    @test isapprox(result.overprediction, 0.0; atol = 1e-10)
    @test isapprox(result.underprediction, 1 / 3; atol = 1e-10)
    @test isapprox(
        result.dispersion + result.overprediction + result.underprediction,
        result.wis; atol = 1e-10)
end

@testset "wis: calibration and non-negativity" begin
    levels = QUANTILE_LEVELS
    z = quantile.(Normal(), levels)
    sigma = 1.0
    y = 5.0

    good_values = y .+ sigma .* z
    bad_values = y .+ 5.0 .+ sigma .* z

    good = wis(y, good_values, levels)
    bad = wis(y, bad_values, levels)

    @test good.wis >= 0
    @test bad.wis >= 0
    @test good.wis < bad.wis
end

@testset "wis: input validation" begin
    @test_throws DimensionMismatch wis(1.0, [1.0, 2.0], [0.25, 0.5, 0.75])
    @test_throws ArgumentError wis(1.0, [1.0, 2.0], [0.25, 0.75])
end

# --- score_forecasts / wis_summary / compare_scales -----------------------

function build_forecast_row(model_id, location, origin_date, horizon,
        level, value)
    (model_id = model_id, location = location, origin_date = origin_date,
     horizon = horizon,
     target_end_date = origin_date + Day(7 * horizon),
     target = TARGET, output_type = "quantile",
     output_type_id = level, value = value)
end

@testset "score_forecasts / wis_summary / compare_scales" begin
    locations = ["US National", "HHS Region 1"]
    true_values = Dict("US National" => 5.0, "HHS Region 1" => 3.0)
    origin_dates = [Date(2018, 1, 6), Date(2018, 1, 13)]
    horizons = 1:2
    z = quantile.(Normal(), QUANTILE_LEVELS)
    sigma = 1.0
    bias = Dict("good-model" => 0.0, "bad-model" => 5.0)

    rows = NamedTuple[]
    for model_id in ("good-model", "bad-model"), location in locations,
        origin_date in origin_dates, horizon in horizons

        y = true_values[location]
        for (level, zk) in zip(QUANTILE_LEVELS, z)
            value = max(y + bias[model_id] + sigma * zk, 0.0)
            push!(rows, build_forecast_row(model_id, location, origin_date,
                                            horizon, level, value))
        end
    end
    forecast_df = DataFrame(rows)

    target_end_dates = Date[]
    for origin_date in origin_dates, horizon in horizons
        push!(target_end_dates, origin_date + Day(7 * horizon))
    end
    unique!(target_end_dates)

    truth_rows = NamedTuple[]
    for location in locations, ted in target_end_dates
        push!(truth_rows,
              (location = location, target_end_date = ted,
               value = true_values[location]))
    end
    truth_df = DataFrame(truth_rows)

    scored = score_forecasts(forecast_df, truth_df; scale = :natural)
    @test all(scored.wis .>= 0)
    n_tasks_expected = 2 * length(locations) * length(origin_dates) *
                        length(horizons)
    @test nrow(scored) == n_tasks_expected

    summary = wis_summary(scored)
    @test Set(summary.model_id) == Set(["good-model", "bad-model"])
    @test "sd_wis" in names(summary)
    good_mean = only(summary.mean_wis[summary.model_id .== "good-model"])
    bad_mean = only(summary.mean_wis[summary.model_id .== "bad-model"])
    @test good_mean < bad_mean

    log_scored = score_forecasts(forecast_df, truth_df; scale = :log)
    @test all(log_scored.wis .>= 0)
    @test nrow(log_scored) == n_tasks_expected

    comparison = compare_scales(forecast_df, truth_df)
    @test Set(comparison.natural.model_id) == Set(["good-model", "bad-model"])
    @test Set(comparison.log.model_id) == Set(["good-model", "bad-model"])
    @test "rank_changed" in names(comparison.comparison)
end
