# Tests for src/hubio.jl. Standalone: includes core.jl + hubio.jl directly,
# so this can run without the full package (see docs/contracts.md).

using Test
using DataFrames
using CSV
using Dates

include(joinpath(@__DIR__, "..", "src", "core.jl"))
include(joinpath(@__DIR__, "..", "src", "hubio.jl"))

"""Build a small synthetic forecast table in the contracts.md schema."""
function synthetic_forecast(model_id, origin_dates, locations)
    rows = NamedTuple[]
    for origin_date in origin_dates, location in locations,
        horizon in HORIZONS, level in QUANTILE_LEVELS

        push!(rows, (;
            model_id, location, origin_date, horizon,
            target_end_date=origin_date + Day(7 * horizon),
            target=TARGET, output_type="quantile",
            output_type_id=level, value=1.0 + level,
        ))
    end
    return DataFrame(rows)
end

@testset "hubio" begin
    model_id = "nfidd-smoketest"
    origin_dates = [Date(2016, 1, 2), Date(2016, 1, 9)]
    locations = LOCATIONS[1:3]
    n_rows_per_file = length(locations) * length(HORIZONS) *
        length(QUANTILE_LEVELS)
    forecast_df = synthetic_forecast(model_id, origin_dates, locations)

    @testset "write_submission" begin
        hub_path = mktempdir()
        results = write_submission(forecast_df, hub_path)

        @test length(results) == length(origin_dates)

        for origin_date in origin_dates
            fname = string(origin_date) * "-" * model_id * ".csv"
            path = joinpath(hub_path, "model-output", model_id, fname)
            @test isfile(path)

            written = CSV.read(path, DataFrame)
            @test names(written) == string.(HUB_COLUMNS)
            @test !("model_id" in names(written))
            @test nrow(written) == n_rows_per_file
        end
    end

    @testset "dry_run" begin
        hub_path = mktempdir()
        results = write_submission(forecast_df, hub_path; dry_run=true)

        @test length(results) == length(origin_dates)
        @test !isdir(joinpath(hub_path, "model-output"))
        @test all(r -> nrow(r.df) == n_rows_per_file, results)
    end

    @testset "write_metadata" begin
        hub_path = mktempdir()
        path = write_metadata(
            model_id, hub_path;
            team_abbr="nfidd", model_abbr="smoketest",
        )
        @test isfile(path)
        @test path == joinpath(
            hub_path, "model-metadata", model_id * ".yml",
        )

        content = read(path, String)
        @test occursin("team_abbr: \"nfidd\"", content)
        @test occursin("model_abbr: \"smoketest\"", content)
        @test occursin("designated_model: true", content)
    end
end
