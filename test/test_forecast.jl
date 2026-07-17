using Test
using DataFrames
using Dates
using Random

include(joinpath(@__DIR__, "..", "src", "core.jl"))
include(joinpath(@__DIR__, "..", "src", "forecast.jl"))

Random.seed!(20260717)

function synthetic_data(; L=length(LOCATIONS), T=12, W=52, S=1, Dmax=6)
    origin_date = Date(2016, 1, 2)
    dates = [origin_date - Day(7 * (T - t)) for t in 1:T]
    woy = [mod1(40 + t, W) for t in 1:T]
    season = fill(1, T)
    Y = Matrix{Union{Missing,Float64}}(undef, T, L)
    for t in 1:T, l in 1:L
        Y[t, l] = to_scale(2.0 + 0.1 * l, :log)
    end
    delay = zeros(Int, T, L)
    return ModelData(Y, delay, woy, season, dates, L, T, W, S, Dmax,
        :log, origin_date)
end

@testset "forecast_quantiles: custom project" begin
    data = synthetic_data()
    L = data.L
    ndraws = 60
    chain = [(scale=1.0 + 0.02 * d,) for d in 1:ndraws]

    trivial_project(draw, d, horizons) =
        fill(log(3.0), d.L, maximum(horizons)) .* draw.scale

    df = forecast_quantiles(chain, data, "nfidd-test";
        project=trivial_project)

    expected_cols = [
        :model_id, :location, :origin_date, :horizon,
        :target_end_date, :target, :output_type, :output_type_id,
        :value,
    ]
    @test propertynames(df) == expected_cols

    nlev = length(QUANTILE_LEVELS)
    nhor = length(HORIZONS)
    @test nrow(df) == L * nhor * nlev
    @test nrow(df) == length(LOCATIONS) * 4 * 23

    @test all(v -> v in QUANTILE_LEVELS, df.output_type_id)
    @test all(>=(0.0), df.value)
    @test all(==(TARGET), df.target)
    @test all(==("quantile"), df.output_type)
    @test all(==("nfidd-test"), df.model_id)
    @test all(==(data.origin_date), df.origin_date)
    @test issubset(Set(df.location), Set(LOCATIONS))

    @test all(
        row -> row.target_end_date ==
               row.origin_date + Day(7 * row.horizon),
        eachrow(df),
    )

    for loc in unique(df.location), h in unique(df.horizon)
        mask = (df.location .== loc) .& (df.horizon .== h)
        sub = sort(df[mask, :], :output_type_id)
        @test issorted(sub.value)
    end
end

@testset "forecast_quantiles: non-contiguous horizons map by value" begin
    # Regression: vals is filled by `project` indexed by horizon VALUE
    # (column h for h in 1:maximum(horizons)), so the output loop must
    # read column h, not the position within `horizons`. With a subset
    # like [1, 3, 4] these differ, and reading by position would attach
    # the wrong horizon's forecast under a correctly-labelled row.
    data = synthetic_data()
    chain = [(dummy=1.0,)]

    # Deterministic: column h carries the constant value (h + 1) on the
    # natural scale, so every quantile for horizon h must equal h + 1.
    perh_project(draw, d, horizons) =
        [to_scale(h + 1.0, :log) for _ in 1:d.L, h in 1:maximum(horizons)]

    df = forecast_quantiles(chain, data, "nfidd-hz";
        horizons=[1, 3, 4], project=perh_project)

    @test Set(df.horizon) == Set([1, 3, 4])
    for h in (1, 3, 4)
        vals_h = df[df.horizon .== h, :value]
        @test all(v -> isapprox(v, h + 1.0; atol=1e-8), vals_h)
    end
end

@testset "forecast_quantiles: default_project" begin
    data = synthetic_data()
    L = data.L
    W = data.W
    ndraws = 40
    chain = [
        (
            seasonal=fill(log(3.0 + 0.01 * d), W, L),
            ar_coef=fill(0.5, L),
            resid_sd=fill(0.1, L),
            last_resid=zeros(L),
        ) for d in 1:ndraws
    ]

    df = forecast_quantiles(chain, data, "nfidd-default")

    @test nrow(df) == L * length(HORIZONS) * length(QUANTILE_LEVELS)
    @test all(>=(0.0), df.value)
    for loc in unique(df.location), h in unique(df.horizon)
        mask = (df.location .== loc) .& (df.horizon .== h)
        sub = sort(df[mask, :], :output_type_id)
        @test issorted(sub.value)
    end
end

@testset "forecast_quantiles: DataFrame-backed fit" begin
    data = synthetic_data()
    ndraws = 30
    raw = DataFrame(scale=1.0 .+ 0.03 .* (1:ndraws))

    trivial_project(draw, d, horizons) =
        fill(log(3.0), d.L, maximum(horizons)) .* draw.scale

    df = forecast_quantiles(raw, data, "nfidd-test2";
        project=trivial_project)

    @test nrow(df) == data.L * length(HORIZONS) * length(QUANTILE_LEVELS)
    @test all(>=(0.0), df.value)
end
