# End-to-end integration smoke: fit -> forecast -> score on ONE validation
# split, through the assembled `SismidILITuring` package (not bare
# includes). Proves the base_model -> posterior_draws -> base_project ->
# forecast_quantiles -> score_forecasts pipeline is coherent and produces a
# valid hub table. Kept tiny (few draws, short window) so it runs fast.

using Test
using DataFrames
using Dates
using Random
using SismidILITuring

Random.seed!(20260717)

@testset "integration: fit -> forecast -> score (one split)" begin
    # Season 1 (a validation season), last split = latest forecast origin.
    split = training_splits(1)[end]
    data = build_model_data(split; Dmax=4, window_weeks=52,
                            transform=:log1p)

    @test size(data.Y) == (data.T, data.L)   # (T x L) orientation
    @test data.L == length(LOCATIONS)         # 11 locations

    model = base_model(data; transform=:log1p)
    fit = fit_pathfinder(model; ndraws=80)
    gdraws = generated_draws(model, fit)

    @test gdraws isa Vector{<:NamedTuple}
    @test !isempty(gdraws)
    # base_project's required fields are present on each draw.
    for key in (:mu0, :mu_w, :delta, :season_eff, :phi, :sigma_ar,
                :residual)
        @test haskey(gdraws[1], key)
    end

    fc = forecast_quantiles(gdraws, data, "nfidd-turing";
                            project=base_project)

    # --- hub schema and shape ---
    expected_cols = [
        :model_id, :location, :origin_date, :horizon, :target_end_date,
        :target, :output_type, :output_type_id, :value,
    ]
    @test propertynames(fc) == expected_cols

    # One origin: 11 locations x 4 horizons x 23 quantiles.
    @test nrow(fc) == length(LOCATIONS) * length(HORIZONS) *
        length(QUANTILE_LEVELS)
    @test nrow(fc) == 11 * 4 * 23
    @test length(unique(fc.origin_date)) == 1
    @test fc.origin_date[1] == data.origin_date
    @test Set(fc.location) == Set(LOCATIONS)
    @test all(==("quantile"), fc.output_type)
    @test all(==(TARGET), fc.target)
    @test all(v -> v in QUANTILE_LEVELS, fc.output_type_id)

    # --- non-negative values, non-decreasing across quantile levels ---
    @test all(isfinite, fc.value)
    @test all(>=(0.0), fc.value)
    for loc in unique(fc.location), h in unique(fc.horizon)
        mask = (fc.location .== loc) .& (fc.horizon .== h)
        sub = sort(fc[mask, :], :output_type_id)
        @test issorted(sub.value)
        @test sub.target_end_date[1] ==
            data.origin_date + Day(7 * h)
    end

    # --- score against the finalized (oracle) series ---
    truth = load_series("flu_data_hhs")
    truth_df = DataFrame(
        location=truth.location,
        target_end_date=truth.origin_date,
        value=truth.wili,
    )
    scored = score_forecasts(fc, truth_df; scale=:natural)
    @test nrow(scored) >= 2                    # a couple of oracle points
    @test all(isfinite, scored.wis)
    @test all(>=(0.0), scored.wis)

    summ = wis_summary(scored)
    @test nrow(summ) == 1
    @test isfinite(summ.mean_wis[1])

    println("example forecast row:")
    println(first(fc, 1))
    println("mean WIS over $(nrow(scored)) oracle tasks: ",
            summ.mean_wis[1])
end

println("test_integration.jl passed")
