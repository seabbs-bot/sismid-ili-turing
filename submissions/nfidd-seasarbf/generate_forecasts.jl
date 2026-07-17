#!/usr/bin/env julia
# nfidd-seasarbf -- full 5-season submission driver for the round-1
# winner of the wide simple round: the seasoncombo "core" model
# (pooled seasonal climatology + per-location AR(6) + backfill
# correction). Validation (seasons 1,2) mean WIS 0.2781, sd 0.334 --
# the best seasonality+backfill model of the round, 22.5% under
# seabbs_bot-ar6bf (0.359).
#
# This is a thin driver: it `include`s the locked, validation-scored
# sweep code from experiments/simple-round/seasoncombo/generate.jl
# (copied verbatim into _seasoncombo_lib.jl, with the single change of
# threading allow_test_season through training_splits so the held-out
# test seasons 3-5 can be forecast for full hub coverage). The sweep's
# main() is guarded by `abspath(PROGRAM_FILE) == @__FILE__`, so
# including it here defines its functions/constants without running the
# validation sweep.
#
# Selection used validation seasons (1,2) ONLY. Coverage spans all 5
# seasons: the seasonal shape is fit to season_year <= 2016 and the
# backfill profile to season_year <= 2016 (both strictly before the
# test seasons, season_year >= 2017), and every split is fit only on
# the vintage data available at its own forecast origin -- no test-
# season data enters tuning or fitting.
#
# Usage:
#   julia --project=<repo> \
#     submissions/nfidd-seasarbf/generate_forecasts.jl <hub_path>

const _LIB = joinpath(@__DIR__, "_seasoncombo_lib.jl")
include(_LIB)

const MODEL_ID = "nfidd-seasarbf"
const ALL_SEASONS = (1, 2, 3, 4, 5)

function build_full_submission()
    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")
    ones_amp = ones(length(LOCATIONS))

    # Pooled seasonal shape and backfill revision profile, both built
    # from pre-test data only -- identical to the sweep's main().
    profile = build_seasonal_profile(
        hist; transform=TRANSFORM, max_season_year=MAX_TRAIN_SEASON_YEAR,
        min_support=MIN_SUPPORT, smooth_window=SMOOTH_WINDOW,
    )
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= MAX_TRAIN_SEASON_YEAR, :,
    ]
    backfill_profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_WINDOW,
        min_support=5, mode=BF_MODE, pooled=false, stat=BF_STAT,
    )

    forecast = build_forecast_table(
        ALL_SEASONS, versions_full, profile, ones_amp;
        backfill_profile=backfill_profile, backfill_window=BF_WINDOW,
        model_id=MODEL_ID,
    )
    return forecast
end

function main()
    hub_path = length(ARGS) >= 1 ? ARGS[1] : nothing
    t0 = time()
    forecast = build_full_submission()
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    if hub_path !== nothing
        write_submission(forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="nfidd", model_abbr="seasarbf", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
