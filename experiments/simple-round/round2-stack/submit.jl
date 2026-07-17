#!/usr/bin/env julia
# Submission driver for the round-2 stack winner
# (log + Student-t(df=10, scale=1.4) intervals + AR(6)-coefficient
# pooling w=0.9 on the deseasonalized residual, on top of the pooled
# seasonal climatology + non-monotonic backfill core). Validation WIS
# 0.2601 (round2-stack/score.txt); selected on validation seasons 1, 2
# only. This driver reuses that experiment's exact functions and just
# runs the winning combo across ALL FIVE seasons for a hub submission --
# each split is still a per-origin vintage fit capped at its own
# forecast origin (allow_test_season inside build_forecast_table), NOT
# training on the test seasons.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> submit.jl <hub_path>

include(joinpath(@__DIR__, "generate.jl"))  # defines everything; main()
                                            # is guarded, does not run here

const SUB_MODEL_ID = "seabbs_bot-seasstack"
const SUB_POOL_W = 0.9

function submit(hub_path)
    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= MAX_TRAIN_SEASON_YEAR, :,
    ]
    profile = build_seasonal_profile(
        hist; transform=:log, max_season_year=MAX_TRAIN_SEASON_YEAR,
        min_support=MIN_SUPPORT, smooth_window=SMOOTH_WINDOW,
    )
    bf_profile = build_revision_profile(
        training_versions; transform=:log, max_delay=BF_WINDOW,
        min_support=MIN_SUPPORT, mode=BF_MODE, stat=BF_STAT,
    )
    forecast = build_forecast_table(
        (1, 2, 3, 4, 5), versions_full, profile; transform=:log,
        backfill_profile=bf_profile, innovation=:student_t,
        pool_w=SUB_POOL_W, model_id=SUB_MODEL_ID,
    )
    write_submission(forecast, hub_path)
    write_metadata(
        SUB_MODEL_ID, hub_path;
        team_abbr="seabbs_bot", model_abbr="seasstack", designated=true,
    )
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    hub_path = length(ARGS) >= 1 ? ARGS[1] :
        error("usage: submit.jl <hub_path>")
    t0 = time()
    fc = submit(hub_path)
    dt = round(time() - t0; digits=1)
    println("wrote $(nrow(fc)) rows across " *
            "$(length(unique(fc.origin_date))) origin dates in $(dt)s " *
            "to $(hub_path)")
end
