#!/usr/bin/env julia
# gen_conformal.jl -- TEST-SEASON (3, 4, 5) forecasts for the "honest
# frontier" (conformal-pooled + window208 + width0.9, validation WIS
# 0.2730), for the held-out test-season evaluation (reports/
# test-evaluation.md).
#
# Reuses experiments/simple-round/conformal-pooled/generate.jl's
# `build_forecast_table` VERBATIM by `include`ing that file: it already
# rebuilds the pooled seasonal profile and the backfill revision
# profile PER SPLIT from only strictly-prior data (leak-free), and its
# own constants already default to WINDOW_WEEKS=208, WIDTH_SCALE=0.9,
# POOL_WEIGHT=0.3 -- exactly the "honest frontier" combination recorded
# in submissions/README.md's leak-free leaderboard (val WIS 0.2730).
#
# Unlike round2-stack, this driver's split-conformal calibration pool is
# WALK-FORWARD state carried between splits (see that file's module
# docstring): every split matures pending calibration tasks using only
# information knowable as of its own origin, then adds its own tasks to
# the pool for later origins to mature. This means seasons MUST be
# walked in ascending chronological order from season 1 -- generating
# TEST_SEASONS alone, without first walking through the validation
# seasons, would start the test seasons with an empty/cold calibration
# pool instead of the one they actually had at generation time. So this
# script builds all FIVE seasons (matching how the model would actually
# be run/submitted) and only FILTERS DOWN to the test-season origin
# dates afterward, for scoring -- the validation-season rows are
# discarded here, not fit to anything new.
#
# Usage: julia --project=<sismid-ili-turing repo> gen_conformal.jl

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
const OUT_DIR = joinpath(PKG_DIR, "experiments", "test-eval", "out")
# NOTE: deliberately not named `HERE` -- the included file below
# defines its own `const HERE = @__DIR__`, which would collide.

# Defines build_forecast_table, build_pooled_seasonal,
# build_revision_profile, apply_backfill_correction!, PendingTask,
# calibrated_quantiles, MODEL_ID ("seabbs_bot-conformal-pooled"),
# WINDOW_WEEKS (208), WIDTH_SCALE (0.9), POOL_WEIGHT (0.3), etc.
# `main()` is guarded by `abspath(PROGRAM_FILE) == @__FILE__`, false
# here, so including this file only defines functions.
include(joinpath(
    PKG_DIR, "experiments", "simple-round", "conformal-pooled",
    "generate.jl",
))

function main()
    t0 = time()
    versions_full = load_series("flu_data_hhs_versions")
    hist_all = load_series("flu_data_hhs")
    vidx = build_vintage_index(versions_full)

    # Walk ALL FIVE seasons in order -- the rolling calibration pool
    # depends on it (see module docstring above and in the included
    # file). Uses the included file's own defaults: WINDOW_WEEKS=208,
    # POOL_WEIGHT=0.3, WIDTH_SCALE=0.9 (the honest-frontier combo).
    forecast = build_forecast_table(
        (1, 2, 3, 4, 5), versions_full, hist_all, vidx;
        window_weeks=WINDOW_WEEKS, pool_weight=POOL_WEIGHT,
        width_scale=WIDTH_SCALE,
    )
    dt = round(time() - t0; digits=2)
    println("built $(nrow(forecast)) rows across " *
            "$(length(unique(forecast.origin_date))) origin date(s) " *
            "(all 5 seasons) in $(dt)s")

    # Filter down to TEST-season (3, 4, 5) origin dates only -- this is
    # a REPORTING filter, not a refit: the validation-season rows above
    # already exist and are simply dropped here, never re-used.
    test_origins = Set{Date}()
    for season in TEST_SEASONS
        for split in training_splits(season; allow_test_season=true)
            push!(test_origins, maximum(split.origin_date))
        end
    end
    test_forecast = forecast[in.(forecast.origin_date, Ref(test_origins)), :]
    n_origins = length(unique(test_forecast.origin_date))
    println("filtered to $(nrow(test_forecast)) rows across " *
            "$(n_origins) TEST-season origin date(s)")

    outpath = joinpath(OUT_DIR, "conformal-pooled.csv")
    CSV.write(outpath, test_forecast)
    println("wrote $(outpath)")
    return test_forecast
end

main()
