#!/usr/bin/env julia
# gen_seasstack.jl -- TEST-SEASON (3, 4, 5) forecasts for the "seasstack
# full stack" (pooled seasonal + log transform + Student-t intervals +
# AR-coefficient pooling w=0.9), for the held-out test-season evaluation
# (reports/test-evaluation.md).
#
# Reuses experiments/simple-round/round2-stack/generate.jl's
# `build_forecast_table` VERBATIM by `include`ing that file: it already
# rebuilds both the pooled seasonal profile and the backfill revision
# profile PER SPLIT from only `origin_date`/`as_of` strictly before that
# split's own forecast origin (leak-free by construction -- see that
# file's module docstring and its "LEAKAGE FIX" note in score.txt). This
# combo (transform=:log, innovation=:student_t, pool_w=0.9) is exactly
# the "full stack, LEAK-FREE (winner)" row scored 0.2891 on validation
# in round2-stack/score.txt, i.e. the model recorded as "seasstack full
# stack" in submissions/README.md's honest leak-free leaderboard.
#
# Each split in `build_forecast_table` is independent (no walk-forward
# state carried between splits), so this can be called directly with
# `seasons = TEST_SEASONS` -- no need to also process the validation
# seasons first.
#
# Usage: julia --project=<sismid-ili-turing repo> gen_seasstack.jl

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
const OUT_DIR = joinpath(PKG_DIR, "experiments", "test-eval", "out")
# NOTE: deliberately not named `HERE` -- the included file below
# defines its own `const HERE = @__DIR__`, which would collide.

# Defines build_forecast_table, build_seasonal_profile,
# build_revision_profile, apply_backfill_correction!, fit_ar,
# fit_ar_pooled, simulate_paths, load_oracle, coverage, score_one, main,
# etc. `main()` is guarded by `abspath(PROGRAM_FILE) == @__FILE__`,
# which is false here (PROGRAM_FILE is this script), so including this
# file only defines functions -- it does not run round2-stack's own
# sweep.
include(joinpath(
    PKG_DIR, "experiments", "simple-round", "round2-stack", "generate.jl",
))

const MODEL_ID = "seabbs_bot-seasstack"

function main()
    t0 = time()
    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")

    forecast = build_forecast_table(
        TEST_SEASONS, hist, versions_full;
        transform=:log, innovation=:student_t, pool_w=0.9,
        model_id=MODEL_ID,
    )
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) TEST-season " *
            "origin date(s) in $(dt)s")
    outpath = joinpath(OUT_DIR, "seabbs_bot-seasstack.csv")
    CSV.write(outpath, forecast)
    println("wrote $(outpath)")
    return forecast
end

main()
