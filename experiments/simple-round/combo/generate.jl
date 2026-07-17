#!/usr/bin/env julia
# generate.jl -- winning "best combination" candidate for the
# simple-round LIGHT + ANALYTIC family: per-location AR(4), fit
# jointly with a ridge-regularised 2-harmonic seasonal term, on top of
# the seabbs_bot-ar6bf backfill correction. All three ingredients are
# fit in a single OLS/ridge regression per (split, location); see
# `search_grid.jl` (this directory) for every helper this file reuses
# and for the full grid search that selected this combination.
#
# WHY AR(4), not AR(6) or AR(8): the grid (search_grid.jl, run on
# validation seasons 1-2) found AR order barely matters once the
# seasonal term and backfill correction are both present -- mean WIS
# is flat to three sig figs across AR(4)/(6)/(8) (0.3349 / 0.3347 /
# 0.3336). AR(4) has the LOWEST SD of the three (0.4200 vs 0.4237 vs
# 0.4254) and the fewest per-location parameters (9 vs 11 vs 13,
# against ~100/98/96 fitted observations per split under
# `window_weeks=104`), so it is picked deliberately over the
# marginally-lower-mean AR(8): a ~0.3% mean-WIS gain from AR(8) is not
# worth the higher SD and higher parameter count on a two-season
# training window (see docs/contracts.md "watch overfitting" framing
# and score.txt in this directory for the full grid table).
#
# Baseline beaten: seabbs_bot-ar6bf (AR(6) + backfill only), mean WIS
# 0.359, SD 0.452 (submissions/seabbs_bot-ar6bf/README.md). This
# candidate: mean WIS 0.3349, SD 0.4200 -- about 6.7% lower mean WIS
# AND lower SD, on the same validation seasons. See score.txt for the
# per-location/season/horizon breakdown.
#
# Deliberately LIGHT + ANALYTIC (no Turing), like nfidd-ar6 and
# seabbs_bot-ar6bf: CSV/DataFrames/Statistics/LinearAlgebra only.
# SELECTED AND SCORED ON VALIDATION SEASONS (1, 2) ONLY
# (docs/contracts.md experimental integrity): the model (AR order,
# backfill profile, seasonal ridge lambda) is locked from the
# validation-only grid search in `search_grid.jl`, never touching
# seasons 3-5. This driver then GENERATES forecasts for all five
# seasons for hub submission (validation 1-2 plus held-out test 3-5,
# `allow_test_season=true` inside `build_forecast_table`), mirroring
# `seabbs_bot-ar6bf`: each split is still just a per-origin vintage
# fit capped at its own forecast origin, so this never trains on or
# tunes against the test seasons -- it only forecasts them with the
# already-locked model. The printed validation WIS below is computed
# on the season-1/2 subset of that same full table (`season_year <=
# 2016`), not a separate fit, so it matches search_grid.jl exactly.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl [hub_path]
# With `hub_path` given, writes a hub-format submission (all 5
# seasons) under `<hub_path>/model-output/simple-combo-ar4bf-sn/` plus
# matching model-metadata, exactly like nfidd-ar6/seabbs_bot-ar6bf do
# for their own coverage.

include("search_grid.jl")  # fit_ar_seasonal, simulate_paths, backfill
                            # helpers, build_forecast_table, scoring;
                            # its own `main()` is guarded and not run
                            # by this `include`

const MODEL_ID = "simple-combo-ar4bf-sn"
const AR_ORDER = 4
const USE_BACKFILL = true
const USE_SEASONAL = true

function main()
    t0 = time()
    hub_path = length(ARGS) >= 1 ? ARGS[1] : nothing

    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=DELAY_CUTOFF,
        min_support=MIN_SUPPORT,
    )

    forecast = build_forecast_table(
        (1, 2, 3, 4, 5), MODEL_ID, AR_ORDER, USE_BACKFILL, USE_SEASONAL,
        profile, versions_full,
    )
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    # Validation-only subset (seasons 1-2, season_year <= 2016) for the
    # printed WIS -- test seasons 3-5 are generated above but never
    # scored/selected on here (docs/contracts.md experimental integrity).
    validation_forecast = forecast[
        season_year.(forecast.origin_date) .<= 2016, :,
    ]
    truth = load_oracle(HUB_PATH)
    scored = score_forecasts(validation_forecast, truth; scale=:natural)
    summ = wis_summary(scored)
    println(
        "validation WIS: mean=$(round(summ.mean_wis[1]; digits=4)) " *
        "sd=$(round(summ.sd_wis[1]; digits=4)) " *
        "n_tasks=$(summ.n_tasks[1])",
    )

    if hub_path !== nothing
        write_submission(forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="seabbs_bot", model_abbr="combo", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
