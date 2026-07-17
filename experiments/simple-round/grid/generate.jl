#!/usr/bin/env julia
# generate.jl -- winning cell of the simple-round SYSTEMATIC GRID
# (search_grid.jl, this directory): AR(6), fit to the residual LEVEL
# (no differencing) after subtracting a POOLED seasonal shape from the
# multiplicative/w6/per-loc/median-backfilled series, with each
# location's AR(6) coefficients blended 50/50 with a fullpool
# common-dynamics anchor. See search_grid.jl for every helper this file
# reuses and ranked_table.txt for the full 48-cell grid.
#
# WHY this cell, not AR(8) or AR(12) (the runners-up at 0.3008/0.3012):
# AR order's main effect across the grid is essentially flat (mean_wis
# 0.3501/0.3516/0.3530 averaged over the other 16 cells each order
# appears in -- see ranked_table.txt's main-effects section), so once
# backfill + pooled season + AR pooling are all present, extra AR lags
# buy nothing worth the added parameters; AR(6) is both the lowest-mean
# AND has among the lowest task-count-adjusted parameter overhead of
# the top three.
#
# WHY diff=false: differencing is the single worst factor in the whole
# grid by a wide margin (main effect mean_wis 0.3321 undifferenced vs.
# 0.3710 differenced -- see ranked_table.txt). Every one of the 24
# differenced cells scores worse than every one of the 24 undifferenced
# cells at the same (order, backfill, season, pool) setting; the
# differenced AR has to be reconstructed via a per-path cumulative sum
# (search_grid.jl's `simulate_paths`), which compounds each step's
# innovation variance across the horizon instead of the level AR's
# self-correcting mean reversion. Differencing this deseasonalised,
# already-fairly-stationary residual trades away that mean reversion
# for no accuracy gain.
#
# WHY pooled season is in, non-negotiably: season has the largest main
# effect of all five factors by a wide margin (0.3856 -> 0.3176, ~18%
# -- see ranked_table.txt), and every one of the top 12 ranked cells
# has season=true; the top season=false cell (order=12, bf=true,
# pool=true, diff=false, 0.3447) is not even close to competitive. This
# is also a hub requirement independent of the ranking, so there is no
# actual tension here between "what the grid says" and "what must
# ship": the grid's own top cell already has it on.
#
# Baseline beaten: experiments/simple-round/seasonpool (AR(6) +
# additive/per-loc/median backfill + pooled season, no AR pooling),
# mean WIS 0.3049. This cell: mean WIS 0.2997, SD 0.3826 -- switching
# the backfill correction to the winning multiplicative/w6 cell
# (experiments/simple-round/backfill/score.txt) and adding fullpool
# AR-coefficient shrinkage (w=0.5, experiments/simple-round/pool) on
# top of seasonpool's design together buy a further ~1.7% mean-WIS
# reduction. Also beats the ar-order sweep's AR(12)+backfill pick
# (0.3518) by ~14.8% and the earlier per-location-Fourier `combo`
# candidate (0.3349) by ~10.5%.
#
# Deliberately LIGHT + ANALYTIC (no Turing), like every other
# simple-round script: CSV/DataFrames/Statistics/LinearAlgebra only.
# SCORED ON VALIDATION SEASONS (1, 2) ONLY (docs/contracts.md
# experimental integrity) -- this is an experiment/selection artefact;
# with a `hub_path` argument it writes a hub-format submission covering
# whatever seasons `build_forecast_table` is called on (here, still
# only the validation seasons -- change `VALIDATION_SEASONS` to the
# full set at the point this becomes an actual test-phase submission
# driver, not before).
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl [hub_path]

include("search_grid.jl")  # every helper, plus the module-level
                           # constants (TRANSFORM, DMAX, N_HARMONICS,
                           # BF_MODE/WINDOW/POOLED/STAT, POOL_WEIGHT,
                           # ...); its own `main()` is guarded and not
                           # run by this `include`

const MODEL_ID = "simple-grid-ar6bf-sn-pl"
const AR_ORDER = 6
const USE_POOL = true
const USE_DIFF = false

function main()
    t0 = time()
    hub_path = length(ARGS) >= 1 ? ARGS[1] : nothing

    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_WINDOW,
        min_support=BF_MIN_SUPPORT, mode=BF_MODE, pooled=BF_POOLED,
        stat=BF_STAT,
    )

    history = load_series("flu_data_hhs")
    shape_coef = fit_pooled_shape(
        history; transform=TRANSFORM, K=N_HARMONICS, period=SEASON_PERIOD,
        cutoff_year=CLIMATOLOGY_YEAR,
    )

    split_caches = build_split_cache(
        versions_full, profile, shape_coef; use_backfill=true,
        use_season=true,
    )
    forecast = build_forecast_table(
        MODEL_ID, split_caches, AR_ORDER; use_pool=USE_POOL,
        use_diff=USE_DIFF,
    )
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    truth = load_oracle(HUB_PATH)
    scored = score_forecasts(forecast, truth; scale=:natural)
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
            team_abbr="seabbs_bot", model_abbr="grid", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
