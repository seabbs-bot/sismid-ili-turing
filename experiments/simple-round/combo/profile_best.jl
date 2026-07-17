#!/usr/bin/env julia
# profile_best.jl -- per-location and per-season/horizon breakdown of
# the winning grid cell from search_grid.jl (AR(4) + backfill +
# seasonal) against the seabbs_bot-ar6bf baseline (AR(6) + backfill,
# no seasonal, mean WIS 0.359). Re-fits both tables (cheap: seconds)
# rather than threading `tables` out of `search_grid.jl`'s `main`, to
# keep this script runnable standalone.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> profile_best.jl

include("search_grid.jl")  # brings in every helper + constant above,
                            # but does NOT run its `main` (guarded by
                            # `abspath(PROGRAM_FILE) == @__FILE__`)

function season_of_origin(origin_date::Date)
    sy = season_year(origin_date)
    return sy == 2015 ? 1 : 2
end

function main_profile()
    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=DELAY_CUTOFF,
        min_support=MIN_SUPPORT,
    )
    truth = load_oracle(HUB_PATH)

    baseline_fc = build_forecast_table(
        (1, 2), "baseline-ar6bf", 6, true, false, profile, versions_full,
    )
    best_fc = build_forecast_table(
        (1, 2), "best-ar4bf-sn", 4, true, true, profile, versions_full,
    )

    baseline_scored = score_forecasts(baseline_fc, truth; scale=:natural)
    best_scored = score_forecasts(best_fc, truth; scale=:natural)

    baseline_scored.season = season_of_origin.(baseline_scored.origin_date)
    best_scored.season = season_of_origin.(best_scored.origin_date)

    println("=== Overall ===")
    for (name, scored) in (
        ("baseline (AR6+backfill)", baseline_scored),
        ("best (AR4+backfill+seasonal)", best_scored),
    )
        println(
            "$(name): mean_wis=$(round(mean(scored.wis); digits=4)) " *
            "sd_wis=$(round(std(scored.wis); digits=4))",
        )
    end

    println("\n=== By location (mean WIS, best - baseline) ===")
    by_loc_base = combine(
        groupby(baseline_scored, :location), :wis => mean => :wis_base,
    )
    by_loc_best = combine(
        groupby(best_scored, :location), :wis => mean => :wis_best,
    )
    by_loc = innerjoin(by_loc_base, by_loc_best; on=:location)
    by_loc.delta = by_loc.wis_best .- by_loc.wis_base
    sort!(by_loc, :delta)
    for row in eachrow(by_loc)
        println(
            "$(rpad(row.location, 14)) base=$(round(row.wis_base; digits=4)) " *
            "best=$(round(row.wis_best; digits=4)) " *
            "delta=$(round(row.delta; digits=4))",
        )
    end

    println("\n=== By season (mean WIS, best - baseline) ===")
    by_season_base = combine(
        groupby(baseline_scored, :season), :wis => mean => :wis_base,
    )
    by_season_best = combine(
        groupby(best_scored, :season), :wis => mean => :wis_best,
    )
    by_season = innerjoin(by_season_base, by_season_best; on=:season)
    by_season.delta = by_season.wis_best .- by_season.wis_base
    for row in eachrow(by_season)
        println(
            "season $(row.season): base=$(round(row.wis_base; digits=4)) " *
            "best=$(round(row.wis_best; digits=4)) " *
            "delta=$(round(row.delta; digits=4))",
        )
    end

    println("\n=== By horizon (mean WIS, best - baseline) ===")
    by_h_base = combine(
        groupby(baseline_scored, :horizon), :wis => mean => :wis_base,
    )
    by_h_best = combine(
        groupby(best_scored, :horizon), :wis => mean => :wis_best,
    )
    by_h = innerjoin(by_h_base, by_h_best; on=:horizon)
    by_h.delta = by_h.wis_best .- by_h.wis_base
    sort!(by_h, :horizon)
    for row in eachrow(by_h)
        println(
            "h=$(row.horizon): base=$(round(row.wis_base; digits=4)) " *
            "best=$(round(row.wis_best; digits=4)) " *
            "delta=$(round(row.delta; digits=4))",
        )
    end

    paired = innerjoin(
        baseline_scored[:, [:location, :origin_date, :horizon, :wis]],
        best_scored[:, [:location, :origin_date, :horizon, :wis]];
        on=[:location, :origin_date, :horizon],
        renamecols="_base" => "_best",
    )
    n_improved = count(r -> r.wis_best < r.wis_base, eachrow(paired))
    n_total = nrow(baseline_scored)
    println(
        "\nTask-level: $(n_improved) of $(n_total) " *
        "($(round(100 * n_improved / n_total; digits=1))%) improved",
    )
end

main_profile()
