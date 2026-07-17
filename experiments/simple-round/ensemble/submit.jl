#!/usr/bin/env julia
# submission driver for the ensemble's winning combination (`ens-mean`,
# see generate.jl / score.txt): averages the same four LOCKED member
# definitions (ar6, ar6bf, climatology, ses) across ALL FIVE seasons
# (1,2 validation + 3,4,5 held-out test), for a hub submission.
#
# generate.jl itself is validation-seasons-only by design (the member
# designs and the ens-mean combination were selected there, on
# validation only, per docs/contracts.md experimental integrity). This
# driver reuses those exact functions/constants unchanged (via
# `include`, which is a no-op for generate.jl's own `main()` since its
# `abspath(PROGRAM_FILE) == @__FILE__` guard checks THIS script's path,
# not generate.jl's) and only extends the generation loop to the
# additional origin dates in seasons 3-5, mirroring
# `submissions/seabbs_bot-ar6bf/generate_forecasts.jl`'s discipline:
# `allow_test_season=true` on `training_splits` governs the TUNING gate
# (no test-season data may be used to pick a model or combination), not
# per-origin vintage generation -- every split's fit is still capped at
# its own forecast origin, so generating forecasts for a test-season
# split never trains on or tunes against that season.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> submit.jl <hub_path>
# With no `hub_path`, builds and returns the forecast table without
# writing anything (dry run).

include(joinpath(@__DIR__, "generate.jl"))

const SUBMIT_MODEL_ID = "seabbs_bot-simpens"
const ALL_SEASONS = (1, 2, 3, 4, 5)

"""
    build_all_season_members(profile, versions_full) -> Dict{String,DataFrame}

Same as `generate.jl`'s `build_member_forecasts`, but over `ALL_SEASONS`
(validation + held-out test) instead of `VALIDATION_ONLY`, using
`allow_test_season=true` for the test seasons.
"""
function build_all_season_members(profile, versions_full)
    rng = MersenneTwister(SEED)
    rows = Dict(m => _empty_rows() for m in MEMBER_NAMES)
    for season in ALL_SEASONS
        splits = training_splits(
            season; allow_test_season=(season in TEST_SEASONS),
        )
        for split in splits
            data = build_model_data(
                split; Dmax=DMAX, transform=TRANSFORM, window_weeks=104,
                versions=versions_full,
            )
            data_bf = deepcopy(data)
            apply_backfill_correction!(data_bf, profile)
            origin = data.origin_date
            for (li, loc) in enumerate(LOCATIONS)
                y = Float64.(data.Y[:, li])
                y_bf = Float64.(data_bf.Y[:, li])

                coef, resid_sd = fit_ar(y, AR_ORDER)
                paths_ar6 = simulate_paths(
                    y, coef, resid_sd, AR_ORDER, HORIZONS, NPATHS; rng=rng,
                )
                coef_bf, resid_sd_bf = fit_ar(y_bf, AR_ORDER)
                paths_ar6bf = simulate_paths(
                    y_bf, coef_bf, resid_sd_bf, AR_ORDER, HORIZONS, NPATHS;
                    rng=rng,
                )
                level, resid_sd_ses, _ = fit_ses(y, SES_ALPHAS)
                paths_ses = simulate_ses_paths(
                    level, resid_sd_ses, HORIZONS, NPATHS; rng=rng,
                )

                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    climo_vals = climatology_quantiles(
                        split, loc, target_end, TRANSFORM; band=CLIMO_BAND,
                        min_n=CLIMO_MIN_N,
                    )
                    for (qi, q) in enumerate(QUANTILE_LEVELS)
                        nat_ar6 = max(
                            from_scale(quantile(paths_ar6[h], q), TRANSFORM),
                            0.0,
                        )
                        nat_ar6bf = max(
                            from_scale(quantile(paths_ar6bf[h], q), TRANSFORM),
                            0.0,
                        )
                        nat_ses = max(
                            from_scale(quantile(paths_ses[h], q), TRANSFORM),
                            0.0,
                        )
                        push!(rows["ar6"], ("ar6", loc, origin, h,
                            target_end, TARGET, "quantile", q, nat_ar6))
                        push!(rows["ar6bf"], ("ar6bf", loc, origin, h,
                            target_end, TARGET, "quantile", q, nat_ar6bf))
                        push!(rows["ses"], ("ses", loc, origin, h,
                            target_end, TARGET, "quantile", q, nat_ses))
                        push!(rows["climatology"], ("climatology", loc,
                            origin, h, target_end, TARGET, "quantile", q,
                            climo_vals[qi]))
                    end
                end
            end
        end
    end
    return rows
end

function main()
    hub_path = length(ARGS) >= 1 ? ARGS[1] : nothing
    t0 = time()

    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=DELAY_CUTOFF,
        min_support=MIN_SUPPORT,
    )

    member_rows = build_all_season_members(profile, versions_full)
    all_members = vcat((member_rows[m] for m in MEMBER_NAMES)...)
    ensemble = combine_members(
        all_members, :mean; model_id=SUBMIT_MODEL_ID,
    )

    dt = round(time() - t0; digits=2)
    n_origins = length(unique(ensemble.origin_date))
    println("built $(nrow(ensemble)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    if hub_path !== nothing
        write_submission(ensemble, hub_path)
        write_metadata(
            SUBMIT_MODEL_ID, hub_path;
            team_abbr="seabbs_bot", model_abbr="simpens", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return ensemble
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
