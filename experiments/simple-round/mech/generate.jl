#!/usr/bin/env julia
# Mechanistic susceptible-depletion + renewal terms, stacked on the
# round-2 stack winner -- simple-round.
#
# Every model up to round2-stack shares one structural weakness: the AR
# on the deseasonalized residual mean-reverts toward the location's OWN
# unconditional mean and the pooled seasonal shape is a fixed, symmetric
# climatology repeated every year. Neither has a built-in reason for
# growth to slow and turn over near a peak that is bigger or smaller
# than a typical season's -- the model only "knows" the peak is coming
# because the climatology says so at that calendar week, not because
# the season's own trajectory has used up susceptibles. This especially
# hurts long-horizon (h=3,4) forecasts made just before a turnaround,
# and worst in an atypical season (2016/17, validation season 2, the
# harder of the two -- see round2-stack/score.txt's by-season table).
#
# Two mechanistic candidates, ANALYTIC (no Turing), sharing one state
# variable, tested ALONE (on top of the plain fourthroot/Gaussian/
# pool_w=0 "core" from seasoncombo, matching round2-stack's own
# ablation style) and STACKED on the round2-stack winner (log +
# Student-t(df=10,scale=1.4) + AR(6)-coefficient pooling w=0.9 on the
# deseasonalized residual; validation mean_wis=0.2891, the LEAK-FREE
# honest rescore -- see experiments/simple-round/round2-stack/
# score.txt's "LEAKAGE FIX" section, and note below that this file's
# own `build_season_totals` had the identical leak and is fixed the
# same way):
#
#   1. SUSCEPTIBLE-DEPLETION damping: at each simulated week, the AR(6)
#      prediction's departure from persistence (`pred - tail[end]`) is
#      shrunk by `(1 - cum_frac)^depletion_gamma`, where `cum_frac` is
#      cumulative natural-scale wILI so far this season (reset at each
#      within-window season boundary) divided by that location's
#      historical mean COMPLETE-season total. As a location's own
#      season fills up toward (or past) a typical total, the AR's own
#      growth/decay signal is progressively flattened toward pure
#      persistence -- a cheap stand-in for "fewer susceptibles left to
#      infect, so growth slows".
#   2. RENEWAL-like momentum: an additive drift term
#      `renewal_m * renewal_decay^h * (1 - cum_frac)^renewal_gamma`,
#      where `renewal_m` is the recent local growth rate (mean of the
#      last `RENEWAL_WINDOW` first differences of the deseasonalized
#      residual at the forecast origin), carried forward and decayed
#      both geometrically in `h` (a discrete recovery-rate analogue)
#      and by the SAME depletion signal as (1) -- a discrete, point-
#      estimate renewal-ish update, not a real Rt/renewal-equation
#      model with its own likelihood.
#
# `cum_frac` is seeded once per (split, location) from the observed,
# backfill-corrected history up to the forecast origin (`cum_to_date`),
# then updated PER SIMULATED PATH: each path's own just-simulated
# future weeks feed back into its own depletion state, so a path that
# happens to simulate an unusually big early season sees ITS OWN
# growth damped harder going forward, not a single shared deterministic
# trajectory.
#
# SCORED ON VALIDATION SEASONS (1, 2) ONLY (docs/contracts.md
# experimental integrity) against the local hub clone's oracle -- a
# tuning sweep, not a submission driver, UNLESS a `hub_path` argument
# is given, in which case the locked-in winning combo (`SUB_*`
# constants below) is ALSO run across all 5 seasons and written as a
# hub submission under model_id "nfidd-mech".
#
# LIGHT + ANALYTIC: reuses round2-stack/generate.jl's helpers
# (CSV/DataFrames/Dates/Statistics/Random/LinearAlgebra/Distributions
# only, no Turing).
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl [hub_path]
# always writes score.txt (the validation sweep) alongside this file;
# with `hub_path`, additionally writes a full 5-season hub submission
# under model_id "nfidd-mech".

include(joinpath(@__DIR__, "..", "round2-stack", "generate.jl"))
# ^ defines build_seasonal_profile, deseasonalize, build_revision_profile
# (both LEAK-FREE, taking a per-split `forecast_origin` -- see that
# file's "LEAKAGE FIX" note), apply_backfill_correction!, ar_design,
# fit_ar, fit_ar_pooled, resid_sd_for, simulate_paths,
# build_forecast_table, load_oracle, score_one, coverage, run_combo,
# plus AR_ORDER/NPATHS/SEED/DMAX/WINDOW_WEEKS/MIN_SUPPORT/
# SMOOTH_WINDOW/VALIDATION_ONLY/HUB_PATH/BF_*/T_DF/T_SCALE constants.
# Its own `main()` is guarded and not run by this `include`.

const RENEWAL_WINDOW = 3   # weeks of R-scale first differences averaged
                           # into the renewal momentum term
const DEPLETION_GAMMAS = (0.5, 1.0, 2.0)
const RENEWAL_DECAYS = (0.5, 0.7, 0.9)

# Locked-in mechanistic combo for the hub driver. IMPORTANT: per the
# (leak-free) sweep in score.txt, NO mechanistic combo beats the plain
# round2-stack winner on validation (best is "winner+depletion" at
# mean_wis=0.2941 vs 0.2891 plain -- see score.txt's "ranked" table and
# verdict). This is the LEAST-BAD mechanistic combo found (depletion
# damping only, no renewal term, at the default gamma=1.0), written to
# the hub anyway because a `nfidd-mech` candidate was requested for
# cross-model comparison -- NOT because it is recommended over round2-
# stack's own submission driver (experiments/simple-round/round2-stack/
# submit.jl).
const SUB_MODEL_ID = "nfidd-mech"
const SUB_USE_DEPLETION = true
const SUB_USE_RENEWAL = false
const SUB_DEPLETION_GAMMA = 1.0
const SUB_RENEWAL_DECAY = 0.7
const SUB_RENEWAL_GAMMA = 1.0

# ---------------------------------------------------------------------
# Season totals (susceptible-depletion normalisation)
# ---------------------------------------------------------------------

"""
    build_season_totals(hist, forecast_origin; min_weeks=40)
        -> Dict{String,Float64}

Historical mean COMPLETE-season total natural-scale wILI, per location,
estimated from `hist` restricted to seasons STRICTLY BEFORE the season
containing `forecast_origin` -- LEAK-FREE, rebuilt PER SPLIT like
`build_seasonal_profile`/`build_revision_profile` in round2-stack/
generate.jl (see that file's "LEAKAGE FIX" note: an earlier version of
THIS file built this once from a fixed `season_year <= 2016` cutoff,
which for a validation-season split included that same season's own
future weeks and the entire other validation season). Excluding the
current season entirely (not just `origin_date < forecast_origin`)
matters here specifically: a date filter alone would let a
forecast_origin late in a season's own in-progress total sneak in as
if it were a finished season, corrupting the "typical complete season"
denominator with a partial sum. A season/location is only counted if it
has at least `min_weeks` observed weeks (guards against genuinely
partial seasons at the start/end of the raw series). This is the
denominator of the susceptible-depletion fraction: how much of a
"typical" season's total wILI has already accumulated by the forecast
origin.
"""
function build_season_totals(
    hist::DataFrame, forecast_origin::Date; min_weeks::Int=40,
)
    cur_syear = season_year(forecast_origin)
    h = hist[
        (hist.origin_date .< forecast_origin) .&
        (season_year.(hist.origin_date) .< cur_syear), :,
    ]
    syear = season_year.(h.origin_date)
    totals = Dict{Tuple{String,Int},Float64}()
    counts = Dict{Tuple{String,Int},Int}()
    for i in eachindex(syear)
        key = (h.location[i], syear[i])
        totals[key] = get(totals, key, 0.0) + h.wili[i]
        counts[key] = get(counts, key, 0) + 1
    end
    per_loc = Dict{String,Vector{Float64}}()
    for (key, tot) in totals
        counts[key] < min_weeks && continue
        push!(get!(per_loc, key[1], Float64[]), tot)
    end
    return Dict(loc => mean(v) for (loc, v) in per_loc)
end

"""
    cum_to_date(data, l) -> Float64

Cumulative natural-scale wILI for location column `l`, summed from the
start of the within-window season containing the forecast origin up to
the forecast origin itself (the last row of `data.Y`), using the
backfill-corrected `data.Y` (the same series the AR(6) fit sees). This
seeds the susceptible-depletion state for forward simulation. Note the
first (partial) season inside a truncated `window_weeks` history has no
true season start in view, so its early weeks understate `cum_frac`; a
minor edge effect, not corrected here (see the not-stacked note at the
bottom of score.txt).
"""
function cum_to_date(data::ModelData, l::Int)
    T = data.T
    cur = data.season[T]
    t0 = T
    while t0 > 1 && data.season[t0 - 1] == cur
        t0 -= 1
    end
    cum = 0.0
    for t in t0:T
        y = data.Y[t, l]
        ismissing(y) && continue
        cum += max(from_scale(y, data.transform), 0.0)
    end
    return cum
end

# ---------------------------------------------------------------------
# Mechanistic path simulation
# ---------------------------------------------------------------------

"""
    simulate_paths_mech(y, coef, resid_sd, order, horizons, npaths;
        rng, innovation, t_df, t_scale, transform, level, profile,
        origin, season_total, cum0, use_depletion, depletion_gamma,
        use_renewal, renewal_m, renewal_decay, renewal_gamma)
        -> Dict{Int,Vector{Float64}}

`simulate_paths` (round2-stack) plus the two mechanistic terms, applied
IN R-SCALE (the deseasonalized residual the AR(6) operates on) at every
simulated step:

  - `use_depletion`: the AR(6) prediction's departure from persistence,
    `pred - tail[end]`, is scaled by `(1 - cum_frac)^depletion_gamma`
    (clamped >= 0).
  - `use_renewal`: adds
    `renewal_m * renewal_decay^h * (1 - cum_frac)^renewal_gamma`.

`cum_frac = cum_state / season_total`. `cum_state` starts at `cum0`
(natural-scale wILI accumulated so far this season, `cum_to_date`) and
is updated EVERY simulated step, per path, by converting that step's
just-simulated R-scale value back to natural scale (`level` + that
step's seasonal offset from `profile`, inverse-`transform`) and adding
it in -- so `cum_frac` tracks each path's OWN simulated trajectory, not
one shared deterministic curve. If `season_total` is `NaN` or <= 0 (no
historical total for this location), both mechanistic terms are
disabled for the call regardless of the `use_*` flags, and this
reproduces `simulate_paths` bit-for-bit (identical draw order).
"""
function simulate_paths_mech(
    y::AbstractVector{Float64}, coef::Vector{Float64}, resid_sd::Float64,
    order::Int, horizons, npaths::Int;
    rng::Random.AbstractRNG, innovation::Symbol,
    t_df::Int=T_DF, t_scale::Float64=T_SCALE, transform::Symbol,
    level::Float64, profile::Dict{Int,Float64}, origin::Date,
    season_total::Float64, cum0::Float64,
    use_depletion::Bool, depletion_gamma::Float64,
    use_renewal::Bool, renewal_m::Float64, renewal_decay::Float64,
    renewal_gamma::Float64,
)
    tdist = TDist(t_df)
    vscale = sqrt((t_df - 2) / t_df)
    innov_sd = innovation == :student_t ? resid_sd * vscale * t_scale : resid_sd

    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    tail0 = y[(end - order + 1):end]
    have_total = !isnan(season_total) && season_total > 0
    mech_on = have_total && (use_depletion || use_renewal)
    s_by_h = Dict(h => get(profile, week_of_season(origin + Day(7 * h)), 0.0)
                  for h in 1:hmax)

    for s in 1:npaths
        tail = copy(tail0)
        cum_state = cum0
        for h in 1:hmax
            pred = coef[1]
            for lag in 1:order
                pred += coef[lag + 1] * tail[end - lag + 1]
            end

            mean_val = pred
            if mech_on
                frac = min(cum_state / season_total, 1.5)
                if use_depletion
                    damp = max(1 - frac, 0.0)^depletion_gamma
                    mean_val = tail[end] + (pred - tail[end]) * damp
                end
                if use_renewal
                    rdamp = max(1 - frac, 0.0)^renewal_gamma
                    mean_val += renewal_m * renewal_decay^h * rdamp
                end
            end

            innov = innovation == :student_t ?
                innov_sd * rand(rng, tdist) : innov_sd * randn(rng)
            val = mean_val + innov
            if h in horizons
                out[h][s] = val
            end
            push!(tail, val)
            popfirst!(tail)

            if mech_on
                nat = max(from_scale(val + level + s_by_h[h], transform), 0.0)
                cum_state += nat
            end
        end
    end
    return out
end

# ---------------------------------------------------------------------
# Forecast table builder: round2-stack core + the two mechanistic terms
# ---------------------------------------------------------------------

"""
    build_forecast_table_mech(seasons, hist, versions_full; transform,
        backfill_window, innovation, pool_w, use_depletion,
        depletion_gamma, use_renewal, renewal_decay, renewal_gamma,
        model_id) -> DataFrame

`build_forecast_table` (round2-stack, LEAK-FREE version: the pooled
seasonal profile, backfill revision profile, AND (this file's own
addition) the susceptible-depletion season totals are all rebuilt FRESH
per split, from only `hist`/`versions_full` rows strictly before that
split's own forecast origin -- see `build_seasonal_profile`/
`build_revision_profile` in round2-stack/generate.jl and
`build_season_totals` above) with `simulate_paths_mech` spliced in for
the two mechanistic terms. Every non-mechanistic argument means exactly
what it does in round2-stack's `build_forecast_table`;
`use_depletion=false, use_renewal=false` reproduces it bit-for-bit.
"""
function build_forecast_table_mech(
    seasons, hist::DataFrame, versions_full::DataFrame;
    transform::Symbol, backfill_window::Int=BF_WINDOW,
    innovation::Symbol=:gaussian, pool_w::Float64=0.0,
    use_depletion::Bool=false, depletion_gamma::Float64=1.0,
    use_renewal::Bool=false, renewal_decay::Float64=0.7,
    renewal_gamma::Float64=1.0, model_id::String,
)
    rng = MersenneTwister(SEED)
    rows = DataFrame(
        model_id=String[], location=String[], origin_date=Date[],
        horizon=Int[], target_end_date=Date[], target=String[],
        output_type=String[], output_type_id=Float64[], value=Float64[],
    )
    for season in seasons
        splits = training_splits(
            season; allow_test_season=(season in TEST_SEASONS),
        )
        for split in splits
            forecast_origin = maximum(split.origin_date)
            profile = build_seasonal_profile(
                hist, forecast_origin; transform=transform,
                min_support=MIN_SUPPORT, smooth_window=SMOOTH_WINDOW,
            )
            backfill_profile = build_revision_profile(
                versions_full, forecast_origin; transform=transform,
                max_delay=backfill_window, min_support=MIN_SUPPORT,
                mode=BF_MODE, stat=BF_STAT,
            )
            season_totals = build_season_totals(hist, forecast_origin)
            data = build_model_data(
                split; Dmax=DMAX, transform=transform,
                window_weeks=WINDOW_WEEKS, versions=versions_full,
            )
            apply_backfill_correction!(
                data, backfill_profile; mode=BF_MODE,
                delay_cutoff=backfill_window,
            )
            R, level = deseasonalize(data.Y, data.woy, profile)
            origin = data.origin_date
            L = data.L

            ys = [R[:, li] for li in 1:L]
            fits = [fit_ar(ys[li], AR_ORDER) for li in 1:L]
            coefs = [f[1] for f in fits]
            Xs = [f[2] for f in fits]
            yresps = [f[3] for f in fits]

            blended = if pool_w <= 0.0
                coefs
            else
                anchor = fit_ar_pooled(ys, AR_ORDER)
                [(1 - pool_w) .* coefs[li] .+ pool_w .* anchor for li in 1:L]
            end

            for (li, loc) in enumerate(LOCATIONS)
                coef = blended[li]
                resid_sd = resid_sd_for(Xs[li], yresps[li], coef, AR_ORDER)
                cum0 = cum_to_date(data, li)
                season_total = get(season_totals, loc, NaN)
                renewal_m = use_renewal ?
                    mean(diff(ys[li][(end - RENEWAL_WINDOW):end])) : 0.0
                paths = simulate_paths_mech(
                    ys[li], coef, resid_sd, AR_ORDER, HORIZONS, NPATHS;
                    rng=rng, innovation=innovation, transform=transform,
                    level=level[li], profile=profile, origin=origin,
                    season_total=season_total, cum0=cum0,
                    use_depletion=use_depletion,
                    depletion_gamma=depletion_gamma,
                    use_renewal=use_renewal, renewal_m=renewal_m,
                    renewal_decay=renewal_decay,
                    renewal_gamma=renewal_gamma,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    s = get(profile, week_of_season(target_end), 0.0)
                    vals = paths[h] .+ level[li] .+ s
                    for q in QUANTILE_LEVELS
                        qval = quantile(vals, q)
                        nat = max(from_scale(qval, transform), 0.0)
                        push!(rows, (
                            model_id, loc, origin, h, target_end,
                            TARGET, "quantile", q, nat,
                        ))
                    end
                end
            end
        end
    end
    return rows
end

# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------

function run_combo_mech(
    label, seasons, hist, versions_full; transform, innovation, pool_w,
    use_depletion, depletion_gamma, use_renewal, renewal_decay,
    renewal_gamma, truth,
)
    fc = build_forecast_table_mech(
        seasons, hist, versions_full; transform=transform,
        innovation=innovation, pool_w=pool_w, use_depletion=use_depletion,
        depletion_gamma=depletion_gamma, use_renewal=use_renewal,
        renewal_decay=renewal_decay, renewal_gamma=renewal_gamma,
        model_id=label,
    )
    summ = score_one(fc, truth)
    cov50 = coverage(fc, truth, 0.5)
    cov90 = coverage(fc, truth, 0.9)
    println("  $(rpad(label, 44)) mean_wis=$(round(summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(summ.sd_wis; digits=4)) " *
            "cov50=$(round(cov50; digits=3)) cov90=$(round(cov90; digits=3))")
    return (
        label=label, use_depletion=use_depletion,
        depletion_gamma=depletion_gamma, use_renewal=use_renewal,
        renewal_decay=renewal_decay, renewal_gamma=renewal_gamma,
        mean_wis=summ.mean_wis, sd_wis=summ.sd_wis, cov50=cov50,
        cov90=cov90, forecast=fc,
    )
end

function main()
    t0 = time()
    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")
    truth = load_oracle(HUB_PATH)
    # NOTE (leakage fix, matching round2-stack/generate.jl): the pooled
    # seasonal profile, backfill revision profile, AND season_totals are
    # no longer precomputed here -- `build_forecast_table_mech` rebuilds
    # all three FRESH, per split, restricted to that split's own
    # forecast origin (see `build_season_totals` above and round2-
    # stack's `build_seasonal_profile`/`build_revision_profile`). `hist`/
    # `versions_full` are passed through unfiltered; the per-origin
    # restriction happens inside.

    results = NamedTuple[]

    println("=== alone, on top of the plain core (fourthroot/gaussian/" *
            "pool=0) ===")
    push!(results, run_combo_mech(
        "core (reproduce)", VALIDATION_ONLY, hist, versions_full;
        transform=:fourthroot, innovation=:gaussian, pool_w=0.0,
        use_depletion=false, depletion_gamma=1.0, use_renewal=false,
        renewal_decay=0.7, renewal_gamma=1.0, truth=truth,
    ))
    push!(results, run_combo_mech(
        "core+depletion", VALIDATION_ONLY, hist, versions_full;
        transform=:fourthroot, innovation=:gaussian, pool_w=0.0,
        use_depletion=true, depletion_gamma=1.0, use_renewal=false,
        renewal_decay=0.7, renewal_gamma=1.0, truth=truth,
    ))
    push!(results, run_combo_mech(
        "core+renewal", VALIDATION_ONLY, hist, versions_full;
        transform=:fourthroot, innovation=:gaussian, pool_w=0.0,
        use_depletion=false, depletion_gamma=1.0, use_renewal=true,
        renewal_decay=0.7, renewal_gamma=1.0, truth=truth,
    ))
    push!(results, run_combo_mech(
        "core+depletion+renewal", VALIDATION_ONLY, hist, versions_full;
        transform=:fourthroot, innovation=:gaussian, pool_w=0.0,
        use_depletion=true, depletion_gamma=1.0, use_renewal=true,
        renewal_decay=0.7, renewal_gamma=1.0, truth=truth,
    ))

    println("\n=== stacked on the round2-stack winner (log+tstudent+" *
            "pool(w=0.9)) ===")
    push!(results, run_combo_mech(
        "winner (reproduce)", VALIDATION_ONLY, hist, versions_full;
        transform=:log, innovation=:student_t, pool_w=0.9,
        use_depletion=false, depletion_gamma=1.0, use_renewal=false,
        renewal_decay=0.7, renewal_gamma=1.0, truth=truth,
    ))
    push!(results, run_combo_mech(
        "winner+depletion", VALIDATION_ONLY, hist, versions_full;
        transform=:log, innovation=:student_t, pool_w=0.9,
        use_depletion=true, depletion_gamma=1.0, use_renewal=false,
        renewal_decay=0.7, renewal_gamma=1.0, truth=truth,
    ))
    push!(results, run_combo_mech(
        "winner+renewal", VALIDATION_ONLY, hist, versions_full;
        transform=:log, innovation=:student_t, pool_w=0.9,
        use_depletion=false, depletion_gamma=1.0, use_renewal=true,
        renewal_decay=0.7, renewal_gamma=1.0, truth=truth,
    ))
    push!(results, run_combo_mech(
        "winner+depletion+renewal  [full mech stack]", VALIDATION_ONLY,
        hist, versions_full;
        transform=:log, innovation=:student_t, pool_w=0.9,
        use_depletion=true, depletion_gamma=1.0, use_renewal=true,
        renewal_decay=0.7, renewal_gamma=1.0, truth=truth,
    ))

    println("\n=== depletion_gamma sensitivity on top of the full mech " *
            "stack (renewal_decay=0.7 fixed) ===")
    gamma_results = NamedTuple[]
    for g in DEPLETION_GAMMAS
        r = run_combo_mech(
            "winner+dep(g=$g)+renewal", VALIDATION_ONLY, hist,
            versions_full;
            transform=:log, innovation=:student_t, pool_w=0.9,
            use_depletion=true, depletion_gamma=g, use_renewal=true,
            renewal_decay=0.7, renewal_gamma=1.0, truth=truth,
        )
        push!(results, r)
        push!(gamma_results, r)
    end
    best_gamma = sort(gamma_results; by=r -> r.mean_wis)[1].depletion_gamma

    println("\n=== renewal_decay sensitivity on top of the full mech " *
            "stack (depletion_gamma=$best_gamma fixed) ===")
    decay_results = NamedTuple[]
    for d in RENEWAL_DECAYS
        r = run_combo_mech(
            "winner+dep+renewal(d=$d)", VALIDATION_ONLY, hist,
            versions_full;
            transform=:log, innovation=:student_t, pool_w=0.9,
            use_depletion=true, depletion_gamma=best_gamma,
            use_renewal=true, renewal_decay=d, renewal_gamma=1.0,
            truth=truth,
        )
        push!(results, r)
        push!(decay_results, r)
    end

    sorted = sort(results; by=r -> r.mean_wis)
    winner = sorted[1]
    core = results[1]
    stack_reproduce = results[5]
    mech_only = filter(r -> r.use_depletion || r.use_renewal, results)
    best_mech = sort(mech_only; by=r -> r.mean_wis)[1]

    println("\n=== ranked ===")
    for r in sorted
        println("  $(rpad(r.label, 46)) mean_wis=$(round(r.mean_wis; digits=4)) " *
                "sd_wis=$(round(r.sd_wis; digits=4))")
    end
    println("\nwinner: $(winner.label) mean_wis=$(round(winner.mean_wis; digits=4))")

    # Winner breakdown: by location, by season, by horizon.
    winner_scored = score_forecasts(winner.forecast, truth; scale=:natural)
    by_loc = combine(groupby(winner_scored, :location),
        :wis => mean => :mean_wis, nrow => :n)
    sort!(by_loc, :mean_wis)

    winner_scored.season_num = [
        season_year(d) == 2015 ? 1 : 2 for d in winner_scored.origin_date
    ]
    by_season = combine(groupby(winner_scored, :season_num),
        :wis => mean => :mean_wis, :wis => std => :sd_wis, nrow => :n)
    sort!(by_season, :season_num)

    by_h = combine(groupby(winner_scored, :horizon),
        :wis => mean => :mean_wis, nrow => :n)
    sort!(by_h, :horizon)

    by_season_h = combine(groupby(winner_scored, [:season_num, :horizon]),
        :wis => mean => :mean_wis, nrow => :n)
    sort!(by_season_h, [:season_num, :horizon])

    # Same season x horizon breakdown for the BEST MECHANISTIC combo
    # (may or may not equal `winner` above -- see the ranked table),
    # for a like-for-like comparison of where the mechanistic terms
    # help or hurt relative to the plain (non-mechanistic) stack.
    mech_scored = score_forecasts(best_mech.forecast, truth; scale=:natural)
    mech_scored.season_num = [
        season_year(d) == 2015 ? 1 : 2 for d in mech_scored.origin_date
    ]
    mech_by_season_h = combine(groupby(mech_scored, [:season_num, :horizon]),
        :wis => mean => :mean_wis, nrow => :n)
    sort!(mech_by_season_h, [:season_num, :horizon])

    # delta > 0 means the mechanistic combo is BETTER (lower WIS) than
    # the plain winner in that (season, horizon) cell.
    delta_by_season_h = innerjoin(
        rename(by_season_h, :mean_wis => :mean_wis_winner),
        rename(mech_by_season_h[:, [:season_num, :horizon, :mean_wis]],
            :mean_wis => :mean_wis_mech),
        on=[:season_num, :horizon],
    )
    delta_by_season_h.delta =
        delta_by_season_h.mean_wis_winner .- delta_by_season_h.mean_wis_mech
    sort!(delta_by_season_h, [:season_num, :horizon])
    target_rows = delta_by_season_h[
        (delta_by_season_h.season_num .== 2) .&
        (delta_by_season_h.horizon .>= 3), :,
    ]
    other_rows = delta_by_season_h[
        .!((delta_by_season_h.season_num .== 2) .&
           (delta_by_season_h.horizon .>= 3)), :,
    ]
    target_delta = mean(target_rows.delta)
    other_delta = mean(other_rows.delta)

    open(joinpath(@__DIR__, "score.txt"), "w") do io
        println(io, "mechanistic susceptible-depletion + renewal -- " *
                     "simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "round2-stack winner (log+tstudent+pool(w=0.9), " *
                     "'winner (reproduce)' below): mean_wis=" *
                     "$(round(stack_reproduce.mean_wis; digits=4)) sd_wis=" *
                     "$(round(stack_reproduce.sd_wis; digits=4)) -- leak-" *
                     "free honest rescore (experiments/simple-round/" *
                     "round2-stack/score.txt's LEAKAGE FIX section)")
        println(io)
        overall_verdict = winner.mean_wis < stack_reproduce.mean_wis - 1e-9 ?
            "VERDICT: a mechanistic combo beats the plain round2-stack " *
            "winner on validation -- see 'ranked' below." :
            "VERDICT: NEITHER mechanistic candidate beats the plain " *
            "round2-stack winner net (best mechanistic combo is " *
            "'$(best_mech.label)' at mean_wis=" *
            "$(round(best_mech.mean_wis; digits=4)), vs " *
            "$(round(stack_reproduce.mean_wis; digits=4)) plain). See " *
            "'does the mechanism actually capture the peak/turnaround?' " *
            "below for whether it is at least doing the right thing in " *
            "the cells it targets, and the Bayesian-benefit section for " *
            "why a point-estimate analytic version may be leaving real " *
            "signal on the table."
        println(io, overall_verdict)
        println(io)
        println(io, "=== alone, on top of the plain core (fourthroot/" *
                     "gaussian/pool=0) ===")
        for r in results[1:4]
            println(io, "  $(rpad(r.label, 26)) mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4)) cov50=" *
                         "$(round(r.cov50; digits=3)) cov90=" *
                         "$(round(r.cov90; digits=3))")
        end
        println(io)
        println(io, "=== marginal contribution on top of the core " *
                     "(mean_wis=$(round(core.mean_wis; digits=4))) ===")
        for r in results[2:4]
            delta = core.mean_wis - r.mean_wis
            pct = 100 * delta / core.mean_wis
            println(io, "  $(rpad(r.label, 24)) delta=$(round(delta; digits=4)) " *
                         "($(round(pct; digits=2))%)")
        end
        println(io)
        println(io, "=== stacked on the round2-stack winner ===")
        for r in results[5:8]
            println(io, "  $(rpad(r.label, 46)) mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4)) cov50=" *
                         "$(round(r.cov50; digits=3)) cov90=" *
                         "$(round(r.cov90; digits=3))")
        end
        println(io)
        println(io, "=== marginal contribution on top of the winner " *
                     "(mean_wis=$(round(stack_reproduce.mean_wis; digits=4))) ===")
        for r in results[6:8]
            delta = stack_reproduce.mean_wis - r.mean_wis
            pct = 100 * delta / stack_reproduce.mean_wis
            println(io, "  $(rpad(r.label, 46)) delta=$(round(delta; digits=4)) " *
                         "($(round(pct; digits=2))%)")
        end
        println(io)
        println(io, "=== depletion_gamma sensitivity (renewal_decay=0.7) ===")
        for r in gamma_results
            println(io, "  $(rpad(r.label, 30)) mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4))")
        end
        println(io)
        println(io, "=== renewal_decay sensitivity (depletion_gamma=" *
                     "$best_gamma) ===")
        for r in decay_results
            println(io, "  $(rpad(r.label, 30)) mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4))")
        end
        println(io)
        println(io, "=== ranked (all combos) ===")
        for r in sorted
            println(io, "  $(rpad(r.label, 46)) mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4))")
        end
        println(io)
        println(io, "=== winner: $(winner.label) ===")
        println(io, "mean_wis=$(round(winner.mean_wis; digits=4)) " *
                     "sd_wis=$(round(winner.sd_wis; digits=4)) " *
                     "cov50=$(round(winner.cov50; digits=3)) " *
                     "cov90=$(round(winner.cov90; digits=3))")
        vs_stack = stack_reproduce.mean_wis - winner.mean_wis
        vs_pct = 100 * vs_stack / stack_reproduce.mean_wis
        println(io, "vs round2-stack winner " *
                     "($(round(stack_reproduce.mean_wis; digits=4))): " *
                     "$(round(vs_stack; digits=4)) ($(round(vs_pct; digits=2))%)")
        println(io)
        println(io, "winner mean WIS by location:")
        for row in eachrow(by_loc)
            println(io, "  $(rpad(row.location, 16)) $(round(row.mean_wis; digits=4)) " *
                         "(n=$(row.n))")
        end
        println(io)
        println(io, "winner mean WIS by season:")
        for row in eachrow(by_season)
            println(io, "  season $(row.season_num): mean_wis=" *
                         "$(round(row.mean_wis; digits=4)) sd_wis=" *
                         "$(round(row.sd_wis; digits=4)) (n=$(row.n))")
        end
        println(io)
        println(io, "winner mean WIS by horizon:")
        for row in eachrow(by_h)
            println(io, "  h=$(row.horizon): $(round(row.mean_wis; digits=4)) " *
                         "(n=$(row.n))")
        end
        println(io)
        println(io, "winner ($(winner.label)) mean WIS by season x " *
                     "horizon (target: h=3,4, season 2 = 2016/17):")
        for row in eachrow(by_season_h)
            println(io, "  season $(row.season_num), h=$(row.horizon): " *
                         "$(round(row.mean_wis; digits=4)) (n=$(row.n))")
        end
        println(io)
        println(io, "best mechanistic combo ($(best_mech.label), " *
                     "mean_wis=$(round(best_mech.mean_wis; digits=4))) mean " *
                     "WIS by season x horizon, for comparison against the " *
                     "row above:")
        for row in eachrow(mech_by_season_h)
            println(io, "  season $(row.season_num), h=$(row.horizon): " *
                         "$(round(row.mean_wis; digits=4)) (n=$(row.n))")
        end
        println(io)
        println(io, "=== does the mechanism actually capture the peak/" *
                     "turnaround? ===")
        println(io, "delta = winner mean_wis - best_mech mean_wis per " *
                     "(season, horizon) cell; positive = mechanism helps " *
                     "there, negative = mechanism hurts there:")
        for row in eachrow(delta_by_season_h)
            println(io, "  season $(row.season_num), h=$(row.horizon): " *
                         "delta=$(round(row.delta; digits=4))")
        end
        println(io, "mean delta, season 2 (2016/17) h=3,4 (the target " *
                     "cells): $(round(target_delta; digits=4))")
        println(io, "mean delta, everywhere else: " *
                     "$(round(other_delta; digits=4))")
        verdict = if target_delta > 0 && target_delta > other_delta
            "The mechanism DOES help disproportionately in the target " *
            "cells (season 2, h=3-4) relative to elsewhere, consistent " *
            "with a genuine depletion/turnaround effect -- but it still " *
            "loses enough elsewhere (other locations/horizons/season 1) " *
            "that the net mean_wis is worse than the plain stack (see " *
            "the ranked table above): as configured here, the mechanism " *
            "is not a net win even though it is doing something real."
        elseif target_delta > 0
            "The mechanism helps in the target cells (season 2, h=3-4) " *
            "but NOT disproportionately more than it helps (or hurts) " *
            "elsewhere -- consistent with it mostly adding/removing " *
            "generic variance rather than specifically capturing " *
            "depletion-driven turnover."
        else
            "The mechanism does NOT help even in the specific cells it " *
            "was designed for (season 2, h=3-4): both the point estimate " *
            "of the depletion/momentum terms and their net effect on WIS " *
            "are working against the hypothesis in this validation data, " *
            "not just failing to pay for the added variance elsewhere."
        end
        println(io, verdict)
        println(io)
        println(io, "=== what this analytic version cannot do, and where " *
                     "a Bayesian (Turing) treatment would genuinely help ===")
        println(io, "1. depletion_gamma and renewal_decay/renewal_gamma " *
                     "are POINT estimates, tuned once by grid search over " *
                     "$(length(DEPLETION_GAMMAS)) x $(length(RENEWAL_DECAYS)) " *
                     "values on 2 validation seasons; a Bayesian model " *
                     "could put a prior on each and let the posterior " *
                     "reflect how little season-level information there " *
                     "really is (5 seasons total) to pin these down, " *
                     "rather than reporting one selected value with no " *
                     "uncertainty and risking overfitting the sweep itself " *
                     "to those 2 seasons.")
        println(io, "2. `season_total` (the depletion denominator) is a " *
                     "single historical MEAN per location, so a location " *
                     "with a genuinely unusual season this year gets its " *
                     "damping calibrated against a number that may not " *
                     "apply to it -- a hierarchical model with partial " *
                     "pooling of the per-location depletion rate/season " *
                     "total toward a shared distribution (shrinking " *
                     "small/noisy locations toward the group, letting " *
                     "well-observed ones like US National speak more for " *
                     "themselves) would let each location's damping adapt " *
                     "without needing the ad hoc `min_weeks=40` filter and " *
                     "flat historical average used here.")
        println(io, "3. `renewal_m` (the momentum term) is a fixed number " *
                     "per split, from a 3-point local finite difference; " *
                     "a Turing renewal/Rt-style model could instead treat " *
                     "the local growth rate as a latent, smoothly-evolving " *
                     "state with its OWN uncertainty (e.g. a random walk " *
                     "on log-Rt), which both regularises the momentum " *
                     "estimate against a noisy last few weeks and " *
                     "naturally propagates that uncertainty into the " *
                     "forecast intervals -- something the point estimate " *
                     "used here cannot do (it is exact given the inputs, " *
                     "but the inputs themselves have no error bars).")
        println(io, "4. the two mechanistic terms and the AR(6)/pool/" *
                     "Student-t/backfill terms are all currently combined " *
                     "by hand-picked functional forms (damping " *
                     "multiplicatively, adding the renewal term, applying " *
                     "both to the same cum_frac); a joint Bayesian model " *
                     "could instead estimate how much weight each " *
                     "mechanism should carry (including possibly zero) " *
                     "directly from the joint likelihood across all " *
                     "locations and seasons, rather than via a sequential " *
                     "ablation-and-pick-the-best-cell sweep like this one.")
    end

    dt = round(time() - t0; digits=1)
    println("\nwrote score.txt in $(dt)s total")

    hub_path = length(ARGS) >= 1 ? ARGS[1] : nothing
    if hub_path !== nothing
        submit_mech(hub_path, hist, versions_full)
    end
    return sorted
end

# ---------------------------------------------------------------------
# Full 5-season hub submission driver (locked-in `SUB_*` combo)
# ---------------------------------------------------------------------

"""
    submit_mech(hub_path, hist, versions_full)

Writes a full-5-season (`allow_test_season=true` inside
`build_forecast_table_mech`, still a per-origin vintage fit capped at
each split's own forecast origin, NOT training on the test seasons) hub
submission under model_id `SUB_MODEL_ID` ("nfidd-mech"), using the
`SUB_*` combo locked in at the top of this file from the validation
sweep above.
"""
function submit_mech(hub_path, hist, versions_full)
    forecast = build_forecast_table_mech(
        (1, 2, 3, 4, 5), hist, versions_full;
        transform=:log, innovation=:student_t,
        pool_w=0.9, use_depletion=SUB_USE_DEPLETION,
        depletion_gamma=SUB_DEPLETION_GAMMA, use_renewal=SUB_USE_RENEWAL,
        renewal_decay=SUB_RENEWAL_DECAY, renewal_gamma=SUB_RENEWAL_GAMMA,
        model_id=SUB_MODEL_ID,
    )
    write_submission(forecast, hub_path)
    write_metadata(
        SUB_MODEL_ID, hub_path; team_abbr="nfidd", model_abbr="mech",
        designated=true,
    )
    println("wrote $(nrow(forecast)) rows across " *
            "$(length(unique(forecast.origin_date))) origin dates to " *
            "$(hub_path)")
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
