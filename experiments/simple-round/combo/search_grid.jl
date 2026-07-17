#!/usr/bin/env julia
# search_grid.jl -- joint grid search for the simple-round "best
# combination" family: per-location AR order x the nfidd-ar6bf
# backfill correction x an optional lightly-regularised seasonal term,
# all on the fourth-root scale (docs/lessons.md item 7).
#
# Deliberately LIGHT + ANALYTIC, like `submissions/nfidd-ar6/
# generate_forecasts.jl` and `submissions/seabbs_bot-ar6bf/
# generate_forecasts.jl` that this extends: CSV/DataFrames/Statistics/
# LinearAlgebra only, no Turing/Mooncake/Pathfinder. `src/scoring.jl`
# (ScoringRules only, no Turing) is also included so this script can
# score its own candidates in-process, without writing a hub
# submission first.
#
# Baseline to beat: seabbs_bot-ar6bf, mean WIS 0.359 (SD 0.452) on the
# validation seasons (submissions/seabbs_bot-ar6bf/README.md), itself
# AR(6) + backfill correction.
#
# SCORES ON VALIDATION SEASONS (1, 2) ONLY (docs/contracts.md
# experimental integrity); `training_splits` refuses seasons 3-5
# unless `allow_test_season=true` is passed explicitly, which this
# script never does.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> search_grid.jl

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))
include(joinpath(PKG_DIR, "src", "hubio.jl"))
include(joinpath(PKG_DIR, "src", "scoring.jl"))

const TRANSFORM = :fourthroot
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12          # matches nfidd-ar6 / seabbs_bot-ar6bf
const DELAY_CUTOFF = 8   # backfill profile support, docs/eda/02
const MIN_SUPPORT = 5
const N_HARMONICS = 2    # sin/cos pairs for the seasonal term
const PERIOD = 52.0      # weeks per season cycle
const LAMBDA_FRAC = 0.3  # ridge strength on seasonal cols, as a
                         # fraction of nobs (see fit_ar_seasonal); a
                         # single fixed "light" value, not itself part
                         # of the grid, to keep the search small
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")

# ---------------------------------------------------------------------
# Backfill correction (identical to seabbs_bot-ar6bf)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile, copied unchanged from
`submissions/seabbs_bot-ar6bf/generate_forecasts.jl` -- see that file
for the full derivation. Estimated once, from training-set origin
dates only (`season_year <= 2016`), and reused across every grid cell.
"""
function build_revision_profile(
    versions::DataFrame; transform::Symbol, max_delay::Int,
    min_support::Int,
)
    raw = Dict{Tuple{String,Int},Vector{Float64}}()
    for g in groupby(versions, [:location, :origin_date])
        settled_idx = argmax(g.as_of)
        settled = to_scale(g.wili[settled_idx], transform)
        settled_as_of = g.as_of[settled_idx]
        loc = g.location[1]
        for row in eachrow(g)
            row.as_of == settled_as_of && continue
            delay = div(Dates.value(row.as_of - row.origin_date), 7)
            (delay < 0 || delay > max_delay) && continue
            vintage = to_scale(row.wili, transform)
            key = (loc, delay)
            push!(get!(raw, key, Float64[]), settled - vintage)
        end
    end
    profile = Dict{Tuple{String,Int},Float64}()
    for (key, vals) in raw
        length(vals) >= min_support && (profile[key] = median(vals))
    end
    return profile
end

"""
    apply_backfill_correction!(data, profile)

Nudge `data.Y` in place, at every `(t, l)` with `0 <= data.delay[t, l]
<= DELAY_CUTOFF` and a matching `profile` entry. Identical to
`seabbs_bot-ar6bf`.
"""
function apply_backfill_correction!(
    data::ModelData, profile::Dict{Tuple{String,Int},Float64},
)
    for l in 1:data.L, t in 1:data.T
        d = data.delay[t, l]
        (d < 0 || d > DELAY_CUTOFF) && continue
        ismissing(data.Y[t, l]) && continue
        key = (LOCATIONS[l], d)
        haskey(profile, key) || continue
        data.Y[t, l] += profile[key]
    end
    return data
end

# ---------------------------------------------------------------------
# AR(p) + optional ridge-regularised seasonal term
# ---------------------------------------------------------------------

"""
    fit_ar_seasonal(y, order; woy, n_harmonics, lambda_season, period)
        -> (coef, resid_sd)

OLS/ridge fit of an AR(`order`) model with intercept and, when
`n_harmonics > 0`, `n_harmonics` sin/cos pairs of a `period`-week
seasonal cycle evaluated at each row's `woy` (`ModelData.woy`: 1-based
week-of-season index, so this lines up with the same seasonal clock
`model_season_backfill` uses for its Turing seasonal term).

`coef = [c, phi_1, ..., phi_order, a_1, b_1, ..., a_K, b_K]`
(`phi_lag` multiplies `y[t-lag]`; `a_k`/`b_k` multiply
`sin`/`cos(2*pi*k*woy/period)`).

When `lambda_season > 0`, the seasonal columns (only) are ridge
penalised: `coef = (X'X + diag(0,...,0,lambda,...,lambda)) \\ (X'y)`,
`lambda` on the `2*n_harmonics` seasonal columns and 0 on the
intercept/AR columns, so the AR dynamics are never shrunk, only the
seasonal term -- "lightly regularised" per the brief, since a
`window_weeks=104` (two-season) fit has too little history to trust
an unpenalised per-week seasonal shape. `lambda_season = 0` (or
`n_harmonics = 0`) recovers plain OLS over whichever columns are
present.
"""
function fit_ar_seasonal(
    y::AbstractVector{Float64}, order::Int;
    woy::Union{Nothing,AbstractVector{Int}}=nothing,
    n_harmonics::Int=0, lambda_season::Float64=0.0, period::Float64=52.0,
)
    n = length(y)
    nobs = n - order
    nobs >= order + 2 ||
        error("series too short for AR($order): n=$n, nobs=$nobs")
    n_season_cols = 2 * n_harmonics
    ncols = order + 1 + n_season_cols
    X = zeros(nobs, ncols)
    yresp = Vector{Float64}(undef, nobs)
    for (row, t) in enumerate((order + 1):n)
        yresp[row] = y[t]
        X[row, 1] = 1.0
        for lag in 1:order
            X[row, lag + 1] = y[t - lag]
        end
        if n_harmonics > 0
            wt = woy[t]
            for k in 1:n_harmonics
                ang = 2pi * k * wt / period
                X[row, order + 2k] = sin(ang)
                X[row, order + 2k + 1] = cos(ang)
            end
        end
    end
    coef = if lambda_season > 0 && n_harmonics > 0
        penalty = zeros(ncols)
        penalty[(order + 2):end] .= lambda_season
        (X' * X + Diagonal(penalty)) \ (X' * yresp)
    else
        X \ yresp
    end
    resid = yresp .- X * coef
    dof = max(nobs - ncols, 1)
    resid_sd = sqrt(sum(abs2, resid) / dof)
    return coef, resid_sd
end

"""
    season_effect(woy_val, coef, order, n_harmonics, period) -> Float64

Fitted seasonal offset at week-of-season `woy_val`, reading the
`a_k, b_k` harmonic coefficients out of `coef` (see
[`fit_ar_seasonal`](@ref)'s layout). Returns 0.0 when `n_harmonics ==
0` (no seasonal columns fitted).
"""
function season_effect(
    woy_val::Int, coef::Vector{Float64}, order::Int, n_harmonics::Int,
    period::Float64,
)
    n_harmonics == 0 && return 0.0
    s = 0.0
    for k in 1:n_harmonics
        ang = 2pi * k * woy_val / period
        s += coef[order + 2k] * sin(ang) + coef[order + 2k + 1] * cos(ang)
    end
    return s
end

"""
    simulate_paths(y, coef, resid_sd, order, horizons, npaths,
                   season_offsets; rng) -> Dict{Int,Vector{Float64}}

Simulate `npaths` Gaussian-innovation sample paths forward from the
end of `y` (modelling scale), for each horizon in `horizons`, adding a
precomputed, non-random seasonal offset `season_offsets[step]` at each
forward step (the seasonal term is a deterministic function of the
target date, so it is computed once outside the simulation loop, not
re-evaluated per path). `season_offsets` all-zero recovers plain AR
simulation (identical to nfidd-ar6 / seabbs_bot-ar6bf).
"""
function simulate_paths(
    y::AbstractVector{Float64}, coef::Vector{Float64}, resid_sd::Float64,
    order::Int, horizons, npaths::Int, season_offsets::Vector{Float64};
    rng::Random.AbstractRNG,
)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    tail0 = y[(end - order + 1):end]  # most recent `order` obs, ascending
    for s in 1:npaths
        tail = copy(tail0)
        for h in 1:hmax
            pred = coef[1]
            for lag in 1:order
                pred += coef[lag + 1] * tail[end - lag + 1]
            end
            pred += season_offsets[h]
            val = pred + resid_sd * randn(rng)
            if h in horizons
                out[h][s] = val
            end
            push!(tail, val)
            popfirst!(tail)
        end
    end
    return out
end

# ---------------------------------------------------------------------
# Forecast table for one grid cell
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, model_id, ar_order, use_backfill,
                          use_seasonal, profile, versions_full)

One grid cell's hub quantile table across every split of `seasons`
(validation seasons only in this script). `use_backfill` toggles the
`profile` correction (as `seabbs_bot-ar6bf`); `use_seasonal` toggles
the `N_HARMONICS`/`LAMBDA_FRAC` ridge seasonal term. When
`use_backfill` is false, `build_model_data` is called without
`versions` (recency-based delay, as `nfidd-ar6`); when true, the full
`versions_full` table is passed (true `as_of`-based delay, as
`seabbs_bot-ar6bf`), matching each baseline's own delay convention.

`seasons` in `TEST_SEASONS` (3-5) are fetched with
`allow_test_season=true`, mirroring `seabbs_bot-ar6bf`: each split is
still just a per-origin vintage fit capped at its own forecast origin,
so covering the test seasons at generation time never trains on or
tunes against them (docs/contracts.md experimental integrity). The
model itself -- AR order, backfill profile, seasonal ridge lambda --
is selected purely from validation-season (1, 2) scoring in
`search_grid.jl`'s `main`; this function is only ever asked to touch
seasons 3-5 by `generate.jl`'s full-coverage submission path.
"""
function build_forecast_table(
    seasons, model_id, ar_order, use_backfill, use_seasonal, profile,
    versions_full,
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
            data = build_model_data(
                split; Dmax=DMAX, transform=TRANSFORM, window_weeks=104,
                versions=use_backfill ? versions_full : nothing,
            )
            use_backfill && apply_backfill_correction!(data, profile)
            origin = data.origin_date
            for (li, loc) in enumerate(LOCATIONS)
                y = Float64.(data.Y[:, li])
                n_harm = use_seasonal ? N_HARMONICS : 0
                lambda_season = use_seasonal ?
                    LAMBDA_FRAC * (length(y) - ar_order) : 0.0
                coef, resid_sd = fit_ar_seasonal(
                    y, ar_order; woy=data.woy, n_harmonics=n_harm,
                    lambda_season=lambda_season, period=PERIOD,
                )
                season_offsets = [
                    season_effect(
                        week_of_season(origin + Day(7 * step)), coef,
                        ar_order, n_harm, PERIOD,
                    )
                    for step in 1:maximum(HORIZONS)
                ]
                paths = simulate_paths(
                    y, coef, resid_sd, ar_order, HORIZONS, NPATHS,
                    season_offsets; rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    vals = paths[h]
                    for q in QUANTILE_LEVELS
                        qval = quantile(vals, q)
                        nat = max(from_scale(qval, TRANSFORM), 0.0)
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
# Scoring helper
# ---------------------------------------------------------------------

"""Hub oracle (`target-data/oracle-output.csv`) as a scoring truth
table, identical to `scripts/run_validation.jl`'s local helper."""
function load_oracle(hub_path)
    path = joinpath(hub_path, "target-data", "oracle-output.csv")
    oracle = CSV.read(path, DataFrame)
    truth = DataFrame(
        location=String.(oracle.location),
        target_end_date=Date.(oracle.target_end_date),
        value=Float64.(oracle.oracle_value),
    )
    return dropmissing(truth)
end

function main()
    t0 = time()
    versions_full = load_series("flu_data_hhs_versions")
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= 2016, :,
    ]
    profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=DELAY_CUTOFF,
        min_support=MIN_SUPPORT,
    )
    truth = load_oracle(HUB_PATH)

    ar_orders = (4, 6, 8)
    backfill_opts = (false, true)
    season_opts = (false, true)

    results = DataFrame(
        ar_order=Int[], backfill=Bool[], seasonal=Bool[],
        mean_wis=Float64[], sd_wis=Float64[], n_tasks=Int[],
    )
    tables = Dict{Tuple{Int,Bool,Bool},DataFrame}()

    for ar_order in ar_orders, use_backfill in backfill_opts,
        use_seasonal in season_opts

        model_id = "combo-ar$(ar_order)" *
            (use_backfill ? "-bf" : "") * (use_seasonal ? "-sn" : "")
        forecast = build_forecast_table(
            (1, 2), model_id, ar_order, use_backfill, use_seasonal,
            profile, versions_full,
        )
        scored = score_forecasts(forecast, truth; scale=:natural)
        summ = wis_summary(scored)
        push!(results, (
            ar_order, use_backfill, use_seasonal, summ.mean_wis[1],
            summ.sd_wis[1], summ.n_tasks[1],
        ))
        tables[(ar_order, use_backfill, use_seasonal)] = forecast
        println(
            "AR($(ar_order)) backfill=$(use_backfill) " *
            "seasonal=$(use_seasonal): mean_wis=" *
            "$(round(summ.mean_wis[1]; digits=4)) sd_wis=" *
            "$(round(summ.sd_wis[1]; digits=4)) " *
            "n_tasks=$(summ.n_tasks[1])",
        )
    end

    sort!(results, :mean_wis)
    dt = round(time() - t0; digits=1)
    println("\ngrid search done in $(dt)s, sorted by mean WIS:")
    show(results; allrows=true, allcols=true)
    println()

    best = results[1, :]
    key = (best.ar_order, best.backfill, best.seasonal)
    println(
        "\nbest: AR($(best.ar_order)) backfill=$(best.backfill) " *
        "seasonal=$(best.seasonal) mean_wis=" *
        "$(round(best.mean_wis; digits=4)) sd_wis=" *
        "$(round(best.sd_wis; digits=4))",
    )
    println("baseline (seabbs_bot-ar6bf): mean_wis=0.359 sd_wis=0.452")

    return results, tables, profile, versions_full, truth, key
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
