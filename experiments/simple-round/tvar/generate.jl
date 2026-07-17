#!/usr/bin/env julia
# generate.jl -- TIME-VARYING AR(6) family for the simple-model wide round
# (analytic, no Turing): per-location AR(6) whose coefficients are fit by
# DISCOUNTED weighted OLS instead of plain OLS, so recent weeks can be
# weighted more heavily than distant ones and the fitted coefficients can
# track a drifting/regime-shifting relationship rather than averaging
# uniformly over the whole 104-week window.
#
# RESULT OF THE SWEEP (see score.txt for the full table): every discount
# < 1.0 tried, and every rolling-window truncation tried as an alternative
# time-varying scheme, scored WORSE than plain (undiscounted) OLS -- and
# monotonically so, all the way down to numerical blow-up at the shortest
# windows/steepest discounts. This held for both AR(6) (this family's
# assigned order) and, as a follow-up check ruling out "AR(6) just has too
# many parameters (7) to support a short effective window", AR(2) too. So
# DISCOUNT = 1.0 below is not a placeholder -- it is the tuned, empirically
# best setting in this family: `fit_ar_discounted` is kept general (a
# smaller DISCOUNT genuinely produces a time-varying fit) so the negative
# result is falsifiable and the machinery is reusable, but this driver
# runs it at the value the validation-season sweep actually preferred.
#
# The one change from plain AR(6) that DOES help, on top of this family's
# static-wins finding, is the backfill correction from
# submissions/seabbs_bot-ar6bf (nudging recently-reported, still-revisable
# weeks toward their expected settled value before fitting) -- included
# below since the brief said to add it if it helps, and it does (see
# score.txt).
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl [hub_path]
#
# With no argument, just fits+forecasts validation seasons (1, 2) and
# prints timing (this experiment is scored on validation only, per the
# brief -- never tune against the held-out test seasons). Pass a hub_path
# to also write a submission-shaped table there, matching the
# nfidd-ar6 / seabbs_bot-ar6bf convention, should this ever be promoted.

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

const MODEL_ID = "tvar-discount-bf"
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12          # matches nfidd-ar6 / seabbs_bot-ar6bf
const DELAY_CUTOFF = 8   # weeks; backfill profile support, docs/eda/02
const MIN_SUPPORT = 5    # min sample size per (location, delay) to trust

# Tuned discount factor (see header + score.txt): 1.0, i.e. plain OLS.
# Set < 1.0 to actually discount older observations; every value tried
# below this scored worse on validation (score.txt).
const DISCOUNT = 1.0

# ---------------------------------------------------------------------
# Discounted weighted OLS AR fit
# ---------------------------------------------------------------------

"""
    fit_ar_discounted(y, order, discount) -> (coef, resid_sd)

Exponentially-discounted weighted OLS fit of an AR(`order`) model with
intercept to `y` (ascending in time, no missing values). Row `i` of the
regression (`i = nobs` is the most recent observation, `i = 1` the
oldest) gets weight `discount^(nobs - i)`: `discount = 1.0` reduces to
equal weights (plain OLS, identical to `fit_ar` in nfidd-ar6 /
seabbs_bot-ar6bf); `discount < 1` down-weights older rows so the fitted
`coef` tracks a drifting relationship instead of averaging over the
whole window -- this is the "coefficients adapt over time" mechanism
for this family.

`coef = [c, phi_1, ..., phi_order]`, `phi_1` multiplying the most recent
lag. `resid_sd` uses the weighted residual variance, with Kish's
effective sample size (`ess = (sum w)^2 / sum w^2`) standing in for the
raw row count in the degrees-of-freedom correction -- a small discount
effectively fits on far fewer than `nobs` independent rows, and the plain
row count would understate the residual uncertainty.
"""
function fit_ar_discounted(
    y::AbstractVector{Float64}, order::Int, discount::Float64,
)
    n = length(y)
    nobs = n - order
    nobs >= order + 2 ||
        error("series too short for AR($order): n=$n, nobs=$nobs")
    X = ones(nobs, order + 1)
    yresp = Vector{Float64}(undef, nobs)
    w = Vector{Float64}(undef, nobs)
    for (row, t) in enumerate((order + 1):n)
        yresp[row] = y[t]
        for lag in 1:order
            X[row, lag + 1] = y[t - lag]
        end
        w[row] = discount^(nobs - row)
    end
    sw = Diagonal(w)
    coef = (X' * sw * X) \ (X' * sw * yresp)
    resid = yresp .- X * coef
    wvar = sum(w .* abs2.(resid)) / sum(w)
    ess = sum(w)^2 / sum(abs2, w)
    dof = max(ess - (order + 1), 1.0)
    resid_sd = sqrt(wvar * ess / dof)
    return coef, resid_sd
end

"""
    simulate_paths(y, coef, resid_sd, order, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Simulate `npaths` Gaussian-innovation AR(`order`) sample paths forward
from the end of `y` (modelling scale), for each horizon in `horizons`.
Identical to nfidd-ar6 / seabbs_bot-ar6bf.
"""
function simulate_paths(
    y::AbstractVector{Float64}, coef::Vector{Float64}, resid_sd::Float64,
    order::Int, horizons, npaths::Int; rng::Random.AbstractRNG,
)
    hmax = maximum(horizons)
    out = Dict(h => Vector{Float64}(undef, npaths) for h in horizons)
    tail0 = y[(end - order + 1):end]
    for s in 1:npaths
        tail = copy(tail0)
        for h in 1:hmax
            pred = coef[1]
            for lag in 1:order
                pred += coef[lag + 1] * tail[end - lag + 1]
            end
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
# Backfill correction (copied from submissions/seabbs_bot-ar6bf, see that
# file for the full derivation and docs/eda/02-backfill.md for the EDA
# behind it)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical location x delay revision profile on the `transform` scale:
median of `to_scale(settled, transform) - to_scale(vintage, transform)`
per `(location, delay)` with at least `min_support` recorded versions at
that delay. `versions` must already be filtered by the caller to the
training set (no test seasons).
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

Nudge `data.Y` in place at every `(t, l)` with `0 <= data.delay[t, l] <=
DELAY_CUTOFF` and a matching `profile` entry. Missing entries and delays
outside the profile's support are left untouched.
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
# Forecast table
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, profile, versions_full) -> DataFrame

Fit and forecast the discounted-AR(6)+backfill model for every
cross-validation split of every season in `seasons`. Training discipline
matches nfidd-ar6 / seabbs_bot-ar6bf: `build_model_data` caps each
split's data at its own forecast origin, `window_weeks=104` further caps
history to the most recent two seasons, and `versions_full` (passed
through to `build_model_data`) only ever looks at `as_of <=
forecast_origin`.
"""
function build_forecast_table(seasons, profile, versions_full)
    rng = MersenneTwister(SEED)
    rows = DataFrame(
        model_id=String[], location=String[], origin_date=Date[],
        horizon=Int[], target_end_date=Date[], target=String[],
        output_type=String[], output_type_id=Float64[], value=Float64[],
    )
    for season in seasons
        splits = training_splits(season)
        for split in splits
            data = build_model_data(
                split; Dmax=DMAX, transform=TRANSFORM, window_weeks=104,
                versions=versions_full,
            )
            apply_backfill_correction!(data, profile)
            origin = data.origin_date
            for (li, loc) in enumerate(LOCATIONS)
                y = Float64.(data.Y[:, li])
                coef, resid_sd = fit_ar_discounted(y, AR_ORDER, DISCOUNT)
                paths = simulate_paths(
                    y, coef, resid_sd, AR_ORDER, HORIZONS, NPATHS;
                    rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    vals = paths[h]
                    for q in QUANTILE_LEVELS
                        qval = quantile(vals, q)
                        nat = max(from_scale(qval, TRANSFORM), 0.0)
                        push!(rows, (
                            MODEL_ID, loc, origin, h, target_end,
                            TARGET, "quantile", q, nat,
                        ))
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
    println("revision profile: $(length(profile)) (location, delay) " *
            "entries with >= $(MIN_SUPPORT) observations")

    forecast = build_forecast_table((1, 2), profile, versions_full)
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    if hub_path !== nothing
        write_submission(forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="nfidd", model_abbr="tvarbf", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
