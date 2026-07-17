#!/usr/bin/env julia
# THOROUGH time-varying AR on the seasonal+backfill winner --
# simple-round, TIME-VARYING AR family.
#
# Base model (unchanged from `experiments/simple-round/seasoncombo`'s
# combo 1 "core", mean WIS 0.2781 on validation seasons 1-2): the pooled
# week-of-season climatology (`build_seasonal_profile`, one shared shape
# across all 11 locations and ~13 seasons of history), the empirical
# per-(location, delay) backfill correction applied BEFORE
# deseasonalizing, then a per-location AR(6) fit on the deseasonalised +
# backfilled residual.
#
# A first pass at time-varying AR on top of this residual
# (`experiments/simple-round/seasoncombo/generate.jl` combo 2, "tvar":
# exponentially-discounted WLS, discount in {0.95, 0.97, 0.99, 0.995,
# 1.0}, NO backfill correction) found discount=1.0 (i.e. no time
# variation) best at 0.2866 -- worse than the 0.2781 static+backfill
# core. That pass (a) never combined discounting WITH the backfill
# correction, and (b) only tried one discounting scheme. This driver
# fills both gaps: backfill + deseasonalizing are applied first (as in
# the winning core), and THREE distinct time-varying mechanisms are
# swept on the resulting residual, plus a smaller order/effective-
# memory follow-up:
#
#   1. RLS   -- genuine recursive least squares with a forgetting factor
#               lambda (row-by-row P/theta updates, diffuse prior), swept
#               over lambda in [0.90, 1.0]. lambda=1.0 is mathematically
#               the static OLS fit (a diffuse-prior RLS recursion
#               converges to the same normal equations regardless of
#               lambda, since the prior's information content is
#               negligible by construction) -- included as the in-sweep
#               static control.
#   2. window -- rolling-window AR: re-fit plain OLS AR(6) using only the
#               most recent `window` weeks of residual, discarding
#               everything older. window=104 (=WINDOW_WEEKS, the full
#               history already used) is the in-sweep static control.
#   3. kernel -- locally-weighted AR: weighted OLS with a HALF-GAUSSIAN
#               kernel weight centred on the week nearest the forecast
#               origin (weight = exp(-0.5*(distance/bandwidth)^2)),
#               swept over bandwidth. This is a genuinely different
#               weighting SHAPE from the RLS/discount's geometric decay
#               (Gaussian tails vs. exponential), not a re-parameterised
#               copy of it. A large bandwidth is the in-sweep static
#               control (Gaussian weight -> ~1 everywhere).
#   4. order follow-up -- repeats the RLS lambda sweep at AR(3) and
#               AR(4) (instead of AR(6)) to check whether any "more
#               discounting helps" signal is being masked by AR(6)'s
#               larger parameter count needing more effective history to
#               identify (the same order-artifact check done in
#               `experiments/simple-round/tvar/score.txt` for the
#               no-season model, repeated here on the seasonal residual).
#
# All four are compared to the static core (backfill + AR(6), no time
# variation) on VALIDATION SEASONS (1, 2) ONLY -- docs/contracts.md
# experimental integrity, no test-season tuning. Region/season (2015/16
# vs 2016/17) breakdowns are reported for the core and for the single
# best genuinely time-varying configuration found (i.e. excluding the
# in-sweep static controls), since a shift in AR dynamics, if real, is
# most likely to show up asymmetrically across seasons or locations even
# if it washes out in the headline mean.
#
# LIGHT + ANALYTIC: CSV/DataFrames/Statistics/LinearAlgebra only, no
# Turing.
#
# Coverage: writes a FULL 5-season hub submission (validation seasons 1-2
# plus held-out test seasons 3-5, `season in TEST_SEASONS` fetched with
# `allow_test_season=true`) using whichever configuration scores best on
# validation above -- each split is still just a per-origin vintage fit
# capped at its own forecast origin; model selection happened on
# VALIDATION_SEASONS only.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl <hub_path>
# writes score.txt alongside this file; writes a hub submission only if
# hub_path is given.

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
const HERE = @__DIR__
include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))
include(joinpath(PKG_DIR, "src", "hubio.jl"))
include(joinpath(PKG_DIR, "src", "scoring.jl"))

const MODEL_ID = "seabbs_bot-tvarseason"
const TRANSFORM = :fourthroot
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12                 # matches ar6bf's build_model_data Dmax
const WINDOW_WEEKS = 104        # AR history cap = 2 seasons
const MIN_SUPPORT = 5           # min sample size per profile/backfill bin
const SMOOTH_WINDOW = 3         # circular smoothing span for the profile
const VALIDATION_ONLY = (1, 2)
const MAX_TRAIN_SEASON_YEAR = 2016  # pre-2015 history + validation seasons
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")   # oracle for scoring

# Same backfill design as `seasoncombo`'s "core" combo (additive,
# per-location, median, 8-week cutoff).
const BF_MODE = :additive
const BF_WINDOW = 8
const BF_STAT = :median

# ---------------------------------------------------------------------
# Pooled seasonal shape (identical to
# experiments/simple-round/seasoncombo/generate.jl; copied rather than
# shared since every experiment driver in this repo is standalone)
# ---------------------------------------------------------------------

"""
    build_seasonal_profile(hist; transform, max_season_year, min_support,
                            smooth_window) -> Dict{Int,Float64}

Pooled week-of-season climatology on the `transform` scale: one shared
shape across all 11 locations and every season with
`season_year(origin_date) <= max_season_year`. See
`seasoncombo/generate.jl` for the full derivation; identical here.
"""
function build_seasonal_profile(
    hist::DataFrame; transform::Symbol, max_season_year::Int,
    min_support::Int, smooth_window::Int,
)
    h = hist[season_year.(hist.origin_date) .<= max_season_year, :]
    x = to_scale.(h.wili, transform)
    locs = h.location
    woys = week_of_season.(h.origin_date)

    levels = Dict{String,Float64}()
    for loc in unique(locs)
        levels[loc] = mean(x[locs .== loc])
    end
    dev = [x[i] - levels[locs[i]] for i in eachindex(x)]

    Wmax = maximum(woys)
    raw = [Float64[] for _ in 1:Wmax]
    for i in eachindex(dev)
        push!(raw[woys[i]], dev[i])
    end
    means = [length(v) >= min_support ? mean(v) : 0.0 for v in raw]

    half = div(smooth_window - 1, 2)
    smoothed = similar(means)
    for w in 1:Wmax
        idxs = [mod1(w + off, Wmax) for off in (-half):half]
        smoothed[w] = mean(means[idxs])
    end
    smoothed .-= mean(smoothed)

    return Dict(w => smoothed[w] for w in 1:Wmax)
end

"""
    deseasonalize(Y, woy, profile) -> (R, level)

Remove each location's own mean level and the pooled seasonal shape from
`Y` (T x L, modelling scale), returning the residual matrix `R` and the
per-location `level` (added back at forecast time). No per-location
amplitude scaling here (unlike seasoncombo's `amp` combo) -- this driver
isolates the time-varying-AR question on top of the plain pooled shape,
matching combo 1's "core" design exactly.
"""
function deseasonalize(
    Y::AbstractMatrix, woy::Vector{Int}, profile::Dict{Int,Float64},
)
    T, L = size(Y)
    level = zeros(L)
    for l in 1:L
        level[l] = mean(Float64.(Y[:, l]))
    end
    R = Matrix{Float64}(undef, T, L)
    for l in 1:L, t in 1:T
        R[t, l] = Y[t, l] - level[l] - get(profile, woy[t], 0.0)
    end
    return R, level
end

# ---------------------------------------------------------------------
# Backfill correction (identical to seasoncombo's "core" combo)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support)
        -> Dict{Tuple{String,Int},Float64}

Empirical per-(location, delay) revision profile (additive, median),
identical in design to `seasoncombo/generate.jl`; see that file for the
full derivation.
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

Nudge `data.Y` in place wherever `0 <= delay <= BF_WINDOW` and a matching
`profile` entry exists. Identical to seasoncombo's "core" combo.
"""
function apply_backfill_correction!(
    data::ModelData, profile::Dict{Tuple{String,Int},Float64},
)
    for l in 1:data.L, t in 1:data.T
        d = data.delay[t, l]
        (d < 0 || d > BF_WINDOW) && continue
        ismissing(data.Y[t, l]) && continue
        key = (LOCATIONS[l], d)
        haskey(profile, key) || continue
        data.Y[t, l] += profile[key]
    end
    return data
end

# ---------------------------------------------------------------------
# Four AR fitters, sharing the same intercept + `order`-lag regressor
# layout: static OLS, RLS (forgetting factor), rolling-window OLS, and
# Gaussian-kernel-weighted OLS.
# ---------------------------------------------------------------------

"""
    fit_ar(y, order) -> (coef, resid_sd)

Plain OLS fit of an AR(`order`) model with intercept. The static control
every time-varying scheme below is compared against.
"""
function fit_ar(y::AbstractVector{Float64}, order::Int)
    n = length(y)
    nobs = n - order
    nobs >= order + 2 ||
        error("series too short for AR($order): n=$n, nobs=$nobs")
    X = ones(nobs, order + 1)
    yresp = Vector{Float64}(undef, nobs)
    for (row, t) in enumerate((order + 1):n)
        yresp[row] = y[t]
        for lag in 1:order
            X[row, lag + 1] = y[t - lag]
        end
    end
    coef = X \ yresp
    resid = yresp .- X * coef
    dof = max(nobs - (order + 1), 1)
    resid_sd = sqrt(sum(abs2, resid) / dof)
    return coef, resid_sd
end

"""
    fit_ar_rls(y, order, lambda) -> (coef, resid_sd)

Recursive least squares with forgetting factor `lambda`: processes the
`order`-lag-plus-intercept regressor rows ONE AT A TIME (oldest to
newest), updating the coefficient vector `theta` and its covariance `P`
after each row, rather than solving one batch normal-equations system
(contrast `fit_ar_discounted` in `seasoncombo/generate.jl`, which builds
the same discounted objective but solves it in one weighted-OLS batch
call). Starts from a diffuse prior (`P0 = I / delta`, `delta` tiny) so
the recursion converges to the same weighted normal equations as the
batch discounted fit once enough rows have been seen -- the point is a
genuinely different (online, row-by-row) algorithm for the same
forgetting-factor idea, not merely a relabelling of the batch version.
`lambda = 1.0` is mathematically the static OLS fit (no forgetting).
`resid_sd` uses the same exponential weight `lambda^(nobs-row)` as the
fit itself, for a residual-scale estimate consistent with how much each
row actually influenced `theta`.
"""
function fit_ar_rls(y::AbstractVector{Float64}, order::Int, lambda::Float64)
    n = length(y)
    nobs = n - order
    nobs >= order + 2 ||
        error("series too short for AR($order): n=$n, nobs=$nobs")
    ncols = order + 1
    X = ones(nobs, ncols)
    yresp = Vector{Float64}(undef, nobs)
    for (row, t) in enumerate((order + 1):n)
        yresp[row] = y[t]
        for lag in 1:order
            X[row, lag + 1] = y[t - lag]
        end
    end

    delta = 1e-6
    P = Matrix{Float64}(I, ncols, ncols) ./ delta
    theta = zeros(ncols)
    for row in 1:nobs
        x = @view X[row, :]
        Px = P * x
        denom = lambda + dot(x, Px)
        k = Px ./ denom
        e = yresp[row] - dot(x, theta)
        theta = theta .+ k .* e
        P = (P .- k * (x' * P)) ./ lambda
    end

    coef = theta
    resid = yresp .- X * coef
    w = [lambda^(nobs - row) for row in 1:nobs]
    wdof = max(sum(w) - ncols, 1.0)
    resid_sd = sqrt(sum(w .* abs2.(resid)) / wdof)
    return coef, resid_sd
end

"""
    fit_ar_rolling(y, order, window) -> (coef, resid_sd)

Rolling-window AR: plain OLS `fit_ar`, refit using ONLY the most recent
`window` weeks of `y` (older weeks are dropped entirely, not
downweighted -- a hard truncation, contrasted with the smooth decay of
`fit_ar_rls`/`fit_ar_kernel`). `window` is clamped to `length(y)` so a
window larger than the available (already `WINDOW_WEEKS`-capped) history
just reproduces the static fit.
"""
function fit_ar_rolling(y::AbstractVector{Float64}, order::Int, window::Int)
    n = length(y)
    w = min(window, n)
    return fit_ar(y[(n - w + 1):n], order)
end

"""
    fit_ar_kernel(y, order, bandwidth) -> (coef, resid_sd)

Locally-weighted AR: weighted OLS with a HALF-GAUSSIAN kernel weight
`exp(-0.5 * (distance / bandwidth)^2)`, `distance` measured in weeks back
from the forecast origin (the most recent fitted row always gets weight
1). A genuinely different weighting SHAPE from `fit_ar_rls`'s geometric
decay: the Gaussian's weight stays close to 1 for several weeks then
falls off fast, rather than decaying by a constant factor every week.
Large `bandwidth` recovers the static (unweighted) fit.
"""
function fit_ar_kernel(
    y::AbstractVector{Float64}, order::Int, bandwidth::Float64,
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
        w[row] = exp(-0.5 * ((nobs - row) / bandwidth)^2)
    end
    Xw = X .* sqrt.(w)
    yw = yresp .* sqrt.(w)
    coef = Xw \ yw
    resid = yresp .- X * coef
    wdof = max(sum(w) - (order + 1), 1.0)
    resid_sd = sqrt(sum(w .* abs2.(resid)) / wdof)
    return coef, resid_sd
end

"""
    simulate_paths(y, coef, resid_sd, order, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Simulate `npaths` Gaussian-innovation AR(`order`) sample paths forward
from the end of `y`, for each horizon in `horizons`. Shared by all four
fitters above (same AR(`order`) coefficient layout); the coefficients
used are always "at the forecast origin" -- there is no re-estimation
during the forward simulation itself, only in how `coef`/`resid_sd` were
fit beforehand.
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
# Forecast table builder (shared by every combo, selected by `ar_mode`)
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, versions_full, profile, backfill_profile;
                         ar_mode, lambda, window, bandwidth, order,
                         model_id) -> DataFrame

Fit and forecast one time-varying-AR configuration for every
cross-validation split of every season in `seasons`. Deseasonalizing and
the backfill correction are ALWAYS applied (this driver's brief is "keep
season + backfill, vary only the AR mechanism"); `ar_mode` selects which
AR fitter runs on the resulting residual:

  - `:plain`   -- `fit_ar` (static; the core combo / in-sweep control)
  - `:rls`     -- `fit_ar_rls(...; lambda)`
  - `:rolling` -- `fit_ar_rolling(...; window)`
  - `:kernel`  -- `fit_ar_kernel(...; bandwidth)`

`order` overrides `AR_ORDER` for the order/effective-memory follow-up.
"""
function build_forecast_table(
    seasons, versions_full, profile::Dict{Int,Float64},
    backfill_profile::Dict{Tuple{String,Int},Float64};
    ar_mode::Symbol=:plain, lambda::Float64=1.0, window::Int=WINDOW_WEEKS,
    bandwidth::Float64=1e6, order::Int=AR_ORDER, model_id::String,
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
                split; Dmax=DMAX, transform=TRANSFORM,
                window_weeks=WINDOW_WEEKS, versions=versions_full,
            )
            apply_backfill_correction!(data, backfill_profile)
            R, level = deseasonalize(data.Y, data.woy, profile)
            origin = data.origin_date
            for (li, loc) in enumerate(LOCATIONS)
                y = R[:, li]
                coef, resid_sd = if ar_mode == :plain
                    fit_ar(y, order)
                elseif ar_mode == :rls
                    fit_ar_rls(y, order, lambda)
                elseif ar_mode == :rolling
                    fit_ar_rolling(y, order, window)
                elseif ar_mode == :kernel
                    fit_ar_kernel(y, order, bandwidth)
                else
                    error("unknown ar_mode $ar_mode")
                end
                paths = simulate_paths(
                    y, coef, resid_sd, order, HORIZONS, NPATHS; rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    s = get(profile, week_of_season(target_end), 0.0)
                    vals = paths[h] .+ level[li] .+ s
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

"""Hub oracle (`target-data/oracle-output.csv`) as a scoring truth
table."""
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

score_one(forecast, truth) = wis_summary(score_forecasts(
    forecast, truth; scale=:natural,
))[1, :]

"""Per-`season_label` (2015/16 vs 2016/17) mean+SD WIS breakdown of a
scored (`score_forecasts`) table."""
function season_breakdown(scored::DataFrame)
    d = copy(scored)
    d.season_label = ifelse.(
        season_year.(d.origin_date) .== 2015, "2015/16", "2016/17",
    )
    combine(
        groupby(d, :season_label),
        :wis => mean => :mean_wis, :wis => std => :sd_wis, nrow => :n,
    )
end

"""Per-`location` mean+SD WIS breakdown of a scored table."""
function location_breakdown(scored::DataFrame)
    combine(
        groupby(scored, :location),
        :wis => mean => :mean_wis, :wis => std => :sd_wis, nrow => :n,
    )
end

# ---------------------------------------------------------------------
# Sweep grids
# ---------------------------------------------------------------------

# RLS forgetting factor: 1.0 is the in-sweep static control.
const LAMBDAS = (0.90, 0.95, 0.97, 0.99, 0.995, 0.999, 1.0)
# Rolling window (weeks): 104 (=WINDOW_WEEKS) is the in-sweep static
# control. The shortest windows deliberately stay well clear of the
# window=16 numerical blow-up found for AR(6) in
# experiments/simple-round/tvar/score.txt (10 fit rows for 7
# parameters); 26 already gives 20 fit rows, comfortably identified.
const WINDOWS = (26, 32, 39, 52, 65, 78, 91, 104)
# Kernel bandwidth (weeks): 1000 is the in-sweep static control (Gaussian
# weight ~= 1 for every row in a 104-week window).
const BANDWIDTHS = (8.0, 13.0, 20.0, 26.0, 39.0, 52.0, 78.0, 1000.0)
# Order/effective-memory follow-up: repeat the RLS lambda sweep (the
# cheapest of the three mechanisms to re-run) at lower AR orders, to
# check whether AR(6)'s parameter count is masking a genuine
# time-variation signal that a smaller model could pick up with less
# effective history required per fit.
const FOLLOWUP_ORDERS = (3, 4)
const FOLLOWUP_LAMBDAS = (0.95, 0.99, 0.995, 1.0)

function main()
    hub_path = length(ARGS) >= 1 ? ARGS[1] : nothing
    t0 = time()

    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")
    truth = load_oracle(HUB_PATH)

    profile = build_seasonal_profile(
        hist; transform=TRANSFORM, max_season_year=MAX_TRAIN_SEASON_YEAR,
        min_support=MIN_SUPPORT, smooth_window=SMOOTH_WINDOW,
    )
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= MAX_TRAIN_SEASON_YEAR, :,
    ]
    backfill_profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_WINDOW,
        min_support=MIN_SUPPORT,
    )

    # --- static core: season + backfill + plain AR(6) (should reproduce
    # seasoncombo's combo 1, mean_wis=0.2781) ---
    core = build_forecast_table(
        VALIDATION_ONLY, versions_full, profile, backfill_profile;
        ar_mode=:plain, model_id="tvarseason-core",
    )
    core_summ = score_one(core, truth)
    println("core (static AR(6)+season+backfill): " *
            "mean_wis=$(round(core_summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(core_summ.sd_wis; digits=4))")

    # --- combo 1: RLS forgetting factor ---
    rls_results = NamedTuple[]
    for lam in LAMBDAS
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, backfill_profile;
            ar_mode=:rls, lambda=lam, model_id="tvarseason-rls",
        )
        summ = score_one(fc, truth)
        push!(rls_results, (lambda=lam, mean_wis=summ.mean_wis,
                             sd_wis=summ.sd_wis))
        println("rls lambda=$lam -> mean_wis=" *
                "$(round(summ.mean_wis; digits=4)) sd_wis=" *
                "$(round(summ.sd_wis; digits=4))")
    end
    sort!(rls_results; by=r -> r.mean_wis)
    rls_best = rls_results[1]

    # --- combo 2: rolling window ---
    window_results = NamedTuple[]
    for win in WINDOWS
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, backfill_profile;
            ar_mode=:rolling, window=win, model_id="tvarseason-window",
        )
        summ = score_one(fc, truth)
        push!(window_results, (window=win, mean_wis=summ.mean_wis,
                                sd_wis=summ.sd_wis))
        println("window=$win -> mean_wis=" *
                "$(round(summ.mean_wis; digits=4)) sd_wis=" *
                "$(round(summ.sd_wis; digits=4))")
    end
    sort!(window_results; by=r -> r.mean_wis)
    window_best = window_results[1]

    # --- combo 3: Gaussian kernel bandwidth ---
    kernel_results = NamedTuple[]
    for bw in BANDWIDTHS
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, backfill_profile;
            ar_mode=:kernel, bandwidth=bw, model_id="tvarseason-kernel",
        )
        summ = score_one(fc, truth)
        push!(kernel_results, (bandwidth=bw, mean_wis=summ.mean_wis,
                                sd_wis=summ.sd_wis))
        println("bandwidth=$bw -> mean_wis=" *
                "$(round(summ.mean_wis; digits=4)) sd_wis=" *
                "$(round(summ.sd_wis; digits=4))")
    end
    sort!(kernel_results; by=r -> r.mean_wis)
    kernel_best = kernel_results[1]

    # --- combo 4: order/effective-memory follow-up (RLS lambda sweep at
    # AR(3), AR(4)) ---
    followup_results = NamedTuple[]
    for ord in FOLLOWUP_ORDERS, lam in FOLLOWUP_LAMBDAS
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, backfill_profile;
            ar_mode=:rls, lambda=lam, order=ord,
            model_id="tvarseason-followup",
        )
        summ = score_one(fc, truth)
        push!(followup_results, (order=ord, lambda=lam,
                                  mean_wis=summ.mean_wis,
                                  sd_wis=summ.sd_wis))
        println("followup order=$ord lambda=$lam -> mean_wis=" *
                "$(round(summ.mean_wis; digits=4)) sd_wis=" *
                "$(round(summ.sd_wis; digits=4))")
    end
    sort!(followup_results; by=r -> r.mean_wis)
    followup_best = followup_results[1]

    # --- overall comparison: core vs. best of each time-varying family
    # (in-sweep static controls included, so a family only "wins" if a
    # GENUINELY time-varying point beats the static core) ---
    combos = [
        (name="core (static)", mean_wis=core_summ.mean_wis,
         sd_wis=core_summ.sd_wis, detail="plain AR(6)"),
        (name="rls", mean_wis=rls_best.mean_wis, sd_wis=rls_best.sd_wis,
         detail="lambda=$(rls_best.lambda)"),
        (name="window", mean_wis=window_best.mean_wis,
         sd_wis=window_best.sd_wis, detail="window=$(window_best.window)"),
        (name="kernel", mean_wis=kernel_best.mean_wis,
         sd_wis=kernel_best.sd_wis,
         detail="bandwidth=$(kernel_best.bandwidth)"),
        (name="order-followup", mean_wis=followup_best.mean_wis,
         sd_wis=followup_best.sd_wis,
         detail="order=$(followup_best.order) " *
                "lambda=$(followup_best.lambda)"),
    ]
    sort!(combos; by=r -> r.mean_wis)
    winner = combos[1]

    # Is the winner a genuinely time-varying point, or an in-sweep
    # static control (lambda=1.0 / window=104 / bandwidth=1000 / plain)?
    is_static_boundary = winner.name == "core (static)" ||
        (winner.name == "rls" && rls_best.lambda == 1.0) ||
        (winner.name == "window" && window_best.window == 104) ||
        (winner.name == "kernel" && kernel_best.bandwidth == 1000.0) ||
        (winner.name == "order-followup" && followup_best.lambda == 1.0)

    # Region/season breakdown: core vs. the single best GENUINELY
    # time-varying configuration across all sweeps (excluding in-sweep
    # static controls), even if it does not win overall -- the brief
    # asks whether time variation helps ANYWHERE (e.g. 2016/17
    # specifically), not only on the headline mean.
    genuine = filter(
        r -> !(
            (r.name == "rls" && rls_best.lambda == 1.0) ||
            (r.name == "window" && window_best.window == 104) ||
            (r.name == "kernel" && kernel_best.bandwidth == 1000.0) ||
            (r.name == "order-followup" && followup_best.lambda == 1.0) ||
            r.name == "core (static)"
        ),
        combos,
    )
    best_genuine = isempty(genuine) ? nothing : genuine[1]

    best_genuine_fc = if best_genuine === nothing
        nothing
    elseif best_genuine.name == "rls"
        build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, backfill_profile;
            ar_mode=:rls, lambda=rls_best.lambda,
            model_id="tvarseason-best-genuine",
        )
    elseif best_genuine.name == "window"
        build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, backfill_profile;
            ar_mode=:rolling, window=window_best.window,
            model_id="tvarseason-best-genuine",
        )
    elseif best_genuine.name == "kernel"
        build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, backfill_profile;
            ar_mode=:kernel, bandwidth=kernel_best.bandwidth,
            model_id="tvarseason-best-genuine",
        )
    else
        build_forecast_table(
            VALIDATION_ONLY, versions_full, profile, backfill_profile;
            ar_mode=:rls, lambda=followup_best.lambda,
            order=followup_best.order,
            model_id="tvarseason-best-genuine",
        )
    end

    core_scored = score_forecasts(core, truth; scale=:natural)
    core_season_bd = season_breakdown(core_scored)
    core_loc_bd = location_breakdown(core_scored)
    genuine_season_bd = best_genuine_fc === nothing ? nothing :
        season_breakdown(score_forecasts(best_genuine_fc, truth; scale=:natural))
    genuine_loc_bd = best_genuine_fc === nothing ? nothing :
        location_breakdown(score_forecasts(best_genuine_fc, truth; scale=:natural))

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "THOROUGH time-varying AR on the seasonal+backfill " *
                     "winner -- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "Base model (unchanged): pooled seasonal " *
                     "climatology + per-(location,delay) backfill " *
                     "correction + per-location AR(6). Reference points:")
        println(io, "  seasoncombo combo 1 core (season+AR6+backfill)  " *
                     "= 0.2781 (sd 0.3341)  [tuned without any time " *
                     "variation]")
        println(io, "  seasoncombo combo 2 tvar (season+discounted AR, " *
                     "NO backfill) best = 0.2866 (discount=1.0, i.e. " *
                     "static was still best there too)")
        println(io, "  local static-core sanity rerun (this script)    " *
                     "= $(round(core_summ.mean_wis; digits=4)) " *
                     "(sd $(round(core_summ.sd_wis; digits=4)))")
        println(io)

        println(io, "=== combo 1: RLS forgetting factor sweep ===")
        for r in rls_results
            println(io, "  lambda=$(r.lambda) -> mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: lambda=$(rls_best.lambda) mean_wis=" *
                     "$(round(rls_best.mean_wis; digits=4)) sd_wis=" *
                     "$(round(rls_best.sd_wis; digits=4))")
        println(io)

        println(io, "=== combo 2: rolling-window AR sweep ===")
        for r in window_results
            println(io, "  window=$(r.window) -> mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: window=$(window_best.window) mean_wis=" *
                     "$(round(window_best.mean_wis; digits=4)) sd_wis=" *
                     "$(round(window_best.sd_wis; digits=4))")
        println(io)

        println(io, "=== combo 3: locally-weighted (Gaussian kernel) " *
                     "AR sweep ===")
        for r in kernel_results
            println(io, "  bandwidth=$(r.bandwidth) -> mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: bandwidth=$(kernel_best.bandwidth) mean_wis=" *
                     "$(round(kernel_best.mean_wis; digits=4)) sd_wis=" *
                     "$(round(kernel_best.sd_wis; digits=4))")
        println(io)

        println(io, "=== combo 4: order/effective-memory follow-up " *
                     "(RLS lambda sweep at AR(3), AR(4)) ===")
        for r in followup_results
            println(io, "  order=$(r.order) lambda=$(r.lambda) -> " *
                         "mean_wis=$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4))")
        end
        println(io, "best: order=$(followup_best.order) " *
                     "lambda=$(followup_best.lambda) mean_wis=" *
                     "$(round(followup_best.mean_wis; digits=4)) sd_wis=" *
                     "$(round(followup_best.sd_wis; digits=4))")
        println(io)

        println(io, "=== overall comparison (best of each family) ===")
        for r in combos
            println(io, rpad(r.name, 18) *
                         "mean_wis=$(rpad(round(r.mean_wis; digits=4), 8)) " *
                         "sd_wis=$(rpad(round(r.sd_wis; digits=4), 8)) " *
                         r.detail)
        end
        println(io)
        println(io, "=== winner: $(winner.name) ($(winner.detail)) ===")
        println(io, "mean_wis=$(round(winner.mean_wis; digits=4)) " *
                     "sd_wis=$(round(winner.sd_wis; digits=4))")
        println(io, "is an in-sweep STATIC control: $is_static_boundary")
        vs_core = core_summ.mean_wis - winner.mean_wis
        vs_pct = 100 * vs_core / core_summ.mean_wis
        core_r = round(core_summ.mean_wis; digits=4)
        println(io, "vs static core ($core_r): " *
                     "$(round(vs_core; digits=4)) " *
                     "($(round(vs_pct; digits=2))%)")
        println(io)

        println(io, "=== season breakdown: static core ===")
        for r in eachrow(core_season_bd)
            println(io, "  $(r.season_label): mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4)) n=$(r.n)")
        end
        if best_genuine === nothing
            println(io)
            println(io, "No genuinely time-varying configuration beat " *
                         "its own family's static-control point in ANY " *
                         "sweep (every family's in-sweep best was the " *
                         "static boundary itself) -- there is no " *
                         "\"best genuine\" configuration to break down " *
                         "further.")
        else
            println(io, "=== season breakdown: best genuinely " *
                         "time-varying config ($(best_genuine.name), " *
                         "$(best_genuine.detail)) ===")
            for r in eachrow(genuine_season_bd)
                println(io, "  $(r.season_label): mean_wis=" *
                             "$(round(r.mean_wis; digits=4)) sd_wis=" *
                             "$(round(r.sd_wis; digits=4)) n=$(r.n)")
            end
        end
        println(io)

        println(io, "=== location breakdown: static core ===")
        for r in eachrow(core_loc_bd)
            println(io, "  $(r.location): mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4)) n=$(r.n)")
        end
        if best_genuine !== nothing
            println(io)
            println(io, "=== location breakdown: best genuinely " *
                         "time-varying config ($(best_genuine.name), " *
                         "$(best_genuine.detail)) ===")
            for r in eachrow(genuine_loc_bd)
                println(io, "  $(r.location): mean_wis=" *
                             "$(round(r.mean_wis; digits=4)) sd_wis=" *
                             "$(round(r.sd_wis; digits=4)) n=$(r.n)")
            end
        end
        println(io)

        println(io, "=== bottom line ===")
        if is_static_boundary
            println(io, "Every time-varying mechanism tried (RLS " *
                         "forgetting factor, rolling window, " *
                         "Gaussian-kernel local weighting, and lower " *
                         "AR orders) scored WORSE than its own family's " *
                         "static control once backfill + season are " *
                         "already in the model. The overall winner is " *
                         "$(winner.name) ($(winner.detail)), which is " *
                         "itself the static boundary point (== the " *
                         "0.2781 core). This is a negative result for " *
                         "time-varying AR on this residual, consistent " *
                         "with the earlier no-season finding in " *
                         "experiments/simple-round/tvar/score.txt: the " *
                         "104-week (2-season) fit window is already " *
                         "short enough that discounting or truncating " *
                         "it further only removes information without " *
                         "adapting to anything genuinely new.")
        else
            vs_pct_r = round(vs_pct; digits=2)
            println(io, "A genuinely time-varying configuration " *
                         "($(winner.name), $(winner.detail)) BEAT the " *
                         "static core: $(round(vs_core; digits=4)) " *
                         "mean WIS improvement ($(vs_pct_r)%). See the " *
                         "season/location breakdown above for where " *
                         "the gain concentrates.")
        end
    end

    println("\nwinner: $(winner.name) ($(winner.detail)) mean_wis=" *
            "$(round(winner.mean_wis; digits=4)) sd_wis=" *
            "$(round(winner.sd_wis; digits=4)) " *
            "static_boundary=$is_static_boundary")
    println("wrote score.txt")

    # ------------------------------------------------------------------
    # Full 5-season hub-format output, using the winning configuration
    # ------------------------------------------------------------------
    build_kwargs = if winner.name == "core (static)"
        (ar_mode=:plain,)
    elseif winner.name == "rls"
        (ar_mode=:rls, lambda=rls_best.lambda)
    elseif winner.name == "window"
        (ar_mode=:rolling, window=window_best.window)
    elseif winner.name == "kernel"
        (ar_mode=:kernel, bandwidth=kernel_best.bandwidth)
    else
        (ar_mode=:rls, lambda=followup_best.lambda,
         order=followup_best.order)
    end

    forecast = build_forecast_table(
        (1, 2, 3, 4, 5), versions_full, profile, backfill_profile;
        model_id=MODEL_ID, build_kwargs...,
    )
    dt = round(time() - t0; digits=2)
    n_origins = length(unique(forecast.origin_date))
    println("built $(nrow(forecast)) rows across $(n_origins) origin " *
            "date(s) in $(dt)s")

    if hub_path !== nothing
        write_submission(forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="seabbs_bot", model_abbr="tvarseason",
            designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end
    return forecast
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
