#!/usr/bin/env julia
# HIERARCHICAL TIME-VARYING PARTIAL-POOLED SEASONALITY -- simple-round.
#
# Starting point: the round-2 stack winner
# (experiments/simple-round/round2-stack/generate.jl,
# "log+tstudent+pool(w=0.9)"): pooled week-of-season climatology on the
# LOG scale + per-location AR(6) on the deseasonalized residual, blended
# 90% toward a fullpool (all-locations-stacked) OLS anchor, simulated
# forward with Student-t(df=10) innovations scaled by 1.4, plus the
# `seabbs_bot-ar6bf`-style additive/per-location/median backfill
# correction. mean_wis=0.2601 (sd 0.2587, validation seasons 1, 2 --
# see round2-stack/score.txt).
#
# This driver keeps backfill + log + Student-t intervals + AR(6)
# pooling FIXED at the round-2 stack's winning settings, and adds TWO
# further partial-pooling levers directly on the seasonal term itself
# (docs/eda/04-cross-location.md's "shared season-severity effect",
# cross-location seasonal-amplitude correlation mean r=0.68, motivates
# both):
#
#   LEVER 1 -- partial pooling of the seasonal SHAPE. The round-2
#   stack's seasonal term is ONE pooled week-of-season profile shared
#   by all 11 locations (`build_seasonal_profile`). Here each
#   location's OWN week-of-season deviation profile
#   (`build_location_profiles`, estimated only from that location's own
#   history, same binning/circular-smoothing rule as the pooled one) is
#   blended toward the pooled profile,
#     blended[loc][w] = (1 - LAMBDA_SHAPE) * pooled[w]
#                        + LAMBDA_SHAPE * own[loc][w],
#   and the blended shape's overall amplitude is itself shrunk toward 1
#   via a second knob, AMP_SHRINK (identical mechanism to
#   `seasonpool2/generate.jl`'s `build_amplitude_scales`, but regressed
#   against the BLENDED shape rather than the bare pooled one). LAMBDA_
#   SHAPE=0 & AMP_SHRINK=0 reproduces the round-2 stack's single shared
#   shape exactly; LAMBDA_SHAPE=1 & AMP_SHRINK=1 is the fully
#   idiosyncratic, unpooled-per-location shape. Swept as a joint grid
#   (the two knobs interact -- seasonpool2/score.txt found the same for
#   its amplitude/AR-pooling grid).
#
#   LEVER 2 -- TIME-VARYING seasonal amplitude, scaled by the current
#   season's severity-so-far, POOLED across all 11 locations. At each
#   split's forecast origin, this season's mean deseasonalized residual
#   so far (net of the LEVER-1 shape+amplitude term, still fit on the
#   pre-severity residual so the AR(6) itself is untouched) is computed
#   PER LOCATION, then pooled across all 11 (median, matching
#   `experiments/simple-round/severity2`'s preferred stat) into ONE
#   shared "is this season running hot or cold" scalar, added (shrunk by
#   LAMBDA_SEVERITY) to every location's and every horizon's forward
#   seasonal contribution for that split. Because the modelling scale
#   here is LOG (round2-stack's winning transform, not severity2's
#   fourthroot), an ADDITIVE log-scale adjustment is already exactly a
#   MULTIPLICATIVE natural-scale one -- severity2's own preferred form --
#   with no separate ratio/division step or `MIN_CLIM_LEVEL` guard
#   needed.
#
# IMPORTANT PRIOR: `experiments/simple-round/severity2` already tested
# almost exactly LEVER 2 (per-location severity-so-far, pooled across
# locations, scaling the forward climatology) on top of the ROUND-1
# baseline (climatology + backfill, no AR pooling, no Student-t, no
# LEVER 1) and found it monotonically WORSE at every nonzero lambda,
# every form (additive/multiplicative), every pooling stat
# (mean/median), and every estimation window tried -- see that
# experiment's score.txt for the full sweep and its conclusion that the
# per-location AR(6) already sees the current season's actual recent
# values directly and has adapted to it running hot/cold, so the pooled
# severity term mostly duplicates information the AR component already
# has. The one thing that changes here: `pool_w=0.9` blends each
# location's AR(6) coefficients 90% toward a FULLPOOL anchor fit on
# every location's residuals stacked together, which dilutes exactly
# that per-location current-trajectory memory severity2's negative
# result rests on -- so LEVER 2 is re-tested here, on this more heavily
# pooled AR family, rather than assumed negative by inheritance.
#
# LEVER 3 (time-varying/recency-weighted pooling weight or
# climatology, "optionally" per the brief) is NOT implemented here --
# left as a follow-up given the runtime budget and LEVER 2's strong
# negative prior; see the closing note in score.txt.
#
# Scored on VALIDATION SEASONS (1, 2) ONLY, against the local hub
# clone's oracle (docs/contracts.md experimental integrity) -- a tuning
# sweep. The winning combo is then used, unchanged, to write a FULL
# 5-season (1-2 validation, 3-5 held-out test) hub submission when a
# `hub_path` argument is given (model_id "nfidd-tvpool") -- each
# origin's fit is still just a vintage fit capped at its own forecast
# origin, so writing forecasts for the held-out test seasons here does
# not touch experimental integrity; only VALIDATION_ONLY informed the
# hyperparameter choices above.
#
# LIGHT + ANALYTIC: CSV/DataFrames/Dates/Statistics/Random/
# LinearAlgebra/Distributions only, no Turing -- this hierarchical
# time-varying pooling is exactly the kind of structure the FULL joint
# Turing model would fit natively (partial pooling via a hierarchical
# prior on each location's seasonal deviation, and a shared latent
# season-severity random effect informing every location's likelihood
# jointly, both with properly propagated posterior uncertainty). What
# is built here is a plug-in-estimate APPROXIMATION: the "pooling" is a
# fixed shrinkage weight tuned by grid search on validation WIS, not a
# posterior variance ratio, and the "severity" term is a point estimate
# fed into forward simulation rather than a jointly-sampled random
# effect that would also inform how much each location's own likelihood
# should move in response to what the other 10 are doing. See score.txt
# closing note for a fuller comparison.
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl [hub_path]
# writes score.txt alongside this file; if `hub_path` is given, ALSO
# writes a full 5-season submission + metadata to that hub clone
# (model_id "nfidd-tvpool") using the sweep's winning configuration.

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra
using Distributions

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
const HERE = @__DIR__
include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))
include(joinpath(PKG_DIR, "src", "hubio.jl"))
include(joinpath(PKG_DIR, "src", "scoring.jl"))

const MODEL_ID = "nfidd-tvpool"
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12
const WINDOW_WEEKS = 104
const MIN_SUPPORT = 5           # min sample size per profile bin to trust
const SMOOTH_WINDOW = 3         # circular smoothing span for the profile
const VALIDATION_ONLY = (1, 2)
const MAX_TRAIN_SEASON_YEAR = 2016  # pre-2015 history + validation seasons
const LOCAL_HUB_PATH = joinpath(PKG_DIR, "scratch-hub")  # oracle for scoring

# Round-2 stack's winning settings, held fixed throughout.
const TRANSFORM = :log
const T_DF = 10
const T_SCALE = 1.4
const POOL_W = 0.9
const BF_MODE = :additive
const BF_WINDOW = 8
const BF_STAT = :median

# Grids for the two new levers.
const LAMBDA_SHAPES = (0.0, 0.25, 0.5, 0.75, 1.0)
const AMP_SHRINKS = (0.0, 0.25, 0.5, 0.75, 1.0)
const LAMBDA_SEVERITIES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
const MIN_SEVERITY_WEEKS = 3    # matches severity2's own guard
const SEVERITY_STAT = :median   # matches severity2's preferred stat

# ---------------------------------------------------------------------
# Pooled + per-location seasonal shape (LEVER 1)
# ---------------------------------------------------------------------

"""
    week_profile(x, woys, wmax; min_support, smooth_window) -> Dict{Int,Float64}

Bin `x` (already location-mean-removed deviations, on the `transform`
scale) by week-of-season `woys` into `wmax` circular bins, mean each bin
with `< min_support` observations falling back to 0.0, circularly smooth
over `smooth_window` weeks, then re-centre to mean 0. Shared binning
core for both `build_seasonal_profile` (pooled across all 11 locations)
and `build_location_profiles` (one location at a time) so both profiles
share the same `1:wmax` key domain and can be blended week-for-week.
"""
function week_profile(
    x::Vector{Float64}, woys::Vector{Int}, wmax::Int;
    min_support::Int, smooth_window::Int,
)
    raw = [Float64[] for _ in 1:wmax]
    for i in eachindex(x)
        push!(raw[woys[i]], x[i])
    end
    means = [length(v) >= min_support ? mean(v) : 0.0 for v in raw]

    half = div(smooth_window - 1, 2)
    smoothed = similar(means)
    for w in 1:wmax
        idxs = [mod1(w + off, wmax) for off in (-half):half]
        smoothed[w] = mean(means[idxs])
    end
    smoothed .-= mean(smoothed)

    return Dict(w => smoothed[w] for w in 1:wmax)
end

"""
    season_wmax(hist, max_season_year) -> Int

Max week-of-season index across the full (all-location) training
history, used as a single shared `wmax` for both the pooled and every
per-location profile so their keys line up for blending.
"""
function season_wmax(hist::DataFrame, max_season_year::Int)
    h = hist[season_year.(hist.origin_date) .<= max_season_year, :]
    return maximum(week_of_season.(h.origin_date))
end

"""
    build_seasonal_profile(hist, wmax; transform, max_season_year,
                            min_support, smooth_window) -> Dict{Int,Float64}

Pooled week-of-season climatology on the `transform` scale: each
location's own mean is removed first, then deviations from ALL 11
locations are pooled into one shared shape. Identical in design to
`round2-stack`/`seasoncombo`'s function of the same name, refactored to
share `week_profile`'s binning core with the per-location profiles
below.
"""
function build_seasonal_profile(
    hist::DataFrame, wmax::Int; transform::Symbol, max_season_year::Int,
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

    return week_profile(dev, woys, wmax; min_support, smooth_window)
end

"""
    build_location_profiles(hist, wmax; transform, max_season_year,
                             min_support, smooth_window)
        -> Dict{String,Dict{Int,Float64}}

Per-location week-of-season deviation profile: for each location,
IDENTICAL derivation to `build_seasonal_profile` (own-location mean
removed, binned, circularly smoothed, re-centred) but estimated from
ONLY that location's own history rather than pooling deviations across
all 11. With ~13 seasons of history per location, weeks with
`< min_support` of that location's own observations fall back to 0.0
pre-smoothing -- the same rule the pooled profile uses, and the reason
a location with only sparse own-history support at a given week gets a
blended shape (`blend_shape_profiles` below) that reverts toward the
pooled shape there rather than injecting noise.
"""
function build_location_profiles(
    hist::DataFrame, wmax::Int; transform::Symbol, max_season_year::Int,
    min_support::Int, smooth_window::Int,
)
    profiles = Dict{String,Dict{Int,Float64}}()
    for loc in LOCATIONS
        h = hist[
            (hist.location .== loc) .&
            (season_year.(hist.origin_date) .<= max_season_year), :,
        ]
        if isempty(h)
            profiles[loc] = Dict(w => 0.0 for w in 1:wmax)
            continue
        end
        x = to_scale.(h.wili, transform)
        dev = x .- mean(x)
        woys = week_of_season.(h.origin_date)
        profiles[loc] = week_profile(dev, woys, wmax; min_support, smooth_window)
    end
    return profiles
end

"""
    blend_shape_profiles(pooled, per_loc, lambda_shape)
        -> Dict{String,Dict{Int,Float64}}

Per-location blended shape: `blended[loc][w] = (1 - lambda_shape) *
pooled[w] + lambda_shape * per_loc[loc][w]`. `lambda_shape = 0.0`
reproduces one shared pooled shape for every location (the round-2
stack's own seasonal term); `lambda_shape = 1.0` is the fully
idiosyncratic per-location shape with no pooling at all.
"""
function blend_shape_profiles(
    pooled::Dict{Int,Float64}, per_loc::Dict{String,Dict{Int,Float64}},
    lambda_shape::Float64,
)
    blended = Dict{String,Dict{Int,Float64}}()
    for loc in LOCATIONS
        own = per_loc[loc]
        blended[loc] = Dict(
            w => (1 - lambda_shape) * pv + lambda_shape * get(own, w, pv)
            for (w, pv) in pooled
        )
    end
    return blended
end

"""
    build_amplitude_scales(hist, blended; transform, max_season_year,
                            shrink) -> Vector{Float64}

Per-location amplitude scale, in `LOCATIONS` order: an OLS slope of
each location's own (mean-removed) deviations against its OWN blended
shape (`blended[loc]`, LEVER 1's output), shrunk toward 1.0 by `shrink`.
`shrink = 0.0` fixes amp=1 everywhere (no extra scaling beyond what
LEVER 1's shape blend already provides); `shrink = 1.0` uses the raw
per-location regression slope. Identical mechanism to
`seasonpool2/generate.jl`'s function of the same name, generalised from
regressing against the single shared pooled profile to regressing
against each location's own blended one.
"""
function build_amplitude_scales(
    hist::DataFrame, blended::Dict{String,Dict{Int,Float64}};
    transform::Symbol, max_season_year::Int, shrink::Float64,
)
    scales = ones(length(LOCATIONS))
    for (li, loc) in enumerate(LOCATIONS)
        h = hist[
            (hist.location .== loc) .&
            (season_year.(hist.origin_date) .<= max_season_year), :,
        ]
        isempty(h) && continue
        x = to_scale.(h.wili, transform)
        dev = x .- mean(x)
        prof = blended[loc]
        s = [get(prof, week_of_season(d), 0.0) for d in h.origin_date]
        denom = sum(abs2, s)
        b = denom > 1e-8 ? sum(dev .* s) / denom : 1.0
        scales[li] = 1.0 + shrink * (b - 1.0)
    end
    return scales
end

"""
    deseasonalize(Y, woy, profiles, amp) -> (R, level)

Remove each location's own mean level and its `amp`-scaled BLENDED
seasonal shape (`profiles[loc]`, LEVER 1's per-location output) from
`Y` (T x L, modelling scale). Generalises `round2-stack`/`seasoncombo`'s
function of the same name from one shared `profile` dict to one dict
PER location.
"""
function deseasonalize(
    Y::AbstractMatrix, woy::Vector{Int},
    profiles::Dict{String,Dict{Int,Float64}}, amp::Vector{Float64},
)
    T, L = size(Y)
    level = zeros(L)
    for l in 1:L
        level[l] = mean(Float64.(Y[:, l]))
    end
    R = Matrix{Float64}(undef, T, L)
    for l in 1:L
        prof = profiles[LOCATIONS[l]]
        for t in 1:T
            s = get(prof, woy[t], 0.0)
            R[t, l] = Y[t, l] - level[l] - amp[l] * s
        end
    end
    return R, level
end

# ---------------------------------------------------------------------
# Time-varying season severity, pooled across locations (LEVER 2)
# ---------------------------------------------------------------------

"""
    season_severity(R, season; min_weeks, stat) -> Float64

Pooled current-season "running hot or cold" year-effect: `stat`
(`:median` or `:mean`) across all 11 locations of that location's own
mean deseasonalized residual (`R`, T x L, already net of LEVER 1's
blended shape+amplitude term -- the AR(6) fit itself is untouched by
this) over the rows belonging to THIS split's current season
(`season .== season[end]`, the split's own forecast-origin season
index within the AR(6) window; every row already satisfies
`date <= forecast_origin` by construction of `ModelData`, so no
separate leakage guard is needed here). Because `R` is on the LOG
modelling scale, adding this pooled scalar (shrunk by
`LAMBDA_SEVERITY`) to the forward seasonal contribution is already
exactly severity2's preferred MULTIPLICATIVE-in-natural-units form.

Returns 0.0 (no adjustment) when fewer than `min_weeks` current-season
rows exist yet -- too early in the season for the pooled estimate to
be trustworthy (identical guard to severity2's `MIN_SEVERITY_WEEKS`).
"""
function season_severity(
    R::Matrix{Float64}, season::Vector{Int}; min_weeks::Int,
    stat::Symbol=:median,
)
    idx = findall(==(season[end]), season)
    length(idx) < min_weeks && return 0.0
    per_loc = [mean(R[idx, l]) for l in 1:size(R, 2)]
    return stat == :median ? median(per_loc) : mean(per_loc)
end

# ---------------------------------------------------------------------
# Backfill correction (identical to round2-stack's "core" combo)
# ---------------------------------------------------------------------

"""
    build_revision_profile(versions; transform, max_delay, min_support,
                            mode, stat) -> Dict{Tuple{String,Int},Float64}

Empirical per-(location, delay) revision profile on the `transform`
scale. Identical to `round2-stack`'s function of the same name.
"""
function build_revision_profile(
    versions::DataFrame; transform::Symbol, max_delay::Int,
    min_support::Int, mode::Symbol, stat::Symbol,
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
            if mode == :multiplicative && abs(vintage) < 1e-6
                continue
            end
            val = mode == :additive ? settled - vintage : settled / vintage
            key = (loc, delay)
            push!(get!(raw, key, Float64[]), val)
        end
    end
    profile = Dict{Tuple{String,Int},Float64}()
    for (key, vals) in raw
        length(vals) < min_support && continue
        profile[key] = stat == :median ? median(vals) : mean(vals)
    end
    return profile
end

"""
    apply_backfill_correction!(data, profile; mode, delay_cutoff)

Nudge `data.Y` in place, identical to `round2-stack`'s function of the
same name.
"""
function apply_backfill_correction!(
    data::ModelData, profile::Dict{Tuple{String,Int},Float64};
    mode::Symbol, delay_cutoff::Int,
)
    for l in 1:data.L, t in 1:data.T
        d = data.delay[t, l]
        (d < 0 || d > delay_cutoff) && continue
        ismissing(data.Y[t, l]) && continue
        key = (LOCATIONS[l], d)
        haskey(profile, key) || continue
        c = profile[key]
        data.Y[t, l] = mode == :additive ? data.Y[t, l] + c : data.Y[t, l] * c
    end
    return data
end

# ---------------------------------------------------------------------
# Per-location AR(6), blended toward a fullpool anchor (fixed at
# POOL_W=0.9, the round-2 stack's winning weight)
# ---------------------------------------------------------------------

"""
    ar_design(y, order) -> (X, yresp)

Design matrix and response for an OLS AR(`order`) fit with intercept.
Identical to `round2-stack`'s function of the same name.
"""
function ar_design(y::AbstractVector{Float64}, order::Int)
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
    return X, yresp
end

"""
    resid_sd_for(X, yresp, coef, order) -> Float64

Residual SD of `coef` (not necessarily the OLS solution for `X`,
`yresp`) evaluated on this design. Identical to `round2-stack`'s
function of the same name.
"""
function resid_sd_for(
    X::Matrix{Float64}, yresp::Vector{Float64}, coef::Vector{Float64},
    order::Int,
)
    nobs = size(X, 1)
    resid = yresp .- X * coef
    dof = max(nobs - (order + 1), 1)
    return sqrt(sum(abs2, resid) / dof)
end

"""
    fit_ar(y, order) -> (coef, X, yresp)

OLS fit of an AR(`order`) model with intercept to `y`.
"""
function fit_ar(y::AbstractVector{Float64}, order::Int)
    X, yresp = ar_design(y, order)
    coef = X \ yresp
    return coef, X, yresp
end

"""
    fit_ar_pooled(ys, order) -> coef

One OLS AR(`order`) fit on the design rows of every column in `ys`
stacked together -- the fullpool anchor. Identical to `round2-stack`'s
function of the same name.
"""
function fit_ar_pooled(ys::Vector{Vector{Float64}}, order::Int)
    designs = [ar_design(y, order) for y in ys]
    Xall = reduce(vcat, first.(designs))
    yall = reduce(vcat, last.(designs))
    return Xall \ yall
end

# ---------------------------------------------------------------------
# Path simulation: Student-t innovations, variance-matched, scaled
# (fixed at T_DF=10, T_SCALE=1.4, the round-2 stack's winning scheme)
# ---------------------------------------------------------------------

"""
    simulate_paths(y, coef, resid_sd, order, horizons, npaths; rng)
        -> Dict{Int,Vector{Float64}}

Simulate `npaths` Student-t(`T_DF`)-innovation AR(`order`) sample paths
forward from the end of `y` (modelling scale, the deseasonalized
residual), for each horizon in `horizons`. Identical to `round2-stack`'s
function of the same name with `innovation=:student_t` fixed (the
round-2 stack's winning choice).
"""
function simulate_paths(
    y::AbstractVector{Float64}, coef::Vector{Float64}, resid_sd::Float64,
    order::Int, horizons, npaths::Int; rng::Random.AbstractRNG,
)
    tdist = TDist(T_DF)
    vscale = sqrt((T_DF - 2) / T_DF)
    innov_sd = resid_sd * vscale * T_SCALE

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
            innov = innov_sd * rand(rng, tdist)
            val = pred + innov
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
# Forecast table builder -- round-2 stack core + LEVER 1 (shape/amp
# partial pooling) + LEVER 2 (pooled time-varying severity)
# ---------------------------------------------------------------------

"""
    build_forecast_table(seasons, versions_full, blended, amp,
                          backfill_profile; lambda_severity, model_id)
        -> DataFrame

Fit and forecast one point on the LEVER 1 x LEVER 2 grid for every
cross-validation split of every season in `seasons`: backfill
correction, then LEVER 1's per-location blended-shape/amplitude
deseasonalization, then per-location AR(`AR_ORDER`) blended `POOL_W`
toward the fullpool anchor, then LEVER 2's pooled current-season
severity term (`lambda_severity`, computed from the pre-severity
residual and applied only to the forward reconstruction -- the AR fit
itself never sees it), simulated forward with Student-t innovations.

`lambda_severity = 0.0` disables LEVER 2 entirely, isolating LEVER 1's
own effect.
"""
function build_forecast_table(
    seasons, versions_full, blended::Dict{String,Dict{Int,Float64}},
    amp::Vector{Float64}, backfill_profile::Dict{Tuple{String,Int},Float64};
    lambda_severity::Float64=0.0, model_id::String,
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
            apply_backfill_correction!(
                data, backfill_profile; mode=BF_MODE, delay_cutoff=BF_WINDOW,
            )
            R, level = deseasonalize(data.Y, data.woy, blended, amp)
            origin = data.origin_date
            L = data.L

            sev = lambda_severity > 0 ?
                lambda_severity * season_severity(
                    R, data.season; min_weeks=MIN_SEVERITY_WEEKS,
                    stat=SEVERITY_STAT,
                ) : 0.0

            ys = [R[:, li] for li in 1:L]
            fits = [fit_ar(ys[li], AR_ORDER) for li in 1:L]
            coefs = [f[1] for f in fits]
            Xs = [f[2] for f in fits]
            yresps = [f[3] for f in fits]
            anchor = fit_ar_pooled(ys, AR_ORDER)

            for (li, loc) in enumerate(LOCATIONS)
                coef = (1 - POOL_W) .* coefs[li] .+ POOL_W .* anchor
                resid_sd = resid_sd_for(Xs[li], yresps[li], coef, AR_ORDER)
                paths = simulate_paths(
                    ys[li], coef, resid_sd, AR_ORDER, HORIZONS, NPATHS;
                    rng=rng,
                )
                prof = blended[loc]
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    s = get(prof, week_of_season(target_end), 0.0)
                    vals = paths[h] .+ level[li] .+ amp[li] * s .+ sev
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

# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------

function main()
    t0 = time()
    hub_path = length(ARGS) >= 1 ? ARGS[1] : nothing

    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")
    truth = load_oracle(LOCAL_HUB_PATH)
    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= MAX_TRAIN_SEASON_YEAR, :,
    ]

    wmax = season_wmax(hist, MAX_TRAIN_SEASON_YEAR)
    pooled_profile = build_seasonal_profile(
        hist, wmax; transform=TRANSFORM,
        max_season_year=MAX_TRAIN_SEASON_YEAR, min_support=MIN_SUPPORT,
        smooth_window=SMOOTH_WINDOW,
    )
    loc_profiles = build_location_profiles(
        hist, wmax; transform=TRANSFORM,
        max_season_year=MAX_TRAIN_SEASON_YEAR, min_support=MIN_SUPPORT,
        smooth_window=SMOOTH_WINDOW,
    )
    backfill_profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_WINDOW,
        min_support=MIN_SUPPORT, mode=BF_MODE, stat=BF_STAT,
    )

    ones_amp = ones(length(LOCATIONS))

    # --- sanity: reproduce the round-2 stack winner exactly ---
    core_blended = blend_shape_profiles(pooled_profile, loc_profiles, 0.0)
    core = build_forecast_table(
        VALIDATION_ONLY, versions_full, core_blended, ones_amp,
        backfill_profile; lambda_severity=0.0, model_id="tvpool-core",
    )
    core_summ = score_one(core, truth)
    println("core (reproduces round2-stack log+tstudent+pool(w=0.9)): " *
            "mean_wis=$(round(core_summ.mean_wis; digits=4)) " *
            "sd_wis=$(round(core_summ.sd_wis; digits=4)) " *
            "(reference: 0.2601)")

    # --- LEVER 1: joint grid, lambda_shape x amp_shrink ---
    println("\n=== LEVER 1: shape/amplitude partial-pooling grid ===")
    shape_results = NamedTuple[]
    for lambda_shape in LAMBDA_SHAPES
        blended = blend_shape_profiles(pooled_profile, loc_profiles, lambda_shape)
        for amp_shrink in AMP_SHRINKS
            amp = build_amplitude_scales(
                hist, blended; transform=TRANSFORM,
                max_season_year=MAX_TRAIN_SEASON_YEAR, shrink=amp_shrink,
            )
            fc = build_forecast_table(
                VALIDATION_ONLY, versions_full, blended, amp,
                backfill_profile; lambda_severity=0.0,
                model_id="tvpool-shape",
            )
            summ = score_one(fc, truth)
            push!(shape_results, (
                lambda_shape=lambda_shape, amp_shrink=amp_shrink,
                mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
            ))
            println("  lambda_shape=$lambda_shape amp_shrink=$amp_shrink " *
                    "-> mean_wis=$(round(summ.mean_wis; digits=4)) " *
                    "sd_wis=$(round(summ.sd_wis; digits=4))")
        end
    end
    sort!(shape_results; by=r -> r.mean_wis)
    shape_best = shape_results[1]
    println("LEVER 1 best: lambda_shape=$(shape_best.lambda_shape) " *
            "amp_shrink=$(shape_best.amp_shrink) " *
            "mean_wis=$(round(shape_best.mean_wis; digits=4))")

    best_blended = blend_shape_profiles(
        pooled_profile, loc_profiles, shape_best.lambda_shape,
    )
    best_amp = build_amplitude_scales(
        hist, best_blended; transform=TRANSFORM,
        max_season_year=MAX_TRAIN_SEASON_YEAR, shrink=shape_best.amp_shrink,
    )

    # --- LEVER 2: severity sweep, on top of BOTH the plain core (to
    # isolate its own marginal effect) AND the LEVER-1 winner (to check
    # for interaction) ---
    println("\n=== LEVER 2: pooled time-varying severity, on the plain " *
            "core ===")
    sev_on_core = NamedTuple[]
    for lam in LAMBDA_SEVERITIES
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, core_blended, ones_amp,
            backfill_profile; lambda_severity=lam, model_id="tvpool-sevA",
        )
        summ = score_one(fc, truth)
        push!(sev_on_core, (
            lambda=lam, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("  lambda_severity=$lam -> " *
                "mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(sev_on_core; by=r -> r.mean_wis)
    sev_on_core_best = sev_on_core[1]

    println("\n=== LEVER 2: pooled time-varying severity, on the LEVER-1 " *
            "winner ===")
    sev_on_shape = NamedTuple[]
    for lam in LAMBDA_SEVERITIES
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, best_blended, best_amp,
            backfill_profile; lambda_severity=lam, model_id="tvpool-sevB",
        )
        summ = score_one(fc, truth)
        push!(sev_on_shape, (
            lambda=lam, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        ))
        println("  lambda_severity=$lam -> " *
                "mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(sev_on_shape; by=r -> r.mean_wis)
    sev_on_shape_best = sev_on_shape[1]
    println("LEVER 2 (on LEVER-1 winner) best: " *
            "lambda_severity=$(sev_on_shape_best.lambda) " *
            "mean_wis=$(round(sev_on_shape_best.mean_wis; digits=4))")

    # --- final combined winner ---
    candidates = [
        (name="core (round2-stack reproduction)",
         mean_wis=core_summ.mean_wis, sd_wis=core_summ.sd_wis),
        (name="lever1-only (shape+amp pooling)",
         mean_wis=shape_best.mean_wis, sd_wis=shape_best.sd_wis),
        (name="lever2-only (severity on core)",
         mean_wis=sev_on_core_best.mean_wis, sd_wis=sev_on_core_best.sd_wis),
        (name="lever1+lever2 (combined)",
         mean_wis=sev_on_shape_best.mean_wis,
         sd_wis=sev_on_shape_best.sd_wis),
    ]
    sort!(candidates; by=r -> r.mean_wis)
    winner = candidates[1]

    # Resolve the winning full hyperparameter set for the final,
    # full-5-season forecast table (used for both the breakdown below
    # and the hub submission, if any).
    final_lambda_shape, final_amp_shrink, final_lambda_severity =
        if winner.name == "core (round2-stack reproduction)"
            (0.0, 0.0, 0.0)
        elseif winner.name == "lever1-only (shape+amp pooling)"
            (shape_best.lambda_shape, shape_best.amp_shrink, 0.0)
        elseif winner.name == "lever2-only (severity on core)"
            (0.0, 0.0, sev_on_core_best.lambda)
        else
            (shape_best.lambda_shape, shape_best.amp_shrink,
             sev_on_shape_best.lambda)
        end
    final_blended = blend_shape_profiles(
        pooled_profile, loc_profiles, final_lambda_shape,
    )
    final_amp = build_amplitude_scales(
        hist, final_blended; transform=TRANSFORM,
        max_season_year=MAX_TRAIN_SEASON_YEAR, shrink=final_amp_shrink,
    )

    # Winner breakdown on VALIDATION seasons: by location, by season,
    # by horizon.
    winner_fc = build_forecast_table(
        VALIDATION_ONLY, versions_full, final_blended, final_amp,
        backfill_profile; lambda_severity=final_lambda_severity,
        model_id=MODEL_ID,
    )
    winner_scored = score_forecasts(winner_fc, truth; scale=:natural)
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

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "hierarchical time-varying partial-pooled " *
                     "seasonality -- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "round-2 stack winner (log+tstudent+pool(w=0.9)): " *
                     "mean_wis=0.2601 sd_wis=0.2587 " *
                     "(experiments/simple-round/round2-stack/score.txt)")
        println(io, "local reproduction: mean_wis=" *
                     "$(round(core_summ.mean_wis; digits=4)) " *
                     "sd_wis=$(round(core_summ.sd_wis; digits=4))")
        println(io)
        println(io, "=== LEVER 1: shape/amplitude partial-pooling grid ===")
        for r in shape_results
            println(io, "  lambda_shape=$(r.lambda_shape) " *
                         "amp_shrink=$(r.amp_shrink) -> mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4))")
        end
        lever1_pct = 100 * (core_summ.mean_wis - shape_best.mean_wis) /
                     core_summ.mean_wis
        println(io, "LEVER 1 best: lambda_shape=$(shape_best.lambda_shape) " *
                     "amp_shrink=$(shape_best.amp_shrink) mean_wis=" *
                     "$(round(shape_best.mean_wis; digits=4)) " *
                     "($(round(lever1_pct; digits=2))% vs core)")
        println(io)
        println(io, "=== LEVER 2: pooled time-varying severity, on the " *
                     "plain core ===")
        for r in sev_on_core
            pct = 100 * (core_summ.mean_wis - r.mean_wis) / core_summ.mean_wis
            println(io, "  lambda_severity=$(r.lambda) -> mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4)) " *
                         "($(round(pct; digits=2))% vs core)")
        end
        println(io, "best: lambda_severity=$(sev_on_core_best.lambda) " *
                     "mean_wis=$(round(sev_on_core_best.mean_wis; digits=4))")
        println(io)
        println(io, "=== LEVER 2: pooled time-varying severity, on the " *
                     "LEVER-1 winner ===")
        for r in sev_on_shape
            pct = 100 * (shape_best.mean_wis - r.mean_wis) /
                  shape_best.mean_wis
            println(io, "  lambda_severity=$(r.lambda) -> mean_wis=" *
                         "$(round(r.mean_wis; digits=4)) sd_wis=" *
                         "$(round(r.sd_wis; digits=4)) " *
                         "($(round(pct; digits=2))% vs lever1-only)")
        end
        println(io, "best: lambda_severity=$(sev_on_shape_best.lambda) " *
                     "mean_wis=$(round(sev_on_shape_best.mean_wis; digits=4))")
        println(io)
        println(io, "=== overall comparison ===")
        for r in candidates
            println(io, rpad(r.name, 38) *
                         "mean_wis=$(rpad(round(r.mean_wis; digits=4), 8)) " *
                         "sd_wis=$(round(r.sd_wis; digits=4))")
        end
        println(io)
        println(io, "=== winner: $(winner.name) ===")
        println(io, "hyperparameters: lambda_shape=$(final_lambda_shape) " *
                     "amp_shrink=$(final_amp_shrink) " *
                     "lambda_severity=$(final_lambda_severity)")
        println(io, "mean_wis=$(round(winner.mean_wis; digits=4)) " *
                     "sd_wis=$(round(winner.sd_wis; digits=4))")
        vs_ref = 0.2601 - winner.mean_wis
        vs_pct = 100 * vs_ref / 0.2601
        println(io, "vs round-2 stack (0.2601): $(round(vs_ref; digits=4)) " *
                     "($(round(vs_pct; digits=2))%)")
        println(io)
        println(io, "winner mean WIS by location:")
        for row in eachrow(by_loc)
            println(io, "  $(rpad(row.location, 16)) " *
                         "$(round(row.mean_wis; digits=4)) (n=$(row.n))")
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
        println(io, "=== analytic approximation vs. what the joint " *
                     "Turing model would do ===")
        println(io, "This is a plug-in-estimate stand-in for the " *
                     "hierarchical structure the full Bayesian joint " *
                     "model fits natively: LAMBDA_SHAPE/AMP_SHRINK are " *
                     "fixed shrinkage weights chosen by grid search on " *
                     "validation WIS, not posterior variance ratios (a " *
                     "hierarchical prior would let the data set how " *
                     "much each location's shape departs from the " *
                     "pooled one, including different amounts of " *
                     "pooling for well- vs sparsely-observed locations, " *
                     "rather than one grid-searched weight applied " *
                     "uniformly to all 11). Likewise LAMBDA_SEVERITY's " *
                     "pooled year-effect is a point estimate fed " *
                     "forward into simulation, not a jointly-sampled " *
                     "latent variable each location's likelihood " *
                     "informs and is informed by simultaneously with " *
                     "its own AR dynamics and quantified posterior " *
                     "uncertainty on the year-effect itself (this " *
                     "analytic version has no uncertainty on lambda_" *
                     "shape/amp_shrink/lambda_severity at all -- they " *
                     "are point-estimated constants for every split). " *
                     "What the approximation likely still captures: " *
                     "the DIRECTION of the AR(6)-pool_w=0.9 finding " *
                     "(round2-stack/seasonpool2) that pooling toward a " *
                     "shared structure helps once seasonality is " *
                     "already accounted for, and whichever LEVER 2 " *
                     "result appears above.")
        println(io)
        println(io, "LEVER 3 (time-varying/recency-weighted pooling " *
                     "weight or climatology) not implemented here -- " *
                     "left as follow-up given severity2's strong prior " *
                     "that season-level scaling on top of an AR(6) " *
                     "that already sees the current trajectory adds " *
                     "little, and the runtime budget of this round.")
    end

    dt = round(time() - t0; digits=1)
    println("\nwinner: $(winner.name) mean_wis=" *
            "$(round(winner.mean_wis; digits=4)) " *
            "sd_wis=$(round(winner.sd_wis; digits=4))")
    println("wrote score.txt in $(dt)s total")

    if hub_path !== nothing
        full_forecast = build_forecast_table(
            (1, 2, 3, 4, 5), versions_full, final_blended, final_amp,
            backfill_profile; lambda_severity=final_lambda_severity,
            model_id=MODEL_ID,
        )
        write_submission(full_forecast, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="nfidd", model_abbr="tvpool", designated=true,
        )
        println("wrote full 5-season submission + metadata to $(hub_path)")
    end

    return candidates
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
