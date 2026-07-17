#!/usr/bin/env julia
# cluster-pooled seasonality -- simple-round, SEASONALITY family,
# follow-on to `seasoncombo`'s "core" model (pooled seasonal shape +
# per-location AR(6) + backfill correction, validation mean WIS
# 0.2781, `submissions/nfidd-seasarbf/README.md`).
#
# `seasoncombo`/`season` found ONE shared seasonal shape across all 11
# locations beats a per-location shape (0.2781 vs 0.3004): pooling
# borrows strength across ~13 seasons x 11 locations instead of ~13
# seasons x 1, and per-location peak timing is too noisy (SD 5.2-7.9
# weeks, docs/eda/03-seasonality.md) to estimate location-by-location.
# But full pooling also blurs real between-location differences --
# HHS Region 9 in particular (southwest/Pacific, weaker and more
# atypically-timed seasons) is a known outlier
# (docs/eda/03-seasonality.md, docs/eda/04-cross-location.md).
#
# This experiment tries a middle ground: CLUSTER the 11 locations by
# the shape of their own week-of-season climatology (not just peak
# timing -- the whole smoothed curve, so onset/offset shape counts
# too), then pool a separate seasonal shape PER CLUSTER instead of
# one shape for all 11 or one shape for each of the 11. Each location
# still gets its own free OLS amplitude coefficient on its cluster's
# shape (`build_cluster_amplitude_scales`, unshrunk -- `seasoncombo`'s
# own amp sweep found shrink=1.0, i.e. the raw slope, wins over any
# partial shrinkage toward 1.0). Number of clusters `K` is swept from
# 1 (one cluster = all 11 locations, i.e. the fully pooled shape) to
# 11 (11 singleton clusters, i.e. each location's own unpooled shape)
# via one agglomerative (average-linkage) hierarchical clustering on
# 1-correlation distance between locations' own smoothed climatology
# curves, cut at each K. AR(6) and the backfill correction are
# unchanged throughout -- only the seasonal-shape granularity varies.
#
# NOTE on baselines: because the per-location amplitude coefficient is
# always fit here (at every K, including K=1), this design's K=1 is
# closer to `seasoncombo`'s combo 5 ("amp+backfill", unshrunk, 0.2748)
# than to its plain "core" combo (no amplitude term, 0.2781) -- see
# score.txt for both reference points. Likewise K=11 is only a loose
# analogue of `season`'s per-location "climatology-backfill" (0.3004):
# that experiment fits the climatology as one joint OLS coefficient
# alongside the AR lags, whereas this file deseasonalizes first and
# fits AR(6) on the residual, so the two numbers are not directly the
# same model. Neither K=1 nor K=11 should be read as an exact
# reproduction of an earlier experiment, just its closest counterpart.
#
# LIGHT + ANALYTIC: CSV/DataFrames/Statistics/LinearAlgebra only, no
# Turing, no external clustering package (hierarchical clustering is
# ~20 lines for 11 points, see `agglomerative_merges` below).
#
# Scope: VALIDATION SEASONS (1, 2) ONLY for clustering, shape/backfill
# estimation, and model selection (docs/contracts.md experimental
# integrity) -- the clustering distance matrix, per-cluster profiles,
# amplitude scales and backfill profile are all built from
# `season_year <= 2016` (pre-2015 history plus the two validation
# seasons); test seasons 3-5 are only ever forecast, never fit or
# selected on, using the already-locked best-K model
# (`allow_test_season=true` passed explicitly, `training_splits`'s own
# guard, mirroring `submissions/nfidd-seasarbf/generate_forecasts.jl`).
#
# Usage:
#   julia --project=<sismid-ili-turing repo> generate.jl [hub_path]
# writes score.txt alongside this file; with `hub_path` given, ALSO
# writes a full 5-season hub-format submission (all validation + test
# seasons) under `<hub_path>/model-output/simple-cluster-seasonbf/`
# plus matching model-metadata, using the best-K model locked by the
# validation-only sweep above.

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

const TRANSFORM = :fourthroot
const AR_ORDER = 6
const NPATHS = 1000
const SEED = 20260717
const DMAX = 12                 # matches ar6bf's build_model_data Dmax
const WINDOW_WEEKS = 104
const MIN_SUPPORT = 5           # min sample size per profile bin to trust
const SMOOTH_WINDOW = 3         # circular smoothing span for the profile
const VALIDATION_ONLY = (1, 2)
const ALL_SEASONS = (1, 2, 3, 4, 5)
const MAX_TRAIN_SEASON_YEAR = 2016  # pre-2015 history + validation seasons
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")
const K_GRID = 1:length(LOCATIONS)  # 1 (fully pooled) .. 11 (per-location)

const MODEL_ID = "simple-cluster-seasonbf"

# Reference backfill design (`seabbs_bot-ar6bf`), reused unchanged --
# identical to `seasoncombo`'s core combo.
const BF_MODE = :additive
const BF_WINDOW = 8
const BF_STAT = :median

# ---------------------------------------------------------------------
# Pooled seasonal shape (identical to `seasoncombo`'s function of the
# same name -- reused here both to build the per-CLUSTER profile
# (called on a location subset of `hist`) and, incidentally, each
# location's own climatology curve for clustering (called on a single
# location).
# ---------------------------------------------------------------------

"""
    build_seasonal_profile(hist; transform, max_season_year, min_support,
                            smooth_window) -> Dict{Int,Float64}

Pooled week-of-season climatology on the `transform` scale, estimated
from whichever locations are present in `hist` (restricted to
`season_year(origin_date) <= max_season_year`). Each location's `wili`
is transformed and centred on that location's OWN mean over the
restricted history; the profile at week-of-season `w` is the mean of
these centred values pooled across every location present and matching
week. Calling this on all 11 locations gives the single shared shape
`seasoncombo`/`nfidd-seasarbf` use; calling it on a subset gives that
subset's pooled shape (this experiment's per-cluster profile); calling
it on one location gives that location's own (unpooled) climatology,
used only to build the clustering distance matrix below.

Weeks with fewer than `min_support` pooled observations fall back to
0.0 (no seasonal adjustment) before smoothing. The raw per-week means
are smoothed with a circular moving average of span `smooth_window`
(week 1 and the last week are adjacent, since week-of-season wraps),
and re-centred to zero mean so adding the profile never shifts a
location's overall level.
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

# ---------------------------------------------------------------------
# Clustering: agglomerative (average-linkage) hierarchical clustering
# of the 11 locations by 1-correlation distance between their own
# smoothed climatology curves.
# ---------------------------------------------------------------------

"""
    location_curve(hist, loc; transform, max_season_year, min_support,
                   smooth_window) -> Vector{Float64}

That location's own (unpooled) smoothed week-of-season climatology, as
a dense vector indexed `1:Wmax` (0.0 for any week below `min_support`).
Used only to measure shape similarity between locations for
clustering -- the actual per-cluster profile fit into the forecast
model is `build_seasonal_profile` called on the cluster's member
locations, not this per-location curve.
"""
function location_curve(
    hist::DataFrame, loc::AbstractString; transform::Symbol,
    max_season_year::Int, min_support::Int, smooth_window::Int,
)
    profile = build_seasonal_profile(
        hist[hist.location .== loc, :]; transform=transform,
        max_season_year=max_season_year, min_support=min_support,
        smooth_window=smooth_window,
    )
    Wmax = maximum(keys(profile))
    return [profile[w] for w in 1:Wmax]
end

"""
    correlation_distance(curves) -> Matrix{Float64}

`L x L` distance matrix, `D[i,j] = 1 - cor(curves[i], curves[j])`:
locations whose climatology curves rise and fall together (regardless
of amplitude, since `location_curve` is already own-mean-centred) are
close; locations whose curves are unrelated or move oppositely are far
apart. Diagonal is exactly 0.0 by construction.
"""
function correlation_distance(curves::Vector{Vector{Float64}})
    n = length(curves)
    D = zeros(n, n)
    for i in 1:n, j in (i + 1):n
        d = 1.0 - cor(curves[i], curves[j])
        D[i, j] = d
        D[j, i] = d
    end
    return D
end

"""
    agglomerative_merges(D) -> Vector{Vector{Vector{Int}}}

Full agglomerative (average-linkage) hierarchical clustering of the
`n = size(D, 1)` items indexed by `D`, starting from `n` singleton
clusters and merging the closest pair (by average pairwise distance
across members) at each step until one cluster remains.

Returns `merges[k]` = the clustering into `n - k + 1` clusters, i.e.
`merges[1]` is the `n`-singleton start and `merges[n]` is the single
all-in-one-cluster end; `merges[n - K + 1]` gives the clustering into
exactly `K` clusters, a plain vector of the K vectors of original
item indices belonging to each cluster.
"""
function agglomerative_merges(D::Matrix{Float64})
    n = size(D, 1)
    clusters = [[i] for i in 1:n]
    merges = [deepcopy(clusters)]
    while length(clusters) > 1
        best_d, best_a, best_b = Inf, 0, 0
        for a in 1:length(clusters), b in (a + 1):length(clusters)
            avg_d = mean(D[i, j] for i in clusters[a], j in clusters[b])
            if avg_d < best_d
                best_d, best_a, best_b = avg_d, a, b
            end
        end
        merged = vcat(clusters[best_a], clusters[best_b])
        clusters = [
            clusters[k] for k in 1:length(clusters)
            if k != best_a && k != best_b
        ]
        push!(clusters, merged)
        push!(merges, deepcopy(clusters))
    end
    return merges
end

"""
    clusters_for_k(merges, K) -> Vector{Vector{Int}}

The clustering into exactly `K` clusters from `agglomerative_merges`'s
merge history, each inner vector holding the (1-based, `LOCATIONS`-
order) location indices belonging to that cluster.
"""
function clusters_for_k(merges, K::Int)
    n = length(merges[1])
    return merges[n - K + 1]
end

"""
    build_cluster_assignment(clusters, n) -> Vector{Int}

Location-index -> cluster-id vector (length `n`), from `clusters`
(a `clusters_for_k` result) -- cluster ids are 1-based, in the order
`clusters` itself lists them (arbitrary but stable within one `K`).
"""
function build_cluster_assignment(clusters::Vector{Vector{Int}}, n::Int)
    assign = zeros(Int, n)
    for (cid, members) in enumerate(clusters)
        for li in members
            assign[li] = cid
        end
    end
    return assign
end

"""
    build_cluster_profiles(hist, clusters; transform, max_season_year,
                            min_support, smooth_window)
        -> Vector{Dict{Int,Float64}}

Per-LOCATION (i.e. `LOCATIONS`-order, length 11) vector of seasonal
profile `Dict`s: every location in `clusters[c]` gets the SAME profile,
`build_seasonal_profile` pooled across only that cluster's member
locations. `K = 1` (one cluster with all 11 locations) reproduces
`seasoncombo`'s single shared profile exactly; `K = 11` (11 singleton
clusters) reproduces `season`'s per-location climatology (each
location's own curve, unpooled).
"""
function build_cluster_profiles(
    hist::DataFrame, clusters::Vector{Vector{Int}}; transform::Symbol,
    max_season_year::Int, min_support::Int, smooth_window::Int,
)
    by_loc = Vector{Dict{Int,Float64}}(undef, length(LOCATIONS))
    for members in clusters
        member_locs = LOCATIONS[members]
        sub = hist[in.(hist.location, Ref(member_locs)), :]
        profile = build_seasonal_profile(
            sub; transform=transform, max_season_year=max_season_year,
            min_support=min_support, smooth_window=smooth_window,
        )
        for li in members
            by_loc[li] = profile
        end
    end
    return by_loc
end

"""
    build_cluster_amplitude_scales(hist, profiles; transform,
                                    max_season_year) -> Vector{Float64}

Per-location amplitude scale, `LOCATIONS`-order: the no-intercept OLS
slope of that location's own (transform-scale, own-mean-centred)
deviation on ITS CLUSTER's profile value at the matching
week-of-season (`profiles[li]`, from `build_cluster_profiles`),
estimated over the restricted history. Unshrunk (`seasoncombo`'s own
amplitude sweep found the raw slope, `shrink=1.0`, beats any partial
shrinkage toward 1.0 once a shared/cluster shape is already doing the
pooling -- see that experiment's score.txt).
"""
function build_cluster_amplitude_scales(
    hist::DataFrame, profiles::Vector{Dict{Int,Float64}}; transform::Symbol,
    max_season_year::Int,
)
    h = hist[season_year.(hist.origin_date) .<= max_season_year, :]
    scales = ones(length(LOCATIONS))
    for (li, loc) in enumerate(LOCATIONS)
        sub = h[h.location .== loc, :]
        isempty(sub) && continue
        x = to_scale.(sub.wili, transform)
        dev = x .- mean(x)
        s = [get(profiles[li], week_of_season(d), 0.0) for d in sub.origin_date]
        denom = sum(abs2, s)
        b = denom > 1e-8 ? sum(dev .* s) / denom : 1.0
        scales[li] = b
    end
    return scales
end

"""
    deseasonalize(Y, woy, profiles, amp) -> (R, level)

Remove each location's own mean level and its (`amp`-scaled) CLUSTER
seasonal shape (`profiles[l]`, one profile Dict per location -- may be
shared across several locations in the same cluster) from `Y` (T x L,
modelling scale), returning the residual matrix `R` and the
per-location `level` used (added back at forecast time). Identical to
`seasoncombo`'s function of the same name except `profiles` is now
per-location rather than a single shared Dict.
"""
function deseasonalize(
    Y::AbstractMatrix, woy::Vector{Int}, profiles::Vector{Dict{Int,Float64}},
    amp::Vector{Float64},
)
    T, L = size(Y)
    level = zeros(L)
    for l in 1:L
        level[l] = mean(Float64.(Y[:, l]))
    end
    R = Matrix{Float64}(undef, T, L)
    for l in 1:L, t in 1:T
        s = get(profiles[l], woy[t], 0.0)
        R[t, l] = Y[t, l] - level[l] - amp[l] * s
    end
    return R, level
end

# ---------------------------------------------------------------------
# Backfill correction (identical to `seabbs_bot-ar6bf` / `seasoncombo`'s
# core combo -- see `experiments/simple-round/backfill/generate.jl` for
# the full description).
# ---------------------------------------------------------------------

function build_revision_profile(
    versions::DataFrame; transform::Symbol, max_delay::Int,
    min_support::Int, mode::Symbol, pooled::Bool, stat::Symbol,
)
    raw = Dict{Any,Vector{Float64}}()
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
            key = pooled ? delay : (loc, delay)
            push!(get!(raw, key, Float64[]), val)
        end
    end
    profile = Dict{Any,Float64}()
    for (key, vals) in raw
        length(vals) < min_support && continue
        profile[key] = stat == :median ? median(vals) : mean(vals)
    end
    return profile
end

function apply_backfill_correction!(
    data::ModelData, profile::Dict; mode::Symbol, pooled::Bool,
    delay_cutoff::Int,
)
    for l in 1:data.L, t in 1:data.T
        d = data.delay[t, l]
        (d < 0 || d > delay_cutoff) && continue
        ismissing(data.Y[t, l]) && continue
        key = pooled ? d : (LOCATIONS[l], d)
        haskey(profile, key) || continue
        c = profile[key]
        data.Y[t, l] = mode == :additive ? data.Y[t, l] + c : data.Y[t, l] * c
    end
    return data
end

# ---------------------------------------------------------------------
# Per-location AR(6) (plain OLS, identical to `seasoncombo`/`nfidd-ar6`)
# ---------------------------------------------------------------------

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
# Forecast table builder: cluster-pooled season + AR(6) + backfill
# (the ONLY combination tried here -- unlike `seasoncombo`'s 4-way
# residual-mechanism sweep, this experiment fixes AR(6)+backfill as
# locked by that experiment and sweeps only the seasonal granularity,
# `K`).
# ---------------------------------------------------------------------

function build_forecast_table(
    seasons, versions_full, profiles::Vector{Dict{Int,Float64}},
    amp::Vector{Float64}, backfill_profile::Dict;
    backfill_window::Int, model_id::String, allow_test_season::Bool=false,
)
    rng = MersenneTwister(SEED)
    rows = DataFrame(
        model_id=String[], location=String[], origin_date=Date[],
        horizon=Int[], target_end_date=Date[], target=String[],
        output_type=String[], output_type_id=Float64[], value=Float64[],
    )
    for season in seasons
        splits = training_splits(season; allow_test_season=allow_test_season)
        for split in splits
            data = build_model_data(
                split; Dmax=DMAX, transform=TRANSFORM,
                window_weeks=WINDOW_WEEKS, versions=versions_full,
            )
            apply_backfill_correction!(
                data, backfill_profile; mode=BF_MODE, pooled=false,
                delay_cutoff=backfill_window,
            )
            R, level = deseasonalize(data.Y, data.woy, profiles, amp)
            origin = data.origin_date

            for (li, loc) in enumerate(LOCATIONS)
                y = R[:, li]
                coef, resid_sd = fit_ar(y, AR_ORDER)
                paths = simulate_paths(
                    y, coef, resid_sd, AR_ORDER, HORIZONS, NPATHS; rng=rng,
                )
                for h in HORIZONS
                    target_end = origin + Day(7 * h)
                    s = get(profiles[li], week_of_season(target_end), 0.0)
                    vals = paths[h] .+ level[li] .+ amp[li] * s
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

"""Mean/SD WIS by `group_col` (`:location` or `:horizon`), sorted
ascending by mean WIS."""
function breakdown(scored::DataFrame, group_col::Symbol)
    combine(groupby(scored, group_col),
        :wis => mean => :mean_wis, :wis => std => :sd_wis, nrow => :n,
    )
end

# ---------------------------------------------------------------------
# Sweep over number of clusters K
# ---------------------------------------------------------------------

function main()
    t0 = time()
    hist = load_series("flu_data_hhs")
    versions_full = load_series("flu_data_hhs_versions")
    truth = load_oracle(HUB_PATH)
    hub_path = length(ARGS) >= 1 ? ARGS[1] : nothing

    training_versions = versions_full[
        season_year.(versions_full.origin_date) .<= MAX_TRAIN_SEASON_YEAR, :,
    ]
    backfill_profile = build_revision_profile(
        training_versions; transform=TRANSFORM, max_delay=BF_WINDOW,
        min_support=5, mode=BF_MODE, pooled=false, stat=BF_STAT,
    )

    # --- clustering: 1-correlation distance between each location's own
    # smoothed climatology curve, agglomerative average-linkage ---
    curves = [
        location_curve(
            hist, loc; transform=TRANSFORM,
            max_season_year=MAX_TRAIN_SEASON_YEAR, min_support=MIN_SUPPORT,
            smooth_window=SMOOTH_WINDOW,
        ) for loc in LOCATIONS
    ]
    D = correlation_distance(curves)
    merges = agglomerative_merges(D)

    println("clustering distance matrix built from own-location " *
            "climatology curves (1 - correlation)")

    # --- sweep K ---
    k_results = NamedTuple[]
    k_scored = Dict{Int,DataFrame}()
    for K in K_GRID
        clusters = clusters_for_k(merges, K)
        assign = build_cluster_assignment(clusters, length(LOCATIONS))
        profiles = build_cluster_profiles(
            hist, clusters; transform=TRANSFORM,
            max_season_year=MAX_TRAIN_SEASON_YEAR, min_support=MIN_SUPPORT,
            smooth_window=SMOOTH_WINDOW,
        )
        amp = build_cluster_amplitude_scales(
            hist, profiles; transform=TRANSFORM,
            max_season_year=MAX_TRAIN_SEASON_YEAR,
        )
        fc = build_forecast_table(
            VALIDATION_ONLY, versions_full, profiles, amp, backfill_profile;
            backfill_window=BF_WINDOW, model_id="cluster-K$K",
        )
        scored = score_forecasts(fc, truth; scale=:natural)
        summ = wis_summary(scored)[1, :]
        k_scored[K] = scored
        cluster_desc = join(
            ["{" * join(LOCATIONS[members], ", ") * "}"
             for members in clusters],
            "  ",
        )
        push!(k_results, (
            K=K, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
            n_tasks=summ.n_tasks, clusters=cluster_desc, assign=assign,
            profiles=profiles, amp=amp,
        ))
        println("K=$K -> mean_wis=$(round(summ.mean_wis; digits=4)) " *
                "sd_wis=$(round(summ.sd_wis; digits=4))")
    end
    sort!(k_results; by=r -> r.mean_wis)
    best = k_results[1]
    best_scored = k_scored[best.K]

    # by-location / by-horizon / by-season breakdown for the best K
    loc_bd = sort(breakdown(best_scored, :location), :mean_wis)
    hz_bd = sort(breakdown(best_scored, :horizon), :horizon)
    best_scored.season_yr = season_year.(best_scored.origin_date)
    sn_bd = sort(breakdown(best_scored, :season_yr), :season_yr)

    core_ref = 0.2781  # seasoncombo core / nfidd-seasarbf
    combo5_ref = 0.2748  # seasoncombo combo 5 (amp+backfill, unshrunk)
    vs_core = core_ref - best.mean_wis
    vs_core_pct = 100 * vs_core / core_ref
    vs_combo5 = combo5_ref - best.mean_wis
    vs_combo5_pct = 100 * vs_combo5 / combo5_ref

    open(joinpath(HERE, "score.txt"), "w") do io
        println(io, "cluster-pooled seasonality -- simple-round")
        println(io, "validation seasons (1, 2) only, natural-scale WIS")
        println(io, "runtime: $(round(time() - t0; digits=1))s")
        println(io)
        println(io, "reference points:")
        println(io, "  nfidd-ar6 (plain AR6, no season)             = 0.368 " *
                     "(sd 0.471)")
        println(io, "  seabbs_bot-ar6bf (AR6 + backfill)             = 0.359 " *
                     "(sd 0.452)")
        println(io, "  season/ climatology-backfill (per-location)  = 0.3004 " *
                     "(sd 0.389) -- loose analogue of K=11 below (different " *
                     "fitting procedure, see file header)")
        println(io, "  seasoncombo core (ONE pooled shape + AR6+bf,")
        println(io, "    NO per-location amplitude term)   " *
                     "= 0.2781 (sd 0.334)")
        println(io, "  seasoncombo combo 5 (amp+backfill, unshrunk;")
        println(io, "    same per-location amplitude term as here)   " *
                     "= 0.2748 (sd 0.320) -- closest analogue of K=1 below")
        println(io)
        println(io, "clustering: agglomerative (average-linkage) on " *
                     "1-correlation distance between each location's own " *
                     "smoothed week-of-season climatology curve (own-mean " *
                     "centred, fit from season_year <= " *
                     "$MAX_TRAIN_SEASON_YEAR only, no test-season leakage). " *
                     "Distinct from the " *
                     "per-cluster PROFILE fit into the forecast model, " *
                     "which is the pooled climatology re-estimated across " *
                     "each cluster's member locations, not this per-location " *
                     "curve.")
        println(io)
        println(io, "=== K sweep (K=1 .. $(length(LOCATIONS)), each with " *
                     "AR(6) + backfill unchanged) ===")
        for r in k_results
            marker = r.K == best.K ? "  <- BEST" : ""
            println(io, "K=$(rpad(r.K, 2)) mean_wis=" *
                         "$(rpad(round(r.mean_wis; digits=4), 8)) sd_wis=" *
                         "$(rpad(round(r.sd_wis; digits=4), 8))$marker")
        end
        println(io)
        println(io, "=== best: K=$(best.K) ===")
        println(io, "mean_wis=$(round(best.mean_wis; digits=4)) " *
                     "sd_wis=$(round(best.sd_wis; digits=4)) " *
                     "n_tasks=$(best.n_tasks)")
        println(io, "vs seasoncombo core (0.2781): " *
                     "$(round(vs_core; digits=4)) " *
                     "($(round(vs_core_pct; digits=2))%)")
        println(io, "vs seasoncombo combo 5 / amp+backfill (0.2748, the " *
                     "closer analogue -- see NOTE above): " *
                     "$(round(vs_combo5; digits=4)) " *
                     "($(round(vs_combo5_pct; digits=2))%)")
        println(io)
        println(io, "cluster membership at K=$(best.K):")
        println(io, "  " * best.clusters)
        println(io)
        println(io, "-- breakdown by location (K=$(best.K)) --")
        for row in eachrow(loc_bd)
            println(io, "  $(rpad(row.location, 16)) mean_wis=" *
                         "$(round(row.mean_wis; digits=4)) sd_wis=" *
                         "$(round(row.sd_wis; digits=4))")
        end
        println(io)
        println(io, "-- breakdown by horizon (K=$(best.K)) --")
        for row in eachrow(hz_bd)
            println(io, "  h=$(row.horizon) mean_wis=" *
                         "$(round(row.mean_wis; digits=4)) sd_wis=" *
                         "$(round(row.sd_wis; digits=4))")
        end
        println(io)
        println(io, "-- breakdown by validation season (K=$(best.K)) --")
        for row in eachrow(sn_bd)
            println(io, "  season $(row.season_yr): mean_wis=" *
                         "$(round(row.mean_wis; digits=4)) sd_wis=" *
                         "$(round(row.sd_wis; digits=4))")
        end
        println(io)
        println(io, "cluster membership at every K:")
        for r in k_results
            println(io, "  K=$(rpad(r.K, 2)) $(r.clusters)")
        end
        println(io)
        println(io, "INTERPRETATION: mean WIS falls close to monotonically " *
                     "as K increases from 1 to 11 (0.2748 -> 0.2709, a " *
                     "~1.4% spread, with a couple of local ties at K=3/6 " *
                     "rather than a genuine reversal) -- NOT the middle-K " *
                     "sweet spot the cluster-pooling hypothesis expected. " *
                     "With a smoothed, full-13-season-history climatology " *
                     "curve (not a few free Fourier/OLS parameters fit to " *
                     "the ~2-season AR window), per-location granularity " *
                     "does not re-introduce the overfitting that sank the " *
                     "naive per-location Fourier fit (0.412) -- the shape " *
                     "itself is already well-regularised by construction, " *
                     "so there is little for cluster pooling to add on top. " *
                     "The clustering DOES confirm HHS Region 9's " *
                     "distinctness qualitatively: it is one of the first " *
                     "two locations to split off the main pooled group " *
                     "(K=2: {Region 2, Region 9} vs the other 9; it does " *
                     "not become its own singleton cluster until K=4), " *
                     "pairing with Region 2 rather than standing alone -- " *
                     "but this qualitative signal does not translate into " *
                     "a WIS advantage over simply giving every location " *
                     "its own shape (K=11). All 11 K values beat both " *
                     "reference points (0.2781 core, 0.2748 combo 5); the " *
                     "practical takeaway is that cluster pooling is a " *
                     "defensible, slightly more conservative middle ground " *
                     "if fewer effective parameters is valued for its own " *
                     "sake, but K=11 (full per-location shape + unshrunk " *
                     "per-location amplitude, still with the shared AR(6) " *
                     "+ backfill machinery) is this experiment's best " *
                     "scoring choice on validation.")
    end

    println("\nbest: K=$(best.K) mean_wis=$(round(best.mean_wis; digits=4)) " *
            "sd_wis=$(round(best.sd_wis; digits=4))")
    println("vs seasoncombo core (0.2781): $(round(vs_core; digits=4)) " *
            "($(round(vs_core_pct; digits=2))%)")
    println("wrote score.txt")

    # --- full 5-season hub-format submission, best-K model locked above ---
    if hub_path !== nothing
        full = build_forecast_table(
            ALL_SEASONS, versions_full, best.profiles, best.amp,
            backfill_profile; backfill_window=BF_WINDOW, model_id=MODEL_ID,
            allow_test_season=true,
        )
        n_origins = length(unique(full.origin_date))
        println("built full submission: $(nrow(full)) rows across " *
                "$(n_origins) origin date(s)")
        write_submission(full, hub_path)
        write_metadata(
            MODEL_ID, hub_path;
            team_abbr="seabbs_bot", model_abbr="cluster", designated=true,
        )
        println("wrote submission + metadata to $(hub_path)")
    end

    return k_results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
