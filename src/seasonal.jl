# Canonical leak-free pooled seasonal climatology and empirical
# backfill/revision profile, rebuilt PER SPLIT from only data strictly
# before that split's own `forecast_origin`.
#
# Nearly every experiments/simple-round/*/generate.jl driver carried its
# own copy of `build_seasonal_profile`/`build_revision_profile`, and most
# of those built the profile ONCE from a fixed `season_year <= 2016`
# cutoff, reused unchanged across every split -- for a validation-season
# split that leaks that same season's own future weeks (and the other
# validation season) into the profile used to correct/deseasonalize it.
# `experiments/simple-round/round2-stack/generate.jl` has the first
# leak-free version (per-split, `forecast_origin`-gated); this file
# canonicalises that version so future drivers `include` it instead of
# hand-rolling another copy. See docs/lessons.md and submissions/
# README.md's "Hub submissions PAUSED" note for the full writeup.
#
# This file is standalone: it can be tested via
#   include("src/core.jl"); include("src/data.jl"); include("src/seasonal.jl")
# assuming `LOCATIONS`, `to_scale`, `week_of_season` (core.jl) and
# `ModelData` (core.jl) are in scope.

using DataFrames
using Dates
using Statistics

"""
    build_seasonal_profile(hist, forecast_origin; transform, min_support,
                            smooth_window) -> Dict{Int,Float64}

Pooled week-of-season climatology on the `transform` scale, rebuilt PER
SPLIT from only `hist` rows strictly before `forecast_origin` --
LEAK-FREE. For each location its own mean level is removed first
(deviation-from-own-mean); deviations are pooled across all `LOCATIONS`,
averaged per week-of-season bin (a bin with fewer than `min_support`
observations falls back to 0.0, no seasonal adjustment), then circularly
smoothed with a `smooth_window`-wide moving average and re-centred to
zero mean across the cycle so adding the profile never shifts a
location's overall level, only its within-year shape.
"""
function build_seasonal_profile(
    hist::DataFrame, forecast_origin::Date; transform::Symbol,
    min_support::Int, smooth_window::Int,
)
    h = hist[hist.origin_date .< forecast_origin, :]
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
    build_revision_profile(versions, forecast_origin; transform,
                            max_delay, min_support, mode, stat)
        -> Dict{Tuple{String,Int},Float64}

Empirical per-`(location, delay)` revision profile on the `transform`
scale, rebuilt PER SPLIT from only `versions` rows with `as_of <
forecast_origin` -- LEAK-FREE. For each `(location, origin_date)` group,
`settled` is the vintage with the latest `as_of` seen so far (i.e. "the
latest vintage known as of THIS split's forecast origin", not the
dataset's true final value -- an honest degradation for origin dates
close to `forecast_origin`, whose true settled value isn't knowable yet
either, not a leak). Every earlier vintage in the group within
`max_delay` weeks of `origin_date` contributes one observation, `mode =
:additive` giving `settled - vintage` or `mode = :multiplicative` giving
`settled / vintage` (skipped if the vintage is ~0 to avoid dividing by
it). `stat` (`:median` or `:mean`) summarises each `(location, delay)`
bin, dropped if it has fewer than `min_support` observations.
"""
function build_revision_profile(
    versions::DataFrame, forecast_origin::Date; transform::Symbol,
    max_delay::Int, min_support::Int, mode::Symbol, stat::Symbol,
)
    vf = versions[versions.as_of .< forecast_origin, :]
    raw = Dict{Tuple{String,Int},Vector{Float64}}()
    for g in groupby(vf, [:location, :origin_date])
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

Nudge `data.Y` in place wherever `0 <= data.delay[t, l] <= delay_cutoff`
and a matching `(location, delay)` entry exists in `profile` (see
[`build_revision_profile`](@ref)); entries with no matching profile key,
a delay outside `[0, delay_cutoff]`, or a `missing` observation are left
untouched. `mode = :additive` adds the correction; `mode =
:multiplicative` scales by it. Returns `data`.
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
