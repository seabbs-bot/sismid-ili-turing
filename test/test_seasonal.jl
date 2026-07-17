# Tests for src/seasonal.jl (canonical leak-free seasonal climatology and
# backfill/revision profile). Standalone: includes core.jl, data.jl, and
# seasonal.jl directly rather than loading the whole package.

using Test
using Dates
using DataFrames
using Statistics

include("../src/core.jl")
include("../src/data.jl")
include("../src/seasonal.jl")

function hist_row(location, origin_date, wili)
    (location = location, origin_date = origin_date, wili = wili)
end

@testset "build_seasonal_profile: leak-free (future rows ignored)" begin
    origin = Date(2016, 1, 2)
    before = [
        hist_row("US National", origin - Day(7 * k), 2.0 + 0.05 * (k % 52))
        for k in 1:200
    ]
    future = [
        hist_row("US National", origin + Day(7 * k), 999.0) for k in 1:10
    ]
    hist_before = DataFrame(before)
    hist_with_future = DataFrame(vcat(before, future))

    p1 = build_seasonal_profile(
        hist_before, origin; transform = :log, min_support = 3,
        smooth_window = 3,
    )
    p2 = build_seasonal_profile(
        hist_with_future, origin; transform = :log, min_support = 3,
        smooth_window = 3,
    )
    @test p1 == p2
end

@testset "build_seasonal_profile: shape" begin
    origin = Date(2016, 1, 2)
    rows = [
        hist_row("US National", origin - Day(7 * k), 2.0 + 0.05 * (k % 52))
        for k in 1:200
    ]
    profile = build_seasonal_profile(
        DataFrame(rows), origin; transform = :log, min_support = 3,
        smooth_window = 3,
    )
    # Re-centred to zero mean across the cycle.
    @test isapprox(mean(values(profile)), 0.0; atol = 1e-8)

    # A week-of-season bin with fewer than `min_support` observations
    # falls back to 0.0 rather than an unstable estimate from one or two
    # points.
    sparse_rows = [hist_row("US National", origin - Day(7), 5.0)]
    sparse_profile = build_seasonal_profile(
        DataFrame(sparse_rows), origin; transform = :log, min_support = 3,
        smooth_window = 1,
    )
    @test all(v -> v == 0.0, values(sparse_profile))
end

function version_row(location, origin_date, as_of, wili)
    (location = location, origin_date = origin_date, as_of = as_of,
     wili = wili)
end

@testset "build_revision_profile: leak-free (future vintages ignored)" begin
    origin = Date(2016, 1, 2)
    # One location/origin_date pair, revised across two `as_of` vintages
    # strictly before `origin`, plus a further "settled" vintage that
    # only becomes known AT `origin` -- must not be visible to a split
    # forecasting from `origin`.
    before = [
        version_row(
            "HHS Region 1", origin - Day(14), origin - Day(14), 2.0,
        ),
        version_row(
            "HHS Region 1", origin - Day(14), origin - Day(7), 3.0,
        ),
    ]
    future_settle = [
        version_row("HHS Region 1", origin - Day(14), origin, 10.0),
    ]
    v_before = DataFrame(before)
    v_with_future = DataFrame(vcat(before, future_settle))

    p1 = build_revision_profile(
        v_before, origin; transform = :log, max_delay = 8, min_support = 1,
        mode = :additive, stat = :median,
    )
    p2 = build_revision_profile(
        v_with_future, origin; transform = :log, max_delay = 8,
        min_support = 1, mode = :additive, stat = :median,
    )
    @test p1 == p2
    # settled (as_of = origin - 7d, the latest as_of strictly before
    # `origin`) is delay 1 week from origin_date; the earlier
    # origin - 14d vintage (delay 0) is corrected toward it.
    @test haskey(p1, ("HHS Region 1", 0))
end

@testset "build_revision_profile: additive vs multiplicative" begin
    origin = Date(2016, 1, 2)
    rows = [
        version_row("US National", origin - Day(14), origin - Day(14), 2.0),
        version_row("US National", origin - Day(14), origin - Day(7), 4.0),
    ]
    versions = DataFrame(rows)

    add = build_revision_profile(
        versions, origin; transform = :fourthroot, max_delay = 8,
        min_support = 1, mode = :additive, stat = :mean,
    )
    mult = build_revision_profile(
        versions, origin; transform = :fourthroot, max_delay = 8,
        min_support = 1, mode = :multiplicative, stat = :mean,
    )
    # row 2 (as_of = origin - 7d, the later vintage) is `settled`; row 1
    # (as_of = origin - 14d, same origin_date, so delay 0 weeks) is
    # corrected toward it.
    settled = to_scale(4.0, :fourthroot)
    vintage = to_scale(2.0, :fourthroot)
    @test isapprox(add[("US National", 0)], settled - vintage; atol = 1e-10)
    @test isapprox(
        mult[("US National", 0)], settled / vintage; atol = 1e-10,
    )
end

@testset "apply_backfill_correction!: additive and multiplicative" begin
    T, L, W, S, Dmax = 3, 2, 52, 1, 8
    dates = [Date(2016, 1, 2) - Day(7 * (T - t)) for t in 1:T]
    woy = fill(1, T)
    season = fill(1, T)
    Y = Matrix{Union{Missing,Float64}}(
        [1.0 2.0; 3.0 4.0; missing 5.0],
    )
    delay = [0 3; 1 20; -1 2]
    data = ModelData(
        Y, delay, woy, season, dates, L, T, W, S, Dmax, :log, dates[end],
    )
    profile = Dict(
        (LOCATIONS[1], 0) => 0.5,
        (LOCATIONS[2], 3) => 2.0,
        (LOCATIONS[2], 2) => 3.0,
    )

    additive = deepcopy(data)
    apply_backfill_correction!(additive, profile; mode = :additive,
        delay_cutoff = 8)
    @test additive.Y[1, 1] == 1.5           # delay 0, corrected
    @test additive.Y[1, 2] == 2.0 + 2.0     # delay 3, corrected
    @test additive.Y[2, 1] == 3.0           # delay 1, no profile entry
    @test additive.Y[2, 2] == 4.0           # delay 20 > cutoff, untouched
    @test ismissing(additive.Y[3, 1])       # missing stays missing
    @test additive.Y[3, 2] == 5.0 + 3.0     # delay 2, corrected

    multiplicative = deepcopy(data)
    apply_backfill_correction!(multiplicative, profile;
        mode = :multiplicative, delay_cutoff = 8)
    @test multiplicative.Y[1, 1] == 1.0 * 0.5
    @test multiplicative.Y[1, 2] == 2.0 * 2.0
end
