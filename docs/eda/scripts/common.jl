# Shared data-loading / season helpers for the EDA figure scripts.
# Validation + history only: season_year <= 2016 (see 01-series-overview.md).
using CSV, DataFrames, Dates, Statistics, StatsBase

const REPO_ROOT = joinpath(@__DIR__, "..", "..", "..")
const FIG_DIR = joinpath(@__DIR__, "..", "figures")

function season_year(dt::Date)
    y = year(dt)
    doy = dayofyear(dt)
    return doy >= 205 ? y : y - 1
end

"""Load the finalized series, add season_year/woy, filter to validation+history."""
function load_finalized()
    d = CSV.read(joinpath(REPO_ROOT, "data", "flu_data_hhs.csv"), DataFrame)
    d.origin_date = Date.(d.origin_date)
    d.season_year = season_year.(d.origin_date)
    d = filter(row -> row.season_year <= 2016, d)

    natl_dates = sort(unique(d.origin_date[d.location .== "US National"]))
    season_of = Dict(dt => season_year(dt) for dt in natl_dates)
    woy_map = Dict{Date,Int}()
    for sy in unique(values(season_of))
        dates_in_season = sort([dt for dt in natl_dates if season_of[dt] == sy])
        for (i, dt) in enumerate(dates_in_season)
            woy_map[dt] = i
        end
    end
    d.woy = [woy_map[dt] for dt in d.origin_date]
    return d
end

"""Load the reporting-version snapshots, add season_year/delay, filter cutoff."""
function load_versions()
    d = CSV.read(joinpath(REPO_ROOT, "data", "flu_data_hhs_versions.csv"), DataFrame)
    d.origin_date = Date.(d.origin_date)
    d.as_of = Date.(d.as_of)
    d.season_year = season_year.(d.origin_date)
    d = filter(row -> row.season_year <= 2016, d)
    d.delay = [round(Int, Dates.value(row.as_of - row.origin_date) / 7) for row in eachrow(d)]
    return d
end

# palette: fixed categorical order (dataviz skill default), used consistently
# across all EDA figures so a location/series keeps the same colour everywhere.
const PALETTE = [
    "#2a78d6", "#008300", "#e87ba4", "#eda100",
    "#1baf7a", "#eb6834", "#4a3aa7", "#e34948",
]
