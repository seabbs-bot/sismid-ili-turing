# Loading raw CSVs and building `ModelData` for one cross-validation split.
# This file is standalone: it can be tested via
#   include("src/core.jl"); include("src/data.jl")
# but assumes `LOCATIONS`, `to_scale` and `ModelData` from core.jl are in
# scope (either via that include, or via the package module).

using CSV
using DataFrames
using Dates

const DATA_DIR = joinpath(@__DIR__, "..", "data")

"""
    load_series(name)::DataFrame

Read `data/<name>.csv` (the `.csv` extension is appended if not already
present). `origin_date` and, when present, `as_of` are parsed as `Date`;
`wili` is parsed as `Float64`. The tscv split files carry a `.split`
column; because `.` is not valid in a Julia identifier, its values must be
accessed as `df[!, ".split"]`, not `df.split`.
"""
function load_series(name::AbstractString)::DataFrame
    fname = endswith(name, ".csv") ? name : name * ".csv"
    df = CSV.read(joinpath(DATA_DIR, fname), DataFrame)
    if eltype(df.origin_date) !== Date
        df.origin_date = Date.(df.origin_date)
    end
    if hasproperty(df, :as_of) && eltype(df.as_of) !== Date
        df.as_of = Date.(df.as_of)
    end
    if eltype(df.wili) !== Float64
        df.wili = Float64.(df.wili)
    end
    return df
end

"""
    training_splits(season::Int)::Vector{DataFrame}

Split `DataFrame`s for `flu_data_hhs_tscv_season<season>.csv`, one per
forecast origin, ordered by ascending forecast origin (`max(origin_date)`
within the split). Splits are identified by the `.split` column.
"""
function training_splits(season::Int)::Vector{DataFrame}
    df = load_series("flu_data_hhs_tscv_season$(season)")
    groups = groupby(df, ".split")
    splits = [DataFrame(g) for g in groups]
    sort!(splits; by=s -> maximum(s.origin_date))
    return splits
end

"""
    season_year(date::Date) -> Int

Calendar year the influenza season containing `date` starts in. Seasons
are taken to start in October, so dates from October to December belong
to the season starting that year, and dates from January to September
belong to the season that started the previous year.
"""
season_year(date::Date)::Int = month(date) >= 10 ? year(date) : year(date) - 1

"""
    season_start(date::Date) -> Date

First Saturday on/after 1 October of the season `date` falls in (see
[`season_year`](@ref)). Reference dates in this data are Saturdays, so
this lines season boundaries up with the data's own weekly grid.
"""
function season_start(date::Date)::Date
    oct1 = Date(season_year(date), 10, 1)
    offset = mod(6 - dayofweek(oct1), 7)  # Saturday == 6 in `Dates`
    return oct1 + Day(offset)
end

"""
    week_of_season(date::Date) -> Int

1-based index of `date` within its influenza season (see
[`season_start`](@ref)).
"""
function week_of_season(date::Date)::Int
    return div(Dates.value(date - season_start(date)), 7) + 1
end

"""
    build_model_data(split_df; Dmax=6, window_weeks=104, transform=:log,
                      versions=nothing)::ModelData

Build a [`ModelData`](@ref) from one tscv split's long `DataFrame`.

The forecast origin is `maximum(split_df.origin_date)`. Training history
is capped to the most recent `window_weeks` distinct reference dates
present in the split (fewer if the split is shorter). `wili` is mapped to
the modelling `transform` scale with `to_scale`; entries with no
observation for a location/week in the window are `missing`.

# Reporting delay

The tscv splits do not carry an `as_of` per row, so by default `delay` is
approximated from recency within the split: for a row with reference date
`d`, `delay = min(Dmax, weeks(forecast_origin - d))`. This is exact for
the true reporting triangle when a split's vintage was taken at its
forecast origin (which is how these tscv splits are constructed): a row's
implicit `as_of` is then the forecast origin, so `as_of - d` in weeks is
precisely this recency term.

If a `versions` `DataFrame` (the `flu_data_hhs_versions.csv` schema:
`location, origin_date, as_of, wili`) is supplied, the true `as_of`-based
delay is used instead wherever available: for each `(location,
origin_date)` present in the window, the latest `as_of <=
forecast_origin` is looked up in `versions`, and `delay = min(Dmax,
weeks(as_of - origin_date))`. Rows without a match fall back to the
recency approximation above. Location/weeks with no observation get
`delay = -1`, matching the `missing` entry in `Y`.
"""
function build_model_data(
    split_df::DataFrame;
    Dmax::Int=6,
    window_weeks::Int=104,
    transform::Symbol=:log,
    versions::Union{Nothing,DataFrame}=nothing,
)::ModelData
    forecast_origin = maximum(split_df.origin_date)

    all_dates = sort(unique(split_df.origin_date))
    dates = length(all_dates) > window_weeks ?
        all_dates[(end - window_weeks + 1):end] : all_dates
    T = length(dates)
    date_idx = Dict(d => i for (i, d) in enumerate(dates))

    L = length(LOCATIONS)
    loc_idx = Dict(loc => i for (i, loc) in enumerate(LOCATIONS))

    Y = Matrix{Union{Missing,Float64}}(missing, T, L)
    delay = fill(-1, T, L)

    # Best (latest, not-in-the-future) as_of per (location, origin_date),
    # used to compute the true reporting delay when `versions` is given.
    as_of_lookup = Dict{Tuple{String,Date},Date}()
    if versions !== nothing
        vf = versions[versions.as_of .<= forecast_origin, :]
        for g in groupby(vf, [:location, :origin_date])
            as_of_lookup[(g.location[1], g.origin_date[1])] = maximum(g.as_of)
        end
    end

    for row in eachrow(split_df)
        haskey(date_idx, row.origin_date) || continue
        haskey(loc_idx, row.location) || continue
        t = date_idx[row.origin_date]
        l = loc_idx[row.location]
        Y[t, l] = to_scale(row.wili, transform)

        recency = div(Dates.value(forecast_origin - row.origin_date), 7)
        true_as_of = get(as_of_lookup, (row.location, row.origin_date), nothing)
        delay[t, l] = if true_as_of !== nothing
            min(Dmax, div(Dates.value(true_as_of - row.origin_date), 7))
        else
            min(Dmax, recency)
        end
    end

    woy = [week_of_season(d) for d in dates]
    W = maximum(woy)

    syears = [season_year(d) for d in dates]
    unique_years = sort(unique(syears))
    season = [findfirst(==(sy), unique_years) for sy in syears]
    S = length(unique_years)

    return ModelData(
        Y, delay, woy, season, dates, L, T, W, S, Dmax, transform,
        forecast_origin,
    )
end
