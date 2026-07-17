# Hub I/O: write forecast quantile tables and metadata into a hubverse-style
# hub clone (model-output/, model-metadata/). See docs/contracts.md for the
# forecast table schema this operates on.
#
# Standalone: this file, alongside src/core.jl, can be `include`d without
# loading the whole package (needs DataFrames, CSV, Dates in scope).

using DataFrames
using CSV
using Dates

"""Hub `model-output` CSV column order (no `model_id`; implied by path)."""
const HUB_COLUMNS = [
    :origin_date, :location, :target, :horizon,
    :target_end_date, :output_type, :output_type_id, :value,
]

"""
    split_forecast(forecast_df) -> Vector{<:NamedTuple}

Split a combined forecast table (docs/contracts.md schema, possibly many
origin dates and model ids) into one group per `(model_id, origin_date)`
pair. Each element is `(; model_id, origin_date, df)`, where `df` has the
hub column order (`HUB_COLUMNS`) and no `model_id` column.
"""
function split_forecast(forecast_df::AbstractDataFrame)
    groups = NamedTuple[]
    for sub in groupby(forecast_df, [:model_id, :origin_date]; sort=true)
        model_id = sub.model_id[1]
        origin_date = sub.origin_date[1]
        df = select(DataFrame(sub), HUB_COLUMNS)
        push!(groups, (; model_id, origin_date, df))
    end
    return groups
end

"""
    write_submission(forecast_df, hub_path; designated=true, dry_run=false)
        -> Vector{<:NamedTuple}

Write one CSV per `origin_date` to
`<hub_path>/model-output/<model_id>/<origin_date>-<model_id>.csv`, in hub
column order (`HUB_COLUMNS`), without the `model_id` column.

`designated` is accepted for symmetry with `write_metadata` (a submission
and its designation are usually decided together) but is not itself used
to alter the CSV; see docs/contracts.md discussion. If `dry_run`, no
files or directories are created and the per-group `(; model_id,
origin_date, path, df)` tuples are returned without writing.
"""
function write_submission(
    forecast_df::AbstractDataFrame, hub_path::AbstractString;
    designated::Bool=true, dry_run::Bool=false,
)
    groups = split_forecast(forecast_df)
    results = NamedTuple[]
    for (; model_id, origin_date, df) in groups
        dir = joinpath(hub_path, "model-output", model_id)
        fname = string(origin_date) * "-" * model_id * ".csv"
        path = joinpath(dir, fname)
        if !dry_run
            mkpath(dir)
            CSV.write(path, df; quotestrings=true)
        end
        push!(results, (; model_id, origin_date, path, df))
    end
    return results
end

"""
    write_metadata(model_id, hub_path; team_abbr, model_abbr,
                   designated=true) -> String

Write `<hub_path>/model-metadata/<model_id>.yml` with the minimal
required fields (`team_abbr`, `model_abbr`, `designated_model`), matching
the style of the existing hub `.yml` files. Returns the path written.
"""
function write_metadata(
    model_id::AbstractString, hub_path::AbstractString;
    team_abbr::AbstractString, model_abbr::AbstractString,
    designated::Bool=true,
)
    dir = joinpath(hub_path, "model-metadata")
    mkpath(dir)
    path = joinpath(dir, model_id * ".yml")
    open(path, "w") do io
        println(io, "team_abbr: \"$(team_abbr)\"")
        println(io, "model_abbr: \"$(model_abbr)\"")
        println(io, "designated_model: $(designated)")
    end
    return path
end
