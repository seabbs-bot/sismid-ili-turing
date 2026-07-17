#!/usr/bin/env julia
# Export the course data objects to CSV for use in this project.
# Reads the .rda files from the SISMID forecasting course package and
# writes flat CSVs into data/. Run from the repo root:
#   julia --project=. scripts/export_data.jl

using CSV
using DataFrames
using RData
using CodecXz
using CodecBzip2

const COURSE = joinpath(
    homedir(), "code", "nfidd", "sismid-forecasting", "data",
)
const OUTDIR = joinpath(homedir(), "code", "seabbs", "sismid-ili-turing",
                         "data")

const OBJS = [
    "flu_data_hhs",
    "flu_data_hhs_versions",
    "flu_data_hhs_tscv_season1",
    "flu_data_hhs_tscv_season2",
    "flu_data_hhs_tscv_season3",
    "flu_data_hhs_tscv_season4",
    "flu_data_hhs_tscv_season5",
]

"""
    restore_r_names!(df) -> df

RData.jl replaces leading dots in R column names with underscores
(e.g. `.split` becomes `_split`). Rename any such column back so the
exported CSV matches the original R column names.
"""
function restore_r_names!(df::DataFrame)
    for name in names(df)
        if startswith(name, "_")
            rename!(df, name => "." * name[2:end])
        end
    end
    return df
end

function main()
    isdir(OUTDIR) || mkpath(OUTDIR)

    for obj in OBJS
        rda = joinpath(COURSE, obj * ".rda")
        if !isfile(rda)
            println("skip (missing): ", rda)
            continue
        end
        data = load(rda)
        df = restore_r_names!(data[obj])
        out = joinpath(OUTDIR, obj * ".csv")
        CSV.write(out, df)
        println(
            "wrote ", out, "  (", nrow(df), " rows, ", ncol(df), " cols)",
        )
    end

    println("done")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
