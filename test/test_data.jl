using CSV, DataFrames, Dates, Statistics
include(joinpath(@__DIR__, "..", "src", "core.jl"))
include(joinpath(@__DIR__, "..", "src", "data.jl"))

splits = training_splits(1)
last_split = splits[end]
data = build_model_data(last_split)

@assert size(data.Y) == (data.T, data.L)
@assert data.L == 11
@assert all(!isnan(v) for v in skipmissing(data.Y))
@assert all(d -> d in -1:data.Dmax, data.delay)
@assert all(w -> w in 1:data.W, data.woy)

# Splits are in ascending forecast-origin order.
@assert issorted([maximum(s.origin_date) for s in splits])

# `versions`-based delay overrides the recency approximation where
# available.
versions = load_series("flu_data_hhs_versions")
data_v = build_model_data(last_split; versions=versions)
@assert size(data_v.Y) == (data_v.T, data_v.L)
@assert all(d -> d in -1:data_v.Dmax, data_v.delay)

println("test_data.jl: all assertions passed")
