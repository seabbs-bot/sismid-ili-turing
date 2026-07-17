# Smoke test: confirm the environment loads and the core pieces work.
# Run from the repo root:  julia --project=. scripts/smoke_test.jl

using Pkg
Pkg.activate(dirname(@__DIR__))

@info "loading packages"
using Turing
using Mooncake
using Pathfinder
using ScoringRules
using CSV, DataFrames
using Statistics

@info "packages loaded"

# Data loads.
datadir = joinpath(dirname(@__DIR__), "data")
flu = CSV.read(joinpath(datadir, "flu_data_hhs.csv"), DataFrame)
@info "flu_data_hhs" nrow = nrow(flu) ncol = ncol(flu) cols = names(flu)
@info "locations" locs = unique(flu[!, :location])

# ScoringRules has interval / quantile scores available.
@info "ScoringRules exports present" interval_score = isdefined(ScoringRules, :interval_score) quantile_score =
    isdefined(ScoringRules, :quantile_score)

# Tiny Turing model with the Mooncake AD backend, sampled a few steps.
@model function demo(y)
    m ~ Normal(0, 1)
    s ~ truncated(Normal(0, 1); lower = 0)
    y .~ Normal(m, s)
    return nothing
end
y = randn(50) .+ 2.0
chain = sample(demo(y), NUTS(; adtype = AutoMooncake()), 50; progress = false)
@info "sampled demo model with Mooncake" mean_m = mean(chain[:m])

@info "smoke test passed"
