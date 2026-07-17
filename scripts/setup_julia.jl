# Resolve the Julia environment for this project.
# Run from the repo root:  julia --project=. scripts/setup_julia.jl
#
# ScoringRules.jl is an EpiAware-org package and may not be in the General
# registry, so it is added by URL. Everything else is registered.

using Pkg

Pkg.activate(dirname(@__DIR__))

# Registered dependencies.
registered = [
    "Turing",
    "Mooncake",
    "Pathfinder",
    "Arrow",
    "CSV",
    "DataFrames",
    "Distributions",
    "LinearAlgebra",
    "Statistics",
    "Random",
    "Dates",
    "LogExpFunctions",
    "StatsBase",
]

for pkg in registered
    try
        Pkg.add(pkg)
    catch err
        @warn "could not add $pkg" err
    end
end

# ScoringRules.jl from the EpiAware org (unregistered).
try
    Pkg.add(url = "https://github.com/EpiAware/ScoringRules.jl")
catch err
    @warn "could not add ScoringRules.jl by URL" err
end

Pkg.instantiate()
Pkg.precompile()

@info "environment resolved"
