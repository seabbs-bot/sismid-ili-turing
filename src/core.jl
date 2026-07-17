# Core types, constants, and transforms shared across the package.
# This file is standalone: component files and their tests can
#   include("src/core.jl")
# to get the shared definitions without loading the whole package.

using Dates

"""23 hub quantile levels."""
const QUANTILE_LEVELS = [
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99,
]

"""Canonical location order used for matrix columns and output."""
const LOCATIONS = [
    "US National",
    "HHS Region 1", "HHS Region 2", "HHS Region 3", "HHS Region 4",
    "HHS Region 5", "HHS Region 6", "HHS Region 7", "HHS Region 8",
    "HHS Region 9", "HHS Region 10",
]

const TARGET = "ili perc"
const HORIZONS = 1:4

logit(p) = log(p / (1 - p))
invlogit(x) = 1 / (1 + exp(-x))

# Transforms between wILI **percentage** (natural scale, 0–100) and the
# modelling scale. `log` is the favoured default; `logit` and `fourthroot`
# are candidate alternatives. Each entry is (forward, inverse).
const EPS = 1e-4
const TRANSFORMS = Dict{Symbol,Tuple{Function,Function}}(
    :log => (w -> log(max(w, EPS)), x -> exp(x)),
    :log1p => (w -> log1p(max(w, 0.0)), x -> expm1(x)),
    :logit => (
        w -> logit(clamp(w / 100, 1e-6, 1 - 1e-6)),
        x -> 100 * invlogit(x),
    ),
    :fourthroot => (w -> max(w, 0.0)^0.25, x -> x^4),
)

to_scale(w, t::Symbol) = TRANSFORMS[t][1](w)
from_scale(x, t::Symbol) = TRANSFORMS[t][2](x)

"""
    ModelData

Model inputs for one forecast origin (one cross-validation split), in
time × location matrix form. See docs/contracts.md.
"""
struct ModelData
    Y::Matrix{Union{Missing,Float64}}  # (T×L) vintage obs on modelling scale
    delay::Matrix{Int}                 # (T×L) reporting delay weeks; -1 missing
    woy::Vector{Int}                   # (T) week-of-season index 1..W
    season::Vector{Int}                # (T) season index 1..S
    dates::Vector{Date}                # (T) reference dates, ascending
    L::Int
    T::Int
    W::Int
    S::Int
    Dmax::Int
    transform::Symbol                  # scale used to fill Y
    origin_date::Date                  # forecast origin (reference date)
end
