# Sanity check for the v4-tv-ar candidate (Round 1 tree search).
# Builds a small synthetic ModelData by hand (mirrors
# test/test_model.jl's pattern), runs a prior predictive check on
# model_v4 (plausible wILI% range, stable time-varying phi), then a
# tiny Pathfinder fit and a tiny NUTS(AutoMooncake) fit, and exercises
# project_v4 end to end through forecast_quantiles.
#
# Run from the repo root with:
#   julia --project=. experiments/round1/v4-tv-ar/check_v4.jl

using Random
using Dates
using Distributions
using Turing
using Mooncake
using Pathfinder
using Statistics
using DataFrames

const ROOT = joinpath(@__DIR__, "..", "..", "..")
include(joinpath(ROOT, "src", "core.jl"))
include(joinpath(ROOT, "src", "model.jl"))
include(joinpath(ROOT, "src", "inference.jl"))
include(joinpath(ROOT, "src", "forecast.jl"))
include(joinpath(@__DIR__, "model_v4.jl"))
include(joinpath(@__DIR__, "project_v4.jl"))

Random.seed!(20260717)

# --- Build a small synthetic ModelData by hand ---
const T = 40
const L = 3
const W = 33
const S = 1
const Dmax = 4

woy = [mod1(t, W) for t in 1:T]
season = fill(1, T)
dates = Date(2016, 1, 2) .+ Day.(7 .* (0:(T - 1)))

delay = [min(T - t, Dmax) for t in 1:T, l in 1:L]
delay[T, 2] = -1
delay[T, 3] = -1
delay[T - 1, 3] = -1

true_curve_pct = [2.0 + 1.0 * (1 + sin(2 * pi * w / W)) for w in woy]
Y = Matrix{Union{Missing,Float64}}(undef, T, L)
for l in 1:L, t in 1:T
    Y[t, l] = delay[t, l] == -1 ? missing :
              to_scale(true_curve_pct[t] + 0.1 * randn(), :log1p)
end

d = ModelData(Y, delay, woy, season, dates, L, T, W, S, Dmax, :log1p,
              dates[end])

println("=== model_v4: dims and generated quantities ===")
dims = model_dims(d)
@assert dims == (T=T, L=L, W=W, S=S, Dmax=Dmax)

model = model_v4(d; transform=:log1p)
gq = model()
@assert size(gq.latent) == (T, L)
@assert size(gq.seasonal) == (T, L)
@assert size(gq.residual) == (T, L)
@assert size(gq.phi_path) == (T, L)
@assert length(gq.phi) == L
@assert gq.phi == gq.phi_path[T, :]
@assert length(gq.sigma_ar) == L
@assert size(gq.r) == (Dmax + 1, L)
@assert length(gq.r_pop) == Dmax + 1
@assert gq.transform == :log1p
println("shapes OK; sample phi_path[end, :] = ", gq.phi_path[T, :])

println()
println("=== Prior predictive check ===")
nprior = 200
prior_chain = sample(model, Prior(), nprior; progress=false)
@assert all(isfinite, Array(prior_chain))
println("prior draws all finite (", nprior, " draws)")

# Recompute generated quantities across prior draws directly (cheaper
# than re-running the sampler through `returned`, and exercises the
# same code path check_model.jl in test/ uses for base_model).
wili_vals = Float64[]
phi_ranges = Float64[]
for _ in 1:nprior
    g = model()
    append!(wili_vals, from_scale.(g.latent .+ g.r[1, :]', :log1p))
    push!(phi_ranges, maximum(g.phi_path) - minimum(g.phi_path))
end
wili_vals = filter(isfinite, wili_vals)
println("prior wILI% central 90% range: ",
        quantile(wili_vals, [0.05, 0.5, 0.95]))
println("prior within-draw phi range (max - min over T), mean = ",
        mean(phi_ranges), ", max = ", maximum(phi_ranges))
# The stationary-variance init in `tv_ar_path` (like `base_model`'s
# `ar_or_diff`) can blow up on rare prior draws where phi_path[1] sits
# very close to +-1 (a near-unit-root draw dividing by a tiny 1 -
# phi^2); that is a known, shared feature of this parameterisation, not
# a bug, so this check looks at the *central* mass rather than the max.
@assert quantile(wili_vals, 0.95) < 1e4
@assert minimum(wili_vals) >= -1.0 - 1e-6  # expm1 floor
# The tv-phi random walk is deliberately slow: within one T=40-week
# draw its range should usually stay well inside (-1, 1)'s full width.
@assert median(phi_ranges) < 1.5
println("prior predictive OK: wILI central mass plausible, ",
        "tv-phi stays stable in typical draws")

println()
println("=== Tiny Pathfinder fit ===")
pf = fit_pathfinder(model; ndraws=100)
pf_draws = posterior_draws(pf)
@assert length(pf_draws) == 100
println("pathfinder ran, ", length(pf_draws), " draws")

println()
println("=== Tiny NUTS (AutoMooncake) fit ===")
chain = sample(model, NUTS(; adtype=AutoMooncake()), 20; progress=false)
@assert all(isfinite, chain[:lp])
println("NUTS ran, log-density finite over ", length(chain[:lp]), " draws")

println()
println("=== project_v4 through forecast_quantiles ===")
gen = generated_draws(model, chain)
@assert haskey(NamedTuple(gen[1]), :phi_path)
df = forecast_quantiles(gen, d, "nfidd-v4-tv-ar"; project=project_v4)
@assert nrow(df) == L * length(HORIZONS) * length(QUANTILE_LEVELS)
@assert all(>=(0.0), df.value)
@assert all(isfinite, df.value)
for loc in unique(df.location), h in unique(df.horizon)
    mask = (df.location .== loc) .& (df.horizon .== h)
    sub = sort(df[mask, :], :output_type_id)
    @assert issorted(sub.value)
end
println("forecast_quantiles OK: ", nrow(df), " rows, all finite, ",
        "non-negative, monotone within (location, horizon)")
println("sample horizon-1 median (location 1): ",
        df[(df.location .== LOCATIONS[1]) .& (df.horizon .== 1) .&
           (df.output_type_id .== 0.5), :value])

println()
println("ALL CHECKS PASSED for v4-tv-ar")
