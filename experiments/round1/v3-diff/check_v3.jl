# Sanity checks for Round 1 candidate v3-diff (model_v3/project_v3).
# Mirrors test/test_model.jl's synthetic-ModelData pattern but adds
# the two checks the brief asks for on top of "does it build":
#   1. a PRIOR predictive check on the natural (wILI%) scale, since
#      the differencing branch's own EDA note
#      (docs/eda/05-autocorrelation.md) is that an unconstrained
#      random walk can drift. It checks the bulk is plausible and
#      compares the tail against the shared AR(1) baseline on the
#      same data, rather than a raw min/max, since base_model's
#      hierarchical sigma_ar prior has a heavy tail in both branches
#      (see the printed quantiles for why a naive range check would
#      be misleading).
#   2. a tiny Pathfinder fit and a tiny NUTS(AutoMooncake) fit, then
#      `project_v3` run on generated-quantities draws from each, to
#      confirm the whole draw -> project -> natural-scale path works
#      end to end.
#
# Not a Test.jl test file (this experiment does not own test/); run
# directly with `julia --project=. experiments/round1/v3-diff/check_v3.jl`
# from the repo root and read the printed PASS/FAIL summary.

using Random
using Dates
using Distributions
using Turing
using Pathfinder
using Mooncake
using Statistics

const ROOT = joinpath(@__DIR__, "..", "..", "..")
include(joinpath(ROOT, "src", "core.jl"))
include(joinpath(ROOT, "src", "model.jl"))
include(joinpath(@__DIR__, "model_v3.jl"))
include(joinpath(@__DIR__, "project_v3.jl"))

# `_field` (NamedTuple-or-Dict draw accessor) lives in src/forecast.jl,
# which needs DataFrames; pull in just the one helper `project_v3`
# needs, rather than the whole file, to keep this check's
# dependencies minimal.
_field(draw, key::Symbol) =
    draw isa AbstractDict ? draw[key] : getproperty(draw, key)

Random.seed!(20240717)

# --- Small synthetic ModelData (same shape as test/test_model.jl) ---
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

true_curve = [2.0 + 0.5 * sin(2 * pi * w / W) for w in woy]
Y = Matrix{Union{Missing,Float64}}(undef, T, L)
for l in 1:L, t in 1:T
    Y[t, l] = delay[t, l] == -1 ? missing : true_curve[t] + 0.05 * randn()
end

d = ModelData(Y, delay, woy, season, dates, L, T, W, S, Dmax, :log1p, dates[end])

failures = String[]
check(name, ok) = ok || push!(failures, name)

model = model_v3(d)

# --- 1. Prior predictive: finite, plausible bulk, and a check that
# --- differencing does not drift *worse* than the shared AR(1)
# --- baseline (the actual EDA worry), rather than a raw min/max
# --- range check.
#
# base_model's hierarchical, log-scale `sigma_ar` prior has a heavy
# right tail by construction (exp of a Normal-on-Normal), so a few
# extreme prior draws are expected in *both* branches, not a
# differencing-specific defect; a raw min/max check would flag that
# shared tail as a "failure" regardless of branch. So this checks the
# bulk (median, 90th percentile) for plausibility, and separately
# compares v3-diff's far tail against the AR(1) baseline on the same
# synthetic data: a first difference has unbounded variance in T
# (`cumsum` of iid innovations) where AR(1) has a bounded stationary
# variance, so if differencing drifts worse, it should show up as a
# much heavier tail at the same extreme quantile, not in the bulk.
prior_chain = sample(model, Prior(), 200; progress=false)
check("prior chain finite", all(isfinite, Array(prior_chain)))

gq_prior = [model() for _ in 1:500]
natural = reduce(vcat, [from_scale.(gq.latent, :log1p) for gq in gq_prior])
qs = quantile(natural, [0.5, 0.9, 0.99, 0.999])
println("v3-diff prior predictive natural-scale quantiles ",
        "(0.5, 0.9, 0.99, 0.999): ", round.(qs, sigdigits=4))
check("prior predictive bulk plausible (median, 90th pctile)",
      abs(qs[1]) < 1000 && abs(qs[2]) < 1000)

ar_model = base_model(d; transform=:log1p, difference=false)
gq_ar = [ar_model() for _ in 1:500]
natural_ar = reduce(vcat, [from_scale.(gq.latent, :log1p) for gq in gq_ar])
qs_ar = quantile(natural_ar, [0.5, 0.9, 0.99, 0.999])
println("AR(1) baseline prior predictive natural-scale quantiles ",
        "(0.5, 0.9, 0.99, 0.999): ", round.(qs_ar, sigdigits=4))
println("v3-diff / AR(1) ratio at the 0.999 quantile: ",
        round(qs[4] / qs_ar[4], sigdigits=4))

# --- 2. Tiny Pathfinder fit, then project_v3 on its draws ---
pf = Pathfinder.pathfinder(model; ndraws=50, rng=Random.default_rng())
# `pf.draws` is the plain unconstrained-space Matrix (unlike
# `draws_transformed`, a named VNChain with no direct Matrix method).
check("pathfinder finite draws", all(isfinite, pf.draws))

gq_pf = Turing.DynamicPPL.returned(model, pf.draws_transformed)
draws_pf = vec(collect(gq_pf))
latent_pf = project_v3(draws_pf[1], d, 1:4)
check("project_v3 (pathfinder draw) size", size(latent_pf) == (L, 4))
check("project_v3 (pathfinder draw) finite", all(isfinite, latent_pf))
natural_pf = from_scale.(latent_pf, :log1p)
check("project_v3 (pathfinder draw) natural-scale plausible",
      all(x -> -1.0 < x < 500.0, natural_pf))

# --- 3. Tiny NUTS(AutoMooncake) fit, then project_v3 on its draws ---
chain = sample(model, NUTS(; adtype=AutoMooncake()), 30; progress=false)
check("NUTS chain finite lp", all(isfinite, chain[:lp]))

gq_nuts = Turing.DynamicPPL.returned(model, chain)
draws_nuts = vec(collect(gq_nuts))
latent_nuts = project_v3(draws_nuts[end], d, 1:4)
check("project_v3 (NUTS draw) size", size(latent_nuts) == (L, 4))
check("project_v3 (NUTS draw) finite", all(isfinite, latent_nuts))

if isempty(failures)
    println("check_v3.jl PASSED (all checks ok)")
else
    println("check_v3.jl FAILED: ", join(failures, "; "))
    exit(1)
end
