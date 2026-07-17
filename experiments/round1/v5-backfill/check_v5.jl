# Prior predictive + tiny-fit check for the v5-backfill Round 1
# candidate. Not part of the automated test suite; run directly with
#   julia --project=. experiments/round1/v5-backfill/check_v5.jl

using Random
using Dates
using Distributions
using Turing
using Mooncake
using Pathfinder
using Statistics

const HERE = @__DIR__
include(joinpath(HERE, "..", "..", "..", "src", "core.jl"))
include(joinpath(HERE, "..", "..", "..", "src", "model.jl"))
include(joinpath(HERE, "model_v5.jl"))
include(joinpath(HERE, "project_v5.jl"))

Random.seed!(20260717)

# --- Small synthetic ModelData, Dmax=12 (this variant's wider window) ---
const T = 60
const L = 3
const W = 33
const S = 1
const Dmax = 12

woy = [mod1(t, W) for t in 1:T]
season = fill(1, T)
dates = Date(2016, 1, 2) .+ Day.(7 .* (0:(T - 1)))

delay = [min(T - t, Dmax) for t in 1:T, l in 1:L]
# A couple of the most recent, delayed cells are genuinely unreported.
delay[T, 2] = -1
delay[T, 3] = -1
delay[T - 1, 3] = -1

true_curve_pct = [3.0 + 2.0 * (0.5 + 0.5 * sin(2 * pi * w / W)) for w in woy]
Y = Matrix{Union{Missing,Float64}}(undef, T, L)
for l in 1:L, t in 1:T
    Y[t, l] = delay[t, l] == -1 ? missing :
        to_scale(max(true_curve_pct[t] + 0.1 * randn(), 0.01), :log1p)
end

d = ModelData(
    Y, delay, woy, season, dates, L, T, W, S, Dmax, :log1p, dates[end],
)

println("--- model_v5 dims ---")
println(model_dims(d))

model = model_v5(d; transform=:log1p)

println("--- prior predictive (Prior() sampler) ---")
prior_chain = sample(model, Prior(), 200; progress=false)
@assert all(isfinite, Array(prior_chain)) "non-finite prior draws"

gq = model()
@assert size(gq.latent) == (T, L)
@assert size(gq.seasonal) == (T, L)
@assert size(gq.residual) == (T, L)
@assert size(gq.r_pop) == (Dmax + 1, 2)
@assert size(gq.r) == (Dmax + 1, L, 2)
@assert length(gq.mu_w) == W
@assert length(gq.phi) == L
@assert length(gq.sigma_ar) == L
@assert gq.transform == :log1p
println("NamedTuple shape checks passed")

# Prior-predictive wILI% for observed cells, back-transformed: check
# it's finite, and that the TYPICAL (median, clamped at 0 as
# forecast_quantiles does downstream) draw is plausible. Two things are
# expected, not bugs, and both are inherited unchanged from base_model:
# (1) a heavy tail once exponentiated back to the natural scale on rare
# draws (verified: base_model itself shows the same tail blow-up on
# identical synthetic data, from the vague hierarchical priors on the
# AR residual / seasonal curve); (2) mu0 ~ Normal(0, 2) is centred at
# 0, i.e. natural wILI% = 0 on the log1p scale, so roughly half of raw
# prior draws are negative before clamping. Clamping first, as
# forecast_quantiles does, is the fair "plausible range" check here.
prior_draws = [model() for _ in 1:100]
natural_vals = Float64[]
for g in prior_draws
    for l in 1:L, t in 1:T
        if !ismissing(d.Y[t, l])
            rev = woy[t] <= cld(W, 2) ?
                g.r[delay[t, l] + 1, l, 1] : g.r[delay[t, l] + 1, l, 2]
            push!(natural_vals, from_scale(g.latent[t, l] + rev, :log1p))
        end
    end
end
@assert all(isfinite, natural_vals) "non-finite prior-predictive wILI%"
lo, hi = extrema(natural_vals)
clamped = max.(natural_vals, 0.0)
med = quantile(clamped, 0.5)
println("prior-predictive wILI% range: ($lo, $hi), clamped median=$med")
@assert 0 <= med < 30 "prior median implausible (median = $med)"

println("--- tiny NUTS fit (Mooncake AD) ---")
chain = sample(model, NUTS(; adtype=AutoMooncake()), 30; progress=false)
@assert all(isfinite, chain[:lp]) "non-finite NUTS log-density"
println("NUTS ok, lp range: ", extrema(chain[:lp]))

println("--- tiny Pathfinder fit ---")
pf = Pathfinder.pathfinder(model; ndraws=50)
@assert all(isfinite, Array(pf.draws_transformed)) "non-finite Pathfinder draws"
println("Pathfinder ok")

println("--- project_v5 smoke check ---")
latent_fc = project_v5(gq, d, 1:4)
@assert size(latent_fc) == (L, 4)
@assert all(isfinite, latent_fc)
println("project_v5 forecast (log1p/modelling scale):")
println(latent_fc)

println("check_v5.jl passed")
