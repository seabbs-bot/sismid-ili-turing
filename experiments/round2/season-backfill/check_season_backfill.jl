# Prior predictive check for the Round 2 season-backfill candidate.
# PRIOR PREDICTIVE ONLY -- no NUTS/Pathfinder fit (shared box is busy;
# see docs/lessons.md #4 on not over-parallelising). Not part of the
# automated test suite; run directly with
#   julia --project=. experiments/round2/season-backfill/check_season_backfill.jl

using Random
using Dates
using Distributions
using Turing
using Statistics

const HERE = @__DIR__
include(joinpath(HERE, "..", "..", "..", "src", "core.jl"))
include(joinpath(HERE, "..", "..", "..", "src", "model.jl"))
include(joinpath(HERE, "model_season_backfill.jl"))
include(joinpath(HERE, "project_season_backfill.jl"))

Random.seed!(20260717)

# --- Small synthetic ModelData, spanning 2 seasons so S=2 is actually
# exercised (the whole point of this candidate). Dmax=12, matching the
# wider backfill window used by the other Round 2 backfill candidates
# (docs/eda/02-backfill.md: revisions settle by delay ~10-15 weeks). ---
const T = 60
const L = 3
const W = 33
const S = 2
const Dmax = 12

woy = [mod1(t, W) for t in 1:T]
season = [t <= 30 ? 1 : 2 for t in 1:T]
dates = Date(2015, 10, 3) .+ Day.(7 .* (0:(T - 1)))

delay = [min(T - t, Dmax) for t in 1:T, l in 1:L]
# A couple of the most recent, delayed cells are genuinely unreported.
delay[T, 2] = -1
delay[T, 3] = -1
delay[T - 1, 3] = -1

true_curve_pct = [3.0 + 2.0 * (0.5 + 0.5 * sin(2 * pi * w / W)) for w in woy]
Y = Matrix{Union{Missing,Float64}}(undef, T, L)
for l in 1:L, t in 1:T
    Y[t, l] = delay[t, l] == -1 ? missing :
        to_scale(max(true_curve_pct[t] + 0.1 * randn(), 0.01), :log)
end

d = ModelData(
    Y, delay, woy, season, dates, L, T, W, S, Dmax, :log, dates[end],
)

println("--- model_season_backfill dims ---")
println(model_dims(d))

model = model_season_backfill(d; transform=:log)

println("--- prior predictive (Prior() sampler) ---")
prior_chain = sample(model, Prior(), 200; progress=false)
@assert all(isfinite, Array(prior_chain)) "non-finite prior draws"

gq = model()
@assert size(gq.latent) == (T, L)
@assert size(gq.seasonal) == (T, L)
@assert size(gq.residual) == (T, L)
@assert size(gq.r_pop) == (Dmax + 1,)
@assert size(gq.r_season) == (Dmax + 1, S)
@assert size(gq.r) == (Dmax + 1, L, S)
@assert length(gq.mu_w) == W
@assert length(gq.phi) == L
@assert length(gq.sigma_ar) == L
@assert gq.transform == :log
println("NamedTuple shape checks passed")

# Confirm the season deviation actually moves the profile: with a
# nonzero sigma_r_season draw, r[:, l, 1] should differ from
# r[:, l, 2] for at least one location (the mechanism this candidate
# adds over base_model, which has no season dimension on `r` at all).
@assert any(gq.r[:, l, 1] != gq.r[:, l, 2] for l in 1:L) ||
        gq.sigma_r_season < 1e-8 "season dimension of r is inert"
println("season deviation on r is active (or sigma_r_season ~ 0 this draw)")

# Prior-predictive wILI% for observed cells, back-transformed: check
# it's finite, and that the TYPICAL (median, clamped at 0 as
# forecast_quantiles does downstream) draw is plausible. As in
# base_model / v5-backfill's checks, a heavy tail on rare draws once
# exponentiated back to the natural scale is expected, not a bug (the
# vague hierarchical priors on the AR residual / seasonal curve are
# unchanged from base_model); the clamped median is the fair
# "plausible range" check here.
prior_draws = [model() for _ in 1:100]
natural_vals = Float64[]
for g in prior_draws
    for l in 1:L, t in 1:T
        if !ismissing(d.Y[t, l])
            rev = g.r[delay[t, l] + 1, l, season[t]]
            push!(natural_vals, from_scale(g.latent[t, l] + rev, :log))
        end
    end
end
@assert all(isfinite, natural_vals) "non-finite prior-predictive wILI%"
lo, hi = extrema(natural_vals)
clamped = max.(natural_vals, 0.0)
med = quantile(clamped, 0.5)
println("prior-predictive wILI% range: ($lo, $hi), clamped median=$med")
@assert 0 <= med < 30 "prior median implausible (median = $med)"

println("--- project_season_backfill smoke check ---")
latent_fc = project_season_backfill(gq, d, 1:4)
@assert size(latent_fc) == (L, 4)
@assert all(isfinite, latent_fc)
println("project_season_backfill forecast (log/modelling scale):")
println(latent_fc)

println("check_season_backfill.jl passed")
