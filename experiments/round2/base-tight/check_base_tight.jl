# Prior predictive check for the round-2 candidate base-tight. PRIOR
# ONLY -- no MCMC/Pathfinder fit here, per this task's brief (round1 is
# mid-run on this box; a heavy fit here would contend with it). Run
# with:
#     julia --project=. experiments/round2/base-tight/check_base_tight.jl
#
# Draws from BOTH `base_model` (src/model.jl, unmodified) and
# `model_base_tight` (model_base_tight.jl) on the SAME synthetic
# `ModelData`, so the printed quantiles are a direct before/after
# comparison of the tightened hyperpriors' effect on the prior
# predictive back-transformed wILI% distribution, not just a check
# that the new candidate looks reasonable in isolation.

include("../../../src/core.jl")
include("../../../src/model.jl")
include("../../../src/forecast.jl")
include("model_base_tight.jl")
include("project_base_tight.jl")

using Dates
using Random
using Statistics

Random.seed!(20260717)

# --- Synthetic ModelData -------------------------------------------------
# Small enough to fit fast, but T well above 1 so the AR(1) recursion is
# not degenerate, and with both missing cells and non-zero delay to
# exercise the observation model's full behaviour. `:fourthroot` is the
# transform round1_run.jl actually screens candidates on
# (docs/lessons.md #7), and the one under which the reported implausible
# tail (`x -> x^4` back-transform) is most dramatic, so that is what
# this check uses too.

T, L, W, S, Dmax = 60, 3, 52, 2, 6
woy = [mod1(t, W) for t in 1:T]
season = [t <= T ÷ 2 ? 1 : 2 for t in 1:T]
dates = [Date(2015, 1, 3) + Day(7 * (t - 1)) for t in 1:T]

transform = :fourthroot
Y = Matrix{Union{Missing,Float64}}(undef, T, L)
delay = fill(-1, T, L)
for l in 1:L, t in 1:T
    wili_pct = 1.5 + 1.0 * sin(2pi * woy[t] / W) + 0.2 * randn()
    wili_pct = max(wili_pct, 0.05)
    if t > T - 3 && l == 2
        Y[t, l] = missing
    else
        Y[t, l] = to_scale(wili_pct, transform)
        delay[t, l] = min(rand(0:3), Dmax)
    end
end

d = ModelData(Y, delay, woy, season, dates, L, T, W, S, Dmax,
              transform, dates[end])

# --- Prior predictive: base_model (BEFORE) vs model_base_tight (AFTER) ---
# `latent` is a deterministic function of the sampled parameters only
# (it does not depend on d.Y), so calling the model directly draws one
# full forward (prior) sample each time.

function prior_wili(build_model, d, transform; n=400)
    all_wili = Float64[]
    for _ in 1:n
        draw = build_model(d; transform=transform)()
        append!(all_wili, from_scale.(vec(draw.latent), transform))
    end
    return all_wili
end

function report(label, wili)
    finite = filter(isfinite, wili)
    n_nonfinite = length(wili) - length(finite)
    q = quantile(finite, [0.5, 0.9, 0.95, 0.99])
    frac_plausible = count(x -> 0 <= x <= 15, finite) / length(finite)
    println(
        "$label: median=$(round(q[1]; digits=2)) ",
        "q90=$(round(q[2]; digits=2)) q95=$(round(q[3]; digits=2)) ",
        "q99=$(round(q[4]; digits=2)) max=$(round(maximum(finite); digits=2)) ",
        "frac_in[0,15]=$(round(frac_plausible; digits=3)) ",
        "n_nonfinite=$n_nonfinite",
    )
    return (q50=q[1], q90=q[2], q95=q[3], q99=q[4], max=maximum(finite),
            frac_plausible=frac_plausible)
end

println("--- BEFORE: base_model (unmodified) ---")
before = report("base_model", prior_wili(base_model, d, transform))

println("--- AFTER: model_base_tight (tightened priors) ---")
after = report("model_base_tight", prior_wili(model_base_tight, d, transform))

# The goal (team-lead brief): prior predictive wILI% stays mostly in
# [0, 15], with the tail (q99) not wildly beyond ~25%, a large
# improvement over base_model's reported q99 in the thousands of
# percent. Assert the improvement is real and the tail is now bounded
# to a plausible order of magnitude (not a hard <25 assert on q99,
# since a single synthetic-data seed can land either side of that by a
# few percent; the frac_plausible and order-of-magnitude checks below
# are the load-bearing ones).
@assert after.frac_plausible > 0.90 (
    "model_base_tight prior predictive bulk not in plausible range: " *
    "$(after.frac_plausible)"
)
@assert after.q99 < 100 (
    "model_base_tight prior predictive q99 still implausible: $(after.q99)"
)
@assert after.frac_plausible > before.frac_plausible (
    "model_base_tight did not improve on base_model's plausible fraction"
)
@assert after.q99 < before.q99 (
    "model_base_tight did not improve on base_model's q99 tail"
)
println(
    "\nImprovement: frac_in[0,15] $(round(before.frac_plausible; digits=3))",
    " -> $(round(after.frac_plausible; digits=3)); q99 ",
    "$(round(before.q99; digits=1))% -> $(round(after.q99; digits=1))%",
)

# --- model_base_tight residual/backfill sanity (mirrors check_v1.jl's
# pattern: residual should sit tightly around 0 regardless of the
# seasonal component) ------------------------------------------------

all_resid = Float64[]
for _ in 1:100
    draw = model_base_tight(d; transform=transform)()
    append!(all_resid, vec(draw.residual))
end
resid_med = median(all_resid)
println("model_base_tight residual median=$(round(resid_med; digits=3))")
@assert abs(resid_med) < 1.0 "residual median far from 0: $resid_med"

# --- project_base_tight shape check (no fit; uses one prior draw) -------

draw = model_base_tight(d; transform=transform)()
latent_fc = project_base_tight(draw, d, 1:4)
@assert size(latent_fc) == (L, 4) "project output size: $(size(latent_fc))"
println("project_base_tight output size: ", size(latent_fc),
        " (expect (", L, ", 4))")

println("\ncheck_base_tight.jl: all checks passed (no fit run)")
