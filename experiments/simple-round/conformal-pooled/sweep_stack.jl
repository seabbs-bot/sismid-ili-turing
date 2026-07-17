#!/usr/bin/env julia
# Stacked structural-lever sweep on the leak-free conformal-pooled model:
#   1. longer AR training window (window_weeks 104 -> 130/156/182/208)
#   2. damped-trend (Gardner) blend (phi x blend_max grid, horizon-ramped)
# One Julia process: data + vintage index are loaded once and
# build_forecast_table is reused per config, so the ~14 configs don't each
# pay Julia startup. Validation-only scoring, leak-free (every profile is
# rebuilt per origin inside build_forecast_table). Prints overall +
# by-horizon WIS, SD, cov50/cov90 for each config; the caller pastes the
# table into score.txt.
#
# Usage: julia --project=. experiments/simple-round/conformal-pooled/sweep_stack.jl

include(joinpath(@__DIR__, "generate.jl"))
using Printf

const DAMP_ALPHA = 0.5   # fixed Holt level-smoothing constant
const DAMP_BETA = 0.1    # fixed Holt trend-smoothing constant

function score_config(forecast, truth, val_origins)
    vf = forecast[in.(forecast.origin_date, Ref(val_origins)), :]
    scored = score_forecasts(vf, truth; scale=:natural)
    summ = wis_summary(scored)[1, :]
    byh = Dict{Int,Float64}()
    for g in groupby(scored, :horizon)
        byh[g.horizon[1]] = mean(g.wis)
    end
    cov50 = coverage(vf, truth, 0.25, 0.75)
    cov90 = coverage(vf, truth, 0.05, 0.95)
    return (mean_wis=summ.mean_wis, sd_wis=summ.sd_wis, byh=byh,
            cov50=cov50, cov90=cov90)
end

function fmt(label, r)
    h = r.byh
    r3(x) = round(x; digits=3)
    "$(rpad(label, 34)) wis=$(round(r.mean_wis; digits=4)) " *
    "sd=$(round(r.sd_wis; digits=4))  h1=$(r3(h[1])) h2=$(r3(h[2])) " *
    "h3=$(r3(h[3])) h4=$(r3(h[4]))  cov50=$(r3(r.cov50)) " *
    "cov90=$(r3(r.cov90))"
end

function main_sweep()
    versions_full = load_series("flu_data_hhs_versions")
    hist_all = load_series("flu_data_hhs")
    vidx = build_vintage_index(versions_full)
    truth = load_oracle(HUB_PATH)
    val_origins = Set{Date}()
    for season in VALIDATION_SEASONS
        for split in training_splits(season)
            push!(val_origins, maximum(split.origin_date))
        end
    end

    runcfg(; window_weeks, damp, width_scale=1.0) = score_config(
        build_forecast_table(
            (1, 2, 3, 4, 5), versions_full, hist_all, vidx;
            window_weeks=window_weeks, pool_weight=POOL_WEIGHT, damp=damp,
            width_scale=width_scale,
        ),
        truth, val_origins,
    )

    lines = String[]
    log(s) = (println(s); push!(lines, s))

    log("=== base (window=104, no damp, pool w=$(POOL_WEIGHT)) ===")
    base = runcfg(; window_weeks=104, damp=nothing)
    log(fmt("base", base))

    log("\n=== window sweep (no damp) ===")
    best_win = 104
    best_win_r = base
    for w in (130, 156, 182, 208)
        r = runcfg(; window_weeks=w, damp=nothing)
        log(fmt("window=$w", r))
        if r.mean_wis < best_win_r.mean_wis
            best_win = w
            best_win_r = r
        end
    end
    log("-> best window (no damp): $(best_win) " *
        "wis=$(round(best_win_r.mean_wis; digits=4))")

    log("\n=== damped-trend sweep at window=$(best_win) " *
        "(alpha=$(DAMP_ALPHA), beta=$(DAMP_BETA)) ===")
    best = best_win_r
    best_label = "window=$(best_win) no-damp"
    for phi in (0.85, 0.9, 0.95), bm in (0.1, 0.2, 0.3)
        damp = (phi=phi, blend_max=bm, alpha=DAMP_ALPHA, beta=DAMP_BETA)
        r = runcfg(; window_weeks=best_win, damp=damp)
        log(fmt("phi=$phi bm=$bm", r))
        if r.mean_wis < best.mean_wis
            best = r
            best_label = "window=$(best_win) phi=$phi bm=$bm"
        end
    end
    # Step 3 (quick check): does an interval WIDTH scale help on top of
    # split-conformal? Motivated by the slight OVER-coverage at window=208
    # (cov50>0.5, cov90>0.9). A global multiplicative scale on the
    # conformal offsets is the dominant per-location-width effect; if it
    # buys nothing, per-location scaling (covbias) is redundant too, since
    # empirical conformal calibration already reads the error quantiles.
    log("\n=== width-scale check at window=$(best_win), no damp ===")
    for ws in (0.85, 0.9, 0.95)
        r = runcfg(; window_weeks=best_win, damp=nothing, width_scale=ws)
        log(fmt("width_scale=$ws", r))
        if r.mean_wis < best.mean_wis
            best = r
            best_label = "window=$(best_win) width_scale=$ws"
        end
    end

    log("\n-> BEST stacked config: $(best_label) " *
        "wis=$(round(best.mean_wis; digits=4)) " *
        "cov50=$(round(best.cov50; digits=3)) " *
        "cov90=$(round(best.cov90; digits=3))")
    log("   vs conformal-pooled base 0.2870 and season 0.3004")

    open(joinpath(@__DIR__, "sweep_stack_results.txt"), "w") do io
        for l in lines
            println(io, l)
        end
    end
    println("\nwrote sweep_stack_results.txt")
end

main_sweep()
