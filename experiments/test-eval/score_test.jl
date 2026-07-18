#!/usr/bin/env julia
# score_test.jl -- score the five TEST-SEASON (3, 4, 5) forecast tables
# built by gen_ar6.jl / gen_ar6bf.jl / gen_season.jl / gen_seasstack.jl /
# gen_conformal.jl against the hub oracle, plus the hub's own hist-avg
# baseline for external context, and print every table needed for
# reports/test-evaluation.md (overall, by horizon, by season, by
# location).
#
# This is a REPORTING script: model selection is locked (validation
# seasons 1, 2 only, see docs/brief.md/docs/contracts.md); nothing here
# tunes or selects on the test seasons, it only scores the already-fixed
# designs.
#
# Usage: julia --project=<sismid-ili-turing repo> score_test.jl

using CSV
using DataFrames
using Dates
using Statistics

const PKG_DIR = "/home/seabbs/code/seabbs/sismid-ili-turing"
const OUT_DIR = joinpath(PKG_DIR, "experiments", "test-eval", "out")
const HUB_PATH = joinpath(PKG_DIR, "scratch-hub")
const EXTERNAL_HUB = "/home/seabbs/code/external/sismid-ili-forecasting-sandbox"

include(joinpath(PKG_DIR, "src", "core.jl"))
include(joinpath(PKG_DIR, "src", "data.jl"))
include(joinpath(PKG_DIR, "src", "scoring.jl"))

"""Hub oracle (`target-data/oracle-output.csv`) as a scoring truth
table -- identical in spirit to every generate.jl's function of the
same name."""
function load_oracle(hub_path)
    path = joinpath(hub_path, "target-data", "oracle-output.csv")
    oracle = CSV.read(path, DataFrame)
    truth = DataFrame(
        location=String.(oracle.location),
        target_end_date=Date.(oracle.target_end_date),
        value=Float64.(oracle.oracle_value),
    )
    return dropmissing(truth)
end

"""Empirical coverage of the nominal `level` central interval (e.g.
`level=0.5` -> [0.25, 0.75])."""
function coverage(forecast::DataFrame, truth::DataFrame, level::Float64)
    a = (1 - level) / 2
    task_cols = [:location, :origin_date, :horizon, :target_end_date]
    lo = forecast[isapprox.(forecast.output_type_id, a; atol=1e-6), :]
    hi = forecast[isapprox.(forecast.output_type_id, 1 - a; atol=1e-6), :]
    lo_r = rename(lo[:, vcat(task_cols, [:value])], :value => :lo)
    hi_r = rename(hi[:, vcat(task_cols, [:value])], :value => :hi)
    joined = innerjoin(lo_r, hi_r, on=task_cols)
    joined = innerjoin(joined, truth, on=[:location, :target_end_date])
    return mean(joined.lo .<= joined.value .<= joined.hi)
end

"""Season number (1-5) from an origin_date, via `season_year`."""
season_num(d::Date) = season_year(d) - 2014

function load_model(path::String, model_id::String)
    df = CSV.read(path, DataFrame)
    df.origin_date = Date.(df.origin_date)
    df.target_end_date = Date.(df.target_end_date)
    if !hasproperty(df, :model_id)
        df.model_id = fill(model_id, nrow(df))
    end
    return df
end

"""Load every per-origin hist-avg CSV from the external hub clone and
concatenate into one forecast table (hub column order, model_id added).
Not every TEST-season origin has a hist-avg file (see report caveats);
only origins present are scored, for external reference only."""
function load_hist_avg(hub_path::String)
    dir = joinpath(hub_path, "model-output", "hist-avg")
    files = filter(f -> endswith(f, ".csv"), readdir(dir))
    dfs = [CSV.read(joinpath(dir, f), DataFrame) for f in files]
    df = vcat(dfs...)
    df.origin_date = Date.(df.origin_date)
    df.target_end_date = Date.(df.target_end_date)
    df.model_id = fill("hist-avg", nrow(df))
    return df
end

function summarise(model_id, forecast, truth)
    scored = score_forecasts(forecast, truth; scale=:natural)
    summ = wis_summary(scored)[1, :]
    cov50 = coverage(forecast, truth, 0.5)
    cov90 = coverage(forecast, truth, 0.9)

    scored.season = season_num.(scored.origin_date)

    by_h = combine(groupby(scored, :horizon),
        :wis => mean => :mean_wis, nrow => :n)
    sort!(by_h, :horizon)

    by_season = combine(groupby(scored, :season),
        :wis => mean => :mean_wis, :wis => std => :sd_wis, nrow => :n)
    sort!(by_season, :season)

    by_loc = combine(groupby(scored, :location),
        :wis => mean => :mean_wis, nrow => :n)
    sort!(by_loc, :mean_wis)

    return (
        model_id=model_id, mean_wis=summ.mean_wis, sd_wis=summ.sd_wis,
        n_tasks=summ.n_tasks, cov50=cov50, cov90=cov90,
        by_horizon=by_h, by_season=by_season, by_location=by_loc,
    )
end

function print_summary(r)
    println("="^77)
    println(r.model_id)
    println("  mean_wis=$(round(r.mean_wis; digits=4)) " *
            "sd_wis=$(round(r.sd_wis; digits=4)) n_tasks=$(r.n_tasks)")
    println("  coverage 50% nominal -> $(round(r.cov50; digits=3))")
    println("  coverage 90% nominal -> $(round(r.cov90; digits=3))")
    println("  by horizon:")
    for row in eachrow(r.by_horizon)
        println("    h=$(row.horizon): $(round(row.mean_wis; digits=4)) " *
                "(n=$(row.n))")
    end
    println("  by season:")
    for row in eachrow(r.by_season)
        println("    season $(row.season): mean_wis=" *
                "$(round(row.mean_wis; digits=4)) sd_wis=" *
                "$(round(row.sd_wis; digits=4)) (n=$(row.n))")
    end
    println("  by location:")
    for row in eachrow(r.by_location)
        println("    $(rpad(row.location, 16)) " *
                "$(round(row.mean_wis; digits=4)) (n=$(row.n))")
    end
end

function main()
    truth = load_oracle(HUB_PATH)

    models = [
        ("nfidd-ar6", joinpath(OUT_DIR, "nfidd-ar6.csv")),
        ("nfidd-ar6bf", joinpath(OUT_DIR, "nfidd-ar6bf.csv")),
        ("seabbs_bot-season", joinpath(OUT_DIR, "seabbs_bot-season.csv")),
        ("seabbs_bot-seasstack", joinpath(OUT_DIR, "seabbs_bot-seasstack.csv")),
        ("conformal-pooled", joinpath(OUT_DIR, "conformal-pooled.csv")),
    ]

    results = NamedTuple[]
    for (model_id, path) in models
        forecast = load_model(path, model_id)
        r = summarise(model_id, forecast, truth)
        push!(results, r)
        print_summary(r)
    end

    # Hub's own hist-avg baseline, external reference only (partial
    # origin-date coverage -- see report caveats).
    hist_avg = load_hist_avg(EXTERNAL_HUB)
    test_origins = Set(vcat([load_model(p, m).origin_date
                              for (m, p) in models]...))
    hist_avg_test = hist_avg[in.(hist_avg.origin_date, Ref(test_origins)), :]
    n_ha_origins = length(unique(hist_avg_test.origin_date))
    println("\nhist-avg: $(n_ha_origins) / $(length(test_origins)) " *
            "TEST-season origins present")
    r_ha = summarise("hist-avg", hist_avg_test, truth)
    push!(results, r_ha)
    print_summary(r_ha)

    println("\n" * "="^77)
    println("RANKED (mean WIS, lower is better):")
    sorted = sort(results; by=r -> r.mean_wis)
    for r in sorted
        println("  $(rpad(r.model_id, 24)) mean_wis=" *
                "$(round(r.mean_wis; digits=4)) sd_wis=" *
                "$(round(r.sd_wis; digits=4)) cov50=" *
                "$(round(r.cov50; digits=3)) cov90=" *
                "$(round(r.cov90; digits=3))")
    end

    return results
end

results = main()
