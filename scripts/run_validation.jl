#!/usr/bin/env julia
# Produce and score the base model's validation-season submission.
#
# Fits base_model on every cross-validation split of the validation
# seasons (1, 2), writes a hub-format submission into a local (scratch)
# hub clone, and scores it against the hub oracle on both the natural and
# log1p scales. This is the "demonstrably submittable, tested simple
# model" milestone: run it, then run scripts/validate_submission.jl on the
# same hub path to confirm the written files are hub-format valid.
#
# Usage:
#   julia --project=. scripts/run_validation.jl [hub_path] [model_id]
# Env knobs (all optional):
#   NDRAWS         Pathfinder draws per split      (default 200)
#   DMAX           max reporting delay             (default 12)
#   WINDOW_WEEKS   training history cap per split  (default 104)
#   SEASONS        comma-separated season ids      (default 1,2)

using SismidILITuring
using CSV
using DataFrames
using Dates
using Statistics

const HUB_PATH = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(@__DIR__, "..", "scratch-hub")
const MODEL_ID = length(ARGS) >= 2 ? ARGS[2] : "nfidd-turing"

getenv(k, default) = haskey(ENV, k) ? parse(Int, ENV[k]) : default
const NDRAWS = getenv("NDRAWS", 200)
const DMAX = getenv("DMAX", 12)
const WINDOW_WEEKS = getenv("WINDOW_WEEKS", 104)
const SEASONS = haskey(ENV, "SEASONS") ?
    parse.(Int, split(ENV["SEASONS"], ",")) : [1, 2]

"""Hub oracle (`target-data/oracle-output.csv`) as a scoring truth table."""
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

println("=== produce_submission: seasons=$(SEASONS) model=$(MODEL_ID) ===")
println("hub=$(HUB_PATH) ndraws=$(NDRAWS) Dmax=$(DMAX) " *
        "window_weeks=$(WINDOW_WEEKS)")
t0 = time()
submission = produce_submission(;
    seasons=SEASONS,
    hub_path=HUB_PATH,
    model_id=MODEL_ID,
    transform=:log1p,
    Dmax=DMAX,
    ndraws=NDRAWS,
    window_weeks=WINDOW_WEEKS,
    write=true,
)
println("produced $(nrow(submission)) rows across " *
        "$(length(unique(submission.origin_date))) origins in " *
        "$(round(time() - t0; digits=1))s")

truth = load_oracle(HUB_PATH)
scored_nat = score_forecasts(submission, truth; scale=:natural)
scored_log = score_forecasts(submission, truth; scale=:log)
summ_nat = wis_summary(scored_nat)
summ_log = wis_summary(scored_log)

println("\n=== validation WIS: $(MODEL_ID) ===")
println("scored tasks (natural): $(nrow(scored_nat))")
println("natural scale: mean_wis=$(round(summ_nat.mean_wis[1]; digits=4)) " *
        "sd_wis=$(round(summ_nat.sd_wis[1]; digits=4))")
println("log1p scale:   mean_wis=$(round(summ_log.mean_wis[1]; digits=4)) " *
        "sd_wis=$(round(summ_log.sd_wis[1]; digits=4))")
println("\ncomponent means (natural): " *
        "dispersion=$(round(summ_nat.mean_dispersion[1]; digits=4)) " *
        "over=$(round(summ_nat.mean_overprediction[1]; digits=4)) " *
        "under=$(round(summ_nat.mean_underprediction[1]; digits=4))")
