#!/usr/bin/env julia
# seabbs_bot-round1 submission generator: fit the Round 1 WINNER on every
# vintage split of ALL FIVE hub seasons and write its hub quantile table.
#
# Candidate SELECTION happened on the validation seasons (1, 2) only
# (experiments/round1/_results + reports/round1.md). This script is the
# SUBMISSION step for the already-chosen winner: the hub scores all five
# seasons (2015-2019), and each origin is a per-week vintage fit using
# only the data available at that origin (build_model_data enforces the
# training-discipline cut), so forecasting the test seasons 3-5 here is
# not selection on test data -- it is producing the required coverage.
#
# Winner-agnostic: pass the winning candidate name; the model/project it
# maps to (below) is fit through the package's canonical
# `fit_and_forecast`, exactly as the round-1 screen scored it, only now
# over seasons 1..5 with `allow_test_season=true`.
#
# Usage:
#   julia --project=<repo> \
#       submissions/seabbs_bot-round1/generate_forecasts.jl \
#       <winner_name> <hub_path>
# e.g.
#   julia --project=. submissions/seabbs_bot-round1/generate_forecasts.jl \
#       nfidd-base ~/code/external/sismid-ili-sandbox-fork

const REPO = "/home/seabbs/code/seabbs/sismid-ili-turing"

# round1_run.jl pulls in SismidILITuring (build_model_data,
# training_splits, fit_and_forecast, write_submission, write_metadata,
# base_model/base_project) and the v1..v5 candidate models/projects.
include(joinpath(REPO, "experiments", "round1_run.jl"))
import SismidILITuring: observation_index
const _R2 = joinpath(REPO, "experiments", "round2")
include(joinpath(_R2, "severity", "model_severity.jl"))
include(joinpath(_R2, "severity", "project_severity.jl"))
include(joinpath(_R2, "season-backfill", "model_season_backfill.jl"))
include(joinpath(_R2, "season-backfill", "project_season_backfill.jl"))
include(joinpath(_R2, "ar-loc", "model_ar_loc.jl"))
include(joinpath(_R2, "ar-loc", "project_ar_loc.jl"))
include(joinpath(_R2, "var", "model_var.jl"))
include(joinpath(_R2, "var", "project_var.jl"))

using DataFrames
using Dates

ar_loc_build(d::ModelData; transform::Symbol=:fourthroot) =
    model_ar_loc(d; transform=transform, Pmax=10)

# name => (build_model, project, transform, ndraws, Dmax). Must match the
# exact configuration each candidate was SCORED with in Round 1
# (experiments/round1_run.jl PRIMARY_TRANSFORM=:fourthroot, ndraws=150,
# Dmax=12; base-log is the transform-axis check on :log).
const WINNERS = Dict(
    "nfidd-base" => (base_model, base_project, :fourthroot, 150, 12),
    "nfidd-base-log" => (base_model, base_project, :log, 150, 12),
    "nfidd-ar-high" => (model_v1, project_v1, :fourthroot, 150, 12),
    "nfidd-mvn-season" => (model_v2, project_v2, :fourthroot, 150, 12),
    "nfidd-diff" => (model_v3, project_v3, :fourthroot, 150, 12),
    "nfidd-tv-ar" => (model_v4, project_v4, :fourthroot, 150, 12),
    "nfidd-backfill" => (model_v5, project_v5, :fourthroot, 150, 12),
    "nfidd-severity" =>
        (model_severity, project_severity, :fourthroot, 150, 12),
    "nfidd-season-backfill" =>
        (model_season_backfill, project_season_backfill, :fourthroot, 150, 12),
    "nfidd-ar-loc" => (ar_loc_build, project_ar_loc, :fourthroot, 150, 12),
    "nfidd-var" => (model_var, project_var, :fourthroot, 150, 12),
)

# Hub model id for the submission. team_abbr uses an underscore because
# the hub's model-metadata schema forbids a hyphen in team_abbr
# (^[a-zA-Z0-9_+]+$); the id itself is "<team_abbr>-<model_abbr>".
const SUBMIT_ID = "seabbs_bot-round1"
const TEAM_ABBR = "seabbs_bot"
const MODEL_ABBR = "round1"

"""
    generate(winner, hub_path)

Fit `winner` over every split of seasons 1..5 (allow_test_season=true)
and write `model-output/seabbs_bot-round1/*.csv` +
`model-metadata/seabbs_bot-round1.yml` into `hub_path`. Every expected
origin must be produced (a submission cannot silently drop origins): a
split that throws is fatal here, unlike the screen's resilient scoring --
the hub needs full coverage.
"""
function generate(winner::AbstractString, hub_path::AbstractString)
    haskey(WINNERS, winner) || error("unknown winner: $winner")
    build_model, project, transform, ndraws, Dmax = WINNERS[winner]
    @info "seabbs_bot-round1 submission" winner transform ndraws Dmax hub_path

    tables = DataFrame[]
    expected = Date[]
    for season in 1:5
        splits = training_splits(season; allow_test_season=true)
        for split in splits
            origin = maximum(split.origin_date)
            push!(expected, origin)
            data = build_model_data(split; Dmax=Dmax, transform=transform,
                                    window_weeks=104)
            result = fit_and_forecast(build_model, data, SUBMIT_ID;
                                      project=project, ndraws=ndraws,
                                      transform=transform)
            push!(tables, result.forecast)
            @info "origin done" season origin nrow=nrow(result.forecast)
        end
    end
    submission = reduce(vcat, tables)

    produced = sort(unique(submission.origin_date))
    missing_dates = setdiff(sort(unique(expected)), produced)
    isempty(missing_dates) ||
        error("missing origins in submission: $missing_dates")

    write_submission(submission, hub_path)
    write_metadata(SUBMIT_ID, hub_path;
                   team_abbr=TEAM_ABBR, model_abbr=MODEL_ABBR)
    @info "wrote submission" n_origins=length(produced) n_rows=nrow(submission)
    return submission
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 ||
        error("usage: generate_forecasts.jl <winner_name> <hub_path>")
    generate(ARGS[1], ARGS[2])
end
