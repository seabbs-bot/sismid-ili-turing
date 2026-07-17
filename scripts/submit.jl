#!/usr/bin/env julia
# Reproducible submission driver: fit -> write -> validate -> print the
# exact commands to commit, push, and open the hub PR.
#
# Runs `produce_submission(...; write=true)` to lay down
# `model-output/<model_id>/*.csv` and `model-metadata/<model_id>.yml` in a
# hub clone, then runs the same checks as
# `scripts/validate_submission.jl` (in that script's own small,
# Turing-free environment, exactly as `task validate` does) against the
# result. By default it stops there and PRINTS the git/gh commands needed
# to commit, push the branch to the fork, and open the PR to reichlab
# without running them, so a run of this script never touches the
# network. Pass `--pr` to actually run those commands.
#
# Usage:
#   julia --project=. scripts/submit.jl [flags]
#
# Flags (all optional):
#   --model-id ID         hub model id, "<team_abbr>-<model_abbr>"
#                          (default: nfidd-turing)
#   --seasons LIST         comma-separated validation season ids
#                          (default: 1,2 -- the two validation seasons;
#                          never the held-out test seasons, see
#                          docs/contracts.md#experimental-integrity)
#   --hub-path PATH        hub clone to write into
#                          (default: our fork clone,
#                          ~/code/external/sismid-ili-sandbox-fork)
#   --ndraws N             Pathfinder draws per split (default: 200)
#   --dmax N               max reporting delay, Dmax (default: 12)
#   --window-weeks N       training history cap per split (default: 104)
#   --branch NAME          branch to use for the commit/push/PR commands
#                          (default: submit-<model-id>)
#   --pr                   actually run the commit/push/PR commands
#                          (default: off -- print-only)
#
# Examples:
#   # Local dress rehearsal: fit + validate, print (don't run) the PR
#   # commands, against our own fork clone.
#   julia --project=. scripts/submit.jl --model-id nfidd-turing
#
#   # The real thing, once ready: actually push and open the PR.
#   julia --project=. scripts/submit.jl --model-id nfidd-turing --pr
#
# Safety: `--pr` refuses to run unless the hub path's `origin` remote is
# a fork (its URL must not point at reichlab/sismid-ili-forecasting-sandbox
# directly) -- this script never pushes straight to the upstream hub.

using SismidILITuring
using DataFrames

const DEFAULT_HUB_PATH = joinpath(
    homedir(), "code", "external", "sismid-ili-sandbox-fork",
)
const HUB_REPO = "reichlab/sismid-ili-forecasting-sandbox"
const VALIDATE_PROJECT = joinpath(@__DIR__, "validate")
const VALIDATE_SCRIPT = joinpath(@__DIR__, "validate_submission.jl")

const FLAG_DEFAULTS = Dict(
    "model-id" => "nfidd-turing",
    "seasons" => "1,2",
    "hub-path" => DEFAULT_HUB_PATH,
    "ndraws" => "200",
    "dmax" => "12",
    "window-weeks" => "104",
    "branch" => "",
)

"""
    parse_args(args) -> (opts::Dict{String,String}, pr::Bool)

Parse `--flag value` pairs (see the header for the flag list) plus the
bare `--pr` switch. Unknown flags or a `--flag` missing its value raise
an error; every other flag falls back to `FLAG_DEFAULTS`.
"""
function parse_args(args)
    opts = copy(FLAG_DEFAULTS)
    pr = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--pr"
            pr = true
            i += 1
        elseif startswith(a, "--")
            key = a[3:end]
            haskey(opts, key) ||
                error("unknown flag --$key (see script header for the list)")
            i + 1 <= length(args) || error("--$key needs a value")
            opts[key] = args[i + 1]
            i += 2
        else
            error("unrecognised argument: '$a' (flags must start with --)")
        end
    end
    return opts, pr
end

"""
    fork_origin_url(hub_path) -> String

`origin`'s URL for the git repo at `hub_path`, used to guard `--pr`
against ever pushing straight to the upstream hub.
"""
function fork_origin_url(hub_path)
    return strip(read(
        Cmd(`git remote get-url origin`; dir=hub_path), String,
    ))
end

function run_local_validation(model_id, hub_path)
    println("\n=== local validation (scripts/validate_submission.jl) ===")
    println("(runs in its own environment, scripts/validate/, so this " *
            "never needs Turing/Mooncake/Pathfinder)")
    cmd = `julia --project=$(VALIDATE_PROJECT) $(VALIDATE_SCRIPT)
        $(model_id) $(hub_path)`
    proc = run(pipeline(ignorestatus(cmd); stdout=stdout, stderr=stderr))
    return proc.exitcode == 0
end

function print_submit_commands(model_id, hub_path, branch, origin_dates)
    files = ["model-output/$(model_id)/$(od)-$(model_id).csv"
             for od in origin_dates]
    metadata_file = "model-metadata/$(model_id).yml"
    println("\n=== commands to commit, push, and open the PR ===")
    println("cd $(hub_path)")
    println("git checkout -b $(branch)")
    println("git add $(metadata_file) $(join(files, " "))")
    println("git commit -m \"Add $(model_id) submission\"")
    println("git push -u origin $(branch)")
    println("gh pr create --repo $(HUB_REPO) --base main \\")
    println("  --head seabbs-bot:$(branch) \\")
    println("  --title \"$(model_id) submission\" \\")
    println("  --body \"<describe the model and validation seasons " *
             "scored here>\"")
end

function main()
    opts, do_pr = parse_args(ARGS)
    model_id = opts["model-id"]
    seasons = parse.(Int, split(opts["seasons"], ","))
    hub_path = opts["hub-path"]
    ndraws = parse(Int, opts["ndraws"])
    dmax = parse(Int, opts["dmax"])
    window_weeks = parse(Int, opts["window-weeks"])
    branch = isempty(opts["branch"]) ? "submit-$(model_id)" : opts["branch"]

    isdir(hub_path) ||
        error("hub_path not found: $hub_path (pass --hub-path, or clone " *
              "the fork first)")

    println("=== submit.jl: model_id=$(model_id) seasons=$(seasons) ===")
    println("hub_path=$(hub_path) ndraws=$(ndraws) Dmax=$(dmax) " *
             "window_weeks=$(window_weeks)")

    t0 = time()
    submission = produce_submission(;
        seasons=seasons, hub_path=hub_path, model_id=model_id,
        transform=:log1p, Dmax=dmax, ndraws=ndraws,
        window_weeks=window_weeks, write=true,
    )
    origin_dates = sort(unique(submission.origin_date))
    println("wrote $(nrow(submission)) rows across " *
            "$(length(origin_dates)) origin date(s) in " *
            "$(round(time() - t0; digits=1))s")

    ok = run_local_validation(model_id, hub_path)
    if !ok
        println("\nLocal validation FAILED -- fix the problems above " *
                "before committing. Not printing submit commands.")
        exit(1)
    end
    println("\nlocal validation PASSED.")

    if !do_pr
        print_submit_commands(model_id, hub_path, branch, origin_dates)
        println("\n(--pr not passed: stopping after local validation; " *
                "nothing committed, nothing pushed, no PR opened.)")
        return
    end

    origin_url = fork_origin_url(hub_path)
    if occursin(HUB_REPO, origin_url) && !occursin("seabbs-bot", origin_url)
        error("--pr refused: $(hub_path)'s origin ($(origin_url)) looks " *
              "like the upstream hub itself, not a fork. Point --hub-path " *
              "at your own fork clone before using --pr.")
    end

    println("\n--pr passed: committing, pushing, and opening the PR now.")
    files = ["model-output/$(model_id)/$(od)-$(model_id).csv"
             for od in origin_dates]
    metadata_file = "model-metadata/$(model_id).yml"
    cd(hub_path) do
        run(ignorestatus(`git checkout -b $(branch)`))
        run(`git add $(metadata_file) $(files)`)
        run(`git commit -m "Add $(model_id) submission"`)
        run(`git push -u origin $(branch)`)
        run(`gh pr create --repo $(HUB_REPO) --base main
            --head seabbs-bot:$(branch)
            --title $(model_id * " submission")
            --body "Submission from scripts/submit.jl."`)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
