#!/usr/bin/env julia
# Local format checker for a prepared hub submission, so problems are
# caught before pushing. This mirrors the hub's own rules from
# hub-config/tasks.json and hub-config/model-metadata-schema.json in the
# hub clone. It is a fast pre-flight, not a replacement for the
# authoritative check: the real hubValidations run happens in the hub's
# PR CI once a pull request is opened.
#
# Run with its own (small, Turing-free) environment so this never needs
# to load Turing/Mooncake/Pathfinder:
#   julia --project=scripts/validate scripts/validate_submission.jl \
#       <model_id> [hub_path]
#   MODEL_ID=nfidd-seasarpp julia --project=scripts/validate \
#       scripts/validate_submission.jl
#
# Env vars (args take precedence when both are given):
#   MODEL_ID    model id, e.g. nfidd-seasarpp (required)
#   HUB_PATH    path to the local hub clone
#               (default: ~/code/external/sismid-ili-forecasting-sandbox)
#   FILE_PATH   a single file to check, relative to model-output/, e.g.
#               "nfidd-seasarpp/2015-11-07-nfidd-seasarpp.csv"; if unset,
#               every CSV under model-output/<model_id>/ is checked
#
# Programmatic use (no environment vars, no exit()):
#   include("scripts/validate_submission.jl")
#   ok, problems = validate_submission("nfidd-seasarpp"; hub_path = "...")

using CSV
using DataFrames
using Dates
using JSON
using YAML

const DEFAULT_HUB_PATH =
    joinpath(homedir(), "code", "external", "sismid-ili-forecasting-sandbox")

const REQUIRED_COLUMNS = [
    :origin_date, :location, :target, :horizon, :target_end_date,
    :output_type, :output_type_id, :value,
]

"""
    HubConfig

The subset of hub-config/tasks.json this checker cares about, read fresh
from the hub clone so it never drifts from the hub's actual rules.
"""
struct HubConfig
    origin_dates::Set{Date}
    locations::Set{String}
    horizons::Set{Int}
    target::String
    quantile_levels::Vector{Float64}
end

function load_hub_config(hub_path::AbstractString)
    path = joinpath(hub_path, "hub-config", "tasks.json")
    if !isfile(path)
        error("hub-config/tasks.json not found at $path")
    end
    cfg = JSON.parsefile(path)
    task = cfg["rounds"][1]["model_tasks"][1]
    task_ids = task["task_ids"]

    origin_dates = Set(
        Date(d) for d in task_ids["origin_date"]["optional"]
    )
    locations = Set{String}(task_ids["location"]["optional"])
    horizons = Set{Int}(task_ids["horizon"]["optional"])
    target = only(task_ids["target"]["required"])
    quantile_levels = Float64.(
        task["output_type"]["quantile"]["output_type_id"]["required"]
    )

    return HubConfig(origin_dates, locations, horizons, target,
                      quantile_levels)
end

"""
    parse_origin_date_from_filename(filename, model_id) -> Union{Date,Nothing}

Extract the origin_date from a `<origin_date>-<model_id>.csv` filename,
or `nothing` if it does not match that pattern.
"""
function parse_origin_date_from_filename(filename, model_id)
    m = match(Regex("^(\\d{4}-\\d{2}-\\d{2})-" * Base.escape_string(model_id) *
                     "\\.csv\$"), filename)
    m === nothing && return nothing
    return tryparse(Date, m.captures[1])
end

function check_filename(filename, model_id)
    problems = String[]
    origin_date = parse_origin_date_from_filename(filename, model_id)
    if origin_date === nothing
        push!(
            problems,
            "filename '$filename' does not match the required " *
            "<origin_date>-<model_id>.csv pattern for model_id " *
            "'$model_id'",
        )
    end
    return problems, origin_date
end

function check_columns(df::DataFrame, label::String)
    problems = String[]
    actual = Symbol.(names(df))
    if actual != REQUIRED_COLUMNS
        push!(
            problems,
            "$label: columns are $(actual), expected exactly " *
            "$(REQUIRED_COLUMNS) in that order",
        )
    end
    return problems
end

function check_target(df::DataFrame, label::String, cfg::HubConfig)
    :target in Symbol.(names(df)) || return String[]
    bad = unique(filter(!=(cfg.target), df.target))
    isempty(bad) && return String[]
    return ["$label: target has values other than '$(cfg.target)': $bad"]
end

function check_locations(df::DataFrame, label::String, cfg::HubConfig)
    :location in Symbol.(names(df)) || return String[]
    problems = String[]
    bad = unique(setdiff(Set(df.location), cfg.locations))
    if !isempty(bad)
        push!(problems, "$label: unrecognised location(s): $bad")
    end
    missing_locs = setdiff(cfg.locations, Set(df.location))
    if !isempty(missing_locs)
        push!(
            problems,
            "$label: missing location(s): $(sort(collect(missing_locs)))",
        )
    end
    return problems
end

function check_horizons(df::DataFrame, label::String, cfg::HubConfig)
    :horizon in Symbol.(names(df)) || return String[]
    bad = unique(filter(h -> !(h in cfg.horizons), df.horizon))
    isempty(bad) && return String[]
    return ["$label: horizon has values outside $(cfg.horizons): $bad"]
end

function check_origin_date(
    df::DataFrame, label::String, cfg::HubConfig,
    filename_origin_date::Union{Date,Nothing},
)
    :origin_date in Symbol.(names(df)) || return String[]
    problems = String[]
    dates = unique(df.origin_date)
    if filename_origin_date !== nothing && dates != [filename_origin_date]
        push!(
            problems,
            "$label: origin_date column value(s) $dates do not match " *
            "the filename's origin_date $filename_origin_date",
        )
    end
    bad = unique(filter(d -> !(d in cfg.origin_dates), dates))
    if !isempty(bad)
        push!(
            problems,
            "$label: origin_date value(s) not in the hub's round list: " *
            "$bad",
        )
    end
    return problems
end

function check_target_end_date(df::DataFrame, label::String)
    needed = Symbol.([:origin_date, :horizon, :target_end_date])
    all(c -> c in Symbol.(names(df)), needed) || return String[]
    expected = df.origin_date .+ Day.(7 .* df.horizon)
    bad_rows = findall(expected .!= df.target_end_date)
    isempty(bad_rows) && return String[]
    return [
        "$label: $(length(bad_rows)) row(s) where target_end_date != " *
        "origin_date + 7*horizon days",
    ]
end

function check_output_type(df::DataFrame, label::String)
    :output_type in Symbol.(names(df)) || return String[]
    bad = unique(filter(!=("quantile"), df.output_type))
    isempty(bad) && return String[]
    return ["$label: output_type has values other than 'quantile': $bad"]
end

function check_quantile_levels(df::DataFrame, label::String, cfg::HubConfig)
    needed = Symbol.([:location, :horizon, :output_type_id])
    all(c -> c in Symbol.(names(df)), needed) || return String[]
    problems = String[]
    expected = Set(cfg.quantile_levels)
    for grp in groupby(df, [:location, :horizon])
        levels = Set(Float64.(grp.output_type_id))
        if levels != expected
            loc, hz = grp.location[1], grp.horizon[1]
            missing_l = setdiff(expected, levels)
            extra_l = setdiff(levels, expected)
            msg = "$label: location='$loc' horizon=$hz has the wrong " *
                  "quantile levels."
            if !isempty(missing_l)
                msg *= " missing: $(sort(collect(missing_l)))"
            end
            if !isempty(extra_l)
                msg *= " unexpected: $(sort(collect(extra_l)))"
            end
            push!(problems, msg)
        end
    end
    return problems
end

function check_value(df::DataFrame, label::String)
    :value in Symbol.(names(df)) || return String[]
    bad = count(v -> ismissing(v) || v < 0, df.value)
    bad == 0 && return String[]
    return ["$label: $bad value(s) are missing or negative (must be >= 0)"]
end

"""
    check_file(path, model_id, cfg) -> Vector{String}

Run every per-file format check on one submission CSV and return the
list of problems found (empty if the file is clean).
"""
function check_file(path::AbstractString, model_id::AbstractString,
                     cfg::HubConfig)
    label = basename(path)
    if !isfile(path)
        return ["$label: file not found at $path"]
    end

    problems, origin_date = check_filename(basename(path), model_id)

    df = try
        CSV.read(path, DataFrame)
    catch err
        push!(problems, "$label: could not read as CSV: $err")
        return problems
    end

    append!(problems, check_columns(df, label))
    append!(problems, check_target(df, label, cfg))
    append!(problems, check_locations(df, label, cfg))
    append!(problems, check_horizons(df, label, cfg))
    append!(problems, check_origin_date(df, label, cfg, origin_date))
    append!(problems, check_target_end_date(df, label))
    append!(problems, check_output_type(df, label))
    append!(problems, check_quantile_levels(df, label, cfg))
    append!(problems, check_value(df, label))
    return problems
end

"""
    check_metadata(hub_path, model_id) -> Vector{String}

Check model-metadata/<model_id>.yml has the required scalar fields from
hub-config/model-metadata-schema.json: team_abbr, model_abbr (both
matching `^[a-zA-Z0-9_+]+\$`, max 25 chars) and designated_model (bool).
"""
function check_metadata(hub_path::AbstractString, model_id::AbstractString)
    label = "model-metadata/$model_id.yml"
    path = joinpath(hub_path, "model-metadata", "$model_id.yml")
    if !isfile(path)
        return ["$label: file not found at $path"]
    end

    meta = try
        YAML.load_file(path)
    catch err
        return ["$label: could not parse as YAML: $err"]
    end

    problems = String[]
    abbr_re = r"^[a-zA-Z0-9_+]+$"

    for key in ("team_abbr", "model_abbr")
        val = get(meta, key, nothing)
        if val === nothing || !(val isa AbstractString) || isempty(val)
            push!(problems, "$label: missing or empty required field '$key'")
        elseif !occursin(abbr_re, val) || length(val) > 25
            push!(
                problems,
                "$label: '$key' = '$val' must match $(abbr_re) and be " *
                "<= 25 characters",
            )
        end
    end

    designated = get(meta, "designated_model", nothing)
    if !(designated isa Bool)
        push!(
            problems,
            "$label: 'designated_model' must be present and boolean " *
            "(true/false), got $(repr(designated))",
        )
    end

    team_abbr = get(meta, "team_abbr", nothing)
    model_abbr = get(meta, "model_abbr", nothing)
    if team_abbr isa AbstractString && model_abbr isa AbstractString
        expected_id = "$team_abbr-$model_abbr"
        if expected_id != model_id
            push!(
                problems,
                "$label: team_abbr-model_abbr = '$expected_id' does not " *
                "match model_id '$model_id'",
            )
        end
    end

    return problems
end

function resolve_files(hub_path, model_id, file_path)
    model_dir = joinpath(hub_path, "model-output", model_id)
    if file_path !== nothing && file_path != ""
        candidates = [
            file_path,
            joinpath(hub_path, "model-output", file_path),
            joinpath(model_dir, file_path),
        ]
        idx = findfirst(isfile, candidates)
        return idx === nothing ? [candidates[1]] : [candidates[idx]]
    end
    isdir(model_dir) || return String[]
    return sort(
        joinpath.(model_dir, filter(f -> endswith(f, ".csv"),
                                     readdir(model_dir))),
    )
end

"""
    validate_submission(model_id; hub_path=DEFAULT_HUB_PATH,
                         file_path=nothing) -> (ok::Bool, problems)

Check every submission file for `model_id` (or just `file_path` if given)
against the hub's format rules, and the model's metadata YAML. Returns
whether everything passed and the full list of problems found (empty
list means a clean pass). Does not read or write anything outside the
hub clone; never submits or pushes.
"""
function validate_submission(
    model_id::AbstractString;
    hub_path::AbstractString = DEFAULT_HUB_PATH,
    file_path::Union{AbstractString,Nothing} = nothing,
)
    isdir(hub_path) || error("hub not found at $hub_path")
    cfg = load_hub_config(hub_path)

    problems = check_metadata(hub_path, model_id)

    files = resolve_files(hub_path, model_id, file_path)
    if isempty(files)
        push!(
            problems,
            "no submission files found for model_id='$model_id' under " *
            "$(joinpath(hub_path, "model-output", model_id))",
        )
    end
    for f in files
        append!(problems, check_file(f, model_id, cfg))
    end

    return isempty(problems), problems
end

function main()
    args = ARGS
    model_id = length(args) >= 1 ? args[1] : get(ENV, "MODEL_ID", "")
    if model_id == ""
        println(stderr, "no model_id given. Pass it as the first " *
                         "argument or set MODEL_ID.")
        exit(1)
    end
    hub_path = length(args) >= 2 ? args[2] : get(ENV, "HUB_PATH", "")
    hub_path = hub_path == "" ? DEFAULT_HUB_PATH : hub_path
    file_path = get(ENV, "FILE_PATH", "")
    file_path = file_path == "" ? nothing : file_path

    println("validating '$model_id' against hub at $hub_path")
    ok, problems = validate_submission(model_id; hub_path, file_path)

    println("\n--- validation summary: $model_id ---")
    if ok
        println("  PASS: no format problems found")
    else
        for p in problems
            println("  FAIL: $p")
        end
        println("\n$(length(problems)) problem(s) found.")
    end
    exit(ok ? 0 : 1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
