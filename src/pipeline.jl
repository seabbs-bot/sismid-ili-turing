# Always-ready submission driver: fit the base model on each validation
# split, forecast the hub quantile table, and optionally write it into a
# hubverse hub clone. This is the front-runner entry point -- runnable at
# any time to produce a valid, submittable forecast set. See
# docs/contracts.md and docs/brief.md.
#
# Included last by src/SismidILITuring.jl, so it may use build_model_data
# (data), base_model (model), fit_pathfinder / posterior_draws
# (inference), forecast_quantiles / base_project (forecast), and
# write_submission / write_metadata (hubio).

using DataFrames
using Dates
using Random

"""
    fit_and_forecast(build_model, data, model_id; project=base_project,
        ndraws=200, nruns=1, transform=data.transform, horizons=HORIZONS,
        levels=QUANTILE_LEVELS, rng=Random.default_rng())

THE canonical fit -> forecast path for one model on one already-built
`data::ModelData` split. This is the single source of truth: every
driver in this package (`produce_submission`, the round search harness
in `experiments/`) calls this rather than hand-rolling its own
build/fit/draws/forecast sequence. See docs/lessons.md for why this
matters (the `generated_draws` vs `posterior_draws` bug had to be fixed
in three places before this existed).

Builds `build_model(data; transform=transform)`, fits it with
[`fit_pathfinder`](@ref) (`ndraws`, `nruns`; single-path by default,
the fast screening path), and turns the fit's [`generated_draws`](@ref)
-- NOT [`posterior_draws`](@ref), which lacks the derived quantities a
`project` function needs -- into the hub quantile table via
[`forecast_quantiles`](@ref) with the given `project`.

`transform` defaults to `data.transform` so the model is built on the
same scale `data.Y` was filled on; pass it explicitly only if a
candidate deliberately fits a different scale than its data was built
with.

Returns a `NamedTuple` `(forecast, model, fit, draws)`: `forecast` is
the hub table (docs/contracts.md schema); `model`, `fit`, `draws` are
returned too so a caller that also needs Bayesian-workflow diagnostics
([`bayesian_checks`](@ref)) can reuse the same fit and draws rather
than refitting.

# Training discipline

Asserts `maximum(data.dates) <= data.origin_date` before fitting: `data`
must never carry training history dated after its own forecast origin,
or future/finalized values would leak into the fit (docs/contracts.md
experimental integrity). [`build_model_data`](@ref) guarantees this by
construction, so this only fires if a caller hand-builds or mutates a
`ModelData` incorrectly -- it is the last gate every driver passes
through, so the guard lives here.
"""
function fit_and_forecast(
    build_model,
    data::ModelData,
    model_id::AbstractString;
    project::Function=base_project,
    ndraws::Int=200,
    nruns::Int=1,
    transform::Symbol=data.transform,
    horizons=HORIZONS,
    levels=QUANTILE_LEVELS,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    max_train_date = maximum(data.dates)
    discipline_msg = "training discipline violation: data's latest " *
        "training date ($max_train_date) is AFTER its forecast origin " *
        "($(data.origin_date)) -- future/finalized data would leak " *
        "into the fit (docs/contracts.md experimental integrity)"
    @assert max_train_date <= data.origin_date discipline_msg

    model = build_model(data; transform=transform)
    fit = fit_pathfinder(model; ndraws=ndraws, nruns=nruns, rng=rng)
    draws = generated_draws(model, fit)
    forecast = forecast_quantiles(draws, data, model_id;
                                  horizons=horizons, levels=levels,
                                  project=project)
    return (forecast=forecast, model=model, fit=fit, draws=draws)
end

"""
    produce_submission(; seasons=[1, 2], hub_path=nothing,
        model_id="nfidd-turing", transform=:log1p, Dmax=12, ndraws=200,
        window_weeks=104, write=false, strict=true,
        allow_test_season=false)

Build the hub quantile forecast table for the base model across every
cross-validation split of each season in `seasons`, and (if `write`)
write it into the hubverse hub clone at `hub_path`.

For each split: [`build_model_data`](@ref) constructs the `ModelData`
(with the given `transform` and `Dmax`), and [`fit_and_forecast`](@ref)
fits `base_model` with [`fit_pathfinder`](@ref) (fast variational
screening) and turns the fit's [`generated_draws`](@ref) into the hub
table via [`base_project`](@ref), which recomputes the derived latent
dynamics from the raw sampled sites. Per-split tables are concatenated
into one submission.

Defaults follow the EDA: `transform=:log1p` (wILI has exact zeros) and
`Dmax=12` (backfill tail is ~10-15 weeks). `window_weeks` caps the
training history per split (fewer weeks fit faster; the seasonal curve
wants ~2 seasons). `seasons=[1, 2]` are the validation seasons;
[`training_splits`](@ref) refuses a test season unless
`allow_test_season=true` is also passed (only for the locked-finalist
test-phase step).

# Coverage (never silent)

Every `training_splits(season)` origin date, for every `season` in
`seasons`, is expected in the result. A split that throws while fitting
is caught -- so one bad split does not sink the whole run -- and its
origin date is logged (`@error`) as about to go missing, rather than
silently dropped. Once every split has been attempted, coverage is
checked explicitly against that expected set: with the default
`strict=true`, any missing origin date raises an error naming exactly
which dates are missing (and `write` never runs, since the error fires
first); with `strict=false` the same is emitted via `@warn` and the
partial submission is returned anyway.

If `write`, `hub_path` must be given: [`write_submission`](@ref) writes
one CSV per origin date under `model-output/<model_id>/` and
[`write_metadata`](@ref) writes `model-metadata/<model_id>.yml`, with
`team_abbr`/`model_abbr` split from `model_id` on the first `-`.

Returns the combined submission `DataFrame` (docs/contracts.md schema).
"""
function produce_submission(;
    seasons=[1, 2],
    hub_path::Union{Nothing,AbstractString}=nothing,
    model_id::AbstractString="nfidd-turing",
    transform::Symbol=:log1p,
    Dmax::Int=12,
    ndraws::Int=200,
    window_weeks::Int=104,
    write::Bool=false,
    strict::Bool=true,
    allow_test_season::Bool=false,
)
    tables = DataFrame[]
    expected_origins = Date[]
    failed_origins = Tuple{Date,String}[]

    for season in seasons
        splits = training_splits(season; allow_test_season=allow_test_season)
        for split in splits
            origin = maximum(split.origin_date)
            push!(expected_origins, origin)
            data = build_model_data(split; Dmax=Dmax, transform=transform,
                                    window_weeks=window_weeks)
            try
                result = fit_and_forecast(base_model, data, model_id;
                                          project=base_project,
                                          ndraws=ndraws, transform=transform)
                push!(tables, result.forecast)
            catch err
                push!(failed_origins, (origin, sprint(showerror, err)))
                @error "produce_submission: split failed; its origin " *
                       "date will be MISSING from the submission" origin err
            end
        end
    end

    if isempty(tables)
        first_err = isempty(failed_origins) ? "n/a" : failed_origins[1][2]
        error("produce_submission: every split failed; no forecasts " *
              "were produced. First error: $first_err")
    end
    submission = reduce(vcat, tables)

    expected = sort(unique(expected_origins))
    produced = sort(unique(submission.origin_date))
    missing_dates = setdiff(expected, produced)
    if !isempty(missing_dates)
        msg = "produce_submission: MISSING forecasts for " *
              "$(length(missing_dates))/$(length(expected)) expected " *
              "origin_date(s): $missing_dates"
        strict ? error(msg) : @warn msg
    end

    if write
        hub_path === nothing &&
            throw(ArgumentError("hub_path is required when write=true"))
        parts = split(model_id, "-"; limit=2)
        team_abbr = parts[1]
        model_abbr = length(parts) == 2 ? parts[2] : model_id
        write_submission(submission, hub_path)
        write_metadata(model_id, hub_path;
                       team_abbr=team_abbr, model_abbr=model_abbr)
    end

    return submission
end
