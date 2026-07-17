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

"""
    produce_submission(; seasons=[1, 2], hub_path=nothing,
        model_id="nfidd-turing", transform=:log1p, Dmax=12, ndraws=200,
        window_weeks=104, write=false)

Build the hub quantile forecast table for the base model across every
cross-validation split of each season in `seasons`, and (if `write`)
write it into the hubverse hub clone at `hub_path`.

For each split: [`build_model_data`](@ref) constructs the `ModelData`
(with the given `transform` and `Dmax`), [`base_model`](@ref) is fit with
[`fit_pathfinder`](@ref) (fast variational screening), and
[`forecast_quantiles`](@ref) turns the fit's [`posterior_draws`](@ref)
into the hub table via [`base_project`](@ref), which recomputes the
derived latent dynamics from the raw sampled sites. Per-split tables are
concatenated into one submission.

Defaults follow the EDA: `transform=:log1p` (wILI has exact zeros) and
`Dmax=12` (backfill tail is ~10-15 weeks). `window_weeks` caps the
training history per split (fewer weeks fit faster; the seasonal curve
wants ~2 seasons). `seasons=[1, 2]` are the validation seasons; the
held-out test seasons are never touched here.

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
)
    tables = DataFrame[]
    for season in seasons
        for split in training_splits(season)
            data = build_model_data(split; Dmax=Dmax, transform=transform,
                                    window_weeks=window_weeks)
            model = base_model(data; transform=transform)
            fit = fit_pathfinder(model; ndraws=ndraws)
            draws = posterior_draws(fit)
            fq = forecast_quantiles(draws, data, model_id;
                                    project=base_project)
            push!(tables, fq)
        end
    end
    submission = reduce(vcat, tables)

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
