"""
    SismidILITuring

Joint Turing model for nowcasting and forecasting weighted influenza-like
illness (wILI) for the SISMID hub session. The package assembles the base
seasonal + AR(1) + backfill model, its inference wrappers, the
posterior-predictive forecaster, WIS scoring, Bayesian-workflow
diagnostics, and hub I/O into one always-submittable pipeline.

`produce_submission()` is the front-runner entry point: fit the base model
on each validation split and build the hub quantile forecast table. See
docs/contracts.md for the shared interfaces and docs/brief.md for the
design rationale.
"""
module SismidILITuring

# Component files are `include`d in dependency order and share the module
# scope (see docs/contracts.md); each brings its own `using` statements.
include("core.jl")
include("data.jl")
include("model.jl")
include("scoring.jl")
include("forecast.jl")
include("hubio.jl")
include("inference.jl")
include("diagnostics.jl")
include("pipeline.jl")

# Core types, constants, transforms (core.jl)
export ModelData, LOCATIONS, QUANTILE_LEVELS, TARGET, HORIZONS,
    to_scale, from_scale

# Data (data.jl)
export load_series, build_model_data, training_splits

# Model (model.jl)
export base_model, model_dims

# Scoring (scoring.jl)
export wis, score_forecasts, wis_summary, compare_scales

# Forecasting (forecast.jl)
export forecast_quantiles, default_project, base_project

# Hub I/O (hubio.jl)
export write_submission, write_metadata

# Inference (inference.jl)
export fit_pathfinder, fit_mcmc, posterior_draws, generated_draws,
    progress_callback

# Diagnostics (diagnostics.jl)
export prior_predictive, posterior_predictive, residual_summary,
    bayesian_checks

# Pipeline (pipeline.jl)
export produce_submission, fit_and_forecast

end # module SismidILITuring
