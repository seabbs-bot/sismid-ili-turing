# Forecast projection for the round-2 candidate base-tight.
#
# `model_base_tight` (model_base_tight.jl) changes only `base_model`'s
# (src/model.jl) hyperprior scales/locations -- see that file's header
# comment for the full before/after table. No sampled site is added,
# removed, or reshaped, and the return `NamedTuple`'s field names are
# identical to `base_model`'s, so `base_project` (src/forecast.jl)
# already forward-simulates this candidate's AR(1)/difference residual
# and seasonal curve correctly with no changes needed.
#
# `project_base_tight` is therefore a plain alias, kept as its own
# name/file (rather than referencing `base_project` directly at call
# sites) so this candidate follows the same `(name, build_model,
# project)` shape as every other candidate in `experiments/README.md`,
# and so a future change to `model_base_tight`'s return fields has an
# obvious, candidate-local place to add a real `project_base_tight`
# without touching call sites.
const project_base_tight = base_project
