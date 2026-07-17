# Forecast projection for the round-2 candidate loc-obs.
#
# `model_loc_obs` (model_loc_obs.jl) changes only the observation noise
# scale, from `base_model`'s single shared `sigma_obs` to a length-L,
# partially-pooled `sigma_obs[l]`. `base_project` (src/forecast.jl)
# never reads `sigma_obs` at all -- the observation noise term only
# widens the likelihood for already-observed cells, it plays no part in
# forward-simulating the latent AR(1)/difference residual or the
# seasonal curve into future weeks -- so it forecasts this candidate's
# draws correctly with no changes needed. Every field `base_project`
# does read (`mu0`, `mu_w`, `delta`, `season_eff`, `phi`, `sigma_ar`,
# `residual`) is unchanged in shape from `base_model`.
#
# `project_loc_obs` is therefore a plain alias, kept as its own name/
# file (rather than referencing `base_project` directly at call sites)
# so this candidate follows the same `(name, build_model, project)`
# shape as every other candidate in `experiments/README.md`, and so a
# future change to `model_loc_obs`'s return fields has an obvious,
# candidate-local place to add a real `project_loc_obs` without
# touching call sites.
const project_loc_obs = base_project
