# Experiments: the tree-search loop

`src/` holds the current front-runner as the loadable `SismidILITuring`
package. `experiments/` iterates around it: each round proposes competing
candidate models, scores them on the two validation seasons, and a reviewer
promotes the winner. See `docs/plan.md` for the search strategy and
`docs/brief.md` for the requirements (especially experimental integrity).

The engine is [`run_round.jl`](run_round.jl). It fits and scores candidates
resiliently and in parallel-safe fashion, so a round never blocks on one slow
or failing fit, and writes a report to `reports/<roundname>.md`.

## Experimental integrity

Tuning and selection use ONLY the validation seasons (2015/16, 2016/17). The
truth loader (`load_validation_truth`) reads the hub oracle output and
refuses to return any test-season (2017/18-2019/20) values. Do not add code
that scores against the test seasons until the finalists are locked.

## What a candidate is

A candidate is a `(name, build_model, project)` triple:

- `name::String` -- the `model_id` (e.g. `"nfidd-seasar"`); it tags the
  forecast rows and names the ranking entry.
- `build_model(data::ModelData; transform::Symbol)` -- returns a Turing model
  in the `base_model` family. `base_model` itself qualifies; so does any
  thin wrapper such as `model_v3(d; transform) = base_model(d;
  transform=transform, difference=true)`.
- `project(draw, data, horizons)` -- the forecast projector that turns one
  `generated_draws` draw into the `(L x maximum(horizons))` latent-scale
  forecast matrix `forecast_quantiles` consumes. `base_project` is the AR(1)
  default; write a `project_vN` when the dynamics differ (see
  `round1/v3-diff/project_v3.jl`).

Pass candidates as a `Vector` of `NamedTuple`s or plain tuples:

```julia
candidates = [
    (name="nfidd-base",  build_model=base_model,  project=base_project),
    (name="nfidd-diff",  build_model=model_v3,    project=project_v3),
]
```

## How to add a candidate

1. Make a directory `experiments/roundN/<slug>/`.
2. Add `model_<slug>.jl` defining `build_model(d; transform)` (usually a thin
   wrapper around `base_model`, or a new `@model`). Reuse `src/model.jl`
   helpers (`ar_or_diff`, `backfill_profile`, `model_dims`) where you can.
3. Add `project_<slug>.jl` defining the projector, mirroring `base_project`
   but for the candidate's dynamics. Skip this and reuse `base_project` if
   the forward dynamics are unchanged.
4. Optionally add a `check_<slug>.jl` with a prior-predictive sanity check.

Each candidate passes the Bayesian-workflow checks (prior/posterior
predictive, residual autocorrelation) automatically inside `run_candidate`
before it is scored.

## How to run a round

From the repository root:

```julia
julia --project=.
julia> include("experiments/run_round.jl")

julia> include("experiments/round1/v3-diff/model_v3.jl")
julia> include("experiments/round1/v3-diff/project_v3.jl")

julia> candidates = [
           (name="nfidd-base", build_model=base_model, project=base_project),
           (name="nfidd-diff", build_model=model_v3,   project=project_v3),
       ];

julia> out = run_round(candidates, "02-ar-vs-diff");
julia> out.ranking          # sorted by mean WIS, then WIS SD
```

Running the script directly (`julia --project=. experiments/run_round.jl`)
executes `smoke_round()`: the base model over one validation split with few
draws, a cheap end-to-end check that produces a ranking row and a report.

### Useful keywords

`run_round(candidates, roundname; ...)`:

- `seasons=[1, 2]` -- validation seasons to score on.
- `max_splits=nothing` -- cap splits per season (e.g. `4` for a fast pass).
- `ndraws=400` -- Pathfinder draws per fit (fast screening; MCMC is reserved
  for promising candidates and finalists).
- `Dmax=12`, `transform=:log1p` -- data build settings (EDA defaults).
- `diag_ndraws=100` -- draws for the Bayesian-workflow checks.
- `timeout=nothing` -- per-candidate wall-clock limit in seconds. A candidate
  that overruns is flagged `:slow` and abandoned so it cannot block the
  round. The limit is only pre-emptive with more than one Julia thread
  (`julia -t auto`); otherwise it protects against an outright hang once the
  fit yields.
- `slow_threshold=600` -- seconds above which a finished candidate is marked
  `:slow` in the report.

## What a round produces

- A ranking table sorted by mean WIS then WIS SD (the overfitting guard).
- Natural- vs log-scale WIS rank comparison, flagging any disagreement (we
  select on natural-scale WIS; log-scale is report-only).
- Per-candidate Bayesian-workflow summaries.
- A "candidates needing a fix" section listing every `:failed` or `:slow`
  candidate with its captured error. We fix complex or failing models rather
  than abandon them for being complex, so failures are recorded, never
  dropped.
- The report file `reports/<roundname>.md`, from `reports/TEMPLATE.md`.

## Resilience

Every candidate runs inside a `try/catch` and an optional timeout. One crash
or hang is recorded with `status=:failed`/`:slow` and its error message; the
other candidates still run and still rank. A single split that fails within a
candidate is caught too, so a candidate that forecasts most splits still
scores, with the failed-split count noted.
