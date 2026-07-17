# Brief: requirements from Sam

This is the authoritative statement of what to build and how to work.
It is written from Sam's instructions.
When in doubt, follow this document; update it if Sam changes the ask.

## What we are building

A custom infectious-disease forecasting model, submitted to the online
`reichlab/sismid-ili-forecasting-sandbox` hub, following the two SISMID
local-hub sessions (`hub-playground.qmd`, `hub-playground-testing.qmd`).
We fit and search on the two validation seasons, stream down to a few models,
test them on the three testing seasons, and submit the finalist(s).

## Experimental integrity (do not cheat)

This is a hard constraint, not a guideline, matching the SISMID session rules.

- Tune and select models only on the two validation seasons: 2015/16 and
  2016/17.
- The three testing seasons (2017/18, 2018/19, 2019/20) are held out.
  Do not fit to, look at, or select on them until the finalists are locked.
- Use only vintage data available at each forecast origin.
  Never let finalized or future values leak into a fit.
- Score against the hub oracle output, so comparisons stay fair against the
  existing hub models.

## Modelling requirements

- Written in Julia using Turing.jl.
- Joint model across all 11 locations with partial pooling throughout.
- Fit on the log-transformed scale by default; Sam's stated preference is
  "I like log transform ARs".
  Logit and fourth-root are candidate alternative transforms.
- Seasonality learned as a random-effect structure, not Fourier terms.
  It varies by location and by season (time).
- Autoregressive noise on top of the seasonal component, partially pooled.
- A reporting-backfill model in the spirit of the baselinenowcast package, but
  with a non-monotonic report profile. wILI revisions can move up or down with
  delay because wILI is a re-weighted percentage, not an accumulating count, so
  the delay-indexed revision is not constrained to a monotonic reporting CDF.
  The revision profile differs by location and over time, partially pooled
  across both. Built into the joint Turing model rather than run as a separate
  step.

Formulations to test as candidate axes in the search:

- Location random effects: multivariate normal (correlated across locations)
  vs independent.
- After seasonality: autoregression vs differencing.
  A single first difference is a favoured candidate within the differencing
  branch.
- Time-varying AR coefficients.
- AR order greater than 2.
- AR as a vector autoregression (VAR) across locations vs independent per
  location.
- Backfill profile: shared vs location-varying vs time-varying.

## Model checking (Bayesian workflow)

Every candidate model goes through the full Bayesian workflow before it is
scored, not only the finalists:

- Prior predictive checks, to confirm the priors give plausible wILI
  trajectories.
- Posterior predictive checks, to confirm the fitted model reproduces the
  observed patterns.
- Residual analysis, to check for structure the model has not captured.

## Search and selection

- Iterate and "tree out" models: start broad, keep the branches that do well,
  refine them, and test again.
- Score with the weighted interval score (WIS) using `ScoringRules.jl` from the
  EpiAware org.
- Guard against overfitting using both WIS and the standard deviation of WIS.
  Reject models with very high WIS variance even if their mean WIS is good.
- Prefer simpler models where performance is comparable.
- Also report WIS computed on the log scale, but do not optimise on it. Flag any
  case where model choice would diverge between natural-scale and log-scale WIS.

## Compute and inference

- MCMC can be slow, so screen candidates with Pathfinder as a fast first pass;
  reserve full MCMC for promising models and finalists.
- Use Mooncake as the reverse-mode automatic differentiation backend for all
  models.
- Use Turing callbacks to monitor fits in real time.

## Ways of working

- Use parallel subagents with implement-and-review loops: 5 to 10 subagents
  running at once per round.
- Move fast; do not let caution slow the search down.
- Run at least 10 rounds of implement-and-review.
- Per round, use multiple implementers (lower-power models are fine) proposing
  competing implementations, and a reviewer picks the preferred one.
- Merge to main via pull requests or directly, depending on the parallel-agent
  setup.
- Always keep a few core jobs running regardless of box load; spin out more work
  when there is headroom. Do not let the shared-box load gate block progress.
- Keep a report for every iteration loop.

## Deliverable and submission

- Repository owned by the `seabbs-bot` account, with clear markdown docs: this
  brief, the plan, infrastructure as it lands, and a report per iteration loop.
- Prepare and validate the hubverse submission for the finalist(s).
- Open a submission smoke-test pull request first, to prove the hubverse
  mechanics work end to end, ahead of the finalist submission.
- Pause for Sam's explicit go-ahead before opening the external pull request to
  the reichlab hub.
