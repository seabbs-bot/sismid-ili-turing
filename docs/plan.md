# Plan

The design, search strategy, and phased execution plan.
See [`brief.md`](brief.md) for the requirements this plan serves.

## Model design

Everything is fit on a transformed scale `y = g(wILI%)`, with log as the
default transform (Sam's stated preference); logit and fourth-root are
candidate alternatives. For location `i`, week `t`:

```
y[i,t] = seasonal[i, week-of-season(t), season(t)] + noise[i,t]
vintage observation = joint-backfill(latent value, reporting delay)
```

- **Seasonality** is a random-effect structure over week-of-season (not Fourier),
  partially pooled across locations, and allowed to vary by location and by
  season. Location effects are tested as multivariate normal (correlated across
  the 11 locations) against independent.
- **Noise** is an autoregressive process on the post-seasonal residual, partially
  pooled. Variants tested: AR vs differencing (a single first difference is a
  favoured candidate within the differencing branch), AR order (including > 2),
  time-varying AR coefficients, and independent AR per location vs a VAR across
  locations.
- **Backfill** is a joint revision model in the spirit of baselinenowcast, but
  with a non-monotonic report profile. A latent "final" value is observed through
  a delay-indexed revision effect (partially pooled), estimated jointly so recent
  incomplete weeks are effectively nowcast before the forecast extends forward.
  Because wILI is a re-weighted percentage rather than an accumulating count, the
  revision can move up or down as the delay grows, so the profile is not
  constrained to a monotonic reporting CDF. The profile differs by location and
  over time, partially pooled across both, and shared vs location-varying vs
  time-varying is itself a candidate axis. Vintage data drives this, matching
  what a real-time forecaster saw.

## Search and selection

A tree/beam search rather than a one-shot factorial screen.
All search, fitting and selection uses only the two validation seasons; see
[`brief.md`](brief.md#experimental-integrity-do-not-cheat) for the
experimental-integrity constraint this must not break.

1. Start from a tractable base (log transform + partially-pooled seasonal
   random effect + independent AR + simple backfill).
2. Score every candidate on the validation seasons.
3. Keep the branches that do well, refine them (the candidate axes above), and
   repeat.
4. Promote the best 1-2 parsimonious finalists to the testing seasons.

Each candidate passes the Bayesian workflow checks in
[`brief.md`](brief.md#model-checking-bayesian-workflow) before it is scored:
prior predictive checks, posterior predictive checks, and residual analysis.

Scoring and selection:

- Primary metric: mean WIS on the natural scale, via `ScoringRules.jl`.
- Overfitting guard: standard deviation of WIS across origin dates and
  locations. Reject high-variance models even when mean WIS looks good.
- Secondary, report-only: WIS on the log scale. We do not optimise on it, but
  flag any case where natural-scale and log-scale WIS disagree on model choice.
- Parsimony tie-breaker: prefer fewer parameters where performance is comparable.

Inference during the search uses Pathfinder for a fast first pass; promising
candidates and all finalists are refit with full MCMC. Mooncake is the AD
backend for every model, and Turing callbacks report fit progress live.

## Phases and checkpoints

Each checkpoint commits and pushes to `seabbs-bot/sismid-ili-turing`, updates the
status line in the README, and (for search loops) adds a report under
`reports/`.

- [ ] **Phase 0 — setup**
  - [x] Create the repo on `seabbs-bot` and clone the target hub.
  - [x] Scaffold the repo and write the docs (brief, plan, infrastructure).
  - [ ] Julia project with Turing, Mooncake, Pathfinder, ScoringRules, Arrow.
  - [ ] Export the course R data objects to Arrow for Julia.
  - [ ] WIS helper (natural and log scale) built on `ScoringRules.jl`, tested.
- [ ] **Phase 1 — machinery**
  - [ ] Base joint model in Turing with the design above.
  - [ ] Forecast generation: posterior predictive to 23 quantiles x 4 horizons
        x 11 locations per origin date.
  - [ ] Hubverse CSV writer and a local `hubValidations` check.
  - [ ] Experiment-report template wired to the scoring output.
- [ ] **Phase 2 — search** (validation seasons 2015/16, 2016/17)
  - [ ] Run the tree search over at least 10 implement-and-review rounds; one
        report per loop; rank by WIS and WIS SD.
  - [ ] Each round: 5 to 10 implementers (lower-power models allowed) propose
        competing candidates; a reviewer picks the preferred one to carry
        forward.
  - [ ] Each candidate passes prior predictive checks, posterior predictive
        checks, and residual analysis before it is scored (see brief.md's
        Bayesian workflow subsection).
- [ ] **Phase 3 — select** the best 1-2 parsimonious finalists, balancing WIS,
      WIS SD, and parsimony.
- [ ] **Phase 4 — test** (seasons 2017/18–2019/20)
  - [ ] Full expanding-window forecasts for finalists; validate submission.
- [ ] **Phase 5 — submit**
  - [ ] Fork the reichlab hub under `seabbs`, add model output and metadata.
  - [ ] Open a submission smoke-test pull request to prove the hubverse
        mechanics work, ahead of the finalist submission.
  - [ ] Pause for Sam's go-ahead, then open the finalist pull request.

## Submission format

- `model_id` follows `[team]-[model]`; default team `nfidd` (e.g.
  `nfidd-seasarpp`), confirmed before submission.
- 23 quantiles: 0.01, 0.025, then 0.05 to 0.95 by 0.05, then 0.975, 0.99.
- Horizons 1–4 weeks; target `ili perc`; one CSV per origin date; a model
  metadata YAML in `model-metadata/`.
