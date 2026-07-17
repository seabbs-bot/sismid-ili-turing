# Loop 00-smoke

- **Date**: 2026-07-17
- **Parent loop**: base
- **Inference**: Pathfinder (fast screening)
- **Seasons scored**: validation (2015/16, 2016/17)

## What was tried

1 candidate(s): nfidd-base.

## Setup

- Seasons: [1] (splits per season: 1)
- Draws (Pathfinder): 60; Dmax: 12; transform: log1p
- Locations: 11; Quantiles: 23; AD backend: Mooncake

## Results

Ranked by mean WIS, then WIS SD (overfitting guard). Natural-scale WIS is the selection metric; log-scale WIS is report-only.

| Rank | Candidate | Mean WIS | WIS SD | Log-scale WIS | Cov50 | Cov90 | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| - | _none scored_ | | | | | | | |

## Bayesian workflow

Per-candidate prior/posterior predictive and residual checks (mean over scored splits).

| Candidate | Prior % outside | Prior % non-finite | Post cov50 | Post cov90 | Max |resid ACF(1)| |
|---|---|---|---|---|---|

## Log-scale divergence check

Natural-scale and log-scale WIS agree on the candidate ordering. No divergence.

## Candidates needing a fix

We fix complex or failing models rather than abandon them for being complex.

- **nfidd-base** (`:failed`): split 1: FieldError: type NamedTuple has no field `mu_w_raw`, available fields: `latent`, `seasonal`, `residual`, `mu0`, `mu_w`, `delta`, `season_eff`, `phi`, `sigma_ar`, `r`, `r_pop`, `sigma_obs`, `transform`

## Decision

_To be completed by the round reviewer: which candidates to keep, refine, or drop, and the next axes to explore._

## Artifacts

- Report: `reports/00-smoke.md`
- Runner: `experiments/run_round.jl`
