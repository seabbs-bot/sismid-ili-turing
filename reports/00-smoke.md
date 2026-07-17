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
| 1 | nfidd-base | 1.0601 | 0.6747 | 0.3610 | 0.0393 | 0.0935 | ok |  |

## Bayesian workflow

Per-candidate prior/posterior predictive and residual checks (mean over scored splits).

| Candidate | Prior % outside | Prior % non-finite | Post cov50 | Post cov90 | Max |resid ACF(1)| |
|---|---|---|---|---|---|
| nfidd-base | 0.6969 | 0.0000 | 0.0393 | 0.0935 | 0.9733 |

## Log-scale divergence check

Natural-scale and log-scale WIS agree on the candidate ordering. No divergence.

## Candidates needing a fix

None: every candidate produced forecasts within time.

## Decision

_To be completed by the round reviewer: which candidates to keep, refine, or drop, and the next axes to explore._

## Artifacts

- Report: `reports/00-smoke.md`
- Runner: `experiments/run_round.jl`
