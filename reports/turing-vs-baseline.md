# Turing joint model vs. the nfidd-ar6 baseline

- **Date**: 2026-07-17
- **Status**: partial.
  Round 1 is still fitting on this box as this report was written (see "What's still pending" below); this will need another pass once more candidates finish.
- **Baseline**: `nfidd-ar6`, independent AR(6) per location, fourth-root scale, OLS-fit, no hierarchy/seasonality/backfill (`submissions/nfidd-ar6/`), merged into the hub in PR #62.
- **Scoring**: `src/scoring.jl`'s `score_forecasts`/WIS, validation seasons (2015/16, 2016/17) only.

## Headline comparison

Two apples-to-apples baseline numbers are given because the round-1 candidates below are (so far) all scored on a 4-origin-date subset (docs/lessons.md: cut from 8 to 4 after repeated whole-round crashes on this shared box), not the baseline's full 59-origin validation set.

| Baseline variant | Origins | n tasks | Mean WIS (natural) | Mean WIS (log) |
|---|---|---|---|---|
| nfidd-ar6, full validation (submissions/nfidd-ar6/README.md) | 59 | 2596 | **0.368** | 0.106 |
| nfidd-ar6, SAME 4 origins as round1 candidates below | 4 | 176 | **0.330** | 0.095 |

Rescored directly from the merged hub submission's `model-output/` files against `target-data/oracle-output.csv`, restricted to round1's `ORIGIN_DATES`.
See this report's provenance note at the bottom.

## Round 1 candidates scored so far

| Candidate | Transform | Mean WIS (nat) | SD | Mean WIS (log) | Cov50 | Cov90 | prior_frac_outside | vs nfidd-ar6 (4-origin, nat) |
|---|---|---|---|---|---|---|---|---|
| **nfidd-ar6 (baseline)** | fourthroot | **0.330** | 0.303 | 0.095 | -- | -- | -- | 1.00x |
| nfidd-base-log | log | 1.220 | 0.961 | 0.404 | 0.035 | 0.087 | 0.21 | **3.7x worse** |
| nfidd-diff | fourthroot | 2.044 | 1.409 | 0.885 | 0.078 | 0.193 | 0.61 | **6.2x worse** |

Source: `experiments/round1/_results/{nfidd-base-log,nfidd-diff}/summary.txt`, 4 origins (2015-11-14, 2016-03-12, 2016-12-24, 2017-02-25), 176 tasks each.

**So far, neither finished Turing candidate beats the simple AR(6) baseline -- both are substantially worse, not just slightly.**
Coverage is also far below nominal (cov50 should be ~0.50, cov90 ~0.90; both candidates are near 0.03-0.19), meaning the intervals are badly miscalibrated, not just centred wrong.

## Why this connects to the prior-tail fix in this session's other task

`prior_frac_outside` (fraction of PRIOR-predictive draws landing outside the plausible `[0, 15]` wILI% range, `src/diagnostics.jl`) is 0.21 for nfidd-base-log and a striking **0.61** for nfidd-diff -- i.e. 61% of nfidd-diff's own prior predictive draws are already implausible before any data is seen.
This is the same `base_model` hyperprior under-regularisation diagnosed and fixed in `base-tight` (`experiments/round2/base-tight/`, this session's other deliverable): an unbounded `mu_w` seasonal random walk and a near-unit-root-capable AR coefficient prior.
A model whose prior already puts most of its mass on implausible values is starting the fit from a bad place, and it shows up directly here as badly miscalibrated, high-WIS forecasts.
The natural next step (outside this report's scope) is scoring `base-tight` itself on the same 4 origins once a fit is run, to see whether the prior fix alone closes some of this gap.

## What's still pending (fill in as Round 1 completes)

As of this report, on this shared box:

- **nfidd-base** (the primary joint model, `:fourthroot`, the actual core deliverable) -- still fitting; 0/4 splits checkpointed after ~30+ minutes on this run, markedly slower than the other candidates at the same settings.
  Its comparison to the baseline is the single most important number missing from this report; re-run this comparison once its `summary.txt` exists.
  (nfidd-base-log, a `:log`-transform variant of the SAME model, IS in the table above -- but the primary/screening transform for the search is `:fourthroot` per `docs/lessons.md` #7, so nfidd-base-log is not a substitute for nfidd-base.)
- **nfidd-ar-high, nfidd-mvn-season, nfidd-tv-ar, nfidd-backfill** (round 1) and **nfidd-severity, nfidd-season-backfill, nfidd-ar-loc, nfidd-var** (round 2) -- a 10-worker pool run (`experiments/round1_pool.jl`) was launched partway through this session to cover these; no `summary.txt` for any of them yet.
- **base-tight** (this session's prior-fixed candidate) -- deliberately NOT fit in this session (only its prior-predictive check was run, per this task's brief, to avoid disrupting the in-flight Round 1 run).
  Its WIS is unknown; scoring it is the natural next step once the box has headroom.

## Provenance

- Baseline full-validation numbers: `submissions/nfidd-ar6/README.md` "Performance" table.
- Baseline 4-origin rescore: ad hoc script reading `model-output/nfidd-ar6/{origin}-nfidd-ar6.csv` from the hub clone (`/home/seabbs/code/external/sismid-ili-sandbox-fork-ar6`) for the 4 origins in `experiments/round1_run.jl`'s `ORIGIN_DATES`, scored with `src/scoring.jl`'s `score_forecasts` against the same `load_validation_truth`-equivalent oracle read `round1_run.jl` uses (validation-season-filtered `target-data/oracle-output.csv`, identical file content in both the ar6-fork and default hub clones -- verified byte-identical before reuse).
- Round 1 candidate numbers: `experiments/round1/_results/<name>/summary.txt`, written by `experiments/round1_run.jl`'s `aggregate_candidate`.

Re-run this report (or ask for it to be regenerated) once more `summary.txt` files land in `experiments/round1/_results/`, especially `nfidd-base` itself.
