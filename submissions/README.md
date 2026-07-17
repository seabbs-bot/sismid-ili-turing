# Submissions tracker

One distinct model per round, submitted to the
[reichlab/sismid-ili-forecasting-sandbox](https://github.com/reichlab/sismid-ili-forecasting-sandbox)
hub. Each row is its own model_id — we never overwrite a previous submission.
Selection is on the validation seasons only (2015/16, 2016/17); every
submission covers all 5 seasons' origin dates (validation + test) because a
test-season forecast is a legitimate per-week vintage fit, not training on it.

WIS is mean weighted interval score on the validation seasons (natural scale;
log1p in brackets), lower is better. The generating code for each model is
archived in `submissions/<model_id>/`.

## Leaderboard

| Round | model_id | Method | Val WIS (nat / log) | Coverage | PR | CI | Status |
|---|---|---|---|---|---|---|---|
| 0 | `nfidd-ar6` | Plain AR(6) per location, fourth-root | 0.368 / 0.106 | 5 seasons | [#62](https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/62) + [#70](https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/70) | pass | merged |
| 1 | `nfidd-ar6bf` | AR(6) + non-monotonic backfill correction | 0.359 / 0.103 | val (test ext. in flight) | [#67](https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/67) | pass | merged |
| 2 | _held — improving_ | seasonality + backfill (+ pooling / differencing) | tbd | 5 seasons | tbd | tbd | holding for a seasonal+backfill model that improves on 0.352 |
| — | `seabbs_bot-round1` (Turing) | Joint partial-pooling Turing model (base-tight) | tbd | 5 seasons | tbd | tbd | scoring |

Closed (not shipped): #71 (Fourier overwrote the baseline and scored worse,
0.412), #73/#74 (`seabbs_bot-*` duplicated already-merged models).

## Wide simple round — validation WIS (in progress, holding submission)

| Variant | Val WIS (mean / sd) | vs ar6bf 0.359 |
|---|---|---|
| **WIS-weighted ensemble** (climatology + ar6 + ar6bf + ses) | **0.294** / 0.341 | **−18%** (best overall, lowest SD) |
| **Seasonal climatology + backfill** | **0.3004** / 0.389 | **−16%** (clean seasonal+backfill) |
| Pooled-seasonal + AR + backfill | 0.3049 / 0.387 | −15% |
| Combo (AR + backfill + seasonal grid) | 0.3349 / 0.420 | −7% |
| AR(12) + backfill | 0.3518 / 0.451 | −2% |
| Backfill: multiplicative, w6, per-loc median | 0.3586 / 0.449 | ≈ |
| ETS/alternatives (best) | 0.3798 / 0.415 | worse |
| Analytic partial pooling (fullpool, w0.5) | 0.3643 / 0.460 | +1.6%; pooling alone ~1% |

Findings: **seasonality + backfill is the winner** — a shared pooled/climatology
seasonal shape + the backfill correction cuts WIS ~16% (0.359 → 0.30), far more than
AR order or pooling alone, and an ensemble of the seasonal+simple models does best
(0.294, lowest variance). Confirms: backfill is the biggest single lever (~2.4%),
seasonality (pooled, not per-location Fourier) adds a large further gain, partial
pooling is modest (~1%). **Round submission: the ensemble / seasonal+backfill model
(~0.294–0.30).**

## Method notes

- **Round 0 → 1**: the backfill correction cut WIS 0.368 → 0.359 (~2.5%) with
  lower variance, with the gains concentrated in Regions 2 and 9 — exactly the
  most heavily/consistently revised locations the EDA flagged.
- **Two tracks in tandem**: a fast analytic path (per-location OLS models,
  seconds to fit) carries the rounds while the joint **Turing** partial-pooling
  model matures. The analytic partial-pooling and multi-location experiments
  tell us which structures genuinely need Turing (full Bayesian pooling, joint
  backfill, propagated uncertainty) versus what OLS already captures.
- **Seasonality**: a shared, pooled week-of-season shape across all 11 locations
  (naive per-location Fourier overfits the ~2-season training window).
