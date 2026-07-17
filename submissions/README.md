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
| 2 | _wide simple round winner_ | tbd (AR order / backfill / seasonal / VAR / pooling / mixtures) | tbd | 5 seasons | tbd | tbd | in progress |
| 3 | _seasonal model_ | pooled-seasonal + AR + backfill | tbd | 5 seasons | tbd | tbd | in progress |
| — | `seabbs_bot-round1` (Turing) | Joint partial-pooling Turing model (base-tight) | tbd | 5 seasons | tbd | tbd | scoring |

Closed (not shipped): #71 (Fourier overwrote the baseline and scored worse,
0.412), #73/#74 (`seabbs_bot-*` duplicated already-merged models).

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
