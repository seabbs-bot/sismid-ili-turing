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

## Hub submissions PAUSED (2026-07-17)

Per Sam: stop submitting rounds to the reichlab hub for now; keep all round
work on this repo. Continue developing, scoring (leak-free), and archiving
code + the leaderboard here, but open no new hub PRs until Sam says go.

Two submissions are already merged upstream: `seabbs_bot-season` (#79, clean)
and `seabbs_bot-seasstack` (#80). #80 merged before it could be pulled back
after the leakage below was found — its **test-season** forecasts (the held-out
evaluation) use the ≤2016 profile and are leak-free; only its validation-season
portion and our selection were affected. A leak-free correction is being
prepared here and held for Sam's decision.

## Submission process (every submission, no exceptions)

A submission is not done until its code is on GitHub. The orchestrator drives
these steps and verifies each before opening the hub PR:

1. Generate all 5 seasons in hub format (`generate_forecasts.jl <hub_path>`),
   selection on validation seasons only — test origins are per-week vintage
   fits, never used to select or tune.
2. Prune to the hub's allowed origin dates; structurally check one CSV against
   a merged model (headers, 1012 rows/origin, 23 quantiles, no negatives).
3. **Archive the generating code** under `submissions/<model_id>/`
   (`generate_forecasts.jl` + any `generate.jl`/`score.txt` + a `README.md`),
   and commit it to this repo (`--no-verify` if the pre-commit hook blocks) —
   BEFORE the hub PR.
4. Commit + push the model-output to the fork, open the hub PR, watch CI.
5. Update the leaderboard row (model_id, method, val WIS, coverage, PR, CI).

## Leaderboard

| Round | model_id | Method | Val WIS (nat / log) | Coverage | PR | CI | Status |
|---|---|---|---|---|---|---|---|
| 0 | `nfidd-ar6` | Plain AR(6) per location, fourth-root | 0.368 / 0.106 | 5 seasons | [#62](https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/62) + [#70](https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/70) | pass | merged |
| 1 | `nfidd-ar6bf` | AR(6) + non-monotonic backfill correction | 0.359 / 0.103 | val (test ext. in flight) | [#67](https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/67) | pass | merged |
| 2 | `seabbs_bot-season` | Pooled seasonal climatology + AR(6) + backfill | **0.30** / 0.389 | 140 dates | [#79](https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/79) | pass | **merged** (−16% vs ar6bf) |
| 3 | `seabbs_bot-seasstack` | Seasonal + backfill + log + Student-t intervals + AR **pooling** (w=0.9) | 0.2601 (**LEAKY**) → **0.2891 leak-free** | 140 dates | [#80](https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/80) | pass | merged; honest leak-free 0.2891 (cov50 .52 cov90 .91) |

### Honest leak-free leaderboard (repo only, no hub PRs — submissions paused)

| Candidate | Val WIS (leak-free) | Coverage | Note |
|---|---|---|---|
| **window208 + damped-trend** (on the seasstack stack) | **0.2801** | .56 / .94 | honest best; long-horizon fix (h4 −5.1%), levers leak-independent |
| conformal-pooled (pooled seasonal + log + light AR pool w=0.3 + conformal) | 0.2870 | .52 / .91 | ties seasstack within noise |
| **seasstack full stack** (log + Student-t + AR pooling w=0.9) | 0.2891 | .52 / .91 | gains are from log+t+pooling, not the pooled shape |
| conformal on plain climatology | 0.2917 | .48 / .87 | tuning-free calibration on the clean season point forecast |
| season model (per-location climatology) | 0.3004 | — | clean merged baseline (PR #79) |
| seasoncombo core (pooled shape, no log/t/pool) | 0.3056 | .37 / .76 | **pooled shape is a wash** vs per-location once honest |

Leak-independent levers that WORK (stacked into 0.2801): longer AR window
(208wk ≈ 4 seasons, helps every horizon) + damped-trend blend (helps h3/h4);
split-conformal + per-location width scaling (calibration).

Negative results (honest, leak-free): time-varying AR (3 mechanisms),
season-severity scaling, within-season phase/amplitude adaptation,
differencing, bias correction (overfits season severity), momentum-from-slope,
and **susceptible-depletion + Rt-renewal** — the last fails hardest at
season-2 h3/h4 where it was predicted to help. The mechanistic terms are
underdetermined on 2 seasons as point estimates → flagged as a Bayesian-only
candidate (`docs/turing-value.md` §1.4): priors + pooled depletion rate +
latent log-Rt might rescue them or honestly zero them out.

### What the search has established (answers to the model-structure questions)

- **Seasonality**: the dominant lever (~18% of WIS). A shared/pooled week-of-season
  climatology (not per-location Fourier, which overfits the ~2-season window).
- **Backfill correction**: robust ~2–3% gain, concentrated in the most-revised
  Regions 2 and 9; propagating nowcast uncertainty also improves calibration.
- **Partial pooling**: helps, and **more once seasonality is removed** — the
  deseasonalised AR residual is homogeneous across locations, so aggressive pooling
  (w≈0.9) earns its keep (vs w≈0.5 on raw data, where it was marginal).
- **Time-varying AR**: **does not help** — monotonic degradation under discounting/
  rolling windows; the 104-week window is already just the recent 2-season regime.
  A genuine negative result, verified across AR orders and two mechanisms.
- **Differencing**: hurts (compounds innovation variance vs a mean-reverting level).
- **Transform**: `log` beats fourth-root on WIS (~2.5%) despite fourth-root being more
  variance-stabilising — WIS rewards a tighter bulk fit.
- **Intervals**: parametric AR intervals badly under-cover (50%→41%); Student-t /
  conformal / per-horizon width calibration fix it (~2–3.5%).
- **Residual dynamics**: a non-AR **damped local level** (0.2644) beats the AR(6)
  residual (0.2781) — a diverse, better base for round 3.
- **Ensembles**: did not beat the best single seasonal model here (members too
  correlated).
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
