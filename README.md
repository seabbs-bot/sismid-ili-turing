# sismid-ili-turing

Forecasting of weighted influenza-like illness (wILI%) for the SISMID reichlab
sandbox hub, in Julia.

Two tracks were developed in parallel:

- A fast **analytic search** (per-location regression: seasonal climatology +
  AR + reporting-backfill correction, scored by the weighted interval score)
  that carried the model rounds and produced every submitted forecaster.
- A joint **Bayesian model in Turing.jl** (`src/`: partial pooling across the
  11 locations, random-effect seasonality, AR dynamics, joint non-monotonic
  backfill). Its infrastructure is complete and fits via Pathfinder + Mooncake,
  but it did **not** reach competitive accuracy — see the conclusions below and
  [`docs/turing-value.md`](docs/turing-value.md).

See **[Results and conclusions](#results-and-conclusions)** for what mattered,
what didn't, and how it held up out of sample.

## Goal

Build a custom model, validate and select it following the two SISMID local-hub
sessions, and submit the finalist(s) to
[`reichlab/sismid-ili-forecasting-sandbox`](https://github.com/reichlab/sismid-ili-forecasting-sandbox)
as a hubverse pull request so the forecasts appear on the
[online dashboard](https://reichlab.io/sismid-ili-forecasting-dashboard/forecast.html).

- **Validation seasons** (fit and search here): 2015/16, 2016/17.
- **Testing seasons** (measure selected models here): 2017/18, 2018/19, 2019/20.
- **Target**: quantile forecasts of wILI%, 1–4 weeks ahead, 11 locations,
  23 quantiles per task.

## Documentation

Read these in order.

1. [`docs/brief.md`](docs/brief.md) — the requirements from Sam (what to build
   and how to work). This is the authoritative brief.
2. [`docs/plan.md`](docs/plan.md) — the model design, search and selection
   strategy, and the phased plan.
3. [`docs/infrastructure.md`](docs/infrastructure.md) — tooling, data, scoring,
   and the parallel-agent workflow, filled in as each piece lands.
4. [`docs/steer-log.md`](docs/steer-log.md) — a running record of Sam's
   guidance and the action taken in response.
5. [`docs/eda/`](docs/eda/) — exploratory analysis, revisited across search
   rounds rather than fixed once, with committed plots under
   `docs/eda/figures/`.
6. [`reports/`](reports/) — one report per iteration loop, plus a running index.

A Documenter site (see `docs/infrastructure.md`) will render the Julia API
docs and link all of the above in one nav.

## Results and conclusions

Selection used the **validation seasons only** (2015/16, 2016/17). The three
test seasons (2017/18–2019/20) were scored **once, after the models were
locked**, as a held-out check ([`reports/test-evaluation.md`](reports/test-evaluation.md)) —
never used to select or tune. All WIS below is leak-free (seasonal and backfill
profiles rebuilt per forecast origin from strictly-prior data; the leak-free
builders live in `src/seasonal.jl`).

### Leaderboard (leak-free mean WIS, lower is better)

| Model | Validation | Test | Test cov 50/90 |
|---|---|---|---|
| `nfidd-ar6` (plain AR, baseline) | 0.368 | 0.623 | .42 / .76 |
| `nfidd-ar6bf` (+ backfill) | 0.359 | 0.621 | .41 / .75 |
| `seabbs_bot-season` (+ seasonality) | 0.300 | 0.550 | .38 / .72 |
| **`seasstack`** (+ log + Student-t + pooling) | 0.289 | **0.464** | .54 / .89 |
| `conformal-pooled` (+ window-208 + width) | **0.273** | 0.481 | .48 / .88 |
| hist-avg (hub baseline, external) | — | 0.922 | .36 / .83 |

All our models beat the hub's `hist-avg` baseline by a wide margin. Test WIS is
uniformly higher than validation because the test seasons are harder (season
2019/20's long-horizon targets fall in March–May 2020, coincident with COVID-19
disrupting US ILI surveillance — flagged as a confound, not a causal claim).

### What mattered
- **A regularised seasonal signal** — the biggest and most *transferable* lever
  (~16% on validation, ~11% on test). A smoothed week-of-season climatology
  borrowed from all history; naive per-location Fourier overfit and hurt.
- **Longer AR memory** (208 weeks ≈ 4 seasons) — the biggest late-stage lever on
  validation; helps every horizon, most at h3/h4.
- **Calibrated intervals** — Student-t / split-conformal / per-location width
  scaling; small WIS gain, large calibration gain (41%/78% → near-nominal).
- **Log transform** and **modest AR-coefficient pooling** — small but real.

### What didn't
- **The *pooled* seasonal shape** vs per-location — a wash once scored honestly.
- **Time-varying AR** — clean negative, three independent mechanisms.
- **Susceptible-depletion / Rt-renewal** — net loss, failing hardest exactly
  where predicted to help; a Bayesian-only candidate (`docs/turing-value.md`).
- **Within-season severity/phase adaptation, bias correction, differencing,
  VAR, AR order > 6, ensembles** — neutral-to-negative.

### What the held-out test taught us
- **Validation ranking did not fully hold.** `conformal-pooled` (best on
  validation) and `seasstack` **swapped** — on test the *simpler* `seasstack`
  wins. The window-208 + width tuning that topped validation was partly
  validation-overfit; its 5.6% validation edge was noise. **`seasstack` is the
  most defensible single best model.**
- **Backfill's edge was 2-season-specific** — +2.6% on validation, +0.3% on
  test. Seasonality, by contrast, transferred strongly.
- **The meta-lesson**: robust, well-regularised structure generalises; finely
  tuned structure and low-data mechanistic terms do not. The ideas that failed
  all share a cause — too many parameters for two seasons — which is precisely
  the case for propagated uncertainty and partial pooling in a Bayesian model.

### The Bayesian (Turing) model — honest status
The joint Turing model — the original goal — is **infrastructure-complete but
not competitive**. It fits (Pathfinder + Mooncake, FlexiChains) and forecasts,
and a full round-1 Turing sweep ran, but scored ~1.2–1.3 WIS (3–5× worse than
the analytic models) from prior-predictive/calibration problems, so no Turing
model was a finalist or entered the test evaluation.
[`docs/turing-value.md`](docs/turing-value.md) specifies what a Bayesian model
should target to earn its complexity (propagated nowcast/backfill uncertainty,
learned hierarchical pooling, mechanistic terms *with* uncertainty) — the
unfinished next phase.

### Integrity
Selection was validation-only throughout. A mid-project data leak (seasonal and
backfill profiles built once from `season_year ≤ 2016`, exposing validation-
season future weeks) was found, fixed at the root (`src/seasonal.jl`), and every
number was rebaselined leak-free; the two already-merged hub submissions'
residual leakage is documented in [`submissions/README.md`](submissions/README.md).

## Repository layout

```
src/          front-runner Julia model, forecasting, hubverse I/O, scoring;
              always loadable and ready to run and submit
scripts/      entry points for data export, screening runs, submission
data/         Julia-ready data (Arrow), exported from the course R package
experiments/  candidate model definitions and run configs for the search;
              winners are promoted into src/
reports/      one markdown report per iteration loop (+ index and template)
docs/         brief, plan, infrastructure, steer-log, eda
test/         package tests
```
