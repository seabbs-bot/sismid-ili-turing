# Hub comparison: our sandbox vs FluSight vs CovidHub

This compares our teaching sandbox
([`reichlab/sismid-ili-forecasting-sandbox`](https://github.com/reichlab/sismid-ili-forecasting-sandbox))
against the two operational hubs it descends from: CDC FluSight
([`cdcepi/FluSight-forecast-hub`](https://github.com/cdcepi/FluSight-forecast-hub))
and the COVID-19 Forecast Hub, now CovidHub
([`CDCgov/covid19-forecast-hub`](https://github.com/CDCgov/covid19-forecast-hub)).
All facts below come from each hub's `hub-config/tasks.json` and `README.md`,
read directly for this comparison.
Our own characterisation draws on
[`docs/brief.md`](brief.md), [`docs/contracts.md`](contracts.md) and
[`docs/turing-value.md`](turing-value.md).

## 1. Our sandbox: `sismid-ili-forecasting-sandbox`

- **Target**: `ili perc`, weighted ILI percentage (wILI%) from ILINet, a
  continuous target in percent units.
- **Signal type**: a re-weighted **percentage**, not a count.
  It is not additive across reporting sites and can revise **up or down**
  with delay, because the weighting scheme itself shifts as more providers
  report (`contracts.md`, `brief.md`).
- **Locations**: 11 — US National plus 10 HHS regions.
- **Horizons**: 1–4 weeks ahead only.
  No nowcast (h=0) or hindcast (h=-1) horizon.
- **Output type**: `quantile`, 23 levels
  (`0.01, 0.025, 0.05, 0.10, …, 0.90, 0.95, 0.975, 0.99`), the standard
  hubverse two-sided quantile set.
- **Revision/backfill**: documented in unusual depth in the hub README.
  Two separate target-data files exist: `time-series.csv`, the
  **as-of-`origin_date`** snapshot a real-time forecaster would actually have
  seen (`as_of = origin_date`, using the release available by
  `origin_date + 9` days, the FluSight-era Monday deadline); and
  `oracle-output.csv`, the **scoring truth**, frozen at the release on or
  before 1 July of the following season to avoid CDC's between-season
  re-baselining leaking into scores.
  Our own `flu_data_hhs_versions.csv` reporting triangle is built from this.
- **Evaluation**: retrospective, not live-scored by the hub itself.
  The README describes it as "a sandbox environment for training, research
  or benchmarking purposes" over five already-completed seasons
  (2015–2019); we score against `oracle-output.csv` with WIS
  (`src/scoring.jl`, `ScoringRules.jl`) ourselves, matching hubverse
  convention.
- **Cadence/format**: no live deadline.
  "Forecasts may be submitted for any of the original submission dates" as
  a hubverse pull request — a backtest, not a running challenge.

## 2. CDC FluSight (`FluSight-forecast-hub`)

- **Targets** (5 `model_tasks` blocks in `tasks.json`, confirmed directly):
  1. `wk inc flu hosp` — weekly confirmed influenza hospital admissions, a
     **count**. Primary/mandatory target. `quantile` and `sample` output
     types (100-sample trajectories accepted alongside quantiles).
  2. `wk inc flu prop ed visits` — proportion of ED visits due to flu, a
     **rate/proportion** (0–1), optional. `quantile` and `sample`.
  3. `wk flu hosp rate change` — categorical direction/magnitude of change
     in hospitalisation rate, `pmf` over 5 ordinal categories
     (`large_decrease, decrease, stable, increase, large_increase`),
     defined by population-scaled rate-difference thresholds.
  4. `peak inc flu hosp` — quantile forecast of peak-week incidence.
  5. `peak week inc flu hosp` — `pmf` over epiweeks, probability the peak
     falls in each week.
  The hub moved from ILINet wILI% (the original FluSight Challenge this
  sandbox is templated on) to NHSN hospital-admission counts as the primary
  target after the 2022 reorganisation.
- **Signal type**: primary target is a **count** (lab-confirmed
  admissions), not a percentage — it does not carry wILI's
  re-weighting-driven non-monotonic revision.
  The optional ED-visits-proportion target is a proportion like wILI%, but
  from a different data source (NSSP, not ILINet).
- **Locations**: national plus "all 50 states, Washington DC, and Puerto
  Rico" per the README — 53 jurisdictions total in `tasks.json`
  (matches the FIPS-style location codes there).
- **Horizons**: `-1, 0, 1, 2, 3`. `-1` is a hindcast (QC only, "will not be
  scored in summary evaluations"); `0` is a genuine **nowcast** of the
  still-incomplete current epiweek; `1–3` are the forward horizons.
- **Output type/quantiles**: same 23-level standard hubverse quantile set
  for the primary target; `pmf` for the two categorical targets.
- **Data revision/backfill**: NHSN hospital admissions are released
  provisionally on Wednesday and finalised on Friday, then continue to
  update as hospitals report late — a reporting-completion problem
  structurally similar to what `baselinenowcast`-style triangle models
  assume, i.e. counts accrue toward a settled total rather than
  re-weighting up or down like wILI%.
- **Evaluation/cadence**: weekly, submissions due Wednesday 11pm ET,
  `reference_date` = the Saturday ending that epiweek; hubverse format,
  scored by hubverse convention (WIS + coverage) same as our own approach.

## 3. COVID-19 Forecast Hub / CovidHub (`covid19-forecast-hub`)

- **Targets**: `wk inc covid hosp` (weekly confirmed COVID hospital
  admissions, a **count**, mandatory) and `wk inc covid prop ed visits`
  (proportion of ED visits due to COVID, optional, added June 2025).
  Both take `quantile` and `sample` output types (confirmed in
  `tasks.json`).
- **Locations**: 53 (national + all 50 states + DC + PR), confirmed
  directly by field count in `tasks.json`.
- **Horizons**: `-1, 0, 1, 2, 3` — explicitly aligned with FluSight's
  horizon span; the README has a dedicated "Alignment between CovidHub and
  FluSight" section stating both hubs share the `-1:3` horizon span and the
  Wednesday 11pm ET deadline.
- **Output type/quantiles**: same 23-level standard hubverse set.
- **Revision/history**: the README notes this is a redesign — "changes
  from previous versions of the COVID-19 Forecast Hub challenges" — from
  the legacy `reichlab/covid19-forecast-hub`, which forecast COVID deaths
  and cases; the current CovidHub forecasts hospital admissions and ED
  visit proportions instead, deliberately mirroring FluSight's target
  structure.
- **Evaluation/cadence**: weekly, Wednesday 11pm ET, hubverse PR-based
  submission with automerge for pre-registered submitters, WIS + coverage
  evaluation.

## 4. Comparison at a glance

| | Our sandbox | FluSight | CovidHub |
|---|---|---|---|
| Primary target | wILI% (rate) | flu hosp admissions (count) | COVID hosp admissions (count) |
| Optional targets | none | ED visit %, rate-change PMF, peak week/incidence | ED visit % |
| Locations | 11 (US + 10 HHS regions) | 53 (US + 50 states/DC/PR) | 53 (US + 50 states/DC/PR) |
| Horizons | 1–4 | -1, 0, 1, 2, 3 | -1, 0, 1, 2, 3 |
| Output types | quantile (23 levels) | quantile, sample, pmf | quantile, sample |
| Revision behaviour | non-monotonic (%) | reporting-completion (count) | reporting-completion (count) |
| Cadence | retrospective, no deadline | live, weekly, Wed 11pm ET | live, weekly, Wed 11pm ET |
| Population scaling | none (wILI% self-normalises) | yes, for rate-change thresholds | not required for primary target |

## 5. What this means for our effort

### Transfers directly

- **Hierarchical partial pooling** (`turing-value.md` §1.2): the argument
  that a single grid-searched shrinkage weight can't capture per-location,
  per-structure pooling needs applies at least as strongly at 53
  states/DC/PR as at 11 HHS regions — more locations, more heterogeneous
  in population and data quality, is exactly the setting where a learned
  hierarchical variance beats a hand-tuned scalar.
- **WIS + coverage as the scoring standard** (`docs/contracts.md`,
  `src/scoring.jl`): both live hubs score on the same hubverse WIS
  convention we already build against; `score_forecasts` would need a new
  truth-table source, not a new scoring method.
- **Seasonal random-effect structure**: flu and COVID hospitalisations are
  seasonal too; the per-location, per-week-of-season random effect
  approach (not Fourier terms) is a reasonable starting point, though the
  shape itself would need re-fitting on the new signal, not re-used
  wholesale.
- **Conformal calibration** as a fallback/complement to native posterior
  calibration is signal-agnostic and ports unchanged.
- **Bayesian workflow discipline** (prior/posterior predictive checks,
  residual analysis before scoring, per `brief.md`) is process, not model
  structure, and transfers regardless of target.

### Needs rework, not a straight port

- **Backfill sign and monotonicity.** Our repo's distinguishing design
  choice is a **non-monotonic** delay-indexed revision model, justified
  specifically because wILI% is a re-weighted percentage
  (`brief.md`: "wILI revisions can move up or down with delay… not
  constrained to a monotonic reporting CDF").
  NHSN hospitalisation counts are a reporting-completion problem instead —
  closer to what `baselinenowcast` itself assumes.
  Porting the backfill block means checking whether the non-monotonic
  formulation is even the right shape here, or whether a conventional
  monotonic reporting-triangle nowcast is a better fit for counts; this is
  an empirical question, not a "keep as is" one.
- **Target transform and observation model.** Our transform search (log,
  logit, fourth-root, log1p) was tuned for a value bounded in [0, 100].
  Hospitalisation counts are unbounded non-negative integers with very
  different variance behaviour (small states near zero, occasional sharp
  peaks); a log1p or negative-binomial/Poisson-based observation model is
  more natural than any of our current candidates, and needs its own
  search rather than reusing the wILI% finding
  (`turing-value.md` "Transform choice" settled result does not carry
  over).
- **Nowcast/hindcast horizons.** Our `HORIZONS = 1:4` are all
  forward-only. Both live hubs need `h=0` (nowcast of the still-incomplete
  current week) and `h=-1` (hindcast). This is not pure loss, though — our
  joint backfill-and-forecast structure (`turing-value.md` §1.1, the
  `r_pop`/`r_loc` machinery that widens the interval for recent,
  still-revising weeks "for free") is closer to what a good `h=0` nowcast
  needs than a point-estimate pipeline is, since real submitted models
  currently patch this with hand-rolled propagated-uncertainty tails much
  like our own analytic `nowcast/` experiment did.
- **Categorical rate-change and peak-week targets.** FluSight's `pmf`
  outputs (5-category rate change, peak week) are a new output type we
  have never had to produce; `forecast_quantiles` would need a sibling
  function turning posterior predictive draws into category
  probabilities via population-scaled thresholds, not just quantiles.
- **Population scaling.** wILI% is already population-normalised by
  ILINet's survey weighting, so our model has no population covariate.
  FluSight's rate-change thresholds are defined per 100k population; a
  port needs state population added as a known covariate, at minimum for
  that target, and possibly to explain why residual variance differs so
  much between a state like Wyoming and a state like California in a way
  11 roughly-similar-sized HHS regions never surfaced.

### Candid take: is it worth spinning out?

Worth doing as a narrowly-scoped follow-on, not a wholesale port.
The strongest case is **not** our backfill story — that argument is
specific to wILI% being a re-weighted percentage, and mostly evaporates
against NHSN's count-based, reporting-completion revision pattern.
The strongest case is the **nowcast uncertainty and pooling** results in
`turing-value.md` §1.1–§1.3: a joint model that propagates backfill/nowcast
uncertainty through the same recursion that produces the forecast, and
learns per-location, per-structure pooling instead of grid-searching it,
addresses problems both live hubs still solve with ad hoc, hand-tuned
patches (their own submitted models' propagated-uncertainty tails and
fixed shrinkage). The highest-value single port would be a joint
nowcast+forecast model for `wk inc covid hosp` or `wk inc flu hosp` at
`h=0`, scored against the hubs' own historical model-output archives
(both are public), before attempting a live submission.

Main risks:

- These are **live, adjudicated hubs** with a real Wednesday-11pm-ET
  deadline and CDC-run evaluation, not a retrospective sandbox with no
  clock — the "10-minute-per-fit" compute budget in `turing-value.md` §4
  would need revisiting against a weekly real-time deadline across 53
  locations and (potentially) two diseases, a much larger joint model than
  11 regions for one signal.
- The count-target backfill/nowcast problem is already fairly well
  studied (NHSN's own reporting-completion behaviour, existing
  `baselinenowcast`-style tools); our edge there is genuinely one of
  degree (learned vs grid-searched, joint vs bolted-on), not of kind, so
  the win has to be demonstrated empirically against the hubs' existing
  model archive, not assumed from the wILI% results.
- Categorical and peak-timing targets are unexplored for us and would
  need real model-development time before any submission, not just a
  data-plumbing change.
