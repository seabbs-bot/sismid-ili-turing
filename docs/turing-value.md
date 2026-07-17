# What Turing is for

The analytic track (`experiments/simple-round/*`) has run a wide,
disciplined search and has already found most of the mean-WIS gain
available in this data: pooled/climatology seasonality, a point
backfill correction, log transform, a damped local-level residual,
and hand-tuned interval calibration.
Turing must not spend its (much higher) per-fit cost re-discovering
any of that.
Its job is the handful of things a point-estimate pipeline structurally
cannot do: propagate uncertainty jointly, and learn how much to pool
rather than have a human grid-search it.

This document says what genuinely needs the Bayesian joint model, what
does not, and gives a focused spec for the next iteration of
`src/model.jl`'s `base_model`.

## 1. What needs Bayesian, and why

### 1.1 Joint backfill with propagated revision uncertainty

The analytic backfill correction (`experiments/simple-round/backfill`)
is a **fixed point correction**: a per-(location, delay) median or mean
revision, computed once, applied as a constant shift.
It has no notion of how uncertain that correction is, so it cannot
widen the forecast for the most recent, least-revised weeks relative to
older, settled weeks — every week gets the same-width interval
regardless of how much backfill risk it still carries.

`experiments/simple-round/nowcast/score.txt` shows this gap is real and
worth closing: bolting a hand-rolled "propagated nowcast uncertainty"
stochastic tail onto the point correction bought a further 0.0065 mean
WIS (~1.8%) on top of the point correction and phase-conditioning, concentrated exactly
where you'd expect — the most heavily revised locations (Region 9: full=0.3297
vs A=0.4042) and, in the coverage breakdown, mostly at short horizons where
recent-week vintage noise dominates. That was a bespoke bootstrap
approximation of what a joint model gets for free.

`base_model` (`src/model.jl`) already has the right shape for this: `r_pop`/`r_loc`
are sampled parameters with a genuine posterior, entangled with `latent`
through the *same* observation model (`Y ~ Normal(latent + r, sigma_obs)`),
so the posterior width of `latent[T, l]` for a recent, still-revising
week should come out wider than for a settled week purely from the
model structure, with no hand-tuned inflation.
**Why analytic can't do this properly:** a point correction has no
distribution to propagate; retrofitting one means hand-designing a
"stochastic tail" per location/phase, which is exactly the ad hoc
mechanism `nowcast/generate.jl` had to build.
**WIS/coverage win only Bayesian can reach:** better-calibrated,
correctly *location- and recency-dependent* h=1–2 interval width at
Regions 2 and 9 specifically, without any hand-set inflation constant —
matching or beating the nowcast experiment's 1.8% gain as an emergent
property of the joint model, not a bolted-on mechanism.

### 1.2 Hierarchical partial pooling of seasonality and residual dynamics, with correct uncertainty

The analytic pooling experiments (`pool`, `seasonpool2`) each grid-search
a *single scalar shrinkage weight* on validation WIS, and the optimal
weight is **different, and sometimes opposite in direction, for
different parameters**: seasonal amplitude wants almost no pooling
(`seasonpool2`: best `shrink=1.0`, i.e. barely shrunk toward the pooled
mean at all), while the AR-residual coefficient wants heavy pooling
(`seasonpool2`: best `weight=1.0`, full pooling) — and naively stacking
both hand-tuned weights together (`combined-naive`) is *worse* than AR
pooling alone, because the two knobs interact and a grid search can't
see that.
A fixed, globally-tuned weight is also blind to *which locations* need
pooling: Region 2 is the worst-performing location in essentially every
analytic experiment in this repo (AR-order, backfill, season, ensemble,
var — it is *always* the worst or near-worst), yet the pooling weight
is the same for Region 2 as for the best-behaved location.

`base_model` already has the right machinery for this — non-centred
hierarchical variances (`sigma_season_loc`, `tau_log_sigma_ar`,
`sigma_r_loc`) are *learned parameters*, not fixed weights, so in
principle the model can independently discover "pool the AR residual
hard, pool seasonal amplitude lightly" without a human running two
separate grid searches and then discovering they don't compose.
**Why analytic can't do this properly:** OLS partial pooling needs one
scalar knob per structure, tuned by an outer validation-WIS loop; it
cannot let per-location shrinkage vary by how much support that
location's own data gives it, and it cannot jointly resolve
interacting shrinkage weights the way `combined-naive`'s failure shows
it needs to.
**WIS/coverage win only Bayesian can reach:** matching or beating the
best single-axis analytic pooling result (0.2685, AR-only full pooling,
`seasonpool2`) *without* a grid search, and — the real test — doing
better than the naively-stacked combination (0.2702) by letting the
posterior find each hierarchical variance's own level rather than
combining two independently-tuned constants. A secondary, diagnostic
win: the posterior should show Region 2 pooled *less* toward the
population phi/sigma than the well-behaved locations, since it is
persistently the outlier.

### 1.3 Full posterior uncertainty propagation, nowcast through multi-horizon (coherent intervals, not bolted-on calibration)

`experiments/simple-round/intervals/score.txt` found that a point AR
forecast's own Gaussian interval badly under-covers (41%/78% at
50%/90% nominal) and only reaches good WIS/coverage after being
inflated by a hand-tuned scale (×1.4) and/or swapped to Student-t.
`conformal/score.txt` reaches similar calibration by an entirely
separate post-hoc split-conformal step layered on top of the point
forecast.
Both are corrections applied *after* the fact because the point
model's own uncertainty quantification is wrong by construction — a
symmetric Gaussian around a point estimate cannot represent the
skew a fourth-root/log back-transform induces, so it has to be blown
up past its natural spread to compensate.

A joint Bayesian model's posterior predictive, if reasonably specified
(hierarchical variances on `sigma_ar`/`sigma_obs`/`sigma_season_*`,
Student-t rather than Gaussian observation noise), should be calibrated
*by construction*, because the width comes from propagating actual
parameter and process uncertainty through the same recursion that
produces the point forecast, not from a separate calibration pass.
**Why analytic can't do this properly:** the point model has to fit a
single best-guess trajectory, then patch its own uncertainty with an
external device (inflation constant, conformal calibration set);
neither of those correct *why* the interval was wrong, they just widen
it until the nominal/actual coverage numbers line up on the validation
set.
**WIS/coverage win only Bayesian can reach:** cov50/cov90 close to
nominal (matching the Student-t analytic result: 0.525/0.892) with no
external scale or conformal step — a strictly harder and more
informative bar than beating mean WIS alone, since it proves the
uncertainty is *right*, not merely *rescaled to fit*.

### 1.4 Mechanistic renewal / susceptible-depletion with parameter uncertainty and pooling

Still being tested analytically (`next-susdep`, `next-longhorizon`,
`next-tvpool`) for long-horizon performance, so this is a lower-priority,
second branch, not part of the next iteration below.
If the analytic point-estimate mechanistic term is shown to help at
h=3–4, the Bayesian-necessary part is **not** the mechanism itself (a
point depletion/Rt-renewal coefficient can be estimated by MLE just
fine) — it is pooling that coefficient across locations with honest
parameter uncertainty, so a data-poor location's forecast interval
widens honestly at long horizon instead of inheriting a point estimate
of unknown reliability.
This directly targets the one place WIS grows fastest in every
analytic experiment run so far (h=4 is routinely ~1.6–1.8x h=1), where
the AR/local-level residual has no mechanism to bend the curve down as
a season runs out of susceptibles, so long-horizon width is currently
carried by the (already-good) local-level/Student-t machinery alone,
not by anything that understands epidemic dynamics.

## 2. What Turing should NOT bother re-doing

The analytic track has already nailed the following; re-deriving them
in Turing wastes the 10-minute-per-fit budget on a known answer.

- **Seasonal shape itself** (the mean climatology curve). Per-location
  climatology (K=11) beats every clustering level down to K=1
  (`cluster/score.txt`) — the smoothed, full-13-season climatology is
  already well regularised by construction. Keep `base_model`'s
  per-location `mu_w`/`delta` structure as is; do not add Fourier terms
  (naive Fourier(3) overfit badly, 0.412) and do not spend a search
  axis re-litigating shape.
- **Backfill point correction** — the fixed multiplicative/
  per-location/median profile already captures the correction; only
  its *uncertainty* (§1.1) is the open Bayesian question, not
  re-deriving the point profile itself.
- **Transform choice.** Log beats fourth-root by ~4% mean WIS
  (`transform/score.txt`), independent of estimation framework — just
  build `ModelData` with `transform=:log` (`Dmax`/`window_weeks`
  unchanged), do not run this as a Turing search axis.
- **AR order.** Flat from order 8–12 (`ar-order/score.txt`); `base_model`
  already uses a single lag structure via the level/AR(1)-or-diff
  residual, not an order sweep — leave it.
- **Time-varying AR coefficients, in every mechanism tried:**
  discounted OLS, rolling window, RLS forgetting factor, and
  Gaussian-kernel local weighting all degrade monotonically, with and
  without seasonality, at multiple AR orders (`tvar`, `tvar-season`).
  This is a settled negative result — do not add a time-varying `phi`
  to `base_model`.
- **Differencing** as the primary residual — hurts once seasonality is
  present (`seasoncombo` combo 6: diff worse than AR-on-residual by
  ~4.4%). `base_model`'s `difference` flag can stay as an option but is
  not worth defaulting to or searching over.
- **Season-severity scaling** — a clean, thoroughly-checked negative
  result across every form/pooling/window tried (`severity2`).
- **Ensembling near-identical seasonal members** — did not beat the
  best single model (`seasonens`); not a Bayesian question anyway.
- **Cross-location VAR predictors as a fixed lag structure.**
  Ridge-VAR(1) helps (~8%, `var/score.txt`) but needs heavy hand-tuned
  regularisation and a flat lambda region — this is a mean-estimation
  question, not an uncertainty one, and correlated location effects
  are already a listed candidate axis (`brief.md`, MVN vs independent)
  that overlaps with §1.2's pooling work rather than being a separate
  new axis. Do not build a bespoke VAR block; if cross-location
  correlation is wanted, it belongs inside the hierarchical structure
  in §1.2 (e.g. correlated, not independent, location deviations).

## 3. Spec for the next Turing iteration

Built on `src/model.jl`'s `base_model`; changes below, nothing else.

**A. Transform.** Fit on `transform=:log` (not the current
`fourthroot`/`log1p` mix across drivers) — administrative, matches the
settled analytic finding, not a modelling change.

**B. Residual dynamics: damped local level, not AR(1).**
Analytic found a damped local level beats the AR(6)/AR(1) residual on
top of the same seasonal+backfill base (`locallevel`: 0.2611 vs 0.2781;
`sesresid`: 0.2644 vs 0.2781 for a slightly different base). Add a
`damped_level(eps, sigma, alpha, phi)` function alongside
`ar_or_diff` in `src/model.jl` (mean-reverting: level accumulates a
damped random walk rather than a stationary AR(1)), and extend the
`difference::Bool` switch to a `dynamics::Symbol` in
`{:ar1, :damped, :diff}`. Pool `alpha`/`phi` hierarchically across
locations exactly as `phi`/`sigma_ar` already are (non-centred, shared
population mean/sd) — this is what lets §1.2 be tested on the better
base rather than a known-inferior AR(1) residual.

**C. Observation noise: Student-t, not Gaussian.**
Change `yobs ~ arraydist(Normal.(mu_obs, sigma_obs))` to a Student-t
location-scale family (`sigma_obs` as scale, a hierarchical or fixed
`nu` — start with `nu ~ Gamma`-type prior or a fixed moderate df like
10, matching the analytic `intervals` finding of a flat optimum across
df 8–20). One-line change, directly targets §1.3: it gives the model a
native heavy tail instead of relying on a symmetric Gaussian that
would otherwise need the same kind of external inflation the analytic
Gaussian family needed.

**D. Do not touch:** seasonal shape structure, backfill point-profile
structure (only validate its *propagated* uncertainty, don't
re-derive the profile), AR order search, time-varying phi, VAR blocks,
severity scaling, differencing-as-primary.

### Validation comparisons that would prove this earns its complexity

1. **Backfill uncertainty (§1.1).** At Regions 2 and 9, h=1–2:
   compare the joint model's native posterior-predictive interval
   width/coverage against (a) the analytic point-backfill + Student-t
   scheme (`intervals`: cov50 0.525/cov90 0.892) and (b) the analytic
   nowcast experiment's hand-rolled propagated-uncertainty tail
   (`nowcast`: 0.0065 gain over point correction alone). Win condition:
   match or beat (b)'s gain with no hand-tuned stochastic-tail
   mechanism, and show interval width is visibly wider for the most
   recent 1–2 weeks at Regions 2/9 specifically, not uniform across all
   weeks.

2. **Learned vs grid-searched pooling (§1.2).** Fit once, read off the
   posterior hierarchical variances (`sigma_season_loc`,
   `tau_log_sigma_ar`, `sigma_r_loc`/equivalent for the new dynamics).
   Win condition: (a) mean WIS matches or beats the single best
   analytic pooling axis (0.2685) without any grid search, (b) beats
   the naively-stacked combination (0.2702) that analytic pooling could
   not resolve, and (c) Region 2's implied posterior shrinkage toward
   the population mean is visibly weaker than the other 10 locations'
   (a diagnostic on the fitted `phi`/`alpha` draws, not a WIS number).

3. **Calibration without a calibration step (§1.3).** Score the raw
   posterior-predictive quantiles directly — no external scale
   inflation, no conformal recalibration. Win condition: cov50/cov90
   at or near nominal (comparable to the Student-t analytic scheme's
   0.525/0.892) at every horizon h=1–4, without a horizon-specific
   adjustment (per-horizon width scaling was itself a *negative*
   analytic result, `intervals/score.txt` family 4 — so a Bayesian
   model reproducing flat, correctly-calibrated horizon growth natively
   is the target, not something to hand-tune toward).

4. **Overall.** Mean WIS on validation should beat the current best
   simple-round stack (0.2601, `round2-stack`) by enough to justify a
   ~10-minute-per-split Turing fit against a sub-second analytic one —
   treat anything within ~1–2% of 0.2601 as *not* clearing that bar
   unless comparisons 1–3 show a genuine calibration win the analytic
   stack cannot reach even in principle (comparison 3 in particular),
   since WIS alone can be matched by a well-tuned point model plus
   enough hand-calibration.
