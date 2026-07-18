# Test-season evaluation: locked models, held-out seasons 3-5

- **Date**: 2026-07-18
- **Status**: model **selection is locked**.
  Every design scored below was chosen on the validation seasons
  (2015/16, 2016/17) only, in earlier rounds, and nothing in this report
  changed that.
  This is the legitimate final step: reporting how the already-locked
  designs perform on the three held-out **test** seasons (2017/18,
  2018/19, 2019/20), which have never been fit to, looked at, or
  selected on before now.
  No model was changed, retuned, or reselected based on the numbers
  below.
- **Seasons scored**: TEST only (season 3 = 2017/18, season 4 = 2018/19,
  season 5 = 2019/20).
  Validation-season numbers are quoted from earlier reports for
  comparison, never recomputed here.
- **Scoring**: `scratch-hub/target-data/oracle-output.csv`, natural
  scale, via `src/scoring.jl`'s `wis`/`score_forecasts` (Bracher et al.
  weighted interval score).

## Leak-free discipline

Every model below is generated leak-free: any seasonal climatology or
backfill/revision profile is rebuilt **per split**, from only the data
strictly before that split's own forecast origin.
For the backfill correction specifically, two of the five models
(`nfidd-ar6bf`, `seabbs_bot-season`) were originally submitted with a
profile built **once** from a fixed `season_year <= 2016` cutoff.
That fixed cutoff is already leak-free for test-season scoring (2016 is
strictly before every 2017-2020 test origin — see
`submissions/README.md`'s "Hub submissions PAUSED" note) but, for this
report, both were regenerated with the profile rebuilt per split via
the canonical `src/seasonal.jl` builders instead, so every model here
sits on the same, auditable, per-origin discipline.
The other three models (`nfidd-ar6`, the seasstack full stack, the
conformal-pooled honest frontier) already used per-origin, leak-free
construction throughout (`experiments/simple-round/round2-stack/` and
`experiments/simple-round/conformal-pooled/` respectively).

Generating code: `experiments/test-eval/gen_ar6.jl`,
`gen_ar6bf.jl`, `gen_season.jl`, `gen_seasstack.jl`, `gen_conformal.jl`;
scoring code: `experiments/test-eval/score_test.jl`.
Raw forecast tables are in `experiments/test-eval/out/` (regenerable,
not tracked beyond this evaluation).

## Models scored

1. **`nfidd-ar6`** — plain AR(6) per location, fourth-root scale, no
   seasonality, no backfill.
2. **`nfidd-ar6bf`** — AR(6) + backfill correction.
3. **`seabbs_bot-season`** — per-location climatology + AR(6) +
   backfill (the merged hub PR #79 design).
4. **`seabbs_bot-seasstack`** — pooled seasonal shape + log transform +
   Student-t intervals + AR-coefficient pooling (w=0.9), the
   leak-free full stack from `round2-stack`.
5. **`conformal-pooled`** — pooled seasonal + backfill + AR-coefficient
   pooling (w=0.3) + split-conformal intervals, window=208 weeks,
   width-scale=0.9: the "honest frontier" (best leak-free validation
   score recorded, 0.2730).

Plus, for external context, the hub's own **`hist-avg`** baseline
(historical-average model, pulled from the real hub clone at
`~/code/external/sismid-ili-forecasting-sandbox`).
`hist-avg` only has forecast files for 75 of the 87 test-season origin
dates in this hub clone (it is missing 12, mostly late in season 5), so
its numbers below are scored on that smaller, uneven set and are
reference-only, not a like-for-like comparison.

## Headline test results

| Model | Test mean WIS | Test SD | Cov 50% | Cov 90% | n tasks |
|---|---|---|---|---|---|
| **`seabbs_bot-seasstack`** | **0.4643** | 0.5895 | 0.540 | 0.891 | 3828 |
| `conformal-pooled` | 0.4806 | 0.6317 | 0.478 | 0.879 | 3828 |
| `seabbs_bot-season` | 0.5499 | 0.7913 | 0.380 | 0.724 | 3828 |
| `nfidd-ar6bf` | 0.6214 | 0.8423 | 0.409 | 0.754 | 3828 |
| `nfidd-ar6` | 0.6232 | 0.8547 | 0.420 | 0.757 | 3828 |
| `hist-avg` (external, 75/87 origins) | 0.9217 | 1.0045 | 0.358 | 0.830 | 3300 |

3828 = 87 test-season origin dates x 11 locations x 4 horizons.
`seasstack` is the best model on the held-out test seasons, narrowly
ahead of `conformal-pooled` — a reversal of the validation ranking (see
below).
All five of our models clear `hist-avg` comfortably.

## Validation vs test: does the ranking hold?

| Model | Val mean WIS | Val SD | Test mean WIS | Test SD | Val rank | Test rank |
|---|---|---|---|---|---|---|
| `conformal-pooled` (honest frontier) | **0.2730** | 0.3181 | 0.4806 | 0.6317 | **1** | 2 |
| `seabbs_bot-seasstack` (full stack) | 0.2891 | 0.2987 | **0.4643** | 0.5895 | 2 | **1** |
| `seabbs_bot-season` | 0.3004 | 0.3890 | 0.5499 | 0.7913 | 3 | 3 |
| `nfidd-ar6bf` | 0.3590 | 0.4521 | 0.6214 | 0.8423 | 4 | 4 |
| `nfidd-ar6` | 0.3684 | 0.4708 | 0.6232 | 0.8547 | 5 | 5 |

Validation numbers are quoted, not recomputed: `nfidd-ar6`/`nfidd-ar6bf`/
`seabbs_bot-season` from `experiments/simple-round/season/score.txt`;
`seasstack` from `experiments/simple-round/round2-stack/score.txt`
(leak-free winner); `conformal-pooled` from
`experiments/simple-round/conformal-pooled/score.txt` (window=208,
width=0.9 row).

Three of the five models keep their exact validation rank on test:
`nfidd-ar6` stays worst, `nfidd-ar6bf` stays fourth, `seabbs_bot-season`
stays third.
The top two swap.
On validation, `conformal-pooled` beat `seasstack` by 5.6% (0.2730 vs
0.2891).
On test, `seasstack` beats `conformal-pooled` by 3.4% (0.4643 vs
0.4806).
That is a genuine ranking reversal at the top of the leaderboard, not
noise from a rounding difference.

## Does the honest 0.2730 approach generalise, or does it overfit?

Partially.
It generalises in the sense that mattered most: `conformal-pooled` is
still the second-best model on test, a wide margin ahead of
`seabbs_bot-season`, `nfidd-ar6bf`, and `nfidd-ar6` — the split-conformal
calibration and the longer (208-week) AR window are real, transferable
wins, not artefacts of the two validation seasons.
It does **not** generalise in the narrower sense the validation
leaderboard implied: the claim that conformal-pooled specifically beats
the full stack does not survive contact with the test seasons.

Two structural reasons this is plausible, not just noise:

- `conformal-pooled`'s split-conformal calibration pool accumulates
  errors walk-forward from season 1 onward, so at the start of the test
  seasons it is calibrated on a pool built almost entirely from the two
  validation seasons' error distribution.
  If the test seasons' error distribution differs (see the season-5
  discussion below), that calibration is exactly the kind of thing that
  would transfer imperfectly.
- The two models were selected on `AR_ORDER`/`WINDOW_WEEKS`/`WIDTH_SCALE`/
  `POOL_WEIGHT` sweeps run only on 2 validation seasons each (see
  `round2-stack/score.txt`'s pool-weight sweep and
  `conformal-pooled/score.txt`'s window/width sweeps).
  A 2-season selection set is small enough that a ~5% validation gap
  between two similarly-built models is within the range a genuinely
  comparable model could lose back on 3 new seasons, which is what
  happened here.

Coverage calibration itself held up well for both: `seasstack` goes from
cov50/cov90 = 0.521/0.914 (validation) to 0.540/0.891 (test);
`conformal-pooled` goes from 0.498/0.907 (validation) to 0.478/0.879
(test).
Both stay close to nominal on test, so the ranking reversal is about
point-forecast/interval-width accuracy on the new seasons, not a
calibration breakdown.

## Backfill's validation edge nearly disappears on test

A smaller, second instance of the same lesson: on validation, the
backfill correction alone (`nfidd-ar6bf` vs `nfidd-ar6`) was worth 2.6%
(0.3684 to 0.3590).
On test it is worth 0.3% (0.6232 to 0.6214) — barely distinguishable
from noise given both models' test SD is around 0.85.
The climatology term, by contrast, transfers strongly: `seabbs_bot-season`
vs `nfidd-ar6bf` is an 11.5% cut on test (0.6214 to 0.5499), comparable
to its 16.3% validation cut (0.3590 to 0.3004).
Seasonality is the lever that reliably survives to held-out seasons;
the backfill correction's validation-measured gain looks partly like a
2-season-specific effect.

## Breakdown by horizon

Mean WIS by forecast horizon (weeks ahead), test seasons only:

| Model | h1 | h2 | h3 | h4 |
|---|---|---|---|---|
| `seabbs_bot-seasstack` | **0.3047** | **0.4338** | **0.5260** | **0.5926** |
| `conformal-pooled` | 0.3040 | 0.4533 | 0.5516 | 0.6137 |
| `seabbs_bot-season` | 0.3221 | 0.5098 | 0.6414 | 0.7263 |
| `nfidd-ar6bf` | 0.3369 | 0.5592 | 0.7316 | 0.8578 |
| `nfidd-ar6` | 0.3384 | 0.5619 | 0.7337 | 0.8589 |
| `hist-avg` | 0.8930 | 0.9069 | 0.9353 | 0.9514 |

Every model degrades monotonically from h1 to h4, as expected.
h4 is the hardest horizon for every model.
`seasstack` wins outright at every horizon except h1, where it is
essentially tied with `conformal-pooled` (0.3047 vs 0.3040, well inside
the noise floor); it pulls ahead at longer horizons — exactly where its
208-week AR window and Student-t/pooling combination were designed to
help.

## Breakdown by season

Mean WIS by test season:

| Model | Season 3 (2017/18) | Season 4 (2018/19) | Season 5 (2019/20) |
|---|---|---|---|
| `seabbs_bot-seasstack` | 0.5010 | **0.2914** | 0.6003 |
| `conformal-pooled` | 0.5192 | 0.3170 | 0.6057 |
| `seabbs_bot-season` | 0.6132 | 0.3241 | 0.7125 |
| `nfidd-ar6bf` | 0.5947 | 0.4265 | 0.8429 |
| `nfidd-ar6` | 0.5953 | 0.4285 | 0.8458 |
| `hist-avg` (n uneven, see caveats) | 0.8940 | 0.6707 | 1.3690 |

Season 4 (2018/19) is the easiest test season for every model by a
wide margin.
Season 5 (2019/20) is the hardest for every model, roughly double
season 4's mean WIS in every one of our five models.
`seasstack` still wins in the hardest season (0.6003 vs
`conformal-pooled`'s 0.6057, `season`'s 0.7125), so the best model does
not lose its edge where the going is hardest.
Season 5's forecast targets run from October 2019 to May 2020: its
h3/h4 targets for the last several origins land in March-May 2020, the
same window when COVID-19 changed US healthcare-seeking behaviour and
ILI surveillance patterns nationally.
That is a plausible, well-known confound for why season 5 is hardest
across every model here, including `hist-avg`; it is a property of the
held-out data, not something any of these designs could have
anticipated from the validation seasons.

## Breakdown by location

Mean WIS by location, test seasons only, sorted by the `seasstack`
column:

| Location | `seasstack` | `conformal-pooled` | `season` | `ar6bf` | `ar6` |
|---|---|---|---|---|---|
| HHS Region 9 | **0.2789** | 0.3244 | 0.3494 | 0.4449 | 0.4483 |
| HHS Region 5 | 0.2917 | 0.3241 | 0.4299 | 0.4571 | 0.4600 |
| HHS Region 1 | 0.3441 | 0.3190 | 0.4088 | 0.4501 | 0.4516 |
| US National | 0.3803 | 0.3977 | 0.4820 | 0.4921 | 0.4918 |
| HHS Region 3 | 0.3895 | 0.3823 | 0.4519 | 0.5271 | 0.5255 |
| HHS Region 8 | 0.4051 | 0.4552 | 0.5307 | 0.5137 | 0.5140 |
| HHS Region 10 | 0.4356 | 0.4121 | 0.4778 | 0.5870 | 0.5881 |
| HHS Region 4 | 0.5546 | 0.5882 | 0.6933 | 0.6890 | 0.6878 |
| HHS Region 7 | 0.6236 | 0.5459 | 0.5986 | 0.7135 | 0.7506 |
| HHS Region 2 | 0.6566 | 0.6694 | 0.7481 | 0.9471 | 0.9259 |
| **HHS Region 6** | **0.7469** | 0.8687 | 0.8786 | 1.0137 | 1.0116 |

HHS Region 6 is the single hardest location for every model, including
`hist-avg` (1.689) — consistent with the round-1/round-2 EDA finding
that Regions 2 and 6/9 are the most heavily revised and volatile
series.
`seasstack` wins outright in the hardest location (Region 6), and in
most others; `conformal-pooled` edges it in a handful of mid-table
locations (Regions 1, 3, 7, 10), so the top-two reversal is not driven
by one or two outlier locations, it is broad-based across the map.

## Honest caveats

- **Small selection set.**
  Model selection ranked five candidates on 2 validation seasons.
  This report is the direct demonstration of why that is a real risk:
  the top-two ranking flipped on 3 new seasons.
  Two validation seasons is not enough data to reliably resolve a ~5%
  WIS gap between similarly-constructed models; it is enough to
  confidently rule out the clearly worse candidates (`nfidd-ar6`,
  `nfidd-ar6bf`), which is exactly what held up here.
- **Season 5 is unusual.**
  Its forecast targets extend into the early COVID-19 period
  (March-May 2020), a documented disruptor of US ILI surveillance.
  It is the hardest season for every model tested, including the
  hub's own `hist-avg`, so this is very unlikely to be specific to our
  designs, but it does mean the "test" seasons are not a uniformly
  representative future.
- **`hist-avg` is a partial, external reference only.**
  It is missing forecast files for 12 of the 87 test-season origin
  dates in the local hub clone (`~/code/external/
  sismid-ili-forecasting-sandbox`), concentrated late in season 5, so
  its by-season/by-location numbers are not computed on the same task
  set as our five models and should not be read as a precise
  apples-to-apples comparison, only as a sanity check that our models
  clear a real external baseline.
- **Task-level variance is large.**
  Every model's test SD is comparable to or larger than its mean WIS
  (e.g. `seasstack` 0.4643 mean vs 0.5895 SD).
  Individual-task variance in wILI% forecasting is high; the mean WIS
  differences reported here are the right point estimates, but they
  are not accompanied by formal significance tests, and some of the
  closer comparisons (h1 seasstack-vs-conformal-pooled, several
  mid-table locations) are well within a plausible noise band.
- **Backfill profiles rebuilt for this report, not re-fit.**
  Two models' backfill profiles were rebuilt with the canonical
  per-origin builder instead of reusing the originally-submitted
  fixed-cutoff profile (see "Leak-free discipline" above).
  This is a strictly more-correct construction, not a re-tuning: no
  new hyperparameter was chosen using test-season information, and the
  original fixed cutoff was already provably leak-free for these test
  origins, so this does not affect the experimental-integrity
  guarantee, only the exact numeric profile values used.

## Verdict

The search's central conclusion survives contact with held-out data:
seasonality is the dominant lever (season model already cuts test WIS
~12% below AR6+backfill, on top of that lever's ~16% validation cut),
and the full stack (pooled seasonal shape + log + Student-t intervals +
AR pooling) is the strongest model actually tested end-to-end on the
held-out seasons, beating the plain AR(6) baseline by 25.5% and the
hub's own `hist-avg` by round 50%.
The one place the validation-era conclusions do not fully hold is the
narrow claim that split-conformal calibration with a longer AR window
(`conformal-pooled`) is better than the full stack: that specific
ranking flips on test, though both remain far ahead of everything
simpler.
Given the two are close on test (0.4643 vs 0.4806, both well within a
season's worth of noise of each other) and `seasstack` is simpler
(no walk-forward calibration state to maintain), `seasstack`
is the more defensible pick as *the* headline model if only one is to
be highlighted going forward — but this report's job is to record what
happened on test, not to re-select, and no hub submission is being
opened based on it (hub submissions remain paused per Sam's 2026-07-17
instruction, see `submissions/README.md`).
