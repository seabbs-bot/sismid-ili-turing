# simple-round experiment index

Snapshot as of 2026-07-17 (this tree is under active, concurrent
development -- statuses below reflect what was on disk at audit time;
re-check before relying on a number for a new decision).

Every experiment here is a "light + analytic" (OLS/ridge, no Turing)
candidate for the wide simple-model round, scored on `mean_wis` against
the validation seasons (1, 2) only, per `docs/brief.md`/`docs/
contracts.md` experimental integrity. **Leak status** below is about
one specific, recurring bug (see `docs/lessons.md`, `submissions/
README.md`'s "Hub submissions PAUSED" note, and `experiments/
simple-round/round2-stack/score.txt`'s "LEAKAGE FIX" section for the
full writeup): a pooled seasonal climatology (`build_seasonal_profile`)
and/or an empirical backfill/revision profile (`build_revision_profile`)
built **ONCE** from a fixed `season_year <= 2016` cutoff (or an
equivalent `max_season_year` constant), then reused unchanged across
every cross-validation split. Because `VALIDATION_SEASONS = (1, 2)` and
season 2 spans Oct 2016-Sep 2017 (`season_year == 2016`), that fixed
cutoff includes ALL of validation season 2's own data -- so a split
early in season 2 gets a profile built partly from that same season's
own future weeks. The fix (`experiments/simple-round/round2-stack/
generate.jl`, now canonicalised in `src/seasonal.jl`) rebuilds both
profiles **per split**, restricted to `hist`/`versions` rows strictly
before that split's own `forecast_origin`.

- **LEAK-FREE**: both profiles (where present) are rebuilt per split
  from `forecast_origin`.
- **MIXED**: the seasonal/climatology term is correctly per-origin, but
  the backfill/revision profile is still built once from a fixed
  cutoff.
- **LEAKY**: at least one profile is built once from a fixed cutoff and
  reused across every split.
- **N/A**: this experiment has no seasonal-profile or backfill-profile
  component at all (a different lever -- AR order, transform, pooling
  weight, differencing, etc.).

## LEAK-FREE

| Experiment | Tests | Headline WIS |
|---|---|---|
| `round2-stack` | log+Student-t+AR-pool(w=0.9) stack on the seasonal core | **0.2891** (honest; 0.2601 was the pre-fix leaky number) |
| `mech` | susceptible-depletion + Rt term on top of round2-stack (via direct `include`) | 0.2891 (mechanistic term: negative result) |

## MIXED (seasonal term clean, backfill profile leaky)

| Experiment | Tests | Headline WIS |
|---|---|---|
| `season` | per-location climatology + backfill (merged hub PR #79) | 0.3004 -- **flag for Sam: submissions/README.md calls this "clean" but its `build_revision_profile` is NOT per-split; only `build_climatology` is** |
| `conformal` | split-conformal intervals on the climatology model | 0.2917 |
| `conformal-pooled` | conformal + pooled AR residual | 0.287 |
| `severity2` | severity-scaled climatology, lambda sweep | 0.3005 |
| `seasonens` | ensemble of climatology-family members | 0.2748 (ampbf member) |
| `robustclim` | robust/outlier-resistant climatology variant | no score.txt yet |

## IN-FLIGHT (leak-free rewrite mid-edit at audit time -- do not trust either score.txt until fixed)

| Experiment | Tests | Note |
|---|---|---|
| `longhz` | long-horizon (h=3,4) momentum/damped-trend fixes on round2-stack | `build_seasonal_profile` was just rewritten to the leak-free 2-arg (`hist, forecast_origin`) signature, but its two call sites (`main()`, ~line 780 and ~933) still pass the old `max_season_year=` keyword -- as committed at audit time this errors (`MethodError`) if re-run. Both `score.txt` and `score-leaky.txt` are present from before this edit. |

## LEAKY

| Experiment | Tests | Headline WIS (against the leaky profile) |
|---|---|---|
| `adaptive` | within-season adaptive AR (backfill leak only, no seasonal) | 0.3518 |
| `ar-order` | AR order sweep + backfill (backfill leak only) | 0.3518 (order 12) |
| `backfill` | the backfill correction itself, mode/window sweep | 0.3684 |
| `calib` | interval-width calibration on the (pre-fix) round2-stack base | 0.2601 -> 0.2566 (base model is the LEAKY 0.2601 reference) |
| `cluster` | location-clustered seasonal profiles, K sweep | 0.2709 (K=11) |
| `combo` | grid search over AR+backfill+seasonal combination | grid-search best cell (`search_grid.jl`; backfill leak confirmed) |
| `dsb` | differencing-order sweep + backfill | 0.2975 (drop-differencing) |
| `ensemble` | ensembles several backfill-leaky members | 0.2902 (ens-mean) |
| `ets` | ETS/exponential-smoothing alternatives + backfill | 0.3684 (plain AR6, no backfill, this run) |
| `features` | extra AR features/regularisation + backfill | 0.2917 |
| `full` | "full stack" seasonal + backfill + AR order | 0.2997 (order 6) |
| `grid` | grid-search harness underlying `combo` | no score.txt yet (backfill leak confirmed in `search_grid.jl`) |
| `horizon` | per-horizon interval width scaling + backfill | 0.2415 (width=0.6) |
| `intervals` | interval scheme (Gaussian/Student-t) + backfill | Student-t(df=10) winner |
| `locallevel` | damped local-level residual dynamics | 0.2781 (sanity reproduction of the LEAKY round-1 headline) |
| `nowcast` | nowcast-uncertainty propagation into backfill | winner "D" (`search_grid.jl`-based) |
| `seasoncombo` | round-1 winner: pooled seasonal + AR(6) + backfill | 0.2781 -- **the LEAKY round-1 headline** (round2-stack's honest, leak-free rescore of this same core is 0.3056) |
| `seasondrift` | seasonal-profile decay/drift sweep | 0.2781 (sanity reproduction) |
| `seasonpool` | AR(6) coefficient pooling weight sweep + backfill (no pooled-seasonal profile of its own) | 0.3684 (baseline-ar6 shown) |
| `seasonpool2` | seasonal amplitude-shrinkage sweep | 0.2748 (shrink=1.0) |
| `sesresid` | SES residual dynamics on the seasonal core | 0.2781 (sanity reproduction) |
| `smoother` | seasonal-profile smoothing window/jitter sweep | 0.285 |
| `tvar` | time-varying AR (discount/window) + backfill | 0.359 (ties static baseline; negative result) |
| `tvar-season` | time-varying AR on the seasonal core | 0.2866 (static still best) |
| `tvpool` | AR-pooling weight sweep on the seasonal+backfill core | cites 0.2601, the LEAKY round-2 reference, as its own baseline |
| `round3` | round-3 candidate screening | cites 0.2601, the LEAKY round-2 reference, as its own baseline |

## N/A (no seasonal or backfill profile in this sweep)

| Experiment | Tests |
|---|---|
| `pool` | AR(6) coefficient partial-pooling weight sweep (raw, pre-seasonal) |
| `transform` | modelling-scale transform choice (log / fourthroot / logit) |
| `var` | VAR model order + ridge sweep |
| `mix` | not examined in this pass -- no `build_seasonal_profile`/`build_revision_profile`/`build_climatology` reference found; check before citing a WIS from it |

## Bottom line

The only two experiments whose reported WIS is currently honest
end-to-end are `round2-stack` (0.2891) and `mech` (0.2891, no
improvement from the mechanistic term). Every LEAKY/MIXED-with-leaky-
backfill number above is optimistic relative to what a true per-origin
re-run would show (see `round2-stack/score.txt`'s own before/after:
0.2601 leaky -> 0.2891 leak-free, roughly +11% honest degradation on
that one stack) -- do not rank candidates against each other, or pick a
round submission, using the numbers in the LEAKY/MIXED tables without
first re-scoring on the leak-free profile
(`src/seasonal.jl`'s `build_seasonal_profile`/`build_revision_profile`/
`apply_backfill_correction!`, now the canonical version to `include`).
