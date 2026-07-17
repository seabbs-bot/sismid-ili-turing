# seabbs_bot-seasstack

Merged as [reichlab PR #80](https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/80).

> **LEAKAGE NOTE.** The `generate.jl`/`generate_forecasts.jl` archived in this
> directory are the ORIGINAL build, which estimated the pooled seasonal and
> backfill profiles once from all `season_year <= 2016` data — for validation
> origins that leaked those seasons' own future weeks. The advertised 0.2601 is
> leak-inflated. The **honest leak-free** score for this exact model (profiles
> rebuilt per-origin from strictly-prior data) is **0.2891** (cov50 0.521,
> cov90 0.914) — still a genuine −3.8% over the clean `season` model (0.3004),
> so the log + Student-t + AR-pooling stack holds up. The leak-free rebuild
> lives in `experiments/simple-round/round2-stack/` (see its `score.txt`). A
> clean submission driver is being centralised via the leak-free `src/`
> profile functions; do not resubmit from the drivers in this dir as-is.

## Method

The pooled-seasonal + backfill core, with three stacked wins:

- Pooled week-of-season climatology (shared shape, pooled deviations across
  all 11 locations) on a **log** scale.
- Non-monotonic delay-indexed **backfill** correction (per location, additive,
  median, 8-week window).
- Per-location **AR(6)** on the deseasonalised residual, partially **pooled**
  toward an all-locations anchor (blend weight w=0.9).
- **Student-t(df=10)** innovations, variance-matched then scaled 1.4, to fix
  the under-coverage of raw Gaussian AR intervals.

## Files

- `generate.jl` — the round-2 stack experiment (scores the ablation sweep on
  the validation seasons; see `score.txt`). Source of the winning combo.
- `generate_forecasts.jl` — the submission driver. Runs the winning combo
  across all five seasons and writes the hubverse submission:
  `julia --project=<repo> generate_forecasts.jl <hub_path>`. Each test-season
  origin is a per-week vintage fit (data up to that origin only), never used
  for selection.
- `score.txt` — the validation ablation table.

## Regenerate

```
julia --project=. submissions/seabbs_bot-seasstack/generate_forecasts.jl <hub_clone>
```

then prune to the hub's allowed origin dates (drops the 6 post-2020-03-21
dates outside the hub's task set) and validate before opening the PR.
