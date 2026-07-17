# seabbs_bot-seasstack

Round-3 hub submission ([reichlab PR #80](https://github.com/reichlab/sismid-ili-forecasting-sandbox/pull/80)).
Validation WIS **0.2601** (50% coverage 0.565, 90% coverage 0.943), selected
on the validation seasons (2015/16, 2016/17) only.

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
