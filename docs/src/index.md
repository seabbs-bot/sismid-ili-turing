# SismidILITuring

A custom infectious-disease forecasting model, built in Julia with Turing.jl, submitted to the online [`reichlab/sismid-ili-forecasting-sandbox`](https://github.com/reichlab/sismid-ili-forecasting-sandbox) hub.
The model is a joint fit across all 11 HHS regions, with partially-pooled seasonality, autoregressive noise, and a non-monotonic reporting-backfill component, searched and selected on the two validation seasons before testing on three held-out seasons.
See [Brief](project/brief.md) for the full requirements and [Plan](project/plan.md) for the design and search strategy.

## Where to look

- **[Project](project/brief.md)**: the brief (what to build and how to work),
  the plan (design and search strategy), infrastructure status, the interface
  contracts shared between modules, and the steer log (a running record of
  guidance and actions taken).
- **[EDA](eda/00-index.md)**: exploratory analysis of the wILI series,
  backfill/revision behaviour, seasonality, cross-location structure, and
  autocorrelation, written to inform the model design.
- **[Reports](reports/README.md)**: one report per implement-and-review loop
  of the model search, recording what was tried, the WIS scores, and the
  decision taken forward.
- **API**: reference documentation for the `SismidILITuring` module,
  generated from docstrings.
  This section appears once `src/SismidILITuring.jl` exists and loads
  cleanly; see [Infrastructure](project/infrastructure.md) for the current
  build status.

## Experimental integrity

Tune and select only on the validation seasons (2015/16, 2016/17); the three testing seasons (2017/18–2019/20) are held out until the finalists are locked.
See [Brief § Experimental integrity](project/brief.md#experimental-integrity-do-not-cheat) for the full constraint.
