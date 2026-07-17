# Weighted interval score (WIS) for quantile forecasts, and forecast-table
# scoring utilities. See docs/contracts.md for the forecast/truth table
# schemas and the WIS formula.

using DataFrames
using Statistics
using ScoringRules

"""
    wis(observation, values, levels)

Weighted interval score for one forecast task.

`levels` are quantile levels symmetric about the median (e.g.
`QUANTILE_LEVELS`), with corresponding predicted `values`, scored against a
scalar `observation`. Symmetric pairs `(a, 1 - a)` for `a < 0.5` form the `K`
central prediction intervals; each is scored with
`ScoringRules.interval_score` and combined with the median absolute error
following Bracher et al. (2021):

    WIS = (1 / (K + 0.5)) * (0.5 * |y - median| + Σ_k (α_k / 2) * IS_{α_k})

where `α_k = 2a` is the miscoverage of the interval formed by level `a` and
its counterpart `1 - a`.

Returns a named tuple `(wis, dispersion, overprediction, underprediction)`;
the three components sum to `wis` and decompose it following the same
reference (interval width vs. one-sided misses above/below).
"""
function wis(observation::Real, values::AbstractVector{<:Real},
        levels::AbstractVector{<:Real})
    length(values) == length(levels) ||
        throw(DimensionMismatch(
            "values and levels must have the same length"))

    lv = collect(Float64, levels)
    val = collect(Float64, values)
    tol = 1e-8

    median_idx = findfirst(a -> abs(a - 0.5) < tol, lv)
    median_idx === nothing &&
        throw(ArgumentError("levels must include the median (0.5)"))
    median = val[median_idx]

    lower_levels = filter(a -> a < 0.5 - tol, lv)
    K = length(lower_levels)
    K == 0 && throw(ArgumentError(
        "levels must include at least one central interval below 0.5"))

    is_sum = 0.0
    dispersion = 0.0
    overprediction = 0.0
    underprediction = 0.0
    for a in lower_levels
        lower_idx = findfirst(x -> abs(x - a) < tol, lv)
        upper_idx = findfirst(x -> abs(x - (1 - a)) < tol, lv)
        upper_idx === nothing && throw(ArgumentError(
            "level $a has no symmetric upper counterpart $(1 - a)"))
        lower = val[lower_idx]
        upper = val[upper_idx]
        alpha_k = 2 * a
        coverage = 1 - alpha_k

        is_k = interval_score(lower, upper, observation, coverage)
        is_sum += (alpha_k / 2) * is_k

        dispersion += (alpha_k / 2) * (upper - lower)
        overprediction += max(lower - observation, 0.0)
        underprediction += max(observation - upper, 0.0)
    end

    denom = K + 0.5
    median_term = 0.5 * abs(observation - median)
    wis_total = (median_term + is_sum) / denom

    dispersion /= denom
    overprediction = (overprediction + 0.5 * max(median - observation, 0.0)) /
                      denom
    underprediction = (underprediction + 0.5 * max(observation - median, 0.0)) /
                        denom

    return (wis = wis_total, dispersion = dispersion,
            overprediction = overprediction, underprediction = underprediction)
end

"""
    score_forecasts(forecast_df, truth_df; scale=:natural)

Score a forecast table (schema in docs/contracts.md) against `truth_df`
(columns `location, target_end_date, value`). Returns one row per forecast
task (`model_id, location, origin_date, horizon, target_end_date`) with the
WIS and its decomposition.

`scale = :natural` scores the wILI percentages as submitted -- this is the
operational target for tuning and model selection. `scale = :log`
transforms both the forecast quantiles and the observation with `log1p`
before scoring; this is report-only and must never be optimised against.
"""
function score_forecasts(forecast_df::DataFrame, truth_df::DataFrame;
        scale::Symbol = :natural)
    scale in (:natural, :log) ||
        throw(ArgumentError("scale must be :natural or :log, got $scale"))

    joined = innerjoin(forecast_df, truth_df,
        on = [:location, :target_end_date], renamecols = "" => "_truth")

    task_cols = [:model_id, :location, :origin_date, :horizon,
                 :target_end_date]

    scored = combine(groupby(joined, task_cols)) do sdf
        values = sdf.value
        levels = sdf.output_type_id
        observation = sdf.value_truth[1]
        if scale == :log
            values = log1p.(values)
            observation = log1p(observation)
        end
        result = wis(observation, values, levels)
        (wis = result.wis, dispersion = result.dispersion,
         overprediction = result.overprediction,
         underprediction = result.underprediction,
         n_quantiles = length(values))
    end
    scored.scale = fill(scale, nrow(scored))
    return scored
end

"""
    wis_summary(scored_df)

Per-`model_id` summary of a scored forecast table (as returned by
[`score_forecasts`](@ref)): mean WIS, its standard deviation across tasks
(origin dates × locations × horizons), the mean component decomposition,
and the task count.

Both moments matter: the mean ranks predictive accuracy, and the SD is the
overfitting guard -- a model with a low mean but a high SD has bought good
average performance with instability across tasks.
"""
function wis_summary(scored_df::DataFrame)
    combine(groupby(scored_df, :model_id),
        :wis => mean => :mean_wis,
        :wis => std => :sd_wis,
        :dispersion => mean => :mean_dispersion,
        :overprediction => mean => :mean_overprediction,
        :underprediction => mean => :mean_underprediction,
        nrow => :n_tasks,
    )
end

"""
    compare_scales(forecast_df, truth_df)

Score `forecast_df` against `truth_df` on both the natural and `log1p`
scales and return `(natural, log, comparison)`: the two `wis_summary`
tables plus a per-model comparison table with mean/SD on each scale, the
rank on each scale, and a `rank_changed` flag.

Log-scale scoring is report-only -- use `comparison` to check whether model
ranking would differ between scales, never to select or tune a model.
"""
function compare_scales(forecast_df::DataFrame, truth_df::DataFrame)
    natural = wis_summary(score_forecasts(forecast_df, truth_df;
                                           scale = :natural))
    logscale = wis_summary(score_forecasts(forecast_df, truth_df;
                                            scale = :log))
    sort!(natural, :mean_wis)
    sort!(logscale, :mean_wis)
    natural.rank_natural = collect(1:nrow(natural))
    logscale.rank_log = collect(1:nrow(logscale))

    natural_sel = rename(natural[:, [:model_id, :mean_wis, :sd_wis,
                                      :rank_natural]],
        :mean_wis => :mean_wis_natural, :sd_wis => :sd_wis_natural)
    logscale_sel = rename(logscale[:, [:model_id, :mean_wis, :sd_wis,
                                        :rank_log]],
        :mean_wis => :mean_wis_log, :sd_wis => :sd_wis_log)
    comparison = innerjoin(natural_sel, logscale_sel, on = :model_id)
    comparison.rank_changed = comparison.rank_natural .!= comparison.rank_log

    return (natural = natural, log = logscale, comparison = comparison)
end
