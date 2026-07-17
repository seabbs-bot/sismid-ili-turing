# Tree-search round-runner for the ILI model search.
#
# Scores a set of candidate models on the VALIDATION seasons (2015/16,
# 2016/17) resiliently and (optionally) with a per-candidate timeout, so a
# round never blocks on one slow or failing fit. Produces a ranking table
# (mean WIS then WIS SD, the overfitting guard), the Bayesian-workflow
# summaries, and a report under reports/<roundname>.md.
#
# Experimental integrity (docs/brief.md, docs/contracts.md): tuning and
# selection use ONLY the two validation seasons. The truth loader here
# refuses to load any test-season (2017/18-2019/20) oracle values.
#
# Usage (see experiments/README.md for adding candidates):
#   julia --project=. experiments/run_round.jl        # runs the base smoke
# or, from an interactive session / another script:
#   include("experiments/run_round.jl")
#   run_round(candidates, "02-my-round")
#
# A candidate is a `(name, build_model, project)` triple (a NamedTuple or a
# plain Tuple). `build_model(data; transform)` returns a Turing model in the
# `base_model` family; `project(draw, data, horizons)` is its forecast
# projector (see src/forecast.jl `base_project`, or a `project_vN`).

# Load the package. It is normally loadable as `SismidILITuring`; if that
# fails (e.g. a half-assembled checkout) fall back to include-ing the
# component src files into Main, which share scope the same way.
const _PKG_ROOT = normpath(joinpath(@__DIR__, ".."))
const _USING_PACKAGE = try
    @eval using SismidILITuring
    true
catch err
    @warn "`using SismidILITuring` failed; include-ing src/ files instead" err
    for f in ("core.jl", "data.jl", "model.jl", "scoring.jl", "forecast.jl",
              "hubio.jl", "inference.jl", "diagnostics.jl", "pipeline.jl")
        include(joinpath(_PKG_ROOT, "src", f))
    end
    false
end

using CSV
using DataFrames
using Dates
using Statistics
using Printf

# --- integrity: validation-only truth ---------------------------------------

"""Path to the hub clone whose oracle output is the scoring truth."""
const HUB_PATH =
    "/home/seabbs/code/external/sismid-ili-forecasting-sandbox"

"""
Season-start years of the VALIDATION seasons only: 2015 -> 2015/16 and
2016 -> 2016/17. The test seasons (2017/18-2019/20) are deliberately absent
so their settled truth is never loaded (docs/brief.md experimental
integrity).
"""
const VALIDATION_SEASON_YEARS = (2015, 2016)

"""
    _season_start_year(d::Date) -> Int

Season-start calendar year of `d`, seasons taken to start in October (a
Saturday on/after 1 October). Kept local so the truth loader does not depend
on package internals; matches `src/data.jl`'s `season_year` on the weekly
Saturday grid the oracle uses.
"""
_season_start_year(d::Date)::Int = month(d) >= 10 ? year(d) : year(d) - 1

"""
    load_validation_truth(; hub_path=HUB_PATH,
                          season_years=VALIDATION_SEASON_YEARS)::DataFrame

Load the settled hub oracle output and return the scoring truth for the
VALIDATION seasons only, as a `DataFrame` with columns `location`,
`target_end_date`, `value` (the schema `score_forecasts` expects).

Reads `target-data/oracle-output.csv` from the hub clone (columns
`location, target_end_date, target, output_type, output_type_id,
oracle_value`; `output_type_id` is `NA` for these point oracle values). Rows
are filtered to `season_years` by `_season_start_year(target_end_date)`, so
test-season truth is never returned. Throws if any test-season year slips
through (a hard integrity guard).
"""
function load_validation_truth(;
    hub_path::AbstractString=HUB_PATH,
    season_years=VALIDATION_SEASON_YEARS,
)::DataFrame
    path = joinpath(hub_path, "target-data", "oracle-output.csv")
    isfile(path) || error("oracle output not found at $path")
    raw = CSV.read(path, DataFrame; missingstring="NA")
    ends = Date.(raw.target_end_date)
    keep = [_season_start_year(d) in season_years for d in ends]
    truth = DataFrame(
        location=String.(raw.location[keep]),
        target_end_date=ends[keep],
        value=Float64.(raw.oracle_value[keep]),
    )
    got_years = sort(unique(_season_start_year.(truth.target_end_date)))
    all(y -> y in season_years, got_years) || error(
        "integrity violation: truth contains non-validation seasons $got_years")
    return truth
end

# --- candidate normalisation -------------------------------------------------

"""
    _candidate_parts(cand) -> (name, build_model, project)

Accept a candidate as a `NamedTuple` (`(; name, build_model, project)`) or a
plain `Tuple`/`Vector` in that order, and return the three parts. Kept
permissive so candidate lists read naturally either way.
"""
function _candidate_parts(cand)
    if cand isa NamedTuple
        return (String(cand.name), cand.build_model, cand.project)
    end
    return (String(cand[1]), cand[2], cand[3])
end

# --- diagnostics predict -----------------------------------------------------

_get(draw, k::Symbol) =
    draw isa AbstractDict ? draw[k] : getproperty(draw, k)

"""
    latent_predict(draw, data)

Map one draw's latent `(T x L)` path (modelling scale) back to the natural
wILI% scale, for the Bayesian-workflow checks. Works for both prior draws
(`model()` returns the `latent` field directly) and posterior
generated-quantities draws (`generated_draws`), so prior/posterior
predictive checks and residual autocorrelation are computed against the
model's own latent field rather than the climatology placeholder in
`src/diagnostics.jl`.
"""
latent_predict(draw, data) = from_scale.(_get(draw, :latent), data.transform)

# --- per-candidate scoring ---------------------------------------------------

"""
    run_candidate(name, build_model, project; seasons=[1, 2], ndraws=400,
                  Dmax=12, transform=:log1p, truth=nothing, max_splits=nothing,
                  diag_ndraws=100, window_weeks=104)

Fit, forecast, score and Bayesian-check one candidate on the validation
seasons, returning a results `NamedTuple`. Fast Pathfinder screening
(`fit_pathfinder`) is used, matching the search plan (docs/plan.md).

For each `training_splits(season)` split (capped at `max_splits` per season
when set) it: builds the `ModelData` (`build_model_data` with `Dmax`,
`transform`, `window_weeks`), fits `build_model(data; transform)`, forecasts
the hub quantile table from the fit's `generated_draws` via `project`, and
runs `bayesian_checks`. Any single split that throws is caught and recorded
(it does not sink the whole candidate).

Scoring uses `compare_scales`, giving mean and SD WIS on BOTH the natural
scale (the selection metric and its overfitting guard) and the report-only
log scale. The returned `NamedTuple` carries:

- `status` -- `:ok` if it produced forecasts, else `:failed`.
- `error` -- captured error message (`nothing` when `:ok`).
- `mean_wis`, `sd_wis` -- natural-scale WIS mean and SD (selection metrics).
- `log_mean_wis`, `log_sd_wis` -- log-scale WIS mean and SD (report only).
- `coverage50`, `coverage90` -- posterior-predictive empirical coverage.
- `prior_frac_outside`, `prior_frac_nonfinite` -- prior-predictive health.
- `max_abs_acf1` -- largest |lag-1 residual autocorrelation| over locations.
- `n_tasks`, `n_splits`, `n_failed_splits` -- counts.
- `forecast_df`, `truth` -- the pooled forecast table and truth used.
"""
function run_candidate(
    name, build_model, project;
    seasons=[1, 2],
    ndraws::Int=400,
    Dmax::Int=12,
    transform::Symbol=:log1p,
    truth::Union{Nothing,DataFrame}=nothing,
    max_splits::Union{Nothing,Int}=nothing,
    diag_ndraws::Int=100,
    window_weeks::Int=104,
)
    truth === nothing && (truth = load_validation_truth())
    tables = DataFrame[]
    coverages50 = Float64[]
    coverages90 = Float64[]
    prior_outside = Float64[]
    prior_nonfinite = Float64[]
    acf1s = Float64[]
    n_splits = 0
    n_failed_splits = 0
    split_errors = String[]

    for season in seasons
        splits = training_splits(season)
        max_splits !== nothing && (splits = splits[1:min(max_splits, end)])
        for split in splits
            n_splits += 1
            try
                data = build_model_data(split; Dmax=Dmax, transform=transform,
                                        window_weeks=window_weeks)
                model = build_model(data; transform=transform)
                fit = fit_pathfinder(model; ndraws=ndraws)
                gdraws = generated_draws(model, fit)
                fq = forecast_quantiles(gdraws, data, name; project=project)
                push!(tables, fq)
                bc = bayesian_checks(fit, model, data; ndraws=diag_ndraws,
                                     draws=gdraws, predict=latent_predict)
                push!(coverages50, bc.posterior.calibration.coverage50)
                push!(coverages90, bc.posterior.calibration.coverage90)
                push!(prior_outside,
                      bc.prior.summary.frac_outside_plausible_range)
                push!(prior_nonfinite, bc.prior.summary.frac_nonfinite)
                acf = bc.residuals.autocorrelation
                if nrow(acf) > 0
                    lag1 = acf[acf.lag .== 1, :acf]
                    isempty(lag1) || push!(acf1s, maximum(abs, lag1))
                end
            catch err
                n_failed_splits += 1
                push!(split_errors,
                      "split $(n_splits): " * _short_error(err))
            end
        end
    end

    if isempty(tables)
        msg = isempty(split_errors) ? "no splits produced forecasts" :
              join(split_errors, "; ")
        return _failed_result(name, msg; n_splits=n_splits,
                              n_failed_splits=n_failed_splits)
    end

    forecast_df = reduce(vcat, tables)
    cmp = compare_scales(forecast_df, truth)
    nat = cmp.natural[cmp.natural.model_id .== name, :]
    lg = cmp.log[cmp.log.model_id .== name, :]

    return (
        name=name,
        status=:ok,  # produced forecasts; partial split failures still :ok
        error=isempty(split_errors) ? nothing : join(split_errors, "; "),
        mean_wis=nat.mean_wis[1],
        sd_wis=nat.sd_wis[1],
        log_mean_wis=lg.mean_wis[1],
        log_sd_wis=lg.sd_wis[1],
        coverage50=_meanor(coverages50),
        coverage90=_meanor(coverages90),
        prior_frac_outside=_meanor(prior_outside),
        prior_frac_nonfinite=_meanor(prior_nonfinite),
        max_abs_acf1=_meanor(acf1s),
        n_tasks=Int(nat.n_tasks[1]),
        n_splits=n_splits,
        n_failed_splits=n_failed_splits,
        forecast_df=forecast_df,
        truth=truth,
    )
end

_meanor(v) = isempty(v) ? NaN : mean(filter(isfinite, v))

function _short_error(err)
    s = sprint(showerror, err)
    return length(s) > 300 ? s[1:300] * "..." : s
end

"""Build a `:failed` results NamedTuple sharing `run_candidate`'s schema."""
function _failed_result(name, msg; n_splits=0, n_failed_splits=0)
    return (
        name=name, status=:failed, error=msg,
        mean_wis=NaN, sd_wis=NaN, log_mean_wis=NaN, log_sd_wis=NaN,
        coverage50=NaN, coverage90=NaN,
        prior_frac_outside=NaN, prior_frac_nonfinite=NaN, max_abs_acf1=NaN,
        n_tasks=0, n_splits=n_splits, n_failed_splits=n_failed_splits,
        forecast_df=nothing, truth=nothing,
    )
end

# --- resilient timeout -------------------------------------------------------

"""
    _with_timeout(f, timeout) -> (:done, value) | (:timed_out, nothing)

Run `f()` on a spawned task and wait at most `timeout` seconds for it. If it
finishes in time, return its value (rethrowing any error it raised). If it
overruns, return `(:timed_out, nothing)` and abandon the task rather than
blocking the round. `timeout === nothing` runs `f()` inline with no limit.

The timeout is only truly pre-emptive when Julia has more than one thread
(`julia -t auto`): a compute-bound fit on a single thread cannot be
interrupted mid-step, so the wait resolves once it yields. It still protects
the round from an outright hang.
"""
function _with_timeout(f, timeout)
    timeout === nothing && return (:done, f())
    result = Ref{Any}(nothing)
    errored = Ref{Any}(nothing)
    finished = Ref(false)
    task = Threads.@spawn begin
        try
            result[] = f()
        catch e
            errored[] = e
        finally
            finished[] = true
        end
    end
    outcome = timedwait(() -> finished[], Float64(timeout); pollint=0.5)
    if outcome == :timed_out
        return (:timed_out, nothing)
    end
    errored[] !== nothing && throw(errored[])
    return (:done, result[])
end

# --- round orchestration -----------------------------------------------------

"""
    run_round(candidates::Vector, roundname; seasons=[1, 2], ndraws=400,
              Dmax=12, transform=:log1p, max_splits=nothing, timeout=nothing,
              slow_threshold=600, reports_dir=<repo>/reports, hub_path=HUB_PATH,
              parent="base", write=true)

Run every candidate in `candidates` RESILIENTLY and write a report to
`reports/<roundname>.md`.

Each candidate is wrapped in `_with_timeout` and a `try/catch`, so one crash
or hang cannot block the others: a failure is recorded with `status=:failed`
and its error rather than dropped, and a run exceeding `slow_threshold`
seconds (or `timeout`) is flagged `:slow`. We fix failing/complex models, so
they are surfaced in a "candidates needing a fix" section, never silently
abandoned.

Results are ranked by mean WIS then WIS SD (the overfitting guard). The
round also compares natural-scale and log-scale WIS ranks across candidates
and flags any rank disagreement. Returns a `NamedTuple`
`(; ranking, results, disagreements, report_path)`.
"""
function run_round(
    candidates::Vector, roundname::AbstractString;
    seasons=[1, 2],
    ndraws::Int=400,
    Dmax::Int=12,
    transform::Symbol=:log1p,
    max_splits::Union{Nothing,Int}=nothing,
    diag_ndraws::Int=100,
    timeout::Union{Nothing,Real}=nothing,
    slow_threshold::Real=600,
    reports_dir::AbstractString=joinpath(_PKG_ROOT, "reports"),
    hub_path::AbstractString=HUB_PATH,
    parent::AbstractString="base",
    write::Bool=true,
)
    truth = load_validation_truth(; hub_path=hub_path)
    results = NamedTuple[]

    for cand in candidates
        name, build_model, project = _candidate_parts(cand)
        @info "running candidate" name
        t0 = time()
        local res
        try
            outcome, value = _with_timeout(timeout) do
                run_candidate(name, build_model, project; seasons=seasons,
                              ndraws=ndraws, Dmax=Dmax, transform=transform,
                              truth=truth, max_splits=max_splits,
                              diag_ndraws=diag_ndraws)
            end
            elapsed = time() - t0
            if outcome == :timed_out
                res = merge(
                    _failed_result(name,
                        "timed out after $(timeout)s (status :slow)"),
                    (status=:slow, elapsed=elapsed),
                )
            else
                status = value.status
                status == :ok && elapsed > slow_threshold && (status = :slow)
                res = merge(value, (status=status, elapsed=elapsed))
            end
        catch err
            elapsed = time() - t0
            res = merge(_failed_result(name, _short_error(err)),
                        (elapsed=elapsed,))
            @warn "candidate crashed" name err
        end
        push!(results, res)
    end

    ranking = _ranking_table(results)
    disagreements = _rank_disagreements(ranking)
    report_path = joinpath(reports_dir, "$(roundname).md")
    if write
        mkpath(reports_dir)
        _write_report(report_path, roundname, results, ranking,
                      disagreements; seasons=seasons, ndraws=ndraws,
                      Dmax=Dmax, transform=transform, parent=parent,
                      max_splits=max_splits)
        @info "wrote report" report_path
    end
    return (ranking=ranking, results=results, disagreements=disagreements,
            report_path=report_path)
end

"""
    _ranking_table(results) -> DataFrame

Ranking of the scored (`:ok`/`:slow`) candidates, sorted by mean WIS then
WIS SD (the overfitting tie-break). Failed candidates are excluded here and
listed separately in the report. Adds natural- and log-scale ranks for the
divergence check.
"""
function _ranking_table(results)
    scored = [r for r in results if isfinite(r.mean_wis)]
    df = DataFrame(
        name=[r.name for r in scored],
        status=[r.status for r in scored],
        mean_wis=[r.mean_wis for r in scored],
        sd_wis=[r.sd_wis for r in scored],
        log_mean_wis=[r.log_mean_wis for r in scored],
        log_sd_wis=[r.log_sd_wis for r in scored],
        coverage50=[r.coverage50 for r in scored],
        coverage90=[r.coverage90 for r in scored],
        max_abs_acf1=[r.max_abs_acf1 for r in scored],
        n_tasks=[r.n_tasks for r in scored],
    )
    nrow(df) == 0 && return df
    sort!(df, [:mean_wis, :sd_wis])
    df.rank_natural = collect(1:nrow(df))
    lg = sort(df[:, [:name, :log_mean_wis]], :log_mean_wis)
    lg.rank_log = collect(1:nrow(lg))
    leftjoin!(df, lg[:, [:name, :rank_log]], on=:name)
    return df
end

"""
    _rank_disagreements(ranking) -> DataFrame

Candidates whose natural-scale rank differs from their log-scale rank. An
empty table means natural and log WIS agree on the ordering.
"""
function _rank_disagreements(ranking::DataFrame)
    (nrow(ranking) == 0 || !hasproperty(ranking, :rank_log)) &&
        return DataFrame()
    return ranking[ranking.rank_natural .!= ranking.rank_log,
                   [:name, :rank_natural, :rank_log, :mean_wis,
                    :log_mean_wis]]
end

# --- report writing ----------------------------------------------------------

_fmt(x) = x isa Real && isfinite(x) ? @sprintf("%.4f", x) : "n/a"

"""
    _write_report(path, roundname, results, ranking, disagreements; ...)

Render a loop report to `path`, following `reports/TEMPLATE.md`: setup, a
results table (mean WIS, WIS SD, log-scale WIS, coverage, notes), the
Bayesian-workflow summaries, the log-scale divergence check, and a
"candidates needing a fix" section listing every `:failed` candidate with
its captured error (we fix them rather than drop them).
"""
function _write_report(
    path, roundname, results, ranking, disagreements;
    seasons, ndraws, Dmax, transform, parent, max_splits,
)
    failed = [r for r in results if r.status == :failed]
    slow = [r for r in results if r.status == :slow]
    io = IOBuffer()
    println(io, "# Loop $(roundname)")
    println(io)
    println(io, "- **Date**: $(Dates.today())")
    println(io, "- **Parent loop**: $(parent)")
    println(io, "- **Inference**: Pathfinder (fast screening)")
    println(io, "- **Seasons scored**: validation (2015/16, 2016/17)")
    println(io)
    println(io, "## What was tried")
    println(io)
    println(io, "$(length(results)) candidate(s): " *
                join([r.name for r in results], ", ") * ".")
    println(io)
    println(io, "## Setup")
    println(io)
    ms = max_splits === nothing ? "all" : string(max_splits)
    println(io, "- Seasons: $(seasons) (splits per season: $(ms))")
    println(io, "- Draws (Pathfinder): $(ndraws); Dmax: $(Dmax); " *
                "transform: $(transform)")
    println(io, "- Locations: 11; Quantiles: 23; AD backend: Mooncake")
    println(io)
    println(io, "## Results")
    println(io)
    println(io, "Ranked by mean WIS, then WIS SD (overfitting guard). " *
                "Natural-scale WIS is the selection metric; log-scale WIS " *
                "is report-only.")
    println(io)
    println(io, "| Rank | Candidate | Mean WIS | WIS SD | Log-scale WIS | " *
                "Cov50 | Cov90 | Status | Notes |")
    println(io, "|---|---|---|---|---|---|---|---|---|")
    if nrow(ranking) == 0
        println(io, "| - | _none scored_ | | | | | | | |")
    else
        for row in eachrow(ranking)
            r = _result_by_name(results, row.name)
            note = r.n_failed_splits > 0 ?
                   "$(r.n_failed_splits)/$(r.n_splits) splits failed" : ""
            println(io, "| $(row.rank_natural) | $(row.name) | " *
                        "$(_fmt(row.mean_wis)) | $(_fmt(row.sd_wis)) | " *
                        "$(_fmt(row.log_mean_wis)) | " *
                        "$(_fmt(row.coverage50)) | $(_fmt(row.coverage90)) | " *
                        "$(r.status) | $(note) |")
        end
    end
    println(io)
    println(io, "## Bayesian workflow")
    println(io)
    println(io, "Per-candidate prior/posterior predictive and residual " *
                "checks (mean over scored splits).")
    println(io)
    println(io, "| Candidate | Prior % outside | Prior % non-finite | " *
                "Post cov50 | Post cov90 | Max |resid ACF(1)| |")
    println(io, "|---|---|---|---|---|---|")
    for r in results
        r.status == :failed && continue
        println(io, "| $(r.name) | $(_fmt(r.prior_frac_outside)) | " *
                    "$(_fmt(r.prior_frac_nonfinite)) | " *
                    "$(_fmt(r.coverage50)) | $(_fmt(r.coverage90)) | " *
                    "$(_fmt(r.max_abs_acf1)) |")
    end
    println(io)
    println(io, "## Log-scale divergence check")
    println(io)
    if nrow(disagreements) == 0
        println(io, "Natural-scale and log-scale WIS agree on the candidate " *
                    "ordering. No divergence.")
    else
        println(io, "Model choice would differ between scales for:")
        println(io)
        println(io, "| Candidate | Rank (natural) | Rank (log) | " *
                    "Mean WIS | Log WIS |")
        println(io, "|---|---|---|---|---|")
        for row in eachrow(disagreements)
            println(io, "| $(row.name) | $(row.rank_natural) | " *
                        "$(row.rank_log) | $(_fmt(row.mean_wis)) | " *
                        "$(_fmt(row.log_mean_wis)) |")
        end
    end
    println(io)
    println(io, "## Candidates needing a fix")
    println(io)
    if isempty(failed) && isempty(slow)
        println(io, "None: every candidate produced forecasts within time.")
    else
        println(io, "We fix complex or failing models rather than abandon " *
                    "them for being complex.")
        println(io)
        for r in failed
            println(io, "- **$(r.name)** (`:failed`): $(r.error)")
        end
        for r in slow
            note = r.error === nothing ? "exceeded the slow threshold" :
                   r.error
            println(io, "- **$(r.name)** (`:slow`): $(note)")
        end
    end
    println(io)
    println(io, "## Decision")
    println(io)
    println(io, "_To be completed by the round reviewer: which candidates " *
                "to keep, refine, or drop, and the next axes to explore._")
    println(io)
    println(io, "## Artifacts")
    println(io)
    println(io, "- Report: `reports/$(roundname).md`")
    println(io, "- Runner: `experiments/run_round.jl`")

    open(path, "w") do f
        Base.write(f, String(take!(io)))
    end
    return path
end

_result_by_name(results, name) =
    results[findfirst(r -> r.name == name, results)]

# --- base-model smoke --------------------------------------------------------

"""
    smoke_round()

Tiny end-to-end proof: run `run_round` on just the base model
(`base_model` + `base_project`) over ONE validation split with few draws, and
confirm it yields a ranking row and a report file without error. Intended to
be cheap on a congested box.
"""
function smoke_round()
    candidates = [(
        name="nfidd-base",
        build_model=base_model,
        project=base_project,
    )]
    return run_round(candidates, "00-smoke"; seasons=[1], max_splits=1,
                     ndraws=60, diag_ndraws=40, parent="base")
end

# Run the smoke when executed as a script (not when include-d).
if abspath(PROGRAM_FILE) == @__FILE__
    out = smoke_round()
    row = nrow(out.ranking) > 0 ? first(eachrow(out.ranking)) : nothing
    if row === nothing
        @warn "smoke produced no ranking row"
    else
        @info "smoke ranking" name=row.name mean_wis=row.mean_wis sd_wis=row.sd_wis
    end
    @info "smoke report" out.report_path
end
