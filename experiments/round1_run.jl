# Round 1 per-candidate driver: runs ONE named candidate from the Round 1
# shortlist, split by split, checkpointing after each split so a crash
# (including a native segfault under shared-host contention, which no
# in-process try/catch can stop) loses at most the split in flight, never
# the whole candidate. A later re-run of the same candidate skips splits
# whose checkpoint file already exists, so it resumes rather than
# refitting from scratch.
#
# Run ONE candidate per OS process (not one process looping over every
# candidate, as the interactive `run_round` engine does): an outer
# `timeout` provides the per-candidate wall-clock limit, and the invoking
# shell can run several of these concurrently, so a crash or hang in one
# candidate cannot block or corrupt the others.
#
# Usage:
#   julia --project=. experiments/round1_run.jl <candidate_name> <outdir>
#
# Writes, under <outdir>/<candidate_name>/:
#   meta.txt                   -- transform used (so aggregation, which
#                                  may run as a separate invocation, does
#                                  not need to re-derive it)
#   splits/<origin_date>.csv   -- that split's forecast_df rows (checkpoint)
#   diagnostics.txt            -- one appended line per completed split
#   summary.txt                -- final key=value pooled summary (this
#                                  candidate's analogue of run_round's
#                                  ranking-table row)
#   region.csv                 -- per-location WIS breakdown
#   time.csv                   -- per-origin-date WIS breakdown (with a
#                                  season-phase label)

include(joinpath(@__DIR__, "run_round.jl"))

# Non-exported helpers `model_v1.jl`..`model_v5.jl` assume are already in
# scope (see experiments/README.md); bring them in explicitly since
# `using SismidILITuring` only imports exported names.
import SismidILITuring: model_dims, backfill_profile, ar_or_diff

const R1 = joinpath(@__DIR__, "round1")
include(joinpath(R1, "v1-ar-high", "model_v1.jl"))
include(joinpath(R1, "v1-ar-high", "project_v1.jl"))
include(joinpath(R1, "v2-mvn-season", "model_v2.jl"))
include(joinpath(R1, "v2-mvn-season", "project_v2.jl"))
include(joinpath(R1, "v3-diff", "model_v3.jl"))
include(joinpath(R1, "v3-diff", "project_v3.jl"))
include(joinpath(R1, "v4-tv-ar", "model_v4.jl"))
include(joinpath(R1, "v4-tv-ar", "project_v4.jl"))
include(joinpath(R1, "v5-backfill", "model_v5.jl"))
include(joinpath(R1, "v5-backfill", "project_v5.jl"))

using Dates
using CSV
using DataFrames
using Statistics: mean, std

# Eight origin dates spanning pre-peak, rising, peak, and post-peak in
# both validation seasons (picked from a mean-recent-wILI% scan of
# `training_splits`, see steer-log.md). Season 1 (2015/16) peaks late
# (~March); season 2 (2016/17) peaks in Feb. The label is used for the
# per-time/failure-mode breakdown, not for fitting.
const ORIGIN_PHASE = Dict(
    Date("2015-11-14") => "pre-peak", Date("2015-12-26") => "rising",
    Date("2016-03-12") => "peak", Date("2016-04-23") => "post-peak",
    Date("2016-11-12") => "pre-peak", Date("2016-12-24") => "rising",
    Date("2017-02-25") => "peak", Date("2017-04-08") => "post-peak",
)
const ORIGIN_SEASON = Dict(
    Date("2015-11-14") => 1, Date("2015-12-26") => 1,
    Date("2016-03-12") => 1, Date("2016-04-23") => 1,
    Date("2016-11-12") => 2, Date("2016-12-24") => 2,
    Date("2017-02-25") => 2, Date("2017-04-08") => 2,
)
const ORIGIN_DATES = sort(collect(keys(ORIGIN_PHASE)))

const PRIMARY_TRANSFORM = :fourthroot

const CANDIDATES = Dict(
    "nfidd-base-log" => (name="nfidd-base-log", build_model=base_model,
                          project=base_project, transform=:log),
    "nfidd-base" => (name="nfidd-base", build_model=base_model,
                      project=base_project),
    "nfidd-ar-high" => (name="nfidd-ar-high", build_model=model_v1,
                         project=project_v1),
    "nfidd-mvn-season" => (name="nfidd-mvn-season", build_model=model_v2,
                            project=project_v2),
    "nfidd-diff" => (name="nfidd-diff", build_model=model_v3,
                      project=project_v3),
    "nfidd-tv-ar" => (name="nfidd-tv-ar", build_model=model_v4,
                       project=project_v4),
    "nfidd-backfill" => (name="nfidd-backfill", build_model=model_v5,
                          project=project_v5),
)

"""
    fit_one_split(build_model, project, split, transform, Dmax, ndraws,
                  diag_ndraws, name)

Fit and forecast ONE training split (mirrors the inner body of
`run_round.jl`'s `run_candidate`, but for a single split so the caller
can checkpoint immediately after), via the shared
[`fit_and_forecast`](@ref) (docs/lessons.md). Returns `(status,
forecast_df, diag, errmsg)`; `status` is `:ok` or `:failed`.
"""
function fit_one_split(build_model, project, split, transform, Dmax,
                        ndraws, diag_ndraws, name)
    try
        data = build_model_data(split; Dmax=Dmax, transform=transform,
                                window_weeks=104)
        result = fit_and_forecast(build_model, data, name;
                                  project=project, ndraws=ndraws,
                                  transform=transform)
        fit, gdraws, fq = result.fit, result.draws, result.forecast
        bc = bayesian_checks(fit, result.model, data; ndraws=diag_ndraws,
                              draws=gdraws, predict=latent_predict)
        diag = (
            frac_bad_draws=_frac_bad_draws(gdraws),
            coverage50=bc.posterior.calibration.coverage50,
            coverage90=bc.posterior.calibration.coverage90,
            prior_frac_outside=bc.prior.summary.frac_outside_plausible_range,
            prior_frac_nonfinite=bc.prior.summary.frac_nonfinite,
            max_abs_acf1=let acf = bc.residuals.autocorrelation
                nrow(acf) > 0 && any(acf.lag .== 1) ?
                    maximum(abs, acf[acf.lag .== 1, :acf]) : NaN
            end,
        )
        return (:ok, fq, diag, nothing)
    catch err
        return (:failed, nothing, nothing, _short_error(err))
    end
end

function run_one_candidate(cname, outdir)
    haskey(CANDIDATES, cname) || error("unknown candidate: $cname")
    cand = CANDIDATES[cname]
    name, build_model, project, opts = _candidate_parts(cand)
    transform = get(opts, :transform, PRIMARY_TRANSFORM)
    ndraws = get(opts, :ndraws, 150)
    Dmax = get(opts, :Dmax, 12)

    cdir = joinpath(outdir, cname)
    splitdir = joinpath(cdir, "splits")
    mkpath(splitdir)
    diagpath = joinpath(cdir, "diagnostics.txt")
    open(joinpath(cdir, "meta.txt"), "w") do io
        println(io, "transform=$transform")
        println(io, "ndraws=$ndraws")
        println(io, "Dmax=$Dmax")
    end

    truth = load_validation_truth()
    all_splits = Dict(1 => training_splits(1), 2 => training_splits(2))

    for origin in ORIGIN_DATES
        ckpt = joinpath(splitdir, "$(origin).csv")
        if isfile(ckpt)
            @info "skipping already-checkpointed split" name origin
            continue
        end
        season = ORIGIN_SEASON[origin]
        splits = all_splits[season]
        idx = findfirst(s -> maximum(s.origin_date) == origin, splits)
        if idx === nothing
            @warn "origin date not found in training_splits" name origin season
            continue
        end
        split = splits[idx]
        t0 = time()
        status, fq, diag, errmsg = fit_one_split(
            build_model, project, split, transform, Dmax, ndraws, 100, name)
        elapsed = time() - t0
        open(diagpath, "a") do io
            if status == :ok
                println(io, join([
                    origin, "ok", elapsed, diag.coverage50, diag.coverage90,
                    diag.prior_frac_outside, diag.prior_frac_nonfinite,
                    diag.max_abs_acf1, diag.frac_bad_draws, "",
                ], ","))
            else
                println(io, join([
                    origin, "failed", elapsed, "", "", "", "", "", "",
                    replace(errmsg, "," => ";", "\n" => " "),
                ], ","))
            end
        end
        if status == :ok
            CSV.write(ckpt, fq)
            @info "split checkpointed" name origin elapsed
        else
            @warn "split failed" name origin errmsg
        end
    end
end

"""
    aggregate_candidate(cname, outdir, truth)

Read back every checkpointed split for `cname`, score the pooled
forecast table against `truth`, and write `summary.txt`, `region.csv`,
`time.csv` under `outdir/cname/`. Safe to call after a partial run (a
crashed candidate is scored on whatever splits it managed to
checkpoint).
"""
function aggregate_candidate(cname, outdir, truth)
    cdir = joinpath(outdir, cname)
    splitdir = joinpath(cdir, "splits")
    files = isdir(splitdir) ? readdir(splitdir; join=true) : String[]
    files = filter(f -> endswith(f, ".csv"), files)

    metapath = joinpath(cdir, "meta.txt")
    transform_used = "unknown"
    if isfile(metapath)
        for line in readlines(metapath)
            if startswith(line, "transform=")
                transform_used = split(line, "=", limit=2)[2]
            end
        end
    end

    diagpath = joinpath(cdir, "diagnostics.txt")
    diag_lines = isfile(diagpath) ? readlines(diagpath) : String[]
    n_splits = length(diag_lines)
    n_failed = count(l -> occursin(",failed,", l), diag_lines)

    if isempty(files)
        open(joinpath(cdir, "summary.txt"), "w") do io
            println(io, "name=$cname")
            println(io, "status=failed")
            println(io, "transform=$transform_used")
            println(io, "error=no splits produced forecasts")
            println(io, "n_splits=$n_splits")
            println(io, "n_failed_splits=$n_failed")
        end
        return
    end

    forecast_df = reduce(vcat, [CSV.read(f, DataFrame) for f in files])
    scored_nat = score_forecasts(forecast_df, truth; scale=:natural)
    scored_log = score_forecasts(forecast_df, truth; scale=:log)

    mean_wis = mean(scored_nat.wis)
    sd_wis = std(scored_nat.wis)
    log_mean_wis = mean(scored_log.wis)
    log_sd_wis = std(scored_log.wis)

    # --- per-region (location) breakdown -----------------------------
    region = combine(groupby(scored_nat, :location),
        :wis => mean => :mean_wis, :wis => std => :sd_wis, nrow => :n)
    sort!(region, :mean_wis; rev=true)
    CSV.write(joinpath(cdir, "region.csv"), region)

    # --- per-time (origin_date) breakdown -----------------------------
    time_df = combine(groupby(scored_nat, :origin_date),
        :wis => mean => :mean_wis, :wis => std => :sd_wis, nrow => :n)
    time_df.phase = [get(ORIGIN_PHASE, d, "?") for d in time_df.origin_date]
    sort!(time_df, :mean_wis; rev=true)
    CSV.write(joinpath(cdir, "time.csv"), time_df)

    worst_region = region[1, :location]
    worst_region_wis = region[1, :mean_wis]
    worst_time = time_df[1, :origin_date]
    worst_time_wis = time_df[1, :mean_wis]

    # diagnostics averaged over OK splits
    ok_rows = [split(l, ",") for l in diag_lines if occursin(",ok,", l)]
    avgf(i) = isempty(ok_rows) ? NaN :
        mean(parse(Float64, r[i]) for r in ok_rows if r[i] != "")

    open(joinpath(cdir, "summary.txt"), "w") do io
        println(io, "name=$cname")
        println(io, "status=ok")
        println(io, "transform=$transform_used")
        println(io, "mean_wis=$mean_wis")
        println(io, "sd_wis=$sd_wis")
        println(io, "log_mean_wis=$log_mean_wis")
        println(io, "log_sd_wis=$log_sd_wis")
        println(io, "coverage50=$(avgf(4))")
        println(io, "coverage90=$(avgf(5))")
        println(io, "prior_frac_outside=$(avgf(6))")
        println(io, "prior_frac_nonfinite=$(avgf(7))")
        println(io, "max_abs_acf1=$(avgf(8))")
        println(io, "frac_bad_draws=$(avgf(9))")
        println(io, "n_tasks=$(nrow(scored_nat))")
        println(io, "n_splits=$n_splits")
        println(io, "n_failed_splits=$n_failed")
        println(io, "worst_region=$worst_region")
        println(io, "worst_region_wis=$worst_region_wis")
        println(io, "worst_time=$worst_time")
        println(io, "worst_time_wis=$worst_time_wis")
    end
end

function main()
    length(ARGS) >= 2 || error("usage: round1_run.jl <candidate> <outdir>")
    cname, outdir = ARGS[1], ARGS[2]
    haskey(CANDIDATES, cname) || error("unknown candidate: $cname")
    try
        run_one_candidate(cname, outdir)
    catch err
        @warn "candidate errored during split loop (partial results kept)" cname err
    end
    truth = load_validation_truth()
    try
        aggregate_candidate(cname, outdir, truth)
    catch err
        @error "aggregation failed" cname err
        rethrow()
    end
    @info "candidate done" cname
end

# Only run when executed as a script (not when `include`d for its
# definitions, e.g. from a test/smoke harness), matching run_round.jl's
# own convention for `smoke_round()`.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
