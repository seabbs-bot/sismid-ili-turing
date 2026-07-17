# Round 1 WIDE screen via a Distributed worker POOL.
#
# Unlike experiments/round1_all.jl (one process, Threads.@threads), this
# spins up N worker PROCESSES, compiles Turing/Pathfinder/Mooncake and the
# package ONCE per worker (@everywhere), then pmaps every (candidate,
# origin) fit across the pool. Process isolation means a native crash
# (segfault under shared-host contention) kills at most one worker and one
# item, not the whole round -- pmap's on_error records it and carries on.
#
# ALL file writing happens on the MASTER as results come back, so no two
# processes ever write the same candidate's diagnostics.txt/checkpoint
# concurrently. Checkpoint format is IDENTICAL to round1_run.jl's
# (splits/<origin>.csv, diagnostics.txt, meta.txt), so `aggregate_candidate`
# and the reviewer read it unchanged, and an already-checkpointed split is
# skipped (resume).
#
# Run:  POOL_WORKERS=3 julia --project=. experiments/round1_pool.jl <outdir>

using Distributed

const REPO = normpath(joinpath(@__DIR__, ".."))
const NWORKERS = parse(Int, get(ENV, "POOL_WORKERS", "3"))

addprocs(NWORKERS; exeflags=["--project=$REPO"])
@info "worker pool up" nworkers=nworkers() workers=workers()

@everywhere begin
    const REPO = $REPO
    # round1_run.jl brings: SismidILITuring (compiled once here), the
    # round1 machinery (fit_one_split, aggregate_candidate, ORIGIN_*,
    # load_validation_truth, latent_predict, _frac_bad_draws), and the
    # v1..v5 candidate models/projects. main() is guarded, so including
    # it does not run anything.
    include(joinpath(REPO, "experiments", "round1_run.jl"))
    import SismidILITuring: observation_index   # round2 models use it

    const _R2 = joinpath(REPO, "experiments", "round2")
    include(joinpath(_R2, "severity", "model_severity.jl"))
    include(joinpath(_R2, "severity", "project_severity.jl"))
    include(joinpath(_R2, "season-backfill", "model_season_backfill.jl"))
    include(joinpath(_R2, "season-backfill", "project_season_backfill.jl"))
    include(joinpath(_R2, "ar-loc", "model_ar_loc.jl"))
    include(joinpath(_R2, "ar-loc", "project_ar_loc.jl"))
    include(joinpath(_R2, "var", "model_var.jl"))
    include(joinpath(_R2, "var", "project_var.jl"))

    # ar-loc needs its ceiling order fixed; fit_and_forecast only threads
    # `transform`, so wrap the Pmax kwarg here (mirrors model_v3's wrap).
    ar_loc_build(d::ModelData; transform::Symbol=:fourthroot) =
        model_ar_loc(d; transform=transform, Pmax=10)

    # cname => (build_model, project, transform, ndraws, Dmax). Everything
    # on :fourthroot (the primary screening scale) except the base-log
    # transform-axis check. ndraws=150 per the screen spec.
    const POOL_CANDIDATES = Dict(
        "nfidd-base" =>
            (base_model, base_project, :fourthroot, 150, 12),
        "nfidd-base-log" =>
            (base_model, base_project, :log, 150, 12),
        "nfidd-ar-high" =>
            (model_v1, project_v1, :fourthroot, 150, 12),
        "nfidd-mvn-season" =>
            (model_v2, project_v2, :fourthroot, 150, 12),
        "nfidd-diff" =>
            (model_v3, project_v3, :fourthroot, 150, 12),
        "nfidd-tv-ar" =>
            (model_v4, project_v4, :fourthroot, 150, 12),
        "nfidd-backfill" =>
            (model_v5, project_v5, :fourthroot, 150, 12),
        "nfidd-severity" =>
            (model_severity, project_severity, :fourthroot, 150, 12),
        "nfidd-season-backfill" =>
            (model_season_backfill, project_season_backfill,
             :fourthroot, 150, 12),
        "nfidd-ar-loc" =>
            (ar_loc_build, project_ar_loc, :fourthroot, 150, 12),
        "nfidd-var" =>
            (model_var, project_var, :fourthroot, 150, 12),
    )

    # Per-worker split cache: training_splits reads a CSV, so resolve it
    # once per season per worker and reuse across every origin.
    const _SPLITS = Dict{Int,Vector{DataFrame}}()
    function _get_split(season, origin)
        haskey(_SPLITS, season) ||
            (_SPLITS[season] = training_splits(season))
        splits = _SPLITS[season]
        idx = findfirst(s -> maximum(s.origin_date) == origin, splits)
        idx === nothing && error("origin $origin not in season $season")
        return splits[idx]
    end

    # One work item -> fit result. PURE: writes no files (the master does
    # that). fit_one_split already try/catches inference errors and returns
    # status=:failed; a native crash instead kills this worker and pmap's
    # on_error handles it.
    function do_item(item)
        bm, proj, transform, ndraws, Dmax = POOL_CANDIDATES[item.cname]
        split = _get_split(item.season, item.origin)
        t0 = time()
        status, fq, diag, errmsg =
            fit_one_split(bm, proj, split, transform, Dmax, ndraws, 100,
                          item.cname)
        elapsed = time() - t0
        return (cname=item.cname, origin=item.origin, status=status,
                fq=fq, diag=diag, errmsg=errmsg, elapsed=elapsed,
                wid=myid())
    end
end

# --- master-side file writing (single writer, no races) ----------------

"""Write one candidate's meta.txt (idempotent; identical across reruns)."""
function _write_meta(cdir, transform, ndraws, Dmax)
    open(joinpath(cdir, "meta.txt"), "w") do io
        println(io, "transform=$transform")
        println(io, "ndraws=$ndraws")
        println(io, "Dmax=$Dmax")
    end
end

"""Append a `do_item` result's diagnostics line and (if ok) checkpoint
CSV, in exactly round1_run.jl's format so aggregate_candidate reads it."""
function _write_result!(outdir, r)
    cdir = joinpath(outdir, r.cname)
    mkpath(joinpath(cdir, "splits"))
    diagpath = joinpath(cdir, "diagnostics.txt")
    open(diagpath, "a") do io
        if r.status == :ok
            println(io, join([
                r.origin, "ok", r.elapsed, r.diag.coverage50,
                r.diag.coverage90, r.diag.prior_frac_outside,
                r.diag.prior_frac_nonfinite, r.diag.max_abs_acf1,
                r.diag.frac_bad_draws, "",
            ], ","))
        else
            msg = r.errmsg === nothing ? "unknown" : r.errmsg
            println(io, join([
                r.origin, "failed", r.elapsed, "", "", "", "", "", "",
                replace(msg, "," => ";", "\n" => " "),
            ], ","))
        end
    end
    if r.status == :ok
        CSV.write(joinpath(cdir, "splits", "$(r.origin).csv"), r.fq)
    end
end

function main()
    length(ARGS) >= 1 || error("usage: round1_pool.jl <outdir>")
    outdir = isabspath(ARGS[1]) ? ARGS[1] : joinpath(REPO, ARGS[1])
    mkpath(outdir)

    # Build the worklist on the master, skipping already-checkpointed
    # splits (resume). meta.txt is (re)written up front so a candidate
    # dir always carries its config even before its first fit lands.
    items = NamedTuple[]
    for (cname, spec) in POOL_CANDIDATES
        _, _, transform, ndraws, Dmax = spec
        cdir = joinpath(outdir, cname)
        mkpath(joinpath(cdir, "splits"))
        _write_meta(cdir, transform, ndraws, Dmax)
        for origin in ORIGIN_DATES
            ckpt = joinpath(cdir, "splits", "$(origin).csv")
            isfile(ckpt) && (@info "skip (checkpointed)" cname origin;
                             continue)
            push!(items, (cname=cname, season=ORIGIN_SEASON[origin],
                          origin=origin))
        end
    end
    @info "worklist" n_items=length(items) n_candidates=length(POOL_CANDIDATES)

    # pmap across the pool. on_error keeps a worker death (segfault ->
    # ProcessExitedException) from sinking the round: the crashed item's
    # slot gets a sentinel, and order is preserved so we can map it back
    # to its item.
    results = pmap(do_item, items;
                   on_error = e -> (crashed=true, err=sprint(showerror, e)))

    n_ok = 0; n_failed = 0; n_crashed = 0
    for (item, r) in zip(items, results)
        if r isa NamedTuple && get(r, :crashed, false)
            n_crashed += 1
            @warn "item crashed (worker died)" item.cname item.origin r.err
            # Record as a failed split so it is visible, not dropped.
            _write_result!(outdir, (cname=item.cname, origin=item.origin,
                status=:failed, fq=nothing, diag=nothing,
                errmsg="worker died: $(r.err)", elapsed=NaN, wid=0))
            continue
        end
        _write_result!(outdir, r)
        if r.status == :ok
            n_ok += 1
            @info "ok" r.cname r.origin elapsed=round(r.elapsed; digits=1) r.wid
        else
            n_failed += 1
            @warn "fit failed" r.cname r.origin r.errmsg
        end
    end
    @info "fits done" n_ok n_failed n_crashed

    # Aggregate every candidate (scored on whatever splits it managed to
    # checkpoint). Reuses round1_run.jl's aggregate_candidate unchanged.
    truth = load_validation_truth()
    for cname in keys(POOL_CANDIDATES)
        try
            aggregate_candidate(cname, outdir, truth)
            @info "aggregated" cname
        catch err
            @error "aggregation failed" cname err
        end
    end
    @info "ROUND1 POOL DONE" outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
