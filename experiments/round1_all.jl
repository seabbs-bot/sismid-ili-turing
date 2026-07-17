# Round 1, all candidates, ONE process: compiles Turing/Pathfinder/Mooncake
# once and reuses it across every (candidate, origin_date) fit via
# `Threads.@threads`, instead of one subprocess per candidate (which pays
# the ~2 min Turing compile cost 7 times over). Reuses `fit_one_split` and
# `aggregate_candidate` from `round1_run.jl` unchanged, so the checkpoint
# format (splits/<origin>.csv, diagnostics.txt, summary.txt, region.csv,
# time.csv) is identical to that driver's.
#
# Run with:
#   JULIA_NUM_THREADS=8 julia --project=. experiments/round1_all.jl <outdir>
# or
#   julia --project=. -t 8 experiments/round1_all.jl <outdir>
#
# Each work item (one candidate's one origin-date split) is independent
# and checkpoints its own file, so a crash mid-run loses at most the
# items in flight; re-running skips already-checkpointed splits.

include(joinpath(@__DIR__, "round1_run.jl"))

using Base.Threads: @threads, ReentrantLock, lock

const DIAG_LOCK = ReentrantLock()

"""
    build_worklist() -> Vector{NamedTuple}

One entry per (candidate, origin_date) pair still needing a fit: all
7 Round 1 candidates (base on fourthroot, base on log as the
transform-axis check, and v1..v5) crossed with the 8 screening origin
dates, i.e. up to 56 items, resolved against each season's
`training_splits` once up front (shared, read-only across threads).
"""
function build_worklist()
    all_splits = Dict(1 => training_splits(1), 2 => training_splits(2))
    items = NamedTuple[]
    for (cname, cand) in CANDIDATES
        name, build_model, project, opts = _candidate_parts(cand)
        transform = get(opts, :transform, PRIMARY_TRANSFORM)
        ndraws = get(opts, :ndraws, 150)
        Dmax = get(opts, :Dmax, 12)
        for origin in ORIGIN_DATES
            season = ORIGIN_SEASON[origin]
            splits = all_splits[season]
            idx = findfirst(s -> maximum(s.origin_date) == origin, splits)
            idx === nothing && continue
            push!(items, (cname=cname, name=name, build_model=build_model,
                          project=project, transform=transform,
                          ndraws=ndraws, Dmax=Dmax, origin=origin,
                          split=splits[idx]))
        end
    end
    return items
end

"""
    run_item!(item, outdir)

Fit + checkpoint ONE (candidate, origin_date) work item, mirroring the
per-split body of `round1_run.jl`'s `run_one_candidate` (same
checkpoint files), guarded so concurrent threads writing the same
candidate's `diagnostics.txt` do not interleave lines.
"""
function run_item!(item, outdir)
    cdir = joinpath(outdir, item.cname)
    splitdir = joinpath(cdir, "splits")
    mkpath(splitdir)
    ckpt = joinpath(splitdir, "$(item.origin).csv")
    if isfile(ckpt)
        @info "skip (already checkpointed)" item.cname item.origin
        return
    end
    metapath = joinpath(cdir, "meta.txt")
    isfile(metapath) || open(metapath, "w") do io
        println(io, "transform=$(item.transform)")
        println(io, "ndraws=$(item.ndraws)")
        println(io, "Dmax=$(item.Dmax)")
    end

    t0 = time()
    status, fq, diag, errmsg = fit_one_split(
        item.build_model, item.project, item.split, item.transform,
        item.Dmax, item.ndraws, 100, item.name)
    elapsed = time() - t0

    diagpath = joinpath(cdir, "diagnostics.txt")
    lock(DIAG_LOCK) do
        open(diagpath, "a") do io
            if status == :ok
                println(io, join([
                    item.origin, "ok", elapsed, diag.coverage50,
                    diag.coverage90, diag.prior_frac_outside,
                    diag.prior_frac_nonfinite, diag.max_abs_acf1,
                    diag.frac_bad_draws, "",
                ], ","))
            else
                println(io, join([
                    item.origin, "failed", elapsed, "", "", "", "", "", "",
                    replace(errmsg, "," => ";", "\n" => " "),
                ], ","))
            end
        end
    end
    if status == :ok
        CSV.write(ckpt, fq)
        @info "checkpointed" item.cname item.origin elapsed Threads.threadid()
    else
        @warn "split failed" item.cname item.origin errmsg
    end
end

function main()
    length(ARGS) >= 1 || error("usage: round1_all.jl <outdir>")
    outdir = ARGS[1]
    mkpath(outdir)
    items = build_worklist()
    @info "worklist built" n_items=length(items) n_threads=Threads.nthreads()

    @threads for i in eachindex(items)
        try
            run_item!(items[i], outdir)
        catch err
            @error "work item errored (checkpoint skipped, continuing)" items[i].cname items[i].origin err
        end
    end

    @info "all work items done, aggregating"
    truth = load_validation_truth()
    for cname in keys(CANDIDATES)
        try
            aggregate_candidate(cname, outdir, truth)
            @info "aggregated" cname
        catch err
            @error "aggregation failed" cname err
        end
    end
    @info "ROUND1 ALL DONE"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
