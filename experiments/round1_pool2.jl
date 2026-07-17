# Round 1 screen, CANDIDATE-parallel worker pool.
#
# Unlike round1_pool.jl (pmap over (candidate,origin) items), this pmaps
# over whole CANDIDATES: one worker owns a candidate end to end, so it
# JIT-compiles that candidate's Turing model ONCE and then does all four
# origin fits warm -- instead of the item-level pool's model-thrash,
# where a worker recompiles a different model on nearly every item (the
# 11 candidates are 11 distinct models). It also means each candidate's
# files are written by exactly one worker (no cross-process races), and
# the worker AGGREGATES its own candidate the moment its fits finish, so
# review_round sees summaries appear candidate-by-candidate and the round
# can be scored on whatever has completed (iterate forward, do not block
# on the slowest candidate).
#
# Checkpoint format is identical to round1_run.jl (splits/<origin>.csv,
# diagnostics.txt, meta.txt, summary.txt/region.csv/time.csv), so an
# already-checkpointed origin is skipped (resume) and the reviewer reads
# it unchanged.
#
# Run:  POOL_WORKERS=6 julia --project=. experiments/round1_pool2.jl <outdir>

using Distributed

const REPO = normpath(joinpath(@__DIR__, ".."))
const NWORKERS = parse(Int, get(ENV, "POOL_WORKERS", "6"))
addprocs(NWORKERS; exeflags=["--project=$REPO"])
@info "candidate-parallel pool up" nworkers=nworkers()

@everywhere begin
    const REPO = $REPO
    include(joinpath(REPO, "experiments", "round1_run.jl"))
    import SismidILITuring: observation_index
    const _R2 = joinpath(REPO, "experiments", "round2")
    include(joinpath(_R2, "severity", "model_severity.jl"))
    include(joinpath(_R2, "severity", "project_severity.jl"))
    include(joinpath(_R2, "season-backfill", "model_season_backfill.jl"))
    include(joinpath(_R2, "season-backfill", "project_season_backfill.jl"))
    include(joinpath(_R2, "ar-loc", "model_ar_loc.jl"))
    include(joinpath(_R2, "ar-loc", "project_ar_loc.jl"))
    include(joinpath(_R2, "var", "model_var.jl"))
    include(joinpath(_R2, "var", "project_var.jl"))

    ar_loc_build(d::ModelData; transform::Symbol=:fourthroot) =
        model_ar_loc(d; transform=transform, Pmax=10)

    const POOL_CANDIDATES = Dict(
        "nfidd-base" => (base_model, base_project, :fourthroot, 150, 12),
        "nfidd-base-log" => (base_model, base_project, :log, 150, 12),
        "nfidd-ar-high" => (model_v1, project_v1, :fourthroot, 150, 12),
        "nfidd-mvn-season" => (model_v2, project_v2, :fourthroot, 150, 12),
        "nfidd-diff" => (model_v3, project_v3, :fourthroot, 150, 12),
        "nfidd-tv-ar" => (model_v4, project_v4, :fourthroot, 150, 12),
        "nfidd-backfill" => (model_v5, project_v5, :fourthroot, 150, 12),
        "nfidd-severity" =>
            (model_severity, project_severity, :fourthroot, 150, 12),
        "nfidd-season-backfill" =>
            (model_season_backfill, project_season_backfill,
             :fourthroot, 150, 12),
        "nfidd-ar-loc" =>
            (ar_loc_build, project_ar_loc, :fourthroot, 150, 12),
        "nfidd-var" => (model_var, project_var, :fourthroot, 150, 12),
    )

    # Fit every origin of ONE candidate (skipping already-checkpointed
    # ones), writing this candidate's own checkpoints/diagnostics, then
    # aggregate it. One worker owns this candidate, so writing here is
    # race-free. Returns a tiny status NamedTuple.
    function run_candidate_pooled(cname, outdir, truth)
        bm, proj, transform, ndraws, Dmax = POOL_CANDIDATES[cname]
        cdir = joinpath(outdir, cname)
        splitdir = joinpath(cdir, "splits")
        mkpath(splitdir)
        open(joinpath(cdir, "meta.txt"), "w") do io
            println(io, "transform=$transform")
            println(io, "ndraws=$ndraws")
            println(io, "Dmax=$Dmax")
        end
        diagpath = joinpath(cdir, "diagnostics.txt")
        all_splits = Dict(1 => training_splits(1), 2 => training_splits(2))
        n_ok = 0
        n_failed = 0
        for origin in ORIGIN_DATES
            ckpt = joinpath(splitdir, "$(origin).csv")
            isfile(ckpt) && (n_ok += 1; continue)
            season = ORIGIN_SEASON[origin]
            splits = all_splits[season]
            idx = findfirst(s -> maximum(s.origin_date) == origin, splits)
            idx === nothing && continue
            t0 = time()
            status, fq, diag, errmsg = fit_one_split(
                bm, proj, splits[idx], transform, Dmax, ndraws, 100, cname)
            elapsed = time() - t0
            open(diagpath, "a") do io
                if status == :ok
                    println(io, join([
                        origin, "ok", elapsed, diag.coverage50,
                        diag.coverage90, diag.prior_frac_outside,
                        diag.prior_frac_nonfinite, diag.max_abs_acf1,
                        diag.frac_bad_draws, "",
                    ], ","))
                else
                    println(io, join([
                        origin, "failed", elapsed, "", "", "", "", "", "",
                        replace(errmsg === nothing ? "?" : errmsg,
                                "," => ";", "\n" => " "),
                    ], ","))
                end
            end
            if status == :ok
                CSV.write(ckpt, fq)
                n_ok += 1
            else
                n_failed += 1
            end
        end
        aggregate_candidate(cname, outdir, truth)
        return (cname=cname, n_ok=n_ok, n_failed=n_failed)
    end
end

function main()
    length(ARGS) >= 1 || error("usage: round1_pool2.jl <outdir>")
    outdir = isabspath(ARGS[1]) ? ARGS[1] : joinpath(REPO, ARGS[1])
    mkpath(outdir)
    truth = load_validation_truth()

    # Order candidates so the cheap/important ones (base first) get a
    # worker immediately; pmap then dispatches the rest as workers free.
    order = ["nfidd-base", "nfidd-diff", "nfidd-base-log", "nfidd-backfill",
             "nfidd-mvn-season", "nfidd-tv-ar", "nfidd-severity",
             "nfidd-season-backfill", "nfidd-ar-high", "nfidd-ar-loc",
             "nfidd-var"]
    cands = filter(c -> haskey(POOL_CANDIDATES, c), order)
    @info "candidates" n=length(cands) cands

    results = pmap(c -> run_candidate_pooled(c, outdir, truth), cands;
                   on_error = e -> (err=sprint(showerror, e),))
    for (c, r) in zip(cands, results)
        if r isa NamedTuple && haskey(r, :err)
            @warn "candidate crashed (worker died)" c r.err
        else
            @info "candidate done" r.cname r.n_ok r.n_failed
        end
    end
    @info "ROUND1 POOL2 DONE" outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
