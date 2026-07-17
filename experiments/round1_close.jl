# Round 1 CLOSE: a focused, well-converged re-screen replacing the
# previous attempt's ndraws=150, 4-origin numbers (base 1.22, diff
# 2.04 -- see reports/turing-vs-baseline.md), which were not a fair
# test of the joint model against the nfidd-ar6 baseline (mean WIS
# 0.368 full validation / 0.330 same-4-origin subset).
#
# Runs FIVE candidates, SEQUENTIALLY, in ONE process (so each
# candidate's Turing model is JIT-compiled once and its own splits run
# warm; no Distributed pool, no per-candidate OS process, matching the
# brief's "at most 2 concurrent (or sequential) in ONE compile-once
# process" and avoiding the resource contention a multi-process pool
# causes on this shared, already-loaded box):
#   - nfidd-base       (base_model, fourthroot -- the primary joint model)
#   - base-tight       (base_model with tightened, better-regularised
#                        hyperpriors -- experiments/round2/base-tight/)
#   - nfidd-diff       (difference residual instead of AR(1))
#   - nfidd-ar-high    (partially-pooled AR(p), p from PACF)
#   - nfidd-mvn-season (correlated cross-location seasonal deviations)
#
# ndraws=300 (vs this session's earlier ndraws=150) is set per-candidate
# in `experiments/round1_run.jl`'s `CANDIDATES` dict, not here.
#
# Writes to a FRESH results directory
# (`experiments/round1/_results_close/`), NOT the shared
# `experiments/round1/_results/` some other still-running processes on
# this box are writing into with the old ndraws=150/4-origin settings
# -- this avoids any cross-process checkpoint race while those finish
# or are cleaned up.
#
# Run:
#   julia --project=. experiments/round1_close.jl

include(joinpath(@__DIR__, "round1_run.jl"))

const CLOSE_CANDIDATES = [
    "nfidd-base", "base-tight", "nfidd-diff", "nfidd-ar-high",
    "nfidd-mvn-season",
]
const CLOSE_OUTDIR = joinpath(@__DIR__, "round1", "_results_close")

function main()
    truth = load_validation_truth()
    for cname in CLOSE_CANDIDATES
        t0 = time()
        @info "=== starting candidate ===" cname outdir=CLOSE_OUTDIR
        try
            run_one_candidate(cname, CLOSE_OUTDIR)
        catch err
            @error "candidate errored during split loop (partial results kept)" cname err
        end
        try
            aggregate_candidate(cname, CLOSE_OUTDIR, truth)
        catch err
            @error "aggregation failed" cname err
            rethrow()
        end
        @info "=== candidate done ===" cname elapsed_s=(time() - t0)
    end
end

main()
