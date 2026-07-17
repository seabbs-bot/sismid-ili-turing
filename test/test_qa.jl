# Advisory QA layer using EpiAwarePackageTools.jl (the EpiAware org's
# "Ecosystem test utilities"), added as an unregistered dependency of the
# isolated test/ environment (test/Project.toml). Picked up automatically
# by test/runtests.jl's discover_modules (matches test_<name>.jl), so no
# edit to that file was needed to wire this in.
#
# Runs test/qa_runner.jl in ITS OWN nested subprocess under
# `--project=test`, isolating Aqua/JuliaFormatter/EpiAwarePackageTools
# from the root modelling environment.
#
# Report-only: a failing check prints its findings, but this file always
# exits 0, so a formatting finding never blocks test/runtests.jl's
# pass/fail gate while src/ is under active, fast-moving search (see
# docs/brief.md "move fast; do not let caution slow the search down").
# Tighten later by propagating the nested exit code once formatting
# conventions are settled and the root Project.toml carries a [compat]
# table (Aqua's stricter checks need that; see docs/infrastructure.md).

const TESTDIR = @__DIR__
const RUNNER = joinpath(TESTDIR, "qa_runner.jl")

julia = Base.julia_cmd()
cmd = `$julia --project=$TESTDIR --startup-file=no $RUNNER`

println("test_qa.jl: running advisory QA checks (EpiAwarePackageTools) ...")
proc = run(cmd; wait = false)
wait(proc)

if proc.exitcode == 0
    println("test_qa.jl: QA checks passed.")
else
    println("test_qa.jl: QA checks found issues (advisory only, see " *
            "output above) -- not failing the suite.")
end

println("test_qa.jl: done (this entry never fails the test suite).")
