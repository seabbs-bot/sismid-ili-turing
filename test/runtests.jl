# Aggregate and run every test/test_<module>.jl file that exists.
# Works before every module lands: missing files are skipped, not errors.
# Run from the repo root:  julia --project=. test/runtests.jl
#
# Each test file runs in its own subprocess rather than being include()d
# in-process. Turing/Mooncake MCMC runs can hard-crash the process (a
# segfault, not a catchable Julia exception) rather than raising a
# normal error; include() would take the whole aggregator down with it
# and print no summary at all. A subprocess crash just becomes one FAIL.

const TESTDIR = @__DIR__
const ROOTDIR = dirname(TESTDIR)

# Discover every test_<module>.jl file present, so new modules are picked
# up automatically without editing this list. KNOWN_ORDER only fixes the
# display/run order for modules we know about; anything else discovered
# still runs, appended alphabetically after the known ones.
const KNOWN_ORDER = [
    "core", "data", "seasonal", "model", "inference", "forecast",
    "scoring", "hubio", "diagnostics", "integration",
]

function discover_modules(testdir)
    names = String[]
    for f in readdir(testdir)
        m = match(r"^test_(.+)\.jl$", f)
        m === nothing && continue
        push!(names, m.captures[1])
    end
    return names
end

discovered = discover_modules(TESTDIR)
known_present = filter(m -> m in discovered, KNOWN_ORDER)
extra = sort(setdiff(discovered, KNOWN_ORDER))
const MODULES = vcat(known_present, extra)

results = NamedTuple{(:name, :status, :output),
                      Tuple{String,Symbol,String}}[]

for name in MODULES
    file = joinpath(TESTDIR, "test_$name.jl")
    if !isfile(file)
        push!(results, (name = name, status = :skipped, output = ""))
        continue
    end
    @info "running $file"
    io = IOBuffer()
    julia = Base.julia_cmd()
    cmd = `$julia --project=$ROOTDIR --startup-file=no $file`
    proc = run(pipeline(cmd; stdout = io, stderr = io); wait = false)
    wait(proc)
    output = String(take!(io))
    print(output)
    status = proc.exitcode == 0 ? :passed : :failed
    push!(results, (name = name, status = status, output = output))
end

println()
println("=== test/runtests.jl summary ===")
for r in results
    marker = r.status == :passed ? "PASS" :
             r.status == :skipped ? "SKIP" : "FAIL"
    println("  [$marker] test_$(r.name).jl")
    if r.status == :failed
        # Last few lines usually carry the actual error/crash message.
        lines = split(rstrip(r.output), '\n')
        for line in lines[max(1, end - 19):end]
            println("      ", line)
        end
    end
end

n_passed = count(r -> r.status == :passed, results)
n_skipped = count(r -> r.status == :skipped, results)
n_failed = count(r -> r.status == :failed, results)
println(
    "\n$n_passed passed, $n_skipped skipped (no file yet), " *
    "$n_failed failed",
)

if n_failed > 0
    exit(1)
end
