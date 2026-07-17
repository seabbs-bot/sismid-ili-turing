# Isolated QA subprocess invoked by test_qa.jl with `--project=test` (see
# test/Project.toml). Kept a light, standalone project so
# EpiAwarePackageTools.jl/JuliaFormatter never become dependencies of the
# root modelling environment (Turing, Mooncake, ...) that every search
# loop re-instantiates.
#
# Checks the repo's own source-formatting directories directly, rather
# than loading `SismidILITuring` and passing the module in: component
# files are assembled by plain `include` (see docs/contracts.md), not
# registered package loading, so `pathof`/`pkgdir` (which
# `test_formatting(mod::Module)` relies on) would not resolve.

using EpiAwarePackageTools

const ROOTDIR = dirname(@__DIR__)
const CHECK_DIRS = [joinpath(ROOTDIR, d) for d in ("src", "test", "docs")]

ok = true
try
    EpiAwarePackageTools.test_formatting(CHECK_DIRS)
catch err
    global ok = false
    println("qa_runner.jl: formatting check failed:")
    showerror(stdout, err)
    println()
end

exit(ok ? 0 : 1)
