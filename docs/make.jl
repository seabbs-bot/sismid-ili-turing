# Documenter site for SismidILITuring.
#
# Robust to the package module not existing/loading yet: it is assembled in
# parallel by another agent (see docs/contracts.md). The narrative site
# (Home, Project, EDA, Reports) always builds; the API section is added
# only when `src/SismidILITuring.jl` is present and loads cleanly.
#
# Run with:  julia --project=docs docs/make.jl

using Documenter

const ROOT = dirname(@__DIR__)
const DOCS_SRC = joinpath(@__DIR__, "src")
const MODULE_FILE = joinpath(ROOT, "src", "SismidILITuring.jl")

# -- keep docs/src/{project,eda,reports} in sync with the real files --------
#
# Each page lives under docs/src as a symlink back to its source file, so the
# markdown is written and owned in one place (docs/, docs/eda/, reports/) and
# the nav below never goes stale: a new reports/NN-*.md or docs/eda/*.md file
# picks up a matching symlink and a nav entry the next time make.jl runs, with
# no edit required here.

function sync_md_symlinks(src_dir::AbstractString, dest_dir::AbstractString)
    mkpath(dest_dir)
    files = filter(f -> endswith(f, ".md"), sort(readdir(src_dir)))
    for f in files
        target = joinpath(src_dir, f)
        link = joinpath(dest_dir, f)
        rel_target = relpath(target, dest_dir)
        if islink(link) && readlink(link) == rel_target
            continue
        end
        rm(link; force = true)
        symlink(rel_target, link)
    end
    return files
end

# Human title for a page file: "01-series-overview.md" -> "01 — Series
# overview"; "steer-log.md" -> "Steer log"; special-cased README/TEMPLATE.
function page_title(filename::AbstractString)
    name = replace(filename, r"\.md$" => "")
    name == "README" && return "Index"
    name == "TEMPLATE" && return "Template"
    m = match(r"^(\d+)-(.+)$", name)
    if m !== nothing
        num, rest = m.captures
        return "$num — " * uppercasefirst(replace(rest, "-" => " "))
    end
    return uppercasefirst(replace(name, r"[-_]" => " "))
end

# -- Project section: fixed order, each file included if present ------------

const PROJECT_FILES = [
    "brief.md", "plan.md", "infrastructure.md", "contracts.md",
    "steer-log.md",
]
sync_md_symlinks(@__DIR__, joinpath(DOCS_SRC, "project"))
project_pages = [
    page_title(f) => joinpath("project", f)
    for f in PROJECT_FILES if isfile(joinpath(@__DIR__, f))
]

# -- EDA section: numeric prefixes sort into the right order -----------------

eda_files = sync_md_symlinks(joinpath(ROOT, "docs", "eda"),
    joinpath(DOCS_SRC, "eda"))
eda_pages = [page_title(f) => joinpath("eda", f) for f in eda_files]

# -- Reports section: index/template first, then loop reports in order ------

reports_dir = joinpath(ROOT, "reports")
all_reports = sync_md_symlinks(reports_dir, joinpath(DOCS_SRC, "reports"))
special = filter(f -> f in ("README.md", "TEMPLATE.md"), all_reports)
loop_reports = sort(filter(f -> !(f in special), all_reports))
report_files = vcat(sort(special), loop_reports)
report_pages = [page_title(f) => joinpath("reports", f) for f in report_files]

# -- API section: only when the package module loads cleanly ----------------

sismid_module = nothing
api_page = joinpath(DOCS_SRC, "api.md")
if isfile(MODULE_FILE)
    # Loaded with a plain `include`, not `Pkg.develop` + `using`: a
    # committed dependency on the local package would need `name`/`uuid`
    # in the root Project.toml to resolve, so a fresh `Pkg.instantiate()`
    # could fail before this robustness logic ever ran. `include` only
    # needs the ordinary deps already listed in docs/Project.toml (the
    # same ones SismidILITuring's component files `using`).
    try
        include(MODULE_FILE)
        if isdefined(Main, :SismidILITuring)
            global sismid_module = Main.SismidILITuring
        end
    catch err
        @warn "SismidILITuring failed to load; building docs without the " *
              "API section" exception = (err, catch_backtrace())
    end
else
    @info "src/SismidILITuring.jl not present yet; building docs without " *
          "the API section"
end

pages = [
    "Home" => "index.md",
    "Project" => project_pages,
    "EDA" => eda_pages,
    "Reports" => report_pages,
]

if sismid_module === nothing
    rm(api_page; force = true)
else
    write(api_page, """
    # API reference

    ```@autodocs
    Modules = [SismidILITuring]
    ```
    """)
    push!(pages, "API" => "api.md")
end

makedocs(;
    sitename = "SismidILITuring",
    pages = pages,
    format = Documenter.HTML(; prettyurls = false, inventory_version = "dev"),
    checkdocs = :none,
)
