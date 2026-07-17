# Round REVIEWER: ranks a round's candidates and picks a winner from
# the on-disk results already written by a per-candidate driver (e.g.
# `experiments/round1_run.jl`'s `aggregate_candidate`) or the
# in-process engine (`experiments/run_round.jl`). Read-only: never
# fits a model, never touches the data or hub clone. Deliberately kept
# free of `using SismidILITuring` so it stays fast to load and can run
# while other candidates are still fitting.
#
# On disk, per candidate, under `results_dir/<candidate_name>/`:
#   summary.txt  -- key=value pooled summary (status, mean_wis, sd_wis,
#                   log_mean_wis, log_sd_wis, coverage50/90,
#                   worst_region(_wis), worst_time(_wis), n_splits,
#                   n_failed_splits) -- see `aggregate_candidate`.
#   region.csv   -- per-location mean/sd WIS breakdown.
#   time.csv     -- per-origin-date mean/sd WIS breakdown, with phase.
#   meta.txt     -- transform/ndraws/Dmax, and optionally a
#                   `params=<n>` or `n_params=<n>` line (not currently
#                   written by any driver; read opportunistically for
#                   the parsimony tie-break, `missing` if absent).
#
# A candidate directory with no `summary.txt` yet is `:missing` (still
# fitting, or never started); one whose `summary.txt` records
# `status=failed` is `:failed`. Both are listed under "needs a fix",
# never silently dropped, so a partially-finished round still produces
# a report shell.
#
# Usage:
#   include("experiments/review_round.jl")
#   winner = review_round("experiments/round1/_results", "round1")

using CSV
using DataFrames
using Dates
using Statistics: mean, std
using Printf: @sprintf

const _EXPERIMENTS_DIR = @__DIR__
const _REPO_ROOT = normpath(joinpath(_EXPERIMENTS_DIR, ".."))

"""Default `reports/<roundname>.md` path, matching every other driver
in this repo (`experiments/README.md`, `reports/TEMPLATE.md`)."""
default_report_path(roundname) =
    joinpath(_REPO_ROOT, "reports", "$(roundname).md")

# How much a candidate's composite (selection) score is inflated per
# unit of "inconsistency": WIS coefficient of variation across tasks,
# plus how much worse its worst region/time is than its own mean. Kept
# as one named constant so the weighting is visible and easy to
# revisit, not buried in an expression (docs/brief.md: guard against
# overfitting on WIS SD; prefer generalisers over one great average).
const PENALTY_WEIGHT = 0.5

# Composite scores within this relative fraction of the leader's are
# treated as a tie and, where param counts are known, broken toward
# the simpler model (docs/brief.md: "prefer simpler models where
# performance is comparable").
const TIE_TOLERANCE = 0.01

# --- reading on-disk candidate results ---------------------------------------

"""Parse a `key=value` text file (summary.txt/meta.txt) into a Dict."""
function _read_kv(path)
    kv = Dict{String,String}()
    for line in readlines(path)
        isempty(line) && continue
        parts = split(line, "="; limit=2)
        length(parts) == 2 && (kv[parts[1]] = parts[2])
    end
    kv
end

_parsef(kv, k) = haskey(kv, k) ? parse(Float64, kv[k]) : NaN
_parsei(kv, k) = haskey(kv, k) ? parse(Int, kv[k]) : 0

"""
    candidate_params(cdir)

Optional parameter count for the parsimony tie-break: a `params=<n>`
or `n_params=<n>` line in `meta.txt`, if a driver ever writes one.
No current driver does, so this normally returns `missing` -- reviewed
gracefully (the tie-break notes that it could not be applied) rather
than guessed at, since guessing model complexity from the file system
without loading the model would be no better than a coin flip.
"""
function candidate_params(cdir)
    path = joinpath(cdir, "meta.txt")
    isfile(path) || return missing
    kv = _read_kv(path)
    for k in ("n_params", "params")
        haskey(kv, k) && return parse(Int, kv[k])
    end
    missing
end

"""
    read_candidate(results_dir, cname) -> NamedTuple

Read one candidate's `summary.txt`/`region.csv`/`time.csv`. `status`
is `:ok`, `:failed` (ran, `summary.txt` says `status=failed`), or
`:missing` (no `summary.txt` written yet -- still fitting or never
started). Never throws on a partial/absent candidate.
"""
function read_candidate(results_dir, cname)
    cdir = joinpath(results_dir, cname)
    summary_path = joinpath(cdir, "summary.txt")
    if !isfile(summary_path)
        return (name=cname, status=:missing,
                reason="no summary.txt yet (still fitting, or the " *
                       "candidate has not been started/aggregated)")
    end
    kv = _read_kv(summary_path)
    if get(kv, "status", "unknown") != "ok"
        return (name=cname, status=:failed,
                reason=get(kv, "error", "summary.txt does not say ok"),
                n_splits=_parsei(kv, "n_splits"),
                n_failed_splits=_parsei(kv, "n_failed_splits"))
    end

    region_path = joinpath(cdir, "region.csv")
    time_path = joinpath(cdir, "time.csv")
    region_df = isfile(region_path) ? CSV.read(region_path, DataFrame) :
                DataFrame()
    time_df = isfile(time_path) ? CSV.read(time_path, DataFrame) :
              DataFrame()

    (
        name=cname, status=:ok,
        transform=get(kv, "transform", "unknown"),
        mean_wis=_parsef(kv, "mean_wis"), sd_wis=_parsef(kv, "sd_wis"),
        log_mean_wis=_parsef(kv, "log_mean_wis"),
        log_sd_wis=_parsef(kv, "log_sd_wis"),
        coverage50=_parsef(kv, "coverage50"),
        coverage90=_parsef(kv, "coverage90"),
        worst_region=get(kv, "worst_region", "?"),
        worst_region_wis=_parsef(kv, "worst_region_wis"),
        worst_time=get(kv, "worst_time", "?"),
        worst_time_wis=_parsef(kv, "worst_time_wis"),
        n_splits=_parsei(kv, "n_splits"),
        n_failed_splits=_parsei(kv, "n_failed_splits"),
        params=candidate_params(cdir),
        region=region_df, time=time_df,
    )
end

"""Every immediate subdirectory of `results_dir`, sorted, i.e. every
candidate that has at least started (has a directory, however
empty)."""
function discover_candidates(results_dir)
    isdir(results_dir) || return String[]
    sort(filter(d -> isdir(joinpath(results_dir, d)), readdir(results_dir)))
end

# --- ranking ------------------------------------------------------------------

"""
    rank_candidates(ok) -> Vector

Rank `:ok` candidates primarily by mean WIS, down-weighted by a
generalisation penalty so a candidate that wins only on average, at
the cost of one bad region or split, does not automatically beat one
that does consistently well everywhere (docs/brief.md: "prefer models
that do generally well ... not just best overall").

`composite = mean_wis * (1 + PENALTY_WEIGHT * penalty)`, where
`penalty` sums three non-negative, unitless "how much worse than my
own mean" terms: the WIS coefficient of variation across tasks, and
the excess of the worst region's/time's mean WIS over the candidate's
own mean WIS. Returns the input, each entry augmented with `penalty`
and `composite`, sorted ascending by `composite` (lower is better).
"""
function rank_candidates(ok)
    augmented = map(ok) do c
        if !(c.mean_wis > 0) || !isfinite(c.mean_wis)
            return merge(c, (penalty=NaN, composite=Inf))
        end
        cv = c.sd_wis / c.mean_wis
        region_excess = max(0.0, c.worst_region_wis / c.mean_wis - 1)
        time_excess = max(0.0, c.worst_time_wis / c.mean_wis - 1)
        penalty = cv + region_excess + time_excess
        composite = c.mean_wis * (1 + PENALTY_WEIGHT * penalty)
        merge(c, (penalty=penalty, composite=composite))
    end
    sort(augmented; by=c -> c.composite)
end

"""
    break_ties_by_parsimony(ranked) -> (ranked, notes)

Among candidates within `TIE_TOLERANCE` of the leader's composite
score, prefer the one with fewest recorded params. If any tied
candidate has no recorded param count, the tie is left as-is (WIS
order) with a note that it needs a manual parsimony check, rather than
guessed at.
"""
function break_ties_by_parsimony(ranked)
    notes = String[]
    (isempty(ranked) || !isfinite(ranked[1].composite)) && return ranked, notes
    best = ranked[1].composite
    tied_idx = findall(c -> isfinite(c.composite) &&
                            abs(c.composite - best) / best <= TIE_TOLERANCE,
                        ranked)
    length(tied_idx) <= 1 && return ranked, notes
    tied = ranked[tied_idx]
    names_str = join([c.name for c in tied], ", ")
    if any(c -> ismissing(c.params), tied)
        push!(notes, "Tie among $names_str within " *
                      "$(round(Int, TIE_TOLERANCE * 100))% composite " *
                      "score, but not every tied candidate has a " *
                      "recorded param count (meta.txt has no `params=` " *
                      "line) -- cannot break the tie on parsimony " *
                      "automatically; keeping WIS-based order, flag for " *
                      "manual review.")
        return ranked, notes
    end
    reordered = copy(ranked)
    reordered[tied_idx] = sort(tied; by=c -> c.params)
    push!(notes, "Tie among $names_str within " *
                  "$(round(Int, TIE_TOLERANCE * 100))% composite score; " *
                  "broke the tie toward parsimony (fewest params): " *
                  "**$(reordered[tied_idx[1]].name)**.")
    reordered, notes
end

"""
    log_scale_report(ranked) -> String

Compares the natural-scale (composite) ranking to a plain log-scale
WIS ranking. Log-scale WIS is report-only and never used to select
(docs/brief.md), but any disagreement in the leader is flagged: it
usually means the leader wins by fitting a few large outbreaks well
rather than doing well proportionally everywhere.
"""
function log_scale_report(ranked)
    isempty(ranked) && return "No `:ok` candidates to compare."
    nat_order = [c.name for c in ranked]
    log_order = [c.name for c in sort(ranked; by=c -> c.log_mean_wis)]
    if nat_order == log_order
        return "Natural-scale and log-scale WIS agree on the full " *
               "candidate ordering. No divergence."
    end
    lines = String[
        "Natural-scale ranking (composite score): " * join(nat_order, " > "),
        "Log-scale ranking (mean log WIS): " * join(log_order, " > "),
    ]
    if nat_order[1] != log_order[1]
        push!(lines, "**Divergence at the top**: the winner would " *
                      "change from `$(nat_order[1])` (natural scale, " *
                      "the selection metric) to `$(log_order[1])` " *
                      "(log scale, report-only). The natural-scale " *
                      "winner stands, but this disagreement is worth " *
                      "investigating before the next round.")
    else
        push!(lines, "Leaders agree (`$(nat_order[1])`); only the " *
                      "order behind the leader differs.")
    end
    join(lines, "\n")
end

# --- per-region / per-time breakdown across candidates -----------------------

"""
    pivot_wis(ok, table_sym, key_col) -> DataFrame

Pivots every `:ok` candidate's `region.csv` (table_sym=:region,
key_col=:location) or `time.csv` (table_sym=:time,
key_col=:origin_date) into one wide table: rows are the key, columns
are candidates, values are mean WIS. Empty DataFrame if no candidate
has that table.
"""
function pivot_wis(ok, table_sym, key_col)
    long = DataFrame()
    for c in ok
        df = getproperty(c, table_sym)
        (nrow(df) == 0 || !hasproperty(df, key_col)) && continue
        append!(long, DataFrame(key=df[:, key_col], candidate=c.name,
                                mean_wis=df[:, :mean_wis]))
    end
    nrow(long) == 0 && return DataFrame()
    wide = unstack(long, :key, :candidate, :mean_wis)
    rename!(wide, :key => key_col)
    candcols = [n for n in names(wide) if n != String(key_col)]
    wide.across_candidates = [
        isempty(collect(skipmissing(Vector(row[candcols])))) ? missing :
            mean(skipmissing(Vector(row[candcols])))
        for row in eachrow(wide)
    ]
    sort!(wide, :across_candidates; rev=true, by=x -> coalesce(x, -Inf))
    wide
end

"""
    worst_tally(ok, field) -> Vector{Pair}

How often each value of `field` (`:worst_region` or `:worst_time`) is
the worst case for some candidate, most-common first: a quick "which
region/time keeps coming up" signal for the failure-modes section.
"""
function worst_tally(ok, field)
    counts = Dict{String,Int}()
    for c in ok
        key = string(getproperty(c, field))
        counts[key] = get(counts, key, 0) + 1
    end
    sort(collect(counts); by=last, rev=true)
end

# --- rendering ----------------------------------------------------------------

_fmt(x) = (x === missing || (x isa Real && !isfinite(x))) ? "n/a" :
          x isa AbstractFloat ? @sprintf("%.4f", x) : string(x)

function df_to_markdown(df::DataFrame)
    nrow(df) == 0 && return "_(no data)_\n"
    cols = names(df)
    io = IOBuffer()
    println(io, "| " * join(cols, " | ") * " |")
    println(io, "|" * repeat("---|", length(cols)))
    for row in eachrow(df)
        println(io, "| " * join([_fmt(row[c]) for c in cols], " | ") * " |")
    end
    String(take!(io))
end

"""
    write_report(report_path, roundname, ranked, failed, missing_,
                 tie_notes, log_note, region_wide, time_wide,
                 worst_region_counts, worst_time_counts, winner)

Render `reports/<roundname>.md` in `reports/TEMPLATE.md`'s shape
(this repo's convention -- see `experiments/README.md`), extended with
the per-region/per-time breakdown and failure-modes sections the
brief asks the reviewer for. Pure string building: no re-scoring.
"""
function write_report(
    report_path, roundname, ranked, failed, missing_, tie_notes, log_note,
    region_wide, time_wide, worst_region_counts, worst_time_counts, winner,
)
    io = IOBuffer()
    println(io, "# $roundname: reviewer report")
    println(io)
    println(io, "- **Date**: $(Dates.today())")
    println(io, "- **Parent loop**: base")
    println(io, "- **Inference**: as recorded per candidate " *
                "(see each candidate's `meta.txt`)")
    println(io, "- **Seasons scored**: validation (2015/16, 2016/17)")
    println(io, "- **Generated by**: `experiments/review_round.jl` " *
                "(`review_round`) -- read-only over on-disk results, " *
                "no model fits")
    println(io)

    println(io, "## What was reviewed")
    println(io)
    n_ok, n_failed, n_missing = length(ranked), length(failed),
                                 length(missing_)
    println(io, "$(n_ok + n_failed + n_missing) candidate " *
                "director$(n_ok + n_failed + n_missing == 1 ? "y" :
                "ies") found: $n_ok ok, $n_failed failed, " *
                "$n_missing not yet aggregated.")
    println(io)

    println(io, "## Results")
    println(io)
    println(io, "Ranked by composite score: mean WIS inflated by a " *
                "generalisation penalty (WIS coefficient of " *
                "variation, plus how much worse the worst region/time " *
                "is than the candidate's own mean), weight " *
                "`PENALTY_WEIGHT=$(PENALTY_WEIGHT)`. Natural scale is " *
                "the selection metric; log-scale WIS is report-only.")
    println(io)
    println(io, "| Rank | Candidate | Mean WIS | WIS SD | Composite | " *
                "Worst region (WIS) | Worst time (WIS) | Log WIS | " *
                "Cov50 | Cov90 | Params | Splits ok/failed |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|")
    if isempty(ranked)
        println(io, "| - | _no `:ok` candidates yet_ | | | | | | | | " *
                    "| | |")
    else
        for (i, c) in enumerate(ranked)
            wr = "$(c.worst_region) ($(_fmt(c.worst_region_wis)))"
            wt = "$(c.worst_time) ($(_fmt(c.worst_time_wis)))"
            splits = "$(c.n_splits - c.n_failed_splits)/$(c.n_splits)" *
                     " ($(c.n_failed_splits) failed)"
            println(io, "| $i | $(c.name) | $(_fmt(c.mean_wis)) | " *
                        "$(_fmt(c.sd_wis)) | $(_fmt(c.composite)) | " *
                        "$wr | $wt | $(_fmt(c.log_mean_wis)) | " *
                        "$(_fmt(c.coverage50)) | $(_fmt(c.coverage90)) | " *
                        "$(_fmt(c.params)) | $splits |")
        end
    end
    println(io)
    if !isempty(tie_notes)
        println(io, "**Parsimony tie-break**:")
        println(io)
        for n in tie_notes
            println(io, "- $n")
        end
        println(io)
    end

    println(io, "## Per-region WIS breakdown")
    println(io)
    println(io, "Mean WIS by location and candidate, worst " *
                "(highest across-candidate mean) region first.")
    println(io)
    print(io, df_to_markdown(region_wide))
    println(io)

    println(io, "## Per-time WIS breakdown")
    println(io)
    println(io, "Mean WIS by forecast origin and candidate, worst " *
                "origin first.")
    println(io)
    print(io, df_to_markdown(time_wide))
    println(io)

    println(io, "## Failure modes")
    println(io)
    if isempty(worst_region_counts) && isempty(worst_time_counts)
        println(io, "No `:ok` candidates to summarise.")
    else
        println(io, "How often each region/time is the WORST case for " *
                    "some candidate (steer toward these for the next " *
                    "round's axes):")
        println(io)
        if !isempty(worst_region_counts)
            println(io, "- Region: " * join(
                ["$k ($v/$n_ok candidates)" for (k, v) in worst_region_counts],
                ", "))
        end
        if !isempty(worst_time_counts)
            println(io, "- Time: " * join(
                ["$k ($v/$n_ok candidates)" for (k, v) in worst_time_counts],
                ", "))
        end
        if nrow(region_wide) > 0 && hasproperty(region_wide, :across_candidates)
            worst_row = first(region_wide)
            println(io, "- Worst region by across-candidate mean WIS: " *
                        "**$(worst_row[1])** " *
                        "($(_fmt(worst_row.across_candidates))).")
        end
        if nrow(time_wide) > 0 && hasproperty(time_wide, :across_candidates)
            worst_row = first(time_wide)
            println(io, "- Worst time by across-candidate mean WIS: " *
                        "**$(worst_row[1])** " *
                        "($(_fmt(worst_row.across_candidates))).")
        end
    end
    println(io)

    println(io, "## Log-scale divergence check")
    println(io)
    println(io, log_note)
    println(io)

    println(io, "## Candidates needing a fix")
    println(io)
    if isempty(failed) && isempty(missing_)
        println(io, "None: every candidate directory has an `ok` " *
                    "summary.")
    else
        for r in failed
            println(io, "- **$(r.name)** (`:failed`): $(r.reason)")
        end
        for r in missing_
            println(io, "- **$(r.name)** (`:missing`): $(r.reason)")
        end
    end
    println(io)

    println(io, "## Decision")
    println(io)
    if winner === nothing
        println(io, "_No `:ok` candidate yet -- nothing to decide. " *
                    "Re-run once at least one candidate's `summary.txt` " *
                    "exists._")
    else
        w = ranked[1]
        println(io, "**Winner: `$(w.name)`.**")
        println(io)
        println(io, "Lowest composite score ($(_fmt(w.composite))): " *
                    "mean WIS $(_fmt(w.mean_wis)), SD $(_fmt(w.sd_wis)) " *
                    "(CV $(_fmt(w.sd_wis / w.mean_wis))), worst region " *
                    "$(w.worst_region) (WIS $(_fmt(w.worst_region_wis))), " *
                    "worst time $(w.worst_time) " *
                    "(WIS $(_fmt(w.worst_time_wis))).")
        if length(ranked) > 1
            println(io, "Runner-up: `$(ranked[2].name)` " *
                        "(composite $(_fmt(ranked[2].composite))).")
        end
        println(io)
        if !isempty(tie_notes)
            println(io, "See the parsimony tie-break note above.")
        end
        println(io, "See Failure modes above for where to focus the " *
                    "next round's axes.")
    end
    println(io)

    println(io, "## Artifacts")
    println(io)
    println(io, "- Report: `reports/$(roundname).md`")
    println(io, "- Reviewer: `experiments/review_round.jl`")

    mkpath(dirname(report_path))
    open(report_path, "w") do f
        Base.write(f, String(take!(io)))
    end
    report_path
end

# --- top-level entry point ---------------------------------------------------

"""
    review_round(results_dir, roundname;
                 report_path=default_report_path(roundname)) -> winner

Read every candidate under `results_dir` (one subdirectory per
candidate, written by a driver such as `round1_run.jl`'s
`aggregate_candidate`), rank the `:ok` ones, write
`reports/<roundname>.md`, and return the winner's name (a `String`),
or `nothing` if no candidate has finished aggregating yet.

LIGHT: no model fits, no re-scoring -- reads whatever
`summary.txt`/`region.csv`/`time.csv` are already on disk. Safe to
call against a partially-finished round (missing/failed candidates
are listed under "needs a fix", never dropped silently) and against
an empty results directory (produces a report shell and returns
`nothing`).
"""
function review_round(results_dir, roundname;
                       report_path=default_report_path(roundname))
    cnames = discover_candidates(results_dir)
    all_read = [read_candidate(results_dir, c) for c in cnames]

    ok = [c for c in all_read if c.status == :ok]
    failed = [c for c in all_read if c.status == :failed]
    missing_ = [c for c in all_read if c.status == :missing]

    ranked = rank_candidates(ok)
    ranked, tie_notes = break_ties_by_parsimony(ranked)
    log_note = log_scale_report(ranked)

    region_wide = pivot_wis(ok, :region, :location)
    time_wide = pivot_wis(ok, :time, :origin_date)
    worst_region_counts = worst_tally(ok, :worst_region)
    worst_time_counts = worst_tally(ok, :worst_time)

    winner = isempty(ranked) ? nothing : ranked[1].name

    write_report(report_path, roundname, ranked, failed, missing_, tie_notes,
                 log_note, region_wide, time_wide, worst_region_counts,
                 worst_time_counts, winner)

    winner
end
