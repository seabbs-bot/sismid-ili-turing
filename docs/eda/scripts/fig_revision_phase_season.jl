# Figure for 02-backfill.md: does the direction reversal seen for
# Region 1/Region 4 between 2015/16 and 2016/17 (see the "revision
# structure by location and by tracked season" section) concentrate
# in a particular part of the season, or hold across both peak and
# off-season weeks? Delay-1 % upward revision, split by settled-value
# phase (above/below that location's own median) x tracked season,
# faceted by location.
include("common.jl")
using CairoMakie

d = load_versions()
gd = groupby(d, [:location, :origin_date])
counts = combine(gd, nrow => :nver)
mult_keys = filter(row -> row.nver > 1, counts)
mult = innerjoin(d, select(mult_keys, [:location, :origin_date]), on=[:location, :origin_date])
gd2 = groupby(mult, [:location, :origin_date])
settled = combine(gd2) do sub
    i = argmax(sub.delay)
    (settled_wili = sub.wili[i], settled_delay = sub.delay[i])
end
rev = innerjoin(mult, settled, on=[:location, :origin_date])
rev.revision = rev.wili .- rev.settled_wili
rev1 = filter(row -> row.delay == 1 && row.settled_wili > 0.05, rev)

locmed = combine(groupby(rev1, :location), :settled_wili => median => :locmed)
rev1 = leftjoin(rev1, locmed, on=:location)
rev1.phase = ifelse.(rev1.settled_wili .>= rev1.locmed, "high", "low")

locs = ["US National", ["HHS Region $i" for i in 1:10]...]
season_colours = Dict(2015 => PALETTE[1], 2016 => PALETTE[4])

fig = Figure(size=(1500, 1250), fontsize=12)
for (i, loc) in enumerate(locs)
    row, col = divrem(i - 1, 4)
    ax = Axis(fig[row + 1, col + 1]; title=loc,
        xlabel = row == 2 ? "" : "",
        ylabel = col == 0 ? "% delay-1 revisions upward" : "",
        xticks=(1:2, ["low season", "high season"]))
    dloc = rev1[rev1.location .== loc, :]
    for sy in (2015, 2016)
        sub = filter(r -> r.season_year == sy, dloc)
        isempty(sub) && continue
        pu = [100 * mean(filter(r -> r.phase == ph, sub).revision .> 0)
              for ph in ("low", "high")]
        ns = [nrow(filter(r -> r.phase == ph, sub)) for ph in ("low", "high")]
        lines!(ax, 1:2, pu; color=season_colours[sy], linewidth=2)
        scatter!(ax, 1:2, pu; color=season_colours[sy], markersize=12,
            label=sy == 2015 ? "2015/16" : "2016/17")
        for (xi, (p, n)) in enumerate(zip(pu, ns))
            text!(ax, xi, p; text="n=$n", fontsize=8, offset=(0, 8),
                align=(:center, :bottom))
        end
    end
    hlines!(ax, [50]; color=:gray, linestyle=:dash, linewidth=1)
    ylims!(ax, 0, 100)
    xlims!(ax, 0.7, 2.3)
end
Legend(fig[1, 5],
    [LineElement(color=season_colours[2015], linewidth=2),
     LineElement(color=season_colours[2016], linewidth=2)],
    ["2015/16", "2016/17"]; framevisible=false)
Label(fig[0, 1:4], "Delay-1 revision direction by season-phase and tracked " *
    "training season, per location: Region 1 and Region 4's reversal " *
    "(see text) holds across both phases within each season, so it is a " *
    "season-level shift, not a peak-vs-off-season effect (validation + " *
    "history only)"; fontsize=14, font=:bold)
save(joinpath(FIG_DIR, "10_revision_phase_season.png"), fig)
println("saved 10_revision_phase_season.png")

# print numbers for the writeup
using Statistics
for loc in ("HHS Region 1", "HHS Region 4")
    dloc = rev1[rev1.location .== loc, :]
    for sy in (2015, 2016)
        for ph in ("low", "high")
            sub = filter(r -> r.season_year == sy && r.phase == ph, dloc)
            isempty(sub) && continue
            println(loc, " ", sy, " ", ph, ": n=", nrow(sub),
                " %up=", round(100*mean(sub.revision .> 0), digits=0))
        end
    end
end
