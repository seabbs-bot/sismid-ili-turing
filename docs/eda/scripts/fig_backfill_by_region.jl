# Figure for 02-backfill.md: revision size vs delay, faceted by
# location (small multiples), split by the two tracked training
# seasons (2015/16 vs 2016/17), to show how much the backfill profile
# itself differs season to season within each location.
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
rev.abs_revision = abs.(rev.revision)
rev_early = filter(row -> row.delay <= 15 && row.delay < row.settled_delay &&
    row.settled_wili > 0.05, rev)

locs = ["US National", ["HHS Region $i" for i in 1:10]...]
season_colours = Dict(2015 => PALETTE[1], 2016 => PALETTE[4])

fig = Figure(size=(1500, 1250), fontsize=12)
for (i, loc) in enumerate(locs)
    row, col = divrem(i - 1, 4)
    ax = Axis(fig[row + 1, col + 1]; title=loc,
        xlabel = row == 2 ? "reporting delay (weeks)" : "",
        ylabel = col == 0 ? "median abs. revision (pp)" : "")
    dloc = rev_early[rev_early.location .== loc, :]
    for sy in (2015, 2016)
        sub = combine(groupby(filter(r -> r.season_year == sy, dloc), :delay),
            :abs_revision => median => :m)
        isempty(sub) && continue
        sort!(sub, :delay)
        lines!(ax, sub.delay, sub.m; color=season_colours[sy], linewidth=1.8,
            label=sy == 2015 ? "2015/16" : "2016/17")
    end
    xlims!(ax, 1, 15)
end
Legend(fig[1, 5],
    [LineElement(color=season_colours[2015], linewidth=2),
     LineElement(color=season_colours[2016], linewidth=2)],
    ["2015/16", "2016/17"]; framevisible=false)
Label(fig[0, 1:4], "Backfill revision size by delay, per location and " *
    "per tracked training season: the two seasons' profiles differ " *
    "within most locations, on top of the larger cross-location " *
    "differences (validation + history only)"; fontsize=15, font=:bold)
save(joinpath(FIG_DIR, "07_backfill_by_region.png"), fig)
println("saved 07_backfill_by_region.png")
