# Figures for 02-backfill.md: revision size and direction by delay,
# pooled and for a few example locations (Region 2 upward-biased,
# Region 9 downward-biased, US National as a baseline).
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
rev_early = filter(row -> row.delay <= 20 && row.delay < row.settled_delay && row.settled_wili > 0.05, rev)

example_locs = ["US National", "HHS Region 2", "HHS Region 9"]
colours = Dict(zip(["pooled"; example_locs], PALETTE[1:4]))

# --- panel 1: median absolute revision vs delay ---
fig = Figure(size=(1300, 550), fontsize=13)
ax1 = Axis(fig[1, 1]; xlabel="reporting delay (weeks)",
    ylabel="median absolute revision (wILI pp)", title="Revision size by delay")
pooled_byd = combine(groupby(rev_early, :delay), :abs_revision => median => :m)
sort!(pooled_byd, :delay)
lines!(ax1, pooled_byd.delay, pooled_byd.m; color=colours["pooled"], linewidth=2.5,
    label="pooled (all locations)")
for loc in example_locs
    sub = combine(groupby(filter(row -> row.location == loc, rev_early), :delay),
        :abs_revision => median => :m)
    sort!(sub, :delay)
    lines!(ax1, sub.delay, sub.m; color=colours[loc], linewidth=1.8, label=loc)
end
xlims!(ax1, 1, 20)
axislegend(ax1; position=:rt, framevisible=false)

# --- panel 2: fraction of upward revisions by delay ---
ax2 = Axis(fig[1, 2]; xlabel="reporting delay (weeks)",
    ylabel="fraction of revisions that are upward", title="Revision direction by delay")
hlines!(ax2, [0.5]; color=:gray, linestyle=:dash, linewidth=1)
pooled_up = combine(groupby(rev_early, :delay), :revision => (x -> mean(x .> 0)) => :fu)
sort!(pooled_up, :delay)
lines!(ax2, pooled_up.delay, pooled_up.fu; color=colours["pooled"], linewidth=2.5,
    label="pooled (all locations)")
for loc in example_locs
    sub = combine(groupby(filter(row -> row.location == loc, rev_early), :delay),
        :revision => (x -> mean(x .> 0)) => :fu)
    sort!(sub, :delay)
    lines!(ax2, sub.delay, sub.fu; color=colours[loc], linewidth=1.8, label=loc)
end
xlims!(ax2, 1, 20)
ylims!(ax2, 0, 1)
axislegend(ax2; position=:rb, framevisible=false)

Label(fig[0, 1:2], "Backfill revisions are non-monotonic and location-varying " *
    "(validation + history only)"; fontsize=15, font=:bold)
save(joinpath(FIG_DIR, "02_backfill_revisions.png"), fig)
println("saved 02_backfill_revisions.png")
