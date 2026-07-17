# Figure for 02-backfill.md: 02-backfill's location-varying revision
# sizes (e.g. Region 2's delay-1 revision 3-7x every other location)
# are all measured on the raw wILI percentage scale, but the model
# fits on a transformed scale (fourth-root recommended in
# 01-series-overview). Direction (sign) of a revision cannot change
# under a monotonic transform, so the non-monotonic-direction and
# season-reversal findings already in 02-backfill hold on any scale.
# But *magnitude* can: this checks whether the location ranking of
# revision size compresses once measured in fourth-root units instead
# of raw percentage points, which matters for whether the model needs
# an extreme location-varying revision-scale parameter or whether the
# transform already absorbs most of the location spread.
include("common.jl")
using CairoMakie, Statistics

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
rev1 = filter(row -> row.delay == 1 && row.settled_wili > 0.05, rev)

fourthroot(x) = max(x, 0.0)^0.25
rev1.raw_abs = abs.(rev1.wili .- rev1.settled_wili)
rev1.fr_abs = abs.(fourthroot.(rev1.wili) .- fourthroot.(rev1.settled_wili))

locs = ["US National", ["HHS Region $i" for i in 1:10]...]
short = [replace(l, "HHS Region " => "R", "US National" => "Natl") for l in locs]

raw_med = Float64[]
fr_med = Float64[]
for loc in locs
    sub = filter(r -> r.location == loc, rev1)
    push!(raw_med, median(sub.raw_abs))
    push!(fr_med, median(sub.fr_abs))
end
raw_pooled = median(rev1.raw_abs)
fr_pooled = median(rev1.fr_abs)
raw_norm = raw_med ./ raw_pooled
fr_norm = fr_med ./ fr_pooled

ord = sortperm(raw_norm, rev=true)

fig = Figure(size=(1400, 600), fontsize=13)
ax = Axis(fig[1, 1]; xlabel="location (sorted by raw-scale revision size)",
    ylabel="median abs. delay-1 revision, relative to pooled median",
    title="Location-varying revision size: raw % vs fourth-root scale",
    xticks=(1:length(locs), short[ord]))
barplot!(ax, (1:length(locs)) .- 0.15, raw_norm[ord]; width=0.28,
    color=PALETTE[1], label="raw wILI (%)")
barplot!(ax, (1:length(locs)) .+ 0.15, fr_norm[ord]; width=0.28,
    color=PALETTE[4], label="fourth-root")
hlines!(ax, [1.0]; color=:gray, linestyle=:dash, linewidth=1)
axislegend(ax; position=:rt, framevisible=false)

Label(fig[0, 1], "Delay-1 revision size relative to the pooled median, raw " *
    "vs fourth-root scale: the location spread compresses on the " *
    "modelling scale but the same locations stay the extremes " *
    "(validation + history only)"; fontsize=14, font=:bold)
save(joinpath(FIG_DIR, "15_backfill_scale.png"), fig)
println("saved 15_backfill_scale.png")

println("raw-scale range (relative to pooled): ", round(minimum(raw_norm), digits=2),
    " to ", round(maximum(raw_norm), digits=2),
    " (ratio ", round(maximum(raw_norm) / minimum(raw_norm), digits=1), "x)")
println("fourthroot-scale range (relative to pooled): ", round(minimum(fr_norm), digits=2),
    " to ", round(maximum(fr_norm), digits=2),
    " (ratio ", round(maximum(fr_norm) / minimum(fr_norm), digits=1), "x)")
for i in ord
    println(locs[i], ": raw=", round(raw_norm[i], digits=2), " fourthroot=", round(fr_norm[i], digits=2))
end
