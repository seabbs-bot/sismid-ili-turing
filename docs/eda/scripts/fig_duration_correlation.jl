# Figure for 03-seasonality.md / 04-cross-location.md: completes the
# "shared vs independent year effect" picture alongside amplitude
# (mean r=0.68) and onset week (mean r=0.24): is season *duration*
# (offset minus onset) also a shared year property, or independent
# per location? (Raw peak height is NOT tested here: since a
# location's off-season baseline is season-invariant, correlating
# raw peak height across locations gives the exact same correlation
# matrix as amplitude -- an additive per-series constant does not
# change a correlation -- so it would be a redundant check.)
include("common.jl")
using CairoMakie, Statistics

d = load_finalized()
d = filter(row -> row.season_year >= 2004, d)
locs = ["US National", ["HHS Region $i" for i in 1:10]...]
short = [replace(l, "HHS Region " => "R", "US National" => "Natl") for l in locs]
seasons = sort(unique(d.season_year))

d.offseason = (d.woy .<= 8) .| (d.woy .>= 45)
baseline = combine(groupby(filter(row -> row.offseason, d), :location), :wili => median => :baseline)

function onset_offset(woy::Vector{Int}, wili::Vector{Float64}, thresh::Float64)
    ord = sortperm(woy)
    w, v = woy[ord], wili[ord]
    above = v .> thresh
    runs = Tuple{Int,Int}[]
    i = 1
    n = length(above)
    while i <= n
        if above[i]
            j = i
            while j <= n && above[j]
                j += 1
            end
            (j - i) >= 2 && push!(runs, (i, j - 1))
            i = j
        else
            i += 1
        end
    end
    isempty(runs) && return (missing, missing)
    return (w[runs[1][1]], w[runs[end][2]])
end

duration = fill(NaN, length(seasons), length(locs))
for (j, loc) in enumerate(locs)
    base = baseline.baseline[findfirst(==(loc), baseline.location)]
    thresh = 1.5 * base
    dloc = d[d.location .== loc, :]
    for (i, sy) in enumerate(seasons)
        sub = dloc[dloc.season_year .== sy, :]
        isempty(sub) && continue
        on, off = onset_offset(sub.woy, sub.wili, thresh)
        ismissing(on) && continue
        duration[i, j] = off - on
    end
end

dur_cor = cor(duration)
n = length(locs)

fig = Figure(size=(1500, 750), fontsize=12)
ax1 = Axis(fig[1, 1]; title="Cross-location correlation of season duration",
    xticks=(1:n, short), yticks=(1:n, short), xticklabelrotation=pi/2, aspect=1)
hm = heatmap!(ax1, 1:n, 1:n, dur_cor'; colormap=:Blues, colorrange=(-0.4, 1))
for i in 1:n, j in 1:n
    val = dur_cor[i, j]
    text!(ax1, i, j; text=string(round(val, digits=2)), align=(:center, :center),
        fontsize=8, color=val > 0.6 ? :white : :black)
end
Colorbar(fig[1, 2], hm; label="correlation")

ax2 = Axis(fig[1, 3]; xlabel="season", ylabel="duration (weeks)",
    title="Season duration by location",
    xticks=(1:length(seasons), string.(seasons .- 2000)), xticklabelrotation=pi/3)
for (j, loc) in enumerate(locs)
    lines!(ax2, 1:length(seasons), duration[:, j];
        color=(PALETTE[mod1(j, length(PALETTE))], 0.7), linewidth=1.5)
end
natl_j = findfirst(==("US National"), locs)
lines!(ax2, 1:length(seasons), duration[:, natl_j]; color=:black, linewidth=3,
    label="US National")
axislegend(ax2; position=:lb, framevisible=false)

Label(fig[0, 1:3], "Season duration (offset - onset): a weak-to-moderate " *
    "shared-year signal, between amplitude (r=0.68) and onset timing " *
    "(r=0.24) in strength (validation + history only)"; fontsize=14, font=:bold)
save(joinpath(FIG_DIR, "14_duration_correlation.png"), fig)
println("saved 14_duration_correlation.png")

offdiag = [dur_cor[i, j] for i in 1:n, j in 1:n if i != j]
println("duration correlation: mean=", round(mean(offdiag), digits=2),
    " range=", round(minimum(offdiag), digits=2), " to ", round(maximum(offdiag), digits=2))
for (j, loc) in enumerate(locs)
    j == natl_j && continue
    println(loc, " duration vs US National: r=", round(dur_cor[natl_j, j], digits=2))
end
