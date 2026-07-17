# Figure for 03-seasonality.md: is there a shared "severity year"
# effect, where a season that is severe (high amplitude) at one
# location tends to be severe everywhere? Correlate each location's
# per-season amplitude (peak wILI minus that location's own
# off-season baseline) against every other location's, across the 13
# training-set seasons, and against the cross-location mean.
include("common.jl")
using CairoMakie

d = load_finalized()
d = filter(row -> row.season_year >= 2004, d)
locs = ["US National", ["HHS Region $i" for i in 1:10]...]
short = [replace(l, "HHS Region " => "R", "US National" => "Natl") for l in locs]
seasons = sort(unique(d.season_year))

d.offseason = (d.woy .<= 8) .| (d.woy .>= 45)
baseline = combine(groupby(filter(row -> row.offseason, d), :location), :wili => median => :baseline)

amp = fill(NaN, length(seasons), length(locs))
for (j, loc) in enumerate(locs)
    base = baseline.baseline[findfirst(==(loc), baseline.location)]
    dloc = d[d.location .== loc, :]
    for (i, sy) in enumerate(seasons)
        sub = dloc[dloc.season_year .== sy, :]
        isempty(sub) && continue
        amp[i, j] = maximum(sub.wili) - base
    end
end

cm = cor(amp)

fig = Figure(size=(1500, 750), fontsize=12)
n = length(locs)
ax1 = Axis(fig[1, 1]; title="Cross-location correlation of per-season amplitude",
    xticks=(1:n, short), yticks=(1:n, short), xticklabelrotation=pi/2, aspect=1)
hm = heatmap!(ax1, 1:n, 1:n, cm'; colormap=:Blues, colorrange=(0, 1))
for i in 1:n, j in 1:n
    val = cm[i, j]
    text!(ax1, i, j; text=string(round(val, digits=2)), align=(:center, :center),
        fontsize=8, color=val > 0.6 ? :white : :black)
end
Colorbar(fig[1, 2], hm; label="correlation")

# each location's amplitude vs the cross-location mean amplitude for that season
mean_amp = vec(mean(amp, dims=2))
ax2 = Axis(fig[1, 3]; xlabel="cross-location mean amplitude that season (pp)",
    ylabel="location amplitude, z-scored", title="Amplitude vs a shared severity index")
for (j, loc) in enumerate(locs)
    z = (amp[:, j] .- mean(amp[:, j])) ./ std(amp[:, j])
    scatter!(ax2, mean_amp, z; color=PALETTE[mod1(j, length(PALETTE))], markersize=8,
        label=short[j])
end
axislegend(ax2; position=:lt, framevisible=false, nbanks=2, labelsize=9)

Label(fig[0, 1:3], "Cross-location correlation of seasonal amplitude: a real " *
    "but moderate shared severity-year signal, weaker than the level " *
    "correlations in 04-cross-location (validation + history only, " *
    "n=13 seasons per pair)"; fontsize=14, font=:bold)
save(joinpath(FIG_DIR, "11_amplitude_correlation.png"), fig)
println("saved 11_amplitude_correlation.png")

# print summary for the writeup
offdiag = [cm[i, j] for i in 1:n, j in 1:n if i != j]
println("amplitude correlation: mean=", round(mean(offdiag), digits=2),
    " range=", round(minimum(offdiag), digits=2), " to ", round(maximum(offdiag), digits=2))
natl_idx = findfirst(==("US National"), locs)
for (j, loc) in enumerate(locs)
    j == natl_idx && continue
    println(loc, " vs US National: r=", round(cm[natl_idx, j], digits=2))
end
