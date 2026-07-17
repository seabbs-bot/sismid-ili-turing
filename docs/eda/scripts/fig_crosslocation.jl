# Figures for 04-cross-location.md: correlation heatmaps for levels and
# for week-to-week differences of log(wILI).
include("common.jl")
using CairoMakie

d = load_finalized()
d.logwili = log.(d.wili .+ 0.01)
wide = unstack(d, :origin_date, :location, :logwili)
sort!(wide, :origin_date)
locs = names(wide)[2:end]
short = [replace(l, "HHS Region " => "R", "US National" => "Natl") for l in locs]
mat = Matrix(wide[:, locs])

cm = cor(mat)
diffmat = diff(mat, dims=1)
cd = cor(diffmat)

function heatmap_panel(fig, pos, m, title_str)
    n = size(m, 1)
    ax = Axis(fig[pos...]; title=title_str, xticks=(1:n, short), yticks=(1:n, short),
        xticklabelrotation=pi/2, aspect=1)
    hm = heatmap!(ax, 1:n, 1:n, m'; colormap=:Blues, colorrange=(0, 1))
    for i in 1:n, j in 1:n
        val = m[i, j]
        txt_colour = val > 0.6 ? :white : :black
        text!(ax, i, j; text=string(round(val, digits=2)), align=(:center, :center),
            fontsize=9, color=txt_colour)
    end
    return hm
end

fig = Figure(size=(1500, 800), fontsize=12)
hm1 = heatmap_panel(fig, (1, 1), cm, "Levels: cor(log wILI)")
hm2 = heatmap_panel(fig, (1, 2), cd, "Week-to-week changes: cor(Δ log wILI)")
Colorbar(fig[1, 3], hm1; label="correlation")
Label(fig[0, 1:3], "Cross-location correlation: high for levels (shared " *
    "seasonality), moderate and uneven once differenced (validation + " *
    "history only)"; fontsize=15, font=:bold)
save(joinpath(FIG_DIR, "04_cross_location_correlation.png"), fig)
println("saved 04_cross_location_correlation.png")
