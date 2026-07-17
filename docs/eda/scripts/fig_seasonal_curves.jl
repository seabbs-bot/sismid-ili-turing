# Figure for 03-seasonality.md: seasonal wILI curves by location, seasons
# overlaid (sequential colour = chronological order), season peaks marked.
include("common.jl")
using CairoMakie

d = load_finalized()
d = filter(row -> row.season_year >= 2004, d)  # drop partial first season
seasons = sort(unique(d.season_year))
locs = ["US National", ["HHS Region $i" for i in 1:10]...]

n_lo, n_hi = extrema(seasons)
seq_colour(sy) = cgrad(:Blues, range(0.35, 1.0, length=length(seasons)))[
    findfirst(==(sy), seasons)]

fig = Figure(size=(1500, 1250), fontsize=13)
for (i, loc) in enumerate(locs)
    row, col = divrem(i - 1, 4)
    ax = Axis(fig[row + 1, col + 1]; title=loc,
        xlabel = row == 2 ? "week of season" : "",
        ylabel = col == 0 ? "wILI (%)" : "",
        xticks = 0:13:52)
    dloc = d[d.location .== loc, :]
    for sy in seasons
        sub = sort(dloc[dloc.season_year .== sy, :], :woy)
        isempty(sub) && continue
        lines!(ax, sub.woy, sub.wili; color=seq_colour(sy), linewidth=1.3)
        pk = argmax(sub.wili)
        scatter!(ax, [sub.woy[pk]], [sub.wili[pk]];
            color=seq_colour(sy), markersize=7, strokewidth=0.8,
            strokecolor=:black)
    end
    xlims!(ax, 0, 53)
end
Colorbar(fig[1:3, 5]; colormap=cgrad(:Blues, range(0.35, 1.0, length=length(seasons))),
    limits=(n_lo, n_hi + 1), label="season (start year)", ticks=n_lo:2:n_hi)
Label(fig[0, 1:4], "Seasonal wILI curves by location, 2004/05-2016/17 seasons " *
    "(validation + history only); dots mark each season's peak week";
    fontsize=15, font=:bold)
save(joinpath(FIG_DIR, "03_seasonal_curves.png"), fig)
println("saved 03_seasonal_curves.png")
