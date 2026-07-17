# Figure for 06-regional-heterogeneity.md: ordered dot charts of three
# per-location summary statistics (off-season baseline, mean seasonal
# amplitude, differenced-series volatility) so regional heterogeneity
# is visible at a glance, alongside each other.
include("common.jl")
using CairoMakie

d = load_finalized()
d = filter(row -> row.season_year >= 2004, d)  # drop partial first season
locs = ["US National", ["HHS Region $i" for i in 1:10]...]
short = [replace(l, "HHS Region " => "R", "US National" => "Natl") for l in locs]

# --- baseline: median wILI over off-season weeks (woy 1-8, 45-52) ---
d.offseason = (d.woy .<= 8) .| (d.woy .>= 45)
baseline = combine(groupby(filter(row -> row.offseason, d), :location),
    :wili => median => :baseline)

# --- amplitude: per-season peak minus off-season baseline, averaged ---
seasons = sort(unique(d.season_year))
amp_rows = NamedTuple[]
for loc in locs
    dloc = d[d.location .== loc, :]
    base = baseline.baseline[findfirst(==(loc), baseline.location)]
    amps = Float64[]
    for sy in seasons
        sub = dloc[dloc.season_year .== sy, :]
        isempty(sub) && continue
        push!(amps, maximum(sub.wili) - base)
    end
    push!(amp_rows, (location=loc, amplitude=mean(amps)))
end
amplitude = DataFrame(amp_rows)

# --- volatility: SD of week-to-week difference of log(wili) ---
d.logwili = log.(d.wili .+ 0.01)
vol_rows = NamedTuple[]
for loc in locs
    sub = sort(d[d.location .== loc, :], :origin_date)
    push!(vol_rows, (location=loc, volatility=std(diff(sub.logwili))))
end
volatility = DataFrame(vol_rows)

function dot_panel!(fig, pos, df, valcol, title_str, xlabel_str, colour)
    df = sort(df, valcol)
    n = nrow(df)
    lab = [replace(l, "HHS Region " => "R", "US National" => "Natl")
           for l in df.location]
    ax = Axis(fig[pos...]; title=title_str, xlabel=xlabel_str,
        yticks=(1:n, lab))
    vals = df[:, valcol]
    for i in 1:n
        lines!(ax, [0, vals[i]], [i, i]; color=(:gray, 0.4), linewidth=1.5)
    end
    scatter!(ax, vals, 1:n; color=colour, markersize=14)
    xlims!(ax, 0, maximum(vals) * 1.15)
    return ax
end

fig = Figure(size=(1500, 550), fontsize=13)
dot_panel!(fig, (1, 1), baseline, :baseline,
    "Off-season baseline level", "median wILI, woy 1-8/45-52 (%)", PALETTE[1])
dot_panel!(fig, (1, 2), amplitude, :amplitude,
    "Mean seasonal amplitude", "mean(season peak - baseline) (pp)", PALETTE[4])
dot_panel!(fig, (1, 3), volatility, :volatility,
    "Differenced-series volatility", "SD of Δlog(wILI)", PALETTE[7])

Label(fig[0, 1:3], "Regional heterogeneity: baseline, amplitude and " *
    "week-to-week volatility all vary several-fold across locations, and " *
    "the three orderings are not the same (validation + history only)";
    fontsize=15, font=:bold)
save(joinpath(FIG_DIR, "06_regional_heterogeneity.png"), fig)
println("saved 06_regional_heterogeneity.png")
