# Figure for 05-autocorrelation.md: does the pooled ACF story (lag-1
# persistence ~0.78-0.96; differencing induces a negative lag-1 ACF
# except at Region 4/Region 5) hold within every individual season,
# or does it drift? Heatmaps of per-location, per-season lag-1 ACF,
# on the undifferenced and first-differenced deseasonalised residual.
include("common.jl")
using CairoMakie, StatsBase

d = load_finalized()
d = filter(row -> row.season_year >= 2004, d)  # drop partial first season
d.logwili = log.(d.wili .+ 0.01)
woymean = combine(groupby(d, [:location, :woy]), :logwili => mean => :woymean)
d = leftjoin(d, woymean, on=[:location, :woy])
d.resid = d.logwili .- d.woymean

locs = ["US National", ["HHS Region $i" for i in 1:10]...]
short = [replace(l, "HHS Region " => "R", "US National" => "Natl") for l in locs]
seasons = sort(unique(d.season_year))

lag1_level = fill(NaN, length(locs), length(seasons))
lag1_diff = fill(NaN, length(locs), length(seasons))
for (i, loc) in enumerate(locs), (j, sy) in enumerate(seasons)
    sub = sort(d[(d.location .== loc) .& (d.season_year .== sy), :], :origin_date)
    nrow(sub) < 10 && continue
    r = Float64.(sub.resid)
    lag1_level[i, j] = autocor(r, 1:1)[1]
    lag1_diff[i, j] = autocor(diff(r), 1:1)[1]
end

function heat_panel(fig, pos, m, title_str, colorrange)
    n_loc, n_s = size(m)
    ax = Axis(fig[pos...]; title=title_str, yticks=(1:n_loc, short),
        xticks=(1:n_s, string.(seasons .- 2000) .* "/" .* lpad.(string.((seasons .+ 1) .- 2000), 2, "0"),),
        xticklabelrotation=pi/3)
    hm = heatmap!(ax, 1:n_s, 1:n_loc, permutedims(m); colormap=:RdBu, colorrange=colorrange)
    for i in 1:n_loc, j in 1:n_s
        isnan(m[i, j]) && continue
        val = m[i, j]
        text!(ax, j, i; text=string(round(val, digits=2)), align=(:center, :center),
            fontsize=8, color=:black)
    end
    return hm
end

fig = Figure(size=(1600, 900), fontsize=12)
hm1 = heat_panel(fig, (1, 1), lag1_level, "Lag-1 ACF, undifferenced residual, by season", (-1, 1))
hm2 = heat_panel(fig, (1, 2), lag1_diff, "Lag-1 ACF, 1st-differenced residual, by season", (-1, 1))
Colorbar(fig[1, 3], hm1; label="lag-1 ACF")
Label(fig[0, 1:2], "Per-season lag-1 ACF: persistence (left) is fairly stable " *
    "season to season, but the differencing sign signature (right) is " *
    "noisier per season than the pooled estimate suggests, including for " *
    "Region 4/Region 5 (validation + history only)"; fontsize=15, font=:bold)
save(joinpath(FIG_DIR, "09_acf_season_drift.png"), fig)
println("saved 09_acf_season_drift.png")

# print summary stats used in the writeup
using Statistics
for (i, loc) in enumerate(locs)
    vals = filter(!isnan, lag1_diff[i, :])
    npos = count(>(0), vals)
    println(loc, ": diff lag-1 ACF season range ", round(minimum(vals), digits=2),
        " to ", round(maximum(vals), digits=2), "; ", npos, "/", length(vals), " seasons positive")
end
