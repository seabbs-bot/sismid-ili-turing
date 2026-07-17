# Figure for 05-autocorrelation.md: ACF/PACF of the deseasonalised
# residual and its first difference, for a spread of locations
# (chosen to span the AR-order table: US National and HHS Region 8
# select order 4, HHS Region 2 selects order 8, HHS Region 7 selects
# order 10).
include("common.jl")
using CairoMakie, StatsBase

d = load_finalized()
d = filter(row -> row.season_year >= 2004, d)  # drop partial first season
d.logwili = log.(d.wili .+ 0.01)

woymean = combine(groupby(d, [:location, :woy]), :logwili => mean => :woymean)
d = leftjoin(d, woymean, on=[:location, :woy])
d.resid = d.logwili .- d.woymean

locs = ["US National", "HHS Region 8", "HHS Region 2", "HHS Region 7"]
nlags = 20

function acf_panel!(fig, pos, x, nlags, title_str; colour)
    n = length(x)
    ac = autocor(x, 1:nlags)
    bound = 1.96 / sqrt(n)
    ax = Axis(fig[pos...]; title=title_str, xlabel="lag", ylabel="ACF")
    barplot!(ax, 1:nlags, ac; color=colour, width=0.6)
    hlines!(ax, [0.0]; color=:black, linewidth=1)
    hlines!(ax, [bound, -bound]; color=:gray, linestyle=:dash, linewidth=1)
    ylims!(ax, -1, 1)
    return ax
end

function pacf_panel!(fig, pos, x, nlags, title_str; colour)
    n = length(x)
    pc = pacf(x, 1:nlags)
    bound = 1.96 / sqrt(n)
    ax = Axis(fig[pos...]; title=title_str, xlabel="lag", ylabel="PACF")
    barplot!(ax, 1:nlags, pc; color=colour, width=0.6)
    hlines!(ax, [0.0]; color=:black, linewidth=1)
    hlines!(ax, [bound, -bound]; color=:gray, linestyle=:dash, linewidth=1)
    ylims!(ax, -1, 1)
    return ax
end

fig = Figure(size=(1500, 1400), fontsize=12)
for (i, loc) in enumerate(locs)
    sub = sort(d[d.location .== loc, :], :origin_date)
    resid = sub.resid
    dresid = diff(resid)
    colour = PALETTE[i]

    acf_panel!(fig, (i, 1), resid, nlags, "$loc: ACF (residual)"; colour=colour)
    pacf_panel!(fig, (i, 2), resid, nlags, "$loc: PACF (residual)"; colour=colour)
    acf_panel!(fig, (i, 3), dresid, nlags, "$loc: ACF (1st diff. residual)";
        colour=colour)
end

Label(fig[0, 1:3], "Deseasonalised residual: ACF decays gradually and PACF " *
    "cuts off near lag 1, but AIC still favours higher AR order; " *
    "differencing induces a negative lag-1 ACF (over-differencing signature)";
    fontsize=15, font=:bold)
save(joinpath(FIG_DIR, "05_autocorrelation.png"), fig)
println("saved 05_autocorrelation.png")
