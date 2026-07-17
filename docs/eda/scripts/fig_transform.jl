# Figures for 01-series-overview.md: Taylor's power law fit, and the
# flatness comparison across identity/sqrt/fourth-root/log.
include("common.jl")
using CairoMakie, LinearAlgebra

d = load_finalized()
d = filter(row -> row.season_year >= 2004, d)

# --- panel 1: Taylor's power law ---
cells = combine(groupby(d, [:location, :woy]),
    :wili => mean => :local_mean, :wili => var => :local_var, nrow => :n)
cells = filter(row -> row.n >= 8 && row.local_mean > 0 && row.local_var > 0, cells)
logmean = log.(cells.local_mean)
logvar = log.(cells.local_var)
X = hcat(ones(length(logmean)), logmean)
beta = X \ logvar
lambda = beta[2]

fig = Figure(size=(1350, 600), fontsize=13)
ax1 = Axis(fig[1, 1]; xlabel="log(local mean wILI)", ylabel="log(local variance)",
    title="Taylor's power law (each point = one location x week-of-season cell)")
scatter!(ax1, logmean, logvar; color=(PALETTE_BLUE_POINT = "#2a78d6"; "#2a78d6"),
    markersize=5, alpha=0.5)
xs = range(minimum(logmean), maximum(logmean), length=100)
lines!(ax1, xs, beta[1] .+ beta[2] .* xs; color="#e34948", linewidth=2.5,
    label="fitted slope λ = $(round(lambda, digits=2))")
axislegend(ax1; position=:rb, framevisible=false)

# --- panel 2: flatness comparison, identity/sqrt/fourthroot/log ---
EPS = 1e-4
transforms = [
    ("identity", w -> w, PALETTE[1]),
    ("sqrt", w -> sqrt(max(w, 0.0)), PALETTE[2]),
    ("fourth-root", w -> max(w, 0.0)^0.25, PALETTE[4]),
    ("log", w -> log(max(w, EPS)), PALETTE[3]),
]
ax2 = Axis(fig[1, 2]; xlabel="raw local mean quintile (low to high wILI)",
    ylabel="mean local SD (post-transform, arbitrary units)",
    title="Flatness: local SD vs level, by transform")
cellkeys = combine(groupby(d, [:location, :woy]), :wili => mean => :local_raw_mean, nrow => :n)
cellkeys = filter(row -> row.n >= 8, cellkeys)
for (name, f, colour) in transforms
    d.tv = f.(d.wili)
    c = combine(groupby(d, [:location, :woy]), :tv => std => :local_sd,
        :wili => mean => :local_raw_mean, nrow => :n)
    c = filter(row -> row.n >= 8 && row.local_sd > 0 && row.local_raw_mean > 0, c)
    q = quantile(c.local_raw_mean, [0.2, 0.4, 0.6, 0.8])
    bin = [searchsortedfirst(q, m) for m in c.local_raw_mean]
    binsd = combine(groupby(DataFrame(bin=bin, sd=c.local_sd), :bin), :sd => mean => :m)
    sort!(binsd, :bin)
    # normalise each transform's curve to its own value at bin 1 so the four
    # curves are visually comparable on one axis despite different units
    normed = binsd.m ./ binsd.m[1]
    lines!(ax2, binsd.bin, normed; color=colour, linewidth=2.2, label=name)
    scatter!(ax2, binsd.bin, normed; color=colour, markersize=9)
end
hlines!(ax2, [1.0]; color=:gray, linestyle=:dash, linewidth=1)
axislegend(ax2; position=:lt, framevisible=false)

Label(fig[0, 1:2], "Variance-stabilisation: fourth-root flattens the mean-" *
    "variance relationship better than log (validation + history only)";
    fontsize=15, font=:bold)
save(joinpath(FIG_DIR, "01_transform_variance_stabilisation.png"), fig)
println("saved 01_transform_variance_stabilisation.png")
