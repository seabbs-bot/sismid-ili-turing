# Figure for 01-series-overview.md: histograms and QQ-plots of raw
# wILI vs fourth-root-transformed wILI, pooled across locations after
# per-location standardisation (z-score), to show the skew/variance-
# stabilisation story from the transform-comparison section visually
# rather than only as summary statistics.
include("common.jl")
using CairoMakie, StatsBase

d = load_finalized()
d = filter(row -> row.season_year >= 2004, d)
d.fr = max.(d.wili, 0.0) .^ 0.25

function zscore_pooled(df, col)
    out = Float64[]
    for loc in unique(df.location)
        sub = df[df.location .== loc, col]
        m, s = mean(sub), std(sub)
        append!(out, (sub .- m) ./ s)
    end
    return out
end

raw_z = zscore_pooled(d, :wili)
fr_z = zscore_pooled(d, :fr)

# inverse-normal-CDF via a standard rational approximation
# (Acklam's algorithm), avoiding a Distributions.jl dependency
function norminv(p::Float64)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    dd = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00)
    plow = 0.02425
    if p < plow
        q = sqrt(-2 * log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((dd[1]*q+dd[2])*q+dd[3])*q+dd[4])*q+1)
    elseif p <= 1 - plow
        q = p - 0.5
        r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
    else
        q = sqrt(-2 * log(1 - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((dd[1]*q+dd[2])*q+dd[3])*q+dd[4])*q+1)
    end
end

function qq_panel!(fig, pos, x, title_str, colour)
    n = length(x)
    xs = sort(x)
    p = ((1:n) .- 0.5) ./ n
    theo = norminv.(p)
    ax = Axis(fig[pos...]; title=title_str, xlabel="theoretical N(0,1) quantile",
        ylabel="sample quantile (z-scored)")
    scatter!(ax, theo, xs; color=colour, markersize=3, alpha=0.35)
    lo, hi = extrema(theo)
    lines!(ax, [lo, hi], [lo, hi]; color=:black, linestyle=:dash, linewidth=1)
end

fig = Figure(size=(1400, 950), fontsize=13)
ax1 = Axis(fig[1, 1]; title="Raw wILI (z-scored within location)",
    xlabel="z-score", ylabel="density")
hist!(ax1, raw_z; bins=60, normalization=:pdf, color=(PALETTE[1], 0.75))
ax2 = Axis(fig[1, 2]; title="Fourth-root wILI (z-scored within location)",
    xlabel="z-score", ylabel="density")
hist!(ax2, fr_z; bins=60, normalization=:pdf, color=(PALETTE[4], 0.75))
qq_panel!(fig, (2, 1), raw_z, "Raw wILI: QQ vs normal", PALETTE[1])
qq_panel!(fig, (2, 2), fr_z, "Fourth-root wILI: QQ vs normal", PALETTE[4])

Label(fig[0, 1:2], "Pooled, per-location-standardised distribution of raw " *
    "vs fourth-root wILI: fourth-root visibly shortens the right tail and " *
    "tracks the normal QQ line much more closely (validation + history " *
    "only)"; fontsize=15, font=:bold)
save(joinpath(FIG_DIR, "08_transform_distribution.png"), fig)
println("saved 08_transform_distribution.png")
