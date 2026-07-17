# Figure for 03-seasonality.md: is season-to-season onset/offset
# variability a smooth spread (as the SD numbers in 03-seasonality
# suggest), or does it hide early/late onset subtypes? And is onset
# timing itself a shared "early/late year" effect across locations,
# the way seasonal amplitude was shown to be in 04-cross-location?
# Same onset/offset heuristic as 03-seasonality: first/last week of a
# run of >= 2 consecutive weeks above 1.5x the location's own
# off-season baseline (median wILI over woy 1-8 and 45-52).
include("common.jl")
using CairoMakie

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
    runs = Tuple{Int,Int}[]  # (start_idx, end_idx) of runs with length >= 2
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
    onset = w[runs[1][1]]
    offset = w[runs[end][2]]
    return (onset, offset)
end

onset_mat = fill(NaN, length(seasons), length(locs))
offset_mat = fill(NaN, length(seasons), length(locs))
for (j, loc) in enumerate(locs)
    base = baseline.baseline[findfirst(==(loc), baseline.location)]
    thresh = 1.5 * base
    dloc = d[d.location .== loc, :]
    for (i, sy) in enumerate(seasons)
        sub = dloc[dloc.season_year .== sy, :]
        isempty(sub) && continue
        on, off = onset_offset(sub.woy, sub.wili, thresh)
        ismissing(on) && continue
        onset_mat[i, j] = on
        offset_mat[i, j] = off
    end
end

fig = Figure(size=(1600, 1500), fontsize=13)

# --- panel 1: pooled histogram of onset week, all locations x seasons ---
ax1 = Axis(fig[1, 1]; title="Pooled onset week (all locations x seasons)",
    xlabel="onset woy", ylabel="count")
hist!(ax1, filter(!isnan, vec(onset_mat)); bins=20, color=(PALETTE[1], 0.8))

# --- panel 2: pooled histogram of offset week ---
ax2 = Axis(fig[1, 2]; title="Pooled offset week (all locations x seasons)",
    xlabel="offset woy", ylabel="count")
hist!(ax2, filter(!isnan, vec(offset_mat)); bins=20, color=(PALETTE[4], 0.8))

# --- panel 3: cross-location correlation of onset week across seasons ---
onset_cor = cor(onset_mat)
n = length(locs)
ax3 = Axis(fig[2, 1:2]; title="Cross-location correlation of onset week (by season)",
    xticks=(1:n, short), yticks=(1:n, short), xticklabelrotation=pi/2, aspect=1)
hm = heatmap!(ax3, 1:n, 1:n, onset_cor'; colormap=:Blues, colorrange=(-0.2, 1))
for i in 1:n, j in 1:n
    val = onset_cor[i, j]
    text!(ax3, i, j; text=string(round(val, digits=2)), align=(:center, :center),
        fontsize=9, color=val > 0.6 ? :white : :black)
end
Colorbar(fig[2, 3], hm; label="correlation")
rowsize!(fig.layout, 1, Relative(0.28))
rowsize!(fig.layout, 2, Relative(0.62))

Label(fig[0, 1:3], "Onset/offset timing: pooled distributions and " *
    "cross-location correlation of onset week (a shared-year test, as for " *
    "amplitude in 04-cross-location); validation + history only";
    fontsize=15, font=:bold)
save(joinpath(FIG_DIR, "13_onset_stability.png"), fig)
println("saved 13_onset_stability.png")

# print summary for the writeup
using Statistics
onset_all = filter(!isnan, vec(onset_mat))
offset_all = filter(!isnan, vec(offset_mat))
println("onset: n=", length(onset_all), " mean=", round(mean(onset_all), digits=1),
    " median=", median(onset_all), " sd=", round(std(onset_all), digits=1))
println("offset: n=", length(offset_all), " mean=", round(mean(offset_all), digits=1),
    " median=", median(offset_all), " sd=", round(std(offset_all), digits=1))
offdiag = [onset_cor[i, j] for i in 1:n, j in 1:n if i != j]
println("onset cross-loc corr: mean=", round(mean(offdiag), digits=2),
    " range=", round(minimum(offdiag), digits=2), " to ", round(maximum(offdiag), digits=2))
natl_idx = findfirst(==("US National"), locs)
for (j, loc) in enumerate(locs)
    j == natl_idx && continue
    println(loc, " onset vs US National: r=", round(onset_cor[natl_idx, j], digits=2))
end
# early/late subtype check: split seasons by national onset median, see if
# regional onsets cluster together within early vs late national seasons
natl_onset = onset_mat[:, natl_idx]
med = median(filter(!isnan, natl_onset))
early_seasons = seasons[natl_onset .<= med]
late_seasons = seasons[natl_onset .> med]
println("early national-onset seasons: ", early_seasons)
println("late national-onset seasons: ", late_seasons)
