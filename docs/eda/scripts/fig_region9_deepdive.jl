# Figure for 07-region9-deepdive.md: Region 9 has been flagged, in
# passing, as an outlier in several separate reports (most Poisson-
# like transform power in 01-series-overview, strongest negative
# delay-1 backfill bias in 02-backfill, weakest amplitude correlation
# in 04-cross-location). This pulls three of those threads into one
# figure to check whether Region 9 is a genuine, consistent outlier
# or is flagged for different, unrelated reasons each time.
include("common.jl")
using CairoMakie, Statistics

d = load_finalized()
d = filter(row -> row.season_year >= 2004, d)
locs = ["US National", ["HHS Region $i" for i in 1:10]...]
seasons = sort(unique(d.season_year))
r9 = "HHS Region 9"

# --- panel 1: Taylor's power law, Region 9 highlighted ---
cells = combine(groupby(d, [:location, :woy]),
    :wili => mean => :local_mean, :wili => var => :local_var, nrow => :n)
cells = filter(row -> row.n >= 8 && row.local_mean > 0 && row.local_var > 0, cells)
logmean_all = log.(cells.local_mean)
logvar_all = log.(cells.local_var)
X = hcat(ones(length(logmean_all)), logmean_all)
beta_all = X \ logvar_all

r9cells = filter(row -> row.location == r9, cells)
logmean_r9 = log.(r9cells.local_mean)
logvar_r9 = log.(r9cells.local_var)
X9 = hcat(ones(length(logmean_r9)), logmean_r9)
beta_r9 = X9 \ logvar_r9

# --- panel 2: amplitude, Region 9 vs national, by season ---
d.offseason = (d.woy .<= 8) .| (d.woy .>= 45)
baseline = combine(groupby(filter(row -> row.offseason, d), :location), :wili => median => :baseline)
amp = Dict{String,Vector{Float64}}()
for loc in (r9, "US National")
    base = baseline.baseline[findfirst(==(loc), baseline.location)]
    dloc = d[d.location .== loc, :]
    a = Float64[]
    for sy in seasons
        sub = dloc[dloc.season_year .== sy, :]
        push!(a, isempty(sub) ? NaN : maximum(sub.wili) - base)
    end
    amp[loc] = a
end
r9_amp_z = (amp[r9] .- mean(amp[r9])) ./ std(amp[r9])
natl_amp_z = (amp["US National"] .- mean(amp["US National"])) ./ std(amp["US National"])
r9_natl_amp_r = cor(amp[r9], amp["US National"])

# --- panel 3: delay-1 revision % up by tracked season, Region 9 vs pooled ---
dv = load_versions()
gd = groupby(dv, [:location, :origin_date])
counts = combine(gd, nrow => :nver)
mult_keys = filter(row -> row.nver > 1, counts)
mult = innerjoin(dv, select(mult_keys, [:location, :origin_date]), on=[:location, :origin_date])
gd2 = groupby(mult, [:location, :origin_date])
settled = combine(gd2) do sub
    i = argmax(sub.delay)
    (settled_wili = sub.wili[i], settled_delay = sub.delay[i])
end
rev = innerjoin(mult, settled, on=[:location, :origin_date])
rev.revision = rev.wili .- rev.settled_wili
rev1 = filter(row -> row.delay == 1 && row.settled_wili > 0.05, rev)
r9pct = Float64[]
pooledpct = Float64[]
for sy in (2015, 2016)
    subr9 = filter(r -> r.location == r9 && r.season_year == sy, rev1)
    push!(r9pct, 100 * mean(subr9.revision .> 0))
    subpool = filter(r -> r.season_year == sy, rev1)
    push!(pooledpct, 100 * mean(subpool.revision .> 0))
end

fig = Figure(size=(1650, 550), fontsize=13)

ax1 = Axis(fig[1, 1]; xlabel="log(local mean wILI)", ylabel="log(local variance)",
    title="Taylor's power law: Region 9 vs pooled")
scatter!(ax1, logmean_all, logvar_all; color=(:gray, 0.3), markersize=4)
scatter!(ax1, logmean_r9, logvar_r9; color=PALETTE[7], markersize=6)
xs = range(minimum(logmean_all), maximum(logmean_all), length=100)
lines!(ax1, xs, beta_all[1] .+ beta_all[2] .* xs; color=:black, linewidth=2,
    label="pooled λ=$(round(beta_all[2], digits=2))")
lines!(ax1, xs, beta_r9[1] .+ beta_r9[2] .* xs; color=PALETTE[7], linewidth=2.5,
    label="Region 9 λ=$(round(beta_r9[2], digits=2))")
axislegend(ax1; position=:rb, framevisible=false)

ax2 = Axis(fig[1, 2]; xlabel="season", ylabel="amplitude, z-scored",
    title="Amplitude: Region 9 vs National (r=$(round(r9_natl_amp_r, digits=2)))",
    xticks=(1:length(seasons), string.(seasons .- 2000)), xticklabelrotation=pi/3)
lines!(ax2, 1:length(seasons), r9_amp_z; color=PALETTE[7], linewidth=2, label="Region 9")
scatter!(ax2, 1:length(seasons), r9_amp_z; color=PALETTE[7], markersize=8)
lines!(ax2, 1:length(seasons), natl_amp_z; color=:black, linewidth=2, label="US National")
scatter!(ax2, 1:length(seasons), natl_amp_z; color=:black, markersize=8)
axislegend(ax2; position=:rt, framevisible=false)

ax3 = Axis(fig[1, 3]; xlabel="tracked training season", ylabel="% delay-1 revisions upward",
    title="Backfill direction: Region 9 vs pooled",
    xticks=(1:2, ["2015/16", "2016/17"]))
barplot!(ax3, (1:2) .- 0.15, r9pct; width=0.28, color=PALETTE[7], label="Region 9")
barplot!(ax3, (1:2) .+ 0.15, pooledpct; width=0.28, color=:gray, label="pooled (all locations)")
hlines!(ax3, [50]; color=:black, linestyle=:dash, linewidth=1)
ylims!(ax3, 0, 100)
axislegend(ax3; position=:rt, framevisible=false)

Label(fig[0, 1:3], "Region 9 across three unrelated axes: an outlier on " *
    "transform power and backfill direction, only moderately decoupled on " *
    "amplitude (validation + history only)"; fontsize=15, font=:bold)
save(joinpath(FIG_DIR, "12_region9_deepdive.png"), fig)
println("saved 12_region9_deepdive.png")
println("Region 9 Taylor's law lambda = ", round(beta_r9[2], digits=2),
    " vs pooled ", round(beta_all[2], digits=2))
println("Region 9 amplitude vs National r = ", round(r9_natl_amp_r, digits=2))
println("Region 9 delay-1 %up: ", round.(r9pct, digits=0), " vs pooled ", round.(pooledpct, digits=0))
