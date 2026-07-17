# 02. Backfill / reporting revisions

Source: `data/flu_data_hhs_versions.csv`.
Reporting delay = `as_of - origin_date`, in weeks.
Computed in Julia (CSV.jl, DataFrames.jl, Statistics, StatsBase).

**Experimental integrity: filtered to `season_year <= 2016`** (see
[[01-series-overview]] for the definition), the full training set —
pre-2015 history plus the two validation seasons.
The three held-out testing seasons are excluded throughout.

## Snapshot structure

- Within the training-set cutoff: 77 distinct `as_of` snapshots,
  2015-10-01 to 2017-06-30 (mostly weekly spacing).
  2015-10-01 is when the reporting-version file itself starts
  recording snapshots (a data-collection artefact of the source
  file, not an experimental-integrity restriction we chose).
- 717 distinct `origin_date` values, back to 2003-08-30.
- 31,086 rows total; 87% of (location, origin_date) series have
  only a single recorded version.
  These are training-set weeks that had already settled by the time
  the version file's tracking began, so only their final value was
  ever captured.
  The remaining 13% (1,023 series) have a genuine multi-version
  revision history and are the basis for the numbers below; because
  tracking only starts 2015-10-01, every one of these 1,023 series
  falls in the two most recent training-set seasons (`2015/16`,
  `2016/17` — 517 and 506 series respectively).
  Earlier training-set seasons (`2004/05`-`2014/15`) are not a gap
  in our filtering, they are simply not covered by this source
  file's snapshot history.
- For those series, the tracked delay runs from 1 week up to 48
  weeks (the last snapshot for old data), with a median of 25 weeks
  between first and last recorded version.

All numbers below use, for each (location, origin_date) with more
than one version, the value at the largest tracked delay as a proxy
for the settled value, and compare earlier versions against it.

## Revision size by delay

| delay (weeks) | median abs revision | median relative revision | % up | % down |
|---|---|---|---|---|
| 1 | 0.104 | 6.6% | 60% | 40% |
| 2 | 0.067 | 3.8% | 71% | 29% |
| 3 | 0.050 | 2.9% | 71% | 29% |
| 4 | 0.038 | 2.2% | 70% | 29% |
| 5 | 0.028 | 1.6% | 71% | 26% |
| 8 | 0.014 | 1.0% | 65% | 23% |
| 10 | 0.008 | 0.5% | 56% | 22% |
| 15 | ~0.00 | ~0.0% | 36% | 20% |
| 20 | 0.00 | 0.0% | 20% | 13% |

(remaining probability mass at each delay beyond ~5 weeks is exact
ties with the settled value; "% up"/"% down" are shares of all
series at that delay, so they no longer sum close to 100% once most
series have already settled).

Revisions are largest in the first 1-2 weeks (median ~6.6% relative
at delay 1) and decay smoothly; by delay ~15-20 weeks the typical
revision is negligible, so a reporting-delay dimension `Dmax` around
10-15 covers the great majority of the revision mass, with a longer
tail of small, occasional adjustments beyond that.

![Revision size (left) and direction (right) by reporting delay,
pooled and for three example locations (Region 2 upward-biased,
Region 9 downward-biased, US National as a
baseline)](figures/02_backfill_revisions.png)

## Non-monotonic direction

Revisions are **not** a one-directional correction.
At delay 1, 60% of revisions are upward and 40% downward, i.e. a
sizeable minority of initial reports are revised down, consistent
with the brief's point that wILI is a re-weighted percentage (not
an accumulating count) so nothing forces the delay profile to be a
monotonic reporting CDF.
The upward share rises slightly from delay 1 to delay 2-5 (60% to
~71%) then drifts down toward 20% by delay 20 as more series have
already reached their settled value (ties inflate the "settled"
share rather than genuine downward revision becoming more common).
Looking at whether the *sign* of the revision is stable across
delays for the same series: sign(delay 1) matches sign(delay 2) in
73% of series, and all three of sign(delay 1,2,3) agree in only
62% — i.e. more than a third of series flip direction at least once
while settling, direct evidence of a non-monotonic revision path.

## Variation by location

Both the size and the direction of the early revision differ
sharply by location (delay 1, mean **signed** revision):

| location | mean signed revision (delay 1) | % up (delay 1) |
|---|---|---|
| HHS Region 2 | +0.414 | 84% |
| HHS Region 7 | +0.118 | 77% |
| HHS Region 3 | +0.104 | 66% |
| HHS Region 6 | +0.094 | 70% |
| HHS Region 9 | -0.271 | 39% |
| HHS Region 1 | -0.057 | 42% |
| HHS Region 10 | -0.049 | 38% |
| US National | +0.044 | 62% |

Region 2 shows a strong, consistent upward bias in the first report
(initial reports undercount, 84% of first revisions are upward);
Region 9, Region 1, Region 10 trend negative at delay 1 (initial
reports overcount, only 38-42% of revisions are upward).
Median absolute revision at delay 1 also spans an ~14x range across
locations, from 0.034 (Region 8) to 0.463 (Region 2).
This supports a location-varying (partially pooled) revision
profile rather than one shared profile.

## Variation over the season

Splitting delay 1-4 revisions by whether the settled value is above
or below the pooled median (a crude peak- vs off-season split):

| phase | delay 1 median rel. revision | delay 1 % up | delay 4 median rel. revision | delay 4 % up |
|---|---|---|---|---|
| high (peak season) | 5.7% | 64% | 1.9% | 72% |
| low (off season) | 7.3% | 54% | 2.4% | 66% |

Relative revision size is larger off-season (low counts amplify
proportional noise) and the upward bias at delay 1 is stronger in
the peak-season phase (64% vs 54% up), so time-of-season affects
both revision *magnitude* (bigger relative swings off-peak) and, to
a smaller extent, revision *direction*.
This is real but weaker than the location effect above.

## Variation across training seasons

The two training-set seasons with tracked revision history also
differ from each other at delay 1 (pooled across locations,
n = 407 each):

| training season | median rel. revision (delay 1) | % up (delay 1) |
|---|---|---|
| `2015/16` | 4.5% | 69% |
| `2016/17` | 3.5% | 73% |

The direction is consistent (majority upward both seasons) but the
size and the upward share both shift season to season, on top of
the larger location-driven differences above.
With only two tracked seasons this is too small a sample to fit a
season-level revision effect confidently, but it is further evidence
that a single fixed revision profile is unlikely to hold everywhere
and every season — a partially-pooled profile that can absorb both
location and season variation is the safer default.

## Revision structure by location and by tracked season

The location and season summaries above are pooled, or use only
three example locations; this section breaks the delay-1 revision
down per location, per tracked training season (`2015/16` vs
`2016/17`, n = 37 per location per season), to check whether the
season-to-season shift noted above is a small pooled effect or
hides larger, location-specific swings.

| location | 2015/16 median abs. rev. | 2015/16 % up | 2016/17 median abs. rev. | 2016/17 % up |
|---|---|---|---|---|
| US National | 0.078 | 70% | 0.091 | 54% |
| HHS Region 1 | 0.104 | 27% | 0.076 | 57% |
| HHS Region 2 | 0.715 | 95% | 0.337 | 73% |
| HHS Region 3 | 0.105 | 51% | 0.214 | 81% |
| HHS Region 4 | 0.133 | 95% | 0.092 | 43% |
| HHS Region 5 | 0.068 | 41% | 0.080 | 49% |
| HHS Region 6 | 0.154 | 76% | 0.065 | 65% |
| HHS Region 7 | 0.080 | 78% | 0.100 | 76% |
| HHS Region 8 | 0.021 | 68% | 0.047 | 65% |
| HHS Region 9 | 0.505 | 27% | 0.211 | 51% |
| HHS Region 10 | 0.042 | 27% | 0.080 | 49% |

Region 2 is the most heavily revised location in both seasons (its
median absolute revision is 3-7x every other location) but the size
roughly halves between seasons (0.715 to 0.337) while staying
strongly upward (95% to 73% up).
More strikingly, **Region 1 and Region 4 flip direction entirely**
between the two tracked seasons.
Region 1 goes from 27% up (i.e. 73% of first reports revised down)
in `2015/16` to 57% up in `2016/17`.
Region 4 goes from 95% up in `2015/16` to 43% up (majority
downward) in `2016/17`.
Region 9 also shifts substantially (27% to 51% up).
This is a stronger, location-specific version of the pooled
season-to-season shift noted above (4.5% to 3.5% median relative
revision, 69% to 73% up): at the pooled level the two seasons look
broadly similar, but several individual locations swing much more,
including full direction reversals.

![Backfill revision size by delay, faceted by all 11 locations and
split by tracked training season (`2015/16` vs `2016/17`); most
locations show a visibly different profile between the two seasons,
on top of the larger cross-location differences already shown
above](figures/07_backfill_by_region.png)

This sharpens the point made in the "variation across training
seasons" section above: with only two tracked seasons, a
location-varying revision profile is necessary (Region 2 alone
needs a materially different scale from every other location), and
a season-varying component is not just a small pooled adjustment,
it needs to represent genuine direction reversals for at least
Region 1 and Region 4.
A model that pools the revision profile fully across locations, or
treats season variation as a small shared shift, would misrepresent
this behaviour for several of the 11 locations.

## Does the Region 1/Region 4 reversal track season-phase or the season as a whole?

The reversal above is large enough to ask whether it is really a
season-level effect, or whether it is secretly a peak-vs-off-season
(phase) effect that happens to correlate with which weeks each
tracked season's snapshots landed on.
This section splits each location's delay-1 revisions further, by
whether the settled value that week is above or below that
location's own median (the same crude phase split used in
"variation over the season" above) crossed with tracked season.

| location, season, phase | n | % up |
|---|---|---|
| Region 1, 2015/16, low | 18 | 28% |
| Region 1, 2015/16, high | 19 | 26% |
| Region 1, 2016/17, low | 19 | 47% |
| Region 1, 2016/17, high | 18 | 67% |
| Region 4, 2015/16, low | 22 | 95% |
| Region 4, 2015/16, high | 15 | 93% |
| Region 4, 2016/17, low | 15 | 33% |
| Region 4, 2016/17, high | 22 | 50% |

![Delay-1 revision direction by season-phase and tracked training
season, faceted by all 11 locations; for Region 1 and Region 4 the
two phase points sit close together within each season and the
separation is almost entirely between seasons, not between phases
(validation + history only)](figures/10_revision_phase_season.png)

For both Region 1 and Region 4, the low-phase and high-phase points
sit close to each other within a season (Region 1: 28% vs 26% in
`2015/16`, 47% vs 67% in `2016/17`; Region 4: 95% vs 93% in
`2015/16`, 33% vs 50% in `2016/17`), while the gap **between**
seasons is large regardless of phase.
So the reversal is a genuine season-level shift in the revision
process, not an artefact of which part of the season each tracked
season happened to sample more of.
A handful of other locations (e.g. Region 3, Region 9) show a
similar within-season phase consistency alongside a between-season
shift in the figure, though not as extreme a reversal as Region 1 or
Region 4.

## Implications for the model

- Do not use a monotonic reporting-CDF backfill (e.g. a plain
  `baselinenowcast`-style completion factor); revisions genuinely
  change sign both across delay and across locations.
- A `Dmax` of ~10-15 weeks captures nearly all revision mass; a
  smaller `Dmax` (contract default 6) will still catch the bulk but
  will miss a real tail — worth testing sensitivity to `Dmax`.
- Partially pool the delay-indexed revision profile across
  locations (means differ more than an order of magnitude and can
  differ in sign) and allow some time variation (both magnitude and,
  more weakly, direction differ off-season vs peak season, and
  between the two tracked training seasons), matching the brief's
  "shared vs location-varying vs time-varying" search axis.
  The per-location, per-season breakdown above shows this is not
  just a small pooled adjustment: Region 1 and Region 4 fully
  reverse the sign of their delay-1 revision bias between `2015/16`
  and `2016/17`, so the season-varying component needs enough
  flexibility to represent a genuine sign flip, not only a
  magnitude change, for at least some locations.
  The phase-crossed breakdown confirms this is a true season-level
  effect (consistent across both peak and off-season weeks within a
  season), so a season random effect on the revision profile is the
  right structure, rather than trying to explain the shift through
  the existing phase/time-of-season term alone.
- The 1,023 series with tracked revision history all fall in the
  two most recent training-set seasons (`2015/16`, `2016/17`),
  because the reporting-version source file only records snapshots
  from 2015-10-01 onward — this is a property of the source data,
  not a subset we chose, and there is no larger already-tracked
  training-set sample to fall back on.
  Treat the exact percentages above as estimated from those two
  seasons specifically, re-derive them if a version file with longer
  tracked history becomes available, and never substitute the
  held-out test seasons as a stand-in for more revision history.
