# 05. Autocorrelation

Source: `data/flu_data_hhs.csv`, log scale (`log(wili + 0.01)`).
Computed in Julia (CSV.jl, DataFrames.jl, Statistics, StatsBase),
with `autocor`/`pacf` from StatsBase and a manual Yule-Walker + AIC
order search.
Two versions are examined: the raw log series (dominated by
seasonality) and a deseasonalised residual (log series minus its
location x week-of-season mean, using the `woy` definition from
[[01-series-overview]]), which better isolates the AR structure the
brief asks the model to place on top of the seasonal component.

**Experimental integrity: filtered to `season_year <= 2016`** (see
[[01-series-overview]] for the definition), the full training set —
pre-2015 history plus the two validation seasons.
The three held-out testing seasons are excluded throughout.

## Raw log series (seasonality still present)

ACF decays very slowly (lag 1 ~0.78-0.96, still 0.6-0.7 by lag 6)
at every location.
This is a seasonality artefact, not evidence of a long genuine AR
memory: a slowly-varying seasonal mean makes the raw series look
highly persistent at any lag.
It is included here only to show why AR order should not be tuned
on the undifferenced, non-deseasonalised series.

## Deseasonalised residual

After removing the location x week-of-season mean, ACF still decays
gradually rather than cutting off, from ~0.78-0.96 at lag 1 down to
0.21-0.53 by lag 8, but PACF drops sharply after lag 1 (lag-1
partial 0.78-0.96, lag-2 partial mostly in -0.13 to +0.3, and lag 3+
mostly inside ±0.13).
AIC-based AR order selection on this residual (Yule-Walker) gives:

| location | selected order |
|---|---|
| HHS Region 8 | 4 |
| HHS Region 9 | 4 |
| US National | 4 |
| HHS Region 3 | 5 |
| HHS Region 4 | 5 |
| HHS Region 5 | 5 |
| HHS Region 1 | 6 |
| HHS Region 2 | 8 |
| HHS Region 6 | 9 |
| HHS Region 10 | 9 |
| HHS Region 7 | 10 |

Median 5, range 4-10 — a more than 2-fold spread across the 11
locations.
So although the PACF looks close to an AR(1) cutoff, AIC still
prefers higher order for every location in this full-training-set
sample — the gradual ACF decay plus small-but-nonzero higher-lag
partials add up over many lags.
This directly supports the brief's instruction to consider AR order
greater than 2, though a single shared order across all 11
locations would be a poor fit to this range; a partially-pooled
order/coefficient structure (shrinking high-order coefficients
toward zero rather than fixing one hard cutoff) fits the spread
better than picking one fixed low order for everyone.
This order search pools all 13 training-set seasons together per
location; whether AR order or coefficients also drift from season to
season within a location has not yet been tested and is worth a
follow-up if per-season identifiability allows it.

![ACF and PACF of the deseasonalised residual, and ACF of its first
difference, for four example locations spanning the AR-order range
(US National and Region 8: order 4; Region 2: order 8; Region 7:
order 10); PACF cuts off near lag 1 but ACF decays gradually, and
differencing induces a negative lag-1 ACF for most
locations](figures/05_autocorrelation.png)

## AR vs differencing

The first difference of the deseasonalised residual has a
**negative** lag-1 ACF at nearly every location (-0.15 to -0.35,
median around -0.2; two locations, Region 4 and Region 5, show a
small positive lag-1 of +0.07/+0.09 instead), which is the classic
signature of over-differencing a series that was already close to
stationary (differencing a stationary AR process introduces exactly
this kind of negative lag-1 autocorrelation).
This, combined with PACF on the undifferenced residual decaying
rather than showing a unit root, indicates the deseasonalised
residual is a stationary AR-type process and should be modelled
with an AR term rather than a first difference; differencing on top
of an already-adequate AR(p) is likely to hurt, not help for most
locations.

## Does the pooled signature hold every season, or only on average?

The numbers above pool all 13 training-set seasons per location.
This section checks per-season stability directly: for each
location and each season separately, the lag-1 ACF of the
undifferenced residual (persistence) and of its first difference
(the AR-vs-differencing signature), computed within that season's
~52 weeks only.

![Per-season lag-1 ACF, undifferenced (left) and first-differenced
(right) deseasonalised residual, by location; persistence is stable
season to season but the differencing sign is much noisier per
season than the pooled estimate suggests (validation + history
only)](figures/09_acf_season_drift.png)

**Persistence is stable.**
The undifferenced lag-1 ACF (left panel) is uniformly high
(0.3-0.98) for every location in every one of the 13 seasons, with
no location ever showing a season where persistence collapses.
The pooled lag-1 range quoted above (0.78-0.96) is a reasonable
single-number summary of a genuinely stable property, not an
average papering over big per-season swings.

**The differencing sign, however, is not stable for most locations.**
Per-season lag-1 ACF of the first-differenced residual:

| location | per-season range | seasons with positive lag-1 (of 13) |
|---|---|---|
| US National | -0.33 to 0.65 | 6 |
| HHS Region 1 | -0.51 to 0.30 | 2 |
| HHS Region 2 | -0.42 to 0.41 | 6 |
| HHS Region 3 | -0.40 to 0.47 | 2 |
| HHS Region 4 | -0.29 to 0.47 | 7 |
| HHS Region 5 | -0.31 to 0.55 | 6 |
| HHS Region 6 | -0.32 to 0.35 | 2 |
| HHS Region 7 | -0.66 to -0.01 | 0 |
| HHS Region 8 | -0.45 to 0.35 | 2 |
| HHS Region 9 | -0.44 to -0.04 | 0 |
| HHS Region 10 | -0.46 to -0.03 | 0 |

Only three locations (Region 7, Region 9, Region 10) show the
negative-lag-1 "over-differencing" signature in **every** one of
the 13 seasons.
Every other location, including several with a clearly negative
*pooled* lag-1 (e.g. US National, Region 2, Region 5, at 6/13
positive seasons each), is roughly evenly split season to season,
and Region 4 (the pooled positive-lag-1 exception) is actually
majority-positive per season (7/13) but not by a wide margin over
the other locations.
So the pooled "AR beats differencing almost everywhere, except
Region 4/Region 5" story is a real central tendency, but for most
locations it is closer to a coin flip season to season than a
consistent per-season property; only Region 7/Region 9/Region 10
show a stable, every-season case against differencing.

## Implications for the model

- Fit AR order on the deseasonalised (post-seasonality) residual,
  never on the raw series — the raw series' apparent long memory is
  a seasonality artefact.
- Favour AR over an additional first-difference step for most
  locations: differencing the residual induces a negative lag-1
  ACF, a standard over-differencing symptom, given the residual
  already looks close to a stationary AR(p).
  Region 4 and Region 5 are the exception (small positive lag-1
  after differencing) and are worth checking individually if a
  location-varying AR-vs-differencing choice is on the table.
  The per-season breakdown above shows this AR-vs-differencing
  signal is itself fairly noisy season to season for most locations
  (roughly even split, not a stable per-season property), so treat
  the pooled recommendation as the right *average* choice rather
  than something that should hold rigidly in any single season; only
  Region 7/Region 9/Region 10 show a consistent every-season case
  for AR over differencing.
- AR order 1-2 with partial pooling toward a low shared order is a
  reasonable starting branch, but should be tested against
  higher-order (up to ~9-10) partially-pooled alternatives per the
  brief, since AIC prefers order >= 4 for every location in this
  sample and order >= 5 for 8 of the 11.
- This is a full-training-set estimate (13 seasons, pooled across
  seasons per location); revisit the order search as a per-season
  breakdown becomes feasible, and never extend it into the three
  held-out test seasons.
