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
[[01-series-overview]] for the definition), i.e. pre-2015 history
plus the two validation seasons only.
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

Median 5, range 4-10.
So although the PACF looks close to an AR(1) cutoff, AIC still
prefers higher order for every location in this validation-period
sample — the gradual ACF decay plus small-but-nonzero higher-lag
partials add up over many lags.
This directly supports the brief's instruction to consider AR order
greater than 2, though a single shared order across all 11
locations would be a poor fit to this range; a partially-pooled
order/coefficient structure (shrinking high-order coefficients
toward zero rather than fixing one hard cutoff) fits the spread
better than picking one fixed low order for everyone.

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
- AR order 1-2 with partial pooling toward a low shared order is a
  reasonable starting branch, but should be tested against
  higher-order (up to ~9-10) partially-pooled alternatives per the
  brief, since AIC prefers order >= 4 for every location in this
  sample and order >= 5 for 8 of the 11.
- This is a validation-period-only estimate (13 seasons); revisit
  the order search as more validation-period detail is added, and
  never extend it into the three held-out testing seasons.
