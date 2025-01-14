# Liberia
Modelers: Célia Escribe, Thomas Nikolakakis

Date: January 2025

## Study overview

This study aims to assess what should be the optimal sizing of solar PV and BESS in Liberia by 2030. This question was raised in the context of the organization of a solar tender to be launched in 2025 with an expected COD in 2028.

## Modeling assumptions

All the data changes made in this version of the model are summarized below. We only highlight here important modeling changes included in this study:
- hydro modeling: we consider two possible ways to model hydro power plants flexibility. 
  - First, the limited flexibility scenario assumes that hydro power plants are operated as full RoR plants, with the historical dispatch. This captures the limited flexibility observed in the historical data, where hydro generation can be shifted to the evening during the dry season to match peak demand. However, this does not adapt to the new energy mix, and in particular to the additional solar PV generation in the middle of the day.
  - Second, the flexibility scenario assumes a daily storage capacity. We estimate daily capacity factor from historical hourly hydro generation data. This daily generation can then be dispatched optimally by the model. In this case, hydro generation can be shifted to accomodate middle-of-the-day solar PV production, and to match evening peak demand.
- demand modeling: LEC provided some demand data accounting for loss load at the monthly resolution. 

### Scenarios

## Main outcomes

## Detailed data summary

### Demand forecast

We have updated the demand forecast that was initially taken from the Feasibility study by Artelia. We rely on an updated forecast made in 2024 by LEC, which is detailed in the Steering Committee meeting from 9 October 2024.

### Hydro capacity

Current available capacity is 88 MW, located in the MtCoffee localization. An extension is envisioned by 2028, bringing total hydro capacity to 148 MW. 

The SP2 project was estimated to have total capacity of 150 MW, with similary hydro generation profile as the MtCoffee power plant. This plant is assumed to come online in 2032.

### Hydro potential

We rely on historical dispatch data from 2023, available in the `Suggested Changes` excel workbook, tab `DispatchGraphs`. To estimate corresponding hourly capacacity factor, we use a capacity of 66 GW (instead of the theoretical 88 GW). This takes into account the fact that one of the MtCoffee hydro power plant was down in this period.

The capacity factor for prospective project SP2 was obtained from Artelia data that estimated monthly capacity factor. The hourly capacity factor was derived from scaling the profile from existing MtCoffee power plant to match the projected monthly capacity factors.

The corresponding processed data is available in the `Processed data` folder.

### Thermal generation

Existing thermal HFO generators were confirmed to have a capacity of 28 MW. 

### Import and export capacity

Import capacity seems uncertain. Its range is assumed to be 10-50 MW. Baseline value is taken at 27 MW. However, one point of attention is that in a country model, import is only based on price. Capacity is taken to be the same across all months and years. This fails to account for correlated production and demand patterns in neighboring countries: when demand is high and hydro production is low in Liberia, it may be also the case in neighboring coutries, thus affecting import capacity.  