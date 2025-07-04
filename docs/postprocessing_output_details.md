# EPM Output documentation

This document describes the key outputs from the Energy Planning Model (EPM), providing a quick summary of what each file contains and example rows to guide users unfamiliar with the model.

---

## `pUtilizationByTechandFuelCountry.csv`

🔹 **Description:** Utilization factor by technology and fuel type, at the country level. Indicates how fully each generation type is used.

```markdown
| Country | TECHNOLOGY | FUEL    | Utilization Factor |
|---------|------------|---------|---------------------|
| CAF     | Oil        | Oil     | 0.38                |
| CAF     | Hydro      | Hydro   | 0.47                |
```

---

## `pCapacityByFuel.csv`

🔹 **Description:** Total installed capacity (MW) per fuel type, by region.

```markdown
| REGION | FUEL    | Capacity (MW) |
|--------|---------|----------------|
| CAF    | Oil     | 55.0           |
```

---

## `pFuelCosts.csv`

🔹 **Description:** Fuel prices over time by fuel type (in $/MMBtu).

---

## `pEmissionsIntensity.csv`

🔹 **Description:** Emissions per unit of electricity generation by technology and fuel.

---

## `pCostsbyPlant.csv`

🔹 **Description:** Cost breakdown (fixed and variable) for each plant.

---

## `pFuelCostsCountry.csv`

🔹 **Description:** Country-specific fuel costs over time.

---

## `pUtilizationByTechandFuel.csv`

🔹 **Description:** Technology and fuel-level utilization, without country breakdown.

---

## `pEnergyByFuel.csv`

🔹 **Description:** Total energy generated by each fuel type across regions.

---

## `pEnergyByFuelCountry.csv`

🔹 **Description:** Country-level generation by fuel type.

---

## `pSpinningReserveByPlantZone.csv`

🔹 **Description:** Spinning reserve requirements by plant and zone.

---

## `pCostSummaryCountry.csv`

🔹 **Description:** Summarized cost breakdown per country (capital, fuel, O&M, etc.).

---

## `pCostSummaryFull.csv`

🔹 **Description:** Full disaggregated cost report for all entities (plants, fuels, etc.).

---

## `pNewCapacityFuelCountry.csv`

🔹 **Description:** New capacity additions by fuel and country (MW).

---

## `pPeakCapacityCountry.csv`

🔹 **Description:** Peak system capacity needed per country.

---

## `pDispatch.csv`

🔹 **Description:** Plant-level dispatch values by time step (MWh produced).

---

## `pEnergyByPlant.csv`

🔹 **Description:** Total annual generation by plant.

---

## `pCostSummary.csv`

🔹 **Description:** Consolidated total cost summary for the entire system.

---

## `pCapacityPlan.csv`

🔹 **Description:** Planned capacity by technology, fuel, and region.

---

## `pEnergyMix.csv`

🔹 **Description:** Percent energy share by fuel or technology.

---

## `pPlantAnnualLOCE.csv`

🔹 **Description:** Levelized cost of electricity per plant.

---

## `pSpinningReserveByPlantCountry.csv`

🔹 **Description:** Spinning reserve assignments by plant and country.

---

## `pDemandSupply.csv`

🔹 **Description:** Demand and supply balance by country and zone.

---

## `pFuelDispatch.csv`

🔹 **Description:** Fuel dispatch values per plant or unit.

---

## `pPlantUtilization.csv`

🔹 **Description:** Overall utilization factor for each plant.

---

## `pPrice.csv`

🔹 **Description:** Marginal system price or shadow price for each time period.

---

## `pEmissions.csv`

🔹 **Description:** Total emissions (CO₂, etc.) by country, plant, or fuel.

---

## `pCapacityCredit.csv`

🔹 **Description:** Capacity credit assigned to variable or firm generation.

---

## `pUtilizationByFuelCountry.csv`

🔹 **Description:** Utilization factor by fuel and country (aggregated across technologies).

---

## `pZonalAverageCost.csv`

🔹 **Description:** Average cost of supply or dispatch in each zone.

---

## `pSpinningReserveByFuelZone.csv`

🔹 **Description:** Spinning reserve requirement by fuel type in each zone.

---

## `pCapacityByFuelCountry.csv`

🔹 **Description:** Installed capacity by fuel and country.

---

## `pNewCapacityFuel.csv`

🔹 **Description:** New capacity additions by fuel (global or regional).

---

## `pPeakCapacity.csv`

🔹 **Description:** Peak capacity requirement per region or the entire system.

---

## `zmap.csv`

🔹 **Description:** Mapping file for zones, plants, or countries.

---

## `pSpinningReserveCostsCountry.csv`

🔹 **Description:** Cost of meeting spinning reserve requirements per country.

---

## `pSpinningReserveCostsZone.csv`

🔹 **Description:** Spinning reserve costs by zone.

---

## `pAnnualTransmissionCapacity.csv`

🔹 **Description:** Annual transmission capacity available between zones or countries.

---

## `pAdditionalCapacity.csv`

🔹 **Description:** Additional required capacity beyond existing/planned levels.

---

## `pInterchangeCountry.csv`

🔹 **Description:** Power exchanged (imports/exports) between countries.

---

## `pInterconUtilization.csv`

🔹 **Description:** Utilization of interconnectors over time.

---

## `pInterchange.csv`

🔹 **Description:** General interchange flows between zones or nodes.

---

## `pCostSummaryWeightedAverageCountry.csv`

🔹 **Description:** Country-level average cost metrics (weighted by generation or capacity).

---

## `pSummary.csv`

🔹 **Description:** High-level summary of key results: total cost, emissions, capacity, generation.

---

## `pSettings.csv`

🔹 **Description:** Model settings used during the simulation (e.g. years, resolution, toggles).

---
