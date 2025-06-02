# Detailed description of input data

## Settings

The **Settings** file allows users to configure the model:
- **General options**: Set Weighted Average Costs of Capital (WACC) for capital investments and the discount rate for present value calculations (in percentages).
- **Penalties**: Define penalties for unserved energy, curtailment, surplus power, etc.
- **Optional Features**: Enable or disable features such as:
  - Carbon prices
  - Energy efficiency
  - Storage optimization
  - CSP optimization
- **Constraints**: Apply or remove limits like:
  - CO2 emissions
  - Fuel constraints
  - Ramp constraints
  - Import share constraints
- **Output options**: Select results the model should produce (default is "yes" for all).
- **Active Fuel Price Scenario**: Select the fuel price scenario (scenarios are defined in the `FuelTechnologies.csv` file).
- **Capacity Credits**: Define capacity credits for variable renewable technologies. These apply only if the "User defined Capacity Credit" option is set to "Yes."

---

## Fuels

The **Fuels** file includes:
- Fuel types and generation technologies used in the model.
- Differentiation between fuel types (e.g., North Sea Gas vs. regular Gas).
- Fuel price scenarios, defined in the `FuelTechnologies.csv` file, with prices listed in `FuelPrices.csv`.
- The default unit for fuels is MMBtu, with conversion factors included in the `FuelTypes.csv` file.

---

## Zone

Define zones in the `ZoneData.csv` file:
- Assign a name, index, and whether the zone is included in the model.
- Map zones to countries in `MapZonesToCountries.csv`.
- Define possible new transmission routes in the `ZonesTransmission.csv` file:
  - From/To zones
  - Earliest entry year
  - Maximum number of lines and capacity
  - Cost and lifetime

---

## Load

The **Load** file (`LoadDefinition.csv`) defines:
- Time periods (years, seasons, day types, hours).
- Peak hours and critical days.
- Weights for time periods in the `Duration.csv` file.

---

## Generator

The **Generator Data** file contains:
- Plant-specific details:
  - Name, zone, commissioning/retirement years, capacity, technology, and fuels.
  - Fixed/Variable O&M costs, heat rates, CAPEX, and lifetimes.
  - Build limits and minimum load/ramp rates.

---

## GenAvailability

The `GenAvailability.csv` file includes:
- Seasonal availability (e.g., hydro plants).
- Daily variations (handled in the `RenewableEnergyProfile.csv` file).

---

## Topology

The **Topology** file defines:
- Zone interconnections (`Topology.csv`).
- Transfer limits per season (`TransferLimit.csv`).

---

## Demand

Define demand in:
- **Detailed mode** (`Demand.csv`): Hourly resolution for each zone, day type, and season.
- **Simplified mode**:
  - Fractional profiles (`DemandProfile.csv`).
  - Forecasts (`DemandForecast.csv`).

---

## Fuel Price

The `FuelPrice.csv` file includes:
- Annual fuel prices by country and scenario.

---

## Fuel Limits

Optional constraints are defined in the `FuelLimits.csv` file:
- Input values for specific years, countries, and fuels.

---

## Renewable Energy

- **Profiles**:
  - Daily/seasonal variations (`RenewableEnergyProfile.csv`).
  - Generator-specific profiles (`RenewableEnergyGeneratorProfile.csv`).
- **Constraints**: Defined in the `Settings.csv` file.

---

## Reserve

- **Spinning Reserve** (`SpinReserve.csv`):
  - System-wide or country-specific requirements.
- **Planning Reserve** (`Reserve.csv`):
  - Per-unit constraints by country.

---

## Emission

- **Emission Factors** (`EmissionFactors.csv`):
  - CO2 prices and emission factors by fuel.
- **Emission Constraints** (`Emissions.csv`):
  - System-wide and country-specific constraints.

---

## Storage

Storage parameters are defined in the `Storage.csv` file:
- Capacity, CAPEX, O&M costs, efficiency, and lifetime.

---

## CSP

CSP plant characteristics are defined in the `CSP.csv` file:
- Thermal field/storage capacity, costs, and efficiency.

---

## Energy Efficiency

The `EnergyEfficiency.csv` file contains:
- Annual efficiency targets per zone.

---

## ImportShare and TradePrices

- **ImportShare.csv**: Set maximum import shares by year and country.
- **TradePrices.csv**: Hourly trade prices for external zones.

---

## CAPEX Trajectories

Define CAPEX changes in the `CapexTrajectories.csv` file:
- Fractional changes by plant and year.

