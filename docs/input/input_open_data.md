# Open Data Sources

A curated list of open data sources for populating EPM inputs, organized by category.

!!! info "Work in progress"
    This page is being expanded. If you have suggestions for additional sources or experience with specific datasets, please [open an issue](https://github.com/ESMAP-World-Bank-Group/EPM/issues) on GitHub.

---

## Electricity Demand

| Source | Resolution | Description | Link |
|---|---|---|---|
| Our World in Data | Yearly | Historical electricity consumption by country | [ourworldindata.org](https://ourworldindata.org/energy) |
| ENTSO-E | Monthly / hourly | Load data for European countries | [Transparency Platform](https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show) |
| synde | Hourly | Modelled demand under SSP scenarios (GEGIS) | [GitHub](https://github.com/euronion/synde) |

---

## Existing Generation Capacity

| Source | Coverage | Description | Link |
|---|---|---|---|
| Global Energy Monitor | Global | Unit-level power plant data worldwide | [GEM](https://globalenergymonitor.org/projects/global-integrated-power-tracker/) |
| PowerPlantMatching | Europe | Matched plant database, CSV download | [GitHub](https://github.com/PyPSA/powerplantmatching) |
| Global Power Plant Database | Global | Plant capacity, fuel, ownership (no longer maintained) | [GitHub](https://github.com/wri/global-power-plant-database) |

---

## Solar and Wind Profiles

| Source | Resolution | Description | Link |
|---|---|---|---|
| Renewables.ninja | Hourly | Simulated PV and wind capacity factors | [renewables.ninja](https://www.renewables.ninja) |
| Global Solar Atlas | — | Solar resource maps and data | [globalsolaratlas.info](https://globalsolaratlas.info/) |
| Global Wind Atlas | — | Wind resource maps and data | [globalwindatlas.info](https://globalwindatlas.info/en/) |
| atlite | Hourly | Python library for weather-derived power profiles | [Docs](https://atlite.readthedocs.io/en/latest/) |
| Sterl et al., 2022 | — | PV and wind supply regions across Africa | [Paper](https://www.nature.com/articles/s41597-022-01786-5) |

---

## Hydropower

| Source | Resolution | Description | Link |
|---|---|---|---|
| EIA | Yearly | Historical hydro generation by country | [EIA](https://www.eia.gov/international/data/world) |
| GRDC | Monthly | Global river discharge and runoff data | [grdc.bafg.de](https://grdc.bafg.de/) |
| IRENA / Sterl et al., 2022 | — | Hydropower generation profiles across Africa | [Paper](https://open-research-europe.ec.europa.eu/articles/1-29/v3) · [Data](https://www.hydroshare.org/resource/5e8ebdc3bfd24207852539ecf219d915/) |
| FAO AQUASTAT | — | Geo-referenced database of dams | [FAO](http://www.fao.org/aquastat/en/databases/dams) |
| Global Dam Watch (GRanD / FHReD) | — | Reservoir and dam database (existing + future) | [globaldamwatch.org](http://globaldamwatch.org/grand/) |

---

## Comprehensive / Multi-Category

| Source | Description | Link |
|---|---|---|
| Ember | Global power data for 85+ geographies (generation, emissions, demand) | [ember-energy.org](https://ember-energy.org/data/) |
| PyPSA-Earth | Global electricity model with full data workflow | [Docs](https://pypsa-earth.readthedocs.io/en/latest/data_workflow.html) |
| ENERGYDATA.INFO | World Bank open data platform for the energy sector | [energydata.info](https://energydata.info/dataset/?organization=world-bank-grou&vocab_topics=Power+system+and+utilities) |
| OSeMOSYS (Brinkerink et al., 2021) | Global energy system data | [Paper](https://www.nature.com/articles/s41597-022-01737-0) |
