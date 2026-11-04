# Greater Malé VRE profiles: method and sources

**Status:** 4 Nov 2026, version 2, calibrated to the ESMAP measurements. Not yet in `data_MDV`. Assumptions, issues and open questions are tracked in [VRE_LOG.md](VRE_LOG.md).
**Where things are:** the scripts, raw downloads and output CSVs are kept in the project working folder (`EPM_MDV/vre/`), not in this repo. Only this document and the log are in the repo.

**Request (3 Nov 2026):** hourly profiles for each of the 5 islands, centred on the ESMAP 2015 candidate sites where possible:
- PV fixed at optimal tilt;
- PV with 1-axis tracking;
- PV with 2-axis tracking;
- onshore wind;
- offshore wind.

That gives **one profile per technology per island, 25 in total.** The email does not ask for several sites or resource classes per island.

**What the profiles are:** hourly capacity factors (output per MW installed, 0–1) in Maldives local time (UTC+5). They are built from weather data (sunlight, wind speed) with the technology assumptions in §3.

---

## 1. Results

Capacity factor (CF) in the weather year 2020. The weather year is chosen in §4.

| Zone | PV fixed | PV 1-axis | PV 2-axis | Onshore wind* | Offshore wind* |
|---|---|---|---|---|---|
| Malé | 18.0% | 22.8% | 23.6% | 35.1% | 31.0% |
| Hulhumalé | 18.0% | 22.9% | 23.7% | 34.8% | 30.7% |
| Villingili | 17.9% | 22.8% | 23.5% | 35.4% | 31.2% |
| Gulhifalhu | 18.0% | 22.8% | 23.5% | 34.4% | 30.3% |
| Thilafushi | 18.0% | 22.9% | 23.6% | 34.3% | 30.2% |

\* Gross: no wake, availability or electrical losses (§6). PV includes a 14% system loss.

Spread across the years 2015–2023, averaged over the five zones:

| Tech | Mean | Lowest year | Highest year |
|---|---|---|---|
| PV fixed | 17.8% | 17.4% | 18.3% |
| PV 1-axis | 22.5% | 21.9% | 23.4% |
| PV 2-axis | 23.3% | 22.6% | 24.1% |
| Onshore wind | 35.6% | 29.6% (2023) | 42.7% (2021) |
| Offshore wind | 31.4% | 25.6% | 38.0% |

**What this means for the model:**
- **The five islands get practically the same profiles.** The hourly correlation between islands is above 0.998.
  - The coarse datasets cannot separate the islands: all five fall in one ERA5 cell of about 28 km.
  - The 250 m atlases can separate them, and show only tiny differences: 0.4% for solar, 2% for wind.
  - The islands are flat, at the same latitude and in the same open ocean, and clouds and monsoon winds arrive in systems hundreds of kilometres wide.
  - What does differ between islands is the space to build, building shade and airport height limits. These belong in EPM's build limits and costs, not in the profiles.
- **Tracking gains over fixed PV, checked against measured sunlight:** +27% for 1-axis and +31% for 2-axis. Their main value is output in the early morning and late afternoon.
- **Wind is moderate and very seasonal.** The measured level at 100 m is 5.6–5.7 m/s long-term at the islands. CF is high in the monsoon months (May–October and December–February) and low in the inter-monsoon months (March–April and November).
- **Wind CF depends heavily on the turbine** (§3.2). The 35% applies to a large-rotor, low-wind turbine. With a small-rotor 3 MW turbine the same wind gives about 19%.

## 2. Files delivered (`EPM_MDV/vre/out/`)

| File | Content |
|---|---|
| `vre_hourly_2020.csv` | **Main deliverable.** 8,760 hours of the weather year (29 Feb dropped), with one column per zone and tech, e.g. `Male_PV1AX` |
| `repdays_input/{tech}_2020.csv` | The same data in the format of the repo's representative-days pipeline: `zone, month, day, hour (0–23), value`, one file per tech |
| `vre_hourly_2015_2023.csv` | All complete years (78,888 h), for choosing another weather year or testing robustness |
| `wind_hourly_2024_2025.csv` | Wind for 2024–2025, the years with STELCO hourly load. PVGIS has no solar data after 2023 |
| `summary_cf.csv` | Annual CF by zone and tech, with the calibration factors and the long-term wind speed |
| `power_curves_ninja.csv` | The two turbine power curves used |
| `validation/` | Comparison with the ESMAP measurements: summary, monthly and daily-cycle tables, and `calibration.json` |
| `build_log.txt`, `validation_log.txt` | Console output of the last runs |

The EPM tech names `PV1AX` and `PV2AX` are proposals (VRE_LOG Q4). Once agreed, they need rows in `resources/pTechData.csv`.

## 3. Method

### 3.1 Solar PV

1. **Hourly shape** comes from **PVGIS 5.3** (EU JRC) using the **PVGIS-ERA5** radiation database for 2014–2023. Each system is 1 kWp, crystalline silicon, with a 14% system loss (the PVGIS default):

   | Tech | PVGIS setting | Geometry |
   |---|---|---|
   | `PV` | `trackingtype=0`, optimal angles | PVGIS picks a 3° slope, facing roughly south |
   | `PV1AX` | `trackingtype=1` | Single horizontal axis aligned north–south, the usual utility tracker |
   | `PV2AX` | `trackingtype=2` | Two-axis |

2. **Level calibration to Global Solar Atlas (GSA).** ERA5 has too much direct beam and too little diffuse light here:

   | Source | Diffuse share of sunlight |
   |---|---|
   | ERA5 | 29% |
   | GSA | 42% |
   | Measured at Malé airport (§5) | 41.5% |

   Too much beam inflates tracker gains, because trackers gain only on the beam. So beam and diffuse are scaled separately, using the ERA5 2015–2023 horizontal means against the GSA long-term values: beam by about 0.81 and diffuse by about 1.40. Each system's output is then multiplied by `(fb·Gb + fd·Gd + fg·Gr) / G`, using its in-plane beam/diffuse/reflected split from PVGIS for 2023.

3. **Tracker gain correction from the measured data.** Even after step 2, the tracker gains from ERA5 are higher than those from the measured beam and diffuse. Both were computed with the same simple sky model; see §5. The tracker profiles are therefore multiplied by the ratio of measured to modelled gain:

   | Tech | Correction factor |
   |---|---|
   | 1-axis | 0.973 |
   | 2-axis | 0.971 |

4. **Result for fixed PV:** 1,576 kWh/kWp in 2020 and 1,559 kWh/kWp on the 2015–2023 average. GSA's own estimate (`PVOUT`) is 1,625–1,632 kWh/kWp, about 4% higher. GSA uses its own loss assumptions, and ours are slightly more conservative.

### 3.2 Wind

1. **Hourly wind speed:** ERA5 at 100 m, 2014–2025, from the Open-Meteo archive API. All five zones fall in **the same ERA5 cell** (4.25° N, 73.5° E).
2. **Level, from the ESMAP Thulusdhoo lidar** (§5). The lidar measured 5.85 m/s at 100 m over 738 days in 2017–2019. That is **8.7% above ERA5** and **17% above Global Wind Atlas** at the same point. The ERA5 speeds are scaled by:

   `k = (lidar / ERA5 at Thulusdhoo) × (GWA at zone / GWA at lidar) / (ERA5 Malé cell / ERA5 Thulusdhoo cell)`

   - The first term carries the measured level over to our data.
   - The other two move it from Thulusdhoo to each zone: GWA supplies the 250 m spatial pattern, net of the difference ERA5 already has between its two cells.
   - The result is k = 1.07–1.09, giving **long-term means of 5.62–5.72 m/s** at the five zones.
   - This assumes Greater Malé relates to ERA5 the way Thulusdhoo does. Both sites are on the rim of North Malé atoll, 25 km apart.
   - *Version 1 used GWA for the level (about 4.75 m/s). The lidar shows that was too low.*
3. **Turbines.** These are proposals (VRE_LOG Q1). Specific power is rated power per m² of rotor; low values suit low wind and give higher CF.

   | Tech | Turbine | Rotor | Hub | Specific power |
   |---|---|---|---|---|
   | Onshore | Gamesa G114, 2.0 MW | 114 m | 100 m | 196 W/m² |
   | Offshore | Vestas V136, 3.45 MW | 136 m | 100 m | 237 W/m² |

   - **Offshore means fixed-bottom on the shallow reef or lagoon.** No dataset shows stronger wind over the lagoon than on the islands, so the offshore profile uses the same wind and differs only through the turbine.
   - **The turbine choice drives the CF.** On the measured Thulusdhoo wind:

     | Turbine | Specific power | Gross CF |
     |---|---|---|
     | G114 2.0 MW | 196 W/m² | 36.6% |
     | Vestas V112 3.0 MW | 305 W/m² | 26.7% |
     | Vestas V90 3.0 MW | 472 W/m² | 18.7% |

     ESMAP's report gives 18.7% *net* for a generic 3 MW turbine with a 100 m rotor (382 W/m²).
   - Large-rotor turbines cost more per MW, so the **capex in EPM must be for the same turbine class**.
4. **Power curve:** taken from Renewables.ninja's own hourly pairs of wind speed and output for each turbine in 2023, as the median in 0.25 m/s bins. Ninja applies its standard smoothed (farm-level) power curve.

   | Speed | 4 m/s | 5 m/s | 6 m/s | 8 m/s | 12 m/s |
   |---|---|---|---|---|---|
   | Onshore CF | 10.5% | 21.4% | 36.1% | 67.8% | 96% |

## 4. Weather year

- **PVGIS-ERA5 ends in 2023.** The common years for all sources are therefore 2015–2023; 2014 lacks its first five local hours.
- The build picks the year whose annual CFs, across all zones and techs, are closest to the 2015–2023 mean. That year is **2020**, with a mean deviation of 1.7%. The other years deviate by 3.5–10%.
- Onshore wind CF by year at Malé:

  | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 33.2% | 41.3% | 37.9% | 32.7% | 30.3% | 35.1% | 43.1% | 40.1% | 29.9% | 34.0% | 38.0% |

  **2023, the latest PVGIS year, has the weakest wind.**
- The STELCO hourly load is for 2024–2025, so the weather and load years differ. In the tropics the link between solar output and load is mostly the daily cycle, so the effect should be small.
- To match the load year instead:
  - wind is available for 2024 and 2025 (`wind_hourly_2024_2025.csv`);
  - solar would need another source, because PVGIS-ERA5 stops at 2023.

## 5. Validation against ESMAP measurements (`validate_esmap.py`)

### 5.1 Wind: Thulusdhoo lidar

DNV GL ran the ZephIR lidar for ESMAP at 4.37453 N, 73.65217 E. Data: 10-minute records, 7 Apr 2017 – 14 Apr 2019, 17,190 hours matched with ERA5.

| Series | Mean at 100 m | vs lidar | Spread (std/mean) | Hourly corr. | Daily corr. | Gross CF, onshore turbine |
|---|---|---|---|---|---|---|
| **Lidar, measured** | **5.85 m/s** | – | 0.45 | – | – | **36.6%** |
| ERA5, Thulusdhoo cell | 5.38 | −8% | 0.47 | 0.88 | 0.96 | 32.3% |
| ERA5, Malé cell | 5.31 | −9% | 0.47 | 0.88 | 0.96 | 31.5% |
| Version 1 method (ERA5 scaled to GWA) | 5.09 | −13% | 0.47 | 0.88 | 0.96 | 28.9% |
| **Version 2 method (ERA5 × lidar/ERA5)** | 5.85 | 0% | 0.47 | 0.88 | 0.96 | **37.6%** |
| Renewables.ninja MERRA-2 | 5.14 | −12% | 0.50 | 0.80 | 0.89 | 30.2% (ninja's own CF 30.3%) |
| Global Wind Atlas (long-term) | 4.87 | −17% | – | – | – | – |

- **ESMAP's 24-month report** gives 5.9 m/s measured and 5.9 m/s long-term at 100 m (Table 4-4: long-term adjustment 100.3%). Our 5.85 m/s matches.
- The report finds R² = 0.97 between monthly ERA5 and the lidar, which supports using ERA5 for the shape.
- **Shape:** ERA5 reproduces the monthly pattern. Measured monthly means at 100 m:

  | Period | Mean speed |
  |---|---|
  | March–April | 4.0–4.4 m/s |
  | November | 4.2 m/s |
  | May | 7.2 m/s |
  | June–October | 6.1–6.7 m/s |
  | December–February | 5.9–6.4 m/s |

  ERA5 runs 0.2–0.8 m/s lower each month. The daily cycle is weak in both (5.6–6.1 m/s measured).
- **Spread:** scaling ERA5 by one factor keeps its slightly wider spread, so the version 2 CF is 1 point above the measured CF (37.6% vs 36.6%).
- **Measured shear 50→100 m:** α = 0.07, a low value typical of open sea.

### 5.2 Solar: Malé airport station

The Suntrace Tier 2 station for ESMAP is at Velana airport, Hulhulé (4.19274 N, 73.52811 E). It measured 1-minute data from Dec 2015 to Mar 2018: a rotating shadowband irradiometer (RSI) for GHI, DNI and DHI, plus an SR20 pyranometer for GHI. Only minutes passing Solargis quality control were kept, giving 8,447 matched daytime hours. The station is compared with the Solargis satellite series at the station (ESMAP, 1999–2018) and with PVGIS-ERA5.

| Dataset | GHI vs measured (RSI) | Diffuse share | Hourly corr. | Daily corr. | 1-axis gain* | 2-axis gain* |
|---|---|---|---|---|---|---|
| **Measured (RSI)** | 1.000 | **41.5%** | – | – | **+15.5%** | **+18.6%** |
| Solargis satellite | 1.077 | 45.2% | 0.96 | 0.98 | +12.6% | +15.3% |
| ERA5, raw | 1.083 | 28.5% | 0.88 | 0.91 | +24.4% | +28.8% |
| ERA5, calibrated as in §3.1 | 1.061 | 40.6% | 0.87 | 0.88 | +18.7% | +22.1% |

\* Gains in sunlight on the panel over fixed 3° PV, all computed with the same simple isotropic sky model and the Solargis sun position. That model ignores light from around the sun's disc, so its gains are lower than PVGIS's. The comparison between datasets is like for like. The ratio of measured to calibrated-ERA5 gain gives the correction factors in §3.1, step 3.

- **The level of total sunlight is confirmed.** The station's two instruments disagree: the pyranometer reads 7.5% above the RSI. The pyranometer, a secondary-standard thermopile, is the better GHI reference, and against it:

  | Source | vs pyranometer |
  |---|---|
  | Solargis | +0.2% |
  | ERA5 | +0.7% |

  GSA long-term GHI at the station is 2,042 kWh/m²; Solargis 1999–2017 gives 2,052. So the GSA level used in §3.1 stands.
- **The diffuse share is confirmed:** measured 41.5%, against GSA 41.7% and our calibrated ERA5 40.6%.
- **Tracker gains:** our calibrated ERA5 still overstated them by about 3 points, now corrected. The 1-axis gain fell from +30% to +27% over fixed PV.
- **Shape:** ERA5 places cloudy days less well than the satellite (daily correlation 0.88–0.91, against 0.98 for Solargis). The daily cycle and the seasons match; May–July is the cloudy season.

## 6. Limitations and open decisions

1. **Wind CFs are gross.** Wake, availability, electrical and curtailment losses are not deducted. For SC we applied 95% availability on top of the profile. A loss assumption is needed, either in the profile or in EPM; I have not chosen one.
2. **Turbine class and cost must match.** The 35% onshore CF assumes a 196 W/m² low-wind turbine (§3.2). If EPM's wind capex is for a standard turbine, the CF should be about 19–27%.
3. **The wind level is transferred from one site.** Thulusdhoo is 25 km from Malé, on the same atoll rim, and the lidar sat in an open compound near the eastern shore. Wind among Malé's dense buildings would be lower; realistic sites are the reef, the lagoon and the open reclaimed land on Thilafushi and Gulhifalhu.
4. **Turbine sites.** The Velana airport flight path covers Hulhulé and Hulhumalé, and nearby islands may be affected too. Turbine height limits there probably rule out 100 m hubs. Onshore wind on Malé itself is unrealistic. The profiles exist for every island as requested, but the build limits in EPM must reflect this.
5. **Solar weather is from a reanalysis.** ERA5's day-to-day cloud pattern is less accurate than satellite data. If cloudy-day variability matters (BESS sizing), the Solargis series at Malé is a better shape source, but it ends in April 2018.
6. **Open decisions** (tracked in VRE_LOG §4):
   - the tech names (`PV1AX`, `PV2AX`);
   - the turbine choice, hub height and matching capex;
   - the weather year (2020 proposed);
   - whether losses go into the profile or into EPM;
   - whether the 1 kWp DC-based PV CF matches how capacity is entered in EPM (MWp or MWac).

## 7. Data sources

### Used

| Dataset | Provider | Resolution | Years | Used for | Licence | Access |
|---|---|---|---|---|---|---|
| PVGIS 5.3, PVGIS-ERA5 | EU Joint Research Centre | ERA5, about 31 km | 2005–2023 | Hourly PV shape and tracker modelling | Free use with acknowledgement | API `re.jrc.ec.europa.eu/api/v5_3/seriescalc`, script `fetch_pvgis.py` |
| Global Solar Atlas 2.x | World Bank / ESMAP, Solargis | About 250 m, satellite | Long-term to 2025 | PV level: GHI, DIF, PVOUT | CC BY 4.0 | API `api.globalsolaratlas.info/data/lta`, `fetch_atlas.py` |
| ERA5 | ECMWF / Copernicus, via Open-Meteo | 0.25° | 2014–2025 | Hourly wind shape at 100 m | CC BY 4.0. Open-Meteo's free API is for non-commercial use | `archive-api.open-meteo.com/v1/archive?models=era5`, `fetch_era5_wind.py` |
| **ESMAP Maldives Wind Measurement**, Thulusdhoo lidar, plus the DNV GL 24-month report | World Bank / ESMAP | Point, 10-minute | Apr 2017 – Apr 2019 | **Wind level** (and validation) | CC BY 4.0 | energydata.info dataset `d2fbd01c-31f5-4180-9796-ad5828c9d742`. Browser download only |
| **ESMAP Maldives Solar Radiation Measurement**, Malé station, plus the Solargis satellite series | World Bank / ESMAP, Suntrace, Solargis | Point, 1-minute; satellite hourly | Dec 2015 – Mar 2018; 1999 – Apr 2018 | **Tracker gain correction** (and validation) | CC BY 4.0 | energydata.info `maldives-solar-radiation-measurement-data`. Browser download only |
| Global Wind Atlas 3 | DTU / World Bank / ESMAP | 250 m | Long-term | Small spatial adjustment between the lidar and the zones only. Its level is 17% low here | CC BY 4.0 | MDV GeoTIFFs from `globalwindatlas.info/api/gis/country/MDV/wind-speed/{50,100}` |
| Renewables.ninja | Pfenninger and Staffell | MERRA-2, about 50 km, interpolated | 1980–2025 | Turbine power curves and cross-checks | CC BY-NC 4.0 | API with token (`pre-analysis/config/api_tokens.ini`), `fetch_ninja.py` |
| ESMAP 2015 Candidate Site Identification Report (DNV GL) | World Bank / ESMAP | – | 2015 | Sample points: sites #25, #26 | World Bank open-access report (licence not checked) | energydata.info, Maldives Wind Measurement Data (Candidate Site Identification Report) |

### Checked, not used

| Dataset | Why not used |
|---|---|
| PVGIS-SARAH3 (satellite, about 5 km), used for Seychelles | Does not reach 73.5° E: the API says the location is out of coverage |
| Renewables.ninja PV | Its 1-axis option turns around a vertical axis, not the horizontal N–S axis of utility trackers. Its 2-axis gain is only 11%. Used as a cross-check only |
| NASA POWER hourly | 0.5° grid, wind only at 50 m; adds nothing over ERA5 |
| Hoarafushi lidar | Far north of the Maldives, a different wind climate |
| ESMAP 2015 Mesoscale Wind Modelling Report | Superseded by the measurements. The 6.5–7 m/s in earlier notes was read from its map colours; the ESMAP 2015 scores (#24 7.4, #25 7.1, #26 6.9) are relative 0–10 scores, not m/s |

## 8. Sample points (`sites.py`)

| Point | Lat | Lon | Basis |
|---|---|---|---|
| Malé | 4.1755 | 73.5093 | Island centre |
| Hulhumalé | 4.2160 | 73.5420 | Island centre (Phase 1/2) |
| Villingili | 4.17088 | 73.48658 | ESMAP 2015 site #26 Vilingilli |
| Gulhifalhu | 4.1789 | 73.4689 | Island centre |
| Thilafushi | 4.18277 | 73.43135 | ESMAP 2015 site #25 Thilafushi |
| ESMAP #24 (offshore check) | 4.25915 | 73.37954 | Kodhdhipparufinolhu; same GWA speed as the zone points |
| Malé solar station | 4.19274 | 73.52811 | ESMAP measurement site, validation only |
| Thulusdhoo lidar | 4.37453 | 73.65217 | ESMAP measurement site, validation and wind level |

## 9. Reproduce

Run with the EPM `.venv` Python, which has rasterio (needed for the GWA tifs). Put the ESMAP files in `raw/esmap/`, and unzip the Malé satellite CSV into `raw/esmap/satellite/`.

```
cd EPM_MDV/vre
python fetch_pvgis.py        # ~260 requests, ~30 min; one year per request (longer requests get dropped)
python fetch_atlas.py        # GSA API + GWA country tifs
python fetch_era5_wind.py    # Open-Meteo ERA5, 6 points
python fetch_ninja.py        # 17 requests; token in EPM/pre-analysis/config/api_tokens.ini
python fetch_validation.py   # model data at the two ESMAP sites
python validate_esmap.py     # comparison tables + out/validation/calibration.json
python build_profiles.py     # writes out/ (needs calibration.json)
```

Raw downloads are kept in `raw/` and are not fetched again if present.
