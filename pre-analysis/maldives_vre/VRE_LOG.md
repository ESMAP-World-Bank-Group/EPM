# Greater Malé VRE preparation: assumptions, issues and questions log

**Scope:** hourly VRE profiles for the five Greater Malé zones of `epm/input/data_MDV`: Male, Hulhumale', Villingili, Gulhifalhu, Thilafushi.
**Techs:** PV fixed at optimal tilt, PV 1-axis tracking, PV 2-axis tracking, onshore wind, offshore wind. One profile per tech per zone, 25 in total.
**Method and sources:** [VRE_PROFILES.md](VRE_PROFILES.md).
**Status (4 Nov 2026):**
- Profiles built and validated against ESMAP measurements.
- Not yet in `data_MDV`. The questions in §4 are open.

Status codes:
- **PROPOSED:** an assumption we used that still needs agreement.
- **OPEN:** a question waiting for an answer.
- **NOTE:** for information.
- **DONE:** settled.

---

## 1. Current results

Capacity factor, weather year 2020:

| | PV fixed | PV 1-axis | PV 2-axis | Onshore wind | Offshore wind |
|---|---|---|---|---|---|
| All 5 zones | 18.0% | 22.8–22.9% | 23.5–23.7% | 34–35% gross | 30–31% gross |
| 2015–2023 range | 17.4–18.3% | 21.9–23.4% | 22.6–24.1% | 30–43% | 26–38% |

**Deliverables:**
- hourly capacity factors (0–1) for 2015–2023, in local time (UTC+5);
- an 8,760-hour table for the weather year;
- one CSV per tech in the representative-days pipeline format (`zone, month, day, hour, value`);
- wind only for 2024–2025.

## 2. Data used

| Purpose | Source |
|---|---|
| Solar, hourly | PVGIS 5.3 (EU JRC) with ERA5 weather: fixed, 1-axis (horizontal N–S) and 2-axis trackers |
| Solar, level | Global Solar Atlas, checked against the ESMAP solar station at Velana airport (1-minute data, Dec 2015 – Mar 2018) |
| Wind, hourly | ERA5 at 100 m (Open-Meteo archive) |
| Wind, level | ESMAP lidar at Thulusdhoo (Apr 2017 – Apr 2019, 25 km NE of Malé): 5.85 m/s at 100 m |
| Turbine power curves | Renewables.ninja |
| Sample points | ESMAP 2015 candidate sites #25 (Thilafushi) and #26 (Villingili); island centres for Malé, Hulhumalé and Gulhifalhu |

## 3. Assumptions

| # | Item | Assumption | Basis | Status |
|---|---|---|---|---|
| A1 | Sample points | One point per zone (§2) | Request to centre the profiles on the ESMAP 2015 sites where available | DONE |
| A2 | PV system | 1 kWp crystalline Si, 14% system loss. Fixed PV at 3° facing south (PVGIS optimum). 1-axis is a horizontal N–S tracker | PVGIS defaults; the standard utility tracker | PROPOSED |
| A3 | PV level | Beam and diffuse calibrated separately to Global Solar Atlas | ERA5 diffuse share is 29%; GSA 42% and measured 41.5% | DONE |
| A4 | Tracker gain | Modelled gains multiplied by 0.973 (1-axis) and 0.971 (2-axis). Result: +27% and +31% over fixed | Gains from measured beam/diffuse at Velana, compared with calibrated ERA5 under the same sky model | DONE |
| A5 | Wind level | ERA5 × 1.087 (lidar ÷ ERA5 at Thulusdhoo), moved to each zone using the Global Wind Atlas pattern. Long-term mean 5.62–5.72 m/s at 100 m | GWA is 17% below the lidar, ERA5 8% below | DONE |
| A6 | Onshore turbine | Gamesa G114, 2.0 MW, 100 m hub (196 W/m², a low-wind class) | Suits about 5.7 m/s | PROPOSED, see Q1 |
| A7 | Offshore wind | Vestas V136, 3.45 MW, 100 m hub, fixed-bottom on the reef or lagoon. Same wind as onshore | No dataset shows a wind gain over the lagoon | PROPOSED, see Q1 |
| A8 | Wind losses | None in the current files (gross) | – | OPEN, see Q2 |
| A9 | Weather year | 2020 | Closest to the 2015–2023 mean for all techs | PROPOSED, see Q3 |
| A10 | Feb 29 | Dropped from the weather-year files (8,760 h) | The representative-days pipeline drops it too | DONE |

## 4. Open questions, each with a proposed default

| # | Question | Proposed default | Why | Status |
|---|---|---|---|---|
| Q1 | **Turbine type**, so that CF and capex refer to the same turbine | Onshore: low-wind large-rotor class (G114-type) with capex for that class. Offshore: V136-type, fixed-bottom | On the measured wind, gross CF is 37% for the G114, 27% for the V112 3 MW and 19% for the V90 3 MW. If the EPM capex is for a standard turbine, switch to a V112-type curve | OPEN |
| Q2 | **Wind losses:** in the profile or in EPM? | In the profile: net = gross × 0.86, before any wake loss. Add a wake loss only for multi-turbine projects, if agreed | 0.86 is net ÷ gross in the ESMAP 24-month report for its single generic 3 MW turbine at Thulusdhoo (4.9 / 5.7 GWh; availability about 92%). That report assumed no wake because it modelled one turbine | OPEN |
| Q3 | **Weather year** | 2020 | 2023, the latest PVGIS year, has the weakest wind. Solar data ends in 2023, so PV can't match the 2024–2025 load years. Wind for 2024–2025 is available | OPEN |
| Q4 | **Names for the tracking-PV techs** | `PV1AX` and `PV2AX`, as new rows in `resources/pTechData.csv` (Hourly Variation = 1, RE = 1) | `pTechData.csv` has `PV` but no tracking type | OPEN |
| Q5 | **PV capacity basis** | MWp (DC). The profiles are per kWp of panels | If capacity is entered in MWac, convert with an agreed inverter loading ratio | OPEN |
| Q6 | **Zones for wind:** profiles on all 5 zones? | Provide all 5, and limit the builds in EPM | Wind on Malé is unrealistic. Hulhulé and Hulhumalé are under the airport flight path, where height limits may rule out 100 m hubs | OPEN |
| Q7 | **Who writes `supply/pVREProfile.csv`?** | Whoever runs the representative-days selection. The profiles are ready as pipeline input | `pVREProfile` needs the chosen seasons and days (q, d, t1–t24) | OPEN |

## 5. Issues found in the data

| # | Issue | Effect | Status |
|---|---|---|---|
| D1 | PVGIS-SARAH3 (satellite, about 5 km) does not cover the Maldives (API: location out of coverage) | PVGIS-ERA5 is used, which is coarser and ends in 2023 | NOTE |
| D2 | All five zones fall in one ERA5 cell (about 28 km). The 250 m atlases show only 0.4% (solar) and 2% (wind) difference between zones | The profiles are practically identical. Zone differences belong in build limits | NOTE |
| D3 | Global Wind Atlas is 17% below the Thulusdhoo lidar (4.87 vs 5.85 m/s at 100 m) | Not used for the wind level | NOTE |
| D4 | ERA5 diffuse share is 29%, against 41.5% measured | Without correction, tracker gains would be overstated (+33% instead of +27% for 1-axis) | DONE, corrected (A3, A4) |
| D5 | At the Velana station the pyranometer GHI is 7.5% above the rotating shadowband (RSI) GHI. The pyranometer agrees with Solargis (+0.2%) and ERA5 (+0.7%) | The pyranometer is taken as the level reference | NOTE |
| D6 | Wind varies a lot between years: onshore CF 30% (2023) to 43% (2021) | A single weather year matters; consider sensitivities | NOTE |
| D7 | The ESMAP 2015 site table gives relative 0–10 scores (#24 7.4, #25 7.1, #26 6.9), not m/s | An earlier 6.5–7 m/s figure, read from the map colours, is superseded by the lidar | DONE |
| D8 | The ESMAP measurement files on energydata.info can only be downloaded in a browser, not by script. Wind dataset ID: `d2fbd01c-31f5-4180-9796-ad5828c9d742` (the short dataset URL returns 404) | Download them manually | NOTE |

## 6. Observations on the `data_MDV` files (branch state 3 Nov 2026)

These are not VRE issues, but they affect whether the model can answer the study question.

| # | Observation | Status |
|---|---|---|
| M1 | `trade/pTransferLimit.csv` and `pNewTransmission.csv` are empty. The existing Malé–Hulhumalé link and the planned link to Villingili, Gulhifalhu and Thilafushi are needed | OPEN |
| M2 | `trade/pLossFactorInternal.csv` pairs Hulhumale' with Villingili; the planned route runs from Malé to Villingili | OPEN |
| M3 | `config.csv` points to 3 files that are not in the folder: `y_alt.csv`, `load/pDemandForecast_sapp_adapted.csv`, `trade/pMaxAnnualInternalTradeShare.csv` | OPEN |
| M4 | In `zcmap.csv`, two zone names end with a space (`"Hulhumale' "`, `"Villingili "`). Fixing the zone IDs before more files use them would avoid mismatches | OPEN |
| M5 | `supply/pVREProfile.csv` has empty rows for Male only, including `CSPPlant`, which does not apply here | OPEN |

## 7. Change history

| Date | Change |
|---|---|
| 2026-11-04 | First profiles, using Global Wind Atlas for the wind level |
| 2026-11-04 | Version 2: wind level from the Thulusdhoo lidar (+20% speed), tracker gains corrected with the Velana station, weather year 2020 |
