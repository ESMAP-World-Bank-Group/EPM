# Black Sea 2026 results report: roadmap and layout

Automatic generator of a results HTML (and later a slide deck) out of an EPM run
folder. Immediate target: **Georgia + Regional overview**, run
`simulations_run_20260819_204446`.

---

## 0. What the data allows (and what it does not)

Inventory made on the run of 2026-08-19. 13 scenarios, 16 years (2025 to 2040), 13
internal zones plus `iran_swap`, 28 blocks (4 seasons x 7 day types) x 24 h = 672 time
steps.

### Sources per chart

| Need | File | Volume | Note |
|---|---|---|---|
| Annual capacity and generation by techfuel | `summary.csv` | 1.2 MB, 9,648 rows | `Capacity: MW` and `Energy: GWh`; `resolution` carries the techfuel |
| Annual imports and exports | `summary.csv` | n/a | `Imports/Exports exchange: GWh`, plus `Annual Energy Imports/Exports External: GWh` (per partner) |
| Transmission capacity | `summary.csv` + `pTransmissionMerged.csv` | n/a | `Transmission Capacity: MW`, per zone pair |
| Directed bilateral exchanges | `pTransmissionMerged.csv` (`Interchange`) | 172 kB/scenario | `Interchange[z][z2]` = flow **from z to z2**, GWh/year, which gives the arrows |
| Congestion | `pTransmissionMerged.csv` (`InterconUtilization`) | n/a | utilisation rate 0 to 1 per pair and per year |
| Hourly dispatch | `pDispatchComplete.csv` | **110 MB/scenario** | `uni` in {techfuels, `Demand`, `Imports`, `Exports`, `Storage Charge`, `Unmet demand`} |
| Hourly marginal cost | `pHourlyPrice.csv` | 7.4 MB/scenario | $/MWh, per zone and hour |
| Block weights | `input/data_blacksea/pHours.csv` | n/a | for the weighted x axis and the share of hours |
| Physical lines and project groups | `pre-analysis/data/reference_lines.csv` | 53 rows | column `project` is the group |
| External NTC (Russia, Iran, Bulgaria and so on) | `input/data_blacksea/trade/pExtTransferLimit_*.csv` | n/a | **not in the outputs**, see 0.2 |
| Geometry | `zones.geojson`, `zones_ext.geojson`, `linestring_countries.geojson` | 450 kB | 8 external neighbours as polygons |

### 0.1 Five traps identified, to handle in the generator

1. **Internal `NetImport` is wrong by a factor of 1000.** Georgia to AzerbaijanMain in
   2035: `Interchange = 2915.7` but `NetImport = -2.9157`. The *external* partners
   (Russia: 2604.9) are consistent with `summary.csv`.
   So **`NetImport` is never used for internal flows**, only `Interchange`.
2. **The capacity of the external corridors does not exist in the outputs.**
   `TransmissionCapacity` only covers internal pairs (Georgia: Armenia,
   AzerbaijanMain, EastAna, not Russia). It has to be read from
   `pExtTransferLimit_<variant>.csv` on the input side, which means resolving
   scenario to file through `input_scenarios.csv`. It is seasonal and directional
   (`q`, `Import`/`Export`), so the max over `q` per direction is taken.
3. **`baseline` is an exact clone of `LC_Baseline`** (no difference over 9,648 rows),
   so it is excluded from every selector.
4. **`CongestionShare` only exists for 3 zones** (AzerbaijanMain, EastAna,
   Nakhchivan), so `InterconUtilization` is displayed instead, available everywhere.
5. **No result per physical line.** EPM reasons per *zone pair*, while several real
   lines share one pair (TUR to GEO = Borcka to Akhaltsikhe HVDC 700 MW **plus** Hopa
   to Batumi 220 kV out of service). Consequence for the "one bar per line" chart: see
   section 2.1.5.

### 0.2 Volume and cache

13 x 110 MB of dispatch: re-reading that on every build is out of the question. A
pre-aggregation step is required:

```
tools/results_report/
  build.py          # CLI: --run, --countries, --scenarios, --out
  extract.py        # EPM run  ->  cache/<run>/<scenario>.json  (aggregates)
  charts.py         # series -> inline SVG
  maps.py           # geojson + flows -> inline SVG
  render.py         # HTML assembly with embedded JS
  templates/report.css, report.js
  cache/            # gitignored
```

`extract.py` streams `pDispatchComplete.csv` in one pass (pandas chunked read), keeps
only the requested dispatch years (2025 / 2030 / 2035), aggregates at **country** level
(sum of the zones) and writes compact JSON. Order of magnitude for Georgia: 672 steps x
about 12 series x 3 years x 2 scenarios, so roughly 48k numbers, about 250 kB once
rounded to one decimal. Same order for Turkiye (8 zones aggregated into 1).

### 0.3 Technical choice, my recommendation

**Self-contained HTML, inlined JSON data, rendering in vanilla JS (SVG), no CDN.**

- Same contract as `calibration_review.html`: one file, openable as `file:///`, sendable
  by mail, no network.
- But **with JS** this time (the calibration review is hand written static SVG). The
  interactivity asked for, hovering the map arrows, dispatch tooltips, the Baseline and
  Iso toggle, is not reachable in static SVG without duplicating every chart per
  scenario.
- No Chart.js and no MapLibre: about 500 lines of in-house JS are enough for (a) the
  stacked area, (b) the stacked bars, (c) the projected map. Loading them from a CDN
  would break offline use, and embedding them would add 900 kB.
- **Map**: equirectangular projection of the geojson into SVG paths, computed in Python
  at generation time (geometries simplified with Douglas-Peucker at about 0.02 degrees).
  No tiled basemap, just a grey background plus borders, like the maps of the calibration
  review. The advantage is that it stays fully offline and fully controllable.

Target budget: **under 4 MB** for Georgia plus Regional. If it goes over, the dispatch
moves to one average day per season (28 blocks down to 4).

---

## I. Baseline vs Iso

Global header bar: scenario selector (Baseline / Iso / delta), language selector (same
`.lf` and `.le` mechanism as the calibration review), and a reminder of the run and date.

### I.a Per country

#### I.a.1 Georgia *(first deliverable)*

**Section 1: annual capacity and generation, Baseline vs Iso**

Two charts side by side, 2025 to 2040, one group of two bars per year (Baseline | Iso),
stacked by techfuel.

- Left: **capacity (GW)**. Techfuels stacked, plus **the interconnection capacity above,
  hatched diagonally** (sum of the internal `Transmission Capacity` and the external
  NTC), on the same MW axis.
- Right: **generation (TWh)**. Techfuels stacked, plus **imports hatched above zero** and
  **exports hatched below zero**.
- Hatching: SVG `<pattern>`, 45 degrees, colour of the dominant partner, opacity 0.55, so
  that it reads as "this is not domestic production".
- Hover: value, share of the total, gap between Baseline and Iso.

**Section 2: comparison with the NDP (Baseline only)**

Direct reuse of the `calibration_review.html` section 5 template: three bars per year
(Plan | Model | delta), capacity and generation. The `ndp_build.py` code is already
written and validated, it gets ported into `charts.py`. Below the chart, the delta panels
sorted by absolute delta, descending.
*(For Georgia the plan is the GSE hydro pipeline. The 2035 and 2040 figures are already
up to date in `_ndp_cmp_data.json`.)*

**Section 3: hourly dispatch, 2025, 2030, 2035**

One chart per year (3 stacked vertically), following the
`epm-data-explorer/src/utils/dispatchSeries.js` convention exactly:

- Stacked area of the techfuels (`TECHFUEL_COLORS` from the explorer).
- `Imports` and `Storage Charge` in the stack, `Exports` and `Storage Charge` **below
  zero** (EPM writes them positive in `pDispatchComplete`, so they are inverted here).
- `Unmet demand` in bright red at the top.
- **Demand line** `#CC0000`, width 1.5.
- **Marginal cost line** on the right axis ($/MWh), from `pHourlyPrice.csv`.
- Grouped x axis: 4 seasons x 7 day types, vertical separators (solid between seasons,
  dotted between day types), and **the share of the year in percent under each block**
  (from `pHours.csv`).
- Selector above: `Full year | Q1 | Q2 | Q3 | Q4` and `all days | average day`.

**Section 4: evolution of the exchanges**

Stacked bars 2025 to 2040: imports (+) and exports (-) **per partner** (AzerbaijanMain,
EastAna, Armenia, Russia, Romania and so on), one colour per partner, with a **net
marker** (dot) per year, like `buildTrade()` in the explorer. Two panels, Baseline and
Iso, or one delta panel depending on the selector.

**Section 5: flow maps, 2026, 2030, 2035**

Three maps side by side, centred on Georgia and its neighbours.

- Internal zones filled (cool palette), external zones dotted grey.
- **Arrows** between centroids: width proportional to the GWh exchanged, orientation is
  the net direction, colour is the utilisation rate (`InterconUtilization`) on a green to
  orange to red gradient, with the saturated line (above 90 percent) in red with a rim.
- Arrows towards the **external zones** included (Russia, Romania and so on), pointing at
  the centroid of the neighbouring polygon.
- **Hover**: partner, GWh imported, GWh exported, net, NTC (MW), utilisation percent.
- Permanent net label in GWh on the three largest corridors.

**Section 6: capacity per line, grouped by project**

Grouped bars: one group per **project** (`reference_lines.csv:project`, so BSTN, CTN,
EWTC, GECO, BSSC, Zangezur, Trans-Caspian, Mid-Continental East, "existing lines"), one
bar per corridor inside the group, and **4 sub bars per corridor** (2025 / 2030 / 2035 /
2040) to read the evolution.

> **Open point (section 0.1, trap 5)**: EPM does not return capacity line by line, only
> per zone pair. So the proposal is **one bar per zone pair**, with the detail of the
> physical lines that make it up (substation name, kV, status, commissioning year) in the
> tooltip. That is honest and readable, whereas a "per line" bar would be an invented
> split.

#### I.a.2 to I.a.4: Turkiye, Azerbaijan, Armenia

Exactly the same template, parameterised by country. Two specifics:

- **Turkiye**: 8 zones. The annual charts and the dispatch are aggregated at country
  level, but the map keeps the 8 zones and also shows the **internal flows** (WestAna to
  CenterAna and so on), which are the bulk of the volume.
- **Azerbaijan**: Nakhchivan is a separate and enclaved zone, so the map has to treat it
  as such (already done in `zones.geojson`, see commit `f5ad8c24`).

### I.b Regional overview *(second deliverable)*

**Section 1: regional capacity and generation**

- Stacked by **country** (4 colours), then in a second chart by **techfuel** for the
  whole region. Baseline and Iso side by side.
- Key figures band above: installed capacity, generation, intra regional exchanges,
  exchanges with the outside, emissions, NPV cost, for each scenario, with the delta.

**Section 2: regional flow maps, 2026, 2030, 2040**

Three maps **in a left column (about 62 percent of the width)**, and **facing them on the
right the key findings** written next to each map, in a box:

```
+---------------------------+----------------------+
|  Map 2026                 |  Key findings 2026   |
|  (flows + congestion)     |  . ...               |
+---------------------------+----------------------+
|  Map 2030                 |  Key findings 2030   |
+---------------------------+----------------------+
|  Map 2040                 |  Key findings 2040   |
+---------------------------+----------------------+
```

The findings are **computed and then written up**: the generator produces the facts (most
loaded corridor, saturated corridors, largest reversal of direction between two years,
maximum external dependency) and the narrative text is written on top, as in the
calibration review. Every figure quoted is an extracted figure, never typed by hand, so
nothing drifts when the run changes.

**Section 3: regional imports and exports**

- Origin destination matrix (14x14 heatmap, GWh) per year, with a year selector.
- Import and export bars per country, and **the balance with the outside of the region**
  (Russia, Iran, Bulgaria, Greece, Romania, Kazakhstan) kept separate from the intra
  regional exchange. That separation is the reading that matters for RETRADE.
- Curve of the average corridor utilisation rate, 2025 to 2040, per scenario.

---

## II. Build order

| Step | Content | Validation |
|---|---|---|
| **1** | `extract.py` plus cache for LC_Baseline and LC_Iso, Georgia only | the cache totals match `summary.csv` within 0.1 percent |
| **2** | I.a.1 sections 1, 2, 4 (annual, NDP, exchanges) | manual comparison against `summary.csv` |
| **3** | I.a.1 section 3 (dispatch) | sum weighted by `pHours` equals the annual `Energy: GWh` |
| **4** | I.a.1 sections 5 and 6 (map and lines) | inbound flows equal outbound flows per corridor |
| **5** | **Review with you** | stop here before going further |
| **6** | I.b complete (Regional) | |
| **7** | **Review** | |
| **8** | Turkiye, Azerbaijan, Armenia (same template) | |
| **9** | Slide export (`python-pptx`) from the same series | |

---

## III. Open decisions

1. **Scenario scope.** The outline only covers Baseline vs Iso. The 11 others (BSTN, CTN,
   EWTC, GECO, BSSC, Zangezur, TransCaspian, 60pct, AllProjects, FreeExp): sections II
   and III later, or a global selector that makes any pair comparable right away? *I
   recommend the selector*, since the marginal cost is nil once the template is
   parameterised.
2. **Map years.** You said 2026/2030/2035 for the countries and 2026/2030/2040 for the
   regional view. Deliberate, or do we harmonise on 2026/2030/2040 everywhere (2035 being
   already covered by the dispatch)?
3. **Bars per line vs per corridor**, see I.a.1 section 6.
4. **Language.** Settled: **English only**, everywhere.
5. **Where the file lives.** Proposal: output in
   `blacksea_2026/Data/results/results_review.html`, code in
   `EPM/tools/results_report/`, cache gitignored.
