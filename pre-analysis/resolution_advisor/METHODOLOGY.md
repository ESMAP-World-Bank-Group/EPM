# Resolution Advisor — detailed methodology

This document explains, step by step, **what the code does**, **why**, and **how each file contributes to the final result**.

---

## The central question

Before building an EPM model, two questions must be answered:

> **How many zones** are needed for the model to be physically credible?
> **How many representative days** are needed for the chronology to be properly captured?

Too few zones → real network constraints are missed, and prices and dispatch are wrong.
Too many zones → the model becomes too heavy and solve times explode.
The same reasoning applies to representative days.

The Resolution Advisor computes a **floor** (the minimum physics demands) and a **ceiling** (the maximum the compute budget allows), then proposes test points between the two.

---

## The two modes

### Manual mode (`--config`)
You supply the parameters yourself in a YAML file (`config/blacksea.yaml`).
The code simply runs the calculations on what you provided.

### Auto mode (`--auto`)
The code **fetches the data itself** from the internet (OSM, Natural Earth, GPPD),
then computes the parameters automatically before running the same calculations.

---

## Step-by-step execution (auto mode)

When you run:
```bash
python advise.py --countries TUR ROU BGR --auto
```

Here is exactly what happens, in order:

---

### Step 1 — Loading the geographic data
**File: `fetch/natural_earth.py`**

Loads two datasets from [Natural Earth](https://www.naturalearthdata.com/):

**Country boundaries** (`ne_110m_admin_0_countries.shp`)
- Border polygons for each country
- Used to: compute country area, draw bounding boxes, clip geometries
- 110m resolution (sufficient for our purposes)

**Populated places** (`ne_110m_populated_places.shp`)
- All cities with their estimated population
- Used to: identify load centers and compute their geographic dispersion

The files are first looked for in `dataset/maps/` (if they exist in the repo),
otherwise downloaded from naciscdn.org and cached in `cache/natural_earth/`.
Subsequent runs use the cache — no re-fetch.

---

### Step 2 — Loading the power plants
**File: `fetch/gppd.py`**

Loads the **Global Power Plant Database** (WRI) — ~35,000 plants worldwide with:
location (lat/lon), fuel, installed capacity (MW), commissioning year.

Used to: identify whether hydro is concentrated far from the load centers
(which justifies an extra zone to represent the transmission corridor).

> **Note**: the WRI URLs are currently dead (404). If the data does not download,
> the hydro parameters are skipped (everything else works normally).
> To enable: download manually from https://datasets.wri.org/dataset/globalpowerplantdatabase
> and place at `cache/gppd/global_power_plant_database.csv`.

---

### Step 3 — Collecting OSM network data
**File: `fetch/osm.py`**

Queries OpenStreetMap's **Overpass API** to retrieve HV electrical infrastructure:

**Substations** (`power=substation`):
- Every substation within the country bounding box
- Attributes retrieved: position (lat/lon), voltage (kV), name

**HV lines** (`power=line` or `power=cable` with a `voltage` attribute):
- All transmission lines >= 100 kV
- Attributes: full geometry (list of coordinates), voltage (kV)

Queries are **cached** (MD5 hash of the query → JSON file in `cache/osm/`).
A 3-second delay is respected between queries so as not to overload the API.
On failure (corporate proxy, API unavailable) it returns an empty list — the following
steps run in degraded mode.

---

### Step 4 — Computing the per-country parameters
**File: `auto.py`** (orchestration) + the `compute/` files

For each country, `auto.py` calls 5 computation functions:

#### 4a. Country area
**File: `compute/area.py`**

```
boundaries_gdf  ->  reprojection EPSG:6933 (equal-area)  ->  area_m2  ->  area_km2
```

Uses the **cylindrical equal-area projection** (EPSG:6933) so that lat/lon polygons are
converted to metres before the area calculation. Without this reprojection, degrees of
latitude and longitude do not correspond to the same real distance (1° of longitude at
the equator = 111 km, but at 45°N = 78 km).

Fallback if the reprojection fails: spherical approximation from the bounding box
(`lat_span × 111 km × lon_span × 111 km × cos(mean_lat)`).

**Why this matters:** a large country (> 500,000 km²) statistically has more resource
diversity and more network distance — it deserves at least one extra zone.

---

#### 4b. RE resource heterogeneity
**File: `compute/re_spread.py`**

Computes a **geographic proxy** for the variability of the RE capacity factor across the country.

```
lat_span = max_latitude - min_latitude  (in degrees)
lon_span = max_longitude - min_longitude
spread = lat_span × 0.022 + lon_span × 0.008 × 0.5
spread = min(spread, 0.50)  # capped at 50%
```

The idea: the north-south extent captures solar irradiation gradients (stronger in the
south), the east-west extent captures wind gradients (different regimes coast vs inland).
The coefficients (0.022 and 0.008) are calibrated empirically to stay consistent with CF
variability studies in Europe and the Middle East.

This is a **proxy that needs no API data** — no access key required, always available.
For more precision, ERA5 or MERRA-2 data would be needed.

**Threshold:** if spread > 0.25 (25%), an extra zone is justified (otherwise zone-averaged
RE gives a biased result).

---

#### 4c. Distance between load centers
**File: `compute/load_centers.py`**

Takes the **5 largest cities** in the country (by population in Natural Earth),
computes all pairwise distances with the haversine formula, returns the maximum distance.

```
If max_distance > 350 km  ->  distant_load_centers = True
```

**Why 350 km?** Beyond that distance, a 220 kV HV line carrying ~1000 MW suffers losses
of roughly 5-8%, and transit constraints can appear under heavy load. It is the empirical
limit used in ENTSO-E studies.

**Why this matters:** if Istanbul and Ankara are 350 km apart (exactly the limit in this
case), modelling Türkiye as a single uniform zone assumes a plant in Istanbul can supply
Ankara without constraint. That is not true at peak.

---

#### 4d. Hydro concentration
**File: `compute/hydro_concentration.py`**

Compares the geographic position of hydro plants with that of the load centers:
1. Computes the **weighted centroid** of the hydro plants (weighted by MW capacity)
2. Computes the **weighted centroid** of the cities (weighted by population)
3. Measures the haversine distance between the two centroids

```
If distance > 150 km  ->  hydro_concentration = True  (hydro far from load)
```

**Why this matters:** if all the hydro is in the mountains to the east and all the demand
is on the western coast, there is necessarily a critical transmission corridor between the
two. Not representing it explicitly in the model biases hydro dispatch and nodal prices.

---

#### 4e. Network congestion corridors
**File: `compute/network_bottlenecks.py`**

This is the most sophisticated calculation. It detects **bottlenecks** in the OSM network
using graph theory.

**Step 1 — Building the graph**
- Each **substation** becomes a node
- Each **HV line** becomes an edge between its two endpoints
- Line endpoints are "snapped" to the nearest substation within a 15 km radius
  (to connect lines that do not pass exactly through the substations in OSM)

**Step 2 — Edge betweenness centrality** (NetworkX algorithm)

For each pair of nodes (A, B) in the network, the shortest path is computed.
An edge's **betweenness centrality** = the fraction of shortest paths that pass through
that edge.

An edge with high betweenness is **critical**: a lot of flow "passes" through it, in the
topological sense. If it were saturated, many node pairs would be disconnected or forced
onto longer paths.

**Step 3 — Identifying the bottlenecks**
```
threshold = mean(betweenness) + 2 × stddev(betweenness)
bottleneck_edges = edges with betweenness > threshold
```

The `mean + 2σ` threshold is a standard statistical convention for identifying outliers in
a distribution — here, the lines that genuinely stand out from the network.

**Step 4 — Counting the corridors**
Adjacent bottleneck edges (sharing a node) are grouped into **connected components**.
Each component = one distinct congestion corridor.

```
TUR : 80 bottleneck edges -> 5 distinct corridors
ROU : 61 bottleneck edges -> 2 distinct corridors
BGR : 38 bottleneck edges -> 1 distinct corridor
```

**Why this matters:** each corridor identified is a place where the network can saturate,
which justifies a zone boundary. This is exactly the PyPSA-Eur logic for segmenting the
European networks.

---

### Step 5 — Assembling the CountryConfig
**File: `auto.py`** + **`schema.py`**

All computed parameters are assembled into a `CountryConfig` object:

```python
CountryConfig(
    name="TUR",
    area_km2=798647,
    n_bidding_zones=1,          # not automated, defaults to 1
    known_congestion_splits=5,  # computed by network_bottlenecks.py
    re_cf_spread=0.21,          # computed by re_spread.py
    distant_load_centers=True,  # computed by load_centers.py
    hydro_concentration=False,  # computed by hydro_concentration.py (GPPD missing)
    data_quality="good",        # inferred from OSM coverage
)
```

`data_quality` is inferred automatically:
- `good` if OSM has > 500 substations and > 500 lines for the country
- `medium` if OSM has 50-500 elements
- `limited` if OSM has < 50 elements or if the fetch failed

---

### Step 6 — Spatial recommendation
**File: `spatial/recommender.py`**

For each country, computes a **physical floor** by accumulating drivers:

| Driver | Contribution |
|--------|-------------|
| Official bidding zones | = n_bidding_zones |
| Congestion corridors | +n corridors |
| RE spread > 25% | +1 zone |
| Area > 500,000 km² | +1 zone |
| Load centers > 350 km | +1 zone |
| Hydro far from load | +1 zone |

Example for TUR: 1 (base) + 5 (congestion) + 0 (RE spread 21% < 25%) + 1 (large area) + 1 (load centers) = **8 zones**

This floor is then **capped by data quality**:
- `good` → max 6 zones (you cannot decompose further than you can calibrate)
- `medium` → max 4 zones
- `limited` → max 1 zone

The logic: even if the physical network justifies 8 zones, if all you have is aggregated
national data, the parameters of a sixth zone would be invented. Better 6 well-calibrated
zones than 8 of which 2 are fiction.

Compute **ceiling**:
```
ceiling = variable_budget / (N_repr_hours × N_years × N_scenarios)
        = 8 000 000 / (384h × 3yr × 1scenario)
        = 30 zones
```

The 8M `variable_budget` is an estimate of the number of LP/MIP variables a CPLEX solver
can handle in < 6h on 64 GB RAM, based on EPM benchmarks.

---

### Step 7 — Temporal recommendation
**File: `temporal/recommender.py`**

Computes the minimum number of **representative days** (not actual days of the year —
aggregated "daytypes" that represent the annual chronology).

**Floor** (cumulative rules):
- Baseline: 4 days (1 per season)
- RE >= 20% → min 8 days (capture weekly wind variability)
- RE >= 35% → min 12 days (capture low-RE/high-demand coincidences)
- RE >= 50% → min 16 days
- Storage `medium` → +2 days (multi-day charge/discharge cycles)
- Storage `high` → +4 days
- Strong seasonal hydro → min 8 days (dry/wet seasons separated)

**Extreme days** (always added on top):
- 2 days minimum: demand peak + min-RE event
- If RE >= 30%: 3 days (adds a wind drought)

These extreme days are distinct from the representative days: they do not represent the
typical chronology but the worst cases that size the backup capacity.

**Ceiling**:
```
max_days = variable_budget / (N_zones × 24h × N_years × N_scenarios)
         = 8 000 000 / (11 zones × 24h × 3yr × 1scenario)
         = 36 days
```

---

### Step 8 — Display and save
**File: `advise.py`**

Assembles the results into a formatted table (or JSON with `--output json`).
With `--save`, writes a JSON file to `output/`.

---

## Summary: who computes what

```
advise.py                 CLI, orchestrates everything, prints the result
auto.py                   Orchestrates fetching + computation in auto mode
schema.py                 Data structures (CountryConfig, AdvisorConfig, ...)

fetch/
  natural_earth.py        Downloads/loads boundaries + cities (Natural Earth)
  osm.py                  Overpass API query -> substations + HV lines
  gppd.py                 Downloads/loads GPPD -> power plants

compute/
  area.py                 area_km2 = EPSG:6933 reprojection -> geometric calculation
  re_spread.py            re_spread = geographic lat/lon proxy
  load_centers.py         distant_load_centers = max distance between cities > 350 km ?
  hydro_concentration.py  hydro_concentration = hydro centroid vs load centroid > 150 km ?
  network_bottlenecks.py  known_congestion_splits = edge betweenness on the OSM graph

spatial/
  recommender.py          floor + ceiling + candidates (number of zones)

temporal/
  recommender.py          floor + ceiling + candidates (number of representative days)
```

---

## What the code does NOT do (yet)

- It does **not generate the zones** — it says how many are needed. To generate them, see `pipelines/zone_pipeline.py`.
- It does **not select the representative days** — for that, see `representative_days/` and the tsam/Poncelet pipeline.
- The `n_bidding_zones` parameter is not automated (always 1 by default in auto mode) — fill it in manually in the YAML if the country has official zonal markets.
- `data_quality` is inferred from OSM density, not from the availability of actual EPM data (zonal hourly load, etc.).
