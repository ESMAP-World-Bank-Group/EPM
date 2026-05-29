# Resolution Advisor

Recommends **spatial** (number of zones) and **temporal** (number of representative days)
resolution for EPM models, based on country characteristics and open data.

Two modes:
- **Manual** — you supply parameters in a YAML config
- **Auto** — fetches OSM grid data, Natural Earth boundaries, and GPPD power plants automatically

---

## Quick start

```bash
conda activate gams_env   # use gams_env, not esmap_env (PROJ works there)
cd pre-analysis/resolution_advisor

# Auto mode (fetches open data — recommended)
python advise.py --countries TUR ROU BGR GEO ARM AZE --auto

# Manual mode (YAML config)
python advise.py --config config/blacksea.yaml

# Save output to JSON
python advise.py --countries TUR ROU BGR --auto --save
```

---

## Outputs

### Console (always)
A formatted table printed to the terminal:
- Per-country zone floor/cap/recommendation
- Data sources used (auto mode) or config values (manual mode)
- Temporal floor/ceiling/candidates
- Convergence test grid: which (N_zones, N_days) combinations to run

### JSON file (with `--save`)
Saved to:
```
pre-analysis/resolution_advisor/output/{model_name}_recommendation.json
```
For auto mode the model name is derived from the country list (e.g. `auto_TUR_ROU_BGR`).
For manual mode it is taken from the `region.name` field in the YAML.

Example JSON structure:
```json
{
  "model": "auto_TUR_ROU_BGR",
  "spatial": {
    "floor_total": 11,
    "ceiling_total": 30,
    "candidates": [11, 14, 17, 30],
    "per_country": { "TUR": 6, "ROU": 3, "BGR": 2 }
  },
  "temporal": {
    "floor_repr_days": 10,
    "floor_extreme_days": 3,
    "ceiling": 36,
    "candidates": [10, 14, 23, 36]
  }
}
```

---

## CLI options

| Flag | Description |
|------|-------------|
| `--countries TUR ROU ...` | ISO_A3 codes (auto mode) |
| `--auto` | Use open data instead of manual YAML |
| `--config <path>` | Path to YAML config (manual mode) |
| `--output table` | Human-readable report (default) |
| `--output json` | Machine-readable JSON printed to stdout |
| `--n_zones N` | Override spatial recommendation for temporal calculation |
| `--save` | Save JSON output to `output/` folder |

---

## How it works

### Spatial resolution

For each country, a **floor** (minimum zones needed physically) is computed from:

| Driver | Source (auto) | Source (manual) |
|--------|--------------|-----------------|
| Documented congestion corridors | OSM HV lines — edge betweenness centrality | `known_congestion_splits` in YAML |
| Country area > 500k km2 | Natural Earth boundaries | `area_km2` in YAML |
| Geographically separated load centers | Natural Earth populated places | `distant_load_centers` in YAML |
| RE capacity factor spread | Geographic proxy (lat/lon span) | `re_cf_spread` in YAML |
| Hydro concentration away from load | GPPD + Natural Earth cities | `hydro_concentration` in YAML |
| Bidding zone structure | Not automated (set in YAML) | `n_bidding_zones` in YAML |

The floor is then **capped by data quality**:
- `good` -> max 6 zones
- `medium` -> max 4 zones
- `limited` -> max 1 zone

The **ceiling** comes from the compute budget:
```
ceiling = variable_budget / (N_hours_repr x N_years x N_scenarios)
```

### Temporal resolution

Floor driven by:
- RE penetration target: >= 20% -> 8 days, >= 35% -> 12 days, >= 50% -> 16 days
- Storage relevance: `high` adds 4 days, `medium` adds 2 days
- Hydro seasonality: adds days if strong seasonal pattern
- Multi-period planning: adds days for more planning years
- Extreme days (peak demand, wind drought, min-solar): always added on top

### Convergence grid

The tool outputs a grid of `(N_zones x N_days)` combinations to test in EPM.
Run at 2-3 points and stop where **total system cost changes < 2%** between levels.

---

## Open data sources (auto mode)

| Source | Data | Cache location |
|--------|------|---------------|
| OpenStreetMap (Overpass API) | HV substations + transmission lines | `cache/osm/` |
| Natural Earth | Country boundaries + populated places | `cache/natural_earth/` |
| GPPD (WRI) | Power plant locations and capacity | `cache/gppd/` |

Results are cached locally — OSM is only re-fetched if you delete `cache/osm/`.

**GPPD note**: WRI moved the download URL. If auto-download fails, download manually:
1. Go to: https://datasets.wri.org/dataset/globalpowerplantdatabase
2. Download `global_power_plant_database.csv`
3. Place at: `cache/gppd/global_power_plant_database.csv`

---

## Manual config (YAML)

Edit `config/blacksea.yaml`. Key fields per country:

```yaml
countries:
  TUR:
    known_congestion_splits: 5   # documented internal bottlenecks (from OSM in auto mode)
    re_cf_spread: 0.21           # max-min CF across country (0-1)
    distant_load_centers: true   # cities > 350km apart?
    hydro_concentration: false   # hydro far from load centers?
    data_quality: good           # good / medium / limited -> caps max zones

model_use:
  re_penetration_target: 0.30   # drives temporal floor
  storage_relevance: medium     # low / medium / high

constraints:
  max_runtime_hours: 6
  ram_gb: 64
  available_solver: cplex
```

---

## File structure

```
resolution_advisor/
├── advise.py                  # CLI entry point
├── schema.py                  # Config dataclasses
├── auto.py                    # Orchestrates open data fetch + compute
├── config/
│   └── blacksea.yaml          # Manual config for Black Sea model
├── fetch/
│   ├── osm.py                 # OSM Overpass API (substations, HV lines)
│   ├── gppd.py                # Global Power Plant Database
│   └── natural_earth.py       # Country boundaries + populated places
├── compute/
│   ├── area.py                # Country area from boundaries
│   ├── re_spread.py           # RE CF spread proxy from lat/lon
│   ├── load_centers.py        # Distance between major cities
│   ├── hydro_concentration.py # Hydro centroid vs load centroid
│   └── network_bottlenecks.py # Edge betweenness on OSM HV graph
├── spatial/
│   └── recommender.py         # Floor/ceiling/candidates logic
├── temporal/
│   └── recommender.py         # Temporal floor/ceiling/candidates
├── cache/                     # Auto-created, cached API responses
└── output/                    # Auto-created, saved JSON recommendations
```
