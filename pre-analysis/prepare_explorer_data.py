"""
Regenerate Explorer cache data for a region:
  1. Admin-1 boundaries (Natural Earth 10m provinces/states)
  2. Country boundaries dissolved from admin-1 (10m resolution, for precise highlight)
  3. OSM substations (fresh fetch, replaces sparse cache)

Usage:
    gams_env python pre-analysis/prepare_explorer_data.py [--region blacksea]
"""
from __future__ import annotations
import json, os, sys, argparse
from pathlib import Path

# Fix pyproj "no database context" error when run via conda run on Windows
def _fix_proj_data():
    for p in sys.path:
        d = Path(p) / "pyproj" / "proj_dir" / "share" / "proj"
        if d.exists():
            os.environ.setdefault("PROJ_DATA", str(d))
            os.environ.setdefault("PROJ_LIB", str(d))
            return
_fix_proj_data()

_BASE     = Path(__file__).resolve().parent
_REPO     = _BASE.parent
_EXPLORER = _REPO.parents[1] / "epm-explorer-v2"
_CACHE    = _EXPLORER / "public" / "data" / "cache"

sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_BASE / "resolution_advisor"))

REGIONS: dict[str, list[str]] = {
    "blacksea":   ["TUR", "ARM", "GEO", "AZE", "ROU", "BGR"],
    "balkans":    ["ALB", "BIH", "MKD", "MNE", "SRB"],
    "centralasia":["KAZ", "KGZ", "TJK", "TKM", "UZB"],
}

NE_ADMIN1_URLS = [
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip",
    "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_1_states_provinces.zip",
]
NE_ADMIN1_CACHE = _BASE / "resolution_advisor" / "cache" / "natural_earth"


# ── 1. Admin-1 boundaries ────────────────────────────────────────────────────

def prepare_admin1(region_id: str, countries: list[str]):
    import geopandas as gpd

    shp = NE_ADMIN1_CACHE / "ne_10m_admin_1_states_provinces.shp"
    if not shp.exists():
        shp = _download_admin1()

    gdf = gpd.read_file(shp)
    iso_col = next((c for c in ["iso_a3", "ISO_A3", "adm0_a3"] if c in gdf.columns), None)
    name_col = next((c for c in ["name", "NAME", "gn_name"] if c in gdf.columns), None)
    type_col = next((c for c in ["type_en", "featurecla"] if c in gdf.columns), None)

    if iso_col:
        gdf = gdf[gdf[iso_col].isin(countries)].copy()
        gdf = gdf.rename(columns={iso_col: "ISO_A3"})

    keep = {"ISO_A3"}
    if name_col: keep.add(name_col); gdf = gdf.rename(columns={name_col: "name"})
    if type_col: keep.add(type_col); gdf = gdf.rename(columns={type_col: "type_en"})
    keep = [c for c in ["ISO_A3", "name", "type_en"] if c in gdf.columns]
    gdf = gdf[keep + ["geometry"]].copy()
    # Same tolerance as countries_10m.geojson so outer edges align with land layer
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01, preserve_topology=True)

    out = _CACHE / f"region_admin1_{region_id}.geojson"
    gdf.to_file(out, driver="GeoJSON")
    print(f"  region_admin1_{region_id}.geojson  ({len(gdf)} features, {gdf['ISO_A3'].value_counts().to_dict()})")


def _download_admin1() -> Path:
    import io, zipfile, requests as req, warnings
    NE_ADMIN1_CACHE.mkdir(parents=True, exist_ok=True)
    for url in NE_ADMIN1_URLS:
        print(f"  Downloading admin-1 from {url.split('/')[-1]}...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = req.get(url, timeout=120, verify=False)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(NE_ADMIN1_CACHE)
            shp = next(NE_ADMIN1_CACHE.rglob("ne_10m_admin_1_states_provinces.shp"), None)
            if shp:
                print(f"  Saved: {shp}")
                return shp
        except Exception as e:
            print(f"  Failed: {e}")
    raise FileNotFoundError("Could not download NE admin-1 data")


# ── 2. World 10m countries (one-time, region-independent) ───────────────────

def prepare_world_countries_10m():
    import geopandas as gpd

    shp = NE_ADMIN1_CACHE / "ne_10m_admin_1_states_provinces.shp"
    if not shp.exists():
        shp = _download_admin1()

    gdf = gpd.read_file(shp)
    iso_raw = next((c for c in ["iso_a3", "ISO_A3", "adm0_a3"] if c in gdf.columns), None)
    if not iso_raw:
        print("  [warn] no ISO column found — skipping world 10m")
        return
    gdf = gdf.rename(columns={iso_raw: "ISO_A3"})
    dissolved = gdf[["ISO_A3", "geometry"]].dissolve(by="ISO_A3").reset_index()
    # Simplify to ~NE 10m precision to keep file web-friendly
    dissolved["geometry"] = dissolved["geometry"].simplify(tolerance=0.01, preserve_topology=True)

    out = _CACHE.parent / "countries_10m.geojson"
    dissolved.to_file(out, driver="GeoJSON")
    sz_mb = out.stat().st_size / 1e6
    print(f"  countries_10m.geojson  ({len(dissolved)} countries, {sz_mb:.1f} MB)")


# ── 3. OSM substations ───────────────────────────────────────────────────────

def prepare_substations(region_id: str, countries: list[str]):
    from fetch.osm import fetch_substations
    from fetch.natural_earth import load_boundaries

    boundaries = load_boundaries(countries)
    all_subs = []

    for iso in countries:
        subset = boundaries[boundaries["ISO_A3"] == iso]
        if subset.empty:
            print(f"  [{iso}] no boundary — skipping")
            continue
        b = subset.geometry.total_bounds  # minx miny maxx maxy
        bbox = (b[1], b[0], b[3], b[2])  # s w n e
        try:
            subs = fetch_substations(bbox)
            for s in subs:
                s["iso"] = iso
            print(f"  {iso}: {len(subs)} substations")
            all_subs.extend(subs)
        except Exception as e:
            print(f"  [{iso}] OSM failed: {e}")

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
            "properties": {"name": s.get("name", ""), "v": int(s.get("voltage_kv", 0) * 1000), "iso": s.get("iso", "")},
        }
        for s in all_subs
    ]
    out = _CACHE / f"region_substations_{region_id}.geojson"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print(f"  -> region_substations_{region_id}.geojson  ({len(features)} total)")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="blacksea", choices=list(REGIONS))
    parser.add_argument("--skip-admin1",      action="store_true")
    parser.add_argument("--skip-world",       action="store_true")
    parser.add_argument("--skip-substations", action="store_true")
    args = parser.parse_args()

    countries = REGIONS[args.region]
    print(f"\nRegion: {args.region}  ({', '.join(countries)})")
    print(f"Explorer cache: {_CACHE}\n")

    if not args.skip_world:
        print("=== World 10m countries ===")
        prepare_world_countries_10m()

    if not args.skip_admin1:
        print("=== Admin-1 boundaries ===")
        prepare_admin1(args.region, countries)

    if not args.skip_substations:
        print("\n=== OSM Substations ===")
        prepare_substations(args.region, countries)

    print("\nDone.")
