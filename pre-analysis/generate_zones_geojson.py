"""
Generate zones.geojson for an EPM region branch.

The output is a GeoJSON FeatureCollection with one feature per EPM zone:
  {z, ISO_A3, c}  — z matches zcmap.csv / pGenDataInput.csv zone IDs.

For countries with multiple zones (e.g. Somalia → Somaliland/Mogadishu/SomaliaROC)
each zone gets the same country polygon geometry; donut markers on the map use
centroid positions derived from linestring endpoints, so they appear at distinct spots.

Usage (from repo root, gams_env):
    python pre-analysis/generate_zones_geojson.py --region eapp
    python pre-analysis/generate_zones_geojson.py --region sapp
    python pre-analysis/generate_zones_geojson.py --region eapp --push
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

_BASE = Path(__file__).resolve().parent
_REPO = _BASE.parent
RAW = "https://raw.githubusercontent.com/ESMAP-World-Bank-Group/EPM"
API = "https://api.github.com/repos/ESMAP-World-Bank-Group/EPM"

REGIONS: dict[str, dict] = {
    "eapp": {"branch": "eapp_2026",    "data_folder": "data_eapp"},
    "sapp": {"branch": "sapp_new2025", "data_folder": "data_sapp"},
}

SIMPLIFY_TOL = 0.02   # degrees — keeps file web-friendly (~50-200 kB per region)


def fetch_csv(branch: str, data_folder: str, relpath: str) -> pd.DataFrame:
    url = f"{RAW}/{branch}/epm/input/{data_folder}/{relpath}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(r.text))


def fetch_geojson(branch: str, data_folder: str, filename: str) -> dict:
    url = f"{RAW}/{branch}/epm/input/{data_folder}/{filename}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def build_zone_iso_map(linestring_gj: dict) -> tuple[dict[str, str], dict[str, str]]:
    """
    Returns:
      zone_iso : zone_name → ISO_A3  (from both endpoints)
      c_iso    : country_display_name → ISO_A3  (from z/c pairs in linestring)
    """
    zone_iso: dict[str, str] = {}
    c_iso: dict[str, str] = {}
    for f in linestring_gj["features"]:
        p = f["properties"]
        z, iso, c = p.get("z"), p.get("ISO_A3", ""), p.get("c", "")
        z2, iso2 = p.get("z_other"), p.get("ISO_A3_other", "")
        if z and z not in zone_iso:
            zone_iso[z] = iso
        if z2 and z2 not in zone_iso:
            zone_iso[z2] = iso2 or ""
        # Build c→ISO from the z side (always has a name)
        if c and iso and iso != "-99":
            c_iso[c] = iso
    return zone_iso, c_iso


# Country display-name → ISO_A3 for names that may not appear in any linestring
_FALLBACK_ISO: dict[str, str] = {
    "South_Africa": "ZAF", "South Africa": "ZAF",
    "Eswatini": "SWZ", "Swaziland": "SWZ",
    "Tanzania": "TZA",
    "DRC": "COD", "Congo": "COD", "DR Congo": "COD",
    "Somaliland": "SOM",
    "Madagascar": "MDG",
    "Lesotho": "LSO",
    "Malawi": "MWI",
}


def load_base_polygons(iso_set: set[str]) -> gpd.GeoDataFrame:
    """Load country polygons from the repo's resources file, filtered to iso_set."""
    src = _REPO / "epm" / "resources" / "postprocess" / "zones.geojson"
    gdf = gpd.read_file(src)
    # Normalise the ISO column (different files use different names)
    for col in ["ISO_A3", "iso_a3", "ADM0_A3"]:
        if col in gdf.columns:
            gdf = gdf.rename(columns={col: "ISO_A3"})
            break
    gdf = gdf[gdf["ISO_A3"].isin(iso_set)].copy()
    # Dissolve multi-row countries
    gdf = gdf.dissolve(by="ISO_A3").reset_index()
    gdf["geometry"] = gdf["geometry"].simplify(SIMPLIFY_TOL, preserve_topology=True)
    return gdf[["ISO_A3", "geometry"]]


def generate(region: str) -> dict:
    cfg = REGIONS[region]
    branch, data_folder = cfg["branch"], cfg["data_folder"]
    print(f"Region: {region}  branch={branch}  folder={data_folder}")

    # 1. zcmap: z → c  (EPM zone → country display name)
    zcmap = fetch_csv(branch, data_folder, "zcmap.csv")
    zone_country: dict[str, str] = dict(zip(zcmap["z"], zcmap["c"]))
    print(f"  {len(zone_country)} zones in zcmap")

    # 2. linestring: zone → ISO_A3
    ls_gj = fetch_geojson(branch, data_folder, "linestring_countries.geojson")
    zone_iso, c_iso = build_zone_iso_map(ls_gj)
    print(f"  {len(zone_iso)} zones with ISO from linestring")

    # 3. Resolve ISO for every zone (same fallback chain used later)
    def resolve_iso(z: str, c: str) -> str:
        iso = zone_iso.get(z, "")
        if not iso or iso == "-99":
            iso = next((zone_iso.get(zz) for zz, cc in zone_country.items()
                        if cc == c and zone_iso.get(zz) not in ("", "-99", None)), None) or ""
        if not iso or iso == "-99":
            iso = c_iso.get(c, "") or c_iso.get(z, "")
        if not iso or iso == "-99":
            iso = _FALLBACK_ISO.get(c, "") or _FALLBACK_ISO.get(z, "")
        return iso

    needed_isos = {resolve_iso(z, c) for z, c in zone_country.items()}
    needed_isos.discard("")
    print(f"  Loading polygons for {len(needed_isos)} ISOs: {sorted(needed_isos)}")
    poly_gdf = load_base_polygons(needed_isos)

    iso_geom: dict[str, object] = dict(zip(poly_gdf["ISO_A3"], poly_gdf["geometry"]))
    print(f"  Got {len(iso_geom)} country polygons from resources")

    # 4. Build one feature per zone
    features = []
    missing = []
    for z, c in zone_country.items():
        iso = resolve_iso(z, c)

        geom = iso_geom.get(iso)
        if geom is None:
            missing.append(z)
            continue

        features.append({
            "type": "Feature",
            "properties": {"z": z, "ISO_A3": iso, "c": c},
            "geometry": json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"],
        })

    if missing:
        print(f"  [warn] No polygon for zones: {missing}")
    print(f"  Built {len(features)} zone features")

    return {"type": "FeatureCollection", "features": features}


def push_to_github(region: str, gj: dict, token: str):
    cfg = REGIONS[region]
    branch = cfg["branch"]
    data_folder = cfg["data_folder"]
    path = f"epm/input/{data_folder}/zones.geojson"

    content = json.dumps(gj, separators=(",", ":"))
    b64 = base64.b64encode(content.encode()).decode()

    # Get current SHA if file exists (needed for updates)
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    check = requests.get(f"{API}/contents/{path}?ref={branch}", headers=headers)
    sha = check.json().get("sha") if check.ok else None

    payload = {
        "message": f"feat: add zones.geojson for EPM data explorer ({region})",
        "content": b64,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(f"{API}/contents/{path}", headers=headers, json=payload)
    r.raise_for_status()
    print(f"  Pushed to {branch}/{path}  (status {r.status_code})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=list(REGIONS), required=True)
    parser.add_argument("--push", action="store_true", help="Push result to GitHub")
    parser.add_argument("--out", help="Local output path (default: pre-analysis/output/<region>_zones.geojson)")
    args = parser.parse_args()

    gj = generate(args.region)

    out_path = Path(args.out) if args.out else _BASE / "output" / f"{args.region}_zones.geojson"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gj, f, separators=(",", ":"))
    sz = out_path.stat().st_size / 1024
    print(f"  Saved: {out_path}  ({sz:.0f} kB)")

    if args.push:
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            print("ERROR: set GITHUB_TOKEN env var")
            sys.exit(1)
        push_to_github(args.region, gj, token)

    print("Done.")
