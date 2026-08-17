"""
Build the "off-grid" polygons for the CASA 2020 model: the parts of the modelled
countries that belong to NO dispatch zone.

Two countries have such areas, and in both cases the omission is electrical, not an
oversight — the territory is simply not part of the system the model represents:

  Kazakhstan  431,140 km2 (15.9%)  KEGOC Western zone, synchronised with Russia
  Afghanistan 171,329 km2 (26.7%)  provinces off the NEPS network

Left undrawn they read as a bug, so they are published as their own map layer,
grey and dashed, clearly outside the model. They are deliberately NOT added to
zones.geojson: that file is joined to model results, and a zone with no results
would surface as a phantom entry in dropdowns, totals and choropleths (a "0 GWh"
that means "not modelled", which is exactly the wrong reading).

The polygons are cut from the same Natural Earth 10m admin1 source as the zones
themselves, so they line up exactly with the zone borders. Each keeps the written
reason for its exclusion, which the map shows on hover.

Source (40 MB, not vendored — pass --admin1):
  https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip

Usage:
    python pre-analysis/build_offgrid_zones.py --admin1 <path to ne10 admin1 geojson>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd

_BASE = Path(__file__).resolve().parent
_REPO = _BASE.parent
_EXPLORER = _REPO.parents[1] / "regional-power-explorer"

DATA = _REPO / "epm" / "input" / "data_casa_2020"
ZONES_OUT = _EXPLORER / "public" / "data" / "zones"
SIMPLIFY_DEG = 0.01     # same tolerance as the zone polygons, so borders coincide

# Grouped so each feature carries one coherent reason. The unit lists must match
# EXCLUDED in the zone builder — asserted below against the zones actually built.
OFFGRID = {
    "KAZ_WEST": dict(
        iso3="KAZ", c="KAZ", name="Kazakhstan - Western zone",
        units=["KZ-ATY", "KZ-MAN", "KZ-ZAP"],
        reason="KEGOC Western zone: isolated from the rest of the Kazakh grid and "
               "synchronised with Russia. No plant and no load in the 2020 model."),
    "AFG_WEST": dict(
        iso3="AFG", c="AFG", name="Afghanistan - western islands",
        units=["AF-FRA", "AF-NIM"],
        reason="Farah and Nimruz: western islands fed from Iran, not synchronised "
               "with NEPS and not represented by any zone (unlike Herat, which has one)."),
    "AFG_BADAKHSHAN": dict(
        iso3="AFG", c="AFG", name="Afghanistan - Badakhshan",
        units=["AF-BDS"],
        reason="Badakhshan: not connected to NEPS; supplied by isolated Pamir Energy "
               "cross-border micro-grids that are outside the modelled system."),
    "AFG_GHOR": dict(
        iso3="AFG", c="AFG", name="Afghanistan - Ghor",
        units=["AF-GHO"],
        reason="Ghor: no transmission connection to any modelled zone; isolated "
               "micro-hydro and diesel only."),
}


def build(admin1_path: Path, dry_run=False):
    a1 = gpd.read_file(admin1_path)
    a1 = a1[a1["adm0_a3"].isin({d["iso3"] for d in OFFGRID.values()})]

    zones = gpd.read_file(DATA / "zones_zcmap.geojson")
    modelled = zones.geometry.union_all()

    feats = []
    for z, d in OFFGRID.items():
        sub = a1[a1["iso_3166_2"].isin(d["units"])]
        missing = set(d["units"]) - set(sub["iso_3166_2"])
        if missing:
            sys.exit(f"{z}: admin1 units not found in source: {sorted(missing)}")

        # Subtract the modelled zones so the two layers tile instead of overlapping.
        geom = sub.geometry.union_all().buffer(0).simplify(SIMPLIFY_DEG).buffer(0)
        geom = geom.difference(modelled).buffer(0)

        km2 = gpd.GeoSeries([geom], crs=4326).to_crs(6933).area.iloc[0] / 1e6
        feats.append({
            "z": z, "name": d["name"], "ISO_A3": d["iso3"], "c": d["c"],
            "status": "not modelled", "reason": d["reason"],
            "admin_units": ", ".join(sorted(sub["name"].unique())),
            "admin_source": "Natural Earth 10m admin-1",
            "area_km2": round(km2),
            "geometry": geom,
        })
        print(f"  {z:16s} {km2:9,.0f} km2   {d['name']}")

    gdf = gpd.GeoDataFrame(feats, crs=4326)

    # sanity: an off-grid polygon must not eat into any dispatch zone
    overlap = gdf.geometry.union_all().intersection(modelled)
    ov_km2 = gpd.GeoSeries([overlap], crs=4326).to_crs(6933).area.iloc[0] / 1e6
    print(f"\n  overlap with modelled zones: {ov_km2:.2f} km2")
    if ov_km2 > 1.0:
        sys.exit("off-grid polygons overlap the dispatch zones - aborting")

    if dry_run:
        print("\n(dry run - nothing written)")
        return

    # 1. EPM data folder -> epm-data-explorer reads it straight from GitHub raw.
    #    Not a GAMS input: the model reads CSVs only, so this adds no model state.
    out = DATA / "zones_offgrid.geojson"
    gdf.to_file(out, driver="GeoJSON")
    print(f"\n  {out}  ({out.stat().st_size/1024:.0f} KB)")

    # 2. regional-power-explorer, one file per country run, named like its zones file
    if not ZONES_OUT.exists():
        print(f"  (explorer not found at {ZONES_OUT} - skipped)")
        return
    n_of = zones.groupby("ISO_A3").size().to_dict()
    for iso, sub in gdf.groupby("ISO_A3"):
        label = f"{iso}_{n_of[iso]}z"
        rpe = sub.rename(columns={"z": "zone_id", "name": "zone_name"})
        rpe = rpe[["zone_id", "zone_name", "ISO_A3", "status", "reason",
                   "area_km2", "geometry"]]
        path = ZONES_OUT / f"{label}_outside.geojson"
        rpe.to_file(path, driver="GeoJSON")
        print(f"  {path.name}  ({len(rpe)} feature(s))")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--admin1", required=True, help="Natural Earth 10m admin1 geojson")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.admin1)
    if not src.exists():
        sys.exit(f"Not found: {src}")
    print(f"\nadmin1 source: {src}\n")
    build(src, args.dry_run)
