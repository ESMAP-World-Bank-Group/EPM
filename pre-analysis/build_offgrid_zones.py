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

The polygons are cut from the same administrative units as the zones themselves
— the World Bank boundary artifact built by `python -m epm.geodata.wb_boundaries`
— so they line up exactly with the zone borders and with the basemap under them.
Each keeps the written reason for its exclusion, which the map shows on hover.

Nothing here is simplified. The units in the artifact are already fitted to the
country outlines the map is drawn from, and simplifying them again would move
their edges off those outlines, which is the one thing this layer must not do.

Usage:
    python pre-analysis/build_offgrid_zones.py                  # report
    python pre-analysis/build_offgrid_zones.py --apply
    python pre-analysis/build_offgrid_zones.py --folder data_casa --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd

_BASE = Path(__file__).resolve().parent
_REPO = _BASE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from epm.geodata.wb_boundaries import resolve_cache  # noqa: E402
from tools.rebuild_reference_zones import find_artifact  # noqa: E402

_EXPLORER = _REPO.parents[1] / "regional-power-explorer"

INPUT = _REPO / "epm" / "input"
DEFAULT_FOLDERS = ("data_casa", "data_casa_2020")
ZONES_OUT = _EXPLORER / "public" / "data" / "zones"

# Grouped so each feature carries one coherent reason. The unit lists must match
# The units are named by the HASC code the World Bank artifact carries, which is
# what the zone recipes key on as well, so the two layers cannot drift apart.
# EXCLUDED in the zone builder — asserted below against the zones actually built.
OFFGRID = {
    "KAZ_WEST": dict(
        iso3="KAZ", c="KAZ", name="Kazakhstan - Western zone",
        units=["KZ.AR", "KZ.MG", "KZ.WK"],
        reason="KEGOC Western zone: isolated from the rest of the Kazakh grid and "
               "synchronised with Russia. No plant and no load in the 2020 model."),
    "AFG_WEST": dict(
        iso3="AFG", c="AFG", name="Afghanistan - western islands",
        units=["AF.FH", "AF.NM"],
        reason="Farah and Nimruz: western islands fed from Iran, not synchronised "
               "with NEPS and not represented by any zone (unlike Herat, which has one)."),
    "AFG_BADAKHSHAN": dict(
        iso3="AFG", c="AFG", name="Afghanistan - Badakhshan",
        units=["AF.BD"],
        reason="Badakhshan: not connected to NEPS; supplied by isolated Pamir Energy "
               "cross-border micro-grids that are outside the modelled system."),
    "AFG_GHOR": dict(
        iso3="AFG", c="AFG", name="Afghanistan - Ghor",
        units=["AF.GR"],
        reason="Ghor: no transmission connection to any modelled zone; isolated "
               "micro-hydro and diesel only."),
}


def build(artifact, data_dir, apply=False, zones_name="zones_zcmap.geojson"):
    """Cut the off-grid polygons for one model folder and, on --apply, write them."""
    units = gpd.read_file(artifact / "adm1.geojson")
    units = units[units.ISO_A3.isin({d["iso3"] for d in OFFGRID.values()})]
    by_code = units.set_index("HASC_1")

    zones = gpd.read_file(data_dir / zones_name)
    modelled = zones.geometry.union_all()

    print(f"{data_dir.name}: {len(zones)} modelled zones")
    feats = []
    for z, d in OFFGRID.items():
        missing = [c for c in d["units"] if c not in by_code.index]
        if missing:
            sys.exit(f"{z}: units not in the artifact: {missing}")
        sub = by_code.loc[d["units"]]

        # Subtract the modelled zones so the two layers tile instead of overlapping.
        # The units need no repair of their own: the artifact is built valid, and
        # anything done to them here would take them off the country outline.
        geom = sub.geometry.union_all().difference(modelled)

        km2 = gpd.GeoSeries([geom], crs=4326).to_crs(6933).area.iloc[0] / 1e6
        feats.append({
            "z": z, "name": d["name"], "ISO_A3": d["iso3"], "c": d["c"],
            "status": "not modelled", "reason": d["reason"],
            "admin_units": ", ".join(sorted(sub["NAM_1"].unique())),
            "admin_source": f"World Bank GAD, artifact {artifact.name}",
            "area_km2": round(km2),
            "geometry": geom,
        })
        print(f"  {z:16s} {km2:9,.0f} km2   {d['name']}")

    gdf = gpd.GeoDataFrame(feats, crs=4326)

    # sanity: an off-grid polygon must not eat into any dispatch zone
    overlap = gdf.geometry.union_all().intersection(modelled)
    ov_km2 = gpd.GeoSeries([overlap], crs=4326).to_crs(6933).area.iloc[0] / 1e6
    print(f"  overlap with modelled zones: {ov_km2:.2f} km2")
    if ov_km2 > 1.0:
        sys.exit("off-grid polygons overlap the dispatch zones - aborting")

    if not apply:
        print("  (report only; pass --apply to write)")
        return gdf

    # 1. EPM data folder -> epm-data-explorer reads it straight from GitHub raw.
    #    Not a GAMS input: the model reads CSVs only, so this adds no model state.
    out = data_dir / "zones_offgrid.geojson"
    gdf.to_file(out, driver="GeoJSON")
    print(f"  wrote {out.relative_to(_REPO)}  ({out.stat().st_size/1024:.0f} KB)")

    # 2. regional-power-explorer, one file per country run, named like its zones file
    if not ZONES_OUT.exists():
        print(f"  (explorer not found at {ZONES_OUT} - skipped)")
        return gdf
    n_of = zones.groupby("ISO_A3").size().to_dict()
    for iso, sub in gdf.groupby("ISO_A3"):
        label = f"{iso}_{n_of[iso]}z"
        rpe = sub.rename(columns={"z": "zone_id", "name": "zone_name"})
        rpe = rpe[["zone_id", "zone_name", "ISO_A3", "status", "reason",
                   "area_km2", "geometry"]]
        path = ZONES_OUT / f"{label}_outside.geojson"
        rpe.to_file(path, driver="GeoJSON")
        print(f"  {path.name}  ({len(rpe)} feature(s))")
    return gdf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", help="boundary artifact (default: the newest built)")
    ap.add_argument("--cache", help="where artifacts live (see wb_boundaries)")
    ap.add_argument("--folder", nargs="*", default=list(DEFAULT_FOLDERS),
                    metavar="NAME", help="model folders under epm/input/")
    ap.add_argument("--apply", action="store_true",
                    help="write the layer; without it nothing is written")
    args = ap.parse_args()

    artifact = find_artifact(resolve_cache(args.cache), args.artifact)
    print(f"artifact: {artifact}")
    for name in args.folder:
        data_dir = INPUT / name
        if not (data_dir / "zones_offgrid.geojson").exists():
            print(f"{name}: no off-grid layer to rebuild - skipped")
            continue
        build(artifact, data_dir, apply=args.apply)


if __name__ == "__main__":
    main()
