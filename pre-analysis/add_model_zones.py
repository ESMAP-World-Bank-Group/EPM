"""
Publish an EPM model's own zoning to the Regional Power Explorer.

The Explorer's zoning layer normally shows *candidate* clusterings produced by
run_zoning_study.py. When a study already has a zoning defined in its EPM input
folder, that zoning should be shown instead of a clustering — this script is the
generic version of studies/blacksea_2026/add_existing_zones.py, which does the same
thing for the 9 Turkiye dispatch zones but is hard-wired to TUR / data_blacksea.

Reads from epm/input/<data_folder>/:
  zones_zcmap.geojson        zone polygons (properties: z, ISO_A3, c)
  zcmap.csv                  zone -> country
  trade/pTransferLimit.csv   internal transfer limits, used for corridor MW

Writes, per country, the same layout a pipeline run produces:
  pre-analysis/output_workflow/zoning_study/<ISO>_<n>z/epm_export/spatial/
      zones.geojson, zcmap.csv, sTopology.csv, pTransferLimit_estimated.csv

and then exports to the Explorer (same outputs as export_zones_to_explorer.py,
plus corridors, which here carry the model's own MW instead of OSM estimates):
  regional-power-explorer/public/data/zones/<ISO>_<n>z_{zones,inner,corridors}.geojson
  regional-power-explorer/public/data/zones/<ISO>_<n>z_topo.json
  regional-power-explorer/public/data/zones/index.json   (merged, other ISOs kept)
  regional-power-explorer/public/data/zones/sources.json (merged; tells the Explorer
      these zones are the model's own, not one of its OSM clusterings)

Usage (from the EPM repo root):
    python pre-analysis/add_model_zones.py --data-folder data_casa_2020
    python pre-analysis/add_model_zones.py --data-folder data_casa_2020 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union

_BASE = Path(__file__).resolve().parent          # pre-analysis/
_REPO = _BASE.parent                             # EPM root
_EXPLORER = _REPO.parents[1] / "regional-power-explorer"

STUDY_ROOT = _BASE / "output_workflow" / "zoning_study"
ZONES_OUT = _EXPLORER / "public" / "data" / "zones"

SOURCE_LABEL = "existing model zoning"
REF_YEAR = "2025"     # snapshot the corridor ratings are reported for
LIVE_MW = 1.0         # below this a limit means "line absent", not "line at 0 MW"


# ── read the model's zoning ───────────────────────────────────────────────────

def read_model_zoning(data_dir: Path):
    """Return (zones GeoDataFrame, zcmap DataFrame, internal corridor MW per pair)."""
    zones = gpd.read_file(data_dir / "zones_zcmap.geojson")
    if "z" not in zones.columns or "ISO_A3" not in zones.columns:
        sys.exit(f"{data_dir/'zones_zcmap.geojson'}: expected 'z' and 'ISO_A3' properties")

    zcmap = pd.read_csv(data_dir / "zcmap.csv")
    zone_col = "zone" if "zone" in zcmap.columns else "z"

    # ISO_A3 is the real country code (TJK, TKM); zcmap 'c' is the model's label
    # (TAJ, TUR). Group by ISO_A3 so the Explorer, which keys on ISO_A3, matches.
    iso_of = dict(zip(zones["z"], zones["ISO_A3"]))

    links = read_links(data_dir / "trade" / "pTransferLimit.csv", iso_of)
    return zones, zcmap.rename(columns={zone_col: "z"}), links


def read_links(path: Path, iso_of: dict[str, str]) -> list[dict]:
    """Internal corridors, one per unordered zone pair, from pTransferLimit.

    Two properties of the CASA data shape this:
      - limits are directional and often asymmetric (KGZ_S->KGZ_N is 1900 MW,
        KGZ_N->KGZ_S is 1000), so both directions are reported;
      - a line not yet built is written as 1e-9 rather than omitted, so a pair that
        is ~0 in REF_YEAR but positive later is a planned line, not a 0 MW one.
    """
    tl = pd.read_csv(path)
    from_col = "From" if "From" in tl.columns else "z"
    to_col = "To" if "To" in tl.columns else "z2"
    years = sorted(c for c in tl.columns if c.isdigit())
    ref = REF_YEAR if REF_YEAR in years else years[-1]

    for y in years:
        tl[y] = pd.to_numeric(tl[y], errors="coerce")
    directed = tl.groupby([from_col, to_col])[years].mean()   # mean over seasons

    pairs = sorted({tuple(sorted(k)) for k in directed.index
                    if iso_of.get(k[0]) and iso_of.get(k[0]) == iso_of.get(k[1])})

    links = []
    for a, b in pairs:
        def mw(y):
            v = [directed.loc[(x, y_), y] for x, y_ in ((a, b), (b, a))
                 if (x, y_) in directed.index]
            return max(v) if v else 0.0

        commissioned = next((y for y in years if mw(y) >= LIVE_MW), None)
        if commissioned is None:
            continue                      # pair declared but never energised
        y = ref if mw(ref) >= LIVE_MW else commissioned
        ab = directed.loc[(a, b), y] if (a, b) in directed.index else None
        ba = directed.loc[(b, a), y] if (b, a) in directed.index else None
        both = [v for v in (ab, ba) if v is not None]
        label = (f"{round(max(both))} MW" if len({round(v) for v in both}) == 1
                 else f"{round(ab)} / {round(ba)} MW")
        links.append({"a": a, "b": b, "mw": round(max(both)),
                      "ab": None if ab is None else round(ab),
                      "ba": None if ba is None else round(ba),
                      "year": y,
                      "status": "existing" if y == ref else "planned",
                      "label": label if y == ref else f"{label} (from {y})"})
    return links


# ── write one study-format run per country ────────────────────────────────────

def write_study_runs(zones, zcmap, all_links, dry_run=False):
    runs = []
    for iso, sub in zones.groupby("ISO_A3"):
        n = len(sub)
        label = f"{iso}_{n}z"
        spatial = STUDY_ROOT / label / "epm_export" / "spatial"

        gj = sub.copy()
        gj["zone_id"] = gj["z"]
        gj["zone_name"] = gj["z"]
        gj = gj[["zone_id", "zone_name", "ISO_A3", "geometry"]]

        zs = set(sub["z"])
        links = [k for k in all_links if k["a"] in zs]
        planned = sum(k["status"] == "planned" for k in links)
        print(f"  {label}: {n} zones ({', '.join(sub['z'])}), {len(links)} internal links"
              + (f" ({planned} planned)" if planned else ""))

        if not dry_run:
            spatial.mkdir(parents=True, exist_ok=True)
            gj.to_file(spatial / "zones.geojson", driver="GeoJSON")
            zcmap[zcmap["z"].isin(zs)].to_csv(spatial / "zcmap.csv", index=False)
            pd.DataFrame([{"z": k["a"], "zz": k["b"]} for k in links]).to_csv(
                spatial / "sTopology.csv", index=False)
            pd.DataFrame([{"z": k["a"], "zz": k["b"], "pTransferLimit": k["mw"],
                           "forward": k["ab"], "reverse": k["ba"], "year": k["year"],
                           "status": k["status"],
                           "note": "actual model values (pTransferLimit.csv)"}
                          for k in links]).to_csv(
                spatial / "pTransferLimit_estimated.csv", index=False)

        runs.append({"iso": iso, "n_zones": n, "label": label,
                     "path": str(STUDY_ROOT / label), "links": links, "gdf": gj})
    return runs


def update_study_index(runs, dry_run=False):
    path = STUDY_ROOT / "index.json"
    idx = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for run in runs:
        entry = {"n_zones": run["n_zones"], "status": "ok", "source": SOURCE_LABEL,
                 "path": run["path"]}
        kept = [r for r in idx.get(run["iso"], []) if r.get("n_zones") != run["n_zones"]]
        idx[run["iso"]] = sorted(kept + [entry], key=lambda r: r["n_zones"])
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  study index: {path}")


# ── export to the Explorer ────────────────────────────────────────────────────

def source_note(data_folder: str) -> dict:
    """Provenance shown under the zone list in the Explorer (ZoningTab).

    Its default note describes the clustered runs ("k-means clustering of OSM
    substations... transfer limits are rough proxies"). That is plainly wrong for
    these zones, which are the model's own and carry the model's own MW, hence the
    per-run override in sources.json.
    """
    return {
        "method": f"Zones of the EPM model itself ({data_folder}): one zone per "
                  "modelled node, no clustering.",
        "boundaries": "Boundaries are the model's zone polygons (zones_zcmap.geojson), "
                      "as used in the dispatch.",
        "limits": f"Corridor ratings are the model's own pTransferLimit for {REF_YEAR}, "
                  "not an estimate; both values are shown when the two directions differ.",
    }


def export_to_explorer(runs, data_folder, dry_run=False):
    """Same outputs as export_zones_to_explorer.py, plus model-derived corridors."""
    countries_path = _EXPLORER / "public" / "data" / "countries_10m.geojson"
    countries = gpd.read_file(countries_path) if countries_path.exists() else None

    if not dry_run:
        ZONES_OUT.mkdir(parents=True, exist_ok=True)

    for run in runs:
        label, gdf = run["label"], run["gdf"]

        # clip to the reference land boundary the Explorer draws countries with
        if countries is not None:
            match = countries[countries["ISO_A3"] == run["iso"]]
            if not match.empty:
                gdf = gdf.clip(unary_union(match.geometry))

        # internal borders = every zone boundary minus the outer country outline
        all_bounds = unary_union([g.boundary for g in gdf.geometry])
        outer = unary_union(gdf.geometry).boundary
        # clipping shatters the difference into hundreds of segments, so stitch the
        # touching ones back into continuous borders before writing
        inner_geom = all_bounds.difference(outer.buffer(1e-6))
        if not inner_geom.is_empty:
            inner_geom = linemerge(inner_geom) if hasattr(inner_geom, "geoms") else inner_geom
        inner = list(inner_geom.geoms) if hasattr(inner_geom, "geoms") else \
            ([inner_geom] if not inner_geom.is_empty else [])

        # corridors between zone centroids, carrying the model's transfer limits
        cent = {r.zone_name: r.geometry.centroid for r in gdf.itertuples()}
        corridors = gpd.GeoDataFrame(
            [{"zone_a": k["a"], "zone_b": k["b"], "mw": k["mw"],
              "status": k["status"], "label": k["label"],
              "geometry": LineString([cent[k["a"]], cent[k["b"]]])}
             for k in run["links"] if k["a"] in cent and k["b"] in cent],
            crs=gdf.crs) if run["links"] else None

        print(f"  {label}: {len(gdf)} zones, {len(inner)} inner edges, "
              f"{0 if corridors is None else len(corridors)} corridors")

        if dry_run:
            continue
        gdf.to_file(ZONES_OUT / f"{label}_zones.geojson", driver="GeoJSON")
        gpd.GeoDataFrame(geometry=inner, crs=gdf.crs).to_file(
            ZONES_OUT / f"{label}_inner.geojson", driver="GeoJSON")
        (ZONES_OUT / f"{label}_topo.json").write_text(
            json.dumps([{"z": k["a"], "zz": k["b"]} for k in run["links"]]),
            encoding="utf-8")
        if corridors is not None:
            corridors.to_file(ZONES_OUT / f"{label}_corridors.geojson", driver="GeoJSON")

    # merge into the Explorer index, leaving every other country untouched
    path = ZONES_OUT / "index.json"
    idx = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for run in runs:
        ks = sorted(set(idx.get(run["iso"], [])) | {run["n_zones"]})
        idx[run["iso"]] = ks
    if not dry_run:
        path.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    print(f"  explorer index: +{len(runs)} countries -> {len(idx)} total")

    # same for the provenance note, so these runs stop claiming to be clustered
    spath = ZONES_OUT / "sources.json"
    src = json.loads(spath.read_text(encoding="utf-8")) if spath.exists() else {}
    note = source_note(data_folder)
    for run in runs:
        src[f"{run['iso']}_{run['n_zones']}z"] = note
    if not dry_run:
        spath.write_text(json.dumps(src, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  explorer sources: +{len(runs)} runs -> {len(src)} total")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-folder", required=True,
                    help="folder under epm/input/ holding the model zoning")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()

    data_dir = _REPO / "epm" / "input" / args.data_folder
    if not data_dir.exists():
        sys.exit(f"Not found: {data_dir}")
    if not _EXPLORER.exists():
        sys.exit(f"Explorer repo not found: {_EXPLORER}")

    print(f"\nModel zoning: {data_dir}")
    zones, zcmap, links = read_model_zoning(data_dir)
    print(f"{len(zones)} zones, {len(links)} internal corridors, MW at {REF_YEAR}\n")

    print("Study runs:")
    runs = write_study_runs(zones, zcmap, links, args.dry_run)
    update_study_index(runs, args.dry_run)

    print("\nExplorer export:")
    export_to_explorer(runs, args.data_folder, args.dry_run)
    print("\nDone." + ("  (dry run — nothing written)" if args.dry_run else ""))
