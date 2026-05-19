"""
Zoning sensitivity study — runs zone_pipeline for each (country, n_zones) combo.
n_zones range is auto-derived from country area (Natural Earth boundaries).

Usage:
    # Auto n_zones, one or more countries:
    gams_env python pre-analysis/run_zoning_study.py --countries TUR ROU BGR

    # Explicit n_zones override:
    gams_env python pre-analysis/run_zoning_study.py --countries TUR --n_zones 3 4 5

    # Full pools (run from repo root):
    gams_env python pre-analysis/run_zoning_study.py --countries \\
        EGY LBY SDN SSD ETH DJI KEN TZA SOM UGA RWA BDI COD \\
        ZAF ZWE ZMB BWA MOZ MWI NAM LSO SWZ MDG AGO \\
        KAZ KGZ TJK TKM UZB

Outputs under pre-analysis/output_workflow/zoning_study/{ISO}_{N}z/
  epm_export/spatial/  -> zcmap.csv, zones.geojson, sTopology.csv
  index.json           -> summary of all completed runs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_BASE = Path(__file__).resolve().parent          # pre-analysis/
_NE_CACHE = _BASE / "resolution_advisor" / "cache" / "natural_earth"

sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_BASE / "pipelines"))

from pipelines.zone_pipeline import run_zone_pipeline

STUDY_ROOT = _BASE / "output_workflow" / "zoning_study"


# ── n_zones auto-sizing ───────────────────────────────────────────────────────

def _country_area_km2(iso: str) -> float:
    """Return country area in km² from the cached NE admin-1 shapefile."""
    try:
        import geopandas as gpd
        shp = _NE_CACHE / "ne_10m_admin_1_states_provinces.shp"
        if not shp.exists():
            return 200_000  # fallback: medium country
        gdf = gpd.read_file(shp)
        iso_col = next((c for c in ["iso_a3", "ISO_A3", "adm0_a3"] if c in gdf.columns), None)
        if not iso_col:
            return 200_000
        subset = gdf[gdf[iso_col] == iso]
        if subset.empty:
            return 0
        union = subset.geometry.unary_union
        # Reproject to equal-area for area computation
        import pyproj
        from shapely.ops import transform
        proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
        area_m2 = transform(proj.transform, union).area
        return area_m2 / 1e6
    except Exception:
        return 200_000


def _auto_n_zones(iso: str) -> list[int]:
    """Return list of n_zones to test based on country area."""
    area = _country_area_km2(iso)
    if area < 30_000:
        return [1, 2]
    elif area < 100_000:
        return [1, 2, 3]
    elif area < 300_000:
        return [1, 2, 3, 4]
    elif area < 800_000:
        return [1, 2, 3, 4, 5]
    else:
        return [1, 2, 3, 4, 5, 6, 8]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_study(
    countries: list[str],
    n_zones_override: list[int] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run all (country, n_zones) combinations.
    Skips runs whose output already contains zones.geojson with status ok.
    """
    log = print if verbose else (lambda *a: None)
    STUDY_ROOT.mkdir(parents=True, exist_ok=True)

    # Build per-country config
    configs: dict[str, list[int]] = {}
    for iso in countries:
        if n_zones_override:
            configs[iso] = list(n_zones_override)
        else:
            n_zones = _auto_n_zones(iso)
            configs[iso] = n_zones
            log(f"  {iso}: area={_country_area_km2(iso):.0f} km²  -> n_zones={n_zones}")

    total = sum(len(ns) for ns in configs.values())
    log(f"\n{'='*60}")
    log(f"  Zoning study: {len(configs)} countries, {total} runs")
    log(f"  Output root : {STUDY_ROOT}")
    log(f"{'='*60}\n")

    # Load previous index
    index_path = STUDY_ROOT / "index.json"
    prev_index: dict = {}
    if index_path.exists():
        try:
            with open(index_path, encoding="utf-8") as f:
                prev_index = json.load(f)
        except Exception:
            pass

    summary: dict[str, list[dict]] = {}
    run_index = 0

    for iso, zone_counts in configs.items():
        summary[iso] = []
        for n in zone_counts:
            run_index += 1
            label = f"{iso}_{n}z"
            out_root = STUDY_ROOT / label

            done_marker = out_root / "epm_export" / "spatial" / "zones.geojson"
            prev_status = next(
                (r["status"] for r in prev_index.get(iso, []) if r["n_zones"] == n),
                None,
            )
            if done_marker.exists() and prev_status == "ok":
                log(f"[{run_index}/{total}] {label} — already done, skipping")
                summary[iso].append({"n_zones": n, "status": "skipped", "path": str(out_root)})
                continue

            log(f"\n[{run_index}/{total}] {label} — running...")
            t0 = time.time()
            try:
                paths = run_zone_pipeline(
                    countries=[iso],
                    n_zones=n,
                    output_root=out_root,
                    verbose=verbose,
                )
                elapsed = time.time() - t0
                log(f"  done in {elapsed:.1f}s")
                summary[iso].append({
                    "n_zones": n,
                    "status": "ok",
                    "elapsed_s": round(elapsed, 1),
                    "path": str(out_root),
                    "outputs": {k: str(v) for k, v in paths.items()},
                })
            except Exception as exc:
                elapsed = time.time() - t0
                log(f"  FAILED after {elapsed:.1f}s: {exc}")
                summary[iso].append({
                    "n_zones": n,
                    "status": "failed",
                    "error": str(exc),
                    "path": str(out_root),
                })

    # Merge into existing index
    merged: dict = {}
    if index_path.exists():
        try:
            with open(index_path, encoding="utf-8") as f:
                merged = json.load(f)
        except Exception:
            pass
    for iso, runs in summary.items():
        existing = {r["n_zones"]: r for r in merged.get(iso, [])}
        for r in runs:
            existing[r["n_zones"]] = r
        merged[iso] = sorted(existing.values(), key=lambda r: r["n_zones"])
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    log(f"\nIndex written -> {index_path}")

    # Summary table
    log(f"\n{'='*60}")
    log("  SUMMARY")
    log(f"{'='*60}")
    for iso, runs in summary.items():
        for r in runs:
            mark = "ok" if r["status"] == "ok" else ("--" if r["status"] == "skipped" else "FAIL")
            t = f"  ({r['elapsed_s']}s)" if "elapsed_s" in r else ""
            log(f"  {mark}  {iso}_{r['n_zones']}z{t}")
    log("")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zoning sensitivity study")
    parser.add_argument("--countries", nargs="+", required=True, metavar="ISO",
                        help="Country ISO_A3 codes to run")
    parser.add_argument("--n_zones", nargs="+", type=int, metavar="N",
                        help="Override n_zones list (applies to all countries; default: auto from area)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_study(
        countries=args.countries,
        n_zones_override=args.n_zones,
        verbose=not args.quiet,
    )
