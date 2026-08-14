"""
Add the existing 9-zone Turkey dispatch configuration as TUR_9z in the zoning study.

Reads from epm/input/data_blacksea/:
  extras/zones_turkiye.geojson  -> zone polygons
  zcmap.csv                     -> zone->country mapping
  extras/sTopology.csv          -> zone connectivity
  trade/pTransferLimit.csv      -> actual transfer limits

Writes to pre-analysis/output_workflow/zoning_study/TUR_9z/ with the same
structure as pipeline-generated runs (epm_export/spatial/ + report/spatial/).
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent          # pre-analysis/
_REPO = _BASE.parent                             # EPM root
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_BASE / "pipelines"))

DATA_DIR  = _REPO / "epm" / "input" / "data_blacksea"
OUT_ROOT  = _BASE / "output_workflow" / "zoning_study" / "TUR_9z"
EPM_DIR   = OUT_ROOT / "epm_export" / "spatial"
REP_DIR   = OUT_ROOT / "report" / "spatial"

EPM_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Convert GeoJSON to pipeline format ────────────────────────────────────

def _convert_geojson():
    src = DATA_DIR / "extras" / "zones_turkiye.geojson"
    with open(src, encoding="utf-8") as f:
        raw = json.load(f)

    # Remap properties to pipeline format: zone_name, ISO_A3
    for feat in raw["features"]:
        p = feat["properties"]
        zone_name = p.get("ADMIN") or p.get("zone_name") or p.get("ISO_A3")
        feat["properties"] = {
            "zone_id":   zone_name,
            "zone_name": zone_name,
            "ISO_A3":    "TUR",
        }

    dst = EPM_DIR / "zones.geojson"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    print(f"  zones.geojson ({len(raw['features'])} zones)")
    return raw


# ── 2. zcmap.csv ─────────────────────────────────────────────────────────────

def _copy_zcmap():
    import pandas as pd
    src = DATA_DIR / "zcmap.csv"
    df = pd.read_csv(src)
    # Keep only internal Turkey zones (exclude Inter_*)
    df = df[~df["z"].str.startswith("Inter_")]
    dst = EPM_DIR / "zcmap.csv"
    df.to_csv(dst, index=False)
    print(f"  zcmap.csv ({len(df)} zones)")


# ── 3. sTopology.csv ─────────────────────────────────────────────────────────

def _copy_topology():
    import pandas as pd
    src = DATA_DIR / "extras" / "sTopology.csv"
    df = pd.read_csv(src)
    # Rename columns to pipeline format (z, zz) and drop external links
    df.columns = ["z", "zz"]
    df = df[~df["z"].str.startswith("Inter_") & ~df["zz"].str.startswith("Inter_")]
    # Deduplicate (keep one direction per pair)
    df["pair"] = df.apply(lambda r: tuple(sorted([r["z"], r["zz"]])), axis=1)
    df = df.drop_duplicates("pair").drop(columns="pair")
    dst = EPM_DIR / "sTopology.csv"
    df.to_csv(dst, index=False)
    print(f"  sTopology.csv ({len(df)} links)")


# ── 4. pTransferLimit.csv ────────────────────────────────────────────────────

def _copy_transfer():
    import pandas as pd
    src = DATA_DIR / "trade" / "pTransferLimit.csv"
    df = pd.read_csv(src)
    # Keep one representative value per pair (average Q1-Q4, year 2025)
    year_cols = [c for c in df.columns if c.isdigit()]
    if year_cols and "2025" in year_cols:
        df["pTransferLimit"] = df["2025"]
    elif year_cols:
        df["pTransferLimit"] = pd.to_numeric(df[year_cols[0]], errors="coerce")
    df = df.rename(columns={"z2": "zz"})
    # Average over seasons
    grp_cols = [c for c in ["z", "zz"] if c in df.columns]
    out = df.groupby(grp_cols)["pTransferLimit"].mean().reset_index()
    out["note"] = "actual model values (data_blacksea/trade/pTransferLimit.csv)"
    dst = EPM_DIR / "pTransferLimit_estimated.csv"
    out.to_csv(dst, index=False)
    print(f"  pTransferLimit_estimated.csv ({len(out)} links, actual values)")


# ── 5. HTML maps ──────────────────────────────────────────────────────────────

def _make_maps(geojson_data):
    try:
        import folium
        import pandas as pd

        zones_list = [
            {"zone_name": f["properties"]["zone_name"], "coords": f["geometry"]}
            for f in geojson_data["features"]
        ]
        n = len(zones_list)
        COLORS = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
                  "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac"]
        color_map = {z["zone_name"]: COLORS[i % len(COLORS)] for i, z in enumerate(zones_list)}

        # Compute centroids
        centroids = {}
        for feat in geojson_data["features"]:
            z = feat["properties"]["zone_name"]
            coords = feat["geometry"]["coordinates"]
            gtype  = feat["geometry"]["type"]
            pts = []
            if gtype == "Polygon":
                pts = coords[0]
            elif gtype == "MultiPolygon":
                for poly in coords:
                    pts.extend(poly[0])
            if pts:
                centroids[z] = (
                    sum(p[1] for p in pts) / len(pts),
                    sum(p[0] for p in pts) / len(pts),
                )

        topo = pd.read_csv(EPM_DIR / "sTopology.csv")

        def _base_map(title):
            m = folium.Map(location=[39.0, 35.0], zoom_start=6, tiles="CartoDB positron")
            for feat in geojson_data["features"]:
                z = feat["properties"]["zone_name"]
                color = color_map[z]
                folium.GeoJson(
                    feat,
                    style_function=lambda f, c=color: {
                        "fillColor": c, "color": "#333",
                        "weight": 1.5, "fillOpacity": 0.35,
                    },
                    tooltip=z,
                ).add_to(m)
                if z in centroids:
                    lat, lon = centroids[z]
                    folium.Marker(
                        [lat, lon],
                        icon=folium.DivIcon(
                            html=f'<div style="font-size:10px;font-weight:bold;color:#111;'
                                 f'background:rgba(255,255,255,0.75);padding:2px 4px;'
                                 f'border-radius:3px;white-space:nowrap">{z}</div>',
                            icon_size=(120, 22), icon_anchor=(60, 11),
                        ),
                    ).add_to(m)
            for _, row in topo.iterrows():
                z1, z2 = row["z"], row["zz"]
                if z1 in centroids and z2 in centroids:
                    folium.PolyLine(
                        [centroids[z1], centroids[z2]],
                        color="#555", weight=2, opacity=0.55, dash_array="5 4",
                        tooltip=f"{z1} ↔ {z2}",
                    ).add_to(m)
            return m

        for fname, label in [
            ("zone_map_simplified.html", "Simplified"),
            ("zone_map_detailed.html",   "Detailed"),
            ("zone_capacity_mix.html",   "Capacity Mix (no GPPD)"),
        ]:
            m = _base_map(label)
            m.save(str(REP_DIR / fname))
            print(f"  {fname}")

    except ImportError as e:
        print(f"  [!] Maps skipped (folium not available): {e}")


# ── 6. Update index.json ─────────────────────────────────────────────────────

def _update_index():
    import time
    index_path = OUT_ROOT.parent / "index.json"
    idx: dict = {}
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            idx = json.load(f)

    entry = {
        "n_zones": 9,
        "status": "ok",
        "source": "existing model (data_blacksea dispatch zones)",
        "path": str(OUT_ROOT),
        "outputs": {
            "zones.geojson": str(EPM_DIR / "zones.geojson"),
            "zcmap.csv": str(EPM_DIR / "zcmap.csv"),
            "sTopology.csv": str(EPM_DIR / "sTopology.csv"),
            "pTransferLimit_estimated.csv": str(EPM_DIR / "pTransferLimit_estimated.csv"),
        },
    }
    idx.setdefault("TUR", [])
    # Replace existing TUR 9z entry if present
    idx["TUR"] = [r for r in idx["TUR"] if r.get("n_zones") != 9]
    idx["TUR"].append(entry)
    # Sort by n_zones
    idx["TUR"].sort(key=lambda r: r["n_zones"])

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)
    print(f"  index.json updated")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nAdding TUR_9z (existing dispatch zones) to zoning study...")
    print(f"Output: {OUT_ROOT}\n")

    geojson_data = _convert_geojson()
    _copy_zcmap()
    _copy_topology()
    _copy_transfer()
    _make_maps(geojson_data)
    _update_index()

    print(f"\nDone. TUR_9z is ready.")
