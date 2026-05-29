"""
Generates a standalone HTML map showing Black Sea zones with external connections
to neighboring countries.

Reads:
  output_workflow/preferred/blacksea_recommended_zones_hd.geojson  (zone polygons)
  data/reference_corridors.csv                                      (internal NTC)
  data/reference_corridors_ext.csv                                  (external NTC)

External anchor points are defined in EXTERNAL_ANCHORS below — placed well
inside each country, not on the border line.

Output:
  output_workflow/preferred/maps/blacksea_external_corridors.html

Usage:
    conda run -n gams_env python pre-analysis/make_external_corridors_map.py
"""
from __future__ import annotations

import csv
from pathlib import Path

_BASE = Path(__file__).resolve().parent

ZONES_GEOJSON = _BASE / "output_workflow" / "preferred" / "blacksea_recommended_zones_hd.geojson"
CORRIDORS_INT = _BASE / "data" / "reference_corridors.csv"
CORRIDORS_EXT = _BASE / "data" / "reference_corridors_ext.csv"
OUTPUT_DIR    = _BASE / "output_workflow" / "preferred" / "maps"
OUTPUT_HTML   = OUTPUT_DIR / "blacksea_external_corridors.html"

_COUNTRY_COLORS: dict[str, str] = {
    "TUR": "#E8A87C",
    "ROU": "#82C0E8",
    "ARM": "#A8D8A8",
    "AZE": "#E8D87C",
    "BGR": "#C8A0E8",
    "GEO": "#E88080",
}

# External zone anchors: key -> (display_name, lon, lat)
# Placed well inside each country, not on the border line.
EXTERNAL_ANCHORS: dict[str, tuple[str, float, float]] = {
    "GRC": ("Greece",  25.5,      40.8),       # Thessaloniki area
    "SRB": ("Serbia",  20.5,      44.8),       # Belgrade area
    "HUN": ("Hungary", 19.1,      47.5),       # near Budapest
    "MDA": ("Moldova", 28.9,      47.0),       # near Chișinău
    "UKR": ("Ukraine", 30.272831, 48.857934),  # Uman Raion, Cherkasy
    "RUS": ("Russia",  45.5,      43.5),       # Nalchik, between GEO and AZE crossings
    "IRN": ("Iran",    46.537879, 37.684531),  # Bostanabad, East Azerbaijan province
    "IRQ": ("Iraq",    43.698292, 35.839264),  # Makhmour, Ninive
    "SYR": ("Syria",   38.553054, 35.499157),  # Mansura
}

LINE_WEIGHT = 2.5

# Override zone centroids used for line endpoints (lat, lon).
# Useful when the polygon centroid is a poor connection point.
ZONE_CENTROID_OVERRIDES: dict[str, tuple[float, float]] = {
    "ARM_1": (39.251372, 46.242487),  # Syunik — actual Iran/Nakhchivan border region
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_corridors(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  WARNING: {path.name} not found — skipping")
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(r for r in f if not r.lstrip().startswith("#"))
        for row in reader:
            rows.append(row)
    return rows


def _mw(val) -> int:
    try:
        return int(float(val or 0))
    except (ValueError, TypeError):
        return 0


def _proj(val) -> str:
    s = str(val or "").strip()
    return "" if s in ("", "nan") else s


def _midpoint(p1: tuple, p2: tuple) -> tuple[float, float]:
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def _mw_label(ex: int, comm: int, cand: int = 0) -> str:
    """Single representative value: existing > committed > candidate."""
    if ex > 0:
        return f"{ex:,} MW"
    if comm > 0:
        return f"{comm:,} MW"
    if cand > 0:
        return f"{cand:,} MW"
    return ""


def _label_marker(location, text, fg):
    import folium
    folium.Marker(
        location=location,
        icon=folium.DivIcon(
            html=(
                f'<div style="display:inline-block;transform:translateX(-50%);'
                f'font-size:8px;color:#222;font-family:sans-serif;'
                f'background:rgba(255,255,255,0.85);padding:1px 4px;'
                f'border-radius:3px;white-space:nowrap;'
                f'box-shadow:0 1px 2px rgba(0,0,0,.18)">{text}</div>'
            ),
            icon_size=(1, 1), icon_anchor=(0, 0),
        ),
    ).add_to(fg)


# ── Map builder ────────────────────────────────────────────────────────────────

def make_map() -> None:
    import folium
    import geopandas as gpd

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading zones: {ZONES_GEOJSON.name}")
    zones_gdf = gpd.read_file(ZONES_GEOJSON)

    centroids: dict[str, tuple[float, float]] = {}
    for _, row in zones_gdf.iterrows():
        zname = str(row.get("zone_name") or row.get("zone_id", ""))
        c = row.geometry.centroid
        centroids[zname] = (c.y, c.x)
    centroids.update(ZONE_CENTROID_OVERRIDES)
    print(f"  {len(centroids)} zones: {sorted(centroids)}")

    int_corridors = _load_corridors(CORRIDORS_INT)
    ext_corridors = _load_corridors(CORRIDORS_EXT)
    print(f"  {len(int_corridors)} internal / {len(ext_corridors)} external corridor entries")

    m = folium.Map(
        location=[41.5, 40.0], zoom_start=5,
        tiles="CartoDB positron", prefer_canvas=True,
    )

    fg_zones      = folium.FeatureGroup(name="Zones",                            show=True)
    fg_int_ex     = folium.FeatureGroup(name="Internal — existing",              show=True)
    fg_int_comm   = folium.FeatureGroup(name="Internal — committed",             show=True)
    fg_int_cand   = folium.FeatureGroup(name="Internal — candidate",             show=False)
    fg_ext_ex     = folium.FeatureGroup(name="External connections — existing",  show=True)
    fg_ext_comm   = folium.FeatureGroup(name="External connections — committed", show=True)
    fg_ext_cand   = folium.FeatureGroup(name="External connections — candidate", show=False)
    fg_ext_nodes  = folium.FeatureGroup(name="Neighboring countries",            show=True)
    fg_labels     = folium.FeatureGroup(name="NTC values",                       show=False)

    # ── Zone polygons + connection dots (no labels) ────────────────────────────
    for _, row in zones_gdf.iterrows():
        zname   = str(row.get("zone_name") or row.get("zone_id", ""))
        country = str(row.get("country", zname[:3]))
        color   = _COUNTRY_COLORS.get(country, "#999999")
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color: {
                "fillColor": c, "color": "#555",
                "weight": 1.0, "fillOpacity": 0.40,
            },
            tooltip=folium.Tooltip(zname, sticky=False),
        ).add_to(fg_zones)
        lat, lon = centroids[zname]
        folium.CircleMarker(
            location=[lat, lon], radius=2,
            color="#444", fill=True, fill_color="#888",
            fill_opacity=0.85, weight=1.0,
            tooltip=zname,
        ).add_to(fg_zones)

    # ── Internal corridors ─────────────────────────────────────────────────────
    for row in int_corridors:
        z  = row.get("z",  "").strip()
        zz = row.get("zz", "").strip()
        if z not in centroids or zz not in centroids:
            continue
        pts  = [centroids[z], centroids[zz]]
        ex   = _mw(row.get("existing_mw"))
        comm = _mw(row.get("committed_mw"))
        cand = _mw(row.get("candidate_mw"))
        proj = _proj(row.get("projects"))
        tip_base = f"{z} ↔ {zz}" + (f" | {proj}" if proj else "")
        mid = _midpoint(pts[0], pts[1])

        if ex > 0:
            folium.PolyLine(pts, color="#1a5fa8", weight=LINE_WEIGHT, opacity=0.85,
                            tooltip=f"{tip_base} | {ex:,} MW existing").add_to(fg_int_ex)
        if comm > 0:
            folium.PolyLine(pts, color="#e07b00", weight=LINE_WEIGHT, opacity=0.82,
                            dash_array="8 4",
                            tooltip=f"{tip_base} | {comm:,} MW committed").add_to(fg_int_comm)
        if cand > 0:
            folium.PolyLine(pts, color="#555555", weight=LINE_WEIGHT, opacity=0.70,
                            dash_array="3 6",
                            tooltip=f"{tip_base} | {cand:,} MW candidate").add_to(fg_int_cand)
        lbl = _mw_label(ex, comm, cand)
        if lbl:
            _label_marker(mid, lbl, fg_labels)

    # ── External connections ───────────────────────────────────────────────────
    ext_used: set[str] = set()
    for row in ext_corridors:
        z   = row.get("z",        "").strip()
        ext = row.get("ext_zone", "").strip()
        if z not in centroids:
            print(f"  WARNING: zone '{z}' not found in zones GeoJSON")
            continue
        if ext not in EXTERNAL_ANCHORS:
            print(f"  WARNING: ext_zone '{ext}' not in EXTERNAL_ANCHORS")
            continue
        ext_used.add(ext)
        ext_name, ext_lon, ext_lat = EXTERNAL_ANCHORS[ext]
        pts  = [centroids[z], (ext_lat, ext_lon)]
        ex   = _mw(row.get("existing_mw"))
        comm = _mw(row.get("committed_mw"))
        cand = _mw(row.get("candidate_mw"))
        proj = _proj(row.get("projects"))
        tip_base = f"{z} ↔ {ext_name}" + (f" | {proj}" if proj else "")
        mid = _midpoint(pts[0], pts[1])

        if ex > 0:
            folium.PolyLine(pts, color="#1a5fa8", weight=LINE_WEIGHT, opacity=0.85,
                            tooltip=f"{tip_base} | {ex:,} MW existing").add_to(fg_ext_ex)
        if comm > 0:
            folium.PolyLine(pts, color="#e07b00", weight=LINE_WEIGHT, opacity=0.82,
                            dash_array="8 4",
                            tooltip=f"{tip_base} | {comm:,} MW committed").add_to(fg_ext_comm)
        if cand > 0:
            folium.PolyLine(pts, color="#555555", weight=LINE_WEIGHT, opacity=0.70,
                            dash_array="3 6",
                            tooltip=f"{tip_base} | {cand:,} MW candidate").add_to(fg_ext_cand)
        lbl = _mw_label(ex, comm, cand)
        if lbl:
            _label_marker(mid, lbl, fg_labels)

    # ── External zone markers ──────────────────────────────────────────────────
    for ext_key in sorted(ext_used):
        ext_name, lon, lat = EXTERNAL_ANCHORS[ext_key]
        folium.CircleMarker(
            location=[lat, lon], radius=3,
            color="#666", fill=True, fill_color="#ccc",
            fill_opacity=0.9, weight=1.0,
            tooltip=ext_name,
        ).add_to(fg_ext_nodes)
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:9px;color:#444;font-family:sans-serif;'
                    f'text-shadow:1px 1px 1px white,-1px -1px 1px white;'
                    f'white-space:pre;text-align:center;margin-top:9px">'
                    f'{ext_name}</div>'
                ),
                icon_size=(80, 28), icon_anchor=(40, -2),
            ),
        ).add_to(fg_ext_nodes)

    for fg in [fg_zones, fg_int_ex, fg_int_comm, fg_int_cand,
               fg_ext_ex, fg_ext_comm, fg_ext_cand, fg_ext_nodes, fg_labels]:
        fg.add_to(m)

    # ── Legend ─────────────────────────────────────────────────────────────────
    country_html = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
        f'<div style="width:12px;height:12px;background:{c};'
        f'opacity:.55;border:1px solid #888"></div><span>{iso}</span></div>'
        for iso, c in _COUNTRY_COLORS.items()
    )
    corridor_html = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
        f'<svg width="24" height="4"><line x1="0" y1="2" x2="24" y2="2" '
        f'stroke="{color}" stroke-width="2.5" stroke-dasharray="{dash}" '
        f'stroke-linecap="round"/></svg>'
        f'<span>{label}</span></div>'
        for label, color, dash in [
            ("Existing",  "#1a5fa8", ""),
            ("Committed", "#e07b00", "8 4"),
            ("Candidate", "#555",    "3 6"),
        ]
    )
    legend = (
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;'
        'background:rgba(255,255,255,.94);padding:10px 14px;border-radius:8px;'
        'box-shadow:0 2px 8px rgba(0,0,0,.22);font-family:sans-serif;'
        'font-size:12px;max-width:160px">'
        f'<b style="font-size:11px">Countries</b>{country_html}'
        '<hr style="margin:6px 0;border-color:#ddd">'
        f'<b style="font-size:11px">Corridors</b>{corridor_html}'
        '<hr style="margin:6px 0;border-color:#ddd">'
        '<span style="color:#999;font-size:9px">Pre-analysis only · Reference NTC data</span>'
        '</div>'
    )
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(str(OUTPUT_HTML))
    print(f"\nSaved -> {OUTPUT_HTML}")


if __name__ == "__main__":
    make_map()
