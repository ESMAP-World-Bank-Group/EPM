"""
make_transmission_map.py
========================
Black Sea EPM transmission map — labelled (MW), internal Türkiye links included,
external neighbours as discreet grey dots.

Lines (with MW label at midpoint):
  existing   solid blue     <- trade/pTransferLimit.csv (internal, incl. Türkiye)
                               + trade/pExtTransferLimit.csv (internal↔external)
  committed  dashed GREEN   <- corridors.ref_generated.csv (committed_*)
  candidate  dashed ORANGE  <- trade/pNewTransmission.csv (Status 3)
                               + pNewTransmissionExt.ref_generated.csv
Output: pre-analysis/output_transmission/blacksea_transmission_map.html
"""
from __future__ import annotations

from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

_BASE = Path(__file__).resolve().parent
_DB = _BASE.parent / "epm" / "input" / "data_blacksea"
_OUT = _BASE / "output_transmission"
YEAR = "2025"

_COUNTRY_COLOR = {"Turkiye": "#4E79C4", "Türkiye": "#4E79C4", "Georgia": "#E88080",
                  "Armenia": "#A8D8A8", "Azerbaijan": "#E8D87C"}
_EXT_FILL = "#D8CDBE"      # warm taupe-grey — distinct from the blue sea
_EXT_LINE = "#A8957E"
_C_EXIST, _C_COMMIT, _C_CAND = "#1f4e79", "#2e9e44", "#e8851a"

# External neighbour anchors (lat, lon) — placed inside each country, discreet dots.
_EXT_ANCHOR = {
    "Bulgaria": (42.7, 25.3), "Greece": (40.9, 24.5), "Romania": (44.3, 27.8),
    "Serbia": (44.8, 20.5), "Hungary": (47.4, 19.1), "Moldova": (47.0, 28.9),
    "Ukraine": (48.3, 30.3), "Russia": (43.5, 45.5), "Iran": (37.9, 46.5),
    "Iraq": (35.8, 43.7), "Syria": (35.5, 38.5), "Kazakhstan": (43.65, 51.2),
}

# Override internal-zone connection points (lat, lon) where the polygon centroid
# is a poor endpoint.
_ZONE_OVERRIDE = {"Armenia": (39.723084, 45.786788)}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _centroids(gdf):
    pts = gdf.to_crs(3857).geometry.representative_point().to_crs(4326)
    return {str(z): (p.y, p.x) for z, p in zip(gdf["z"].astype(str), pts)}


def _w(mw):
    return 1.6 + min(7.0, _num(mw) / 1500.0)


def main() -> int:
    internal_zones = set(pd.read_csv(_DB / "zcmap.csv")["z"].astype(str).str.strip())
    zones = gpd.read_file(_DB / "zones.geojson")
    zi = zones[zones["z"].astype(str).isin(internal_zones)]
    cz = _centroids(zi)
    cz.update(_ZONE_OVERRIDE)               # manual connection-point overrides
    allc = {**cz, **_EXT_ANCHOR}

    lat = sum(p[0] for p in cz.values()) / len(cz)
    lon = sum(p[1] for p in cz.values()) / len(cz)
    m = folium.Map(location=[lat, lon + 2], zoom_start=6, tiles="CartoDB positron")
    folium.TileLayer("cartodbpositronnolabels", name="Basemap — no labels").add_to(m)

    # opaque "all countries" background (toggleable) — clean look without basemap labels
    bg_path = _DB / "extras" / "background_countries.geojson"
    if bg_path.exists():
        bg = gpd.read_file(bg_path)
        folium.GeoJson(bg.to_json(), name="Countries (opaque base)",
                       style_function=lambda f: {"fillColor": "#EFEADD", "color": "#D6CDBA",
                                                 "weight": 0.5, "fillOpacity": 0.95}).add_to(m)

    # internal zones: light country fill
    folium.GeoJson(zi.to_json(), name="Model zones",
                   style_function=lambda f: {"fillColor": _COUNTRY_COLOR.get(f["properties"].get("c", ""), "#D8D8D8"),
                                             "color": "#888", "weight": 0.8, "fillOpacity": 0.55},
                   tooltip=folium.GeoJsonTooltip(fields=["z", "c"])).add_to(m)

    # external neighbours: faint grey-blue polygons + discreet dots; names on a
    # SEPARATE toggleable layer.
    ext = gpd.read_file(_DB / "zones_ext.geojson")
    folium.GeoJson(ext.to_json(), name="External neighbours",
                   style_function=lambda f: {"fillColor": _EXT_FILL, "color": _EXT_LINE,
                                             "weight": 0.8, "fillOpacity": 0.65},
                   tooltip=folium.GeoJsonTooltip(fields=["z"])).add_to(m)
    # names only (no dot/pin markers), placed at the country CENTRE (centroid of the
    # polygon clipped to the visible region), away from the border connection points.
    fg_names = folium.FeatureGroup(name="Neighbour labels").add_to(m)
    _view = box(16, 31, 58, 50)
    for _, row in ext.iterrows():
        g = row.geometry.intersection(_view)
        if g.is_empty:
            g = row.geometry
        p = gpd.GeoSeries([g], crs=4326).to_crs(3857).representative_point().to_crs(4326).iloc[0]
        folium.Marker((p.y, p.x), icon=folium.DivIcon(class_name="",
            html=f'<div style="display:inline-block;font-size:9px;color:#5b6b7d;font-weight:600;'
                 f'transform:translate(-50%,-50%);white-space:nowrap">{row["z"]}</div>',
            icon_size=(1, 1))).add_to(fg_names)

    fg = {"e": folium.FeatureGroup(name="Existing").add_to(m),
          "c": folium.FeatureGroup(name="Committed").add_to(m),
          "n": folium.FeatureGroup(name="Candidate").add_to(m)}
    _style = {"e": (_C_EXIST, None), "c": (_C_COMMIT, "8,6"), "n": (_C_CAND, "6,6")}

    def draw(a, b, kind, mw, tip, label=None, t=0.5):
        """Draw a line AND its MW label into the SAME layer (toggle together).
        `t` = position fraction along the line (offset by kind to reduce overlap)."""
        if a not in allc or b not in allc or _num(mw) <= 0:
            return
        col, dash = _style[kind]
        pa, pb = allc[a], allc[b]
        folium.PolyLine([pa, pb], color=col, weight=_w(mw), opacity=0.85,
                        dash_array=dash, tooltip=tip).add_to(fg[kind])
        pt = (pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]))
        txt = label if label is not None else f"{_num(mw):,.0f} MW"
        folium.Marker(pt, icon=folium.DivIcon(class_name="", html=(
            f'<div style="display:inline-block;transform:translateX(-50%);font-size:8px;'
            f'color:#222;background:rgba(255,255,255,.88);padding:1px 5px;border-radius:3px;'
            f'white-space:nowrap;box-shadow:0 1px 2px rgba(0,0,0,.15)">{txt}</div>'),
            icon_size=(1, 1))).add_to(fg[kind])

    # existing internal (incl. Türkiye) from pTransferLimit
    tl = pd.read_csv(_DB / "trade" / "pTransferLimit.csv")
    tl = tl[tl["q"] == "Q1"]
    seen = set()
    pair = {}
    for _, r in tl.iterrows():
        pair.setdefault(frozenset([r["z"], r["z2"]]), {})[(r["z"], r["z2"])] = _num(r[YEAR])
    for pr, dirs in pair.items():
        z = list(pr)
        a, b = (z[0], z[1]) if len(z) == 2 else (z[0], z[0])
        mw = max(dirs.values())
        draw(a, b, "e", mw, f"existing {a}↔{b}: {mw:,.0f} MW")

    # existing external from pExtTransferLimit
    et = pd.read_csv(_DB / "trade" / "pExtTransferLimit.csv")
    dcol = [c for c in et.columns if c not in ("z", "zext", "q") and not str(c).isdigit()][0]
    for (z, ze), g in et[et["q"] == "Q1"].groupby(["z", "zext"]):
        mw = max(_num(g[YEAR].max()), 0)
        if mw > 0:
            draw(z, ze, "e", mw, f"existing {z}↔{ze}: {mw:,.0f} MW")

    # committed from corridors — "+xxx" label when it reinforces an EXISTING corridor
    cor = pd.read_csv(_OUT / "corridors.ref_generated.csv")
    for _, r in cor.iterrows():
        cm = max(_num(r.get("committed_lohi", 0)), _num(r.get("committed_hilo", 0)))
        if cm > 0:
            ex = max(_num(r.get("existing_lohi", 0)), _num(r.get("existing_hilo", 0)))
            lbl = f"+{cm:,.0f} MW" if ex > 0 else None     # reinforcement -> incremental
            kind_txt = "reinforcement" if ex > 0 else "new line"
            draw(str(r["z"]), str(r["z2"]), "c", cm,
                 f"committed {kind_txt} {r['z']}↔{r['z2']}: +{cm:,.0f} MW (COD {r.get('committed_cod','?')})",
                 label=lbl, t=0.4)

    # candidates: internal (Status 3) + external
    nt = pd.read_csv(_DB / "trade" / "pNewTransmission.csv")
    for _, r in nt[nt["Status"] == 3].iterrows():
        draw(str(r["From"]), str(r["To"]), "n", r["CapacityPerLine"],
             f"candidate {r['From']}→{r['To']}: {_num(r['CapacityPerLine']):,.0f} MW (COD {r.get('EarliestEntry','?')})",
             t=0.6)
    nte = pd.read_csv(_OUT / "pNewTransmissionExt.ref_generated.csv")
    for _, r in nte.iterrows():
        draw(str(r["From"]), str(r["To"]), "n", r["CapacityPerLine"],
             f"candidate {r['From']}→{r['To']}: {_num(r['CapacityPerLine']):,.0f} MW — {r.get('Project','')}",
             t=0.6)

    legend = ("<div style='position:fixed;bottom:18px;left:18px;z-index:9999;background:white;"
              "padding:7px 11px;border:1px solid #999;border-radius:5px;font-size:12px'>"
              "<b>Transmission</b><br><span style='color:#1f4e79'>──</span> existing<br>"
              "<span style='color:#2e9e44'>– –</span> committed<br>"
              "<span style='color:#e8851a'>– –</span> candidate</div>")
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    out = _OUT / "blacksea_transmission_map.html"
    m.save(str(out))
    print(f"Wrote {out}")
    print(f"  internal pairs(existing)={len(pair)} | ext.candidates={len(nte)} | "
          f"int.candidates={int((nt['Status']==3).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
