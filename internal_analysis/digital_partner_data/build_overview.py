"""Build the overview of the infrastructure data the Digital team delivered on 2026-10-13.

One self-contained page: what was delivered, four maps (the region, Georgia with
Open Net, Moldova with Orange, the grid and fibre co-location), the update of the
September screening figures, the gaps, and a proposal for the regional power
explorer. Figures come from analysis.py; the page reuses the map engine, styles
and small renderers of the sector briefs.

The page draws operator data shared under a transfer permission only. Keep it
internal until the Digital team confirms the publication rights.

Usage
    python -I build_overview.py                 # writes beside this script
    python -I build_overview.py --out <dir>     # writes into another folder as well
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import statistics
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "sector_briefs")]

import geopandas as gpd  # noqa: E402
import shapely  # noqa: E402

import analysis  # noqa: E402
from build_brief import CSS, JS  # noqa: E402
from mapkit import BASEMAP, PALETTE, CountryIndex, Map, lines_of, load_json  # noqa: E402
from power import GRID, esc, html_legend, imap, svg_of, sw_box, sw_dot, sw_line  # noqa: E402
from transport import bars, bullets, chain_svg, tiles  # noqa: E402

OUT = HERE / "digital_partner_data_overview.html"
WEB = analysis.WEB
ISOS = list(analysis.COUNTRIES)
STUDY = set(analysis.COUNTRIES.values())

STATUS = {  # one reading on every map, as in the sector briefs
    "operational": (PALETTE["operational"], None, "operational"),
    "under_construction": (PALETTE["building"], None, "under construction"),
    "planned": (PALETTE["planned"], "5 3", "planned"),
    "design": (PALETTE["planned"], "5 3", "design"),
}
SEPT = "#b8c2c9"        # the ITU catalogue as fetched in September
NEAR = PALETTE["accent"]
FAR = "#e07b39"
ACCESS = "#00acc1"
DISTRIB = "#4f6d82"
COVER = {"4g": "#e6f0f7", "5g": "#90bfe0"}
SHORT = {"operational": "operational", "under_construction": "building", "design": "design"}

LABELS = {  # country names, placed by hand
    "region": [("Ukraine", 31.5, 49.3), ("Moldova", 28.6, 47.2), ("Romania", 24.6, 45.9),
               ("Bulgaria", 25.2, 42.7), ("Turkiye", 34.5, 39.2), ("Georgia", 43.6, 42.15),
               ("Armenia", 44.9, 40.25), ("Azerbaijan", 47.6, 40.55), ("Russia", 41.0, 46.2),
               ("Iran", 48.0, 38.4), ("Black Sea", 34.5, 43.3)],
    "geo": [("Georgia", 43.2, 42.55), ("Russia", 44.4, 43.45), ("Turkiye", 41.2, 41.15),
            ("Armenia", 44.6, 41.08), ("Azerbaijan", 46.4, 41.3), ("Black Sea", 40.6, 42.6)],
    "mda": [("Moldova", 28.45, 47.55), ("Romania", 26.2, 46.4), ("Ukraine", 30.6, 48.1)],
    "grid": [("Moldova", 28.6, 47.3), ("Turkiye", 34.5, 39.3), ("Georgia", 43.6, 42.15),
             ("Armenia", 44.9, 40.25), ("Azerbaijan", 47.6, 40.55), ("Black Sea", 34.5, 43.3)],
}


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def fitted(bbox, width=980) -> Map:
    """A canvas whose height follows the aspect of the window."""
    probe = Map(width=width, height=width, bbox=bbox, pad=0)
    span_x = (bbox[2] - bbox[0]) * probe.kx
    return Map(width=width, height=round(width * (bbox[3] - bbox[1]) / span_x), bbox=bbox)


def base(bbox) -> Map:
    m = fitted(bbox)
    m.basemap(INDEX, highlight=STUDY)
    return m


def names(m: Map, key: str):
    for text, lon, lat in LABELS[key]:
        sea = text == "Black Sea"
        m.label(lon, lat, text, size=12 if sea else 11, colour="#7aa4bd" if sea else "#8a857c",
                weight="400" if sea else "600", halo=not sea)


def pix_path(m: Map, coords, seen: set) -> str:
    """Whole pixel path, culled to the window. Identical paths are kept once:
    the Orange layer holds 84,790 short segments that mostly share pixels."""
    lons = [q[0] for q in coords]
    lats = [q[1] for q in coords]
    if max(lons) < m.x0 or min(lons) > m.x1 or max(lats) < m.y0 or min(lats) > m.y1:
        return ""
    pts = []
    for lon, lat in coords:
        x, y = m.xy(lon, lat)
        q = (round(x), round(y))
        if not pts or q != pts[-1]:
            pts.append(q)
    if len(pts) < 2:
        return ""
    d = "M" + "L".join(f"{x},{y}" for x, y in pts)
    if d in seen:
        return ""
    seen.add(d)
    return d


def stroke(m: Map, lines, colour, width, dash=None, opacity=1.0):
    seen: set = set()
    d = "".join(pix_path(m, c, seen) for c in lines)
    if d:
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m.add(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"{da} '
              f'stroke-opacity="{opacity}" stroke-linecap="round" stroke-linejoin="round"/>')


def coords_of(geom):
    """Coordinate lists of a shapely line geometry."""
    if geom is None or geom.is_empty:
        return []
    return [list(p.coords) for p in shapely.get_parts(geom) if p.geom_type == "LineString"]


def web(name):
    return load_json(WEB / f"{name}.geojson")["features"]


def web_lines(name, key=None, value=None):
    return [c for f in web(name) if key is None or f["properties"].get(key) == value
            for c in lines_of(f["geometry"])]


def inside(lines, countries):
    """Lines whose middle vertex falls in one of the countries."""
    return [c for c in lines if INDEX.at(*c[len(c) // 2][:2]) in countries]


def hv_lines(country):
    return inside([c for f in load_json(analysis.OPEN / "osm_hv_lines.geojson")["features"]
                   for c in lines_of(f["geometry"])], {country})


def itu_status(m: Map, isos, width=1.6):
    for status in ("operational", "under_construction", "planned"):
        col, dash, _ = STATUS[status]
        lines = [c for iso in isos for c in web_lines(f"{iso}_backbone_itu", "status", status)]
        stroke(m, lines, col, width, dash)


def legend_status(statuses):
    return [(sw_line(STATUS[s][0], dash=STATUS[s][1]), STATUS[s][2]) for s in statuses]


# ---------------------------------------------------------------------------
# maps
# ---------------------------------------------------------------------------
def region_map() -> str:
    m = base((21.5, 37.6, 51.5, 52.6))
    sept = [c for f in load_json(analysis.OPEN / "itu_fibre.geojson")["features"]
            for c in lines_of(f["geometry"])]
    stroke(m, inside(sept, STUDY | {"Turkey"}), SEPT, 1.0)
    stroke(m, web_lines("submarine_cables"), PALETTE["cable"], 2.2, "2 3")
    itu_status(m, ISOS, 1.3)
    for iso in ISOS:
        for f in web(f"{iso}_nodes_itu"):
            lon, lat = f["geometry"]["coordinates"][:2]
            m.dot(lon, lat, 1.6, PALETTE["ink"], sw=0.4, opacity=0.8)
        for f in web(f"{iso}_facilities_itu"):
            if f["properties"]["kind"] == "landing_point":
                lon, lat = f["geometry"]["coordinates"][:2]
                m.dot(lon, lat, 5, "#ffffff", stroke=PALETTE["cable"], sw=2.2)
    names(m, "region")
    legend = html_legend([
        ("ITU backbone, delivery", legend_status(["operational", "under_construction", "planned"])),
        ("For comparison", [(sw_line(SEPT), "ITU catalogue as fetched in September")]),
        ("Subsea", [(sw_line(PALETTE["cable"], dash="2 3"), "submarine cable"),
                    (sw_dot(PALETTE["cable"], ring=True), "landing point")]),
        ("Nodes", [(sw_dot(PALETTE["ink"]), "ITU node (city or exchange)")]),
    ])
    cap = ("ITU Broadband Maps backbone for the five countries of the delivery, drawn over the "
           "catalogue fetched in September for the same countries and Turkiye. Grey alone marks "
           "routes the delivery does not cover, Turkiye above all. Links are traced node to "
           "node, not along the cable.")
    return imap(svg_of(m), cap, legend)


def georgia_map() -> str:
    m = base((39.7, 40.95, 46.9, 43.75))
    stroke(m, hv_lines("Georgia"), GRID, 3.4, opacity=0.45)
    stroke(m, web_lines("GEO_backbone_itu"), PALETTE["ink"], 0.9, "1 2")
    for status in ("design", "under_construction", "operational"):
        col, dash, _ = STATUS[status]
        stroke(m, web_lines("GEO_backbone_opennet", "status", status), col, 1.7, dash)
    for f in web("GEO_pop_opennet"):
        lon, lat = f["geometry"]["coordinates"][:2]
        m.dot(lon, lat, 2.1, STATUS[f["properties"]["status"]][0], sw=0.5)
    names(m, "geo")
    legend = html_legend([
        ("Open Net backbone", legend_status(["operational", "under_construction", "design"])),
        ("Open Net points of presence", [(sw_dot(STATUS[s][0]), STATUS[s][2])
                                         for s in ("operational", "under_construction", "design")]),
        ("Context", [(sw_line(PALETTE["ink"], dash="1 2"), "ITU backbone"),
                     (sw_line(GRID, h=5), "HV line 220 kV and above (OpenStreetMap)")]),
    ])
    cap = ("Open Net is the state open access network of Georgia's rural broadband programme. "
           "Its routes reach villages from the existing backbone and mostly run on poles: "
           "only a fifth of the network lies within 1 km of a high voltage line.")
    return imap(svg_of(m), cap, legend)


def moldova_map(orange: gpd.GeoDataFrame) -> str:
    m = Map(width=980, height=860, bbox=(26.5, 45.4, 30.3, 48.55))
    m.basemap(INDEX, highlight=STUDY)
    cover = {f["properties"]["tech"]: f["geometry"]
             for f in load_json(WEB / "MDA_coverage_orange.geojson")["features"]}
    for tech in ("4g", "5g"):
        geom = shapely.geometry.shape(cover[tech]).simplify(0.004, preserve_topology=False)
        polys = [p for p in shapely.get_parts(geom) if p.area > 2e-5]
        seen: set = set()
        d = " ".join(r + "Z" for p in polys for r in (
            pix_path(m, list(ring.coords), seen) for ring in [p.exterior, *p.interiors]) if r)
        m.add(f'<path d="{d}" fill="{COVER[tech]}" fill-rule="evenodd" stroke="none"/>')
    for access, col, width in ((False, DISTRIB, 0.9), (True, ACCESS, 1.1)):
        part = orange[orange.access == access]
        stroke(m, [c for g in part.geometry for c in coords_of(g)], col, width)
    stroke(m, hv_lines("Moldova"), GRID, 3.0, opacity=0.6)
    itu_status(m, ["MDA"], 1.8)
    names(m, "mda")
    legend = html_legend([
        ("Orange Moldova fibre", [(sw_line(DISTRIB), "distribution and other cables"),
                                  (sw_line(ACCESS), "access (cable named FTTH or PON)")]),
        ("Orange outdoor coverage", [(sw_box(COVER["4g"]), "4G"), (sw_box(COVER["5g"]), "5G")]),
        ("Context", [(sw_line(PALETTE["operational"]), "ITU backbone in service"),
                     (sw_line(PALETTE["planned"], dash="5 3"), "ITU backbone planned"),
                     (sw_line(GRID, h=5), "HV line 220 kV and above (OpenStreetMap)")]),
    ])
    cap = ("Orange Moldova's fibre is mostly access and distribution plant in and around "
           "towns, with thin links between them. 4G covers almost the whole territory, 5G only central "
           "Chisinau. The blue lines are the ITU backbone of all Moldovan operators.")
    return imap(svg_of(m), cap, legend)


def grid_map(stats) -> str:
    m = base((26.0, 37.9, 51.2, 48.9))
    for iso, parts in stats["grid"].items():
        stroke(m, coords_of(parts["far"]), FAR, 2.2)
        stroke(m, coords_of(parts["near"]), NEAR, 2.2)
    names(m, "grid")
    legend = html_legend([
        ("HV lines 220 kV and above", [(sw_line(NEAR), "mapped fibre within 3 km"),
                                       (sw_line(FAR), "no mapped fibre within 3 km")]),
        ("Fibre counted", [("", "ITU backbone of the delivery, plus Open Net in Georgia and "
                                "Orange non access cables in Moldova")]),
    ])
    cap = ("Grid lines from OpenStreetMap, cut into 500 m pieces and tested against every fibre "
           "layer of the delivery. Turkiye is outside the delivery and Ukraine is left out: "
           "the grid extract stops near 49.8 N.")
    return imap(svg_of(m), cap, legend)


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------
def pct(a, b) -> float:
    return 100.0 * a / b if b else 0.0


def table(head, rows, numeric=()) -> str:
    th = "".join(f'<th class="n">{esc(h)}</th>' if i in numeric else f"<th>{esc(h)}</th>"
                 for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(
        f'<td class="n">{c}</td>' if i in numeric else f"<td>{c}</td>"
        for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<table class="kt"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def messages(rows) -> str:
    return '<div class="msgs">' + "".join(
        f'<div class="msg"><div class="msg-n">{i}</div><div>{body}</div></div>'
        for i, body in enumerate(rows, 1)) + "</div>"


def card(head, body, legend="") -> str:
    leg = f'<div class="mixleg">{legend}</div>' if legend else ""
    return (f'<div class="mixcard"><div class="mixhead">{esc(head)}</div>{body}{leg}</div>')


def vertices_median() -> int:
    counts = [len(c) for iso in ISOS for c in web_lines(f"{iso}_backbone_itu")]
    return round(statistics.median(counts))


def headline(stats) -> str:
    c = stats["countries"]
    now = sum(r["itu_km"].get("operational", 0) for r in c.values())
    sept = sum(r["itu_sept_km"] for r in c.values())
    on, og = stats["opennet"], stats["orange"]
    geo, mda, arm = c["GEO"], c["MDA"], c["ARM"]
    return messages([
        f"<b>A real step up on the open data.</b> The ITU backbone of the five countries "
        f"comes to {now:,.0f} km in service against {sept:,.0f} km in the catalogue we fetched "
        f"in September, most of the gain in Ukraine. Two operator datasets that exist nowhere "
        f"in the open come with it: Open Net in Georgia ({sum(on['km_by_status'].values()):,.0f} km, "
        f"{sum(on['pops_by_status'].values())} points of presence) and Orange in Moldova "
        f"({og['km']:,.0f} km of fibre and 2G to 5G coverage).",
        f"<b>The grid and the fibre sit closer than the September note said.</b> Within 3 km "
        f"of a 220 kV and above line, the share of grid with mapped fibre rises in every "
        f"country: Georgia {geo['sept_share']}% to {pct(geo['hv_near_any_km'], geo['hv_km']):.0f}%, "
        f"Armenia {arm['sept_share']}% to {pct(arm['hv_near_any_km'], arm['hv_km']):.0f}%, "
        f"Moldova {mda['sept_share']}% to {pct(mda['hv_near_any_km'], mda['hv_km']):.0f}%. "
        "Section 7.2 of the screening note should be updated.",
        f"<b>Close is not on the towers.</b> ITU links are traced node to node, with a median "
        f"of {vertices_median()} vertices per link, so a 3 km test screens corridors, it does "
        "not find optical ground wire. No layer says which fibre is strung on transmission "
        "towers. That question goes to the grid operators, and it is the one the Digital team "
        "cares about most.",
        f"<b>The operator networks are access networks.</b> Only "
        f"{pct(geo['partner_along_hv_km'], geo['partner_km']):.0f}% of Open Net runs within 1 km "
        f"of a high voltage line and {pct(geo['partner_along_rail_km'], geo['partner_km']):.0f}% "
        f"along a railway. In Moldova {pct(og['access_km'], og['km']):.0f}% of Orange's cable is "
        "named FTTH or PON and the rest is town distribution. They describe who is connected, "
        "much more than the long haul routes a power or rail corridor could carry.",
        "<b>The rights must be settled before anything is published.</b> The operator layers "
        "are shared under a transfer permission, which is not a licence to publish, and pull "
        "request #43 puts them in a public repository that production would serve once merged.",
    ])


def delivered(stats) -> str:
    c = stats["countries"]
    on, og = stats["opennet"], stats["orange"]
    rows = [
        ("ITU Broadband Maps", "backbone, nodes, landing points, earth stations",
         "GEO, ARM, AZE, MDA, UKR", "2013 to 2019 go live dates",
         "CC BY-NC-SA, attribution to ITU", "Same source as September, refetched: 2.2 times the km"),
        ("ITU", "submarine cables", "Black Sea and Caspian", "2000 to 2026",
         "CC BY-NC-SA", "3 cables, already known"),
        ("ITU indicators", "fibre reach, Global Cybersecurity Index tier, G5 benchmark",
         "all countries", "2021 to 2025", "CC BY-NC-SA", "Comparators for the country panels"),
        ("Open Net", "backbone by status, points of presence with population served",
         "Georgia", "target dates in 2025", "transfer permitted, publication unclear",
         "New: rural open access network"),
        ("Orange Moldova", "fibre cables, outdoor coverage 2G, 3G, 4G, 5G", "Moldova",
         "not dated", "transfer permitted, publication unclear", "New: operator plant and coverage"),
    ]
    now = sum(r["itu_km"].get("operational", 0) for r in c.values())
    cards = tiles([
        (f"{now:,.0f} km", "ITU backbone in service, five countries"),
        (f"{sum(r['itu_nodes'] for r in c.values())}", "ITU nodes"),
        (f"{sum(on['km_by_status'].values()):,.0f} km", "Open Net backbone, all statuses"),
        (f"{og['km']:,.0f} km", f"Orange Moldova fibre, {og['segments']:,} segments"),
    ])
    return (f'<div class="stats four">{cards}</div>'
            + table(["Source", "Content", "Coverage", "Vintage", "Rights", "What it adds"],
                    [[esc(x) for x in r] for r in rows])
            + '<p class="note">Delivered as one GeoPackage of eight layers in EPSG:4326 and, in '
              'pull request #43, as one GeoJSON per country and layer under public/data/digital/, '
              'with data-source/sources/DIGITAL.yaml and a README.</p>')


def country_table(stats) -> str:
    rows = []
    for iso, r in stats["countries"].items():
        years = sorted(r["itu_years"].items(), key=lambda kv: -kv[1])
        dated = ", ".join(f"{y} ({n})" for y, n in years[:2])
        ops = ", ".join(r["itu_operators"])
        rows.append([
            esc(r["name"]),
            f"{r['itu_km'].get('operational', 0):,.0f}",
            f"{r['itu_sept_km']:,.0f}",
            f"{r['itu_km'].get('planned', 0) + r['itu_km'].get('under_construction', 0):,.0f}",
            f"{r['itu_nodes']}",
            f"{r['cross_border']}",
            esc(ops),
            esc(dated),
        ])
    return table(["Country", "In service, km", "September, km", "Planned, km", "Nodes",
                  "Cross border links", "Main operators", "Go live year (links)"],
                 rows, numeric=(1, 2, 3, 4, 5))


def km_bars(stats) -> str:
    rows = [(r["name"], r["itu_km"].get("operational", 0), "#9cc3dc", r["itu_sept_km"])
            for r in stats["countries"].values()]
    return bars(rows, lambda v: f"{v:,.0f}", width=520)


def georgia_block(stats) -> str:
    on, geo = stats["opennet"], stats["countries"]["GEO"]
    order = ["operational", "under_construction", "design"]
    km_rows = [(SHORT[s], on["km_by_status"].get(s, 0), STATUS[s][0], None) for s in order]
    pop_rows = [(SHORT[s], on["pops_by_status"].get(s, 0), STATUS[s][0], None) for s in order]
    kinds = ", ".join(f"{n} {k}s" for k, n in sorted(on["pops_by_type"].items(),
                                                       key=lambda kv: -kv[1]))
    return (
        georgia_map()
        + '<div class="grid3">'
        + card("Backbone by status, km", bars(km_rows, lambda v: f"{v:,.0f}", width=300))
        + card("Points of presence by status", bars(pop_rows, lambda v: f"{v:,.0f}", width=300))
        + '<div class="stats">' + tiles([
            (f"{on['population']:,}", "people in the areas the points of presence serve"),
            (f"{on['households']:,}", "households"),
            (f"{pct(geo['partner_along_hv_km'], geo['partner_km']):.0f}%",
             "of the network within 1 km of an HV line"),
            (f"{pct(geo['hv_near_partner_km'], geo['hv_km']):.0f}%",
             "of the HV grid within 3 km of Open Net"),
        ]) + "</div></div>"
        + f'<p class="note">Points of presence by type: {esc(kinds)}. The availability field '
          f'mixes a flag ("1" for the {on["pops_by_status"].get("operational", 0)} in service) '
          'with quarters from 2025-Q1 to 2025-Q4 for the others, which read as target dates.</p>'
    )


def moldova_block(stats, orange) -> str:
    og, mda = stats["orange"], stats["countries"]["MDA"]
    cov = [(t.upper(), v, "#9cc3dc", None) for t, v in og["coverage_pct"].items()]
    split = [("access", og["access_km"], ACCESS, None),
             ("other", og["km"] - og["access_km"], DISTRIB, None)]
    return (
        moldova_map(orange)
        + '<div class="grid3">'
        + card("Outdoor coverage, share of territory", bars(cov, lambda v: f"{v:.1f}", width=300,
                                                            unit="%"))
        + card("Fibre by role, km", bars(split, lambda v: f"{v:,.0f}", width=300))
        + '<div class="stats">' + tiles([
            (f"{og['segments']:,}", "cable segments"),
            (f"{pct(mda['hv_near_partner_km'], mda['hv_km']):.0f}%",
             "of the HV grid within 3 km of Orange cables (access excluded)"),
        ]) + "</div></div>"
        + '<p class="note">Coverage shares are the README figures, measured on World Bank '
          'boundaries. On the basemap of this note they come out 1 to 2 points higher for 2G, '
          '3G and 4G, the same for 5G.</p>'
    )


def colocation(stats) -> str:
    c = stats["countries"]
    rows = []
    bar_rows = []
    for iso, r in c.items():
        any_share = pct(r["hv_near_any_km"], r["hv_km"])
        partner = (f"{pct(r['hv_near_partner_km'], r['hv_km']):.0f}%"
                   if "hv_near_partner_km" in r else "")
        rows.append([esc(r["name"]) + (" <span class=\"small\">(partial grid)</span>"
                                       if iso == "UKR" else ""),
                     f"{r['hv_km']:,.0f}", f"{r['sept_share']}%",
                     f"{pct(r['hv_near_itu_km'], r['hv_km']):.0f}%", partner,
                     f"<b>{any_share:.0f}%</b>"])
        bar_rows.append((r["name"], any_share, "#9cc3dc", r["sept_share"]))
    legend = (f'<span class="lg">{sw_box(PALETTE["ink"])}September note</span>'
              f'<span class="lg">{sw_box("#9cc3dc")}with the delivery</span>')
    return (
        grid_map(stats)
        + '<div class="headrow">'
        + '<div>' + table(["Country", "HV grid, km", "September note", "ITU, delivery",
                           "Operator layer", "All sources"], rows, numeric=(1, 2, 3, 4, 5))
        + '<p class="note">Share of the 220 kV and above grid within 3 km of mapped fibre, '
          'the test of section 7.2 of the screening note. Operator layer: Open Net in Georgia, '
          'Orange cables other than access in Moldova.</p></div>'
        + card("Share of HV grid near fibre", bars(bar_rows, lambda v: f"{v:.0f}", width=300,
                                                  unit="%"), legend)
        + "</div>"
        + bullets([
            ("What it changes.", "The grid is a credible host for fibre in Armenia, Georgia and "
             "Moldova: three fifths to three quarters of the high voltage network already runs near "
             "fibre. The case for new optical ground wire is about the gaps in orange on the "
             "map, and about Azerbaijan, where the grid and the backbone part ways."),
            ("What it does not say.", "Whether that fibre is on the towers. The ITU traces are "
             "too coarse to tell, and none of the layers carries a support or owner type that "
             "would mark optical ground wire."),
            ("Next step.", "Ask the grid operators for their optical ground wire and dark fibre "
             "lengths by line: GSE in Georgia, the Armenian high voltage network, Azerenerji, "
             "Moldelectrica and Ukrenergo."),
        ])
    )


def indicators(stats) -> str:
    rows = []
    for iso, r in stats["indicators"].items():
        name = analysis.COUNTRIES.get(iso) or {"TUR": "Turkiye", "ROU": "Romania",
                                              "BGR": "Bulgaria"}[iso]
        reach = r.get("pop_within_10km_fibre_pct")
        rows.append([esc(name) + ("" if iso in analysis.COUNTRIES
                                  else ' <span class="small">(comparator)</span>'),
                     f"{reach:.0f}%" if reach is not None else "",
                     esc(r.get("gci2024_tier") or ""), esc(r.get("tracker2024_generation") or ""),
                     esc(r.get("g5_2021") or ""), esc(r.get("g5_2025") or "")])
    return (table(["Country", "Population within 10 km of fibre", "Cybersecurity Index 2024",
                   "ICT regulatory generation 2024", "G5 benchmark 2021", "G5 benchmark 2025"],
                  rows, numeric=(1,))
            + '<p class="note">From itu_indicators.json in the delivery, which holds these '
              'fields for every ITU member. Turkiye has the lowest fibre reach of the set, and is '
              'also the study country the spatial layers leave out.</p>')


def gaps() -> str:
    return table(["Topic", "What we found", "Ask"], [[esc(a), esc(b), esc(c)] for a, b, c in [
        ("Rights", "Open Net and Orange are shared under a transfer permission. The README's "
         "rights section is incomplete. The repository is public and a merge serves the files "
         "from production.",
         "Written confirmation from Open Net and Orange of what can be shown, at what detail. "
         "Until then, keep these layers out of the repository."),
        ("Licence", "ITU data are CC BY-NC-SA. Non commercial and share alike fit a World Bank "
         "explorer, but the share alike clause reaches derived layers.",
         "Credit ITU on the map and on the About page, keep derived files under the same "
         "licence."),
        ("Optical ground wire", "Nothing marks fibre on transmission towers.",
         "Ask the grid operators, or the Digital team's contacts, for OPGW by line."),
        ("Turkiye", "Not in the delivery, though it is one of the four study countries.",
         "Add Turkiye from ITU, and Romania and Bulgaria as comparators."),
        ("Data centre demand", "The inputs Himmat Sandhu offered for the EPM scenarios have not "
         "arrived. This is the first axis of the collaboration.",
         "Planned data centre sites and MW by year, by country."),
        ("Vintage", "Go live dates are 2013 for Armenia, Azerbaijan and most of Moldova, 2019 "
         "for Ukraine and Georgia. Ukraine's network is mapped before 2022.",
         "Date stamp the layers in the explorer and say so in the popups."),
        ("Geometry", "ITU links are straight node to node traces, so lengths and corridors are "
         "approximate.", "Show them as schematic links, not surveyed routes."),
        ("Earth stations", "The 35 Moldovan earth stations are satellite filings registered "
         "with the ITU, mostly broadcasting.", "Leave them out of the explorer."),
        ("README", "The section headed Rights and restrictions holds only the boundary "
         "disclaimer.", "Fill it in the pull request: licence, permission and allowed detail "
         "per source."),
    ]])


REGISTRY = """// src/layers/sectors/digital.js
export default {
  id: 'digital',
  label: 'Digital',
  sublayers: [
    { id: 'itu_backbone', label: 'Fibre backbone (ITU)', geom: 'line',
      file: iso => `digital/${iso}_backbone_itu.geojson`,
      colorBy: 'status', popup: ['operator', 'from', 'to', 'golive'],
      source: 'itu', scope: ['region', 'country'] },
    { id: 'landing', label: 'Cable landings', geom: 'point', ... },
    { id: 'opennet', label: 'Open Net (Georgia)', geom: 'line', countries: ['GEO'],
      file: 'digital/GEO_backbone_opennet.geojson', rights: 'pending', ... },
  ],
  sources: { itu: { name: 'ITU Broadband Maps', licence: 'CC BY-NC-SA', url: ... } },
}
// src/layers/sectors/transport.js, later: rail (OSM), ports, border crossings"""


def integration() -> str:
    flow = chain_svg([("Sector registry", "digital.js, transport.js"),
                      ("Data files", "GeoJSON per country"),
                      ("useSectorLayers", "load, show, popups"),
                      ("Layer panel", "Other infrastructure"),
                      ("About page", "credits from registry")], height=92)
    plan = table(["Phase", "Content", "Condition"], [[esc(a), esc(b), esc(c)] for a, b, c in [
        ("0. Rights", "Split pull request #43: the ITU layers and indicators on one side, the "
         "operator layers held back.", "Answer from the Digital team on Open Net and Orange."),
        ("1. Open layers", "ITU backbone by status, nodes, landing points, submarine cables, on "
         "the region and country pages. ITU indicators in the country panel.",
         "README rights section completed."),
        ("2. Operator layers", "Open Net backbone and points of presence on the Georgia page. "
         "Orange fibre and 4G and 5G coverage on the Moldova page, as vector tiles.",
         "Written publication permission, at the agreed detail."),
        ("3. Synergies", "Share of the HV grid near fibre in the country panel, OPGW when the "
         "grid operators answer, data centre sites. Transport through the same registry: rail, "
         "ports and crossings from the sector briefs data.", "Data centre inputs, OPGW data."),
    ]])
    return (
        '<p>The explorer has no notion of a sector today. Each page adds its MapLibre sources '
        'and layers by hand, and the layer panel hard codes its five sections (basemap, plants, '
        'transmission, substations, load centres) through some forty props. Adding digital the same way would add one more set of states, toggles and '
        'popups to each of the three pages, and transport would add a fourth. The proposal is a '
        'small registry that describes non power layers as data, read by one hook and one panel '
        'section.</p>'
        + f'<div class="primer-fig">{flow}</div>'
        + bullets([
            ("Registry.", "One file per sector lists its sub layers: id, label, geometry type, "
             "file pattern by country or region, style by status, legend swatch, popup fields, "
             "pages where it shows, and source credit. Status uses one vocabulary across "
             "sectors (operational, under construction, planned), as in the sector briefs."),
            ("Hook.", "useSectorLayers(map, sector, countries) loads a file only on first "
             "toggle, through the existing data cache and data manifest, so a missing file is "
             "skipped, adds the layers under the power layers, and opens one generic popup."),
            ("Panel.", "A single section, Other infrastructure, with a chip per sector (Digital, "
             "later Transport) and the sub layers below. The power sections stay as they are."),
            ("Heavy layers.", "Orange fibre and coverage go through the existing line tiles "
             "tool as PMTiles, on the Moldova country page only."),
            ("Credits.", "The About page reads the sources of each registry file, so a new "
             "sector cannot ship without its credit line."),
        ])
        + f'<pre class="code">{html.escape(REGISTRY)}</pre>'
        + plan
        + '<p class="note">Based on origin/main of regional-power-explorer as read on '
          '2026-10-27: RegionPage.jsx, CountryPage.jsx, WorldPage.jsx, LayerPanel.jsx, '
          'dataCache.js, dataGuard.js, AboutPage.jsx and tools/prepare_line_tiles.py.</p>'
    )


EXTRA_CSS = """
.stats.four{grid-template-columns:repeat(4,1fr);margin:6px 0 4px}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:4px 0 8px;
  align-items:start}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:22px;margin:8px 0;
  align-items:center}
.grid3 .stats{grid-template-columns:1fr}
section.block{padding:6px 0 18px}
section.block>p{font-size:.9rem;max-width:860px}
pre.code{background:#f7f8fa;border:1px solid var(--line);border-radius:8px;padding:12px 14px;
  font:12px/1.5 Consolas,Menlo,monospace;overflow-x:auto;color:#33404a}
.internal{display:inline-block;font-size:.74rem;font-weight:600;color:#8a3b12;
  background:#fdeee4;border:1px solid #f2c9ad;border-radius:4px;padding:3px 9px;margin:0 0 10px}
@media (max-width:760px){.stats.four,.grid3{grid-template-columns:1fr 1fr}
  .grid2{grid-template-columns:1fr}}
"""


def section(sid, title, body) -> str:
    return f'<section class="block" id="{sid}"><div class="mh">{esc(title)}</div>{body}</section>'


def page(stats, orange) -> str:
    zoom_js = JS[JS.index("/* Maps zoom"):]
    body = "".join([
        section("brief", "In brief", headline(stats)),
        section("delivered", "What was delivered", delivered(stats)),
        section("region", "The backbone across the five countries",
                region_map() + country_table(stats) + '<div class="grid2">'
                + card("In service, km: delivery and September (dark)", km_bars(stats))
                + bullets([
                    ("Ukraine carries the gain.", "Three times the September length, from 584 "
                     "links mostly dated 2019: the pre-war network."),
                    ("The Caucasus and Moldova move little.", "4 to 20% more length, with "
                     "planned links added, notably 563 km in Armenia."),
                    ("Dates are old.", "Most links in Armenia, Azerbaijan and Moldova carry a "
                     "2013 go live date."),
                ]) + "</div>"),
        section("georgia", "Georgia: Open Net", georgia_block(stats)),
        section("moldova", "Moldova: Orange", moldova_block(stats, orange)),
        section("grid", "The grid as a host for fibre: the September figures revisited",
                colocation(stats)),
        section("indicators", "ITU country indicators", indicators(stats)),
        section("gaps", "Gaps and questions for the Digital team", gaps()),
        section("explorer", "Adding the layer to the regional power explorer", integration()),
    ])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Digital infrastructure data</title>
<style>{CSS}{EXTRA_CSS}</style></head>
<body>
<header class="top"><div class="wrap">
<h1>Digital infrastructure data from the Digital and AI team</h1>
<p class="sub">What the delivery of 13 October 2026 holds, what it changes for the grid and fibre
screening, and how it could enter the regional power explorer.</p>
<span class="internal">Internal: draws operator data shared under a transfer permission</span>
<p class="aidisc">The production of this page includes AI processes. Double-checking the sources cited is recommended.</p>
</div></header>
<main class="wrap">{body}</main>
<footer class="wrap">Built {date.today().isoformat()} by
internal_analysis/digital_partner_data/build_overview.py from rpe_digital_infra_2026-10-13.gpkg,
the files of pull request #43 and the open layers of the September screening note.</footer>
<script>{zoom_js}</script>
</body></html>
"""


def main(extra_out: Path | None = None) -> None:
    global INDEX
    INDEX = CountryIndex(BASEMAP)
    stats = analysis.compute()
    orange = analysis.layer("orange_fibre")
    orange["access"] = orange["cable_name"].fillna("").str.upper().str.contains("FTTH|PON")
    OUT.write_text(page(stats, orange), encoding="utf8")
    (HERE / "data" / "stats.json").write_text(
        json.dumps({k: v for k, v in stats.items() if k != "grid"}, indent=1, default=str),
        encoding="utf8")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    if extra_out:
        target = extra_out / OUT.name
        shutil.copyfile(OUT, target)
        print(f"Copied to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="extra folder to copy the page into")
    args = parser.parse_args()
    main(args.out)
