"""Water pane of the Black Sea sector briefs.

Rivers and lakes are Natural Earth 10m, basins are HydroBASINS level 4, both
extracted to data/ by extract_water.py. Large hydro dams come from the GEM plant
tracker used in the Power pane. Country figures are FAO AQUASTAT and the SDG
6.4.2 series, quoted with their year. Projects are placed by hand.
"""

from __future__ import annotations

import math

from mapkit import FOCUS, PALETTE, CountryIndex, Map, lines_of, load_json
from power import (BBOX, H, HERE, W, esc, html_legend, imap, label_countries, svg_of,
                   sw_box, sw_dot, sw_line)
from transport import bars, batch, bullets, chain_svg, headrow, primer_block, projects_map, \
    timeline

RIVER = "#3f88c5"
LAKE = "#bcdcef"
RESERVOIR = "#7fb6dc"
DAM = PALETTE["ink"]
MUT = "#6f6a61"

_CACHE = {}


def layer(name):
    if name not in _CACHE:
        _CACHE[name] = load_json(HERE / "data" / f"water_{name}.geojson")
    return _CACHE[name]


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def river_width(rank, scale=1.0):
    rank = rank if rank is not None else 12
    return scale * (2.2 if rank <= 4 else 1.6 if rank <= 8 else 1.1 if rank <= 10 else 0.7)


def draw_rivers(m: Map, colour=RIVER, scale=1.0, max_rank=12, opacity=0.9, min_px=1.5):
    groups = {}
    for f in layer("rivers")["features"]:
        p = f["properties"]
        if (p["rank"] or 12) > max_rank or p["kind"] == "Intermittent River":
            continue
        groups.setdefault(river_width(p["rank"], scale), []).extend(lines_of(f["geometry"]))
    for width, lines in sorted(groups.items()):
        batch(m, lines, f'stroke="{colour}" stroke-width="{width:.2f}" '
                        f'stroke-opacity="{opacity}"', min_px)


def draw_lakes(m: Map, reservoirs=True):
    for f in layer("lakes")["features"]:
        p = f["properties"]
        is_res = p["kind"] == "Reservoir" or "Baraj" in p["name"] or "Reservoir" in p["name"]
        fill = RESERVOIR if (reservoirs and is_res) else LAKE
        for poly in f["geometry"]["coordinates"]:
            d = " ".join(r + "Z" for r in (m.path(ring, min_px=0.8) for ring in poly) if r)
            if d:
                m.add(f'<path d="{d}" fill="{fill}" stroke="{RIVER}" stroke-width="0.5" '
                      f'fill-rule="evenodd"><title>{esc(p["name"])}</title></path>')


def hydro_dams(min_mw=100):
    plants = load_json(HERE / "data" / "gem_plants.json")
    return [p for p in plants if p["type"] == "hydropower" and (p.get("operating") or 0) >= min_mw
            and BBOX[0] <= p["lon"] <= BBOX[2] and BBOX[1] <= p["lat"] <= BBOX[3]]


def draw_dams(m: Map, min_mw=100, k=0.22):
    for p in sorted(hydro_dams(min_mw), key=lambda q: -q["operating"]):
        x, y = m.xy(p["lon"], p["lat"])
        r = max(2.2, k * math.sqrt(p["operating"]))
        m.add(f'<path d="M{x:.1f},{y - r:.1f}L{x + r:.1f},{y + r * 0.8:.1f}'
              f'L{x - r:.1f},{y + r * 0.8:.1f}Z" fill="{DAM}" fill-opacity="0.8" '
              f'stroke="#fff" stroke-width="0.8"><title>{esc(p["name"])}, '
              f'{p["operating"]:,.0f} MW, {p.get("start") or "n/a"}</title></path>')


# Shared and closed basins, by HydroBASINS level 4 id. Everything else is drawn
# as a plain basin outline.
BASINS = {
    2040070050: ("Kura and Araks", "#8cc0de"),
    2040005120: ("Coruh", "#9fd1c4"),
    2040785900: ("Euphrates", "#e6cf9b"),
    2040816320: ("Tigris", "#dcc18a"),
    2040085990: ("Lake Van, closed", "#dcd6cb"),
    2040085690: ("Lake Urmia, closed", "#dcd6cb"),
    2040005130: ("Rioni and Enguri", "#cfe5f1"),
}
OTHER_BASIN = "#f3f1ec"


def draw_basins(m: Map, opacity=0.85):
    for f in layer("basins")["features"]:
        bid = f["properties"]["HYBAS_ID"]
        name, fill = BASINS.get(bid, (None, OTHER_BASIN))
        d = " ".join(r + "Z" for r in (m.path(ring, min_px=1.0)
                                        for ring in f["geometry"]["coordinates"]) if r)
        if not d:
            continue
        tip = (f"<title>{esc(name)}: {f['properties']['SUB_AREA']:,.0f} km2 in this unit"
               "</title>") if name else ""
        m.add(f'<path d="{d}" fill="{fill}" fill-opacity="{opacity}" stroke="#fff" '
              f'stroke-width="1.4" stroke-linejoin="round">{tip}</path>')


def draw_borders(m: Map, index: CountryIndex):
    """Country outlines on top of the basins, so both read."""
    for name, poly, *_ in index.entries:
        d = " ".join(r + "Z" for r in (m.path(ring, min_px=0.8) for ring in poly) if r)
        if d:
            m.add(f'<path d="{d}" fill="none" stroke="#8a857c" stroke-width="0.7" '
                  'stroke-dasharray="3 2"/>')


def flow_arrow(m: Map, lon, lat, angle, label=None, anchor="start", dx=0.2, dy=-0.05):
    """Small arrow where a river crosses a border, pointing downstream."""
    x, y = m.xy(lon, lat)
    a = math.radians(angle)
    ux, uy = math.cos(a), -math.sin(a)
    tip = (x + 9 * ux, y + 9 * uy)
    base = (x - 7 * ux, y - 7 * uy)
    px, py = -uy, ux
    m.add(f'<line x1="{base[0]:.1f}" y1="{base[1]:.1f}" x2="{tip[0]:.1f}" y2="{tip[1]:.1f}" '
          f'stroke="{PALETTE["ink"]}" stroke-width="2.6"/>')
    m.add(f'<path d="M{tip[0] + 5 * ux:.1f},{tip[1] + 5 * uy:.1f}'
          f'L{tip[0] + 5 * px:.1f},{tip[1] + 5 * py:.1f}'
          f'L{tip[0] - 5 * px:.1f},{tip[1] - 5 * py:.1f}Z" fill="{PALETTE["ink"]}"/>')
    if label:
        m.label(lon + dx, lat + dy, label, size=9.5, colour=PALETTE["ink"], anchor=anchor,
                weight="600", halo=True)


# ---------------------------------------------------------------------------
# 1. primer
# ---------------------------------------------------------------------------
def primer() -> str:
    svg = chain_svg(
        [("Snow and rain", "Caucasus, Anatolia"), ("Rivers", "Kura, Araks, Euphrates"),
         ("Reservoirs", "Storage and hydro"), ("Withdrawals", "Farms take most"),
         ("Downstream", "Next country, then sea")],
        notes=[(712, 40, "Each border hands the", None), (712, 54, "flow to a new owner.", None)],
        mid_fill=("Reservoirs",))
    return f'<div class="primer-fig">{svg}</div>' + bullets([
        ("Upstream holds the tap.",
         "Turkiye is upstream on the Coruh, Kura, Araks, Euphrates and Tigris. Azerbaijan is "
         "downstream of everyone: 77% of its water comes from abroad."),
        ("Farms drink most.",
         "Agriculture takes 78 to 92% of withdrawals in Armenia, Azerbaijan and Turkiye."),
        ("One reservoir, two masters.",
         "Hydro wants water in winter, irrigation in summer. The same dam cannot give both."),
        ("Few rules.",
         "The Kura has no basin agreement. A Georgia and Azerbaijan draft waits since 2014."),
    ])


# ---------------------------------------------------------------------------
# 2. key figures
# ---------------------------------------------------------------------------
# Renewable water per person, m3, and the share that comes from abroad.
PER_CAPITA = [("Georgia", 16633, 0.082), ("Azerbaijan", 3361, 0.766),
              ("Armenia", 2639, 0.117), ("Turkiye", 2425, 0.015)]

WATER_ROWS = [
    ("Turkiye", "211.6", "1.5%", "64.6 (2022)", "87%", "48%", "111.6 (2010)", "3,564 (2022)",
     "Upstream on all five shared rivers."),
    ("Georgia", "63.3", "8%", "1.3 (2022)", "34%", "4%", "3.4", "488 (2007)",
     "Water rich. Coruh and Kura arrive from Turkiye."),
    ("Armenia", "7.8", "12%", "3.1 (2022)", "78%", "62%", "1.4 (1993)", "156 (2009)",
     "Lake Sevan is the reserve. Araks shared with Turkiye and Iran."),
    ("Azerbaijan", "34.7", "77%", "13.0 (2021)", "92%", "58%", "21.5 (2000)", "1,485 (2021)",
     "Downstream of all. Kura and Araks end here."),
]


def key_figures() -> str:
    tile_rows = [
        ("77%", "Azerbaijan's renewable water that comes from abroad. Armenia 12%, Georgia 8%."),
        ("62%", "Water stress in Armenia, 2022. Azerbaijan 58%, Turkiye 48%, Georgia 4%."),
        ("112 km³", "Turkiye's reservoir storage, 2010. Ataturk alone holds 48.7."),
        ("0%", "Georgia's shared basin area under an operational agreement, 2023."),
    ]
    chart = bars([(c, v, "#9fc2d6", v * s) for c, v, s in PER_CAPITA],
                 fmt=lambda v: f"{v:,.0f}", width=300)
    leg = (f'<span class="lg">{sw_box("#9fc2d6")}total</span>'
           f'<span class="lg">{sw_box(PALETTE["ink"])}from abroad</span>')
    head = headrow(tile_rows, "Renewable water per person, m³ a year", chart, leg)
    body = "".join(
        f"<tr><td><b>{esc(r[0])}</b></td>"
        + "".join(f"<td class='n'>{esc(v)}</td>" for v in r[1:8])
        + f"<td class='small'>{esc(r[8])}</td></tr>"
        for r in WATER_ROWS
    )
    table = ('<table class="kt"><thead><tr><th>Country</th><th class="n">Renewable, km³</th>'
             '<th class="n">From abroad</th><th class="n">Withdrawal, km³</th>'
             '<th class="n">Farms</th><th class="n">Water stress</th>'
             '<th class="n">Dam storage, km³</th><th class="n">Irrigated, 1000 ha</th>'
             f'<th>Position</th></tr></thead><tbody>{body}</tbody></table>')
    note = ('<p class="note">FAO AQUASTAT. Water stress is SDG 6.4.2, 2022. Storage and '
            "irrigated area come from old surveys, year in brackets. Turkiye irrigated area is "
            "the official 2022 figure.</p>")
    return head + table + note


# ---------------------------------------------------------------------------
# 3. background
# ---------------------------------------------------------------------------
TIMELINE = [
    ("Jan 1927", "Kars protocol splits the border rivers 50/50 between the USSR and Turkey.",
     "Climate Diplomacy"),
    ("1933 on", "Soviet drawdown of Lake Sevan. It falls from 58.5 to about 33 km³.",
     "World Bank"),
    ("1974", "Keban, the first large Euphrates dam in Turkiye.", "DSI"),
    ("1980", "Turkiye and the USSR complete the joint Akhurian reservoir.", "Climate Diplomacy"),
    ("1987 to 1990", "Turkiye promises Syria 500 m³/s. Syria and Iraq split it 42/58. "
                     "Ataturk fills in 1990.", "ESCWA"),
    ("Jan 2014", "Georgia and Azerbaijan finalise a Kura agreement. It is never signed.",
     "UNECE, OSCE"),
    ("Jul 2019", "Ilisu starts filling on the Tigris. Hasankeyf goes under.", "DSI"),
    ("Jun 2021", "Drought in Armenia. Sevan releases rise from 170 to 245 million m³.",
     "Azatutyun"),
    ("Nov 2022", "Yusufeli opens on the Coruh, upstream of Georgia.", "DSI"),
    ("Mar 2023", "Lower Kura at a record low. Sea water pushes far upstream.", "JAMnews"),
    ("Apr 2024", "Turkiye and Iraq sign a water framework agreement.", "Anadolu"),
    ("May 2024", "Azerbaijan and Iran open Khudafarin and Giz Galasi on the Araks.",
     "President.az"),
    ("2025", "Caspian at a record low. Istanbul reservoirs at 20% in November.", "ISKI"),
    ("Sep 2026", "Iraq starts paying for Turkish water works with oil.", "The Arab Weekly"),
]


# ---------------------------------------------------------------------------
# 4. the region on the ground
# ---------------------------------------------------------------------------
RIVER_LABELS = [
    (45.3, 41.45, "Kura"), (45.6, 39.35, "Araks"), (41.15, 40.55, "Coruh"),
    (39.3, 38.55, "Euphrates"), (41.7, 37.75, "Tigris"), (42.1, 42.6, "Rioni"),
    (34.4, 40.85, "Kizilirmak"), (30.4, 40.05, "Sakarya"),
]
LAKE_LABELS = [
    (45.75, 40.75, "Sevan"), (42.9, 38.55, "Van"), (45.55, 37.6, "Urmia"),
    (47.2, 41.2, "Mingachevir"), (38.6, 37.25, "Ataturk"), (39.4, 39.0, "Keban"),
]
DAM_SWATCH = ('<svg class="lsw" width="14" height="12"><path d="M7,1L13,11L1,11Z" '
              f'fill="{DAM}"/></svg>')


def ground(index) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index)
    m.add('<g class="zoomable">')
    draw_rivers(m)
    draw_lakes(m)
    draw_dams(m)
    label_countries(m, size=9)
    for lon, lat, name in RIVER_LABELS:
        m.label(lon, lat, name, size=10, colour=RIVER, weight="600", halo=True)
    for lon, lat, name in LAKE_LABELS:
        m.label(lon, lat, name, size=9.5, colour=PALETTE["ink"], weight="600", halo=True)
    m.add("</g>")
    legend = html_legend([
        ("Water", [(sw_line(RIVER, 2), "river"), (sw_box(LAKE), "lake"),
                   (sw_box(RESERVOIR), "reservoir")]),
        ("Dams", [(DAM_SWATCH, "hydro, 100 MW or more, size by MW")]),
    ])
    cap = ("Rivers and lakes: Natural Earth. Dams: GEM plant tracker, hover for name and MW. "
           "Small reservoirs are not drawn.")
    return imap(svg_of(m), cap, legend)


# ---------------------------------------------------------------------------
# 5. basins and transboundary flows
# ---------------------------------------------------------------------------
BASIN_LABELS = [
    (46.6, 39.75, "KURA AND ARAKS", "About 190,000 km². Five countries."),
    (40.9, 40.1, "CORUH", "Turkiye, then Georgia"),
    (38.3, 38.15, "EUPHRATES", "Turkiye, Syria, Iraq"),
    (40.8, 37.55, "TIGRIS", "Turkiye, Syria, Iraq, Iran"),
    (42.85, 39.1, "LAKE VAN", "closed basin"),
    (46.3, 36.85, "LAKE URMIA", "closed basin"),
]
# lon, lat, downstream angle in degrees, label, anchor, label dx, dy
FLOWS = [
    (41.68, 41.47, 80, "Coruh 6.3", "end", -0.25, 0.1),
    (42.95, 41.5, 60, "Kura 0.9", "end", -0.15, -0.35),
    (44.62, 41.17, 90, "Debed 0.9", "end", -0.22, 0.05),
    (45.4, 41.25, -15, "From Georgia 11.9", "start", 0.3, 0.12),
    (46.2, 39.05, 20, "From Iran 7.5", "start", 0.3, -0.25),
    (45.55, 40.75, 10, "From Armenia 6.0", "start", 0.3, 0.08),
    (38.05, 36.85, -90, "Euphrates, 500 m³/s floor", "start", 0.25, -0.2),
    (42.2, 37.15, -60, "Tigris", "start", 0.25, -0.15),
]
FLOW_SWATCH = ('<svg class="lsw" width="22" height="10"><line x1="1" y1="5" x2="15" y2="5" '
               f'stroke="{PALETTE["ink"]}" stroke-width="2.6"/><path d="M14,1L21,5L14,9Z" '
               f'fill="{PALETTE["ink"]}"/></svg>')


def structure(index) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index, highlight=set())
    m.add('<g class="zoomable">')
    draw_basins(m)
    draw_borders(m, index)
    draw_rivers(m, colour=RIVER, scale=0.9, max_rank=10, min_px=2.0)
    draw_lakes(m, reservoirs=False)
    for lon, lat, head, sub in BASIN_LABELS:
        m.label(lon, lat, head, size=10.5, colour=PALETTE["ink"], weight="700", halo=True)
        m.label(lon, lat - 0.28, sub, size=9, colour=MUT, weight="500", halo=True)
    for lon, lat, ang, lab, anchor, dx, dy in FLOWS:
        flow_arrow(m, lon, lat, ang, lab, anchor, dx, dy)
    m.add("</g>")
    legend = html_legend([
        ("Shared basins", [(sw_box(BASINS[2040070050][1]), "Kura and Araks"),
                           (sw_box(BASINS[2040005120][1]), "Coruh"),
                           (sw_box(BASINS[2040785900][1]), "Euphrates and Tigris"),
                           (sw_box("#dcd6cb"), "closed lake basin"),
                           (sw_box(OTHER_BASIN), "other basin")]),
        ("Flows", [(FLOW_SWATCH, "inflow across a border, km³ a year")]),
    ])
    cap = ("Basins: HydroBASINS level 4. Inflows: FAO AQUASTAT. Inflows to Azerbaijan from "
           "Iran and Armenia sum several rivers. Arrows placed by hand.")
    return imap(svg_of(m), cap, legend)


# ---------------------------------------------------------------------------
# 6. projects and agreements
# ---------------------------------------------------------------------------
PROJECTS = [
    {"name": "Vedi reservoir", "map": "Vedi", "status": "committed", "point": (44.75, 39.95),
     "label": (44.25, 39.6, "end"), "scale": "30 million m³, 6,000 ha", "year": "n/a",
     "corridor": "Araks", "link": "Ararat valley irrigation, Armenia",
     "stage": "Opened May 2025, filling. AFD EUR 75m, EU EUR 10m.",
     "stake": "The only one of 17 reservoirs pledged for 2021 to 2026 delivered."},
    {"name": "Kaps reservoir", "map": "Kaps", "status": "committed", "point": (43.75, 40.95),
     "label": (43.35, 41.25, "end"), "scale": "Redesign to 60 million m³", "year": "n/a",
     "corridor": "Araks", "link": "Akhurian basin, Shirak, Armenia",
     "stage": "KfW EUR 68.5m. Contract terminated Dec 2025. Re-tender.",
     "stake": "Irrigation for Shirak and water for Gyumri."},
    {"name": "Sevan transfer tunnels repair", "map": "Sevan tunnels", "status": "committed",
     "anchors": [(45.75, 39.6), (45.35, 39.75), (45.25, 40.2)],
     "label": (45.9, 39.95, "start"), "scale": "AMD 1.6bn", "year": "n/a",
     "corridor": "Araks", "link": "Vorotan and Arpa water into Lake Sevan",
     "stage": "Government programme.",
     "stake": "Sevan recovery against Ararat valley irrigation."},
    {"name": "Georgia and Azerbaijan Kura agreement", "map": "Kura agreement",
     "status": "planned", "point": (47.05, 40.78), "label": (47.7, 41.65, "start"),
     "scale": "Whole Kura basin", "year": "n/a", "corridor": "Kura",
     "link": "With the EU funded basin plan of 2021",
     "stage": "Text final since 2014, unsigned. Plan not adopted.",
     "stake": "The only route to a legal Kura regime."},
    {"name": "Georgia irrigation and land market, GRAIL", "map": "GRAIL", "status": "committed",
     "point": (45.5, 41.7), "label": (45.9, 42.1, "start"), "scale": "USD 75m",
     "year": "n/a", "corridor": "Kura", "link": "Soviet era canals in eastern Georgia",
     "stage": "World Bank, approved 2023.",
     "stake": "Irrigated farming in Kakheti and Kvemo Kartli."},
    {"name": "Water and Irrigation Services Enhancement", "map": None, "status": "committed",
     "scale": "USD 185m, of a 435m programme", "year": "n/a", "corridor": "Armenia, national",
     "link": "650,000 people", "stage": "World Bank, approved Jun 2025.",
     "stake": "Losses near 75% in water supply."},
    {"name": "Armenia water reservoirs, phase I", "map": None, "status": "planned",
     "scale": "EUR 63m", "year": "n/a", "corridor": "Armenia, national",
     "link": "Several small reservoirs", "stage": "EBRD. Board Nov 2026.",
     "stake": "Storage for a country short of it."},
    {"name": "Absheron desalination", "map": "Absheron desalination", "status": "committed",
     "point": (49.9, 40.5), "label": (50.7, 39.35, "middle"), "scale": "300,000 m³/day",
     "year": "2027", "corridor": "Caspian coast", "link": "Baku water from the Caspian",
     "stage": "ACWA Power and IC Ictas, contract Dec 2025.",
     "stake": "Baku supply as the Kura shrinks."},
    {"name": "Silvan dam", "map": "Silvan", "status": "committed", "point": (41.0, 38.1),
     "label": (40.6, 38.45, "end"), "scale": "7.3 km³, 245,000 ha", "year": "n/a",
     "corridor": "Tigris", "link": "GAP, on a Tigris tributary",
     "stage": "DSI. Commissioning date unclear.",
     "stake": "More Tigris water used upstream of Iraq."},
    {"name": "Turkiye and Iraq oil for water", "map": "Oil for water", "status": "committed",
     "point": (43.1, 36.6), "label": (43.5, 36.25, "start"), "scale": "Six dams in Iraq",
     "year": "2026", "corridor": "Tigris", "link": "Turkish works in Iraq paid in oil",
     "stage": "Signed Nov 2025, in force Sep 2026.",
     "stake": "Iraq's water tied to Turkish releases."},
    {"name": "Flood and drought management", "map": "Flood and drought", "status": "committed",
     "point": (35.5, 37.3), "label": (35.1, 37.75, "end"), "scale": "USD 600m",
     "year": "n/a", "corridor": "Turkiye, national", "link": "DSI river basins",
     "stage": "World Bank, approved Jun 2024.",
     "stake": "The 2025 drought hit Istanbul and Izmir."},
]


def projects(index) -> str:
    cap = ("Points placed by hand. The Sevan tunnels are drawn schematically. Two national "
           "programmes are in the table only.")

    def background(m):
        draw_rivers(m, colour="#a9c9de", scale=0.8, max_rank=10, min_px=2.5)
        draw_lakes(m, reservoirs=False)

    return projects_map(index, PROJECTS, cap, background)


# ---------------------------------------------------------------------------
# pane
# ---------------------------------------------------------------------------
def pane(index: CountryIndex, heads) -> str:
    primer_title, map_b, map_c = heads
    return "".join([
        primer_block(primer_title, primer()),
        '<h4 class="mh">Key figures</h4>', key_figures(),
        '<h4 class="mh">Background</h4>', timeline(TIMELINE),
        '<h4 class="mh">The region on the ground</h4>', ground(index),
        f'<h4 class="mh">{esc(map_b)}</h4>', structure(index),
        f'<h4 class="mh">{esc(map_c)}</h4>', projects(index),
    ])
