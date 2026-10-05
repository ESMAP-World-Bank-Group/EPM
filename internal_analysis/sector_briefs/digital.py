"""Digital pane of the Black Sea sector briefs.

Condensed from the digital connectivity screening note. Reads the datasets that
digital_connectivity/fetch_data.py downloads: submarine cables and landings
(submarinecablemap.com), terrestrial fibre (ITU BBmaps), exchanges and facilities
(PeeringDB) and ICT indicators (World Bank WDI). Projects are placed by hand.
"""

from __future__ import annotations

import math

from mapkit import DIGITAL_DATA, FOCUS, PALETTE, CountryIndex, Map, lines_of, load_json
from power import (BBOX, H, W, esc, html_legend, imap, label_countries, svg_of,
                   sw_box, sw_dot, sw_line)
from transport import bars, batch, bullets, chain_svg, headrow, primer_block, projects_map, \
    timeline

DC = DIGITAL_DATA
CABLE = PALETTE["ink"]
FIBRE = "#5b9fcc"
FAC = "#37474f"
_CACHE = {}


def data(name):
    if name not in _CACHE:
        _CACHE[name] = load_json(DC / name)
    return _CACHE[name]


# PeeringDB country codes to basemap names.
PDB = {"BG": "Bulgaria", "RO": "Romania", "TR": "Turkey", "GE": "Georgia", "AM": "Armenia",
       "AZ": "Azerbaijan", "MD": "Moldova", "UA": "Ukraine", "KZ": "Kazakhstan",
       "GR": "Greece"}
IX_AT = {"sofia": (23.32, 42.70), "bucharest": (26.10, 44.44), "istanbul": (28.98, 41.01),
         "ankara": (32.86, 39.93), "varna": (27.92, 43.21), "braila": (27.96, 45.27),
         "craiova": (23.80, 44.33), "tbilisi": (44.79, 41.72), "yerevan": (44.51, 40.18),
         "chisinau": (28.86, 47.01), "thessaloniki": (22.94, 40.64), "aqtau": (51.17, 43.65),
         "odessa": (30.72, 46.48)}
BAKU = (49.87, 40.41)
BELOW = {"yerevan"}  # label under the bubble, clear of the country name


def city(value: str) -> str:
    """PeeringDB lists every site of a multi-city exchange. Keep the first."""
    return value.split(",")[0].split("/")[0].strip().lower().replace("i̇", "i")


def largest_ix():
    """Networks at the largest exchange in each country. Summing members across
    exchanges would count a network once per exchange it joins."""
    out = {}
    for code, rows in data("peeringdb.json")["ix"].items():
        if code in PDB:
            out[PDB[code]] = max((ix.get("net_count") or 0 for ix in rows), default=0)
    return out


def ix_cities():
    best = {}
    for code, rows in data("peeringdb.json")["ix"].items():
        for ix in rows:
            name = city(ix.get("city") or "")
            if name in IX_AT and (ix.get("net_count") or 0) > best.get(name, 0):
                best[name] = ix["net_count"]
    return best


def landings_by_country():
    out = {}
    for f in data("submarine_landings.geojson")["features"]:
        country = f["properties"]["name"].rsplit(",", 1)[-1].strip()
        out[country] = out.get(country, 0) + 1
    return out


def bssc_route():
    """The BSSC has no published alignment. It is traced on the surveyed
    Caucasus Cable System crossing, with the landfalls moved to Anaklia and
    Constanta, as in the screening note."""
    for f in data("submarine_cables.geojson")["features"]:
        if f["properties"]["name"] == "Caucasus Cable System":
            line = [tuple(p) for p in max(lines_of(f["geometry"]), key=len)]
            if line[0][0] < line[-1][0]:
                line.reverse()
            return [(41.573, 42.395)] + line[1:-1] + [(28.75, 43.90), (28.66, 44.17)]
    return []


def cable_line(name):
    for f in data("submarine_cables.geojson")["features"]:
        if name in f["properties"]["name"]:
            return [tuple(p) for p in max(lines_of(f["geometry"]), key=len)]
    return []


def draw_cables(m: Map, colour=CABLE, width=2.0, skip=("Trans-Caspian",), opacity=1.0):
    for f in data("submarine_cables.geojson")["features"]:
        if any(s in f["properties"]["name"] for s in skip):
            continue
        for line in lines_of(f["geometry"]):
            d = m.path(line, min_px=0.8)
            if d:
                m.add(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}" '
                      f'stroke-opacity="{opacity}" stroke-linecap="round"><title>{esc(f["properties"]["name"])}</title>'
                      "</path>")


def draw_fibre(m: Map, status="Operational", colour=FIBRE, width=0.7, nodes=True):
    lines = [line for f in data("itu_fibre.geojson")["features"]
             if (f["properties"].get("status") or "") == status
             for line in lines_of(f["geometry"])]
    batch(m, lines, f'stroke="{colour}" stroke-width="{width}" stroke-opacity="0.75"', 0.5)
    if nodes:
        pts = {(round(p[0], 3), round(p[1], 3)) for line in lines for p in (line[0], line[-1])}
        for lon, lat in pts:
            x, y = m.xy(lon, lat)
            m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.1" fill="{colour}" fill-opacity="0.8"/>')


def draw_landings(m: Map):
    for f in data("submarine_landings.geojson")["features"]:
        lon, lat = f["geometry"]["coordinates"]
        x, y = m.xy(lon, lat)
        m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="#fff" stroke="{CABLE}" '
              f'stroke-width="1.6"><title>{esc(f["properties"]["name"])}</title></circle>')


def draw_facilities(m: Map):
    for rows in data("peeringdb.json")["fac"].values():
        for fac in rows:
            lat, lon = fac.get("latitude"), fac.get("longitude")
            if lat is None or lon is None:
                continue
            if not (BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]):
                continue
            x, y = m.xy(lon, lat)
            m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.3" fill="{FAC}" fill-opacity="0.6">'
                  f'<title>{esc(fac.get("name") or "")}</title></circle>')


# ---------------------------------------------------------------------------
# 1. primer
# ---------------------------------------------------------------------------
def primer() -> str:
    svg = chain_svg(
        [("Submarine cable", "Between countries"), ("Landing station", "Where it comes ashore"),
         ("Backbone", "Along road, rail, pipe"), ("Exchange", "Where networks meet"),
         ("Access", "Homes and masts")],
        notes=[(712, 40, "Each stage has its", None), (712, 54, "own owners.", None)],
        mid_fill=("Exchange",))
    return f'<div class="primer-fig">{svg}</div>' + bullets([
        ("Capacity is rarely the limit.",
         "On an existing route, more fibre pairs or wavelengths are cheap. Rights of way and "
         "landings cost the most."),
        ("Exchanges set the price.",
         "Traffic that cannot meet at a local exchange pays for foreign transit. Tbilisi's "
         "exchange has 14 member networks, Sofia's largest has 134."),
        ("Fibre rides other corridors.",
         "SOCAR Fiber runs in the TANAP gas trench. The BSSC power cable will carry fibre. "
         "TRIPP plans a fibre duct beside the rail."),
        ("Data centres are power load.",
         "Romania's announced AI gigafactory would draw about 1.5 GW."),
    ])


# ---------------------------------------------------------------------------
# 2. key figures
# ---------------------------------------------------------------------------
TABLE_COUNTRIES = [("Turkiye", "Turkey", "TR", "Several cables, exchanges in Istanbul."),
                   ("Georgia", "Georgia", "GE", "One useful cable, Poti, from 2008."),
                   ("Armenia", "Armenia", "AM", "Landlocked. Exits through Georgia and Iran."),
                   ("Azerbaijan", "Azerbaijan", "AZ", "Caspian gateway. No public exchange."),
                   ("Romania", "Romania", "RO", "Comparator, EU shore."),
                   ("Bulgaria", "Bulgaria", "BG", "Comparator, EU shore.")]


def key_figures() -> str:
    tile_rows = [
        ("1 cable", "Direct link from the Caucasus to the EU: Poti to Balchik, 2008, 12.6 Tbps."),
        ("14 against 134", "Networks at Tbilisi's exchange against Sofia's largest."),
        ("0", "Public exchanges listed in Azerbaijan. It has 7 interconnection facilities."),
        ("400 Tbps", "Trans-Caspian cable, Sumgait to Aktau. Service from Q3 2026."),
    ]
    nets = largest_ix()
    order = ["Bulgaria", "Romania", "Turkey", "Armenia", "Georgia", "Azerbaijan"]
    shown = {"Turkey": "Turkiye"}
    chart = bars([(shown.get(c, c), nets.get(c, 0),
                   "#9fc2d6" if c in FOCUS or c == "Turkey" else "#d5dde3", None)
                  for c in order], fmt=lambda v: f"{v:,.0f}", width=300)
    head = headrow(tile_rows, "Networks at the largest exchange", chart)

    wdi = data("wdi_ict.json")
    pdb = data("peeringdb.json")
    land = landings_by_country()

    def ind(key, name):
        v = wdi[key]["values"].get(name)
        return f"{v[1]:.0f} ({v[0]})" if v else "n/a"

    body = "".join(
        f"<tr><td><b>{esc(c)}</b></td><td class='n'>{ind('IT.NET.USER.ZS', c)}</td>"
        f"<td class='n'>{ind('IT.NET.BBND.P2', c)}</td>"
        f"<td class='n'>{land.get(base, 0)}</td>"
        f"<td class='n'>{sum(1 for ix in pdb['ix'].get(cc, []) if ix.get('net_count'))}</td>"
        f"<td class='n'>{nets.get(base, 0)}</td>"
        f"<td class='n'>{len(pdb['fac'].get(cc, []))}</td>"
        f"<td class='small'>{esc(role)}</td></tr>"
        for c, base, cc, role in TABLE_COUNTRIES
    )
    table = ('<table class="kt"><thead><tr><th>Country</th><th class="n">Internet users, %</th>'
             '<th class="n">Fixed broadband per 100</th><th class="n">Cable landings</th>'
             '<th class="n">Exchanges</th><th class="n">Networks at largest</th>'
             '<th class="n">Facilities</th><th>Position</th></tr></thead>'
             f"<tbody>{body}</tbody></table>")
    note = ('<p class="note">Users and broadband: World Bank WDI. Landings: '
            "submarinecablemap.com, inside the map frame. Exchanges with at least one member "
            "network, and facilities: PeeringDB, self reported, September 2026.</p>")
    return head + table + note


# ---------------------------------------------------------------------------
# 3. background
# ---------------------------------------------------------------------------
TIMELINE = [
    ("2008", "Caucasus Cable System opens, Poti to Balchik. Still the only direct link to "
             "the EU.", "Caucasus Online"),
    ("2013", "SOCAR Fiber starts laying fibre along the TANAP gas route across Turkiye.",
     "SOCAR Fiber"),
    ("Oct 2020", "Georgian regulator puts a special manager in Caucasus Online. Arbitration "
                 "follows.", "ComCom"),
    ("Jul 2024", "EXA and SOCAR Fiber agree a land route from Greece to Georgia.", "EXA"),
    ("Dec 2025", "EU lists the BSSC as a Project of Mutual Interest, power and fibre "
                 "together.", "European Commission"),
    ("Feb 2026", "Transelectrica and GSE sign a memorandum on the BSSC.", "Transelectrica"),
    ("Jun 2026", "AzerTelecom and Telecom Armenia sign reciprocal transit. A first since "
                 "1991.", "AzerTelecom"),
    ("Jun 2026", "EU launches its Connectivity Agenda Platform, up to EUR 2bn.",
     "European Commission"),
    ("Q3 2026", "Trans-Caspian cable enters commercial service.", "AzerTelecom"),
]


# ---------------------------------------------------------------------------
# 4. the region on the ground
# ---------------------------------------------------------------------------
LANDING_LABELS = [(41.85, 42.0, "Poti", "start"), (27.95, 43.55, "Balchik", "end"),
                  (49.85, 40.95, "Sumgait", "start"), (51.0, 43.85, "Aktau", "end"),
                  (29.1, 41.35, "Istanbul", "start"), (28.75, 43.7, "Mangalia", "start")]


def ground(index) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index)
    m.add('<g class="zoomable">')
    draw_fibre(m)
    draw_cables(m, width=2.6, skip=())
    draw_landings(m)
    draw_facilities(m)
    label_countries(m, size=9)
    for lon, lat, name, anchor in LANDING_LABELS:
        m.label(lon, lat, name, size=9.5, colour=CABLE, anchor=anchor, weight="600", halo=True)
    m.add("</g>")
    legend = html_legend([
        ("Lines", [(sw_line(CABLE, 3), "submarine cable"),
                   (sw_line(FIBRE, 2), "terrestrial fibre, operational")]),
        ("Points", [(sw_dot(CABLE, ring=True), "cable landing"),
                    (sw_dot(FAC), "interconnection facility")]),
    ])
    cap = ("Cables: submarinecablemap.com. Fibre: ITU BBmaps, a node to node graph, so links "
           "are straight by construction. Facilities: PeeringDB.")
    return imap(svg_of(m), cap, legend)


# ---------------------------------------------------------------------------
# 5. where networks exchange traffic
# ---------------------------------------------------------------------------
SCALE = [(100, "#0277bd", "100 or more"), (50, "#4a9ed1", "50 to 99"),
         (20, "#8fc6e3", "20 to 49"), (1, "#cfe6f2", "1 to 19"),
         (0, "#f2dfa0", "none listed")]


def shade(n):
    for threshold, colour, _ in SCALE:
        if n >= threshold:
            return colour
    return SCALE[-1][1]


def structure(index) -> str:
    nets = largest_ix()
    m = Map(width=W, height=H, bbox=BBOX)
    m.add(f'<rect width="{W}" height="{H}" fill="{PALETTE["sea"]}"/>')
    for name, poly, *_ in index.entries:
        fill = shade(nets[name]) if name in nets else PALETTE["land"]
        d = " ".join(r + "Z" for r in (m.path(ring, min_px=0.5) for ring in poly) if r)
        if d:
            m.add(f'<path d="{d}" fill="{fill}" stroke="{PALETTE["border"]}" '
                  f'stroke-width="0.6" fill-rule="evenodd"/>')
    m.add('<g class="zoomable">')
    draw_cables(m, colour="#9fb3c2", width=1.4, skip=())
    draw_facilities(m)
    label_countries(m, size=9)
    cities = sorted(ix_cities().items(), key=lambda kv: -kv[1])
    for name, n in cities:
        lon, lat = IX_AT[name]
        x, y = m.xy(lon, lat)
        r = 3.0 + math.sqrt(n) * 1.3
        m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#fff" fill-opacity="0.9" '
              f'stroke="{PALETTE["ink"]}" stroke-width="1.8"><title>{name.title()}: {n} '
              "networks</title></circle>")
    for name, n in cities:
        lon, lat = IX_AT[name]
        r = 3.0 + math.sqrt(n) * 1.3
        dy = -(r + 13) if name in BELOW else r + 6
        m.label(lon, lat + dy / m.scale, f"{name.title()} {n}", size=9.5,
                colour=PALETTE["ink"], weight="700", halo=True)
    x, y = m.xy(*BAKU)
    m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="none" stroke="{PALETTE["planned"]}" '
          'stroke-width="2" stroke-dasharray="3 2.5"/>')
    m.label(BAKU[0], BAKU[1] - 16 / m.scale, "Baku, none listed", size=9.5,
            colour=PALETTE["ink"], weight="700", halo=True)
    m.add("</g>")
    legend = html_legend([
        ("Largest exchange", [(sw_box(c), t) for _, c, t in SCALE]
         + [(sw_box(PALETTE["land"]), "not screened")]),
        ("Points", [(sw_dot(PALETTE["ink"], ring=True), "exchange city, size by networks"),
                    (sw_dot(FAC), "facility")]),
    ])
    cap = ("Shading is the member count of the largest exchange in each country. PeeringDB, "
           "self reported. Russia and Iran were not screened.")
    return imap(svg_of(m), cap, legend)


# ---------------------------------------------------------------------------
# 6. projects
# ---------------------------------------------------------------------------
def projects_list():
    return [
        {"name": "Trans-Caspian fibre, Digital Silk Way", "map": "Trans-Caspian",
         "status": "committed", "anchors": cable_line("Trans-Caspian"), "dash": "10 5",
         "label": (49.4, 42.6, "end"), "scale": "400 Tbps, 380 km", "year": "2026",
         "corridor": "Caspian", "link": "Sumgait to Aktau",
         "stage": "Laid. AzerTelecom and Kazakhtelecom.",
         "stake": "Closes the middle route east of Baku."},
        {"name": "BSSC fibre", "map": "BSSC fibre", "status": "planned",
         "anchors": bssc_route(), "label": (35.0, 43.6, "middle"), "scale": "40 Tbps",
         "year": "~2030", "corridor": "Black Sea", "link": "In the HVDC cable lay, Anaklia to "
         "Constanta", "stage": "EU PMI Dec 2025. Transelectrica and GSE memorandum Feb 2026.",
         "stake": "Second direct path to the EU. Not valued in the power case."},
        {"name": "TRIPP fibre duct", "map": "TRIPP", "status": "planned", "osm": "tripp",
         "label": (46.3, 38.4, "middle"), "leader": (46.2, 38.88), "scale": "About 43 km", "year": "n/a",
         "corridor": "Aras", "link": "Beside the rail, road, gas and power line",
         "stage": "TRIPP development company set up.",
         "stake": "A southern route for Armenia and Nakhchivan."},
        {"name": "Armenia and Azerbaijan reciprocal transit", "map": None,
         "status": "committed", "scale": "n/a", "year": "2026", "corridor": "Aras",
         "link": "AzerTelecom and Telecom Armenia", "stage": "Signed Jun 2026.",
         "stake": "First Armenian traffic across Azerbaijan since 1991."},
        {"name": "EXA and SOCAR Fiber land route", "map": None, "status": "planned",
         "scale": "n/a", "year": "n/a", "corridor": "Turkiye",
         "link": "Greece, Turkiye, Georgia, toward Iraq",
         "stage": "Agreement Jul 2024. In development.",
         "stake": "Land diversity against Red Sea cable cuts."},
        {"name": "Black Sea AI gigafactory", "map": "AI gigafactory", "status": "planned",
         "point": (28.05, 44.33), "label": (27.75, 44.65, "end"),
         "scale": "About 1.5 GW, up to EUR 5bn", "year": "2028", "corridor": "Romania",
         "link": "Cernavoda and Doicesti", "stage": "Announced.",
         "stake": "A demand block for the power scenarios."},
    ]


def projects(index) -> str:
    cap = ("The BSSC has no published route. It is drawn on the surveyed Caucasus Cable "
           "System crossing, landfalls moved to Anaklia and Constanta. Two agreements are in "
           "the table only.")

    def background(m):
        draw_cables(m, colour="#b9c9d6", width=1.4)

    return projects_map(index, projects_list(), cap, background)


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
