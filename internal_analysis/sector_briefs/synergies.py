"""Comparison and synergies pane of the Black Sea sector briefs.

Reads nothing new. The ground map stacks the layers of the sector panes, the
corridor matrix is written by hand from those panes, and the project counts are
taken from the project lists of each pane, so they move when a pane changes.
"""

from __future__ import annotations

import digital
import gas
import power
import transport
import water
from mapkit import DIGITAL_DATA, FOCUS, PALETTE, CountryIndex, Map, lines_of, load_json
from power import (BBOX, GRID, H, W, esc, html_legend, imap, label_countries, svg_of,
                   sw_box, sw_line, voltage_of)
from transport import (batch, bars, bullets, headrow, primer_block, projects_map,
                       timeline)

SERVICE = "#0277bd"
STATUS_FILL = {"service": SERVICE, "committed": PALETTE["building"],
               "planned": PALETTE["planned"], None: "#ffffff"}
STATUS_WORD = {"service": "In service", "committed": "Committed", "planned": "Planned",
               None: "n/a"}
SECTORS = [("P", "Power"), ("G", "Gas"), ("T", "Transport"), ("D", "Digital"),
           ("W", "Water")]
MUT = "#6f6a61"
# Ground map only: the gas tab brown sits too close to the grid gold once stacked.
PIPE = "#7b3f1a"
RAIL = "#9aa3a8"


# ---------------------------------------------------------------------------
# 1. primer
# ---------------------------------------------------------------------------
def primer_svg() -> str:
    ink, pipe, cab = PALETTE["ink"], gas.PIPE, digital.CABLE
    g = 118
    o = ['<svg viewBox="0 0 860 205" width="100%" font-family="Segoe UI, Arial, sans-serif">',
         f'<rect x="10" y="{g}" width="590" height="52" fill="#efe6d2"/>',
         f'<line x1="10" y1="{g}" x2="600" y2="{g}" stroke="#a89a7c" stroke-width="1.5"/>']
    # power line: lattice tower, conductors, earth wire on top
    o.append(f'<path d="M62,{g}L82,26L102,{g}M67,{g - 20}L97,{g - 20}M72,{g - 50}L92,{g - 50}" '
             f'fill="none" stroke="{GRID}" stroke-width="2"/>')
    o.append(f'<line x1="46" y1="52" x2="118" y2="52" stroke="{GRID}" stroke-width="2"/>')
    o.append(f'<line x1="30" y1="26" x2="134" y2="26" stroke="{cab}" stroke-width="1.8" '
             'stroke-dasharray="4 2"/>')
    o.append(f'<text x="140" y="30" font-size="10.5" fill="{cab}">fibre in the earth wire</text>')
    # rail on ballast, road
    o.append(f'<path d="M200,{g}L214,{g - 12}L266,{g - 12}L280,{g}Z" fill="#c9c2b4"/>')
    o.append(f'<rect x="216" y="{g - 16}" width="48" height="4" fill="{ink}"/>')
    o.append(f'<rect x="330" y="{g - 7}" width="110" height="7" fill="#9e9e9e"/>')
    o.append(f'<line x1="340" y1="{g - 3.5}" x2="430" y2="{g - 3.5}" stroke="#fff" '
             'stroke-width="1" stroke-dasharray="8 6"/>')
    # trench with gas pipe and fibre duct
    o.append(f'<path d="M478,{g}L486,{g + 44}L566,{g + 44}L574,{g}" fill="#e2d6bb" '
             'stroke="#a89a7c" stroke-dasharray="3 2"/>')
    o.append(f'<circle cx="512" cy="{g + 28}" r="13" fill="{pipe}"/>')
    o.append(f'<circle cx="548" cy="{g + 34}" r="5" fill="{cab}"/>')
    for x, text in [(82, "Power line"), (240, "Rail"), (385, "Road"),
                    (526, "Gas pipe and fibre duct")]:
        o.append(f'<text x="{x}" y="{g + 72}" text-anchor="middle" font-size="12" '
                 f'font-weight="700" fill="{ink}">{esc(text)}</text>')
    for i, line in enumerate(["One right of way.", "Land, permits and border",
                              "agreements are settled once.", "",
                              "TRIPP plans all five on", "about 43 km."]):
        o.append(f'<text x="640" y="{50 + i * 17}" font-size="11.5" fill="{MUT}">'
                 f'{esc(line)}</text>')
    o.append("</svg>")
    return "".join(o)


def primer() -> str:
    return f'<div class="primer-fig">{primer_svg()}</div>' + bullets([
        ("The slow part is the corridor.",
         "Land, permits and border agreements take longer than the build. A second network "
         "on a settled route skips most of it."),
        ("Fibre rides the others.",
         "In the earth wire of power lines, in pipeline and rail trenches, in subsea power "
         "cables. SOCAR Fiber runs in the TANAP trench. The BSSC will carry fibre."),
        ("Loads link sectors.",
         "Electric trains, data centres and desalination are power demand. Romania's AI "
         "gigafactory alone would draw about 1.5 GW."),
        ("Dams serve water and power.",
         "Hydropower release rules set the flow that farms and neighbours get downstream."),
        ("The same few crossings.",
         "Most projects in the four tabs use six corridors. They are mapped below."),
    ])


# ---------------------------------------------------------------------------
# corridor matrix, written from the sector panes
# ---------------------------------------------------------------------------
# path: schematic, through the main nodes. box: label block position (lon, lat, anchor).
CORRIDORS = [
    {"name": "Black Sea crossing",
     "path": [(41.6, 42.3), (38.0, 42.95), (34.0, 43.45), (30.5, 43.95), (28.7, 44.15)],
     "box": (33.0, 44.85, "middle"),
     "P": ("planned", "BSSC 1,300 MW, then GEC"),
     "G": (None, "No pipeline"),
     "T": ("service", "Ro-Ro ferries from Poti and Batumi"),
     "D": ("service", "Caucasus Cable System since 2008. BSSC fibre planned"),
     "W": (None, "n/a")},
    {"name": "Georgian east-west axis",
     "path": [(49.85, 40.4), (47.6, 40.65), (46.36, 40.68), (44.8, 41.7), (43.48, 41.4),
              (43.1, 40.6), (41.27, 39.9)],
     "box": (44.6, 43.75, "middle"),
     "P": ("service", "500 kV Azerbaijan to Georgia, back to back to Turkiye"),
     "G": ("service", "SCP and TANAP, beside the BTC oil line"),
     "T": ("service", "BTK railway since 2017, East to West Highway"),
     "D": ("service", "Terrestrial fibre along road and rail"),
     "W": ("planned", "Kura shared, agreement never signed")},
    {"name": "Caspian crossing",
     "path": [(49.85, 40.4), (50.6, 41.6), (51.17, 43.65)],
     "box": (52.5, 45.2, "end"),
     "P": ("planned", "Trans-Caspian HVDC 1,000 MW"),
     "G": ("planned", "Trans-Caspian Gas Pipeline, no FID"),
     "T": ("service", "Rail ferries Alat to Aktau and Kuryk"),
     "D": ("committed", "Trans-Caspian fibre, Q3 2026"),
     "W": ("committed", "Absheron desalination on the Caspian")},
    {"name": "Aras valley and Nakhchivan",
     "path": [(47.0, 39.45), (46.24, 38.9), (45.6, 38.95), (45.41, 39.21), (44.04, 39.92),
              (43.1, 40.6)],
     "box": (50.2, 37.45, "end"),
     "P": ("committed", "TRIPP 330 kV to Nakhchivan, then Igdir"),
     "G": ("planned", "TRIPP pipeline"),
     "T": ("committed", "Horadiz to Aghband, Kars to Dilucu, TRIPP rail"),
     "D": ("planned", "TRIPP fibre duct"),
     "W": ("service", "Khudafarin and Giz Galasi dams, 2024")},
    {"name": "Thrace to the EU",
     "path": [(28.98, 41.01), (27.6, 41.4), (26.56, 41.68), (24.75, 42.15)],
     "box": (25.6, 39.75, "middle"),
     "P": ("service", "Lines to Bulgaria and Greece. EWTC planned"),
     "G": ("service", "TurkStream, TANAP into TAP, IGB"),
     "T": ("committed", "Halkali to Kapikule railway"),
     "D": ("service", "KAFOS to Romania, land fibre to Sofia"),
     "W": (None, "n/a")},
    {"name": "North to south through Armenia",
     "path": [(44.8, 41.7), (44.6, 41.0), (44.51, 40.18), (45.3, 39.6), (46.24, 38.9),
              (46.29, 38.08)],
     "box": (43.6, 37.55, "end"),
     "P": ("committed", "AGIR to Iran. CTN to Georgia planned"),
     "G": ("service", "From Russia via Georgia, from Iran"),
     "T": ("committed", "North to South Road, Sisian to Kajaran"),
     "D": ("service", "Exits through Georgia and Iran"),
     "W": (None, "n/a")},
]


def present(c) -> int:
    return sum(1 for k, _ in SECTORS if c[k][0])


# ---------------------------------------------------------------------------
# 2. key figures
# ---------------------------------------------------------------------------
def pane_projects():
    return {"Power": power.PROJECTS, "Gas": gas.PROJECTS, "Transport": transport.PROJECTS,
            "Digital": digital.projects_list(), "Water": water.PROJECTS}


COMPARE = [
    ("Power", "AC lines, back to back stations, subsea HVDC",
     "Three synchronous systems meet around Georgia", "TSOs and regulators"),
    ("Gas", "Trunk pipelines, LNG into Turkiye",
     "TANAP and TAP size, no expansion decision", "SOCAR, BOTAS, shippers"),
    ("Transport", "Rail, road, ferries, ports",
     "Caspian crossing, gauge breaks, border waits", "Railways, port operators, customs"),
    ("Digital", "Subsea and land fibre, exchanges",
     "One direct cable to the EU, small exchanges", "Private carriers"),
    ("Water", "Rivers shared upstream and downstream",
     "Few basin agreements in force", "Ministries, joint commissions"),
]


def key_figures() -> str:
    lists = pane_projects()
    total = sum(len(v) for v in lists.values())
    committed = sum(1 for v in lists.values() for p in v if p["status"] == "committed")
    full = sum(1 for c in CORRIDORS if present(c) == 5)
    tile_rows = [
        (f"{full} of 6", "Corridors that carry all five sectors, built or planned."),
        (f"{total} projects", f"Listed across the four tabs. {committed} committed."),
        ("5 networks", "Rail, road, power, gas and fibre planned in the TRIPP strip."),
        ("3 at Anaklia", "Deep sea port, BSSC landfall and its fibre on one site."),
    ]
    chart = bars([(k, len(v), "#9fc2d6", sum(1 for p in v if p["status"] == "committed"))
                  for k, v in lists.items()], fmt=lambda v: f"{v:,.0f}", width=300)
    head = headrow(tile_rows, "Projects listed per tab, committed in dark", chart)
    body = "".join(
        f"<tr><td><b>{esc(s)}</b></td><td>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td>"
        f"<td class='n'>{sum(1 for p in lists[s] if p['status'] == 'committed')}</td>"
        f"<td class='n'>{sum(1 for p in lists[s] if p['status'] == 'planned')}</td></tr>"
        for s, a, b, c in COMPARE)
    table = ('<table class="kt"><thead><tr><th>Sector</th><th>Crosses borders as</th>'
             '<th>Binding constraint</th><th>Who decides</th><th class="n">Committed</th>'
             f'<th class="n">Planned</th></tr></thead><tbody>{body}</tbody></table>')
    note = ('<p class="note">Counts are the project tables of each tab. A project with '
            "several legs counts once per row in its tab.</p>")
    return head + table + note


# ---------------------------------------------------------------------------
# 3. background
# ---------------------------------------------------------------------------
TIMELINE = [
    ("2006", "BTC oil pipeline enters service. SCP gas follows in the same corridor.",
     "Energy"),
    ("2008", "Caucasus Cable System opens, Poti to Balchik.", "Digital"),
    ("Oct 2017", "Baku to Tbilisi to Kars railway opens, along the pipeline corridor.",
     "Transport"),
    ("Jun 2018", "TANAP opens. SOCAR Fiber runs in its trench.", "Energy, digital"),
    ("Dec 2022", "Azerbaijan, Georgia, Romania and Hungary sign the Black Sea cable "
                 "agreement.", "Energy, digital"),
    ("May 2024", "Azerbaijan and Iran open two dams on the Aras.", "Water, energy"),
    ("Aug 2025", "TRIPP announced in Washington: rail, road, pipeline, power, fibre.",
     "All"),
    ("Dec 2025", "EU lists the BSSC as a Project of Mutual Interest, power and fibre "
                 "together.", "Energy, digital"),
    ("Jun 2026", "World Bank TC-GATE: electric locomotives for the Georgian main line.",
     "Transport, energy"),
    ("Q3 2026", "Trans-Caspian fibre enters service. Power and gas cables still at study.",
     "Digital"),
]


# ---------------------------------------------------------------------------
# 4. the region on the ground
# ---------------------------------------------------------------------------
def ground(index) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index)
    m.add('<g class="zoomable">')
    water.draw_rivers(m, max_rank=6, scale=0.8, opacity=0.7)
    transport.draw_rail(m, colour=RAIL, width=0.8, min_px=2, status=False)
    hv = [line for f in load_json(DIGITAL_DATA / "osm_hv_lines.geojson")["features"]
          if voltage_of(f["properties"]) >= 330 for line in lines_of(f["geometry"])]
    batch(m, hv, f'stroke="{GRID}" stroke-width="1.1" stroke-opacity="0.9"', 1.5)
    pipes = [line for f in load_json(DIGITAL_DATA / "osm_pipelines.geojson")["features"]
             if f["properties"].get("substance") in ("gas", "natural_gas")
             for line in lines_of(f["geometry"])]
    batch(m, pipes, f'stroke="{PIPE}" stroke-width="1.4" stroke-opacity="0.9"', 1.5)
    digital.draw_cables(m, width=1.8, skip=())
    label_countries(m, size=9)
    m.add("</g>")
    legend = html_legend([
        ("Lines", [(sw_line(GRID, 2), "power, 330 kV and above"),
                   (sw_line(PIPE, 2), "gas pipeline"),
                   (sw_line(RAIL, 2), "railway"),
                   (sw_line(digital.CABLE, 3), "submarine cable"),
                   (sw_line(water.RIVER, 2), "main river")]),
    ])
    cap = ("The layers of the sector tabs on one map. Grid, pipelines and rail: "
           "OpenStreetMap. Cables: submarinecablemap.com. Rivers: Natural Earth.")
    return imap(svg_of(m), cap, legend)


# ---------------------------------------------------------------------------
# 5. shared corridors
# ---------------------------------------------------------------------------
def badges(m: Map, x, y, c):
    for i, (k, _) in enumerate(SECTORS):
        status = c[k][0]
        fill = STATUS_FILL[status]
        text = "#fff" if status in ("service", "committed") else (
            PALETTE["ink"] if status else "#c0c6ca")
        stroke = fill if status else "#cfd6db"
        bx = x + i * 19
        m.add(f'<rect x="{bx:.1f}" y="{y:.1f}" width="16" height="16" rx="3" fill="{fill}" '
              f'stroke="{stroke}" stroke-width="1.2"><title>{esc(SECTORS[i][1])}: '
              f'{esc(c[k][1])}</title></rect>')
        m.add(f'<text x="{bx + 8:.1f}" y="{y + 12:.1f}" text-anchor="middle" font-size="10" '
              f'font-weight="700" fill="{text}" font-family="Segoe UI, Arial, sans-serif">'
              f"{k}</text>")


def structure(index) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index)
    m.add('<g class="zoomable">')
    label_countries(m, size=9)
    for c in CORRIDORS:
        d = m.smooth(c["path"])
        m.add(f'<path d="{d}" fill="none" stroke="{PALETTE["ink"]}" stroke-width="9" '
              'stroke-opacity="0.16" stroke-linecap="round"/>')
        m.add(f'<path d="{d}" fill="none" stroke="{PALETTE["ink"]}" stroke-width="1.6" '
              'stroke-linecap="round"/>')
    for i, c in enumerate(CORRIDORS, 1):
        lon, lat = c["path"][len(c["path"]) // 2]
        x, y = m.xy(lon, lat)
        m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{PALETTE["ink"]}" '
              'stroke="#fff" stroke-width="1.5"/>')
        m.add(f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="11" '
              f'font-weight="700" fill="#fff" font-family="Segoe UI, Arial, sans-serif">{i}'
              "</text>")
        bl, bt, anchor = c["box"]
        title = f"{i}. {c['name']}"
        w = max(6.6 * len(title), 5 * 19) + 12
        bx, by = m.xy(bl, bt)
        left = {"start": bx, "end": bx - w, "middle": bx - w / 2}[anchor]
        m.add(f'<rect x="{left:.1f}" y="{by - 14:.1f}" width="{w:.1f}" height="42" rx="6" '
              f'fill="#fff" fill-opacity="0.94" stroke="{PALETTE["border"]}"/>')
        m.add(f'<text x="{left + 6:.1f}" y="{by:.1f}" font-size="11" font-weight="700" '
              f'fill="{PALETTE["ink"]}" font-family="Segoe UI, Arial, sans-serif">'
              f"{esc(title)}</text>")
        badges(m, left + 6, by + 6, c)
    m.add("</g>")
    legend = html_legend([
        ("Sectors", [("", "P power, G gas, T transport, D digital, W water")]),
        ("Status", [(sw_box(STATUS_FILL["service"]), "in service"),
                    (sw_box(STATUS_FILL["committed"]), "committed"),
                    (sw_box(STATUS_FILL["planned"]), "planned"),
                    (sw_box("#ffffff"), "none")]),
    ])
    cap = ("Corridors are schematic lines through their main nodes. Each badge shows the most "
           "advanced asset of that sector. Hover a badge for its name.")
    return imap(svg_of(m), cap, legend) + matrix_table()


def matrix_table() -> str:
    def cell(status, text):
        if not status:
            return "<td class='small' style='color:#a8a39a'>n/a</td>"
        col = STATUS_FILL[status]
        ink = "#8a6d00" if status == "planned" else col
        return (f"<td><span class='st' style='border-color:{col};color:{ink}'>"
                f"{STATUS_WORD[status]}</span><br><span class='small'>{esc(text)}</span></td>")

    rows = "".join(
        f"<tr><td><b>{i}. {esc(c['name'])}</b></td>"
        + "".join(cell(*c[k]) for k, _ in SECTORS) + "</tr>"
        for i, c in enumerate(CORRIDORS, 1))
    head = "".join(f"<th>{esc(name)}</th>" for _, name in SECTORS)
    return (f'<table class="kt pt"><thead><tr><th>Corridor</th>{head}</tr></thead>'
            f"<tbody>{rows}</tbody></table>")


# ---------------------------------------------------------------------------
# 6. projects that join sectors
# ---------------------------------------------------------------------------
def projects_list():
    tanap = next(p for p in gas.PROJECTS if p["name"] == "TANAP expansion")
    return [
        {"name": "Anaklia: port, cable landfall, fibre", "map": "Anaklia",
         "status": "committed", "point": power.ANAKLIA, "label": (41.0, 43.15, "end"),
         "scale": "Port, power, fibre", "year": "about 2029",
         "corridor": "Black Sea crossing", "link": "Deep sea port and the BSSC landfall",
         "stage": "Port works under way. Cable on the EU PMI list.",
         "stake": "One site for the port, the cable and its fibre."},
        {"name": "BSSC power and fibre", "map": "BSSC", "status": "planned",
         "anchors": digital.bssc_route(), "label": (33.0, 43.75, "middle"),
         "scale": "Power, fibre", "year": "2031", "corridor": "Black Sea crossing",
         "link": "1,300 MW and 40 Tbps in one cable lay",
         "stage": "EU PMI Dec 2025. Transelectrica and GSE memorandum Feb 2026.",
         "stake": "A second direct data path to the EU at small extra cost."},
        {"name": "Black Sea AI gigafactory", "map": "AI gigafactory", "status": "planned",
         "point": (28.05, 44.33), "label": (27.75, 44.65, "end"),
         "scale": "Data centre, power", "year": "2028", "corridor": "Black Sea crossing",
         "link": "Beside the Cernavoda nuclear plant",
         "stage": "Announced. Up to EUR 5bn.",
         "stake": "About 1.5 GW of load near the BSSC landfall."},
        {"name": "TANAP and SOCAR Fiber", "map": "TANAP", "status": "planned",
         "anchors": tanap["anchors"], "label": (36.6, 40.3, "middle"),
         "scale": "Gas, fibre", "year": "n/a", "corridor": "Georgian east-west axis",
         "link": "Georgian border to Kipoi",
         "stage": "Pipeline and fibre in service. Expansion has no FID.",
         "stake": "An expansion could add fibre at trench cost."},
        {"name": "TC-GATE electric traction", "map": None, "status": "committed",
         "scale": "Rail, power", "year": "n/a", "corridor": "Georgian east-west axis",
         "link": "Electric locomotives for the main line",
         "stage": "World Bank USD 372m, Jun 2026.",
         "stake": "Rail freight growth becomes power demand in Georgia."},
        {"name": "Trans-Caspian cables", "map": "Trans-Caspian", "status": "committed",
         "anchors": digital.cable_line("Trans-Caspian"), "dash": "10 5",
         "label": (49.4, 42.6, "end"), "scale": "Fibre, power, gas", "year": "2026",
         "corridor": "Caspian crossing", "link": "Sumgait to Aktau",
         "stage": "Fibre laid. HVDC at feasibility, gas pipeline at concept.",
         "stake": "Three subsea projects on one sea floor. Surveys could be shared."},
        {"name": "Absheron desalination", "map": "Desalination", "status": "committed",
         "point": (49.9, 40.5), "label": (50.7, 39.35, "middle"), "scale": "Water, power",
         "year": "2027", "corridor": "Caspian crossing", "link": "300,000 m³/day for Baku",
         "stage": "ACWA Power and IC Ictas, contract Dec 2025.",
         "stake": "Baku's water becomes a power load."},
        {"name": "TRIPP", "map": "TRIPP", "status": "planned", "osm": "tripp",
         "label": (46.3, 38.4, "middle"), "leader": (46.2, 38.88),
         "scale": "Rail, road, power, gas, fibre", "year": "n/a",
         "corridor": "Aras valley and Nakhchivan", "link": "Southern Armenia, about 43 km",
         "stage": "Framework Jun 2026. The power leg to Nakhchivan is under construction.",
         "stake": "The only route that plans all five networks together."},
        {"name": "Kars to Dilucu rail and Igdir power", "map": "Kars to Dilucu",
         "status": "committed", "osm": "kars_dilucu", "label": (43.1, 39.55, "end"),
         "scale": "Rail, power", "year": "2030", "corridor": "Aras valley and Nakhchivan",
         "link": "Rail and a 400 kV line to the Nakhchivan border",
         "stage": "Rail groundbreaking Aug 2025. Power line planned for 2032.",
         "stake": "Rail and line reach the same border crossing."},
        {"name": "Turkiye and Iraq oil for water", "map": None, "status": "committed",
         "scale": "Water, oil", "year": "2026", "corridor": "Tigris",
         "link": "Iraqi oil pays for Turkish water works",
         "stage": "Payments started Sep 2026.", "stake": "Energy money finances water."},
    ]


def projects(index) -> str:
    cap = ("Projects that carry two sectors or more. BSSC on the Caucasus Cable System "
           "crossing, landfalls moved to Anaklia and Constanta. TANAP is schematic. Two "
           "projects are in the table only.")

    def background(m):
        digital.draw_cables(m, colour="#b9c9d6", width=1.4)

    return projects_map(index, projects_list(), cap, background, cap_head="Sectors")


# ---------------------------------------------------------------------------
# pane
# ---------------------------------------------------------------------------
def pane(index: CountryIndex) -> str:
    return "".join([
        primer_block("Why the sectors meet", primer()),
        '<h4 class="mh">Key figures</h4>', key_figures(),
        '<h4 class="mh">Background</h4>', timeline(TIMELINE),
        '<h4 class="mh">The region on the ground</h4>', ground(index),
        '<h4 class="mh">Shared corridors</h4>', structure(index),
        '<h4 class="mh">Projects that join sectors</h4>', projects(index),
    ])
