"""Transport panes of the Black Sea sector briefs: Overview, Rail and road, Ports.

Volumes are national, operator and TITR statistics, quoted with their year. The
rail layer is the OpenStreetMap extract of the digital note, roads are Natural
Earth (expressways and major highways only, the classes are not consistent
across countries). Ports, crossings, ferry services and corridor lines are
placed by hand and drawn as schematic routes.
"""

from __future__ import annotations

import math

from gas import table
from mapkit import DIGITAL_DATA, FOCUS, PALETTE, CountryIndex, Map, lines_of, load_json
from power import (BBOX, H, HERE, OTHER_FILL, STATUS_STYLE, W, esc, html_legend, imap,
                   label_countries, pill, svg_of, sw_box, sw_dot, sw_line)

RAIL_E = PALETTE["ink"]     # electrified rail
RAIL_N = "#8fb3c9"          # rail without catenary
ROAD = "#c9a27a"            # expressways and major highways
FERRY = PALETTE["accent"]
CLOSED = "#c0504d"
MUT = "#6f6a61"


# ---------------------------------------------------------------------------
# shared data
# ---------------------------------------------------------------------------
_CACHE = {}


def railways():
    if "rail" not in _CACHE:
        _CACHE["rail"] = load_json(DIGITAL_DATA / "osm_railways.geojson")
    return _CACHE["rail"]


def roads():
    if "road" not in _CACHE:
        _CACHE["road"] = load_json(HERE / "data" / "ne_roads_region.geojson")
    return _CACHE["road"]


def is_electric(props):
    return props.get("electrified") in ("contact_line", "yes", "rail")


def coarse_path(m: Map, coords, min_px):
    """Like Map.path, on whole pixels: the background layers need no more."""
    lons = [q[0] for q in coords]
    lats = [q[1] for q in coords]
    if max(lons) < BBOX[0] or min(lons) > BBOX[2] or max(lats) < BBOX[1] or min(lats) > BBOX[3]:
        return ""
    pts = []
    last = None
    for i, (lon, lat) in enumerate(coords):
        x, y = m.xy(lon, lat)
        if last is not None and i != len(coords) - 1:
            if abs(x - last[0]) < min_px and abs(y - last[1]) < min_px:
                continue
        pts.append((round(x), round(y)))
        last = (x, y)
    if len(pts) < 2:
        return ""
    out = [pts[0]] + [q for a, q in zip(pts, pts[1:]) if q != a]
    return "M" + "L".join(f"{x},{y}" for x, y in out) if len(out) > 1 else ""


def batch(m: Map, lines, style, min_px):
    """One path element for many lines: keeps the page small."""
    d = "".join(x for x in (coarse_path(m, line, min_px) for line in lines) if x)
    if d:
        m.add(f'<path d="{d}" fill="none" {style} stroke-linecap="round" '
              'stroke-linejoin="round"/>')


def draw_roads(m: Map, width=1.0, opacity=0.7, min_px=1.5):
    lines = [line for feat in roads()["features"]
             if feat["properties"].get("expressway")
             or feat["properties"].get("type") == "Major Highway"
             for line in lines_of(feat["geometry"])]
    batch(m, lines, f'stroke="{ROAD}" stroke-width="{width}" stroke-opacity="{opacity}"', min_px)


def draw_rail(m: Map, by_electric=True, width=1.1, min_px=1.2, colour=None, status=True):
    """OSM rail. Lines in service by traction, then works and closed lines on top."""
    groups = {}
    for feat in railways()["features"]:
        p = feat["properties"]
        kind = p.get("railway")
        if kind == "rail":
            key = colour or (RAIL_E if (by_electric and is_electric(p)) else RAIL_N)
        elif status:
            key = "works" if kind == "construction" else "closed"
        else:
            continue
        groups.setdefault(key, []).extend(lines_of(feat["geometry"]))
    for key, lines in sorted(groups.items(), key=lambda kv: kv[0] in ("works", "closed")):
        if key == "works":
            batch(m, lines, f'stroke="{PALETTE["building"]}" stroke-width="2.4" '
                            'stroke-dasharray="5 3"', 0.8)
        elif key == "closed":
            batch(m, lines, f'stroke="{CLOSED}" stroke-width="1.6" stroke-dasharray="1.5 2.5"',
                  0.8)
        else:
            batch(m, lines, f'stroke="{key}" stroke-width="{width}"', min_px)


def osm_lines(key, index):
    """Real alignments for the projects OpenStreetMap already carries."""
    out = []
    for feat in railways()["features"]:
        p = feat["properties"]
        name = p.get("name") or ""
        for line in lines_of(feat["geometry"]):
            lon = sum(q[0] for q in line) / len(line)
            lat = sum(q[1] for q in line) / len(line)
            if key == "horadiz" and "Horadiz" in name:
                out.append(line)
            elif key == "kars_dilucu" and "Kars-Dilucu" in name:
                out.append(line)
            elif (key == "tripp" and p.get("layer") == "corridor" and lat < 39.25
                  and 45.8 < lon < 46.6 and index.at(lon, lat) == "Armenia"):
                out.append(line)
    return out


def timeline(rows) -> str:
    items = "".join(
        f'<li><span class="tl-y">{esc(y)}</span><span class="tl-t">{esc(t)}</span>'
        f'<span class="tl-s">{esc(s)}</span></li>'
        for y, t, s in rows
    )
    return f'<ol class="tl">{items}</ol>'


def tiles(rows) -> str:
    return "".join(f'<div class="stat"><div class="v">{esc(v)}</div>'
                   f'<div class="l">{esc(lab)}</div></div>' for v, lab in rows)


def bars(rows, fmt, width=270, unit="") -> str:
    """Horizontal bars. rows: (label, value, colour, part) where part is an optional
    sub value drawn darker inside the bar."""
    top = max(v for _, v, _, _ in rows)
    bar_h, gap, left = 22, 10, 92
    span = width - left - 52
    out = []
    for i, (lab, v, col, part) in enumerate(rows):
        y = i * (bar_h + gap)
        w = span * v / top
        out.append(f'<text x="{left - 8}" y="{y + 15}" text-anchor="end" font-size="11.5" '
                   f'fill="#403b35">{esc(lab)}</text>')
        out.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" '
                   f'fill="{col}"/>')
        if part:
            out.append(f'<rect x="{left}" y="{y}" width="{span * part / top:.1f}" '
                       f'height="{bar_h}" rx="3" fill="{PALETTE["ink"]}"/>')
        out.append(f'<text x="{left + w + 6:.1f}" y="{y + 15}" font-size="11.5" '
                   f'font-weight="700" fill="{PALETTE["ink"]}">{esc(fmt(v))}{unit}</text>')
    h = len(rows) * (bar_h + gap)
    return (f'<svg viewBox="0 0 {width} {h}" width="100%" '
            f'font-family="Segoe UI, Arial, sans-serif">{"".join(out)}</svg>')


def headrow(tile_rows, card_head, card_body, card_leg="") -> str:
    leg = f'<div class="mixleg">{card_leg}</div>' if card_leg else ""
    return ('<div class="headrow">'
            f'<div class="hr-kpi"><div class="stats">{tiles(tile_rows)}</div></div>'
            f'<div class="hr-mix"><div class="mixcard"><div class="mixhead">{esc(card_head)}'
            f'</div>{card_body}{leg}</div></div></div>')


def chain_svg(top, bottom=None, notes=(), height=None, top_y=18, bottom_x0=None,
              branch=None, mid_fill=()):
    """Boxes linked by arrows, as in the energy primers. bottom is a second row
    that leaves the top row at the box index given by branch."""
    ink, acc = PALETTE["ink"], PALETTE["accent"]
    bw, step = 118, 140
    h = height or (210 if bottom else 100)
    out = [f'<svg viewBox="0 0 860 {h}" width="100%" font-family="Segoe UI, Arial, sans-serif">',
           '<defs><marker id="ta" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
           f'markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="{acc}"/></marker>'
           '</defs>']

    def box(x, y, head, sub, fill="#fff"):
        out.append(f'<rect x="{x}" y="{y}" width="{bw}" height="54" rx="7" fill="{fill}" '
                   f'stroke="{ink}" stroke-width="1.2"/>')
        out.append(f'<text x="{x + bw / 2}" y="{y + 23}" text-anchor="middle" font-size="12.5" '
                   f'font-weight="700" fill="{ink}">{esc(head)}</text>')
        out.append(f'<text x="{x + bw / 2}" y="{y + 40}" text-anchor="middle" font-size="10.5" '
                   f'fill="{MUT}">{esc(sub)}</text>')

    def arrow(x1, y1, x2, y2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{acc}" '
                   f'stroke-width="2"{d} marker-end="url(#ta)"/>')

    for i, (hd, sb) in enumerate(top):
        x = 12 + i * step
        box(x, top_y, hd, sb, "#eaf1f8" if hd in mid_fill else "#fff")
        if i < len(top) - 1:
            arrow(x + bw + 2, top_y + 27, x + step - 2, top_y + 27)
    if bottom:
        x0 = bottom_x0 if bottom_x0 is not None else 12
        by = top_y + 112
        for i, (hd, sb) in enumerate(bottom):
            x = x0 + i * step
            box(x, by, hd, sb, "#eaf1f8" if hd in mid_fill else "#fff")
            if i < len(bottom) - 1:
                arrow(x + bw + 2, by + 27, x + step - 2, by + 27)
        bx = 12 + branch * step + bw / 2
        arrow(bx, top_y + 56, x0 + bw / 2 if x0 > bx - 10 else bx, by - 2)
    for x, y, text, col in notes:
        out.append(f'<text x="{x}" y="{y}" font-size="10.5" fill="{col or MUT}">{esc(text)}</text>')
    out.append("</svg>")
    return "".join(out)


def bullets(rows) -> str:
    items = "".join(f"<li><b>{esc(h)}</b> {esc(t)}</li>" for h, t in rows)
    return f'<ul class="primer-list">{items}</ul>'


def primer_block(title, body) -> str:
    return ('<details class="primer">'
            f'<summary>{esc(title)}</summary>'
            f'<div class="primer-body">{body}</div></details>')


# ---------------------------------------------------------------------------
# places, placed by hand, approximate to a few kilometres
# ---------------------------------------------------------------------------
# name, lon, lat, label anchor, label dx, label dy (degrees)
PORTS = [
    ("Poti", 41.65, 42.15, "end", -0.2, 0.0), ("Batumi", 41.64, 41.65, "end", -0.2, -0.1),
    ("Anaklia", 41.56, 42.39, "end", -0.2, 0.15), ("Alat", 49.42, 39.95, "start", 0.2, -0.1),
    ("Aktau", 51.17, 43.62, "end", -0.2, 0.1), ("Kuryk", 51.68, 43.18, "start", 0.2, -0.1),
    ("Constanta", 28.66, 44.17, "start", 0.25, 0.05),
    ("Novorossiysk", 37.78, 44.72, "start", 0.2, 0.1),
    ("Odesa", 30.75, 46.5, "start", 0.2, 0.1), ("Chornomorsk", 30.66, 46.3, "end", -0.2, -0.25),
    ("Varna", 27.93, 43.19, "end", -0.2, 0.05), ("Burgas", 27.48, 42.49, "end", -0.2, -0.1),
    ("Samsun", 36.35, 41.3, "start", 0.2, 0.25), ("Trabzon", 39.75, 41.0, "middle", 0.0, 0.35),
    ("Ambarli", 28.68, 40.97, "end", -0.15, -0.3), ("Kocaeli", 29.9, 40.75, "start", 0.2, -0.3),
    ("Aliaga", 26.97, 38.8, "end", -0.2, 0.0), ("Mersin", 34.64, 36.78, "middle", 0.0, -0.35),
    ("Iskenderun", 36.18, 36.6, "start", 0.25, -0.1), ("Filyos", 32.03, 41.56, "middle", 0.0, 0.3),
]
PORT_AT = {name: (lon, lat) for name, lon, lat, *_ in PORTS}

# Containers, thousand TEU, latest year found.
TEU = [("Ambarli", 3430, "2025"), ("Mersin", 1940, "2023"), ("Poti", 636, "2025"),
       ("Samsun", 262, "2024"), ("Alat", 105, "2025")]
# Tonnage, Mt, latest year found, for ports with no container figure here.
TONNES = [("Novorossiysk", 168, "2025"), ("Aliaga", 89.5, "2025"), ("Kocaeli", 83.9, "2025"),
          ("Iskenderun", 70.9, "2025"), ("Constanta", 67, "2025"), ("Batumi", 6, "2025, 11 months")]

# Ferry and Ro-Ro services running today. Sea paths are schematic.
FERRIES = [
    ("Poti to Constanta", "E60 Shipping, twice a week since 2023",
     [(41.65, 42.15), (38.5, 42.85), (33.5, 43.35), (30.0, 43.9), (28.66, 44.17)]),
    ("Batumi to Varna", "since January 2025",
     [(41.64, 41.65), (37.5, 42.2), (32.5, 42.6), (29.5, 42.95), (27.93, 43.19)]),
    ("Batumi to Chornomorsk", "UkrFerry, resumed July 2024",
     [(41.64, 41.65), (37.0, 43.15), (33.2, 44.05), (31.3, 45.4), (30.66, 46.3)]),
    ("Caspian rail ferries", "ASCO, Alat to Aktau and Kuryk",
     [(49.42, 39.95), (50.35, 41.6), (51.17, 43.62)]),
    ("Caspian rail ferries", "ASCO, Alat to Kuryk",
     [(49.42, 39.95), (50.8, 41.6), (51.68, 43.18)]),
    ("Caspian rail ferries", "ASCO, Alat to Turkmenbashi",
     [(49.42, 39.95), (51.0, 40.1), (52.6, 40.05)]),
]

# Border crossings: name, lon, lat, mode, open, note, label anchor or None.
CROSSINGS = [
    ("Upper Lars", 44.63, 42.74, "road", True,
     "Georgia to Russia. The only open road. Queues of 2,000 to 4,000 trucks at peaks.", "start"),
    ("Sarpi", 41.55, 41.52, "road", True, "Georgia to Turkiye on the coast.", "end"),
    ("Kartsakhi", 43.25, 41.22, "rail", True, "BTK, Georgia to Turkiye.", None),
    ("Red Bridge", 45.1, 41.33, "road and rail", True, "Georgia to Azerbaijan.", "start"),
    ("Sadakhlo", 44.81, 41.25, "road and rail", True, "Georgia to Armenia.", None),
    ("Samur", 48.55, 41.85, "road and rail", True, "Azerbaijan to Russia.", "start"),
    ("Astara", 48.87, 38.43, "road and rail", True,
     "Azerbaijan to Iran. Rail gauge break since 2018.", "start"),
    ("Julfa", 45.62, 38.96, "rail", True, "Nakhchivan to Iran. Gauge break.", None),
    ("Meghri", 46.25, 38.89, "road", True, "Armenia to Iran, the Agarak bridge.", None),
    ("Dilucu", 44.7, 39.66, "road", True, "Turkiye to Nakhchivan.", None),
    ("Kapikule", 26.65, 41.72, "road and rail", True, "Turkiye to Bulgaria.", "end"),
    ("Kapikoy", 44.3, 38.4, "rail", True, "Turkiye to Iran, same gauge.", None),
    ("Kars to Gyumri", 43.67, 40.83, "rail", False, "Turkiye to Armenia. Closed since 1993.",
     "end"),
    ("Margara", 44.18, 40.03, "road", False, "Turkiye to Armenia. Closed since 1993.", None),
    ("Yeraskh", 44.77, 39.75, "rail", False, "Armenia to Nakhchivan. Closed since the early "
                                             "1990s.", None),
    ("Ijevan to Gazakh", 45.25, 41.12, "rail", False, "Armenia to Azerbaijan. Closed since the "
                                                      "early 1990s.", None),
    ("Inguri", 41.86, 42.42, "rail", False, "Abkhazia line, cut since 1992 to 1993.", "start"),
]
GAUGE_BREAKS = [("Akhalkalaki", 43.48, 41.40), ("Astara", 48.87, 38.43),
                ("Julfa", 45.62, 38.96)]


def draw_crossings(m: Map, labels=True):
    for name, lon, lat, mode, ok, note, anchor in CROSSINGS:
        x, y = m.xy(lon, lat)
        tip = f"<title>{esc(name)}, {esc(mode)}: {esc(note)}</title>"
        if ok:
            m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="#fff" stroke="{PALETTE["ink"]}" '
                  f'stroke-width="1.8">{tip}</circle>')
        else:
            m.add(f'<g stroke="{CLOSED}" stroke-width="2.6" stroke-linecap="round">'
                  f'<line x1="{x - 4.5:.1f}" y1="{y - 4.5:.1f}" x2="{x + 4.5:.1f}" y2="{y + 4.5:.1f}"/>'
                  f'<line x1="{x - 4.5:.1f}" y1="{y + 4.5:.1f}" x2="{x + 4.5:.1f}" y2="{y - 4.5:.1f}"/>'
                  f'{tip}</g>')
        if labels and anchor:
            dx = 0.18 if anchor == "start" else -0.18
            m.label(lon + dx, lat - 0.05, name, size=9.5,
                    colour=PALETTE["ink"] if ok else CLOSED, anchor=anchor, weight="600",
                    halo=True)


def draw_ports(m: Map, names=None, r=4.0, labels=True, size=9.5):
    for name, lon, lat, anchor, dx, dy in PORTS:
        if names and name not in names:
            continue
        x, y = m.xy(lon, lat)
        m.add(f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" rx="1.5" '
              f'fill="{FERRY}" stroke="#fff" stroke-width="1.2"><title>{esc(name)} port</title>'
              f'</rect>')
        if labels:
            m.label(lon + dx, lat + dy - 0.05, name, size=size, colour="#403b35", anchor=anchor,
                    halo=True)


def draw_ferries(m: Map, width=2.0):
    for name, note, path in FERRIES:
        m.add(f'<path d="{m.smooth(path)}" fill="none" stroke="{FERRY}" stroke-width="{width}" '
              f'stroke-dasharray="6 4" stroke-linecap="round" stroke-opacity="0.85">'
              f'<title>{esc(name)}: {esc(note)}</title></path>')


# ---------------------------------------------------------------------------
# projects, shared by the three panes
# ---------------------------------------------------------------------------
# mode: rail, road or port. Geometry: "osm" key, "anchors" (schematic), "point",
# or none (table only).
PROJECTS = [
    {"name": "Horadiz to Aghband railway", "map": "Horadiz to Aghband", "mode": "rail",
     "status": "committed", "osm": "horadiz", "scale": "110 km, 15 Mt/y",
     "year": "end 2026", "corridor": "Azerbaijan to Nakhchivan and Kars",
     "label": (47.45, 39.05, "start"), "link": "Along the Aras to the Armenian border",
     "stage": "69% built in Dec 2025. State budget.",
     "stake": "Brings Azerbaijani rail to the Armenian border."},
    {"name": "TRIPP rail, Meghri section", "map": "TRIPP", "mode": "rail",
     "status": "planned", "osm": "tripp", "scale": "About 43 km", "year": "n/a",
     "corridor": "Azerbaijan to Nakhchivan and Kars", "label": (46.3, 38.5, "middle"),
     "link": "Southern Armenia, with road, power and fibre",
     "stage": "Framework signed Jun 2026. US 74%, Armenia 26%. Feasibility to early 2027.",
     "stake": "Links Azerbaijan to Nakhchivan across southern Armenia."},
    {"name": "Kars to Igdir to Dilucu railway", "map": "Kars to Dilucu", "mode": "rail",
     "status": "committed", "osm": "kars_dilucu", "scale": "224 km", "year": "2030",
     "corridor": "Azerbaijan to Nakhchivan and Kars", "label": (43.1, 39.55, "end"),
     "link": "Turkish rail to the Nakhchivan border",
     "stage": "Groundbreaking Aug 2025. EUR 2.4bn, MUFG led lenders with EKN, OeKB, IsDB.",
     "stake": "Closes the Turkish end of the route through Nakhchivan."},
    {"name": "Kars to Gyumri reopening", "map": "Kars to Gyumri", "mode": "rail",
     "status": "planned", "anchors": [(43.10, 40.60), (43.45, 40.72), (43.67, 40.83),
                                      (43.84, 40.79)],
     "scale": "21 km in Armenia", "year": "n/a", "corridor": "Turkiye and Armenia",
     "label": (43.2, 41.0, "end"), "link": "Line closed since 1993",
     "stage": "Joint working group Apr 2026. Armenian design tender, about USD 15m.",
     "stake": "A second rail exit for Armenia. Gauge break still to solve."},
    {"name": "East to West Highway, Rikoti section", "map": "Rikoti", "mode": "road",
     "status": "committed", "anchors": [(43.6, 41.99), (43.35, 42.04), (43.08, 42.1)],
     "scale": "About 50 km", "year": "2026", "corridor": "Georgia",
     "label": (43.4, 42.8, "middle"), "link": "Shorapani to Argveta, tunnels and bridges",
     "stage": "ADB, World Bank, EIB, state budget. Completion reported for 2025 to 2026.",
     "stake": "Removes the main truck bottleneck between Tbilisi and the ports."},
    {"name": "Trans-Caspian corridor, TC-GATE", "map": None, "mode": "rail",
     "status": "committed", "scale": "USD 750m+", "year": "n/a", "corridor": "Georgia",
     "link": "Electric locomotives, Kakheti roads",
     "stage": "Approved Jun 2026. World Bank USD 372m, with AIIB and ADB.",
     "stake": "Rolling stock and feeder roads for the corridor."},
    {"name": "Tbilisi rail bypass", "map": "Tbilisi bypass", "mode": "rail",
     "status": "planned", "point": (44.85, 41.76), "scale": "About GEL 0.9bn",
     "year": "n/a", "corridor": "Georgia", "label": (45.1, 41.95, "start"),
     "link": "Freight out of central Tbilisi", "stage": "Pending.",
     "stake": "Frees the city section that limits through trains."},
    {"name": "North to South Road, Sisian to Kajaran", "map": "Sisian to Kajaran",
     "mode": "road", "status": "committed",
     "anchors": [(46.03, 39.52), (46.12, 39.35), (46.15, 39.15)],
     "scale": "60 km, Bargushat tunnel", "year": "2033", "corridor": "Armenia",
     "label": (45.55, 37.85, "end"), "leader": (46.12, 39.3), "link": "Part of the 556 km Bavra to Meghri road",
     "stage": "EIB EUR 236m signed Nov 2024, plus ADB. Works from 2026.",
     "stake": "Main road from Iran to Georgia through Armenia."},
    {"name": "Halkali to Kapikule railway", "map": "Halkali to Kapikule", "mode": "rail",
     "status": "committed",
     "anchors": [(28.78, 41.03), (28.0, 41.28), (27.35, 41.4), (26.75, 41.67),
                 (26.65, 41.72)],
     "scale": "229 km, 200 km/h", "year": "2026 to 2028", "corridor": "Turkiye to the EU",
     "label": (27.6, 41.95, "middle"), "link": "Istanbul to the Bulgarian border",
     "stage": "About EUR 1bn. EU IPA EUR 275m, EBRD up to EUR 100m.",
     "stake": "Faster rail between Istanbul and the EU."},
    {"name": "Development Road", "map": "Development Road", "mode": "rail",
     "status": "planned", "anchors": [(43.75, 35.4), (43.13, 36.34), (42.37, 37.1),
                                      (41.2, 37.07)],
     "scale": "1,200 km, USD 17bn+", "year": "n/a", "corridor": "Gulf to Turkiye and Russia",
     "label": (42.0, 36.0, "end"), "link": "Grand Faw to Fishkhabour, rail and road",
     "stage": "Launched with Turkiye Jul 2026. Rail design 95% done. Financing open.",
     "stake": "Gulf freight to Turkiye by land."},
    {"name": "Rasht to Astara railway", "map": "Rasht to Astara", "mode": "rail",
     "status": "planned", "anchors": [(48.87, 38.43), (48.95, 37.9), (49.58, 37.28)],
     "scale": "162 km", "year": "n/a", "corridor": "Gulf to Turkiye and Russia",
     "label": (50.75, 37.8, "middle"), "link": "Missing link of the INSTC west branch",
     "stage": "Russia and Iran agreement May 2023, Russian loan. Land acquisition.",
     "stake": "Last gap in rail from Russia to the Gulf along the Caspian."},
    # ports and maritime
    {"name": "Anaklia deep sea port", "map": "Anaklia", "mode": "port",
     "status": "committed", "point": PORT_AT["Anaklia"], "scale": "Phase 1 about USD 1.1bn",
     "year": "about 2029", "corridor": "Georgian Black Sea coast",
     "label": (40.9, 42.75, "end"), "link": "New deep water port north of Poti",
     "stage": "State landlord model since Jul 2026. Dredging and breakwater under way.",
     "stake": "Georgia's first deep water port."},
    {"name": "Poti port expansion", "map": "Poti +", "mode": "port", "status": "committed",
     "point": PORT_AT["Poti"], "scale": "+150k TEU, then 1m TEU+", "year": "n/a",
     "corridor": "Georgian Black Sea coast", "label": (40.9, 42.05, "end"),
     "link": "1,700 m breakwater, 400 m quay at 13.5 m",
     "stage": "APM Terminals. Stage 1 under way since 2024.",
     "stake": "Larger ships at Georgia's main container port."},
    {"name": "Alat port, phase 2", "map": "Alat +", "mode": "port", "status": "committed",
     "point": PORT_AT["Alat"], "scale": "15 to 25 Mt/y, 500k TEU", "year": "n/a",
     "corridor": "Caspian", "label": (50.0, 40.35, "start"), "link": "Port of Baku at Alat",
     "stage": "Works started Dec 2024. State owned.",
     "stake": "Caspian handling capacity on the Azerbaijani side."},
    {"name": "Aktau and Kuryk expansion", "map": "Aktau, Kuryk", "mode": "port",
     "status": "committed", "point": PORT_AT["Kuryk"], "scale": "21 to 30 Mt/y",
     "year": "2028", "corridor": "Caspian", "label": (50.9, 42.65, "end"),
     "link": "Kazakh Caspian ports",
     "stage": "Aktau container hub with Lianyungang opened Jun 2025.",
     "stake": "Caspian handling capacity on the Kazakh side."},
    {"name": "Caspian fleet renewal", "map": None, "mode": "port", "status": "committed",
     "scale": "13 rail ferries today", "year": "n/a", "corridor": "Caspian",
     "link": "ASCO ferries and Ro-Ro", "stage": "EBRD USD 60m loan for new ships.",
     "stake": "The Caspian crossing is the slowest leg."},
    {"name": "Mersin port capacity", "map": "Mersin +", "mode": "port", "status": "committed",
     "point": PORT_AT["Mersin"], "scale": "2.6 to 3.6m TEU", "year": "2026",
     "corridor": "Turkiye", "label": (34.0, 36.3, "end"), "link": "Mediterranean gateway",
     "stage": "Ministry of Transport.", "stake": "Turkiye's main gateway to the Middle East."},
]


def draw_projects(m: Map, projects, index, pills=True):
    nodes = []
    for p in projects:
        col, dash, _ = STATUS_STYLE[p["status"]]
        dash = p.get("dash", dash)  # submarine cables use a longer dash
        if p.get("point"):
            nodes.append((p["point"], col, 5.5))
            continue
        if p.get("osm"):
            lines = osm_lines(p["osm"], index)
            paths = [m.path(line, min_px=0.5) for line in lines]
        elif p.get("anchors"):
            paths = [m.smooth(p["anchors"])]
            nodes += [(p["anchors"][0], col, 3.6), (p["anchors"][-1], col, 3.6)]
        else:
            continue
        for d in paths:
            m.add(f'<path d="{d}" fill="none" stroke="#fff" stroke-width="5.4" '
                  f'stroke-linecap="round" stroke-opacity="0.9"/>')
            m.add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="3" '
                  f'stroke-dasharray="{dash}" stroke-linecap="round"><title>{esc(p["name"])}'
                  f'</title></path>')
    for (lon, lat), col, r in nodes:
        x, y = m.xy(lon, lat)
        m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="#fff" stroke="{col}" '
              f'stroke-width="2"/>')
    if pills:
        for p in projects:
            if p.get("leader"):
                x1, y1 = m.xy(*p["label"][:2])
                x2, y2 = m.xy(*p["leader"])
                m.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                      f'stroke="{PALETTE["muted"]}" stroke-width="1"/>')
        for p in projects:
            if p.get("map"):
                pill(m, *p["label"], p["map"], p["status"])


def projects_map(index, projects, cap, background, with_table=True, cap_head="Scale"):
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index)
    m.add('<g class="zoomable">')
    background(m)
    label_countries(m, size=9)
    draw_projects(m, projects, index)
    m.add("</g>")
    legend = html_legend([
        ("Status", [(sw_line(PALETTE["building"], 3, "7 4"), "committed"),
                    (sw_line(PALETTE["planned"], 3, "1.5 4"), "planned")]),
    ])
    out = imap(svg_of(m), cap, legend)
    if with_table:
        out += table(projects, cap_head, "scale")
    return out


def thin_rail(m: Map):
    draw_rail(m, colour="#c6d3dc", width=0.8, min_px=3, status=False)


# ===========================================================================
# Transport > Overview
# ===========================================================================
TITR_MT = [("2022", 1.5), ("2023", 2.76), ("2024", 4.48), ("2025", 4.12)]

COUNTRY_ROWS = [
    ("Turkiye", "13,919 km, 51% electric", "About 25 Mt (2025)",
     "553 Mt and 14m TEU, all ports (2025)", "Land bridge to the EU. BTK ends at Kars."),
    ("Georgia", "1,593 km of line, all electric", "13.3 Mt (2025)",
     "Poti 636k TEU (2025). Batumi 6 Mt+ (11 months, 2025)",
     "Black Sea gateway. Gauge break at Akhalkalaki."),
    ("Azerbaijan", "2,918 km, 44% electric", "16.8 Mt, 6.5 Mt transit (2025)",
     "Alat 105k TEU (2025)", "Caspian gateway. Builds the line to Nakhchivan."),
    ("Armenia", "780 km, all electric", "1.2 Mt (Jan to Sep 2025)", "None, landlocked",
     "Rail exits through Georgia only. Turkiye and Azerbaijan links closed."),
]

OVERVIEW_TIMELINE = [
    ("1992 to 1993", "Abkhazia war cuts the coastal rail line from Russia to Georgia.",
     "Wikipedia"),
    ("1993", "Turkiye closes its border with Armenia, and the Kars to Gyumri line with it.",
     "Armenpress"),
    ("May 1993", "EU launches TRACECA, the first Europe to Caucasus to Asia corridor plan.",
     "TRACECA"),
    ("Oct 2013", "Marmaray rail tunnel links Europe and Asia under the Bosphorus.", "TCDD"),
    ("Oct 2017", "Baku to Tbilisi to Kars railway opens.", "ADY"),
    ("2022", "Sanctions on Russia push the Middle Corridor to 1.5 Mt, 2.5 times 2021.",
     "TITR"),
    ("Jun 2023", "Poti to Constanta ferry starts.", "Investor.ge"),
    ("Aug 2025", "Washington summit announces TRIPP through southern Armenia.",
     "US State Department"),
    ("Oct 2025", "Azerbaijan lifts its restrictions on transit to Armenia.", "ARKA"),
    ("Jun 2026", "BTK upgraded to 5 Mt/y. World Bank approves TC-GATE in Georgia.",
     "Astana Times, World Bank"),
    ("Jul 2026", "Iraq and Turkiye launch the Development Road.", "AGBI"),
    ("Sep 2026", "World Bank Trans-Caspian report: trade can triple by 2040.", "World Bank"),
]


def overview_primer() -> str:
    svg = chain_svg(
        top=[("China, Kazakhstan", "Rail, 1,520 mm"), ("Aktau, Kuryk", "Port"),
             ("Caspian crossing", "Rail ferry, Ro-Ro"), ("Alat", "Port"),
             ("Azerbaijan, Georgia", "Rail, 1,520 mm"), ("Akhalkalaki", "Gauge break")],
        bottom=[("Poti, Batumi", "Port"), ("Black Sea", "Ferry to Constanta")],
        branch=4, bottom_x0=572, mid_fill=("Caspian crossing", "Black Sea"),
        notes=[(12, 160, "Two seas, two gauges, five or more", None),
               (12, 176, "handovers between China and Europe.", None),
               (712, 105, "then Kars and Turkish rail", PALETTE["accent"]),
               (712, 119, "to Istanbul, 1,435 mm", PALETTE["accent"])],
    )
    return f'<div class="primer-fig">{svg}</div>' + bullets([
        ("Every handover costs days.",
         "Each port, ferry and gauge change adds waiting time and a new operator."),
        ("The Caspian is the bottleneck.",
         "Few ships, weather stops and falling water levels set the corridor's pace."),
        ("The Northern route is the benchmark.",
         "Through Russia there is one gauge from China to Poland and no sea. It still carries "
         "about ten times more boxes."),
        ("Closed borders lengthen routes.",
         "Turkiye to Armenia and Armenia to Azerbaijan are shut. Freight detours through "
         "Georgia."),
    ])


def overview_key_figures() -> str:
    tile_rows = [
        ("4.1 Mt", "Middle Corridor freight, 2025. Down from 4.5 Mt in 2024."),
        ("77k TEU", "Middle Corridor containers, 2025. Up 36%."),
        ("746k TEU", "Northern route through Russia, 2024. About ten times more."),
        ("14 to 23 days", "China to Europe today. World Bank target: 10 to 15 days by 2030."),
    ]
    chart = bars([(y, v, "#9fc2d6", None) for y, v in TITR_MT],
                 fmt=lambda v: f"{v:.1f}", unit=" Mt")
    head = headrow(tile_rows, "Middle Corridor freight, Mt", chart)
    body = "".join(
        f"<tr><td><b>{esc(c)}</b></td><td>{esc(net)}</td><td>{esc(fr)}</td>"
        f"<td class='small'>{esc(port)}</td><td class='small'>{esc(role)}</td></tr>"
        for c, net, fr, port, role in COUNTRY_ROWS
    )
    table_html = ('<table class="kt"><thead><tr><th>Country</th><th>Rail network</th>'
                  '<th>Rail freight</th><th>Main ports, latest year</th><th>Position</th>'
                  f'</tr></thead><tbody>{body}</tbody></table>')
    note = ('<p class="note">Middle Corridor: TITR Association, scopes differ across sources. '
            "Northern route: UTLC ERA. Rail and ports: TCDD, Geostat, ADY, Armstat, operators."
            "</p>")
    return head + table_html + note


def overview_ground(index) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index)
    m.add('<g class="zoomable">')
    draw_roads(m, width=1.4, opacity=0.9, min_px=2.0)
    draw_rail(m, colour=RAIL_E, width=0.9, min_px=2.5, status=False)
    draw_ferries(m, width=1.8)
    label_countries(m)
    draw_ports(m, names={"Poti", "Batumi", "Alat", "Aktau", "Kuryk", "Constanta", "Varna",
                         "Novorossiysk", "Chornomorsk", "Samsun", "Ambarli", "Mersin",
                         "Iskenderun"})
    m.add("</g>")
    legend = html_legend([
        ("Network", [(sw_line(RAIL_E, 2), "railway"), (sw_line(ROAD, 2), "major road"),
                     (sw_line(FERRY, 3, "6 4"), "ferry service")]),
        ("Points", [('<i class="d sq" style="background:#0277bd"></i>', "port")]),
    ])
    cap = ("Rail: OpenStreetMap. Roads: Natural Earth, major roads only. Ferry routes are "
           "schematic.")
    return imap(svg_of(m), cap, legend)


# Corridors: name, colour, dash, path, label (lon, lat, anchor), note
CORRIDORS = [
    ("Middle Corridor", PALETTE["accent"], None,
     [(52.6, 43.9), (51.17, 43.62), (50.4, 41.8), (49.42, 39.95), (47.15, 40.65),
      (44.79, 41.72), (43.48, 41.40), (43.10, 40.60), (41.27, 39.90), (38.5, 39.75),
      (34.8, 39.8), (30.5, 40.5), (28.9, 41.0), (26.65, 41.72), (23.5, 42.6), (22.0, 43.0)],
     (37.0, 40.35, "middle"), "4.1 Mt, 2025"),
    ("Middle Corridor, Black Sea branch", PALETTE["accent"], "8 5",
     [(44.79, 41.72), (43.0, 42.1), (41.65, 42.15), (38.5, 42.85), (33.5, 43.35),
      (30.0, 43.9), (28.66, 44.17)], (34.0, 43.65, "middle"), "ferry to Constanta"),
    ("INSTC, west branch", "#8B7E72", None,
     [(47.0, 47.6), (47.5, 43.0), (48.55, 41.85), (49.6, 40.6), (48.87, 38.43),
      (49.0, 37.9), (49.58, 37.28), (50.0, 36.27), (50.4, 35.4)],
     (47.9, 44.4, "start"), "Russia to Iran, gap at Rasht to Astara"),
    ("North to South through Armenia", "#5389AE", "3 3",
     [(44.68, 43.4), (44.63, 42.74), (44.79, 41.72), (44.81, 41.25), (44.66, 41.10),
      (44.51, 40.18), (45.4, 39.5), (46.03, 39.52), (46.25, 38.89), (46.3, 38.3)],
     (43.4, 39.2, "end"), "road, Russia to Iran"),
    ("Development Road", PALETTE["planned"], "1.5 4",
     [(43.75, 35.4), (43.13, 36.34), (42.37, 37.1), (40.0, 37.1), (37.4, 37.07),
      (35.3, 37.0)], (39.0, 36.6, "middle"), "planned, Gulf to Turkiye"),
    ("TRIPP route", PALETTE["planned"], "1.5 4",
     [(47.15, 40.65), (47.05, 39.45), (46.25, 38.89), (45.4, 39.2), (44.7, 39.66),
      (44.04, 39.92), (43.10, 40.60)], (46.6, 38.35, "start"), "planned, via Nakhchivan"),
]


def overview_structure(index) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.add(f'<rect width="{W}" height="{H}" fill="{PALETTE["sea"]}"/>')
    m.add('<g class="zoomable">')
    for name, poly, *_ in index.entries:
        fill = "#f5eeda" if name in FOCUS else OTHER_FILL
        d = " ".join(r + "Z" for r in (m.path(ring, min_px=0.5) for ring in poly) if r)
        if d:
            m.add(f'<path d="{d}" fill="{fill}" stroke="#fff" stroke-width="0.8" '
                  f'fill-rule="evenodd"/>')
    for name, col, dash, path, _, note in CORRIDORS:
        d = m.smooth(path)
        width = 6 if name == "Middle Corridor" else 4
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m.add(f'<path d="{d}" fill="none" stroke="#fff" stroke-width="{width + 3}" '
              f'stroke-linecap="round" stroke-opacity="0.8"/>')
        m.add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{width}"{da} '
              f'stroke-linecap="round" stroke-opacity="0.9"><title>{esc(name)}: {esc(note)}'
              f'</title></path>')
    draw_crossings(m, labels=False)
    for name, col, _, _, (lon, lat, anchor), note in CORRIDORS:
        if name == "Middle Corridor, Black Sea branch":
            name = "Black Sea branch"
        text_col = PALETTE["ink"] if col == PALETTE["planned"] else col
        m.label(lon, lat, name, size=11, colour=text_col, anchor=anchor, weight="700", halo=True)
        m.label(lon, lat - 0.42, note, size=9.5, colour=MUT, anchor=anchor, halo=True)
    x, _ = m.xy(40.0, 47.6)
    m.add(f'<text x="{x:.1f}" y="18" text-anchor="middle" font-size="11" font-weight="700" '
          f'fill="#8B7E72" font-family="Segoe UI, Arial, sans-serif">Northern route through '
          f'Russia runs north of this map: 746k TEU, 2024</text>')
    for lon, lat, name in [(33.0, 38.6, "TURKIYE"), (43.0, 42.4, "GEORGIA"),
                           (44.9, 40.55, "ARMENIA"), (48.2, 40.95, "AZERBAIJAN"),
                           (41.5, 46.4, "RUSSIA"), (47.5, 36.6, "IRAN"),
                           (51.5, 46.5, "KAZAKHSTAN")]:
        m.label(lon, lat, name, size=10, colour="#8a857c", weight="600", halo=True)
    m.add("</g>")
    legend = html_legend([
        ("Corridors", [(sw_line(PALETTE["accent"], 4), "Middle Corridor, rail"),
                       (sw_line(PALETTE["accent"], 3, "8 5"), "Middle Corridor, ferry"),
                       (sw_line("#8B7E72", 3), "INSTC west branch"),
                       (sw_line("#5389AE", 3, "3 3"), "North to South road"),
                       (sw_line(PALETTE["planned"], 3, "1.5 4"), "planned")]),
        ("Border crossings", [(sw_dot(PALETTE["ink"], ring=True), "open"),
                              (f'<b style="color:{CLOSED};width:11px;text-align:center">'
                               '&times;</b>', "closed")]),
    ])
    cap = ("Corridors drawn schematically, not on their tracks. Crossings: hover for mode and "
           "status. Details in the Rail and road tab.")
    return imap(svg_of(m), cap, legend)


def overview_projects(index) -> str:
    cap = ("Rail and road projects with real alignments where OpenStreetMap has them: "
           "Horadiz to Aghband, Kars to Dilucu, TRIPP. Other routes are schematic. Tables in the "
           '<a href="#transport/rail">Rail and road</a> and '
           '<a href="#transport/ports">Ports</a> tabs.')
    return projects_map(index, PROJECTS, cap, thin_rail, with_table=False)


def overview_pane(index: CountryIndex, heads) -> str:
    primer_title, map_b, map_c = heads
    return "".join([
        primer_block(primer_title, overview_primer()),
        '<h4 class="mh">Key figures</h4>', overview_key_figures(),
        '<h4 class="mh">Background</h4>', timeline(OVERVIEW_TIMELINE),
        '<h4 class="mh">The region on the ground</h4>', overview_ground(index),
        f'<h4 class="mh">{esc(map_b)}</h4>', overview_structure(index),
        f'<h4 class="mh">{esc(map_c)}</h4>', overview_projects(index),
    ])


# ===========================================================================
# Transport > Rail and road
# ===========================================================================
RAIL_ROWS = [
    ("Turkiye", "13,919", "51%", "1,435", "About 25 (2025)", "TCDD",
     "Armenia, since 1993"),
    ("Georgia", "1,593", "99%", "1,520", "13.3 (2025)", "Georgian Railway",
     "Abkhazia line, since 1992 to 1993"),
    ("Azerbaijan", "2,918", "44%", "1,520", "16.8, 6.5 transit (2025)", "ADY",
     "Armenia, early 1990s"),
    ("Armenia", "780", "100%", "1,520", "1.2 (Jan to Sep 2025)",
     "South Caucasus Railway, RZD concession since 2008", "Turkiye and Azerbaijan"),
]
RAIL_FREIGHT = [("Turkiye", 25.0, None), ("Azerbaijan", 16.8, 6.5), ("Georgia", 13.3, None),
                ("Armenia", 1.6, None)]

RAIL_TIMELINE = [
    ("1993", "Kars to Gyumri line closed with the Turkish border.", "Armenpress"),
    ("2008", "RZD takes a 30 year concession of Armenia's railways.", "South Caucasus Railway"),
    ("Oct 2017", "BTK opens. Bogies change at Akhalkalaki.", "ADY"),
    ("May 2024", "BTK Georgian section upgraded, 1 to 5 Mt/y.", "Railway Gazette"),
    ("Aug 2025", "Turkiye breaks ground on Kars to Dilucu.", "RailFreight"),
    ("Oct 2025", "Upper Lars makes the electronic truck queue mandatory.", "Vestnik Kavkaza"),
    ("2026", "Georgia completes the mountain pass section and the Kvishkheti tunnel.",
     "Caspian Post"),
    ("Apr 2026", "Turkiye and Armenia form a working group on Kars to Gyumri.",
     "Turkish Minute"),
    ("Jun 2026", "TRIPP framework signed. TRIPP Development Company set up.",
     "US State Department"),
    ("End 2026", "Horadiz to Aghband due.", "Report.az"),
]


def rail_primer() -> str:
    svg = chain_svg(
        top=[("Rail, 1,520 mm", "Former Soviet states"), ("Gauge break", "Bogies or crane"),
             ("Rail, 1,435 mm", "Turkiye, Iran, EU"), ("Border post", "Customs, crews"),
             ("Terminal", "Truck for the last leg")],
        mid_fill=("Gauge break",), height=110,
        notes=[(712, 40, "Each step adds", None), (712, 54, "hours to days.", None),
               (12, 98, "Electric lines cost less per tonne-km. Single track caps how many "
                        "trains pass.", None)],
    )
    return f'<div class="primer-fig">{svg}</div>' + bullets([
        ("Two gauges meet here.",
         "Turkiye, Iran and the EU run 1,435 mm. Georgia, Armenia and Azerbaijan run 1,520 mm. "
         "Every through train changes bogies or reloads."),
        ("Electric lines are cheap to run.",
         "Georgia and Armenia are fully electric. Turkiye and Azerbaijan about half."),
        ("Trucks win short hauls.",
         "Road carries most trade with neighbours. Border queues decide the timing."),
        ("A closed border is a detour.",
         "Armenian freight to Turkiye or Europe goes through Georgia."),
    ])


def rail_key_figures() -> str:
    tile_rows = [
        ("5 Mt/y", "BTK capacity in Georgia since the upgrade. Traffic is far below."),
        ("6.5 Mt", "Rail transit through Azerbaijan, 2025. 39% of its rail freight."),
        ("99%", "Georgia's lines that are electric. Turkiye 51%, Azerbaijan 44%."),
        ("2,000 to 4,000", "Trucks queuing at Upper Lars at peaks. Georgia's only road to "
                           "Russia."),
    ]
    chart = bars([(c, v, "#9fc2d6", part) for c, v, part in RAIL_FREIGHT],
                 fmt=lambda v: f"{v:.1f}" if v < 10 else f"{v:.0f}", unit=" Mt")
    leg = (f'<span class="lg">{sw_box("#9fc2d6")}total</span>'
           f'<span class="lg">{sw_box(PALETTE["ink"])}of which transit</span>')
    head = headrow(tile_rows, "Rail freight, 2025, Mt", chart, leg)
    body = "".join(
        f"<tr><td><b>{esc(c)}</b></td><td class='n'>{esc(km)}</td><td class='n'>{esc(el)}</td>"
        f"<td class='n'>{esc(g)}</td><td>{esc(fr)}</td><td class='small'>{esc(op)}</td>"
        f"<td class='small'>{esc(cl)}</td></tr>"
        for c, km, el, g, fr, op, cl in RAIL_ROWS
    )
    table_html = ('<table class="kt"><thead><tr><th>Country</th><th class="n">Network, km</th>'
                  '<th class="n">Electric</th><th class="n">Gauge, mm</th><th>Freight, Mt</th>'
                  '<th>Operator</th><th>Closed links</th></tr></thead>'
                  f"<tbody>{body}</tbody></table>")
    note = ('<p class="note">TCDD, Geostat, ADY and CAREC, Armstat. Armenia 2025: nine months, '
            "the bar extrapolates to a year. Georgia: main line, without sidings.</p>")
    return head + table_html + note


def rail_ground(index) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index)
    m.add('<g class="zoomable">')
    draw_roads(m, width=1.4, opacity=0.85)
    draw_rail(m)
    label_countries(m)
    for name, lon, lat in GAUGE_BREAKS:
        x, y = m.xy(lon, lat)
        m.add(f'<rect x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" '
              f'transform="rotate(45 {x:.1f} {y:.1f})" fill="{PALETTE["planned"]}" '
              f'stroke="{PALETTE["ink"]}" stroke-width="1.2"><title>{esc(name)}, gauge break'
              f'</title></rect>')
    m.label(43.3, 41.62, "Akhalkalaki", size=9.5, colour=PALETTE["ink"], anchor="end",
            weight="600", halo=True)
    m.label(42.1, 40.25, "BTK", size=11, colour=PALETTE["ink"], weight="700", halo=True)
    m.add("</g>")
    legend = html_legend([
        ("Rail", [(sw_line(RAIL_E, 2), "electric"), (sw_line(RAIL_N, 2), "not electric"),
                  (sw_line(PALETTE["building"], 3, "5 3"), "under construction"),
                  (sw_line(CLOSED, 3, "1.5 2.5"), "disused or abandoned")]),
        ("Road", [(sw_line(ROAD, 2), "expressway or major highway")]),
        ("Points", [('<i class="d sq" style="background:#ddc32c;transform:rotate(45deg);'
                     'border:1px solid #256081"></i>', "gauge break")]),
    ])
    cap = ("Rail: OpenStreetMap, traction as tagged. Roads: Natural Earth, major roads only, "
           "classes vary by country.")
    return imap(svg_of(m), cap, legend)


GAUGE = {
    "1,435 mm": ("#CFDDE7", {"Turkey", "Iran", "Iraq", "Syria", "Bulgaria", "Greece", "Romania",
                             "Republic of Serbia", "Macedonia", "Albania", "Hungary",
                             "Lebanon", "Jordan", "Kosovo", "Montenegro"}),
    "1,520 mm": ("#E6DCC2", {"Russia", "Georgia", "Armenia", "Azerbaijan", "Ukraine",
                             "Moldova", "Kazakhstan", "Turkmenistan", "Uzbekistan", "Belarus"}),
}


def rail_structure(index) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.add(f'<rect width="{W}" height="{H}" fill="{PALETTE["sea"]}"/>')
    m.add('<g class="zoomable">')
    for name, poly, *_ in index.entries:
        fill = OTHER_FILL
        for col, members in GAUGE.values():
            if name in members:
                fill = col
        d = " ".join(r + "Z" for r in (m.path(ring, min_px=0.5) for ring in poly) if r)
        if d:
            stroke = "#6f6a61" if name in FOCUS else "#ffffff"
            m.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
                  f'stroke-width="{1.0 if name in FOCUS else 0.8}" fill-rule="evenodd"/>')
    draw_rail(m, by_electric=False, colour="#9aa8b1", width=0.8, min_px=3, status=False)
    for name, lon, lat in GAUGE_BREAKS:
        x, y = m.xy(lon, lat)
        m.add(f'<rect x="{x - 5.5:.1f}" y="{y - 5.5:.1f}" width="11" height="11" '
              f'transform="rotate(45 {x:.1f} {y:.1f})" fill="{PALETTE["planned"]}" '
              f'stroke="{PALETTE["ink"]}" stroke-width="1.2"><title>{esc(name)}, gauge break'
              f'</title></rect>')
    draw_crossings(m)
    m.label(43.3, 41.62, "Akhalkalaki", size=9.5, colour=PALETTE["ink"], anchor="end",
            weight="600", halo=True)
    for lon, lat, name in [(33.0, 38.9, "TURKIYE"), (43.0, 42.65, "GEORGIA"),
                           (44.9, 40.5, "ARMENIA"), (48.0, 40.95, "AZERBAIJAN"),
                           (41.5, 46.4, "RUSSIA"), (47.5, 36.6, "IRAN"),
                           (25.2, 45.6, "ROMANIA"), (25.4, 42.85, "BULGARIA")]:
        m.label(lon, lat, name, size=10.5, colour=PALETTE["ink"], weight="700", halo=True)
    m.add("</g>")
    legend = html_legend([
        ("Track gauge", [(sw_box(col), name) for name, (col, _) in GAUGE.items()]),
        ("Points", [('<i class="d sq" style="background:#ddc32c;transform:rotate(45deg);'
                     'border:1px solid #256081"></i>', "gauge break"),
                    (sw_dot(PALETTE["ink"], ring=True), "crossing open"),
                    (f'<b style="color:{CLOSED};width:11px;text-align:center">&times;</b>',
                     "crossing closed")]),
    ])
    cap = ("Crossings placed by hand. Hover for mode and status. Gauge breaks also at the EU "
           "borders of Moldova and Ukraine, not marked.")
    return imap(svg_of(m), cap, legend)


def rail_pane(index: CountryIndex, heads) -> str:
    primer_title, map_b, map_c = heads
    land = [p for p in PROJECTS if p["mode"] != "port"]
    cap = ("Horadiz to Aghband, Kars to Dilucu and TRIPP on their OpenStreetMap alignments. "
           "Other routes are schematic.")

    def background(m):
        draw_roads(m, width=0.8, opacity=0.45, min_px=2.0)
        thin_rail(m)

    return "".join([
        primer_block(primer_title, rail_primer()),
        '<h4 class="mh">Key figures</h4>', rail_key_figures(),
        '<h4 class="mh">Background</h4>', timeline(RAIL_TIMELINE),
        '<h4 class="mh">The region on the ground</h4>', rail_ground(index),
        f'<h4 class="mh">{esc(map_b)}</h4>', rail_structure(index),
        f'<h4 class="mh">{esc(map_c)}</h4>', projects_map(index, land, cap, background),
    ])


# ===========================================================================
# Transport > Ports and maritime
# ===========================================================================
PORT_ROWS = [
    ("Ambarli", "Turkiye", "n/a", "3.43m (2025)", "Istanbul's container port."),
    ("Mersin", "Turkiye", "n/a", "1.94m (2023)", "Mediterranean gateway."),
    ("Novorossiysk", "Russia", "168 (2025)", "n/a", "Largest Black Sea port. Oil and grain."),
    ("Constanta", "Romania", "67 (2025)", "n/a", "EU gateway. Down from 92.6 Mt in 2023."),
    ("Poti", "Georgia", "n/a", "636k (2025)", "Georgia's container port. A record year."),
    ("Batumi", "Georgia", "6+ (11 months, 2025)", "n/a", "Oil products and general cargo."),
    ("Alat", "Azerbaijan", "About 8 (2025)", "105k (2025)", "Caspian gateway. Up 37% in TEU."),
    ("Aktau and Kuryk", "Kazakhstan", "About 4.2 on TITR (2024)", "n/a",
     "Caspian gateway, east side."),
    ("Turkmenbashi", "Turkmenistan", "7.3 (2025)", "n/a", "Caspian, off the map to the east."),
]

PORT_TIMELINE = [
    ("Jun 2023", "Poti to Constanta ferry starts, twice a week.", "Investor.ge"),
    ("2024", "APM Terminals starts Poti stage 1.", "Seatrade Maritime"),
    ("May 2024", "CCCC consortium wins the Anaklia tender. No agreement follows.", "Civil.ge"),
    ("Jul 2024", "Chornomorsk to Batumi ferry resumes.", "Investor.ge"),
    ("Dec 2024", "Alat phase 2 starts, to 25 Mt/y.", "Port of Baku"),
    ("Jan 2025", "Varna to Batumi ferry starts.", "Investor.ge"),
    ("Jun 2025", "Aktau container hub opens with China's Lianyungang port.", "Trend"),
    ("2025", "Constanta falls to 67 Mt as Ukrainian transit drops.", "USM"),
    ("2025", "Turkish ports handle a record 553 Mt.", "Ministry of Transport"),
    ("Jul 2026", "Georgia takes Anaklia on as a state landlord port.", "Civil.ge"),
]


def ports_primer() -> str:
    svg = chain_svg(
        top=[("Rail or truck", "Hinterland"), ("Port", "Berths, cranes, depth"),
             ("Ship", "Container, Ro-Ro, ferry"), ("Port", "Customs, storage"),
             ("Rail or truck", "Hinterland")],
        mid_fill=("Ship",), height=110,
        notes=[(712, 40, "Rail ferries carry", None), (712, 54, "wagons, no reloading.", None),
               (12, 98, "Water depth sets ship size. Ship size sets the cost per box.", None)],
    )
    return f'<div class="primer-fig">{svg}</div>' + bullets([
        ("Depth sets the ship.",
         "Poti's new quay reaches 13.5 m. Anaklia is planned as Georgia's first deep port."),
        ("Ferries skip reloading.",
         "Caspian rail ferries take wagons whole. Black Sea Ro-Ro takes trucks and trailers."),
        ("Short sea links avoid borders.",
         "Ferries from Georgia to Romania, Bulgaria and Ukraine bypass Turkish and Russian "
         "roads."),
        ("War moved the trade.",
         "Ukrainian grain shifted to Constanta in 2022 to 2023, then back to Odesa."),
    ])


def ports_key_figures() -> str:
    tile_rows = [
        ("553 Mt", "All Turkish ports, 2025. A record."),
        ("636k TEU", "Poti, 2025. Georgia's record."),
        ("105k TEU", "Alat, 2025. Up 37% on 2024."),
        ("67 Mt", "Constanta, 2025. Down from 92.6 Mt in 2023."),
    ]
    chart = bars([(n, v / 1000, "#9fc2d6", None) for n, v, _ in TEU],
                 fmt=lambda v: f"{v:.2f}" if v < 1 else f"{v:.1f}", unit="m")
    head = headrow(tile_rows, "Containers, million TEU, latest year", chart)
    body = "".join(
        f"<tr><td><b>{esc(p)}</b></td><td>{esc(c)}</td><td class='n'>{esc(t)}</td>"
        f"<td class='n'>{esc(teu)}</td><td class='small'>{esc(r)}</td></tr>"
        for p, c, t, teu, r in PORT_ROWS
    )
    table_html = ('<table class="kt"><thead><tr><th>Port</th><th>Country</th>'
                  '<th class="n">Cargo, Mt</th><th class="n">Containers, TEU</th><th>Role</th>'
                  f'</tr></thead><tbody>{body}</tbody></table>')
    note = ('<p class="note">Ministry of Transport of Turkiye, port operators, PortNews, USM. '
            "Mersin 2023, Samsun 2024. Alat and Aktau tonnages are estimates.</p>")
    return head + table_html + note


def ports_ground(index) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index)
    m.add('<g class="zoomable">')
    thin_rail(m)
    draw_ferries(m)
    label_countries(m)
    draw_ports(m)
    m.add("</g>")
    legend = html_legend([
        ("Network", [(sw_line(FERRY, 3, "6 4"), "ferry or Ro-Ro service"),
                     (sw_line("#c6d3dc", 2), "railway")]),
        ("Points", [('<i class="d sq" style="background:#0277bd"></i>', "port")]),
    ])
    cap = ("Ports placed by hand. Ferry services running in 2026, sea paths schematic. Hover "
           "a line for its operator.")
    return imap(svg_of(m), cap, legend)


# Label offsets on the capacity map, clear of the larger circles.
STRUCT_OFF = {"Ambarli": (-1.05, 0.15, "end"), "Kocaeli": (0.75, -0.2, "start"),
              "Mersin": (-0.8, -0.15, "end"), "Iskenderun": (0.45, -0.1, "start")}


def ports_structure(index) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index)
    m.add('<g class="zoomable">')
    draw_ferries(m, width=2.4)
    for name, mt, year in TONNES:
        lon, lat = PORT_AT[name]
        x, y = m.xy(lon, lat)
        r = 2.2 * math.sqrt(mt)
        m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#8B7E72" fill-opacity="0.18" '
              f'stroke="#8B7E72" stroke-width="1.4"><title>{esc(name)}: {mt:g} Mt, {year}'
              f'</title></circle>')
    for name, teu, year in TEU:
        lon, lat = PORT_AT[name]
        x, y = m.xy(lon, lat)
        r = 0.55 * math.sqrt(teu)
        m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{FERRY}" fill-opacity="0.55" '
              f'stroke="#fff" stroke-width="1.2"><title>{esc(name)}: {teu:,}k TEU, {year}'
              f'</title></circle>')
    label_countries(m, size=9)
    values = {n: f"{v:g} Mt" for n, v, _ in TONNES}
    values.update({n: f"{v / 1000:.2f}m TEU" if v >= 1000 else f"{v}k TEU" for n, v, _ in TEU})
    for name, lon, lat, anchor, dx, dy in PORTS:
        if name in values:
            dx, dy, anchor = STRUCT_OFF.get(name, (dx, dy, anchor))
            m.label(lon + dx, lat + dy - 0.05, f"{name} {values[name]}", size=9.5,
                    colour="#403b35", anchor=anchor, weight="600", halo=True)
    m.label(51.2, 41.3, "Rail ferries", size=10, colour=FERRY, anchor="middle", weight="700",
            halo=True)
    m.add("</g>")
    legend = html_legend([
        ("Ports", [(sw_dot(FERRY), "containers, area by TEU"),
                   (sw_dot("#8B7E72", ring=True), "cargo, area by Mt")]),
        ("Links", [(sw_line(FERRY, 3, "6 4"), "ferry or Ro-Ro service")]),
    ])
    cap = ("Latest year found per port, from 2023 to 2025. Two scales: containers and total "
           "cargo. Ferry paths schematic.")
    return imap(svg_of(m), cap, legend)


def ports_pane(index: CountryIndex, heads) -> str:
    primer_title, map_b, map_c = heads
    sea = [p for p in PROJECTS if p["mode"] == "port"]
    cap = "Ports placed by hand. The Caspian fleet is not mapped."

    def background(m):
        thin_rail(m)
        draw_ferries(m, width=1.4)

    return "".join([
        primer_block(primer_title, ports_primer()),
        '<h4 class="mh">Key figures</h4>', ports_key_figures(),
        '<h4 class="mh">Background</h4>', timeline(PORT_TIMELINE),
        '<h4 class="mh">The region on the ground</h4>', ports_ground(index),
        f'<h4 class="mh">{esc(map_b)}</h4>', ports_structure(index),
        f'<h4 class="mh">{esc(map_c)}</h4>', projects_map(index, sea, cap, background),
    ])
