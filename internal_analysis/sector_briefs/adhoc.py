"""Ad hoc pane of the Black Sea sector briefs: one off notes that cut across tabs.

Digital and power: the key message, then a precise map (where data meets power,
on the ground) and a visual one (cheap clean power against the short data path).
"""

from __future__ import annotations

import heapq
import math
import re
from collections import defaultdict

import digital
import power
from mapkit import DIGITAL_DATA, FOCUS, PALETTE, CountryIndex, Map, lines_of, load_json
from power import (BBOX, GRID, H, W, esc, html_legend, imap, label_countries, svg_of,
                   sw_dot, sw_line, voltage_of)
from transport import bars, batch, headrow

SUBS = [("digital-power", "Digital and power")]

MESSAGE = [
    ("Power and data cross the Black Sea on the same route.",
     "The BSSC lays 1,300 MW and a fibre pair in one cable. It would be the second "
     "direct data path to the EU after the Caucasus Cable System."),
    ("Fibre is cheap to add to power assets.",
     "Optical earth wire on a new 400 or 500 kV line, or a duct in its trench, costs a "
     "fraction of a separate route. Every new interconnector can carry fibre by default."),
    ("Data centres are a load that can follow cheap, clean power.",
     "Today they sit where the exchanges are: Istanbul, Bucharest, Sofia. Cheap, low "
     "carbon power is elsewhere: Georgian hydro, cheapest in spring, and Caspian wind. A "
     "site needs both the power and the exchange."),
]


def message() -> str:
    items = "".join(
        f'<div class="msg"><div class="msg-n">{i}</div><div><b>{esc(h)}</b> {esc(t)}</div>'
        "</div>"
        for i, (h, t) in enumerate(MESSAGE, 1))
    return f'<div class="msgs">{items}</div>'


def stub(text: str) -> str:
    return f'<div class="stub tall">{esc(text)}</div>'


# ---------------------------------------------------------------------------
# where data meets power
# ---------------------------------------------------------------------------
STUDY = ("Turkey", "Georgia", "Armenia", "Azerbaijan")  # basemap names
STUDY_PDB = ("TR", "GE", "AM", "AZ")
# the maps draw lines, plants and data centres in these six countries only
SIX = STUDY + ("Romania", "Bulgaria")
SIX_PDB = STUDY_PDB + ("RO", "BG")
SHARED = "#1d4f91"   # fibre along a power line
LAND = "#5f9472"     # other land fibre
# ITU publishes links as straight lines between nodes, so a link "runs along" a
# line when most of it stays inside a band around a 330 kV or higher line. The
# band is a judgement, hence two widths and a range on the tile.
BANDS = (10.0, 15.0)
ALONG = 0.8
CELL = 15.0
KX = 111.32 * math.cos(math.radians(41.0))
KY = 110.57


def km(lon, lat):
    return lon * KX, lat * KY


def hv_lines():
    return [line for f in load_json(DIGITAL_DATA / "osm_hv_lines.geojson")["features"]
            if voltage_of(f["properties"]) >= 330 for line in lines_of(f["geometry"])]


def hv_grid(lines):
    """Points every kilometre along the lines, bucketed in CELL km squares."""
    grid = defaultdict(list)
    for line in lines:
        for a, b in zip(line, line[1:]):
            ax, ay = km(*a)
            bx, by = km(*b)
            n = max(1, int(math.hypot(bx - ax, by - ay)))
            for i in range(n + 1):
                x, y = ax + (bx - ax) * i / n, ay + (by - ay) * i / n
                grid[(int(x // CELL), int(y // CELL))].append((x, y))
    return grid


def near(grid, x, y, r):
    cx, cy = int(x // CELL), int(y // CELL)
    return any((px - x) ** 2 + (py - y) ** 2 <= r * r
               for i in (-1, 0, 1) for j in (-1, 0, 1)
               for px, py in grid.get((cx + i, cy + j), ()))


def frac_near(grid, line, r, step=2.0):
    """Share of points, sampled every step km along a polyline, within r km of the grid."""
    hits = total = 0
    for a, b in zip(line, line[1:]):
        (ax, ay), (bx, by) = km(*a), km(*b)
        n = max(1, int(math.hypot(bx - ax, by - ay) / step))
        for i in range(n):
            total += 1
            hits += near(grid, ax + (bx - ax) * i / n, ay + (by - ay) * i / n, r)
    return hits / total if total else 0.0


def band_of(grid, line):
    """The narrowest band the line runs along, None when neither."""
    for r in BANDS:
        if frac_near(grid, line, r) >= ALONG:
            return r
    return None


OSM_COVER = 5.0  # km: an ITU link this close to a traced OSM cable is drawn by the trace


def osm_cables():
    return [line for f in load_json(power.HERE / "data" / "osm_telecom_cables.geojson")["features"]
            for line in lines_of(f["geometry"])]


def road_lines():
    return [line for f in load_json(power.HERE / "data" / "ne_roads_region.geojson")["features"]
            for line in lines_of(f["geometry"])]


def fibre_links(index: CountryIndex):
    """ITU links, every status, with their length, country of the first node, the
    narrowest band they run along (None when neither) and a drawn route.

    ITU publishes links node to node. A link is drawn along the power line it follows,
    or not at all when a traced OpenStreetMap cable covers it, or along the road
    network. The band, not the drawn route, feeds the shares."""
    lines_hv = hv_lines()
    grid, graph = hv_grid(lines_hv), hv_graph(lines_hv)
    osm = osm_cables()
    osm_grid = hv_grid(osm)
    roads = hv_graph(road_lines())
    out = []
    for f in digital.data("itu_fibre.geojson")["features"]:
        status = f["properties"].get("status")
        for line in lines_of(f["geometry"]):
            (ax, ay), (bx, by) = km(*line[0]), km(*line[-1])
            length = math.hypot(bx - ax, by - ay)
            band = band_of(grid, [line[0], line[-1]])
            route = covered = None
            if band:
                route = route_along(graph, line, detour=2.0, slack=30.0)
            if not route:
                covered = frac_near(osm_grid, [line[0], line[-1]], OSM_COVER) >= ALONG
            if not route and not covered:
                route = (route_along(roads, line, snap=25.0, detour=2.0, slack=40.0)
                         or route_along(roads, line, snap=40.0, detour=3.0, slack=60.0)
                         or route_along(graph, line, snap=40.0, detour=3.0, slack=60.0))
            out.append({"line": line, "km": length, "band": band, "route": route,
                        "covered": covered, "country": index.at(*line[0]),
                        "status": status})
    return out, [(line, band_of(grid, line)) for line in osm]


SNAP = 3.0     # km: line ends this close to another line's vertex meet at a substation
DETOUR = 1.5   # a routed path longer than this times the straight link is rejected


def hv_graph(lines):
    """Vertices of the 330 kV+ lines as a graph in km, with line ends snapped to the
    nearest vertex of another line so that lines meeting at a substation connect."""
    node, xy, adj = {}, [], defaultdict(list)

    def nid(p):
        k = (round(p[0], 4), round(p[1], 4))
        if k not in node:
            node[k] = len(xy)
            xy.append(km(*p) + (p[0], p[1]))
        return node[k]

    owner = {}
    for li, line in enumerate(lines):
        ids = [nid(p) for p in line]
        for a, b in zip(ids, ids[1:]):
            if a != b:
                d = math.hypot(xy[a][0] - xy[b][0], xy[a][1] - xy[b][1])
                adj[a].append((b, d))
                adj[b].append((a, d))
        for i in ids:
            owner.setdefault(i, li)
    cells = defaultdict(list)
    for i, (x, y, *_) in enumerate(xy):
        cells[(int(x // CELL), int(y // CELL))].append(i)
    for li, line in enumerate(lines):
        for end in (nid(line[0]), nid(line[-1])):
            x, y = xy[end][:2]
            best = min(((math.hypot(xy[j][0] - x, xy[j][1] - y), j)
                        for i in (-1, 0, 1) for k in (-1, 0, 1)
                        for j in cells.get((int(x // CELL) + i, int(y // CELL) + k), ())
                        if owner[j] != li), default=(SNAP + 1, None))
            if best[0] <= SNAP:
                adj[end].append((best[1], best[0]))
                adj[best[1]].append((end, best[0]))
    return xy, adj, cells


def nearest_node(graph, x, y, r):
    xy, _, cells = graph
    n = int(r // CELL) + 1
    return min(((math.hypot(xy[j][0] - x, xy[j][1] - y), j)
                for i in range(-n, n + 1) for k in range(-n, n + 1)
                for j in cells.get((int(x // CELL) + i, int(y // CELL) + k), ())),
               default=(r + 1, None))


def route_along(graph, line, snap=BANDS[1], detour=DETOUR, slack=10.0):
    """The network path between the two ends of a link, as lon, lat, or None when
    the network offers no path close to the straight link (A* on km)."""
    xy, adj, _ = graph
    (ax, ay), (bx, by) = km(*line[0]), km(*line[-1])
    da, s = nearest_node(graph, ax, ay, snap)
    db, t = nearest_node(graph, bx, by, snap)
    if s is None or t is None or da > snap or db > snap or s == t:
        return None
    limit = detour * math.hypot(bx - ax, by - ay) + slack
    h = lambda i: math.hypot(xy[i][0] - xy[t][0], xy[i][1] - xy[t][1])
    g, prev, heap = {s: 0.0}, {}, [(h(s), s)]
    while heap:
        f, u = heapq.heappop(heap)
        if u == t:
            break
        if f > limit:
            return None
        for v, d in adj[u]:
            nd = g[u] + d
            if nd < g.get(v, math.inf):
                g[v], prev[v] = nd, u
                heapq.heappush(heap, (nd + h(v), v))
    if t not in g:
        return None
    path, u = [t], t
    while u != s:
        u = prev[u]
        path.append(u)
    return [xy[i][2:] for i in reversed(path)]


def in_six(index: CountryIndex, p) -> bool:
    """Inside one of the six countries, or at sea (a coastal vertex can miss the land)."""
    c = index.at(*p)
    return c is None or c in SIX


def clip(index: CountryIndex, line):
    """The runs of a polyline that stay in the six countries."""
    runs, run = [], []
    for p in line:
        if in_six(index, p):
            run.append(p)
            continue
        if len(run) > 1:
            runs.append(run)
        run = []
    if len(run) > 1:
        runs.append(run)
    return runs


def hv_six(index: CountryIndex):
    return [run for line in hv_lines() for run in clip(index, line)]


def bow(m: Map, a, b, bend=0.08) -> str:
    """A gentle arc for the few links no network can route, so none reads as surveyed."""
    (ax, ay), (bx, by) = m.xy(*a), m.xy(*b)
    cx, cy = (ax + bx) / 2 - (by - ay) * bend, (ay + by) / 2 + (bx - ax) * bend
    return f"M{ax:.1f},{ay:.1f}Q{cx:.1f},{cy:.1f} {bx:.1f},{by:.1f}"


def offset_path(m: Map, coords, off=2.2, min_px=1.2) -> str:
    """Pixel polyline shifted sideways by off pixels, so a fibre drawn on its power
    line sits beside it instead of on top."""
    pts = []
    for lon, lat in coords:
        p = m.xy(lon, lat)
        if not pts or math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) >= min_px:
            pts.append(p)
    if len(pts) < 2:
        return ""
    out = []
    for i, (x, y) in enumerate(pts):
        a, b = pts[max(0, i - 1)], pts[min(len(pts) - 1, i + 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy) or 1.0
        out.append(f"{x - dy / d * off:.1f},{y + dx / d * off:.1f}")
    return "M" + "L".join(out)


DC_RADIUS = 30.0  # km: PeeringDB writes districts (Sisli, Koropi) as cities


def dc_cities():
    """PeeringDB facilities with a location, grouped within DC_RADIUS of the
    largest city around, named after the most common city in the group."""
    sites = []
    for code, rows in digital.data("peeringdb.json")["fac"].items():
        for fac in rows:
            lat, lon = fac.get("latitude"), fac.get("longitude")
            if lat is None or lon is None:
                continue
            if not (BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]):
                continue
            if code not in SIX_PDB:
                continue
            sites.append((code, digital.city(fac.get("city") or ""), lon, lat))
    by_name = defaultdict(int)
    for s in sites:
        by_name[s[1]] += 1
    sites.sort(key=lambda s: (-(s[1] in CITY_NAME), -by_name[s[1]]))
    groups = []
    for code, name, lon, lat in sites:
        x, y = km(lon, lat)
        for g in groups:
            if g["code"] == code and math.hypot(g["x"] - x, g["y"] - y) <= DC_RADIUS:
                g["pts"].append((lon, lat))
                break
        else:
            groups.append({"code": code, "city": name, "x": x, "y": y, "pts": [(lon, lat)]})
    return [{"code": g["code"], "city": g["city"], "n": len(g["pts"]),
             "lon": sum(p[0] for p in g["pts"]) / len(g["pts"]),
             "lat": sum(p[1] for p in g["pts"]) / len(g["pts"])} for g in groups]


CITY_NAME = {"istanbul": "Istanbul", "bucharest": "Bucharest", "sofia": "Sofia",
             "ankara": "Ankara", "cluj napoca": "Cluj", "baku": "Baku", "yerevan": "Yerevan",
             "tbilisi": "Tbilisi", "chisinau": "Chisinau", "thessaloniki": "Thessaloniki",
             "varna": "Varna", "izmir": "Izmir"}
# label offsets in degrees and anchor, clear of the country names
CITY_LABEL = {"istanbul": (0.0, 0.62, "middle"), "bucharest": (0.0, 0.55, "middle"),
              "sofia": (0.0, 0.52, "middle"), "ankara": (-0.3, 0.3, "end"),
              "cluj napoca": (0.0, 0.45, "middle"), "baku": (0.35, -0.05, "start"),
              "yerevan": (0.0, -0.4, "middle"), "tbilisi": (0.25, 0.3, "start"),
              "chisinau": (0.3, -0.05, "start")}


SOFT = "#a8a297"  # background labels: no halo, under the lines
# text scale on these maps, so a screenshot stays legible at slide width (about 5 inches)
TXT = 1.6
_SIZE = re.compile(r'(font-size|dy)="([0-9.]+)"')


def scale_text(svg: str) -> str:
    return _SIZE.sub(lambda g: f'{g[1]}="{float(g[2]) * TXT:.1f}"', svg)


def soft_labels(m: Map):
    """Country and sea names drawn first, light and without halo, so they sit behind."""
    for lon, lat, name in power.COUNTRY_LABELS:
        focus = name.title() in {n.title() for n in SIX} | {"Turkiye"}
        m.label(lon, lat, name, size=9.5 if focus else 8.5, colour=SOFT if focus else "#c2bdb3",
                weight="600" if focus else "500")
    for lon, lat, name in [(34.0, 43.2, "Black Sea"), (51.3, 38.8, "Caspian Sea"),
                           (31.5, 35.8, "Mediterranean")]:
        m.label(lon, lat, name, size=10.5, colour="#9cbfd0", weight="400")


_GRAPH = []


def thin(line, step):
    """One vertex every step km, so short spurs and snaps vanish under the smoothing."""
    out = [line[0]]
    for p in line[1:-1]:
        (ax, ay), (bx, by) = km(*out[-1]), km(*p)
        if math.hypot(bx - ax, by - ay) >= step:
            out.append(p)
    return out + [line[-1]]


def planned_paths(m: Map, index: CountryIndex, p, cables, railways):
    """Planned interconnector routes: the BSSC and GEC as on the projects map, the
    Trans-Caspian on its fibre cable, the others along existing 330 kV+ lines where the
    network offers a path, else a smooth curve through the anchors."""
    if p.get("route") in ("bssc", "gec", "tripp"):
        return power.project_paths(m, index, p, cables, railways)
    if p["name"] == "Trans-Caspian":
        line = digital.cable_line("Trans-Caspian")
        return [m.path(line, min_px=0.5)], [line[0], line[-1]]
    if not _GRAPH:
        _GRAPH.append(hv_graph(hv_lines()))
    paths, ends = [], []
    for line in [p["anchors"]] + p.get("extra", []):
        route = route_along(_GRAPH[0], line, snap=20.0, detour=2.0, slack=30.0)
        if route:  # cut what overshoots the ends on the way to a snapped node
            d2 = lambda q, e: (q[0] - e[0]) ** 2 + (q[1] - e[1]) ** 2
            i0 = min(range(len(route)), key=lambda i: d2(route[i], line[0]))
            i1 = min(range(len(route)), key=lambda i: d2(route[i], line[-1]))
            route = route[i0:i1 + 1] if i0 < i1 else None
        if route and len(route) > 2:
            paths.append(m.smooth(thin(line[:1] + route + line[-1:], 25.0)))
        else:
            paths.append(m.smooth(line))
        ends += [line[0], line[-1]]
    return paths, ends


def dc_radius(n):
    return 2.4 + 1.9 * math.sqrt(n)


def dc_square(m: Map, x, y, half, title, hollow=False, opacity=0.85):
    """Data centres are rounded squares, plants are circles: the shape tells them apart."""
    if hollow:
        paint = (f'fill="#fff" fill-opacity="0.6" stroke="{digital.FAC}" stroke-width="1.6" '
                 f'stroke-opacity="{opacity}"')
    else:
        paint = (f'fill="{digital.FAC}" fill-opacity="{opacity}" stroke="#fff" '
                 'stroke-width="1.1"')
    m.add(f'<rect x="{x - half:.1f}" y="{y - half:.1f}" width="{2 * half:.1f}" '
          f'height="{2 * half:.1f}" rx="{min(2.2, half / 2):.1f}" {paint}>'
          f'<title>{esc(title)}</title></rect>')


def sw_square(col, hollow=False):
    if hollow:
        return (f'<i class="d" style="background:#fff;border:2px solid {col};'
                'border-radius:2px"></i>')
    return f'<i class="d" style="background:{col};border-radius:2px"></i>'


def link_d(m: Map, index: CountryIndex, l) -> str:
    """Drawn path of one ITU link, clipped to the six countries: beside its power line
    when it runs along one, on its own route otherwise, a gentle bow as a last resort."""
    if l["route"]:
        off = 2.2 if l["band"] else 0
        return "".join(offset_path(m, run, off) for run in clip(index, l["route"]))
    a, b = l["line"][0], l["line"][-1]
    if l["covered"] or not (in_six(index, a) and in_six(index, b)):
        return ""
    return bow(m, a, b)


def osm_d(m: Map, index: CountryIndex, line, off=2.2) -> str:
    return "".join(offset_path(m, run, off) for run in clip(index, line))


def ground_figures(links, cities) -> str:
    study = [l for l in links if l["country"] in STUDY and l["status"] == "Operational"]
    total = sum(l["km"] for l in study)
    share = [sum(l["km"] for l in study if l["band"] and l["band"] <= r) / total
             for r in BANDS]
    n_study = sum(c["n"] for c in cities if c["code"] in STUDY_PDB)
    by = {c["city"]: c["n"] for c in cities}
    west = sum(by.get(k, 0) for k in ("istanbul", "bucharest", "sofia"))
    east = sum(by.get(k, 0) for k in ("tbilisi", "baku", "yerevan"))
    tile_rows = [
        (f"{share[0]:.0%} to {share[1]:.0%}",
         "Of fibre km in the four countries that run along a 330 kV or higher line."),
        (f"{n_study} data centres", "Located in PeeringDB in the four countries. "
         f"{sum(c['n'] for c in cities if c['code'] == 'TR')} of them in Turkiye."),
        (f"{west} against {east}", "Data centres in Istanbul, Bucharest and Sofia, "
                                   "against Tbilisi, Baku and Yerevan."),
        ("2 landfalls", "Anaklia and Constanta take the BSSC power and fibre together."),
    ]
    top = sorted(cities, key=lambda c: -c["n"])[:9]
    chart = bars([(CITY_NAME.get(c["city"], c["city"].title()), c["n"], "#c5d3dc",
                   c["n"] if c["code"] in STUDY_PDB else 0) for c in top],
                 fmt=lambda v: f"{v:,.0f}", width=300)
    return headrow(tile_rows, "Data centres per city, study countries in dark", chart)


def ground_map(index: CountryIndex, links, cities) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index)
    m.add('<g class="zoomable">')
    soft_labels(m)
    batch(m, hv_six(index), f'stroke="{GRID}" stroke-width="1.2" stroke-opacity="0.75"', 1.2)
    links, osm = links
    links = [l for l in links if l["status"] == "Operational"]
    land = "".join(link_d(m, index, l) for l in links if not l["band"])
    land += "".join(osm_d(m, index, line, 0) for line, band in osm if not band)
    m.add(f'<path d="{land}" fill="none" stroke="{LAND}" stroke-width="0.9" '
          'stroke-opacity="0.15" stroke-linejoin="round"/>')
    for r, width, op in ((BANDS[1], 1.5, 0.6), (BANDS[0], 2.0, 0.95)):
        d = "".join(link_d(m, index, l) for l in links if l["band"] == r)
        d += "".join(osm_d(m, index, line) for line, band in osm if band == r)
        m.add(f'<path d="{d}" fill="none" stroke="{SHARED}" stroke-width="{width}" '
              f'stroke-opacity="{op}" stroke-linejoin="round"/>')
    digital.draw_cables(m, width=1.2, skip=("Trans-Caspian",), opacity=0.09)
    plants = load_json(power.HERE / "data" / "gem_plants.json")
    big = [p for p in plants if p.get("operating", 0) + p.get("construction", 0) >= 300
           and index.at(p["lon"], p["lat"]) in SIX]
    big.sort(key=lambda p: -(p.get("operating", 0) + p.get("construction", 0)))
    for p in big:
        mw = p.get("operating", 0) + p.get("construction", 0)
        col = power.PLANT_COLOUR.get(p["type"], "#999")
        x, y = m.xy(p["lon"], p["lat"])
        r = max(1.5, 0.13 * math.sqrt(mw))
        m.add(f'<circle class="pl" data-r="{r:.1f}" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
              f'fill="{col}" fill-opacity="0.4" stroke="#fff" stroke-width="0.5">'
              f'<title>{esc(p["name"])}, {mw:,.0f} MW</title></circle>')
    named = set()
    for c in sorted(cities, key=lambda c: -c["n"]):
        x, y = m.xy(c["lon"], c["lat"])
        name = CITY_NAME.get(c["city"], c["city"].title())
        dc_square(m, x, y, 0.9 * dc_radius(c["n"]), f'{name}: {c["n"]} data centres')
        if c["city"] in CITY_LABEL and c["city"] not in named:
            named.add(c["city"])
            dx, dy, anc = CITY_LABEL[c["city"]]
            m.label(c["lon"] + dx, c["lat"] + dy, name, size=9, colour="#5d6b72",
                    anchor=anc, weight="600")
    planned_layer(m, index)
    m.add("</g>")
    legend = html_legend([
        ("Power", [(sw_line(GRID, 2.4), "line, 330 kV+"),
                   (sw_dot("#5389AE"), "plant, 300 MW+")]),
        ("Data", [(sw_line(SHARED, 3), "fibre on a power line"),
                  (sw_line(SHARED, 2), "fibre near a power line"),
                  (sw_line(LAND, 2), "other fibre"),
                  (sw_line(digital.CABLE, 2), "submarine cable"),
                  (sw_square(digital.FAC), "data centres")]),
        ("Planned", [(sw_line(PALETTE["planned"], 3, PLAN_DASH), "interconnector"),
                     (sw_dot_ring("#5389AE"), "plant"),
                     (sw_square(digital.FAC, hollow=True), "data centre")]),
    ])
    cap = (f"A fibre link runs along a power line when {ALONG:.0%} of it stays inside the "
           "band, and is drawn along that line. ITU publishes links node to node, without "
           "the route: other links follow the cable where OpenStreetMap traced it, else "
           "the main roads, so their routes are indicative. Tick the planned boxes to add planned "
           "interconnectors, plants or data centres, one by one. Sources: OpenStreetMap, ITU, "
           "PeeringDB, Global Energy Monitor, Natural Earth, press reports.")
    toggle = ('<span class="ptog" style="margin-left:10px;font:12px/1 Segoe UI,sans-serif;'
              'color:#6f6a61">planned:') + "".join(
        f'<label style="margin-left:8px;cursor:pointer"><input type="checkbox" '
        f'onchange="this.closest(\'.imap-box\').querySelectorAll(\'.plan-layer[data-k={k}]\')'
        f'.forEach(function(g){{g.style.display=this.checked?\'\':\'none\'}},this)"> {t}</label>'
        for k, t in (("lines", "lines"), ("plants", "plants"), ("dc", "data centres"))) + "</span>"
    html = imap(scale_text(svg_of(m)), cap, legend, plants=True)
    return html.replace('data-z="reset">reset</button>', 'data-z="reset">reset</button>' + toggle, 1)


PLAN_DASH = "6 4"


def sw_dot_ring(col):
    return f'<i class="d" style="background:#fff;border:1.5px solid {col}"></i>'


def planned_layer(m: Map, index: CountryIndex):
    """Planned interconnectors, plants and data centres, hidden until ticked."""
    m.add('<g class="plan-layer" data-k="lines" style="display:none">')
    cables = load_json(DIGITAL_DATA / "submarine_cables.geojson")
    railways = load_json(DIGITAL_DATA / "osm_railways.geojson")
    col = PALETTE["planned"]
    for p in power.PROJECTS:
        if p["status"] != "planned":
            continue
        paths, ends = planned_paths(m, index, p, cables, railways)
        for path in paths:
            m.add(f'<path d="{path}" fill="none" stroke="#fff" stroke-width="5" '
                  'stroke-linecap="round" stroke-opacity="0.85"/>')
            m.add(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2.6" '
                  f'stroke-dasharray="{PLAN_DASH}" stroke-linecap="round"><title>'
                  f'{esc(p["name"])}, {esc(p["mw"])} MW, {esc(p["year"])}</title></path>')
        for lon, lat in ends:
            x, y = m.xy(lon, lat)
            m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="#fff" stroke="{col}" '
                  'stroke-width="1.6"/>')
    m.add("</g>")
    m.add('<g class="plan-layer" data-k="plants" style="display:none">')
    plants = load_json(power.HERE / "data" / "gem_plants.json")
    for q in plants:
        mw = q.get("pre_construction", 0) + q.get("announced", 0)
        if mw < 300 or index.at(q["lon"], q["lat"]) not in SIX:
            continue
        x, y = m.xy(q["lon"], q["lat"])
        r = max(1.5, 0.13 * mw ** 0.5)
        m.add(f'<circle class="pl" data-r="{r:.1f}" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
              f'fill="#fff" fill-opacity="0.5" stroke="{power.PLANT_COLOUR.get(q["type"], "#999")}" '
              f'stroke-width="1.4"><title>{esc(q["name"])}, {mw:,.0f} MW planned</title></circle>')
    m.add("</g>")
    m.add('<g class="plan-layer" data-k="dc" style="display:none">')
    for dc in PLANNED_DC:
        x, y = m.xy(*dc["lonlat"])
        size = f'{dc["mw"]:,} MW' if dc["mw"] else "MW n/a"
        dc_square(m, x, y, planned_half(dc["mw"]),
                  f'{dc["name"]}, {dc["site"]}: {size}, {dc["year"]}', hollow=True)
    m.add("</g>")


# ---------------------------------------------------------------------------
# what is planned
# ---------------------------------------------------------------------------
# Announced sites with a location. MW is the announced IT or grid load, None when not
# published. Georgia has no announced site with a size.
PLANNED_DC = [
    {"name": "Black Sea AI Gigafactory, phase I", "site": "Cernavoda", "lonlat": (28.05, 44.33),
     "mw": 1500, "year": "2028", "note": "1.5 GW across both phases",
     "source": "Press, government statements, 2025"},
    {"name": "Black Sea AI Gigafactory, phase II", "site": "Doicesti", "lonlat": (25.40, 44.99),
     "mw": None, "year": "after 2028", "note": "Beside the planned SMR",
     "source": "Press, government statements, 2025"},
    {"name": "BRAIN++ EU AI factory", "site": "Sofia Tech Park", "lonlat": (23.37, 42.67),
     "mw": None, "year": "2026 to 2028", "note": "EUR 90m, EuroHPC",
     "source": "Sofia Tech Park, 2025"},
    {"name": "Khazna and G42", "site": "Baskent OIZ, Ankara", "lonlat": (32.43, 39.86),
     "mw": 100, "year": "n/a", "note": "Up to 100 MW", "source": "Turkiye market review"},
    {"name": "Trendyol Castle", "site": "Temelli, Ankara", "lonlat": (32.36, 39.73),
     "mw": 48, "year": "2026", "note": "48 MW IT", "source": "Turkiye market review"},
    {"name": "Turksat", "site": "Golbasi, Ankara", "lonlat": (32.81, 39.79),
     "mw": 33, "year": "2028", "note": "33 MVA", "source": "Turkiye market review"},
    {"name": "Digital Realty and Ronesans", "site": "Ankara", "lonlat": (32.86, 39.95),
     "mw": 22, "year": "2028", "note": "22 MW IT and more", "source": "Turkiye market review"},
    {"name": "Vodafone and DAMAC", "site": "Izmir", "lonlat": (27.14, 38.42),
     "mw": 20, "year": "n/a", "note": "20 MW target", "source": "Turkiye market review"},
    {"name": "Firebird AI factory", "site": "Hrazdan", "lonlat": (44.77, 40.50),
     "mw": 300, "year": "2027", "note": "100 MW opened 2026, 300 MW targeted",
     "source": "Firebird release and press, 2026"},
    {"name": "AzInTelecom, Tier III", "site": "Absheron", "lonlat": (49.75, 40.45),
     "mw": None, "year": "n/a", "note": "EIB loan, EUR 43m for both sites",
     "source": "Press, EIB loan, 2025"},
    {"name": "AzInTelecom, Tier III", "site": "Hajigabul", "lonlat": (48.94, 40.04),
     "mw": None, "year": "n/a", "note": "EIB loan, EUR 43m for both sites",
     "source": "Press, EIB loan, 2025"},
]
UNSITED_DC = "Google Cloud with Turkcell, 2028 to 2029, and Alibaba Cloud, 2027, in Turkiye."
FIBRE_DASH = {"Under Construction": "7 4", "Planned": "1.5 3"}
# corridors that carry both planned power and fibre, drawn with a soft halo
SYNERGY = ("BSSC", "Trans-Caspian")
# label (lon, lat), text and anchor for the planned items
PLANNED_LABELS = [
    (28.35, 44.55, "Gigafactory, 1.5 GW", "start"),
    (25.4, 44.7, "Doicesti", "middle"),
    (22.4, 42.35, "Sofia AI factory", "start"),
    (33.0, 39.98, "Ankara cluster, about 200 MW", "start"),
    (27.45, 38.33, "Izmir", "start"),
    (45.05, 40.42, "Hrazdan, 300 MW", "start"),
    (49.0, 39.62, "Absheron, Hajigabul", "middle"),
    (31.0, 43.15, "BSSC", "middle"),
    (31.0, 42.2, "GEC", "middle"),
    (50.6, 42.6, "Trans-Caspian", "end"),
    (46.3, 38.95, "TRIPP", "start"),
]


def planned_radius(mw):
    return 3.6 if mw is None else 3.0 + 0.38 * math.sqrt(mw)


def planned_half(mw):
    return 3.2 if mw is None else 2.5 + 0.22 * math.sqrt(mw)


def hollow_squares(sizes):
    out = []
    for mw in sizes:
        h = planned_half(mw)
        out.append(f'<svg width="{2 * h + 4:.0f}" height="{2 * h + 4:.0f}" '
                   f'style="vertical-align:middle"><rect x="2" y="2" width="{2 * h:.1f}" '
                   f'height="{2 * h:.1f}" rx="{min(2.2, h / 2):.1f}" fill="none" '
                   f'stroke="{digital.FAC}" stroke-width="1.6"/></svg>')
    return "".join(out)


def planned_map(index: CountryIndex, links, cities) -> str:
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index)
    m.add('<g class="zoomable">')
    soft_labels(m)
    batch(m, hv_six(index), f'stroke="{GRID}" stroke-width="1.0" stroke-opacity="0.3"', 1.2)
    links, osm = links
    today = [l for l in links if l["status"] == "Operational"]
    d = "".join(link_d(m, index, l) for l in today)
    d += "".join(osm_d(m, index, line, 2.2 if band else 0) for line, band in osm)
    m.add(f'<path d="{d}" fill="none" stroke="{SHARED}" stroke-width="1.0" '
          'stroke-opacity="0.18" stroke-linejoin="round"/>')
    digital.draw_cables(m, width=1.2, skip=("Trans-Caspian",), opacity=0.15)
    for c in cities:
        x, y = m.xy(c["lon"], c["lat"])
        dc_square(m, x, y, 0.9 * dc_radius(c["n"]), f'{c["n"]} data centres', opacity=0.22)

    cables = load_json(DIGITAL_DATA / "submarine_cables.geojson")
    railways = load_json(DIGITAL_DATA / "osm_railways.geojson")
    projects = {p["name"]: p for p in power.PROJECTS}
    # one element, so the halo does not darken where pieces overlap
    halo = "".join(path for name in SYNERGY
                   for path in planned_paths(m, index, projects[name], cables, railways)[0])
    m.add(f'<path d="{halo}" fill="none" stroke="{SHARED}" stroke-width="16" '
          'stroke-opacity="0.09" stroke-linecap="round" stroke-linejoin="round"/>')
    ends = []
    for p in power.PROJECTS:
        if p["status"] != "planned":
            continue
        col, dash, _ = power.STATUS_STYLE[p["status"]]
        paths, e = planned_paths(m, index, p, cables, railways)
        ends += [(pt, col) for pt in e]
        for path in paths:
            m.add(f'<path d="{path}" fill="none" stroke="#fff" stroke-width="5" '
                  'stroke-linecap="round" stroke-opacity="0.85"/>')
            m.add(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="3" '
                  f'stroke-dasharray="{dash}" stroke-linecap="round"><title>'
                  f'{esc(p["name"])}, {esc(p["mw"])} MW, {esc(p["year"])}</title></path>')
    for (lon, lat), col in ends:
        x, y = m.xy(lon, lat)
        m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#fff" stroke="{col}" '
              'stroke-width="1.8"/>')
    for status, dash in FIBRE_DASH.items():
        if status != "Planned":
            continue
        d = "".join(link_d(m, index, l) for l in links if l["status"] == status)
        m.add(f'<path d="{d}" fill="none" stroke="{SHARED}" stroke-width="1.8" '
              f'stroke-dasharray="{dash}" stroke-linecap="round" stroke-linejoin="round"/>')
    riding = [
        (offset_path(m, digital.bssc_route(), 4.5, 0.8), "1.5 3", "BSSC fibre, planned"),
        (offset_path(m, digital.cable_line("Trans-Caspian"), 4.5, 0.8), "10 5",
         "Trans-Caspian fibre, laid 2026"),
    ]  # the TRIPP duct shares the power line's trench: the halo shows it
    for path, dash, title in riding:
        m.add(f'<path d="{path}" fill="none" stroke="{SHARED}" stroke-width="1.8" '
              f'stroke-dasharray="{dash}" stroke-linecap="round"><title>{esc(title)}'
              '</title></path>')
    for dc in sorted(PLANNED_DC, key=lambda dc: -(dc["mw"] or 0)):
        x, y = m.xy(*dc["lonlat"])
        size = f'{dc["mw"]:,} MW' if dc["mw"] else "MW n/a"
        dc_square(m, x, y, planned_half(dc["mw"]),
                  f'{dc["name"]}, {dc["site"]}: {size}, {dc["year"]}', hollow=True)
    for lon, lat, text, anc in PLANNED_LABELS:
        m.label(lon, lat, text, size=8.5, colour="#4f5a60", anchor=anc, weight="600",
                halo=True)
    m.add("</g>")
    rings = hollow_squares((50, 300, 1500))
    legend = html_legend([
        ("Power", [(sw_line(PALETTE["planned"], 3, "1.5 4"), "interconnector"),
                   (sw_line(GRID, 2), "existing line")]),
        ("Data", [(sw_line(SHARED, 2, "1.5 3"), "fibre planned"),
                  (sw_line(SHARED, 2, "10 5"), "Trans-Caspian fibre"),
                  (rings, "announced, 50 to 1,500 MW"),
                  (sw_square("#b0b8bc"), "today")]),
        ("Both", [(sw_line("#c9d6e6", 8), "power and fibre")]),
    ])
    cap = ("Planned power and fibre over today's network, drawn pale. Planned "
           "interconnectors follow existing 330 kV and higher lines where the network offers "
           "a path, else a schematic curve. The BSSC is traced on the Caucasus Cable System "
           "with landfalls at Anaklia and Constanta, the Trans-Caspian on its fibre cable. "
           "Planned ITU fibre links are routed as on the map above. Data centres: announced "
           "sites only, locations approximate, sizes as announced. Without a published "
           f"site: {UNSITED_DC} Sources: ITU, OpenStreetMap, PeeringDB, press reports and "
           "operator releases 2025 to 2026, Turkiye data centre market review.")
    rows = []
    for dc in PLANNED_DC:
        size = f"{dc['mw']:,}" if dc["mw"] else "n/a"
        rows.append(
            f"<tr><td><b>{esc(dc['name'])}</b><br><span class='small'>{esc(dc['site'])}"
            f"</span></td><td class='n'>{size}</td><td>{esc(dc['year'])}</td>"
            f"<td class='small'>{esc(dc['note'])}</td>"
            f"<td class='small'>{esc(dc['source'])}</td></tr>")
    table = ('<table class="kt pt"><thead><tr><th>Announced data centre</th>'
             '<th class="n">MW</th><th>Entry</th><th>Note</th><th>Source</th></tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table>')
    return imap(scale_text(svg_of(m)), cap, legend) + table


# ---------------------------------------------------------------------------
# cheap clean power, short data path
# ---------------------------------------------------------------------------
MODEL = load_json(power.HERE / "data" / "model_power.json")
CHEAP_COL, DEAR_COL = (0, 131, 143), (200, 98, 43)
PRICE_OF = {"Georgia": "Georgia", "Armenia": "Armenia", "Azerbaijan": "Azerbaijan",
            "Turkey": "Turkiye", "Romania": "Romania", "Bulgaria": "Bulgaria"}
# price label (lon, lat) per basemap country
PRICE_AT = {"Georgia": (43.2, 42.75), "Armenia": (45.9, 40.75), "Azerbaijan": (48.3, 41.4),
            "Turkey": (35.0, 38.9), "Romania": (25.0, 45.9), "Bulgaria": (26.4, 42.85)}
CITY = {"Tbilisi": (44.80, 41.72), "Baku": (49.87, 40.41), "Yerevan": (44.51, 40.18),
        "Ganja": (46.36, 40.68), "Akhaltsikhe": (42.98, 41.64), "Erzurum": (41.27, 39.90),
        "Ankara": (32.85, 39.93), "Istanbul": (28.98, 41.01), "Edirne": (26.56, 41.68),
        "Plovdiv": (24.75, 42.15), "Sofia": (23.32, 42.70), "Ruse": (25.97, 43.85),
        "Bucharest": (26.10, 44.43), "Constanta": (28.65, 44.17), "Nakhchivan": (45.41, 39.21),
        "Igdir": (44.04, 39.92), "Horadiz": (47.03, 39.45), "Aktau": (51.17, 43.65),
        "Anaklia": (41.57, 42.40), "Kutaisi": (42.70, 42.27), "Ardahan": (42.70, 41.11),
        "Sivas": (37.02, 39.75)}
# corridor: city chain, power status, fibre status (None when the sector is absent)
CORRIDORS = [
    ("Baku", "Ganja", "Tbilisi", "Akhaltsikhe", "Ardahan", "Erzurum", "Sivas", "Ankara", "Istanbul",
     "service", "service"),
    ("Istanbul", "Edirne", "Plovdiv", "Sofia", "service", "service"),
    ("Sofia", "Ruse", "Bucharest", "Constanta", "service", "service"),
    ("Tbilisi", "Yerevan", "service", "service"),
    ("Baku", "Horadiz", "Nakhchivan", "Igdir", "committed", "planned"),
    ("Baku", "Aktau", "planned", "committed"),
    ("Tbilisi", "Kutaisi", "Anaklia", "service", "service"),
]
DASH = {"service": None, "committed": "7 4", "planned": "1.5 4"}
VISUAL_LEAD = (
    '<div class="msgs"><div class="msg"><div><b>Cheap power on one shore, data centres on '
    'the other.</b> Power in the Caucasus costs about 40 $/MWh, against about 100 $/MWh in '
    "Romania and Bulgaria. Most data centres sit on the expensive side. Power and fibre "
    "laid together can bring the cheap power to the data, or the data to the cheap power."
    "</div></div></div>")


def price_colour(price: float) -> str:
    t = min(1.0, max(0.0, (price - 40.0) / 60.0))
    return "#" + "".join(f"{round(a + (b - a) * t):02x}" for a, b in zip(CHEAP_COL, DEAR_COL))


def price_of(name):
    key = PRICE_OF[name]
    if key in MODEL["countries"]:
        return MODEL["countries"][key]["price"]
    return MODEL["eu_border_price"][key]


def spline(coords, n=14):
    """Catmull-Rom through the cities, densified, in lon and lat."""
    out = []
    for i in range(len(coords) - 1):
        p0 = coords[i - 1] if i else coords[0]
        p1, p2 = coords[i], coords[i + 1]
        p3 = coords[i + 2] if i + 2 < len(coords) else p2
        for k in range(n):
            t = k / n
            out.append(tuple(
                0.5 * (2 * p1[j] + (-p0[j] + p2[j]) * t
                       + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t * t
                       + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t ** 3)
                for j in (0, 1)))
    return out + [coords[-1]]


def strand(m: Map, coords, colour, status, off, width=2.6):
    d = offset_path(m, coords, off, 0.8)
    dash = f' stroke-dasharray="{DASH[status]}"' if DASH[status] else ""
    m.add(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}" '
          f'stroke-linecap="round" stroke-linejoin="round"{dash}/>')


def visual_map(index: CountryIndex, cities) -> str:
    year = MODEL["year"]
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index, highlight=())
    for name, poly, *_ in index.entries:
        if name not in PRICE_OF:
            continue
        d = " ".join(r + "Z" for r in (m.path(ring, min_px=0.5) for ring in poly) if r)
        if d:
            m.add(f'<path d="{d}" fill="{price_colour(price_of(name))}" fill-opacity="0.28" '
                  f'stroke="#fff" stroke-width="0.8" fill-rule="evenodd"/>')
    m.add('<g class="zoomable">')
    # corridors: power and fibre side by side, through the real cities
    for c in CORRIDORS:
        *names, pw, fb = c
        pts = spline([CITY[n] for n in names])
        m.add(f'<path d="{offset_path(m, pts, 0, 0.8)}" fill="none" stroke="#fff" '
              'stroke-width="11" stroke-opacity="0.75" stroke-linecap="round"/>')
        if pw:
            strand(m, pts, GRID, pw, -2.2)
        if fb:
            strand(m, pts, SHARED, fb, 2.2)
    bssc = digital.bssc_route()
    m.add(f'<path d="{offset_path(m, bssc, 0, 0.8)}" fill="none" stroke="#fff" '
          'stroke-width="11" stroke-opacity="0.6" stroke-linecap="round"/>')
    strand(m, bssc, GRID, "planned", -2.2)
    strand(m, bssc, SHARED, "planned", 2.2)
    ccs = digital.cable_line(power.CCS)
    if ccs:
        strand(m, ccs, SHARED, "service", 9, width=1.6)
    # data centres today, by city, and announced sites
    for c in sorted(cities, key=lambda c: -c["n"]):
        if c["n"] < 2:
            continue
        x, y = m.xy(c["lon"], c["lat"])
        m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{dc_radius(c["n"]):.1f}" '
              f'fill="{digital.FAC}" fill-opacity="0.85" stroke="#fff" stroke-width="1.2">'
              f'<title>{c["n"]} data centres</title></circle>')
    for dc in sorted(PLANNED_DC, key=lambda dc: -(dc["mw"] or 0)):
        x, y = m.xy(*dc["lonlat"])
        m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{planned_radius(dc["mw"]):.1f}" '
              f'fill="#fff" fill-opacity="0.5" stroke="{digital.FAC}" stroke-width="1.8">'
              f'<title>{esc(dc["name"])}, {esc(dc["site"])}</title></circle>')
    # prices, then a few names
    font = 'font-family="Segoe UI, Arial, sans-serif"'
    for name, (lon, lat) in PRICE_AT.items():
        x, y = m.xy(lon, lat)
        price = price_of(name)
        m.add(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" font-size="10" '
              f'font-weight="700" fill="#5d5850" stroke="#fff" stroke-width="3" '
              f'paint-order="stroke" {font}>{esc(PRICE_OF[name].upper())}</text>')
        m.add(f'<text x="{x:.1f}" y="{y + 19 * TXT:.1f}" text-anchor="middle" font-size="17" '
              f'font-weight="700" fill="{price_colour(price)}" stroke="#fff" stroke-width="3.5" '
              f'paint-order="stroke" {font}>{price:.0f}<tspan font-size="9" '
              f'font-weight="600"> $/MWh</tspan></text>')
    for lon, lat, text, anc in VISUAL_LABELS:
        m.label(lon, lat, text, size=9.5, colour=PALETTE["ink"], anchor=anc, weight="700",
                halo=True)
    m.add("</g>")
    ramp = "".join(f'<span style="display:inline-block;width:16px;height:10px;'
                   f'background:{price_colour(v)};opacity:0.75"></span>'
                   for v in (40, 55, 70, 85, 100))
    rings = "".join(
        f'<svg width="{2 * r + 4:.0f}" height="{2 * r + 4:.0f}" style="vertical-align:middle">'
        f'<circle cx="{r + 2:.1f}" cy="{r + 2:.1f}" r="{r:.1f}" fill="none" '
        f'stroke="{digital.FAC}" stroke-width="1.6"/></svg>'
        for r in (planned_radius(50), planned_radius(300)))
    legend = html_legend([
        ("Power price", [(ramp, "40 to 100 $/MWh")]),
        ("Corridors", [(sw_line(GRID, 3), "power"), (sw_line(SHARED, 3), "fibre"),
                       (sw_line("#6f6a61", 3), "in service"),
                       (sw_line("#6f6a61", 3, dash="7 4"), "committed"),
                       (sw_line("#6f6a61", 3, dash="1.5 4"), "planned")]),
        ("Data centres", [(sw_dot(digital.FAC), "today, by city"), (rings, "announced")]),
    ])
    cap = (f"Projected average wholesale power prices in {year}. Corridors are schematic, "
           "drawn through the cities they serve. BSSC traced on the Caucasus Cable System, "
           "landfalls at Anaklia and Constanta. Data centres: PeeringDB and announcements, "
           "as on the map above.")
    return VISUAL_LEAD + imap(scale_text(svg_of(m)), cap, legend)


VISUAL_LABELS = [
    (33.0, 42.35, "BSSC", "middle"),
    (34.0, 43.5, "Caucasus Cable System", "middle"),
    (50.6, 43.1, "Trans-Caspian", "end"),
    (45.9, 38.75, "TRIPP", "middle"),
    (28.98, 41.45, "Istanbul", "middle"),
    (32.85, 40.4, "Ankara", "middle"),
    (26.1, 44.85, "Bucharest", "middle"),
    (23.32, 43.1, "Sofia", "middle"),
    (44.95, 41.95, "Tbilisi", "start"),
    (50.1, 40.15, "Baku", "start"),
    (44.35, 40.3, "Yerevan", "end"),
]

# ---------------------------------------------------------------------------
# the same story, city by city
# ---------------------------------------------------------------------------
# station: lon, lat, label dx and dy in pixels, anchor; hubs are written bold
STATION = {
    "Baku": (49.87, 40.41, 10, 6, "start"), "Mingachevir": (47.06, 40.77, 0, 12, "middle"),
    "Ganja": (46.36, 40.68, 0, -8, "middle"), "Tbilisi": (44.80, 41.72, 9, -6, "start"),
    "Akhaltsikhe": (42.98, 41.64, -7, 4, "end"), "Kutaisi": (42.70, 42.27, 6, 12, "start"),
    "Enguri": (42.03, 42.65, 0, -9, "middle"), "Anaklia": (41.57, 42.40, -8, 4, "end"),
    "Hrazdan": (44.77, 40.50, -8, -20, "end"), "Yerevan": (44.51, 40.18, 0, 14, "middle"),
    "Ardahan": (42.70, 41.11, -7, 4, "end"), "Erzurum": (41.27, 39.90, 0, 16, "middle"),
    "Erzincan": (39.49, 39.75, 0, 16, "middle"), "Sivas": (37.02, 39.75, 0, -9, "middle"),
    "Kirikkale": (33.52, 39.85, 0, 16, "middle"), "Ankara": (32.85, 39.93, -2, -30, "middle"),
    "Bolu": (31.61, 40.73, 0, -9, "middle"), "Izmit": (29.92, 40.77, 0, -9, "middle"),
    "Istanbul": (28.98, 41.01, 0, -14, "middle"), "Bursa": (29.06, 40.19, 9, 4, "start"),
    "Izmir": (27.14, 38.42, -10, 4, "end"), "Edirne": (26.56, 41.68, 8, 14, "start"),
    "Plovdiv": (24.75, 42.15, 6, 14, "start"), "Sofia": (23.32, 42.70, 0, 22, "middle"),
    "Burgas": (27.47, 42.50, 9, 4, "start"), "Varna": (27.91, 43.22, -9, 4, "end"),
    "Pleven": (24.62, 43.41, 0, -9, "middle"), "Ruse": (25.97, 43.85, 0, 16, "middle"),
    "Bucharest": (26.10, 44.43, -6, -12, "end"), "Cernavoda": (28.05, 44.33, 12, -32, "start"),
    "Constanta": (28.65, 44.17, 9, 12, "start"), "Horadiz": (47.03, 39.45, 8, 4, "start"),
    "Nakhchivan": (45.41, 39.21, 0, 16, "middle"), "Igdir": (44.04, 39.92, 0, 0, None),
    "Aktau": (51.17, 43.65, -8, 4, "end"),
}
HUBS = {"Baku", "Tbilisi", "Yerevan", "Ankara", "Istanbul", "Sofia", "Bucharest",
        "Anaklia", "Constanta"}
NETWORK = [
    ("Baku", "Mingachevir", "Ganja", "Tbilisi", "service", "service"),
    ("Tbilisi", "Akhaltsikhe", "Ardahan", "Erzurum", "Erzincan", "Sivas", "Kirikkale",
     "Ankara", "service", "service"),
    ("Ankara", "Bolu", "Izmit", "Istanbul", "service", "service"),
    ("Izmit", "Bursa", "Izmir", "service", "service"),
    ("Istanbul", "Edirne", "Plovdiv", "Sofia", "service", "service"),
    ("Plovdiv", "Burgas", "Varna", "Constanta", "service", "service"),
    ("Sofia", "Pleven", "Ruse", "Bucharest", "Cernavoda", "Constanta", "service", "service"),
    ("Tbilisi", "Hrazdan", "Yerevan", "service", "service"),
    ("Tbilisi", "Kutaisi", "Enguri", "Anaklia", "service", "service"),
    ("Baku", "Horadiz", "Nakhchivan", "Igdir", "committed", "planned"),
    ("Baku", "Aktau", "planned", "committed"),
]
# announced sites, grouped on the nearest station; note under the station name
ANNOUNCED = {
    "Cernavoda": (1500, "AI gigafactory, 1.5 GW"), "Hrazdan": (300, "AI factory,|300 MW"),
    "Ankara": (200, "four sites, about 200 MW"), "Sofia": (None, "EU AI factory"),
    "Baku": (None, "two Tier III|sites"), "Izmir": (20, "20 MW"),
}
COUNTRY_TAG = [  # basemap name, lon, lat
    ("Turkey", 35.0, 38.6), ("Georgia", 43.6, 42.95), ("Armenia", 46.2, 39.9),
    ("Azerbaijan", 48.3, 41.8), ("Bulgaria", 25.5, 42.95), ("Romania", 24.0, 45.85),
]


def network_map(index: CountryIndex, cities) -> str:
    year = MODEL["year"]
    m = Map(width=W, height=H, bbox=BBOX)
    m.basemap(index)
    m.add('<g class="zoomable">')
    font = 'font-family="Segoe UI, Arial, sans-serif"'
    for lon, lat, name in power.COUNTRY_LABELS:
        if name.title() not in {"Greece", "Russia", "Iran", "Ukraine", "Moldova", "Iraq",
                                "Syria", "Kazakhstan"}:
            continue
        m.label(lon, lat, name, size=8.5, colour="#c2bdb3", weight="500")
    for name, lon, lat in COUNTRY_TAG:
        x, y = m.xy(lon, lat)
        m.add(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" font-size="9.5" '
              f'font-weight="700" letter-spacing="1.2" fill="{SOFT}" '
              f'{font}>{esc(PRICE_OF[name].upper())}'
              f'<tspan x="{x:.1f}" dy="12" font-size="8.5" font-weight="500" '
              f'letter-spacing="0" fill="#6f6a61">{price_of(name):.0f} $/MWh</tspan></text>')
    m.label(34.6, 43.9, "Black Sea", size=10.5, colour="#7aa6bd", weight="400")
    m.label(51.3, 38.7, "Caspian Sea", size=10.5, colour="#7aa6bd", weight="400")

    def pts_of(names):
        return spline([STATION[n][:2] for n in names], n=10)

    lines = []
    for c in NETWORK:
        *names, pw, fb = c
        lines.append((pts_of(names), pw, fb))
    lines.append((digital.bssc_route(), "planned", "planned"))
    for pts, *_ in lines:
        m.add(f'<path d="{offset_path(m, pts, 0, 0.8)}" fill="none" stroke="#fff" '
              'stroke-width="7" stroke-opacity="0.8" stroke-linecap="round" '
              'stroke-linejoin="round"/>')
    ccs = digital.cable_line(power.CCS)
    if ccs:
        strand(m, ccs, SHARED, "service", 8, width=1.1)
    for pts, pw, fb in lines:
        strand(m, pts, GRID, pw, -1.5, width=1.7)
        strand(m, pts, SHARED, fb, 1.5, width=1.7)

    # data centres today, gathered on the nearest station within reach
    count, loose = defaultdict(int), []
    for c in cities:
        near = min(STATION, key=lambda k: (STATION[k][0] - c["lon"]) ** 2
                   + (STATION[k][1] - c["lat"]) ** 2)
        d2 = (STATION[near][0] - c["lon"]) ** 2 + (STATION[near][1] - c["lat"]) ** 2
        if d2 < 0.45 ** 2:
            count[near] += c["n"]
        elif c["n"] >= 2:
            loose.append(c)
    for c in loose:
        x, y = m.xy(c["lon"], c["lat"])
        name = CITY_NAME.get(c["city"], c["city"].title())
        dc_square(m, x, y, 0.9 * dc_radius(c["n"]), f'{name}: {c["n"]} data centres',
                  opacity=0.55)
        if c["n"] >= 5:
            m.label(c["lon"], c["lat"] + 0.3, name, size=8, colour="#5d5850", halo=True)
    for name, (lon, lat, dx, dy, anc) in STATION.items():
        x, y = m.xy(lon, lat)
        n = count.get(name, 0)
        if name in ANNOUNCED:
            mw, _ = ANNOUNCED[name]
            h = planned_half(mw)
            ox = (0.9 * dc_radius(n) + h + 1.5) if n else 0
            dc_square(m, x + ox, y, h, f"{name}: announced", hollow=True)
        if n:
            half = 0.9 * dc_radius(n)
            dc_square(m, x, y, half, f"{name}: {n} data centres")
        elif name not in ANNOUNCED:
            r = 3.2 if name in HUBS else 2.4
            m.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="#fff" stroke="#6f6a61" '
                  'stroke-width="1.2"/>')
    for name, (lon, lat, dx, dy, anc) in STATION.items():
        x, y = m.xy(lon, lat)
        hub = name in HUBS
        if anc is None:
            continue
        note = ANNOUNCED.get(name, (None, ""))[1]
        sub = "".join(f'<tspan x="{x + dx:.1f}" dy="10" font-size="7.5" font-weight="500" '
                      f'font-style="italic" fill="{digital.FAC}">{esc(part)}</tspan>'
                      for part in note.split("|") if part)
        dy = dy * TXT if dy > 0 else dy  # labels below grow with the scaled text
        m.add(f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anc}" '
              f'font-size="{9 if hub else 8}" font-weight="{700 if hub else 500}" '
              f'fill="{"#4f5a60" if hub else "#7d776d"}" {font}>{esc(name)}{sub}</text>')
    for lon, lat, text, anc in NETWORK_LABELS:
        m.label(lon, lat, text, size=8, colour="#6f6a61", anchor=anc, weight="600",
                halo=True)
    m.add("</g>")
    legend = html_legend([
        ("Corridors", [(sw_line(GRID, 2), "power"), (sw_line(SHARED, 2), "fibre"),
                       (sw_line("#6f6a61", 2), "in service"),
                       (sw_line("#6f6a61", 2, dash="7 4"), "committed"),
                       (sw_line("#6f6a61", 2, dash="1.5 4"), "planned")]),
        ("Data centres", [(sw_square(digital.FAC), "today"),
                          (sw_square(digital.FAC, hollow=True), "announced")]),
    ])
    cap = (f"Average wholesale power price per country, projected for {year}. Corridors are "
           "schematic, drawn through the cities and plants they serve; the BSSC is traced on "
           "the Caucasus Cable System, landfalls at Anaklia and Constanta. Data centres: "
           "PeeringDB and announcements, as on the maps above.")
    return imap(scale_text(svg_of(m)), cap, legend)


NETWORK_LABELS = [
    (35.5, 42.0, "BSSC, power and fibre", "middle"),
    (35.5, 42.85, "Caucasus Cable System", "middle"),
    (50.1, 42.75, "Trans-Caspian", "start"),
    (46.1, 38.85, "TRIPP", "middle"),
]


def digital_power(index: CountryIndex) -> str:
    links, cities = fibre_links(index), dc_cities()
    return "".join([
        '<style>.dp .lg,.dp .lgh{font-size:16px}.dp .mapcap{font-size:15px}'
        '.dp .lg i.d{width:12px;height:12px}.dp .lg i.ln{width:20px;height:3px}'
        '.dp .imap-box{flex-direction:row;align-items:flex-start;gap:14px}'
        '.dp .imap-main{flex:1 1 auto;min-width:0}.dp .imap-side{flex:0 0 190px;'
        'padding-top:36px}.dp .lgrow{flex-direction:column;align-items:flex-start;gap:5px}'
        '.dp .lgh{min-width:0}</style>'
        '<div class="dp">',
        '<h4 class="mh">What we want to say</h4>', message(),
        '<h4 class="mh">Where data meets power</h4>', ground_figures(links[0], cities),
        ground_map(index, links, cities),
        '<h4 class="mh">What is planned</h4>', planned_map(index, links, cities),
        '<h4 class="mh">Cheap power, short data path</h4>', visual_map(index, cities),
        '<h4 class="mh">The same story, city by city</h4>', network_map(index, cities),
        '<h4 class="mh">Key points</h4>', key_points(), "</div>",
    ])


def key_points() -> str:
    lines = sum(p["status"] in ("planned", "committed") for p in power.PROJECTS)
    announced = sum(dc["mw"] or 0 for dc in PLANNED_DC)
    points = [
        f"<b>Many power lines in the pipeline.</b> {lines} interconnectors are planned or "
        "committed across the region. Each new corridor can carry fibre at a low marginal "
        "cost, as the BSSC and the Trans-Caspian already plan to do.",
        f"<b>Growing data centre demand.</b> Announced sites with a published size add about "
        f"{announced:,} MW, led by the 1.5 GW Black Sea AI Gigafactory in Romania.",
        "<b>Additional demand in the west.</b> Data centres cluster in Istanbul, Bucharest "
        "and Sofia, where wholesale power costs about 100 $/MWh in Romania and Bulgaria.",
        "<b>Low cost production in the east.</b> Power in the Caucasus costs about "
        "40 $/MWh.",
        "<b>Grid reinforcement.</b> Bringing eastern power to western demand needs "
        "interconnectors and stronger internal networks behind them.",
    ]
    items = "".join(f'<li style="margin:4px 0">{t}</li>' for t in points)
    return f'<ul style="margin:6px 0 14px;padding-left:20px;line-height:1.5">{items}</ul>'


PANES = {"digital-power": digital_power}


def pane(index: CountryIndex) -> str:
    out = ['<div class="subtabs">']
    out += [f'<button class="subtab" data-sub="{s}">{esc(label)}</button>' for s, label in SUBS]
    out.append("</div>")
    out += [f'<div class="subpane" data-sub="{s}">{PANES[s](index)}</div>' for s, _ in SUBS]
    return "".join(out)
