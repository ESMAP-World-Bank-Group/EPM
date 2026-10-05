"""Build the internal HTML screening note on digital connectivity.

Reads the datasets downloaded by fetch_data.py, computes the screening statistics,
renders every map and chart as inline SVG, and writes a single self-contained
HTML file. No external assets, no network access at build time.

Usage
    python fetch_data.py                 # once, to populate ./data
    python build_note.py                 # writes beside this script
    python build_note.py --out <dir>     # writes into another folder as well
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "digital_connectivity_screening.html"
BASEMAP = HERE.parent.parent / "epm" / "input" / "data_blacksea" / "extras" / "background_countries.geojson"
EPM_LINES = HERE.parent.parent / "epm" / "input" / "data_blacksea" / "linestring_zcmap_geco.geojson"

BBOX = (26.0, 36.0, 53.0, 48.5)

# One family across the whole note. Status has a single reading everywhere:
# blue is in service, teal is being built, mustard is announced.
PALETTE = {
    "ink": "#256081",
    "accent": "#0277bd",
    "pale": "#b2ebf2",
    "muted": "#78909c",
    "land": "#ffffff",
    "land_focus": "#f5eeda",
    "border": "#b0bec5",
    "sea": "#e8f4f8",
    "operational": "#0277bd",
    "building": "#00838f",
    "planned": "#ddc32c",
    # Layers on the combined map, where the legend names each one.
    "fibre": "#0277bd",
    "cable": "#00838f",
    "power": "#256081",
    "pipe": "#b0bec5",
    "rail": "#90a4ae",
}

FOCUS = {"Georgia", "Armenia", "Azerbaijan"}
REGION = {
    "Georgia", "Armenia", "Azerbaijan", "Turkey", "Romania", "Bulgaria",
    "Moldova", "Ukraine", "Kazakhstan", "Russia", "Iran", "Greece",
}
DISPLAY = {"Turkey": "Turkiye"}

# The BSSC route is not published as geometry. Rather than draw a straight line,
# the note traces the Caucasus Cable System, the surveyed crossing of the Black
# Sea that already runs Poti to Balchik in the same latitude band, and shifts the
# two landfalls to the announced ones, Anaklia and Constanta. Captions say so.
BSSC_ANCHOR = "Caucasus Cable System"
BSSC_LANDFALL_EAST = (41.573, 42.395)   # Anaklia, Georgia
BSSC_LANDFALL_WEST = [(28.75, 43.90), (28.66, 44.17)]  # onto the Constanta shelf

# The TRIPP alignment is the closed Soviet-era railway along the Aras. It is
# mapped in OpenStreetMap as disused, so the note traces the real alignment
# between these two points rather than joining the endpoints.
TRIPP_BBOX = (44.6, 38.7, 47.6, 39.95)  # lon_min, lat_min, lon_max, lat_max,
# a tight band along the Aras from Yeraskh to Horadiz, so the filter keeps the
# corridor alignment and not the Yerevan hub or the Kura valley main line.
TRIPP_ENDS = [(47.35, 39.45), (45.41, 39.21)]  # Horadiz (AZ), Nakhchivan (AZ)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def haversine(a, b) -> float:
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, (a[1], a[0], b[1], b[0]))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def lines_of(geom):
    """Yield every coordinate list of a Line or MultiLine geometry."""
    if not geom:
        return
    if geom["type"] == "LineString":
        yield geom["coordinates"]
    elif geom["type"] == "MultiLineString":
        yield from geom["coordinates"]


def polys_of(geom):
    if geom["type"] == "Polygon":
        yield geom["coordinates"]
    elif geom["type"] == "MultiPolygon":
        yield from geom["coordinates"]


def point_in_ring(x, y, ring) -> bool:
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        if (y1 > y) != (y2 > y):
            xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xint:
                inside = not inside
    return inside


class CountryIndex:
    """Point in polygon lookup over the repository basemap.

    Two accelerations keep the screening tractable on large polyline datasets:
    a half-degree cell index that shortlists candidate polygons, and a memo on
    the query point rounded to about one kilometre.
    """

    CELL = 0.5
    MEMO = 100.0  # 1/degree, so 0.01 degree resolution

    def __init__(self, path: Path):
        self.entries = []
        for feat in json.load(open(path, encoding="utf8"))["features"]:
            name = feat["properties"]["name"]
            for poly in polys_of(feat["geometry"]):
                outer = poly[0]
                xs = [p[0] for p in outer]
                ys = [p[1] for p in outer]
                self.entries.append((name, poly, min(xs), min(ys), max(xs), max(ys)))
        self.cells = defaultdict(list)
        for entry in self.entries:
            _, _, x0, y0, x1, y1 = entry
            for gx in range(int(math.floor(x0 / self.CELL)), int(math.floor(x1 / self.CELL)) + 1):
                for gy in range(
                    int(math.floor(y0 / self.CELL)), int(math.floor(y1 / self.CELL)) + 1
                ):
                    self.cells[(gx, gy)].append(entry)
        self.memo = {}

    def _lookup(self, x, y):
        cell = (int(math.floor(x / self.CELL)), int(math.floor(y / self.CELL)))
        for name, poly, x0, y0, x1, y1 in self.cells.get(cell, ()):
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                continue
            if point_in_ring(x, y, poly[0]) and not any(
                point_in_ring(x, y, hole) for hole in poly[1:]
            ):
                return name
        return None

    def at(self, x, y):
        key = (int(x * self.MEMO), int(y * self.MEMO))
        hit = self.memo.get(key)
        if hit is None:
            hit = self.memo[key] = self._lookup(x, y) or ""
        return hit or None


# ---------------------------------------------------------------------------
# SVG canvas
# ---------------------------------------------------------------------------
class Map:
    """Equirectangular canvas with latitude correction, fitted to the window."""

    def __init__(self, width=980, height=460, bbox=BBOX, pad=6):
        self.w, self.h, self.pad = width, height, pad
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.kx = math.cos(math.radians((self.y0 + self.y1) / 2))
        span_x = (self.x1 - self.x0) * self.kx
        span_y = self.y1 - self.y0
        self.scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y)
        self.ox = (width - span_x * self.scale) / 2
        self.oy = (height - span_y * self.scale) / 2
        self.parts = []

    def xy(self, lon, lat):
        return (
            self.ox + (lon - self.x0) * self.kx * self.scale,
            self.oy + (self.y1 - lat) * self.scale,
        )

    def path(self, coords, min_px=1.0) -> str:
        """Project a coordinate list, dropping vertices below the pixel threshold.

        Keeps the rendered file small: the OpenStreetMap extracts carry far more
        vertices than a 980 pixel canvas can show. A line whose own bounding box
        misses the window is dropped outright, which matters on the zoomed maps,
        where most of the railway and pipeline extract falls outside the frame.
        """
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]
        if max(lons) < self.x0 or min(lons) > self.x1:
            return ""
        if max(lats) < self.y0 or min(lats) > self.y1:
            return ""
        pts = []
        last = None
        for i, (lon, lat) in enumerate(coords):
            x, y = self.xy(lon, lat)
            if last is not None and i != len(coords) - 1:
                if abs(x - last[0]) < min_px and abs(y - last[1]) < min_px:
                    continue
            pts.append(f"{x:.1f},{y:.1f}")
            last = (x, y)
        if len(pts) < 2:
            return ""
        return "M" + "L".join(pts)

    def smooth(self, coords, tension=0.5) -> str:
        """Catmull-Rom through the anchors, emitted as cubic beziers.

        Used only by the corridor overview. That map is a schematic at regional
        scale, and a polyline through a handful of cities would read as surveyed
        geometry, which it is not. The curve makes the abstraction visible.
        """
        pts = [self.xy(lon, lat) for lon, lat in coords]
        if len(pts) < 2:
            return ""
        out = [f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"]
        for i in range(len(pts) - 1):
            p0 = pts[i - 1] if i else pts[0]
            p1, p2 = pts[i], pts[i + 1]
            p3 = pts[i + 2] if i + 2 < len(pts) else p2
            c1 = (p1[0] + (p2[0] - p0[0]) * tension / 3,
                  p1[1] + (p2[1] - p0[1]) * tension / 3)
            c2 = (p2[0] - (p3[0] - p1[0]) * tension / 3,
                  p2[1] - (p3[1] - p1[1]) * tension / 3)
            out.append(
                f"C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} "
                f"{p2[0]:.1f},{p2[1]:.1f}"
            )
        return "".join(out)

    def add(self, markup: str):
        # A path that was culled or collapsed comes back empty. Dropping it here
        # saves every caller a guard, and saves megabytes on the railway layer.
        if 'd=""' in markup:
            return
        self.parts.append(markup)

    def basemap(self, index: CountryIndex, highlight=FOCUS):
        self.add(f'<rect width="{self.w}" height="{self.h}" fill="{PALETTE["sea"]}"/>')
        for name, poly, *_ in index.entries:
            fill = PALETTE["land_focus"] if name in highlight else PALETTE["land"]
            rings = [self.path(ring, min_px=0.5) for ring in poly]
            d = " ".join(r + "Z" for r in rings if r)
            if not d:
                continue
            self.add(
                f'<path d="{d}" fill="{fill}" stroke="{PALETTE["border"]}" '
                f'stroke-width="0.6" fill-rule="evenodd"/>'
            )

    def label(self, lon, lat, text, size=10, colour=None, anchor="middle",
              weight="600", halo=False):
        x, y = self.xy(lon, lat)
        colour = colour or PALETTE["muted"]
        ring = (' stroke="#ffffff" stroke-width="3" stroke-linejoin="round" '
                'paint-order="stroke"') if halo else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{colour}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{ring} '
            f'font-family="Segoe UI, Arial, sans-serif">{html.escape(text)}</text>'
        )

    def dot(self, lon, lat, r, fill, stroke="#ffffff", sw=0.8, opacity=1.0):
        x, y = self.xy(lon, lat)
        self.add(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
            f'fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def legend(self, items, x=14, y=None, box=True):
        y = y if y is not None else self.h - 14 - 15 * len(items)
        if box:
            width = 12 + max(len(t) for _, t, *_ in items) * 5.6 + 26
            self.add(
                f'<rect x="{x - 8}" y="{y - 14}" width="{width:.0f}" '
                f'height="{15 * len(items) + 10}" rx="4" fill="#ffffff" '
                f'fill-opacity="0.88" stroke="{PALETTE["border"]}" stroke-width="0.6"/>'
            )
        for i, item in enumerate(items):
            colour, text = item[0], item[1]
            dash = item[2] if len(item) > 2 else None
            yy = y + i * 15
            if dash == "dot":
                self.add(f'<circle cx="{x + 5}" cy="{yy - 3}" r="4" fill="{colour}"/>')
            elif dash == "ring":
                self.add(
                    f'<circle cx="{x + 5}" cy="{yy - 3}" r="4.5" fill="#ffffff" '
                    f'stroke="{colour}" stroke-width="2"/>'
                )
            elif dash == "box":
                self.add(
                    f'<rect x="{x}" y="{yy - 9}" width="15" height="10" rx="2" '
                    f'fill="{colour}" stroke="{PALETTE["border"]}" stroke-width="0.6"/>'
                )
            else:
                da = f' stroke-dasharray="{dash}"' if dash else ""
                self.add(
                    f'<line x1="{x}" y1="{yy - 3}" x2="{x + 18}" y2="{yy - 3}" '
                    f'stroke="{colour}" stroke-width="2.4"{da}/>'
                )
            self.add(
                f'<text x="{x + 24}" y="{yy}" font-size="10" fill="#4a5a68" '
                f'font-family="Segoe UI, Arial, sans-serif">{html.escape(text)}</text>'
            )

    def render(self) -> str:
        return (
            f'<svg viewBox="0 0 {self.w} {self.h}" width="100%" '
            f'xmlns="http://www.w3.org/2000/svg" role="img">'
            + "".join(self.parts)
            + "</svg>"
        )


def bar_chart(rows, width=980, bar_h=18, gap=7, pad_left=130, colours=None, unit=""):
    """Horizontal stacked bar chart. rows = [(label, {series: value})]."""
    colours = colours or {}
    series = list({s for _, d in rows for s in d})
    totals = [sum(d.values()) for _, d in rows]
    vmax = max(totals) if totals else 1
    height = len(rows) * (bar_h + gap) + 34
    plot_w = width - pad_left - 70
    parts = [f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    for i, (label, values) in enumerate(rows):
        y = i * (bar_h + gap) + 6
        x = pad_left
        for s in series:
            v = values.get(s, 0)
            if v <= 0:
                continue
            w = v / vmax * plot_w
            parts.append(
                f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" '
                f'fill="{colours.get(s, PALETTE["accent"])}"/>'
            )
            x += w
        parts.append(
            f'<text x="{pad_left - 8}" y="{y + bar_h - 5}" font-size="11.5" '
            f'text-anchor="end" fill="#2c3e50" '
            f'font-family="Segoe UI, Arial, sans-serif">{html.escape(label)}</text>'
        )
        parts.append(
            f'<text x="{x + 6:.1f}" y="{y + bar_h - 5}" font-size="11" fill="#7f8c8d" '
            f'font-family="Segoe UI, Arial, sans-serif">{totals[i]:,.0f}{unit}</text>'
        )
    ly = len(rows) * (bar_h + gap) + 20
    lx = pad_left
    for s in series:
        parts.append(
            f'<rect x="{lx}" y="{ly - 9}" width="11" height="11" '
            f'fill="{colours.get(s, PALETTE["accent"])}"/>'
        )
        parts.append(
            f'<text x="{lx + 16}" y="{ly}" font-size="10.5" fill="#4a5a68" '
            f'font-family="Segoe UI, Arial, sans-serif">{html.escape(s)}</text>'
        )
        lx += 22 + len(s) * 6.2
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'xmlns="http://www.w3.org/2000/svg">' + "".join(parts) + "</svg>"
    )


# ---------------------------------------------------------------------------
# data loading and statistics
# ---------------------------------------------------------------------------
def load(name):
    path = DATA / name
    if not path.exists():
        return None
    return json.load(open(path, encoding="utf8"))


def fibre_km_by_country(fibre, index: CountryIndex):
    km = defaultdict(Counter)
    for feat in fibre["features"]:
        status = feat["properties"].get("status") or "Unknown"
        for line in lines_of(feat["geometry"]):
            for i in range(len(line) - 1):
                a, b = line[i], line[i + 1]
                mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                country = index.at(*mid)
                if country in REGION:
                    km[country][status] += haversine(a, b)
    return km


class SegmentGrid:
    """Coarse spatial hash over fibre segments, for the co-location metric."""

    CELL = 0.25  # degrees

    def __init__(self, fibre):
        self.cells = defaultdict(list)
        for feat in fibre["features"]:
            for line in lines_of(feat["geometry"]):
                for i in range(len(line) - 1):
                    a, b = line[i], line[i + 1]
                    for key in self._keys(a, b):
                        self.cells[key].append((a, b))

    def _keys(self, a, b):
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        keys = set()
        gx = int(x0 / self.CELL)
        while gx <= int(x1 / self.CELL):
            gy = int(y0 / self.CELL)
            while gy <= int(y1 / self.CELL):
                keys.add((gx, gy))
                gy += 1
            gx += 1
        return keys

    def near(self, x, y, radius_km):
        span = radius_km / 100.0
        out = []
        for gx in range(int((x - span) / self.CELL), int((x + span) / self.CELL) + 1):
            for gy in range(int((y - span) / self.CELL), int((y + span) / self.CELL) + 1):
                out.extend(self.cells.get((gx, gy), ()))
        return out


def point_segment_km(p, a, b) -> float:
    """Approximate distance from point p to segment ab, in km."""
    kx = math.cos(math.radians(p[1])) * 111.32
    ky = 110.57
    px, py = p[0] * kx, p[1] * ky
    ax, ay = a[0] * kx, a[1] * ky
    bx, by = b[0] * kx, b[1] * ky
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    t = 0.0 if denom == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def colocation(hv, fibre, index: CountryIndex, radius_km=3.0):
    """Share of mapped HV line length with a mapped fibre link within radius_km."""
    grid = SegmentGrid(fibre)
    total = defaultdict(float)
    near = defaultdict(float)
    for feat in hv["features"]:
        for line in lines_of(feat["geometry"]):
            for i in range(len(line) - 1):
                a, b = line[i], line[i + 1]
                mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                country = index.at(*mid)
                if country not in REGION:
                    continue
                length = haversine(a, b)
                total[country] += length
                for sa, sb in grid.near(mid[0], mid[1], radius_km):
                    if point_segment_km(mid, sa, sb) <= radius_km:
                        near[country] += length
                        break
    return total, near


# ---------------------------------------------------------------------------
# maps
# ---------------------------------------------------------------------------
def bssc_route(cables):
    """Trace the announced BSSC fibre on surveyed geometry rather than a bearing.

    The Caucasus Cable System already crosses the Black Sea from Poti to Balchik
    in the latitude band the BSSC would use, south of the Crimean shelf. The
    route below keeps that open-sea alignment and moves the two landfalls to the
    announced ones, Anaklia in Georgia and Constanta in Romania.
    """
    for feat in cables["features"]:
        if feat["properties"].get("name") != BSSC_ANCHOR:
            continue
        line = [tuple(p) for p in max(lines_of(feat["geometry"]), key=len)]
        if line[0][0] < line[-1][0]:
            line.reverse()
        return [BSSC_LANDFALL_EAST] + line[1:-1] + BSSC_LANDFALL_WEST
    return []


def tripp_alignment(railways, index):
    """The real Aras valley railway, which the TRIPP corridor would reopen.

    OpenStreetMap carries the Soviet-era line as disused or abandoned. Segments
    on the Iranian bank of the river are dropped, since the corridor runs on the
    Armenian and Azerbaijani side.
    """
    out = []
    if not railways:
        return out
    x0, y0, x1, y1 = TRIPP_BBOX
    for feat in railways["features"]:
        if feat["properties"].get("layer") != "corridor":
            continue
        for line in lines_of(feat["geometry"]):
            lon = sum(p[0] for p in line) / len(line)
            lat = sum(p[1] for p in line) / len(line)
            if not (x0 <= lon <= x1 and y0 <= lat <= y1):
                continue
            if index.at(lon, lat) not in ("Armenia", "Azerbaijan"):
                continue
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# Corridor overview
# ---------------------------------------------------------------------------
# Six corridors carry most of the regional investment pipeline. The anchors are
# real cities and landing points, but the spine drawn between them is schematic:
# this map is for orientation, and the surveyed geometry is in the maps that
# follow. Sector order is fixed so the badges line up down the table.
SECTORS = ["Power", "Gas", "Transport", "Digital"]

CORRIDORS = [
    {
        "name": "Southern Gas Corridor and the Baku to Kars axis",
        "status": "operational",
        "anchors": [(49.87, 40.41), (47.15, 40.65), (44.79, 41.72), (43.48, 41.40),
                    (43.10, 40.60), (41.27, 39.90), (38.50, 39.75), (34.80, 39.80)],
        "label": (40.0, 40.35, "middle"),
        "sectors": ["Power", "Gas", "Transport", "Digital"],
        "carries": "SCP and TANAP gas, the Georgia to Turkiye 400 kV link, the "
                   "Baku to Tbilisi to Kars railway, SOCAR Fiber",
        "meaning": "The one corridor where all four layers are already built and "
                   "share an easement. It is the working precedent for the rest.",
    },
    {
        "name": "Middle Corridor, Caspian crossing",
        "status": "building",
        "anchors": [(51.16, 43.65), (50.75, 42.60), (50.05, 41.20), (49.87, 40.41)],
        "label": (50.9, 42.6, "start"),
        "sectors": ["Power", "Transport", "Digital"],
        "carries": "Aktau to Baku rail ferry, Trans-Caspian fibre at 400 Tbps, "
                   "the proposed Caspian green energy cable",
        "meaning": "Closes the middle route east of Baku. The fibre leg enters "
                   "service in 2026, well ahead of the power leg.",
    },
    {
        "name": "Black Sea energy and digital corridor",
        "status": "planned",
        "anchors": [],  # traced on the surveyed crossing, filled at draw time
        "label": (35.0, 43.25, "middle"),
        "sectors": ["Power", "Digital"],
        "carries": "BSSC HVDC cable, Anaklia to Constanta, with 40 Tbps of fibre "
                   "in the same lay",
        "meaning": "One project, two sectors, one cost base. The fibre is not "
                   "valued in the power benefit case.",
    },
    {
        "name": "TRIPP, the Aras corridor",
        "status": "planned",
        "anchors": [(44.72, 39.85), (45.41, 39.21), (46.24, 38.90), (47.35, 39.45)],
        "label": (45.9, 38.45, "middle"),
        "sectors": ["Power", "Gas", "Transport", "Digital"],
        "carries": "Rail, road, a gas line, a power line and a fibre duct in one "
                   "43 km easement through southern Armenia",
        "meaning": "The only corridor in the region designed as a single "
                   "multi-sector asset from the start.",
    },
    {
        "name": "Georgia to Armenia axis",
        "status": "building",
        "anchors": [(44.79, 41.72), (44.81, 41.48), (44.66, 41.10), (44.49, 40.81),
                    (44.51, 40.18)],
        "label": (44.56, 40.58, "middle"),
        "sectors": ["Power", "Transport", "Digital"],
        "carries": "The Georgia to Armenia interconnector, the M6 road, and the "
                   "fibre that follows both",
        "meaning": "Armenia's only route to the Black Sea that avoids both Russia "
                   "and, until 2026, Azerbaijan.",
    },
    {
        "name": "Northern axis through Russia",
        "status": "legacy",
        "anchors": [(44.68, 43.02), (44.64, 42.66), (44.79, 41.72)],
        "label": (44.70, 42.86, "middle"),
        "sectors": ["Power", "Gas", "Digital"],
        "carries": "The Vladikavkaz to Tbilisi gas line and its Yerevan leg, the Georgia to "
                   "Russia power link, the BSFOCS cable of 2001",
        "meaning": "The dependency the other five corridors exist to reduce. It is "
                   "shown here because it sets the counterfactual.",
    },
]

CORRIDOR_COLOUR = {
    "operational": PALETTE["operational"],
    "building": PALETTE["building"],
    "planned": PALETTE["planned"],
    "legacy": PALETTE["muted"],
}

CORRIDOR_STATUS = {
    "operational": "In service",
    "building": "Under construction",
    "planned": "Announced",
    "legacy": "Legacy route",
}


def map_corridors(index, cables):
    """High-level orientation: the named corridors and what each one carries."""
    # Cropped north: every corridor sits below 44 N and the full screening window
    # would leave the top third of the canvas empty.
    m = Map(height=500, bbox=(26.0, 36.0, 53.5, 46.5))
    m.basemap(index)
    for spec in CORRIDORS:
        colour = CORRIDOR_COLOUR[spec["status"]]
        if spec["anchors"]:
            d = m.smooth(spec["anchors"])
        else:
            route = bssc_route(cables)
            d = m.path(route) if route else ""
        if not d:
            continue
        dash = ' stroke-dasharray="14 8"' if spec["status"] == "planned" else ""
        # A wide translucent band under a thin core line: the band carries the
        # corridor, the core keeps it readable where corridors overlap.
        m.add(
            f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="13" '
            f'stroke-opacity="0.20" stroke-linecap="round"/>'
        )
        m.add(
            f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2.4" '
            f'stroke-opacity="0.95" stroke-linecap="round"{dash}/>'
        )
    for i, spec in enumerate(CORRIDORS, start=1):
        lon, lat, anchor = spec["label"]
        x, y = m.xy(lon, lat)
        m.add(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" '
            f'fill="{CORRIDOR_COLOUR[spec["status"]]}" stroke="#ffffff" '
            f'stroke-width="1.6"/>'
        )
        m.add(
            f'<text x="{x:.1f}" y="{y + 3.5:.1f}" font-size="11" fill="#ffffff" '
            f'text-anchor="middle" font-weight="700" '
            f'font-family="{FONT}">{i}</text>'
        )
    m.legend(
        [
            (PALETTE["operational"], "In service", None),
            (PALETTE["building"], "Under construction", None),
            (PALETTE["planned"], "Announced", "14 8"),
            (PALETTE["muted"], "Legacy route through Russia", None),
        ],
        y=None,
    )
    return m.render()


def corridor_table() -> str:
    """The reading key for the corridor map: who carries what, and why it matters."""
    rows = []
    for i, spec in enumerate(CORRIDORS, start=1):
        badges = []
        for sector in SECTORS:
            on = sector in spec["sectors"]
            cls = "on" if on else "off"
            badges.append(f'<span class="sector {cls}">{sector}</span>')
        rows.append(
            [
                f'<span class="cnum" style="background:{CORRIDOR_COLOUR[spec["status"]]}">'
                f"{i}</span> {html.escape(spec['name'])}",
                " ".join(badges),
                CORRIDOR_STATUS[spec["status"]],
                html.escape(spec["carries"]),
                html.escape(spec["meaning"]),
            ]
        )
    return table(
        ["Corridor", "Layers it carries", "Status", "Assets", "Why it matters here"],
        rows,
    )


def map_submarine(index, cables, landings):
    m = Map(height=470)
    m.basemap(index)
    for feat in cables["features"]:
        name = feat["properties"].get("name", "")
        new = "Trans-Caspian" in name
        colour = PALETTE["building"] if new else PALETTE["operational"]
        dash = ' stroke-dasharray="10 5"' if new else ""
        for line in lines_of(feat["geometry"]):
            m.add(
                f'<path d="{m.path(line)}" fill="none" stroke="{colour}" '
                f'stroke-width="2.1" stroke-opacity="0.9"{dash}/>'
            )
    route = bssc_route(cables)
    if route:
        m.add(
            f'<path d="{m.path(route)}" fill="none" stroke="{PALETTE["planned"]}" '
            f'stroke-width="3.0" stroke-dasharray="10 5" stroke-linecap="round"/>'
        )
    for feat in landings["features"]:
        lon, lat = feat["geometry"]["coordinates"]
        m.dot(lon, lat, 3.1, "#ffffff", stroke=PALETTE["ink"], sw=1.3)
    for lon, lat, text, anchor in [
        (41.90, 42.25, "Poti / Anaklia", "start"),
        (27.60, 42.95, "Balchik", "end"),
        (28.63, 44.55, "Constanta", "start"),
        (49.67, 40.59, "Sumgait", "start"),
        (51.16, 43.65, "Aktau", "start"),
        (29.0, 41.10, "Istanbul", "end"),
    ]:
        m.label(lon, lat - 0.45, text, size=10, colour=PALETTE["ink"], anchor=anchor)
    m.legend(
        [
            (PALETTE["operational"], "In service", None),
            (PALETTE["building"], "Trans-Caspian, service 2026", "10 5"),
            (PALETTE["planned"], "BSSC fibre, on the surveyed crossing", "10 5"),
            (PALETTE["ink"], "Landing point", "dot"),
        ]
    )
    return m.render()


def map_terrestrial(index, fibre):
    m = Map(height=470)
    m.basemap(index)
    order = ["Operational", "Under Construction", "Planned"]
    colours = {
        "Operational": PALETTE["operational"],
        "Under Construction": PALETTE["building"],
        "Planned": PALETTE["planned"],
    }
    # The ITU catalogue is a node and link graph: every record is a single
    # segment between two mapped points, with no intermediate geometry. The
    # nodes are drawn so the map reads as the topology it is.
    style = {
        "Operational": (0.8, ""),
        "Under Construction": (1.8, ' stroke-dasharray="7 4"'),
        "Planned": (1.8, ' stroke-dasharray="1.6 3.4" stroke-linecap="round"'),
    }
    for status in order:
        width, dash = style[status]
        for feat in fibre["features"]:
            if (feat["properties"].get("status") or "") != status:
                continue
            for line in lines_of(feat["geometry"]):
                m.add(
                    f'<path d="{m.path(line)}" fill="none" stroke="{colours[status]}" '
                    f'stroke-width="{width}" stroke-opacity="0.85"{dash}/>'
                )
    nodes = set()
    for feat in fibre["features"]:
        for line in lines_of(feat["geometry"]):
            for lon, lat in (line[0], line[-1]):
                nodes.add((round(lon, 3), round(lat, 3)))
    for lon, lat in nodes:
        m.dot(lon, lat, 1.3, PALETTE["operational"], sw=0, opacity=0.8)
    m.legend(
        [
            (colours["Operational"], "Operational", None),
            (colours["Under Construction"], "Under construction", "7 4"),
            (colours["Planned"], "Planned", "1.6 3.4"),
            (PALETTE["operational"], "Mapped node", "dot"),
        ]
    )
    return m.render()


def map_projects(index, fibre, cables, railways):
    m = Map(height=470)
    m.basemap(index)
    nodes = set()
    for feat in fibre["features"]:
        status = feat["properties"].get("status") or ""
        if status == "Operational":
            continue
        building = status == "Under Construction"
        colour = PALETTE["building"] if building else PALETTE["planned"]
        # Same convention as Map 3: dashed is being built, dotted is announced.
        dash = ' stroke-dasharray="7 4"' if building else \
               ' stroke-dasharray="1.8 3.6" stroke-linecap="round"'
        for line in lines_of(feat["geometry"]):
            m.add(
                f'<path d="{m.path(line)}" fill="none" stroke="{colour}" '
                f'stroke-width="2.2"{dash}/>'
            )
            for lon, lat in (line[0], line[-1]):
                nodes.add((round(lon, 3), round(lat, 3), colour))
    for feat in cables["features"]:
        if "Trans-Caspian" not in feat["properties"].get("name", ""):
            continue
        for line in lines_of(feat["geometry"]):
            m.add(
                f'<path d="{m.path(line)}" fill="none" stroke="{PALETTE["building"]}" '
                f'stroke-width="2.6" stroke-dasharray="10 5"/>'
            )
    route = bssc_route(cables)
    if route:
        m.add(
            f'<path d="{m.path(route)}" fill="none" stroke="{PALETTE["planned"]}" '
            f'stroke-width="3.0" stroke-dasharray="10 5" stroke-linecap="round"/>'
        )
    tripp = tripp_alignment(railways, index)
    for line in tripp:
        m.add(
            f'<path d="{m.path(line, min_px=0.6)}" fill="none" stroke="{PALETTE["ink"]}" '
            f'stroke-width="2.4" stroke-linecap="round"/>'
        )
    m.label(35.0, 43.9, "BSSC fibre, 40 Tbps", size=10, colour=PALETTE["ink"])
    m.label(50.4, 41.6, "Trans-Caspian, 400 Tbps", size=10, colour=PALETTE["ink"])
    m.label(46.3, 38.25, "TRIPP, Aras alignment", size=10, colour=PALETTE["ink"])
    for lon, lat, colour in nodes:
        m.dot(lon, lat, 1.8, colour, sw=0, opacity=0.9)
    items = [
        (PALETTE["building"], "Terrestrial fibre under construction", "7 4"),
        (PALETTE["planned"], "Terrestrial fibre planned", "1.8 3.6"),
        (PALETTE["building"], "Submarine, in commissioning", "10 5"),
        (PALETTE["planned"], "Announced submarine route", "10 5"),
    ]
    if tripp:
        items.append((PALETTE["ink"], "TRIPP, the existing rail alignment", None))
    m.legend(items)
    return m.render()


# Where networks meet, by country. The choropleth is the point of the map: the
# gradient runs from the EU shore of the Black Sea to the Caucasus, and it is a
# gradient in interconnection, not in cable.
PDB_COUNTRY = {
    "BG": "Bulgaria", "RO": "Romania", "TR": "Turkey", "GE": "Georgia",
    "AM": "Armenia", "AZ": "Azerbaijan", "MD": "Moldova", "UA": "Ukraine",
    "KZ": "Kazakhstan", "GR": "Greece",
}

PEERING_SCALE = [
    (150, "#0277bd", "More than 150"),
    (75, "#4a9ed1", "75 to 150"),
    (25, "#8fc6e3", "25 to 75"),
    (1, "#cfe6f2", "1 to 25"),
    (0, "#f2dfa0", "No exchange listed"),
]


def peering_fill(total: int) -> str:
    for threshold, colour, _ in PEERING_SCALE:
        if total >= threshold and (threshold > 0 or total == 0):
            return colour
    return PALETTE["land"]


def map_interconnection(index, pdb):
    """Networks peering per country, with the exchange cities on top."""
    totals = {}
    for code, rows in pdb["ix"].items():
        name = PDB_COUNTRY.get(code)
        if name:
            totals[name] = sum(ix.get("net_count") or 0 for ix in rows)

    # Extended west of the screening window so Sofia, the largest exchange in
    # the region and the one Turkish traffic actually uses, is on the canvas.
    m = Map(height=520, bbox=(22.0, 36.0, 53.0, 48.5))
    m.add(f'<rect width="{m.w}" height="{m.h}" fill="{PALETTE["sea"]}"/>')
    for name, poly, *_ in index.entries:
        fill = peering_fill(totals[name]) if name in totals else PALETTE["land"]
        rings = [m.path(ring, min_px=0.5) for ring in poly]
        d = " ".join(r + "Z" for r in rings if r)
        if d:
            m.add(
                f'<path d="{d}" fill="{fill}" stroke="{PALETTE["border"]}" '
                f'stroke-width="0.6" fill-rule="evenodd"/>'
            )

    for code, rows in pdb["fac"].items():
        for fac in rows:
            lat, lon = fac.get("latitude"), fac.get("longitude")
            if lat is None or lon is None or not (BBOX[0] <= lon <= BBOX[2]):
                continue
            if not (BBOX[1] <= lat <= BBOX[3]):
                continue
            m.dot(lon, lat, 2.2, "#37474f", stroke="none", sw=0, opacity=0.55)

    cities = defaultdict(lambda: {"nets": 0, "ix": 0, "lon": None, "lat": None})
    for code, rows in pdb["ix"].items():
        for ix in rows:
            city = short_city(ix.get("city") or code)
            key = (code, city.lower())
            entry = cities[key]
            entry["nets"] = max(entry["nets"], ix.get("net_count") or 0)
            entry["ix"] += 1
            entry["name"] = city
            if entry["lon"] is None:
                pos = IX_POSITIONS.get(entry["name"].lower())
                if pos:
                    entry["lon"], entry["lat"] = pos

    placed = []
    for entry in sorted(cities.values(), key=lambda e: -e["nets"]):
        if entry["lon"] is None or entry["nets"] == 0:
            continue
        r = 3.0 + math.sqrt(entry["nets"]) * 1.5
        m.dot(entry["lon"], entry["lat"], r, "#ffffff", stroke=PALETTE["ink"],
              sw=2.0, opacity=0.92)
        placed.append((entry["name"], entry["lon"], entry["lat"], entry["nets"], r))

    # Capitals with no exchange at all are the finding, so they are marked
    # explicitly rather than left blank.
    for name in ("baku",):
        lon, lat = IX_POSITIONS[name]
        m.add(
            '<circle cx="%.1f" cy="%.1f" r="7" fill="none" stroke="%s" '
            'stroke-width="2" stroke-dasharray="3 2.5"/>'
            % (m.xy(lon, lat) + (PALETTE["planned"],))
        )
        m.label(lon, lat + 13.0 / m.scale, "Baku, none listed", size=10,
                colour=PALETTE["ink"], anchor="middle", halo=True)

    for name, lon, lat, nets, r in placed:
        m.label(lon, lat - (r + 10.0) / m.scale, f"{name.title()} {nets}", size=10,
                colour=PALETTE["ink"], halo=True)

    m.legend(
        [(colour, text, "box") for _, colour, text in PEERING_SCALE]
        + [
            (PALETTE["ink"], "Exchange city, area = networks at its largest exchange", "ring"),
            ("#37474f", "Interconnection facility", "dot"),
        ],
        x=14,
    )
    return m.render()


def short_city(value: str) -> str:
    """PeeringDB lists every site of a multi-city exchange. Keep the first."""
    return value.split(",")[0].split("/")[0].strip()


IX_POSITIONS = {
    "sofia": (23.32, 42.70),
    "bucharest": (26.10, 44.44),
    "istanbul": (28.98, 41.01),
    "ankara": (32.86, 39.93),
    "varna": (27.92, 43.21),
    "braila": (27.96, 45.27),
    "craiova": (23.80, 44.33),
    "tbilisi": (44.79, 41.72),
    "yerevan": (44.51, 40.18),
    "artashat": (44.55, 39.95),
    "baku": (49.87, 40.41),
    "kyiv": (30.52, 50.45),
    "chisinau": (28.86, 47.01),
}


def map_master(index, fibre, cables, hv, pipelines, railways, epm_lines,
               bbox=BBOX, height=500, min_px=1.8, label_zoom=False):
    """Every mapped corridor on one canvas. Also used zoomed on the Caucasus."""
    m = Map(height=height, bbox=bbox)
    m.basemap(index)
    if pipelines:
        for feat in pipelines["features"]:
            for line in lines_of(feat["geometry"]):
                m.add(
                    f'<path d="{m.path(line, min_px=min_px)}" fill="none" '
                    f'stroke="{PALETTE["pipe"]}" stroke-width="1.8" stroke-opacity="0.9"/>'
                )
    if railways:
        for feat in railways["features"]:
            for line in lines_of(feat["geometry"]):
                m.add(
                    f'<path d="{m.path(line, min_px=min_px)}" fill="none" '
                    f'stroke="{PALETTE["rail"]}" stroke-width="1.2" '
                    f'stroke-dasharray="4 3" stroke-opacity="0.9"/>'
                )
    if hv:
        for feat in hv["features"]:
            for line in lines_of(feat["geometry"]):
                m.add(
                    f'<path d="{m.path(line, min_px=min_px)}" fill="none" '
                    f'stroke="{PALETTE["power"]}" stroke-width="1.0" stroke-opacity="0.7"/>'
                )
    elif epm_lines:
        for feat in epm_lines["features"]:
            for line in lines_of(feat["geometry"]):
                m.add(
                    f'<path d="{m.path(line)}" fill="none" stroke="{PALETTE["power"]}" '
                    f'stroke-width="1.0" stroke-opacity="0.45"/>'
                )
    for feat in fibre["features"]:
        for line in lines_of(feat["geometry"]):
            m.add(
                f'<path d="{m.path(line)}" fill="none" stroke="{PALETTE["fibre"]}" '
                f'stroke-width="1.1" stroke-opacity="0.85"/>'
            )
    for feat in cables["features"]:
        for line in lines_of(feat["geometry"]):
            m.add(
                f'<path d="{m.path(line)}" fill="none" stroke="{PALETTE["cable"]}" '
                f'stroke-width="1.9"/>'
            )
    route = bssc_route(cables)
    if route:
        m.add(
            f'<path d="{m.path(route)}" fill="none" stroke="{PALETTE["planned"]}" '
            f'stroke-width="2.6" stroke-dasharray="10 5" stroke-linecap="round"/>'
        )
    if label_zoom:
        for lon, lat, text, anchor in [
            (44.79, 42.10, "Tbilisi", "middle"),
            (49.87, 40.75, "Baku", "middle"),
            (44.51, 40.45, "Yerevan", "middle"),
            (46.35, 38.75, "Aras corridor (TRIPP)", "middle"),
        ]:
            m.label(lon, lat, text, size=10.5, colour=PALETTE["ink"], anchor=anchor)
    items = [
        (PALETTE["power"], "High-voltage line" + ("" if hv else " (EPM corridor, schematic)"), None),
        (PALETTE["fibre"], "Terrestrial fibre", None),
        (PALETTE["cable"], "Submarine cable", None),
        (PALETTE["planned"], "Announced route", "10 5"),
    ]
    if pipelines:
        items.insert(1, (PALETTE["pipe"], "Gas or oil pipeline", None))
    if railways:
        items.insert(2, (PALETTE["rail"], "Railway", "4 3"))
    m.legend(items)
    return m.render()


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; color: #37474f;
       background: #f4f6f7; }
.page { max-width: 1060px; margin: 0 auto; padding: 20px 30px 60px; }
h1 { font-size: 26px; color: #256081; margin-bottom: 6px; }
h2 { font-size: 18px; color: #256081; border-bottom: 3px solid #0277bd;
     padding-bottom: 6px; margin: 38px 0 14px; }
h3 { font-size: 14px; color: #0277bd; margin: 22px 0 8px; }
h4 { font-size: 13px; color: #546e7a; margin: 14px 0 6px; font-weight: 600; }
p { line-height: 1.62; margin-bottom: 10px; }
ul, ol { margin: 0 0 12px 20px; line-height: 1.62; }
li { margin-bottom: 5px; }
.subtitle { color: #78909c; font-size: 15px; margin-bottom: 4px; }
.meta { color: #90a4ae; font-size: 12px; margin-bottom: 20px; }
.cover { background: #256081; color: white; padding: 38px 30px; border-radius: 8px;
         margin-bottom: 26px; }
.cover h1 { color: white; font-size: 28px; }
.cover .subtitle { color: #b2ebf2; font-size: 16px; }
.cover .meta { color: #b0bec5; margin-bottom: 0; }
.keymsg { background: white; border-radius: 8px; padding: 18px 22px;
          box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 8px; }
.keymsg ol { margin-left: 18px; }
.keymsg li { margin-bottom: 9px; }
.card { background: white; border-radius: 8px; padding: 16px 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin: 14px 0; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin: 14px 0; }
.fig-block { background: white; border-radius: 8px; padding: 14px 16px;
             box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin: 16px 0; }
.fig-block .figtitle { font-size: 12px; font-weight: 700; color: #256081;
                       letter-spacing: 0.3px; margin-bottom: 8px; }
.fig-block .caption { font-size: 11px; color: #78909c; margin-top: 8px; line-height: 1.5; }
table.metrics { border-collapse: collapse; width: 100%; font-size: 12px; margin: 10px 0; }
table.metrics th { background: #256081; color: white; padding: 7px 10px; text-align: left;
                   font-weight: 600; }
table.metrics td { padding: 5px 10px; border-bottom: 1px solid #eceff1; vertical-align: top; }
table.metrics td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.metrics tr.focus { background: #e8f6f9; }
table.metrics tr:hover { background: #f4f6f7; }
.note { background: #fbf6e0; border-left: 3px solid #ddc32c; padding: 10px 14px;
        border-radius: 0 5px 5px 0; margin: 12px 0; font-size: 12px; line-height: 1.55; }
.caution { background: #eaf6f8; border-left: 3px solid #00838f; padding: 10px 14px;
           border-radius: 0 5px 5px 0; margin: 12px 0; font-size: 12px; line-height: 1.55; }
.badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 10.5px;
         font-weight: 700; }
.badge.ok { background: #b2ebf2; color: #00636d; }
.badge.gap { background: #f7edb8; color: #8a7410; }
.badge.no { background: #cfd8dc; color: #37474f; }
.src { font-size: 11px; color: #90a4ae; }
.sector { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px;
          font-weight: 700; margin-right: 3px; letter-spacing: 0.2px; }
.sector.on { background: #0277bd; color: #ffffff; }
.sector.off { background: #eceff1; color: #b0bec5; }
.cnum { display: inline-block; width: 17px; height: 17px; border-radius: 9px;
        color: #ffffff; font-size: 11px; font-weight: 700; text-align: center;
        line-height: 17px; margin-right: 5px; }
a { color: #0277bd; }
/* Intro schematics */
.diagram { background: white; border-radius: 8px; padding: 16px 18px 12px;
           box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin: 14px 0; }
.diagram .figtitle { font-size: 12px; font-weight: 700; color: #256081;
                     letter-spacing: 0.3px; margin-bottom: 10px; }
.diagram .caption { font-size: 11px; color: #78909c; margin-top: 6px; line-height: 1.5; }
.lede { font-size: 14px; line-height: 1.7; color: #37474f; }
.lede strong { color: #256081; }
"""


FONT = "Segoe UI, Arial, sans-serif"


def svg_text(x, y, text, size=11, colour="#37474f", weight="400", anchor="start"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{colour}" font-weight="{weight}" '
        f'text-anchor="{anchor}" font-family="{FONT}">{html.escape(text)}</text>'
    )


def svg_lines(x, y, lines, size=10.5, colour="#546e7a", step=13.5, anchor="start"):
    return "".join(
        svg_text(x, y + i * step, line, size=size, colour=colour, anchor=anchor)
        for i, line in enumerate(lines)
    )


# The five stages of a connection, from the seabed to the user. Each entry is
# a title, what the stage physically is, and what usually goes wrong there.
CHAIN = [
    ("Submarine cable",
     ["A few fibre pairs inside an", "armoured tube on the seabed.",
      "Carries a whole country's traffic."],
     ["3 to 5 years to build,", "hundreds of millions"]),
    ("Landing station",
     ["Where the cable comes ashore.", "A building, a power supply, and",
      "the legal point of entry."],
     ["A single national gateway", "is a chokepoint"]),
    ("Terrestrial backbone",
     ["Long-haul fibre between cities", "and borders, laid in a duct along",
      "a road, a rail line or a pipeline."],
     ["Cheap in a trench that", "is already open"]),
    ("Exchange and data centre",
     ["Where networks meet and content", "sits. Traffic that stops here does",
      "not pay for a foreign transit hop."],
     ["The usual binding", "constraint in this region"]),
    ("Access network",
     ["The last kilometres to homes,", "businesses and mobile masts.",
      "Where the user feels the service."],
     ["Most of the capital cost,", "least of the distance"]),
]


def diagram_chain(width=980):
    """The physical chain from the seabed to the user."""
    n = len(CHAIN)
    gap, pad = 18, 8
    bw = (width - 2 * pad - (n - 1) * gap) / n
    top, bh = 26, 44
    parts = [f'<rect width="{width}" height="192" fill="#ffffff"/>']
    for i, (title, body, risk) in enumerate(CHAIN):
        x = pad + i * (bw + gap)
        parts.append(
            f'<rect x="{x:.1f}" y="{top}" width="{bw:.1f}" height="{bh}" rx="5" '
            f'fill="{PALETTE["ink"]}"/>'
        )
        parts.append(svg_text(x + bw / 2, top + 19, f"{i + 1}", size=11,
                              colour=PALETTE["pale"], weight="700", anchor="middle"))
        parts.append(svg_text(x + bw / 2, top + 35, title, size=11.5,
                              colour="#ffffff", weight="700", anchor="middle"))
        if i < n - 1:
            ax = x + bw + gap / 2
            parts.append(
                f'<path d="M{ax - 5:.1f},{top + bh / 2 - 4}L{ax + 5:.1f},{top + bh / 2}'
                f'L{ax - 5:.1f},{top + bh / 2 + 4}Z" fill="{PALETTE["border"]}"/>'
            )
        parts.append(svg_lines(x, top + bh + 20, body))
        parts.append(
            f'<rect x="{x:.1f}" y="{top + bh + 62}" width="{bw:.1f}" height="44" rx="4" '
            f'fill="#f4f6f7" stroke="{PALETTE["border"]}" stroke-width="0.6"/>'
        )
        parts.append(svg_text(x + 8, top + bh + 76, "WHERE IT BINDS", size=8.5,
                              colour=PALETTE["accent"], weight="700"))
        parts.append(svg_lines(x + 8, top + bh + 89, risk, size=9.5, step=11))
    parts.append(svg_text(pad, 16, "International capacity", size=10.5,
                          colour=PALETTE["muted"], weight="600"))
    parts.append(svg_text(width - pad, 16, "National reach", size=10.5,
                          colour=PALETTE["muted"], weight="600", anchor="end"))
    parts.append(
        f'<line x1="{pad + 120}" y1="12" x2="{width - pad - 90}" y2="12" '
        f'stroke="{PALETTE["border"]}" stroke-width="0.8" stroke-dasharray="3 3"/>'
    )
    return (
        f'<svg viewBox="0 0 {width} 192" width="100%" '
        f'xmlns="http://www.w3.org/2000/svg">' + "".join(parts) + "</svg>"
    )


def diagram_couplings(width=980):
    """The three ways the digital layer touches the power and transport layers."""
    cw, gap = 306, 31
    h = 256
    parts = [f'<rect width="{width}" height="{h}" fill="#ffffff"/>']

    def panel(i, title):
        x = i * (cw + gap)
        parts.append(
            f'<rect x="{x}" y="0" width="{cw}" height="{h}" rx="6" fill="#fbfcfc" '
            f'stroke="{PALETTE["border"]}" stroke-width="0.7"/>'
        )
        parts.append(svg_text(x + 14, 22, title, size=12, colour=PALETTE["ink"], weight="700"))
        return x

    # 1. One right of way carries several networks.
    x = panel(0, "1. The same trench")
    parts.append(
        f'<rect x="{x + 18}" y="46" width="{cw - 36}" height="86" rx="4" '
        f'fill="{PALETTE["land_focus"]}" stroke="{PALETTE["border"]}" stroke-width="0.7"/>'
    )
    for j, (colour, wdt, dash, name) in enumerate([
        (PALETTE["pipe"], 5.0, "", "Gas pipeline"),
        (PALETTE["power"], 2.6, "", "Transmission line"),
        (PALETTE["rail"], 2.6, ' stroke-dasharray="5 3"', "Railway"),
        (PALETTE["fibre"], 2.2, "", "Fibre duct"),
    ]):
        y = 60 + j * 20
        parts.append(
            f'<line x1="{x + 30}" y1="{y}" x2="{x + 120}" y2="{y}" stroke="{colour}" '
            f'stroke-width="{wdt}"{dash} stroke-linecap="round"/>'
        )
        parts.append(svg_text(x + 130, y + 3.5, name, size=10, colour="#546e7a"))
    parts.append(svg_text(x + 18, 148, "One easement, one survey, one permit",
                          size=10.5, colour=PALETTE["accent"], weight="700"))
    parts.append(svg_lines(x + 18, 166, [
        "Most of the cost of laying fibre is the right of way",
        "and the digging, not the glass. A duct pulled in while",
        "a pipeline or a rail trench is already open costs a",
        "fraction of a standalone build. TANAP, TAP and the",
        "BSSC cable all carry fibre for this reason.",
    ]))

    # 2. The transmission line itself is the carrier.
    x = panel(1, "2. The grid is already a carrier")
    for px in (x + 60, x + 200):
        parts.append(
            f'<path d="M{px},130 L{px},64 M{px - 20},80 L{px + 20},80 '
            f'M{px - 14},96 L{px + 14},96" stroke="{PALETTE["power"]}" '
            f'stroke-width="2.2" fill="none" stroke-linecap="round"/>'
        )
    parts.append(
        f'<path d="M{x + 60},64 Q{x + 130},74 {x + 200},64" fill="none" '
        f'stroke="{PALETTE["fibre"]}" stroke-width="3.2"/>'
    )
    for yy, sag in ((80, 94), (96, 110)):
        parts.append(
            f'<path d="M{x + 60},{yy} Q{x + 130},{sag} {x + 200},{yy}" fill="none" '
            f'stroke="{PALETTE["power"]}" stroke-width="1.3" stroke-opacity="0.65"/>'
        )
    parts.append(
        f'<line x1="{x + 130}" y1="70" x2="{x + 232}" y2="52" '
        f'stroke="{PALETTE["fibre"]}" stroke-width="0.9" stroke-dasharray="2 2"/>'
    )
    parts.append(svg_text(x + 236, 55, "OPGW", size=10, colour=PALETTE["fibre"], weight="700"))
    parts.append(svg_text(x + 18, 148, "The earth wire contains the fibre",
                          size=10.5, colour=PALETTE["accent"], weight="700"))
    parts.append(svg_lines(x + 18, 166, [
        "Optical ground wire replaces the earth wire strung",
        "above the conductors. The utility installs it to",
        "protect and monitor the line, then has spare pairs",
        "it never uses. Leasing that dark fibre is revenue on",
        "an asset already paid for by the tariff.",
    ]))

    # 3. Digital is also a load on the power system.
    x = panel(2, "3. Digital is a new electricity load")
    parts.append(
        f'<rect x="{x + 28}" y="58" width="76" height="72" rx="4" fill="#ffffff" '
        f'stroke="{PALETTE["ink"]}" stroke-width="1.6"/>'
    )
    for j in range(4):
        yy = 68 + j * 16
        parts.append(
            f'<rect x="{x + 38}" y="{yy}" width="56" height="10" rx="2" '
            f'fill="{PALETTE["pale"]}" stroke="{PALETTE["border"]}" stroke-width="0.5"/>'
        )
    parts.append(svg_text(x + 66, 145, "Data centre", size=10,
                          colour=PALETTE["muted"], weight="600", anchor="middle"))
    parts.append(
        f'<path d="M{x + 112},94 L{x + 136},94 M{x + 130},89 L{x + 136},94 '
        f'L{x + 130},99" stroke="{PALETTE["planned"]}" stroke-width="2" fill="none"/>'
    )
    parts.append(
        f'<rect x="{x + 146}" y="58" width="130" height="72" rx="4" fill="#ffffff" '
        f'stroke="{PALETTE["border"]}" stroke-width="0.7"/>'
    )
    parts.append(
        f'<rect x="{x + 154}" y="70" width="114" height="48" '
        f'fill="{PALETTE["operational"]}" fill-opacity="0.22"/>'
    )
    parts.append(
        f'<line x1="{x + 154}" y1="70" x2="{x + 268}" y2="70" '
        f'stroke="{PALETTE["operational"]}" stroke-width="2"/>'
    )
    parts.append(svg_text(x + 211, 145, "Flat, firm, all year", size=10,
                          colour=PALETTE["muted"], weight="600", anchor="middle"))
    parts.append(svg_text(x + 18, 172, "Load that goes where power is cheap",
                          size=10.5, colour=PALETTE["accent"], weight="700"))
    parts.append(svg_lines(x + 18, 190, [
        "A data centre draws a near constant load and picks",
        "its site on tariff, reliability and carbon content.",
        "One 100 MW campus is a demand block on the scale",
        "of a small national peak, so it belongs in the",
        "capacity expansion, not in a separate note.",
    ]))
    return (
        f'<svg viewBox="0 0 {width} {h}" width="100%" '
        f'xmlns="http://www.w3.org/2000/svg">' + "".join(parts) + "</svg>"
    )


def diagram(title, svg, caption):
    return (
        f'<div class="diagram"><div class="figtitle">{html.escape(title)}</div>'
        f'{svg}<div class="caption">{caption}</div></div>'
    )


def table(headers, rows, aligns=None, focus_rows=()):
    aligns = aligns or [""] * len(headers)
    out = ['<table class="metrics"><thead><tr>']
    out += [f"<th>{html.escape(h)}</th>" for h in headers]
    out.append("</tr></thead><tbody>")
    for i, row in enumerate(rows):
        cls = ' class="focus"' if i in focus_rows else ""
        out.append(f"<tr{cls}>")
        for cell, align in zip(row, aligns):
            klass = ' class="num"' if align == "num" else ""
            out.append(f"<td{klass}>{cell}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def figure(title, svg, caption):
    return (
        f'<div class="fig-block"><div class="figtitle">{html.escape(title)}</div>'
        f"{svg}<div class=\"caption\">{caption}</div></div>"
    )


def main(extra_out: Path | None = None) -> None:
    index = CountryIndex(BASEMAP)
    fibre = load("itu_fibre.geojson")
    cables = load("submarine_cables.geojson")
    landings = load("submarine_landings.geojson")
    pdb = load("peeringdb.json")
    wdi = load("wdi_ict.json")
    hv = load("osm_hv_lines.geojson")
    pipelines = load("osm_pipelines.geojson")
    railways = load("osm_railways.geojson")
    epm_lines = json.load(open(EPM_LINES, encoding="utf8")) if EPM_LINES.exists() else None

    if not all([fibre, cables, landings, pdb, wdi]):
        raise SystemExit("Missing data. Run fetch_data.py first.")

    km = fibre_km_by_country(fibre, index)
    order = sorted(km, key=lambda c: -sum(km[c].values()))
    total_op = sum(v.get("Operational", 0) for v in km.values())
    total_new = sum(
        v.get("Planned", 0) + v.get("Under Construction", 0) for v in km.values()
    )

    # -- interconnection statistics
    ix_rows = []
    for code, rows in pdb["ix"].items():
        for ix in rows:
            raw = ix.get("city", "") or ""
            # A multi-site exchange lists every city, so name the country instead.
            place = code if "," in raw else short_city(raw)
            ix_rows.append((place, code, ix["name"], ix.get("net_count") or 0))
    ix_rows.sort(key=lambda r: -r[3])
    fac_counts = {code: len(rows) for code, rows in pdb["fac"].items()}
    ix_counts = {code: len(rows) for code, rows in pdb["ix"].items()}
    # A network can join several exchanges, so summing member counts across
    # exchanges double counts. Report the largest exchange in each country.
    largest_ix = defaultdict(int)
    for city, code, name, nets in ix_rows:
        largest_ix[code] = max(largest_ix[code], nets)

    # -- co-location metric
    coloc_total, coloc_near = (None, None)
    if hv:
        coloc_total, coloc_near = colocation(hv, fibre, index)

    parts = [
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>Digital Connectivity in the Black Sea and South Caucasus</title>",
        f"<style>{CSS}</style></head><body><div class=\"page\">",
        '<div class="cover"><h1>Digital Connectivity in the Black Sea '
        "and South Caucasus</h1>"
        '<div class="subtitle">Infrastructure screening and synergies with '
        "energy and transport</div>"
        f'<div class="meta">Internal analysis note &middot; Black Sea regional power '
        f"trade study &middot; {date.today().isoformat()}</div></div>",
    ]

    # ---------------- 1. how the sector works, for a reader new to it
    parts.append("<h2>1. What this sector is, in one page</h2>")
    parts.append(
        '<p class="lede">International data traffic is carried on optical fibre: '
        "submarine cables between countries, terrestrial backbone along road, rail and "
        "transmission corridors, and access networks to the end user. Transmission "
        "capacity on an existing route is comparatively cheap and rarely binding. Cost "
        "and risk concentrate in <strong>rights of way, landing points and the "
        "arrangements under which networks exchange traffic</strong>. This section sets "
        "out the components of that chain and the points at which it meets the power "
        "and transport sectors.</p>"
    )
    parts.append(
        diagram(
            "The chain, from the seabed to the user",
            diagram_chain(),
            "Each stage is a separate business with separate owners. A country can hold "
            "abundant capacity at stage 1 and still buy expensive international transit "
            "because stage 4 is missing. That is the case across the South Caucasus, and "
            "it is why this note spends more space on exchange points than on cable "
            "kilometres.",
        )
    )
    parts.append(
        "<p>The link to the rest of this study is not an analogy. It is the same ground, "
        "the same rights of way and, increasingly, the same load. Three couplings do the "
        "work, and each one is measurable.</p>"
    )
    parts.append(
        diagram(
            "Three couplings between the digital, power and transport layers",
            diagram_couplings(),
            "The first two make fibre cheaper when it follows an energy or transport "
            "corridor. The third runs the other way: digital infrastructure is itself a "
            "growing electricity load, sited on the strength of the tariff and the grid. "
            "Sections 7.1 to 7.3 quantify each one for this region.",
        )
    )
    parts.append(
        '<div class="note"><b>The vocabulary used below.</b> '
        "<i>Backbone</i> is long-haul fibre between cities, as opposed to the access "
        "network reaching the end user. A <i>landing point</i> is where a submarine cable "
        "comes ashore. An <i>internet exchange</i>, or IXP, is a neutral site where "
        "networks swap traffic directly instead of paying a third party to carry it; the "
        "number of member networks is the usual measure of how useful it is. <i>Transit</i> "
        "is that paid carriage. <i>Dark fibre</i> is installed but unlit glass. <i>OPGW</i> "
        "is optical ground wire, the earth wire of a transmission line with fibres inside "
        "it.</div>"
    )

    # ---------------- 2. key messages
    parts.append("<h2>2. Key messages</h2>")
    parts.append(
        '<div class="keymsg"><ol>'
        "<li><b>The physical corridor is already shared.</b> Fibre follows the TANAP gas "
        "pipeline across Turkiye over 1,850 km, follows TAP into Italy, will follow the "
        "BSSC HVDC cable across the Black Sea at 40 Tbps, and TRIPP bundles rail, road, "
        "gas, power and fibre into a single 43 km corridor through Armenia. Co-location "
        "is established practice in this region, not a concept to be introduced.</li>"
        "<li><b>The South Caucasus is a bottleneck, not a corridor.</b> "
        f"Georgia, Armenia and Azerbaijan together hold "
        f"{sum(km[c].get('Operational', 0) for c in FOCUS if c in km):,.0f} km of mapped "
        "operational backbone, less than Romania alone. Georgia has one usable submarine "
        "landing, Poti, on a cable commissioned in 2008. This mirrors the power system "
        "diagnosis exactly.</li>"
        "<li><b>The binding constraint is interconnection, not cable.</b> Sofia and "
        "Bucharest each host exchanges with more than 120 peering networks. Tbilisi has "
        "14, and Azerbaijan has no public exchange listed at all. Capacity added upstream "
        "will not convert into lower prices without exchange points and neutral "
        "facilities downstream.</li>"
        "<li><b>2026 is the hinge year.</b> The Trans-Caspian cable enters service, "
        "Armenia and Azerbaijan signed reciprocal internet transit on 22 June, and the "
        "EU launched its Connectivity Agenda Platform on 23 June with up to EUR 2 billion "
        "coordinating transport, energy, digital and trade.</li>"
        "<li><b>The reverse link matters as much.</b> Romania's announced Black Sea AI "
        "gigafactory would draw 1.5 GW. Data centre load belongs inside the demand "
        "scenarios of the power model, not in a separate digital workstream.</li>"
        "</ol></div>"
    )

    # ---------------- 2. why now
    parts.append("<h2>3. Why now: three corridors converging</h2>")
    parts.append(
        "<p>Three separate corridor programmes reach decision points within the same "
        "eighteen months, and all three cross the same geography as the electricity "
        "interconnections already modelled in this study.</p>"
    )
    parts.append(
        table(
            ["Milestone", "Date", "What it changes"],
            [
                [
                    "Caucasus Cable System commissioned",
                    "2008",
                    "The only direct Caucasus to EU submarine link, 12.6 Tbps after upgrade",
                ],
                [
                    "SOCAR Fiber enters service along TANAP",
                    "2013",
                    "1,850 km of fibre in the gas pipeline trench across Turkiye",
                ],
                [
                    "EXA Infrastructure and SOCAR Fiber agreement",
                    "Jul 2024",
                    "Terrestrial Greece to Georgia route as Red Sea diversity",
                ],
                [
                    "BSSC added to the EU list of Projects of Mutual Interest",
                    "Dec 2025",
                    "Electricity and fibre treated as one project",
                ],
                [
                    "TRIPP development company established",
                    "Dec 2025",
                    "Rail, road, gas, power and fibre in one corridor through Armenia",
                ],
                [
                    "Transelectrica and GSE memorandum on BSSC",
                    "Feb 2026",
                    "Coordinated studies, surveys and joint financing",
                ],
                [
                    "World Bank TC-GATE approved, USD 372 m",
                    "Jun 2026",
                    "Georgian segment of the Middle Corridor",
                ],
                [
                    "AzerTelecom and Telecom Armenia transit agreement",
                    "22 Jun 2026",
                    "First Armenian traffic across Azerbaijani territory since 1991",
                ],
                [
                    "EU Connectivity Agenda Platform launched",
                    "23 Jun 2026",
                    "Up to EUR 2 bn, transport, energy, digital and trade in one frame",
                ],
                [
                    "Trans-Caspian fibre commercial service",
                    "Q3 2026",
                    "400 Tbps Sumgait to Aktau, closes the middle route",
                ],
                [
                    "BSSC fibre in service",
                    "~2030",
                    "40 Tbps alongside the Georgia to Romania HVDC cable",
                ],
            ],
        )
    )

    parts.append(
        "<p>Those milestones sit on six corridors. Each one is a route that some "
        "combination of the power, gas, transport and telecom sectors is already "
        "using or has announced, and the overlap is what makes the digital layer "
        "relevant to a power study. The map below is the orientation for everything "
        "that follows; the surveyed geometry is in Maps 2 to 7.</p>"
    )
    parts.append(
        figure(
            "Map 1. The six corridors, and what each one carries",
            map_corridors(index, cables),
            "Corridor spines are schematic, drawn through the real cities and "
            "landing points each corridor connects. They are for orientation: the "
            "surveyed routes are in the maps that follow. The Black Sea corridor is "
            "the one exception, traced on the surveyed Caucasus Cable System "
            "crossing. "
            '<span class="src">Source: project announcements, EU Global Gateway and '
            "Connectivity Agenda documents, operator statements.</span>",
        )
    )
    parts.append(corridor_table())
    parts.append(
        "<p>Two readings follow directly. Corridor 1 is the proof that the "
        "co-location works: gas, power, rail and fibre already share one easement "
        "from Baku to central Turkiye, and the fibre was laid in a trench opened for "
        "gas. Corridors 3 and 4 are the ones this study can still influence, because "
        "neither has reached financial close and in both cases the fibre is a "
        "marginal cost on an asset that is being built anyway.</p>"
    )

    # ---------------- 3. what exists
    parts.append("<h2>4. What exists today</h2>")
    parts.append("<h3>4.1 Submarine cables</h3>")
    parts.append(
        f"<p>Twelve submarine systems intersect the screening window, landing at "
        f"{len(landings['features'])} points. The structure is asymmetric. Bulgaria, "
        "Romania and Turkiye sit on several routes each. The entire South Caucasus "
        "depends on a single useful cable, the Caucasus Cable System from Poti to "
        "Balchik, commissioned in 2008, or on transit through Russia.</p>"
    )
    parts.append(
        figure(
            "Map 2. Submarine cable systems, Black Sea and Caspian",
            map_submarine(index, cables, landings),
            "Cable routes are the published alignments from the source dataset. The "
            "BSSC fibre is not published as geometry, so it is traced along the "
            "Caucasus Cable System, the surveyed crossing that already runs in the "
            "same latitude band south of the Crimean shelf, with the two landfalls "
            "moved to the announced ones at Anaklia and Constanta. "
            '<span class="src">Source: submarinecablemap.com API v3 (CC BY-NC-SA 3.0); '
            "basemap from the study repository.</span>",
        )
    )
    parts.append(
        table(
            ["System", "Route", "In service", "Capacity", "Note"],
            [
                [
                    "Caucasus Cable System",
                    "Poti (GE) to Balchik (BG), 1,182 km",
                    "2008",
                    "12.6 Tbps",
                    "Only direct Caucasus to EU link. Owned by Caucasus Online",
                ],
                ["KAFOS", "Istanbul to Bucharest, 504 km", "2021 relaunch", "8 Tbps", ""],
                [
                    "BSFOCS",
                    "Russia, Turkiye, Bulgaria, 1,300 km",
                    "2001",
                    "20 Gbps",
                    "Effectively obsolete",
                ],
                [
                    "Trans-Caspian FOCL",
                    "Sumgait (AZ) to Aktau (KZ), 380 km",
                    "Q3 2026",
                    "400 Tbps",
                    "AzerTelecom and Kazakhtelecom, laying complete",
                ],
                [
                    "BSSC fibre",
                    "Anaklia (GE) to Constanta (RO)",
                    "~2030",
                    "40 Tbps initial",
                    "Laid with the HVDC cable",
                ],
                [
                    "Georgia-Russia, Kerch, Energy Bridge",
                    "Russian axis",
                    "various",
                    "n/a",
                    "Carries geopolitical exposure",
                ],
            ],
            focus_rows=(0, 3, 4),
        )
    )

    parts.append("<h3>4.2 Terrestrial backbone</h3>")
    parts.append(
        f"<p>The ITU transmission catalogue holds {len(fibre['features']):,} mapped fibre "
        f"links inside the window, {total_op:,.0f} km operational and {total_new:,.0f} km "
        "planned or under construction. The terrestrial picture confirms the submarine "
        "one. Turkiye and Russia dominate, Ukraine and Romania follow, and the three "
        "South Caucasus countries together account for a small fraction.</p>"
    )
    parts.append(
        figure(
            "Map 3. Mapped terrestrial fibre backbone, by status",
            map_terrestrial(index, fibre),
            "Solid is operational, dashed is under construction, dotted is "
            "planned. The ITU catalogue is a node and link graph: every record is a "
            "single segment between two mapped points, with no intermediate "
            "geometry, so the links are straight by construction and not by "
            "simplification. The nodes are drawn to make that explicit. Link "
            "lengths are therefore lower bounds on the fibre actually in the "
            "ground. "
            '<span class="src">Source: ITU BBmaps geocatalogue, layer '
            "itu-geocatalogue:trx_geocatalogue, retrieved via WFS.</span>",
        )
    )
    bar_rows = [
        (
            DISPLAY.get(c, c),
            {
                "Operational": km[c].get("Operational", 0),
                "Under construction": km[c].get("Under Construction", 0),
                "Planned": km[c].get("Planned", 0),
            },
        )
        for c in order
    ]
    parts.append(
        figure(
            "Figure 1. Mapped backbone length by country, kilometres",
            bar_chart(
                bar_rows,
                colours={
                    "Operational": PALETTE["operational"],
                    "Under construction": PALETTE["building"],
                    "Planned": PALETTE["planned"],
                },
                unit=" km",
            ),
            "Segment level attribution: each fibre segment is assigned to the country "
            "containing its midpoint, so cross-border links split correctly. Lengths "
            "are great-circle sums over the mapped polylines and are clipped to the "
            "screening window, so Russia, Turkiye, Ukraine and Kazakhstan are "
            "truncated. "
            '<span class="src">Source: ITU BBmaps, own calculation.</span>',
        )
    )
    parts.append(
        '<div class="note"><b>Read this as mapped backbone, not as total national '
        "fibre.</b> The ITU catalogue compiles operator disclosures and public sources. "
        "Coverage is uneven between countries and it understates national distribution "
        "networks. It is reliable for comparing corridors, not for national fibre "
        "inventories.</div>"
    )

    parts.append("<h3>4.3 Named corridors</h3>")
    parts.append(
        table(
            ["Corridor", "Route", "Status", "Why it matters here"],
            [
                [
                    "Digital Silk Way (NEQSOL, AzerTelecom)",
                    "China, Kazakhstan, Caspian, Azerbaijan, Georgia, Turkiye, EU",
                    "Service Q3 2026",
                    "The missing middle closes this year",
                ],
                [
                    "<b>SOCAR Fiber</b>",
                    "1,850 km buried along the TANAP gas pipeline, east to west Turkiye",
                    "Operational since 2013",
                    "<b>Fibre already follows the southern gas corridor</b>",
                ],
                [
                    "<b>EXA Trans Adriatic Express</b>",
                    "Along the TAP gas pipeline into Italy",
                    "Operational",
                    "<b>Same logic, extended into the EU</b>",
                ],
                [
                    "EXA and SOCAR Fiber joint route",
                    "Greece, Turkiye, Georgia, extension toward Iraq",
                    "In development since Jul 2024",
                    "Terrestrial diversity against Red Sea outages",
                ],
                [
                    "<b>TRIPP</b>",
                    "43 km through Armenia: rail, road, gas, power and fibre",
                    "Company established Dec 2025, rail target 2028",
                    "<b>The multi-infrastructure corridor test case</b>",
                ],
                [
                    "EPEG",
                    "Frankfurt, Ukraine, Russia, Azerbaijan, Iran, Oman, 10,000 km",
                    "Operational since 2012",
                    "3.2 Tbps southern alternative",
                ],
                [
                    "Armenia to Georgia and to Iran",
                    "GNC-Alfa, Telecom Armenia",
                    "Operational",
                    "Iran link is backup only, Georgia is the single real route",
                ],
                [
                    "Armenia and Azerbaijan transit",
                    "Reciprocal, opens fibre to Nakhchivan through Armenia",
                    "Signed 22 Jun 2026",
                    "First since 1991, mirrors the power trade question",
                ],
            ],
            focus_rows=(1, 2, 4),
        )
    )

    # ---------------- 4. pipeline
    parts.append("<h2>5. What is being built</h2>")
    parts.append(
        f"<p>Of the mapped backbone, {total_new:,.0f} km is planned or under "
        "construction inside the window. Armenia and Moldova show the largest planned "
        "additions relative to their installed base, which is consistent with both "
        "countries seeking route diversity away from a single transit neighbour.</p>"
    )
    parts.append(
        figure(
            "Map 4. Project pipeline: fibre under construction and planned",
            map_projects(index, fibre, cables, railways),
            "Operational links are removed for legibility. Dashed is under "
            "construction, dotted is planned, and the longer dash is submarine. "
            "Terrestrial links carry the straight node to node geometry of the ITU "
            "catalogue, as in Map 3. The two submarine routes do not: the BSSC "
            "follows the surveyed Caucasus Cable System crossing with the announced "
            "landfalls, and TRIPP follows the real closed railway alignment along "
            "the Aras as mapped in OpenStreetMap, not a line between endpoints. "
            '<span class="src">Source: ITU BBmaps, submarinecablemap.com, project '
            "announcements.</span>",
        )
    )

    # ---------------- 5. interconnection
    parts.append("<h2>6. The binding constraint is interconnection</h2>")
    parts.append(
        "<p>Cable capacity is only half of the connectivity cost. The other half is "
        "where networks meet. An internet exchange with many members keeps traffic "
        "local and cheap. An exchange with few members forces traffic onto paid "
        "international transit. On this measure the regional gradient is severe.</p>"
    )
    top_ix = [r for r in ix_rows if r[3] > 0][:14]
    parts.append(
        figure(
            "Figure 2. Networks peering at each internet exchange",
            bar_chart(
                [(f"{name} ({city})", {"Networks": nets}) for city, code, name, nets in top_ix],
                colours={"Networks": PALETTE["accent"]},
                pad_left=210,
            ),
            "Member counts are self-reported by participating networks. "
            '<span class="src">Source: PeeringDB API.</span>',
        )
    )
    parts.append(
        figure(
            "Map 5. Internet exchanges and interconnection facilities",
            map_interconnection(index, pdb),
            "The shading is the finding: it is the total number of networks "
            "peering anywhere in each country, and it falls by an order of magnitude "
            "between the EU shore of the Black Sea and the Caucasus. Bubbles mark "
            "the exchange cities, scaled by the networks at the largest exchange in "
            "each. Dark dots are interconnection facilities, typically "
            "carrier-neutral data centres, and they show that the shortage is not "
            "buildings: Azerbaijan has facilities and no public exchange at all. "
            '<span class="src">Source: PeeringDB API.</span>',
        )
    )
    cc_names = {
        "GE": "Georgia", "AM": "Armenia", "AZ": "Azerbaijan", "TR": "Turkiye",
        "RO": "Romania", "BG": "Bulgaria", "MD": "Moldova", "UA": "Ukraine",
        "KZ": "Kazakhstan", "GR": "Greece",
    }
    wdi_by_country = {}
    for code, payload in wdi.items():
        for country, (year, value) in payload["values"].items():
            wdi_by_country.setdefault(country, {})[code] = (year, value)
    rows = []
    focus_idx = []
    for i, (cc, name) in enumerate(cc_names.items()):
        if cc in ("GR",):
            continue
        w = wdi_by_country.get(name, {})
        def fmt(key):
            if key not in w:
                return "n/a"
            year, value = w[key]
            return f"{value:,.1f} <span class='src'>({year})</span>"
        if name in ("Georgia", "Armenia", "Azerbaijan"):
            focus_idx.append(len(rows))
        rows.append(
            [
                name,
                f"{ix_counts.get(cc, 0)}",
                f"{largest_ix.get(cc, 0)}",
                f"{fac_counts.get(cc, 0)}",
                fmt("IT.NET.USER.ZS"),
                fmt("IT.NET.BBND.P2"),
            ]
        )
    parts.append(
        table(
            [
                "Country",
                "Exchanges",
                "Networks at the largest exchange",
                "Interconnection facilities",
                "Internet users, % of population",
                "Fixed broadband per 100",
            ],
            rows,
            aligns=["", "num", "num", "num", "num", "num"],
            focus_rows=tuple(focus_idx),
        )
    )
    parts.append(
        "<p>Three readings stand out. Azerbaijan promotes itself as the regional digital "
        "hub yet has no public exchange listed at all, so its interconnection happens "
        "abroad or inside single operators. Georgia has one exchange, IXP.ge in Tbilisi, "
        "with 14 member networks, against 134 at NetIX in Sofia and 125 at InterLAN-IX "
        "in Bucharest. And the Turkish operator TurkIX runs two sites: the Sofia site "
        "carries 37 member networks, the Istanbul site carries 4, so Turkiye exports "
        "interconnection value to Bulgaria. Demand is not the constraint anywhere. "
        "Internet use runs from 77 to 93 percent of the population across the region.</p>"
    )

    # ---------------- 6. energy and digital
    parts.append("<h2>7. Energy and digital: four channels</h2>")
    parts.append("<h3>7.1 The same trench</h3>")
    parts.append(
        "<p>The strongest synergy is the least conceptual. Linear infrastructure shares "
        "rights of way, survey work, permitting and civil works. In this region that is "
        "already the dominant model: SOCAR Fiber runs 1,850 km in the TANAP trench and "
        "serves 20 provinces, EXA runs its Trans Adriatic Express along TAP, the BSSC "
        "will carry 40 Tbps of fibre on the HVDC route, and TRIPP puts rail, road, gas, "
        "power and fibre in one 43 km envelope. The marginal cost of adding fibre to an "
        "energy project under construction is small relative to a standalone build.</p>"
    )
    parts.append(
        figure(
            "Map 6. Energy, digital and transport corridors overlaid",
            map_master(index, fibre, cables, hv, pipelines, railways, epm_lines),
            (
                "High-voltage lines from OpenStreetMap. "
                if hv
                else "Electricity corridors shown as schematic zone-to-zone links from "
                "the EPM topology, not as surveyed line routes. "
            )
            + "Fibre from the ITU catalogue, submarine cables from "
            "submarinecablemap.com, railways from OpenStreetMap. The announced BSSC route "
            "follows the surveyed Caucasus Cable System crossing. "
            '<span class="src">Sources: OpenStreetMap contributors (ODbL), ITU BBmaps, '
            "submarinecablemap.com, study repository.</span>",
        )
    )

    parts.append("<h3>7.2 The grid as a carrier</h3>")
    if coloc_total:
        rows = []
        focus_idx = []
        for country in sorted(coloc_total, key=lambda c: -coloc_total[c]):
            if coloc_total[country] < 200:
                continue
            share = coloc_near[country] / coloc_total[country] * 100
            if country in FOCUS:
                focus_idx.append(len(rows))
            rows.append(
                [
                    DISPLAY.get(country, country),
                    f"{coloc_total[country]:,.0f}",
                    f"{coloc_near[country]:,.0f}",
                    f"{share:.0f} %",
                ]
            )
        parts.append(
            "<p>Optical ground wire turns a transmission line into a telecom asset, and "
            "spare fibre pairs can be leased. To size the opportunity we measured how "
            "much of the mapped high-voltage network already runs within 3 km of a "
            "mapped fibre link. Where the share is low, the grid is a corridor that "
            "digital infrastructure has not used.</p>"
        )
        parts.append(
            table(
                [
                    "Country",
                    "Mapped HV line length, km",
                    "Within 3 km of mapped fibre, km",
                    "Share",
                ],
                rows,
                aligns=["", "num", "num", "num"],
                focus_rows=tuple(focus_idx),
            )
        )
        parts.append(
            '<div class="note"><b>Indicative only.</b> This is a co-location rate '
            "between two independently mapped datasets, not a survey of installed "
            "optical ground wire. The extract covers lines tagged at 220 kV and above "
            "inside the screening window, so national totals are truncated at the window "
            "edge and OpenStreetMap coverage is uneven between countries. The ITU fibre "
            "catalogue understates national networks. Both biases push the measured "
            "share down. The figure identifies where to ask the transmission operators, "
            "it does not replace asking them.</div>"
        )
    else:
        parts.append(
            '<div class="caution">The OpenStreetMap high-voltage extract was not '
            "available at build time, so the co-location metric between the transmission "
            "network and the fibre backbone could not be computed. Re-run "
            "<code>fetch_data.py</code> when the Overpass service responds and rebuild.</div>"
        )
    parts.append(
        "<p>The definitive answer sits with the transmission operators. GSE, Azerenerji, "
        "Transelectrica, ESO and TEIAS each know their optical ground wire inventory, "
        "spare pairs and any existing leases. None of it is public, and it is the single "
        "highest-value data request coming out of this screening.</p>"
    )

    parts.append("<h3>7.3 Digital demand on the power system</h3>")
    parts.append(
        "<p>The causality also runs the other way, and this is the channel that belongs "
        "directly inside the existing model. Romania's announced Black Sea AI "
        "gigafactory, valued at up to EUR 5 billion across Cernavoda and Doicesti, would "
        "require about 1.5 GW of supply, with commissioning targeted at the end of 2028. "
        "Georgia is marketing itself for AI hosting on the strength of low-carbon "
        "generation, while carrying a seasonal deficit that the current scenarios already "
        "capture. A data centre load block is a demand assumption, and it should be "
        "tested as such rather than treated as a separate workstream.</p>"
    )

    parts.append("<h3>7.4 Corridor governance</h3>")
    parts.append(
        "<p>Shared corridors share risks. Rights of way, permitting, environmental "
        "approval and political exposure are common to the electricity and the digital "
        "layer, and the regional record is not reassuring. The ownership of Caucasus "
        "Online, which holds the only direct Caucasus to EU cable, has been contested "
        "since 2019: the Georgian regulator appointed a special manager in October 2020 "
        "after the electronic communications law was amended, and the dispute went to "
        "international arbitration. Any assessment that treats the existing submarine "
        "link as a stable asset is understating the case for the BSSC fibre.</p>"
    )

    # ---------------- 7. transport
    parts.append("<h2>8. The transport layer</h2>")
    parts.append(
        "<p>The same geography carries the Middle Corridor. The World Bank approved "
        "TC-GATE in June 2026, USD 372 million for the Georgian segment of the "
        "Trans-Caspian route. The Baku to Tbilisi to Kars railway, open since 2017 and "
        "upgraded in 2024, runs the length of the same valley system as the electricity "
        "interconnections and the fibre. TRIPP is the explicit bundling of all of it. "
        "The European Commission estimates TRIPP cuts transit time by about 25 percent "
        "against the Baku to Tbilisi to Kars alternative, though commercial demand is "
        "unproven and Georgian transit incentives cut against the route.</p>"
    )
    if railways:
        parts.append(
            figure(
                "Map 7. The South Caucasus corridor at corridor scale",
                map_master(
                    index, fibre, cables, hv, pipelines, railways, epm_lines,
                    bbox=(42.6, 38.4, 50.6, 42.8), height=430, min_px=0.8,
                    label_zoom=True,
                ),
                "The same layers as Map 6, zoomed on the Kura and Aras valleys. The "
                "point of the zoom is that the rail, pipeline, transmission and fibre "
                "alignments are not merely in the same country, they are in the same "
                "valleys, often within a few kilometres of each other. "
                '<span class="src">Sources: OpenStreetMap contributors (ODbL), ITU '
                "BBmaps, submarinecablemap.com.</span>",
            )
        )
    parts.append(
        '<div class="note">For the purposes of this study the transport layer matters '
        "in one specific way: it establishes that a corridor authority, a financing "
        "vehicle and an environmental process already exist for these alignments. "
        "Adding a fibre duct to a project that is already trenching is an "
        "administrative decision more than an engineering one.</div>"
    )

    # ---------------- 8. limits
    parts.append("<h2>9. Data gaps, limits and what to request</h2>")
    parts.append(
        table(
            ["Item", "Status", "Comment"],
            [
                [
                    "ITU terrestrial fibre catalogue",
                    '<span class="badge ok">open</span>',
                    "Retrieved by WFS, no credentials. Mapped backbone only, uneven "
                    "national coverage, and node to node segments rather than "
                    "routed geometry, so lengths are lower bounds",
                ],
                [
                    "Submarine cable routes",
                    '<span class="badge ok">open</span>',
                    "CC BY-NC-SA 3.0, non-commercial. Routes are schematic, not geodesic",
                ],
                [
                    "PeeringDB exchanges and facilities",
                    '<span class="badge ok">open</span>',
                    "Self-reported by networks. Undercounts closed or private interconnection",
                ],
                [
                    "World Bank ICT indicators",
                    '<span class="badge ok">open</span>',
                    "CC BY 4.0, latest available year per country",
                ],
                [
                    "OpenStreetMap HV lines and pipelines",
                    '<span class="badge ok">open</span>'
                    if hv
                    else '<span class="badge gap">partial</span>',
                    "ODbL. Coverage varies by country, Overpass is a shared service "
                    "and can time out",
                ],
                [
                    "Ookla broadband performance tiles",
                    '<span class="badge gap">next step</span>',
                    "CC BY-NC-SA 4.0, roughly 600 m tiles, quarterly. Also served "
                    "directly by the ITU GeoServer",
                ],
                [
                    "Transmission operator optical ground wire inventories",
                    '<span class="badge no">to request</span>',
                    "<b>The highest-value gap.</b> GSE, Azerenerji, Transelectrica, "
                    "ESO, TEIAS",
                ],
                [
                    "Lit capacity and IP transit prices",
                    '<span class="badge no">not open</span>',
                    "Commercial datasets only. Without them the landlocked cost "
                    "penalty cannot be quantified",
                ],
                [
                    "EU4Digital and DESI broadband monitoring",
                    '<span class="badge gap">next step</span>',
                    "World Bank implements the facility, internal access likely",
                ],
            ],
        )
    )
    parts.append(
        '<div class="caution"><b>Licence caution.</b> The submarine cable dataset is '
        "CC BY-NC-SA 3.0 and the Ookla tiles are CC BY-NC-SA 4.0. Both are "
        "non-commercial with attribution and share-alike. Clear the licensing before any "
        "external publication or reuse of the maps in a client deliverable.</div>"
    )

    # ---------------- 9. implications
    parts.append("<h2>10. What this means for the power study</h2>")
    parts.append(
        "<ol>"
        "<li><b>The BSSC benefit case is understated.</b> The study currently values the "
        "cable on power system economics alone. The fibre carried on the same route "
        "provides the first non-Russian, non-2008 digital path from the Caucasus to the "
        "EU. That benefit accrues to the same project at close to zero marginal cost.</li>"
        "<li><b>Add a data centre demand sensitivity.</b> A 1.5 GW block in Romania is "
        "material for the regional balance, and a smaller Georgian block interacts "
        "directly with the seasonal deficit already in the model.</li>"
        "<li><b>The Armenia question repeats exactly.</b> The June 2026 transit agreement "
        "is the digital analogue of the Armenia to Azerbaijan power trade scenarios. The "
        "same political precondition unlocks both, which strengthens the case for "
        "treating them together rather than separately.</li>"
        "<li><b>Request the optical ground wire inventories now.</b> They are needed for "
        "the co-location analysis, they are not public, and the transmission operators "
        "are already engaged on this study.</li>"
        "</ol>"
    )

    # ---------------- 10. annex
    parts.append("<h2>11. Annex: sources and reproducibility</h2>")
    parts.append(
        "<p>Every figure in this note is produced from a script in this folder. "
        "<code>fetch_data.py</code> downloads the open datasets into <code>data/</code>, "
        "and <code>build_note.py</code> computes the statistics and renders this page. "
        "No manual data entry is involved except for the corridor inventory table and "
        "the milestone table, which are compiled from the sources listed below.</p>"
    )
    parts.append(
        table(
            ["Dataset", "Endpoint", "Licence"],
            [
                [
                    "ITU terrestrial transmission links",
                    "bbmaps.itu.int/geoserver/ows, layer itu-geocatalogue:trx_geocatalogue",
                    "ITU, attribution",
                ],
                [
                    "Submarine cables and landing points",
                    "submarinecablemap.com/api/v3",
                    "CC BY-NC-SA 3.0",
                ],
                ["Internet exchanges and facilities", "peeringdb.com/api", "Open, attribution"],
                ["ICT indicators", "api.worldbank.org/v2", "CC BY 4.0"],
                ["High-voltage lines, pipelines", "Overpass API, OpenStreetMap", "ODbL"],
                [
                    "Country boundaries",
                    "epm/input/data_blacksea/extras/background_countries.geojson",
                    "Study repository",
                ],
            ],
        )
    )
    parts.append(
        "<h3>Narrative sources</h3>"
        "<ul class='src'>"
        "<li>Georgian State Electrosystem, Black Sea Submarine Cable Project feasibility "
        "study summary</li>"
        "<li>European Commission, Black Sea Connectivity Submarine Electricity Cable, "
        "Global Gateway</li>"
        "<li>European Commission, Connectivity Agenda Platform launch, 23 June 2026</li>"
        "<li>Telecom Armenia and AzerTelecom, internet traffic transit agreement, "
        "22 June 2026</li>"
        "<li>Carnegie Endowment, Rewiring the South Caucasus: TRIPP and the New "
        "Geopolitics of Connectivity, March 2026</li>"
        "<li>EXA Infrastructure and SOCAR Fiber, Red Sea route diversity partnership, "
        "July 2024</li>"
        "<li>NEQSOL Holding, Digital Silk Way project documentation</li>"
        "<li>World Bank, TC-GATE project approval, June 2026, and BSSC preparatory "
        "activities, May 2024</li>"
        "<li>Jamestown Foundation and Forbes Georgia reporting on the Caucasus Online "
        "dispute</li>"
        "</ul>"
    )
    parts.append(
        f'<p class="src" style="margin-top:26px">Generated {date.today().isoformat()} '
        "by build_note.py. Internal working document, not for external circulation "
        "without a licence review.</p>"
    )
    parts.append("</div></body></html>")

    OUT.write_text("".join(parts), encoding="utf8")
    if extra_out is not None:
        target = extra_out / OUT.name if extra_out.is_dir() else extra_out
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OUT, target)
        print(f"Copied to {target}")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"  fibre links: {len(fibre['features'])}, countries scored: {len(km)}")
    if coloc_total:
        print(f"  co-location computed for {len(coloc_total)} countries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="folder or file path to copy the rendered note to, in addition to "
        "writing it beside this script",
    )
    main(parser.parse_args().out)
