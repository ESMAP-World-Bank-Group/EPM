"""Shared map engine for the Black Sea internal analysis notes.

Lifted from the digital connectivity screening so the sector briefs and that
note draw from one engine: geometry helpers, a point in polygon country index
over the repository basemap, and an equirectangular SVG canvas.
"""

from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_BLACKSEA = ROOT / "epm" / "input" / "data_blacksea"
BASEMAP = DATA_BLACKSEA / "extras" / "background_countries.geojson"
DIGITAL_DATA = ROOT / "internal_analysis" / "digital_connectivity" / "data"

BBOX = (26.0, 36.0, 53.0, 48.5)  # lon_min, lat_min, lon_max, lat_max

# Status has a single reading everywhere: blue is in service, teal is being
# built, mustard is announced.
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
    "fibre": "#0277bd",
    "cable": "#00838f",
    "power": "#256081",
    "pipe": "#b0bec5",
    "rail": "#90a4ae",
}

# The four countries the study optimises. Neighbours are drawn, not shaded.
FOCUS = {"Georgia", "Armenia", "Azerbaijan", "Turkey"}
# shaded on every basemap: the four study countries and the EU shore of the Black Sea
HIGHLIGHT = FOCUS | {"Romania", "Bulgaria"}
DISPLAY = {"Turkey": "Turkiye"}


def load_json(path: Path):
    with open(path, encoding="utf8") as fh:
        return json.load(fh)


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

    def basemap(self, index: CountryIndex, highlight=HIGHLIGHT):
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
