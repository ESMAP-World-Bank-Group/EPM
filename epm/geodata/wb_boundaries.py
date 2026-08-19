"""Build the World Bank boundary artifact every EPM map and zone derives from.

Sources, all World Bank, all under CC BY 4.0:

    Admin 0     World Bank Official Boundaries, Data Catalog dataset 0038272
                https://datacatalog.worldbank.org/search/dataset/0038272
                wb_countries_admin0_10m.zip -> WB_countries_Admin0_10m.shp
    ADM1/ADM2   WB Global Administrative Divisions (WB_GAD_Medium_Resolution),
                layers 4 and 3
    Disputes    same service, layer 6 (WB_GAD_Disputes)
    Line styles same service, layer 0 (WB_GAD_ADM0_Bdys)

The Admin 0 file is Natural Earth derived but carries the Bank's own treatment
of disputed areas (Kashmir line of control, Western Sahara, Cyprus, Taiwan,
Somaliland, ...), and it deliberately leaves the areas the Bank attributes to no
country as holes. Those live in the disputes layer and are appended here with
STATUS 'non-determined', so the maps show land rather than sea there without
ever painting them as part of a country.

Why this module exists
----------------------
EPM used to cut its zones from Natural Earth (epm/resources/postprocess/
zones.geojson for countries, Natural Earth 10m admin1 for the sub-national
zones) while the data explorer had moved its basemap to the Bank's boundaries.
Two independent geometries meant visible offsets, and worse, a country polygon
that swallowed territory the Bank does not attribute. Everything now comes from
one artifact so that cannot happen again.

The artifact
------------
Written to <cache>/wb_boundaries_<YYYYMMDD>/, dated and never edited in place:

    adm0.geojson            countries + non-determined areas, full detail
    countries_10m.geojson   adm0, simplified; the layer every map draws and
                            every zone is cut from
    countries_110m.geojson  adm0, simplified harder, for world views
    boundaries_10m.geojson  the Bank's broken-border lines, traced on the
    boundaries_110m.geojson   polygons of the matching resolution
    adm1.geojson            first-level units, fitted to countries_10m
    adm2_subset.geojson     second-level units, only for the countries a model
                            needs below adm1 (see ADM2_ISO_DEFAULT)
    source.json             dataset ids, URLs, fetch date, licence, checksums

The one rule that makes the sub-national layers usable: **the outer contour of
a country always comes from the country layer, only the internal limits come
from adm1**. adm1 is a medium-resolution layer and would not follow the 10m
coastline, so it is clipped to its country and the leftover slivers are given
to the nearest unit -- which is what lets a dissolve of any set of adm1 units
tile its country exactly, with no hairline gaps along the border.

The country layer it is fitted to is countries_10m, not the full-detail adm0:
countries_10m is what the explorer draws, and a zone has to end exactly where
the basemap under it ends. Full-detail adm0 is kept for provenance and for
anything that needs the untouched source, but nothing should cut zones from it.

Usage
-----
    python -m epm.geodata.wb_boundaries                 # build today's artifact
    python -m epm.geodata.wb_boundaries --refresh       # re-download the sources
    python -m epm.geodata.wb_boundaries --verify        # rebuild, compare, write nothing
    python -m epm.geodata.wb_boundaries --adm2-iso PAK IND
    python -m epm.geodata.wb_boundaries --cache PATH --out PATH

Requires: geopandas, shapely, topojson, pandas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd
import topojson as tp
from shapely.geometry import LineString, MultiPolygon, Polygon, box, mapping
from shapely import set_precision
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid

# -- Sources ------------------------------------------------------------------

SOURCE_URL = (
    "https://datacatalogfiles.worldbank.org/ddh-published/0038272/DR0046659/"
    "wb_countries_admin0_10m.zip"
)
SOURCE_NAME = "World Bank Official Boundaries (Data Catalog dataset 0038272)"
SOURCE_LICENSE = "CC BY 4.0"
SHP_IN_ZIP = "WB_countries_Admin0_10m/WB_countries_Admin0_10m.shp"

GAD_SERVICE = ("https://geowb.worldbank.org/hosting/rest/services/Hosted/"
               "WB_GAD_Medium_Resolution/FeatureServer")
GAD_NAME = "World Bank Global Administrative Divisions (WB_GAD_Medium_Resolution)"

# layer id, output fields, cache file name. The two small layers keep the names
# the explorer already caches, so an existing cache is reused as it stands.
GAD_LAYERS = {
    "adm0_bdys": (0, ("style",), "wb_gad_adm0_bdys.geojson"),
    "adm2": (3, ("iso_a3", "wb_a3", "hasc_1", "hasc_2", "gaul_1", "gaul_2",
                 "nam_0", "nam_1", "nam_2"), "wb_gad_adm2_{iso}.geojson"),
    "adm1": (4, ("iso_a3", "wb_a3", "hasc_1", "gaul_1", "nam_0", "nam_1"),
             "wb_gad_adm1.geojson"),
    "disputes": (6, ("nam_0", "nam_0_alt"), "wb_gad_disputes.geojson"),
}
# The service caps a response at 2000 features; anything larger is paged.
GAD_PAGE = 2000

# Countries a model needs below adm1. Karachi is a district, not a province, and
# is its own control area (K-Electric), hence its own EPM zone.
ADM2_ISO_DEFAULT = ("PAK",)

# -- Codes and special cases --------------------------------------------------

# The WB file leaves ISO_A3 as -99 on a handful of features and uses World Bank
# codes rather than ISO for a few countries. Map WB_A3 -> ISO_A3 for those cases.
WB_A3_TO_ISO = {
    "ZAR": "COD",   # Congo, Dem. Rep.
    "ROM": "ROU",   # Romania
    "TMP": "TLS",   # Timor-Leste
    "KSV": "KOS",   # Kosovo -- KOS is the code used across the explorer regions
}

# Natural Earth, which the WB file derives from, leaves the two-letter code at
# -99 for a handful of entities. WB_A2 covers most of them; these it does not.
ISO_A3_TO_A2 = {
    "NOR": "NO",    # Norway
    "CCK": "CC",    # Cocos (Keeling) Islands
    "CXR": "CX",    # Christmas Island
}

# Metropolitan France + Corsica. The WB France polygon includes the overseas
# departments; every map built here shows metropolitan France only.
METRO_FRANCE_BBOX = (-5.5, 41.0, 10.0, 51.5)

# Antarctica is a non-determined area too, but it is out of scope for power
# system maps and on its own is larger than every other one put together.
UNDETERMINED_SKIP = {"Antarctica"}
# A disputed polygon that the Admin 0 layer already paints as part of a country
# only leaves a coastal or border sliver behind; keep the ones that are a real
# hole, by uncovered fraction and by absolute size (square degrees).
MIN_UNDETERMINED_FRACTION = 0.5
MIN_UNDETERMINED_AREA = 1e-3

# -- Drawing ------------------------------------------------------------------

# The Bank's own drawing instructions for national boundaries. Every segment
# carries a `style`: null where the line is solid, otherwise one of Dashed,
# Tightly Dashed or Dotted. Only the styles are used -- that layer is medium
# resolution and sits up to ~2 km off the Admin 0 edges, so drawing its own
# geometry would double every border. See build_boundaries().
WB_LINE_STYLES = ("Dashed", "Tightly Dashed", "Dotted")
UNDETERMINED_STYLE = "Tightly Dashed"
# Lateral tolerance, in degrees, for recognising one of our own border edges in
# a styled WB segment. Ends are squared off so a segment cannot bleed past its
# own tips onto the next border along.
STYLE_MATCH_TOL = 0.05
# How close an area's outline has to run to a country for that stretch to count
# as a land boundary rather than a shore. Testing against the country areas
# instead of their edges survives the vertex mismatch left by simplification.
# The two source layers are drawn at different resolutions, so a land boundary
# can leave a void up to ~0.08 deg wide between the area and its neighbour; a
# real shore is degrees away from the nearest other country, so 0.1 separates
# the two cleanly.
LAND_EDGE_TOL = 0.1
# Near a tripoint a styled segment still passes within the tolerance of the
# neighbouring border and tags a stub of it. A real dashed border contributes
# degrees of line; those strays contribute hundredths, so drop the small ones.
MIN_STYLED_BORDER = 0.25

# Simplification tolerances in degrees. 0.001 deg is ~110 m, well below the
# positional accuracy of a 1:10m source, so the detail layer keeps the WB
# geometry intact for practical purposes.
TOL_10M = 0.001
TOL_110M = 0.08
PREC_10M = 4    # ~11 m
PREC_110M = 3   # ~110 m
PREC_ADM = 5    # ~1 m, for the untouched adm0; the fitted sub-national layers
                # are written at PREC_10M, the precision of the layer they follow

ADM0_PROPS = ("ISO_A3", "WB_A3", "WB_NAME", "WB_REGION", "STATUS")
# adm0 additionally carries the two-letter code: nothing on the map needs it,
# but the EPM reference layers do, and this is where they get it from.
ADM0_FULL_PROPS = ("ISO_A3", "ISO_A2", "WB_A3", "WB_NAME", "WB_REGION", "STATUS")
ADM1_PROPS = ("ISO_A3", "WB_A3", "HASC_1", "GAUL_1", "NAM_0", "NAM_1")
ADM2_PROPS = ("ISO_A3", "WB_A3", "HASC_1", "HASC_2", "GAUL_1", "GAUL_2",
              "NAM_0", "NAM_1", "NAM_2")


# -- Cache and artifact layout ------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]     # <repo>/epm/geodata/ -> <repo>


def resolve_cache(explicit=None):
    """Where the downloaded sources and the built artifacts live.

    Outside the repository: these are large files with their own lifecycle, and
    several clones of EPM on one machine should share a single copy. In order:
    an explicit path, $EPM_GEODATA_CACHE, the first `maps/` directory found
    above the repo, and failing that an ignored directory inside it.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("EPM_GEODATA_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    for parent in _REPO.parents:
        candidate = parent / "maps"
        if candidate.is_dir():
            return candidate
    return _REPO / ".geodata_cache"


def artifact_name(day=None):
    return f"wb_boundaries_{(day or date.today()).strftime('%Y%m%d')}"


# -- Fetching -----------------------------------------------------------------

def fetch_admin0(cache, explicit=None, refresh=False):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"source not found: {p}")
        return p
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / "wb_countries_admin0_10m.zip"
    if refresh or not local.exists():
        print(f"  downloading {SOURCE_URL}")
        tmp = local.with_suffix(".zip.part")
        with urllib.request.urlopen(SOURCE_URL, timeout=300) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(local)
    print(f"  admin0 source: {local}")
    return local


def fetch_gad(cache, key, where="1=1", iso=None, refresh=False):
    """Download one GAD layer as GeoJSON, paging past the service's cap.

    The response carries no clue that it was truncated, so the feature count is
    checked against the layer's own count and a short read is an error rather
    than a quietly incomplete layer.
    """
    layer, fields, pattern = GAD_LAYERS[key]
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / pattern.format(iso=(iso or "").lower())
    if local.exists() and not refresh:
        return local

    expected = _gad_count(layer, where)
    print(f"  downloading GAD layer {layer} ({key}): {expected} features")
    features = []
    while len(features) < expected:
        page = _gad_page(layer, fields, where, offset=len(features))
        if not page:
            break
        features.extend(page)
        if len(features) < expected:
            print(f"    {len(features)}/{expected}")
    if len(features) != expected:
        raise SystemExit(f"GAD layer {layer}: got {len(features)} of {expected} features")

    tmp = local.with_suffix(".geojson.part")
    tmp.write_text(json.dumps({"type": "FeatureCollection", "features": features},
                              separators=(",", ":"), ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(local)
    return local


def _gad_get(layer, params):
    url = f"{GAD_SERVICE}/{layer}/query?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))


def _gad_count(layer, where):
    return _gad_get(layer, {"where": where, "returnCountOnly": "true", "f": "json"})["count"]


def _gad_page(layer, fields, where, offset):
    doc = _gad_get(layer, {
        "where": where,
        "outFields": ",".join(fields),
        "returnGeometry": "true",
        "outSR": "4326",
        "orderByFields": "objectid",
        "resultOffset": offset,
        "resultRecordCount": GAD_PAGE,
        "f": "geojson",
    })
    if "error" in doc:
        raise SystemExit(f"GAD layer {layer}: {doc['error']}")
    return doc.get("features", [])


# -- Geometry helpers ---------------------------------------------------------

def polygons_only(geom):
    """Repair a geometry and keep only its polygonal parts."""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    parts, stack = [], [geom]
    while stack:
        g = stack.pop()
        if g.is_empty:
            continue
        if isinstance(g, Polygon):
            parts.append(g)
        elif hasattr(g, "geoms"):
            stack.extend(g.geoms)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def lines_only(geom):
    """Flatten a geometry to its non-empty LineString parts."""
    out, stack = [], [geom]
    while stack:
        part = stack.pop()
        if part is None or part.is_empty:
            continue
        if isinstance(part, LineString):
            if part.length > 0:
                out.append(part)
        elif hasattr(part, "geoms"):
            stack.extend(part.geoms)
    return out


def round_coords(obj, precision):
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, (list, tuple)):
        return [round_coords(x, precision) for x in obj]
    return obj


# -- Admin 0 ------------------------------------------------------------------

def read_admin0(path):
    if path.suffix.lower() == ".zip":
        return gpd.read_file(f"zip://{path.as_posix()}!{SHP_IN_ZIP}")
    return gpd.read_file(path)


def iso_code(row):
    for field in ("ISO_A3", "ISO_A3_EH"):
        v = (row.get(field) or "").strip()
        if v and v != "-99":
            return v
    wb = (row.get("WB_A3") or "").strip()
    return WB_A3_TO_ISO.get(wb, wb)


def clip_france(gdf):
    clip = box(*METRO_FRANCE_BBOX)
    mask = gdf.ISO_A3 == "FRA"
    if not mask.any():
        return gdf
    gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].apply(
        lambda g: polygons_only(g.intersection(clip))
    )
    print("  France clipped to metropolitan France + Corsica")
    return gdf


def build_countries(src):
    gdf = read_admin0(src)
    gdf["ISO_A3"] = gdf.apply(iso_code, axis=1)

    unknown = gdf[gdf.ISO_A3.isin(("", "-99"))]
    if len(unknown):
        raise SystemExit(f"unresolved country codes: {sorted(unknown.WB_NAME)}")

    # Several dependencies share a country code (Guantanamo Bay -> USA,
    # Clipperton -> FRA, the US minor outlying islands -> UMI). Merge them so
    # every code maps to exactly one feature, which is what every join assumes.
    with warnings.catch_warnings():
        # degree-based area, only used to rank parts and pick a representative name
        warnings.simplefilter("ignore")
        gdf["_area"] = gdf.geometry.area
    main = (gdf.sort_values("_area", ascending=False)
                .drop_duplicates("ISO_A3")
                .set_index("ISO_A3"))
    out = gdf.dissolve(by="ISO_A3", as_index=False)
    for col in ("ISO_A2", "WB_A2", "WB_A3", "WB_NAME", "WB_REGION"):
        out[col] = out.ISO_A3.map(main[col])

    def two_letter(iso_a2, wb_a2, iso_a3):
        for code in (iso_a2, wb_a2):
            code = (code or "").strip() if isinstance(code, str) else ""
            if code and code != "-99":
                return code
        return ISO_A3_TO_A2.get(iso_a3, "")

    out["ISO_A2"] = [two_letter(*v) for v in
                     zip(out.ISO_A2, out.WB_A2, out.ISO_A3)]
    out["STATUS"] = ""
    out = out[list(ADM0_FULL_PROPS) + ["geometry"]]
    out["geometry"] = out.geometry.apply(polygons_only)
    return out.sort_values("ISO_A3").reset_index(drop=True)


def build_undetermined(countries, cache, refresh=False):
    """The WB disputed areas, reduced to the holes the Admin 0 layer leaves.

    Most disputed polygons sit on top of a country the Bank does attribute
    (Arunachal Pradesh is drawn inside India in some products, the Kurils inside
    Russia, ...). Subtracting the country union keeps only what is genuinely
    unpainted, so these features never overpaint a country, they just fill in.
    """
    gdf = gpd.read_file(fetch_gad(cache, "disputes", refresh=refresh))
    gdf["WB_NAME"] = gdf.nam_0_alt.fillna(gdf.nam_0).str.strip()
    gdf = gdf[~gdf.WB_NAME.isin(UNDETERMINED_SKIP)]

    covered = unary_union([g.buffer(0) for g in countries.geometry if g is not None])
    geoms, kept = [], []
    with warnings.catch_warnings():
        # degree-based area, only ever compared against another degree-based area
        warnings.simplefilter("ignore")
        for g in gdf.geometry:
            g = polygons_only(g)
            hole = None if g is None else polygons_only(g.difference(covered))
            geoms.append(hole)
            kept.append(hole is not None
                        and hole.area >= MIN_UNDETERMINED_AREA
                        and hole.area >= MIN_UNDETERMINED_FRACTION * g.area)
    gdf["geometry"] = geoms
    gdf = gdf[kept]

    out = gdf.dissolve(by="WB_NAME", as_index=False)[["WB_NAME", "geometry"]]
    out["ISO_A3"] = ""
    out["ISO_A2"] = ""
    out["WB_A3"] = ""
    out["WB_REGION"] = ""
    out["STATUS"] = "non-determined"
    out = out.sort_values("WB_NAME").reset_index(drop=True)
    print(f"  {len(out)} non-determined areas: {', '.join(out.WB_NAME)}")
    return out[list(ADM0_FULL_PROPS) + ["geometry"]]


# -- Sub-national layers ------------------------------------------------------

def _fit_to_country(units, country_geom):
    """Cut a country's sub-national units to its adm0 outline, and tile it.

    Two edits, in this order. The units are clipped, so nothing sticks out past
    the country outline or into territory the Bank does not attribute to this
    country. Then whatever the units fail to cover -- the strips the medium
    resolution outline leaves along the coast and the border -- is handed to the
    nearest unit, so the units tile the country exactly. Without the second
    step, any zone built by dissolving a subset of units would show hairline
    holes against the basemap drawn from the same country layer.
    """
    clipped = units.copy()
    clipped["geometry"] = [polygons_only(g.intersection(country_geom))
                           for g in clipped.geometry]
    clipped = clipped[clipped.geometry.notna()].reset_index(drop=True)
    if clipped.empty:
        return clipped

    covered = unary_union([g.buffer(0) for g in clipped.geometry])
    residual = polygons_only(country_geom.difference(covered))
    if residual is None:
        return clipped

    parts = gpd.GeoDataFrame(
        geometry=list(getattr(residual, "geoms", [residual])), crs=clipped.crs)
    # nearest unit by distance: a sliver is by construction adjacent to the
    # unit it was cut from, so this reproduces the source assignment
    with warnings.catch_warnings():
        # nearest in degrees rather than metres, which the projection warning is
        # about: harmless here, since every sliver touches the unit it came from
        # and so is at distance zero from it whatever the units of measure.
        warnings.simplefilter("ignore")
        joined = gpd.sjoin_nearest(parts, clipped[["geometry"]], how="left")
    extra = defaultdict(list)
    for part, idx in zip(joined.geometry, joined.index_right):
        if pd.notna(idx):
            extra[int(idx)].append(part)
    for idx, pieces in extra.items():
        clipped.at[idx, "geometry"] = polygons_only(
            unary_union([clipped.at[idx, "geometry"], *pieces]))

    # Absorbing a sliver can leave a ring touching itself at the point where it
    # was joined. Every overlay downstream -- and dissolving units into zones is
    # nothing but overlays -- fails outright on that, so repair here rather than
    # leave it in the artifact.
    broken = ~clipped.geometry.is_valid
    if broken.any():
        clipped.loc[broken, "geometry"] = [
            polygons_only(make_valid(g)) for g in clipped.loc[broken, "geometry"]]
    return clipped


def _build_subnational(raw_path, countries, props, level):
    """Shared body of build_adm1/build_adm2: normalise codes, then fit."""
    gdf = gpd.read_file(raw_path)
    gdf.columns = [c.upper() if c != "geometry" else c for c in gdf.columns]
    def normalise(value, wb_a3):
        # the sub-national layers carry the Bank's own code for a few places
        # (KSV for Kosovo); the country outline is keyed on the ISO one, and a
        # unit that does not match its country's key would be dropped as
        # unattributed.
        code = (value or "").strip()
        if not code or code == "-99":
            code = (wb_a3 or "").strip()
        return WB_A3_TO_ISO.get(code, code)

    gdf["ISO_A3"] = [normalise(v, w)
                     for v, w in zip(gdf.get("ISO_A3", ""), gdf.get("WB_A3", ""))]
    known = set(countries.ISO_A3)
    dropped = sorted(set(gdf.ISO_A3) - known)
    if dropped:
        # units of areas the Bank attributes to no country: they have no adm0
        # outline to be fitted to, and no zone may be built from them
        print(f"    {level}: dropped {len(dropped)} unattributed code(s): "
              f"{', '.join(c or '(blank)' for c in dropped)}")
    gdf = gdf[gdf.ISO_A3.isin(known)]

    outlines = countries.set_index("ISO_A3").geometry
    out = []
    for iso, units in gdf.groupby("ISO_A3", sort=True):
        fitted = _fit_to_country(units.reset_index(drop=True), outlines[iso])
        if not fitted.empty:
            out.append(fitted)
    merged = gpd.GeoDataFrame(pd.concat(out, ignore_index=True), crs=gdf.crs)
    for col in props:
        if col not in merged.columns:
            merged[col] = ""
        merged[col] = merged[col].fillna("")
    merged = merged[list(props) + ["geometry"]]
    print(f"  {level}: {len(merged)} units in {merged.ISO_A3.nunique()} countries")
    return merged.sort_values(list(props)).reset_index(drop=True)


def build_adm1(countries, cache, refresh=False):
    return _build_subnational(fetch_gad(cache, "adm1", refresh=refresh),
                              countries, ADM1_PROPS, "adm1")


def build_adm2(countries, cache, isos, refresh=False):
    if not isos:
        return None
    where = "iso_a3 IN ({})".format(", ".join(f"'{i}'" for i in isos))
    path = fetch_gad(cache, "adm2", where=where, iso="_".join(isos), refresh=refresh)
    return _build_subnational(path, countries, ADM2_PROPS, "adm2")


# -- Broken borders -----------------------------------------------------------

def _style_lookup(styles):
    """Split one of our edges into the styled stretches a WB segment covers."""
    tree = STRtree(list(styles.geometry))

    def lookup(edge):
        matched, rest = [], edge
        for k in tree.query(edge.buffer(STYLE_MATCH_TOL)):
            band = styles.geometry[k].buffer(STYLE_MATCH_TOL, cap_style=2)
            for piece in lines_only(rest.intersection(band)):
                matched.append((styles["style"][k], piece))
            rest = unary_union(lines_only(rest.difference(band)))
            if rest.is_empty:
                break
        return matched, lines_only(rest)

    return lookup


def build_boundaries(layer, cache, refresh=False):
    """The Bank's broken borders, redrawn on our own edges.

    Two things end up in this layer. The outline of every non-determined area,
    split so that the land boundary is dashed and the coastline stays solid
    like any other shore. And the national borders the Bank itself draws
    broken -- the Line of Control, the line of actual control, the Korean
    DMZ -- matched onto our geometry by position, so that the styled line and
    the solid one underneath can never disagree by a pixel.
    """
    styles = gpd.read_file(fetch_gad(cache, "adm0_bdys", refresh=refresh))
    styles = styles[styles["style"].isin(WB_LINE_STYLES)].reset_index(drop=True)
    lookup = _style_lookup(styles)

    nd = layer[layer.STATUS == "non-determined"]
    countries = layer[layer.STATUS != "non-determined"].reset_index(drop=True)
    cgeoms, cnames = list(countries.geometry), list(countries.WB_NAME)
    tree = STRtree(cgeoms)
    rows = []

    for area in nd.itertuples():
        outline = area.geometry.boundary
        neighbours = [cgeoms[k].buffer(0)
                      for k in tree.query(area.geometry.buffer(LAND_EDGE_TOL))]
        band = unary_union(neighbours).buffer(LAND_EDGE_TOL)
        for piece in lines_only(outline.difference(band)):
            rows.append((area.WB_NAME, "", piece))
        for edge in lines_only(outline.intersection(band)):
            matched, plain = lookup(edge)
            rows.extend((area.WB_NAME, st, piece) for st, piece in matched)
            rows.extend((area.WB_NAME, UNDETERMINED_STYLE, piece) for piece in plain)

    tagged = defaultdict(list)
    for i, gi in enumerate(cgeoms):
        boundary = gi.boundary
        for j in tree.query(gi):
            if j <= i:
                continue
            for edge in lines_only(boundary.intersection(cgeoms[j].boundary)):
                for st, piece in lookup(edge)[0]:
                    tagged[(" / ".join(sorted((cnames[i], cnames[j]))), st)].append(piece)
    for (pair, st), pieces in sorted(tagged.items()):
        if sum(p.length for p in pieces) >= MIN_STYLED_BORDER:
            rows.extend((pair, st, p) for p in pieces)

    out = gpd.GeoDataFrame(rows, columns=["NAME", "STYLE", "geometry"],
                           geometry="geometry", crs=layer.crs)
    kept = defaultdict(float)
    for r in out.itertuples():
        kept[r.STYLE or "solid (coastline)"] += r.geometry.length
    print("    " + ", ".join(f"{k} {v:.1f} deg" for k, v in sorted(kept.items())))
    return out


def simplify(gdf, tolerance):
    """Topology-preserving simplification: shared borders stay shared, no slivers."""
    simplified = tp.Topology(gdf, prequantize=1e6,
                             shared_coords=False).toposimplify(tolerance).to_gdf()
    simplified["geometry"] = simplified.geometry.apply(polygons_only)
    # Very small states can be simplified out of existence; keep the source shape.
    lost = simplified.geometry.isna()
    if lost.any():
        names = list(simplified.loc[lost, "WB_NAME"])
        simplified.loc[lost, "geometry"] = gdf.loc[lost, "geometry"].values
        print(f"    kept unsimplified (too small): {', '.join(names)}")
    return simplified


# -- Writing ------------------------------------------------------------------

def _document(name, features):
    return {
        "type": "FeatureCollection",
        "name": name,
        "source": SOURCE_NAME,
        "license": SOURCE_LICENSE,
        "crs": {"type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }


def _write(doc, path):
    path.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False),
                    encoding="utf-8")
    print(f"  {path.name}: {len(doc['features'])} features, "
          f"{path.stat().st_size / 1e6:.1f} MB")


def write_geojson(gdf, path, precision, name, props=ADM0_PROPS, drop_empty=("STATUS",),
                  snap=False):
    """Write a polygon layer, dropping the properties that carry no value.

    snap moves the geometry onto the output grid before writing instead of
    letting the final rounding do it blindly. The two are the same operation
    to within a rounding step, except that snapping repairs the topology it
    breaks: rounding alone collapses neighbouring vertices onto each other and
    leaves rings crossing themselves. The simplified layers do not need it --
    simplification has already pulled their vertices apart -- and are written
    without it so their bytes do not move.
    """
    grid = 10 ** -precision
    features, emptied = [], 0
    for row in gdf.itertuples():
        if row.geometry is None:
            continue
        geometry = row.geometry
        if snap:
            geometry = polygons_only(make_valid(set_precision(geometry, grid)))
            if geometry is None:
                emptied += 1
                continue
        geom = mapping(geometry)
        attrs = {}
        for col in props:
            value = getattr(row, col, "")
            if col in drop_empty and not value:
                continue
            attrs[col] = value
        features.append({
            "type": "Feature",
            "properties": attrs,
            "geometry": {"type": geom["type"],
                         "coordinates": round_coords(geom["coordinates"], precision)},
        })
    if emptied:
        print(f"    {name}: {emptied} unit(s) smaller than the output grid, dropped")
    _write(_document(name, features), path)


def write_lines(gdf, path, precision, name):
    features = [{
        "type": "Feature",
        "properties": {"NAME": row.NAME, "STYLE": row.STYLE},
        "geometry": {"type": "LineString",
                     "coordinates": round_coords(list(row.geometry.coords), precision)},
    } for row in gdf.itertuples()]
    _write(_document(name, features), path)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_source(out_dir, adm2_iso):
    """The provenance record: what was used, when, and what came out.

    Read by the maps that have to cite their boundary source, and by
    tools/rebuild_geometry.py to tell whether a model's zones are still in step
    with the artifact they were cut from.
    """
    files = sorted(p for p in out_dir.iterdir() if p.name != "source.json")
    doc = {
        "artifact": out_dir.name,
        "built": date.today().isoformat(),
        "license": SOURCE_LICENSE,
        "sources": {
            "admin0": {"name": SOURCE_NAME, "url": SOURCE_URL,
                       "dataset": "0038272", "layer": SHP_IN_ZIP},
            "adm1": {"name": GAD_NAME, "url": f"{GAD_SERVICE}/4", "layer": "WB_GAD_ADM1"},
            "adm2": {"name": GAD_NAME, "url": f"{GAD_SERVICE}/3", "layer": "WB_GAD_ADM2",
                     "countries": list(adm2_iso)},
            "disputes": {"name": GAD_NAME, "url": f"{GAD_SERVICE}/6",
                         "layer": "WB_GAD_Disputes"},
            "boundary_styles": {"name": GAD_NAME, "url": f"{GAD_SERVICE}/0",
                                "layer": "WB_GAD_ADM0_Bdys"},
        },
        "outputs": {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)}
                    for p in files},
    }
    path = out_dir / "source.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"  {path.name}: {len(files)} outputs recorded")
    return doc


# -- Build --------------------------------------------------------------------

def build(out_dir, cache, source=None, adm2_iso=ADM2_ISO_DEFAULT,
          keep_france_overseas=False, refresh=False):
    """Build the whole artifact into out_dir. Returns the source.json document."""
    out_dir.mkdir(parents=True, exist_ok=True)

    countries = build_countries(fetch_admin0(cache, source, refresh))
    if not keep_france_overseas:
        countries = clip_france(countries)
    print(f"  {len(countries)} country features")

    # Simplified together with the countries so the shared edges stay shared.
    layer = pd.concat([countries, build_undetermined(countries, cache, refresh)],
                      ignore_index=True)
    layer = gpd.GeoDataFrame(layer, geometry="geometry", crs=countries.crs)

    write_geojson(layer, out_dir / "adm0.geojson", PREC_ADM, "adm0",
                  props=ADM0_FULL_PROPS, drop_empty=("STATUS", "ISO_A2"))

    # The broken borders are traced on the simplified polygons of each
    # resolution, so that every dash sits exactly on the edge it belongs to.
    detail = None
    for label, tol, prec, suffix in (("detail", TOL_10M, PREC_10M, "10m"),
                                     ("world", TOL_110M, PREC_110M, "110m")):
        print(f"  simplifying {label} layer")
        simplified = simplify(layer, tol)
        write_geojson(simplified, out_dir / f"countries_{suffix}.geojson",
                      prec, f"countries_{suffix}")
        print(f"  tracing {label} boundary styles")
        write_lines(build_boundaries(simplified, cache, refresh),
                    out_dir / f"boundaries_{suffix}.geojson", prec,
                    f"boundaries_{suffix}")
        if suffix == "10m":
            detail = simplified

    # Fitted to the 10m countries, the layer zones are cut from -- see the
    # module docstring. The non-determined areas are left out: no zone may be
    # built from territory the Bank attributes to no country.
    print("  fitting sub-national units to the 10m country outlines")
    reference = detail[detail.STATUS != "non-determined"].reset_index(drop=True)
    write_geojson(build_adm1(reference, cache, refresh), out_dir / "adm1.geojson",
                  PREC_10M, "adm1", props=ADM1_PROPS, drop_empty=(), snap=True)
    adm2 = build_adm2(reference, cache, adm2_iso, refresh)
    if adm2 is not None:
        write_geojson(adm2, out_dir / "adm2_subset.geojson", PREC_10M, "adm2_subset",
                      props=ADM2_PROPS, drop_empty=(), snap=True)

    return write_source(out_dir, adm2_iso)


def verify(out_dir, cache, **kw):
    """Rebuild into a temporary directory and compare with an existing artifact.

    The recipe test: from one cache, the build has to be reproducible. Only the
    build date may differ, so the checksums are compared, not source.json.
    """
    if not (out_dir / "source.json").exists():
        raise SystemExit(f"nothing to verify against: {out_dir}/source.json not found")
    reference = json.loads((out_dir / "source.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        fresh = build(Path(tmp), cache, **kw)
    ref, new = reference["outputs"], fresh["outputs"]
    problems = []
    for name in sorted(set(ref) | set(new)):
        if name not in new:
            problems.append(f"  missing now: {name}")
        elif name not in ref:
            problems.append(f"  new output:  {name}")
        elif ref[name]["sha256"] != new[name]["sha256"]:
            problems.append(f"  differs:     {name}")
    if problems:
        print(f"NOT reproducible ({out_dir.name}):")
        print("\n".join(problems))
        return False
    print(f"reproducible: {len(ref)} outputs identical to {out_dir.name}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="local wb_countries_admin0_10m.zip or .shp")
    ap.add_argument("--cache", help="where sources and artifacts live "
                                    "(default: $EPM_GEODATA_CACHE, else ../maps)")
    ap.add_argument("--out", help="artifact directory "
                                  "(default: <cache>/wb_boundaries_<date>)")
    ap.add_argument("--adm2-iso", nargs="*", default=list(ADM2_ISO_DEFAULT),
                    metavar="ISO3", help="countries to also fetch adm2 for")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download the sources instead of using the cache")
    ap.add_argument("--keep-france-overseas", action="store_true",
                    help="keep the French overseas departments in the FRA polygon")
    ap.add_argument("--verify", action="store_true",
                    help="rebuild and compare with the artifact, writing nothing")
    args = ap.parse_args()

    cache = resolve_cache(args.cache)
    out_dir = Path(args.out).resolve() if args.out else cache / artifact_name()
    print(f"cache:    {cache}")
    print(f"artifact: {out_dir}")

    kw = dict(source=args.source, adm2_iso=tuple(args.adm2_iso),
              keep_france_overseas=args.keep_france_overseas, refresh=args.refresh)
    if args.verify:
        sys.exit(0 if verify(out_dir, cache, **kw) else 1)
    build(out_dir, cache, **kw)
    print("Done.")


if __name__ == "__main__":
    main()
