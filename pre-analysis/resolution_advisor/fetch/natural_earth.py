"""
Load Natural Earth data: country boundaries and populated places.

Tries the local dataset/maps/ folder first (already present in the repo),
then falls back to downloading from Natural Earth.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import sys, os

# Fix pyproj PROJ database path before geopandas loads CRS context
sys.path.insert(0, str(Path(__file__).parent))
import _proj_fix  # noqa: F401

import pandas as pd

# Paths relative to pre-analysis/dataset/maps/ (existing repo structure)
_REPO_DATASET = Path(__file__).resolve().parents[3] / "dataset" / "maps"
_CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "natural_earth"

_BOUNDARIES_OPTIONS = [
    _REPO_DATASET / "ne_10m_admin_0_countries" / "ne_10m_admin_0_countries.shp",
    _CACHE_DIR / "ne_10m_admin_0_countries.shp",
    _REPO_DATASET / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp",
    _CACHE_DIR / "ne_110m_admin_0_countries.shp",
]
_PLACES_OPTIONS = [
    _REPO_DATASET / "ne_110m_populated_places" / "ne_110m_populated_places.shp",
    _REPO_DATASET / "ne_10m_populated_places" / "ne_10m_populated_places.shp",
    _CACHE_DIR / "ne_110m_populated_places.shp",
]

# Natural Earth download URLs (zip) — 10m preferred, 110m as fallback
_NE_BOUNDARIES_URLS = [
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip",
    "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip",
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
    "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip",
]
_NE_PLACES_URLS = [
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_populated_places.zip",
    "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_populated_places.zip",
]
_NE_BOUNDARIES_URL = _NE_BOUNDARIES_URLS[0]
_NE_PLACES_URL = _NE_PLACES_URLS[0]


def load_boundaries(countries: List[str] | None = None):
    """
    Load country boundary polygons.
    countries: list of ISO_A3 codes (None = all)
    Returns GeoDataFrame with columns: ISO_A3, NAME, geometry
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas required: pip install geopandas")

    path = _find_or_download(_BOUNDARIES_OPTIONS, _NE_BOUNDARIES_URLS,
                             "ne_10m_admin_0_countries")
    gdf = gpd.read_file(path)

    # Normalise ISO column -- Natural Earth uses ISO_A3 or ADM0_A3
    iso_col = next(
        (c for c in ["ISO_A3", "ADM0_A3", "iso_a3"] if c in gdf.columns), None
    )
    name_col = next(
        (c for c in ["NAME", "NAME_LONG", "name"] if c in gdf.columns), None
    )
    if iso_col:
        gdf = gdf.rename(columns={iso_col: "ISO_A3"})
    if name_col:
        gdf = gdf.rename(columns={name_col: "NAME"})

    if countries:
        gdf = gdf[gdf["ISO_A3"].isin(countries)]

    return gdf[["ISO_A3", "NAME", "geometry"]].copy()


def load_cities(countries: List[str] | None = None,
                min_pop: int = 100_000) -> pd.DataFrame:
    """
    Load populated places.
    Returns DataFrame with columns: name, pop, lat, lon, iso_a3
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas required: pip install geopandas")

    path = _find_or_download(_PLACES_OPTIONS, _NE_PLACES_URLS,
                             "ne_110m_populated_places")
    gdf = gpd.read_file(path)

    # Normalise columns
    col_map = {}
    for src, dst in [
        ("NAME", "name"), ("name", "name"),
        ("POP_MAX", "pop"), ("pop_max", "pop"), ("POP_MIN", "pop"),
        ("ISO_A3", "iso_a3"), ("iso_a3", "iso_a3"), ("ADM0_A3", "iso_a3"),
    ]:
        if src in gdf.columns and dst not in col_map.values():
            col_map[src] = dst
    gdf = gdf.rename(columns=col_map)

    # Extract lat/lon from geometry if not present
    if "lat" not in gdf.columns:
        gdf["lat"] = gdf.geometry.y
        gdf["lon"] = gdf.geometry.x

    keep = [c for c in ["name", "pop", "lat", "lon", "iso_a3"] if c in gdf.columns]
    df = gdf[keep].copy()
    df["pop"] = pd.to_numeric(df.get("pop", 0), errors="coerce").fillna(0)

    if min_pop > 0:
        df = df[df["pop"] >= min_pop]
    if countries and "iso_a3" in df.columns:
        df = df[df["iso_a3"].isin(countries)]

    return df.reset_index(drop=True)


_ADMIN1_OPTIONS = [
    _REPO_DATASET / "ne_10m_admin_1_states_provinces" / "ne_10m_admin_1_states_provinces.shp",
    _CACHE_DIR / "ne_10m_admin_1_states_provinces.shp",
]
_NE_ADMIN1_URLS = [
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip",
    "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_1_states_provinces.zip",
]


def load_admin1(countries: List[str] | None = None):
    """
    Load admin-1 state/province polygons (Natural Earth 10m).
    Returns GeoDataFrame with columns: ISO_A3, name, geometry
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas required: pip install geopandas")

    path = _find_or_download(_ADMIN1_OPTIONS, _NE_ADMIN1_URLS,
                             "ne_10m_admin_1_states_provinces")
    gdf = gpd.read_file(path)

    # adm0_a3 is the most reliable country ISO column in this dataset
    iso_col = next((c for c in ["adm0_a3", "ADM0_A3", "iso_a3", "ISO_A3"] if c in gdf.columns), None)
    name_col = next((c for c in ["name", "NAME", "name_en"] if c in gdf.columns), None)
    if iso_col:
        gdf = gdf.rename(columns={iso_col: "ISO_A3"})
    if name_col and name_col != "name":
        gdf = gdf.rename(columns={name_col: "name"})

    if countries:
        gdf = gdf[gdf["ISO_A3"].isin(countries)]

    keep = [c for c in ["ISO_A3", "name", "geometry"] if c in gdf.columns]
    return gdf[keep].copy().reset_index(drop=True)


# -- download helper -----------------------------------------------------------

def _find_or_download(options: list, url_or_urls, name: str) -> Path:
    for p in options:
        if Path(p).exists():
            return Path(p)

    # Try multiple download URLs
    import io, zipfile, requests as req, warnings
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    urls = url_or_urls if isinstance(url_or_urls, list) else [url_or_urls]

    for url in urls:
        print(f"  [NaturalEarth] Downloading {name} from {url.split('/')[-1]}...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress SSL warnings
                resp = req.get(url, timeout=90, verify=False)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                # Only extract files matching the dataset name
                target_files = [n for n in z.namelist()
                                if name.split("_master")[0] in n or n.endswith(".shp")
                                or n.endswith(".dbf") or n.endswith(".prj") or n.endswith(".shx")]
                for fname in target_files:
                    z.extract(fname, _CACHE_DIR)
            shp = next(_CACHE_DIR.rglob(f"*{name}*.shp"), None) or next(_CACHE_DIR.rglob("*.shp"), None)
            if shp:
                print(f"  [NaturalEarth] Saved to {shp}")
                return shp
        except Exception as e:
            print(f"  [NaturalEarth] Failed ({url.split('/')[-1]}): {e}")

    raise FileNotFoundError(
        f"Could not find or download {name}.\n"
        f"Manual option: download from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/\n"
        f"and place the .shp files in: {_REPO_DATASET}"
    )
