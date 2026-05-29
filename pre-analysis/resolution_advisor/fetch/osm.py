"""
Fetch power infrastructure from OpenStreetMap via Overpass API.
Results cached locally (cache/osm/) to avoid repeated API calls.
"""
from __future__ import annotations
import json
import time
import hashlib
from pathlib import Path
from typing import Tuple

import requests

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "osm"
REQUEST_DELAY = 3  # seconds between Overpass requests (rate limit)

_HEADERS = {
    "User-Agent": "EPM-ResolutionAdvisor/1.0 (World Bank energy planning tool)",
    "Accept": "application/json",
    "Accept-Charset": "utf-8",
    "Content-Type": "application/x-www-form-urlencoded",
}


# -- cache helpers -------------------------------------------------------------

def _cache_path(query: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    qhash = hashlib.md5(query.encode()).hexdigest()
    return CACHE_DIR / f"{qhash}.json"


def _query_overpass(query: str, timeout: int = 120) -> dict:
    cached = _cache_path(query)
    if cached.exists():
        with open(cached, encoding="utf-8") as f:
            return json.load(f)

    time.sleep(REQUEST_DELAY)
    last_err = None
    for url in OVERPASS_URLS:
        try:
            resp = requests.post(
                url,
                data={"data": query},
                timeout=timeout,
                headers=_HEADERS,
                verify=False,
            )
            if resp.status_code == 406:
                # Some proxies reject POST with Content-Type -- try GET with encoded query
                import urllib.parse
                get_url = f"{url}?data={urllib.parse.quote(query)}"
                resp = requests.get(get_url, timeout=timeout, headers=_HEADERS, verify=False)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            last_err = e
            print(f"    [OSM] {url.split('/')[2]} failed: {e}")
            continue
    else:
        print(f"    [OSM] All Overpass endpoints failed. Last: {last_err}")
        return {"elements": []}

    with open(cached, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


# -- public functions ----------------------------------------------------------

def fetch_substations(bbox: Tuple[float, float, float, float]) -> list[dict]:
    """
    Fetch power substations within a bounding box.
    bbox: (south, west, north, east)
    Returns list of dicts: {lat, lon, voltage_kv, name, osm_id}
    """
    s, w, n, e = bbox
    query = f"""
[out:json][timeout:120];
(
  node["power"="substation"]({s},{w},{n},{e});
  way["power"="substation"]({s},{w},{n},{e});
  relation["power"="substation"]({s},{w},{n},{e});
);
out center;
"""
    data = _query_overpass(query)
    rows = []
    for el in data.get("elements", []):
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        tags = el.get("tags", {})
        voltage_kv = _parse_voltage_kv(tags.get("voltage", ""))
        rows.append({
            "osm_id": el["id"],
            "lat": float(lat),
            "lon": float(lon),
            "voltage_kv": voltage_kv,
            "name": tags.get("name", ""),
        })
    return rows


def fetch_hv_lines(bbox: Tuple[float, float, float, float],
                   min_voltage_kv: int = 100) -> list[dict]:
    """
    Fetch HV transmission lines within a bounding box.
    Returns list of dicts: {osm_id, voltage_kv, coords [(lon, lat), ...]}
    """
    s, w, n, e = bbox
    query = f"""
[out:json][timeout:180];
(
  way["power"="line"]["voltage"]({s},{w},{n},{e});
  way["power"="cable"]["voltage"]({s},{w},{n},{e});
);
out geom;
"""
    data = _query_overpass(query)
    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        voltage_kv = _parse_voltage_kv(tags.get("voltage", ""))
        if voltage_kv < min_voltage_kv:
            continue
        geom_nodes = el.get("geometry", [])
        coords = [
            (float(nd["lon"]), float(nd["lat"]))
            for nd in geom_nodes
            if "lon" in nd and "lat" in nd
        ]
        if len(coords) < 2:
            continue
        rows.append({
            "osm_id": el["id"],
            "voltage_kv": voltage_kv,
            "coords": coords,
            "name": tags.get("name", ""),
        })
    return rows


# -- helpers -------------------------------------------------------------------

def _parse_voltage_kv(voltage_str: str) -> int:
    """Parse OSM voltage tag (e.g. '400000', '220000;110000') -> kV int."""
    if not voltage_str:
        return 0
    try:
        raw = int(voltage_str.split(";")[0].replace(",", "").strip())
        return raw // 1000 if raw >= 1000 else raw
    except (ValueError, AttributeError):
        return 0


def fetch_generators(bbox: Tuple[float, float, float, float]) -> list[dict]:
    """
    Fetch power plants from OSM within a bounding box.
    Returns list of dicts: {lat, lon, fuel, capacity_mw, name, osm_id}
    """
    import re
    s, w, n, e = bbox
    query = f"""
[out:json][timeout:120];
(
  node["power"="plant"]({s},{w},{n},{e});
  way["power"="plant"]({s},{w},{n},{e});
  relation["power"="plant"]({s},{w},{n},{e});
);
out center;
"""
    data = _query_overpass(query)
    rows = []
    for el in data.get("elements", []):
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        tags = el.get("tags", {})
        rows.append({
            "osm_id":      el["id"],
            "lat":         float(lat),
            "lon":         float(lon),
            "fuel":        _parse_plant_fuel(tags),
            "capacity_mw": _parse_plant_capacity_mw(tags, re),
            "name":        tags.get("name", ""),
        })
    return rows


def _parse_plant_fuel(tags: dict) -> str:
    source = (tags.get("plant:source") or tags.get("generator:source") or
              tags.get("plant:method") or "").lower()
    for key, label in [
        ("solar", "Solar"), ("wind", "Wind"), ("hydro", "Hydro"),
        ("water", "Hydro"), ("nuclear", "Nuclear"), ("gas", "Gas"),
        ("coal", "Coal"), ("oil", "Oil"), ("biomass", "Biomass"),
        ("geothermal", "Geothermal"), ("storage", "Storage"),
    ]:
        if key in source:
            return label
    return "Other"


def _parse_plant_capacity_mw(tags: dict, re) -> float:
    raw = (tags.get("plant:output:electricity") or
           tags.get("generator:output:electricity") or "")
    if not raw:
        return 0.0
    m = re.match(r"([0-9]+(?:\.[0-9]+)?)\s*(GW|MW|kW)?", str(raw).strip(), re.IGNORECASE)
    if not m:
        return 0.0
    val  = float(m.group(1))
    unit = (m.group(2) or "MW").upper()
    return val * 1000 if unit == "GW" else val / 1000 if unit == "KW" else val


def clear_cache():
    """Remove all cached OSM responses (forces fresh API calls)."""
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
    print(f"[OSM] Cache cleared: {CACHE_DIR}")
