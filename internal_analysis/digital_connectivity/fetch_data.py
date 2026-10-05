"""Fetch open digital-infrastructure data for the Black Sea and South Caucasus screening.

All sources are public and require no credentials. Each output file is written to
./data/ and is re-downloadable by re-running this script.

Sources
    ITU BBmaps GeoServer (WFS)      terrestrial transmission links (fibre), with status
    submarinecablemap.com API v3    submarine cable routes and landing points
    PeeringDB API                   internet exchange points and interconnection facilities
    World Bank Indicators API       ICT indicators
    Overpass API (OpenStreetMap)    high-voltage power lines, gas pipelines (best effort)

Usage
    python fetch_data.py
"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

UA = {"User-Agent": "WB-BlackSea-digital-screening/1.0"}

# Screening window: Black Sea, South Caucasus and the Caspian crossing.
BBOX = (26.0, 36.0, 53.0, 48.5)  # lon_min, lat_min, lon_max, lat_max

COUNTRIES = ["GE", "AM", "AZ", "TR", "RO", "BG", "MD", "UA", "KZ", "GR", "RU", "IR"]
WDI_ISO3 = "GEO;ARM;AZE;TUR;ROU;BGR;MDA;UKR;KAZ"
WDI_INDICATORS = {
    "IT.NET.USER.ZS": "Internet users (% of population)",
    "IT.NET.BBND.P2": "Fixed broadband subscriptions (per 100 people)",
    "IT.CEL.SETS.P2": "Mobile cellular subscriptions (per 100 people)",
}


def get_json(url: str, timeout: int = 180):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def save(name: str, obj) -> None:
    path = DATA / name
    with open(path, "w", encoding="utf8") as fh:
        json.dump(obj, fh)
    size = path.stat().st_size / 1e6
    print(f"    -> {name} ({size:.1f} MB)")


def in_bbox(x: float, y: float) -> bool:
    return BBOX[0] <= x <= BBOX[2] and BBOX[1] <= y <= BBOX[3]


# --------------------------------------------------------------------------
# 1. ITU terrestrial fibre links
# --------------------------------------------------------------------------
def fetch_itu_fibre() -> None:
    print("[1/5] ITU BBmaps terrestrial transmission links")
    url = (
        "https://bbmaps.itu.int/geoserver/ows?service=WFS&version=1.0.0"
        "&request=GetFeature&typeName=itu-geocatalogue:trx_geocatalogue"
        "&outputFormat=application/json&bbox=%s" % ",".join(str(v) for v in BBOX)
    )
    fc = get_json(url)
    print(f"    {len(fc['features'])} links in window")
    save("itu_fibre.geojson", fc)


# --------------------------------------------------------------------------
# 2. Submarine cables
# --------------------------------------------------------------------------
def fetch_submarine() -> None:
    print("[2/5] Submarine cable routes and landing points")
    cables = get_json("https://www.submarinecablemap.com/api/v3/cable/cable-geo.json")
    landings = get_json(
        "https://www.submarinecablemap.com/api/v3/landing-point/landing-point-geo.json"
    )

    keep = []
    for feat in cables["features"]:
        coords = feat["geometry"]["coordinates"]
        if any(in_bbox(x, y) for line in coords for x, y in line):
            keep.append(feat)
    cables["features"] = keep

    landings["features"] = [
        f for f in landings["features"] if in_bbox(*f["geometry"]["coordinates"])
    ]
    print(f"    {len(keep)} cables, {len(landings['features'])} landing points")
    save("submarine_cables.geojson", cables)
    save("submarine_landings.geojson", landings)


# --------------------------------------------------------------------------
# 3. PeeringDB
# --------------------------------------------------------------------------
def fetch_peeringdb() -> None:
    print("[3/5] PeeringDB internet exchanges and facilities")
    out = {"ix": {}, "fac": {}}
    for code in COUNTRIES:
        for kind in ("ix", "fac"):
            try:
                rows = get_json(f"https://www.peeringdb.com/api/{kind}?country={code}")["data"]
            except Exception as exc:  # noqa: BLE001
                print(f"    {code}/{kind}: {exc}")
                rows = []
            out[kind][code] = rows
            time.sleep(0.4)  # be polite with the public API
    n_ix = sum(len(v) for v in out["ix"].values())
    n_fac = sum(len(v) for v in out["fac"].values())
    print(f"    {n_ix} IXPs, {n_fac} facilities")
    save("peeringdb.json", out)


# --------------------------------------------------------------------------
# 4. World Bank ICT indicators
# --------------------------------------------------------------------------
def fetch_wdi() -> None:
    print("[4/5] World Bank ICT indicators")
    out = {}
    for code, label in WDI_INDICATORS.items():
        url = (
            f"https://api.worldbank.org/v2/country/{WDI_ISO3}/indicator/{code}"
            "?format=json&date=2015:2024&per_page=1000"
        )
        payload = get_json(url)
        rows = payload[1] if len(payload) > 1 and payload[1] else []
        latest = {}
        for row in rows:
            if row["value"] is None:
                continue
            country, year = row["country"]["value"], int(row["date"])
            if country not in latest or year > latest[country][0]:
                latest[country] = (year, row["value"])
        out[code] = {"label": label, "values": latest}
    save("wdi_ict.json", out)


# --------------------------------------------------------------------------
# 5. OpenStreetMap layers (best effort: Overpass is a shared public service)
# --------------------------------------------------------------------------
OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]


def overpass(query: str, timeout: int = 240):
    for mirror in OVERPASS_MIRRORS:
        try:
            data = urllib.parse.urlencode({"data": query}).encode()
            req = urllib.request.Request(mirror, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001
            print(f"    {mirror.split('/')[2]}: {exc}")
    return None


def ways_to_geojson(payload, props_keys) -> dict:
    features = []
    for el in payload.get("elements", []):
        geom = el.get("geometry")
        if not geom:
            continue
        tags = el.get("tags", {})
        features.append(
            {
                "type": "Feature",
                "properties": {k: tags.get(k) for k in props_keys},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[p["lon"], p["lat"]] for p in geom],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def fetch_osm() -> None:
    print("[5/5] OpenStreetMap high-voltage lines and gas pipelines (best effort)")

    # Voltages are tagged in volts. The alternatives cover the regional backbone
    # classes, including the Turkish 380 kV and the Russian 750 kV levels.
    hv_query = """[out:json][timeout:280];
way["power"="line"]["voltage"~"^(220|275|330|345|380|400|500|750)000"](%s,%s,%s,%s);
out geom;""" % (BBOX[1], BBOX[0], BBOX[3], BBOX[2])
    payload = overpass(hv_query)
    if payload:
        fc = ways_to_geojson(payload, ["voltage", "name", "operator"])
        print(f"    {len(fc['features'])} HV line segments")
        save("osm_hv_lines.geojson", fc)
    else:
        print("    HV lines unavailable, master map will fall back to the EPM topology")

    pipe_query = """[out:json][timeout:200];
way["man_made"="pipeline"]["substance"~"gas|natural_gas|oil",i](%s,%s,%s,%s);
out geom;""" % (BBOX[1], BBOX[0], BBOX[3], BBOX[2])
    payload = overpass(pipe_query)
    if payload:
        fc = ways_to_geojson(payload, ["name", "substance", "operator"])
        print(f"    {len(fc['features'])} pipeline segments")
        save("osm_pipelines.geojson", fc)
    else:
        print("    pipelines unavailable, the note describes them in text only")


def fetch_railways() -> None:
    """Main-line railways, plus the alignment the TRIPP corridor would reopen.

    Two queries. The regional one keeps main lines only, which is the Middle
    Corridor and the national trunk networks rather than every siding. The
    corridor one drops the usage filter over southern Armenia and the Aras
    valley, because the Soviet-era line through Meghri is mapped as disused or
    abandoned and carries no usage tag.
    """
    print("[6/6] OpenStreetMap railways")

    # Plain tag equality, not a regex: Overpass indexes key-value pairs, and a
    # regex over this window makes the query time out on every mirror.
    main_query = """[out:json][timeout:600];
way["railway"="rail"]["usage"="main"](%s,%s,%s,%s);
out geom;""" % (BBOX[1], BBOX[0], BBOX[3], BBOX[2])
    payload = overpass(main_query, timeout=700)
    features = []
    if payload:
        fc = ways_to_geojson(payload, ["railway", "name", "usage", "electrified"])
        for feat in fc["features"]:
            feat["properties"]["layer"] = "main"
        features += fc["features"]
        print(f"    {len(fc['features'])} main-line segments")

    # Yeraskh to Meghri to Horadiz, the closed alignment along the Aras.
    corridor_query = """[out:json][timeout:280];
way["railway"~"^(rail|disused|abandoned|construction)$"](38.6,44.4,40.2,47.7);
out geom;"""
    payload = overpass(corridor_query, timeout=400)
    if payload:
        fc = ways_to_geojson(payload, ["railway", "name", "usage", "electrified"])
        for feat in fc["features"]:
            feat["properties"]["layer"] = "corridor"
        features += fc["features"]
        print(f"    {len(fc['features'])} segments on the Aras alignment")

    if features:
        save("osm_railways.geojson", {"type": "FeatureCollection", "features": features})
    else:
        print("    railways unavailable, the transport map falls back to text")


def main() -> None:
    print(f"Writing to {DATA}\n")
    fetch_itu_fibre()
    fetch_submarine()
    fetch_peeringdb()
    fetch_wdi()
    fetch_osm()
    fetch_railways()
    print("\nDone.")


if __name__ == "__main__":
    main()
