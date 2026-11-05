"""Fetch hourly PV output from PVGIS 5.3 (JRC) for each Greater Malé zone, 2014-2023.

PVGIS-SARAH3 (satellite, ~5 km) does not cover the Maldives (API: "Location out of the spatial
coverage"), so the radiation database is PVGIS-ERA5 (reanalysis, ~31 km). ERA5 ends in 2023 in PVGIS 5.3.

Systems, all 1 kWp, crystalline Si, 14% system loss (PVGIS default):
  fixed  trackingtype=0, optimal slope and azimuth (PVGIS picks about 3 deg, facing south)
  1ax    trackingtype=1, single horizontal axis aligned north-south (the usual utility tracker)
  2ax    trackingtype=2, two-axis
  ghi    trackingtype=0, slope 0  -> G(i) is global horizontal irradiance, used to compare with Global Solar Atlas

One request per zone, system and year (raw/pvgis/pvgis_{zone}_{system}_{year}.json), plus annual
beam/diffuse/reflected totals in raw/pvgis/components/.
P [W] / 1000 = capacity factor. PVGIS stamps each hour in UTC at the half hour (e.g. 20230101:0630);
the hour is kept and shifted to local time in build_profiles.py.

Run:  python fetch_pvgis.py
"""
import json
import time
import urllib.request
from pathlib import Path

from sites import ZONES

HERE = Path(__file__).resolve().parent
OUT = HERE / "raw" / "pvgis"
START, END = 2014, 2023
BASE = ("https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?lat={lat}&lon={lon}"
        "&startyear={y}&endyear={y}&pvcalculation=1&peakpower=1&loss=14"
        "&raddatabase=PVGIS-ERA5&outputformat=json")
SYSTEMS = {
    "fixed": "&trackingtype=0&optimalangles=1",
    "1ax": "&trackingtype=1",
    "2ax": "&trackingtype=2",
    "ghi": "&trackingtype=0&angle=0&aspect=0",
}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EPM pre-analysis)"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # PVGIS drops long requests now and then (WinError 10054)
            print(f"  retry after {e}")
            time.sleep(20)
    raise SystemExit(f"failed: {url}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for zone, (lat, lon, _) in ZONES.items():
        for sysname, q in SYSTEMS.items():
            for y in range(START, END + 1):  # one year per request: 10-year requests get dropped
                path = OUT / f"pvgis_{zone}_{sysname}_{y}.json"
                if path.exists():
                    continue
                d = fetch(BASE.format(lat=lat, lon=lon, y=y) + q)
                path.write_text(json.dumps(d))
                h = d["outputs"]["hourly"]
                print(f"{zone:11s} {sysname:5s} {y} {len(h)} h  mean P/1000 {sum(x['P'] for x in h) / len(h) / 1000:.4f}"
                      f"  mounting {list(d['inputs']['mounting_system'])[0]}")
                time.sleep(1)
    # Beam / diffuse / reflected split (components=1) for calibrating to Global Solar Atlas in build_profiles.py:
    # horizontal for 2015-2023, the three PV systems for 2023 only (the in-plane split is stable year to year).
    for zone, (lat, lon, _) in ZONES.items():
        for sysname, q in SYSTEMS.items():
            for y in (range(2015, END + 1) if sysname == "ghi" else [END]):
                path = OUT / "components" / f"pvgis_{zone}_{sysname}_{y}_components.json"
                if path.exists():
                    continue
                path.parent.mkdir(exist_ok=True)
                d = fetch(BASE.format(lat=lat, lon=lon, y=y) + q + "&components=1")
                h = d["outputs"]["hourly"]
                tot = {c: round(sum(x[c] for x in h) / 1000, 1) for c in ("Gb(i)", "Gd(i)", "Gr(i)", "P")}
                path.write_text(json.dumps(tot))  # annual kWh/m2 (P: kWh/kWp); the hourly detail is not needed
                print(f"{zone:11s} {sysname:5s} {y} components {tot}")
                time.sleep(1)


if __name__ == "__main__":
    main()
