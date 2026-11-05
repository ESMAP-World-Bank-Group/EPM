"""Fetch hourly ERA5 wind (10 m and 100 m) for each sample point, 2014-2025, via the Open-Meteo archive API.

ERA5 (ECMWF / Copernicus, 0.25 deg ~ 28 km) is the reanalysis Global Wind Atlas is built from, and it is
the only free hourly source here that reaches 2025 at 100 m. Open-Meteo serves ERA5 as is
(models=era5); data licence CC BY 4.0 (Copernicus). Times are UTC.

Run:  python fetch_era5_wind.py
"""
import json
import time
import urllib.request
from pathlib import Path

from sites import OFFSHORE, ZONES

HERE = Path(__file__).resolve().parent
OUT = HERE / "raw" / "era5"
START, END = 2014, 2025
URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
       f"&start_date={START}-01-01&end_date={END}-12-31"
       "&hourly=wind_speed_10m,wind_speed_100m,wind_direction_100m&wind_speed_unit=ms&models=era5&timezone=GMT")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (lat, lon, _) in {**ZONES, **OFFSHORE}.items():
        path = OUT / f"era5_{name}_{START}_{END}.json"
        if path.exists():
            continue
        req = urllib.request.Request(URL.format(lat=lat, lon=lon), headers={"User-Agent": "Mozilla/5.0 (EPM pre-analysis)"})
        d = json.loads(urllib.request.urlopen(req, timeout=600).read())
        path.write_text(json.dumps(d))
        ws = [v for v in d["hourly"]["wind_speed_100m"] if v is not None]
        print(f"{name:11s} grid {d['latitude']},{d['longitude']}  {len(ws)} h  mean 100 m {sum(ws) / len(ws):.2f} m/s")
        time.sleep(3)


if __name__ == "__main__":
    main()
