"""Fetch model data at the two ESMAP measurement sites, for validate_esmap.py.

  MLE_solar         PVGIS-ERA5 hourly horizontal beam/diffuse (components=1), 2015-2018
  Thulusdhoo_lidar  ERA5 hourly 100 m wind (Open-Meteo), 2017-2019
  both              Global Solar Atlas / Global Wind Atlas long-term values (raw/atlas_validation.json)

Run:  python fetch_validation.py
"""
import json
import time
import urllib.request
from pathlib import Path

import rasterio

from fetch_pvgis import BASE, SYSTEMS, fetch
from sites import VALIDATION

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
ERA5 = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2017-01-01"
        "&end_date=2019-12-31&hourly=wind_speed_10m,wind_speed_100m,wind_direction_100m&wind_speed_unit=ms&models=era5&timezone=GMT")


def main():
    lat, lon, _ = VALIDATION["MLE_solar"]
    for y in range(2015, 2019):
        path = RAW / "pvgis" / "validation" / f"pvgis_MLE_solar_ghi_{y}_components.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(fetch(BASE.format(lat=lat, lon=lon, y=y) + SYSTEMS["ghi"] + "&components=1")))
            print("PVGIS", y)
            time.sleep(1)
    lat, lon, _ = VALIDATION["Thulusdhoo_lidar"]
    path = RAW / "era5" / "validation_era5_Thulusdhoo_lidar_2017_2019.json"
    if not path.exists():
        req = urllib.request.Request(ERA5.format(lat=lat, lon=lon), headers={"User-Agent": "Mozilla/5.0 (EPM pre-analysis)"})
        d = json.loads(urllib.request.urlopen(req, timeout=300).read())
        path.write_text(json.dumps(d))
        print("ERA5 Thulusdhoo grid", d["latitude"], d["longitude"])
    out = {}
    for name, (lat, lon, _) in VALIDATION.items():
        req = urllib.request.Request(f"https://api.globalsolaratlas.info/data/lta?loc={lat},{lon}", headers={"User-Agent": "Mozilla/5.0"})
        out[name] = {"gsa": json.loads(urllib.request.urlopen(req, timeout=120).read())["annual"]["data"]}
    for h in (50, 100):
        with rasterio.open(RAW / "gwa" / f"gwa_ws{h}.tif") as src:
            for name, v in zip(VALIDATION, src.sample([(lon, lat) for lat, lon, _ in VALIDATION.values()])):
                out[name][f"gwa_ws{h}"] = round(float(v[0]), 3)
    (RAW / "atlas_validation.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {**{x: v for x, v in d.items() if x != "gsa"}, "GSA_GHI": d["gsa"]["GHI"], "GSA_DIF": d["gsa"]["DIF"]} for k, d in out.items()}))


if __name__ == "__main__":
    main()
