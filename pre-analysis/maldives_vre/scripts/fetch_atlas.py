"""Long-term annual reference values from the World Bank / ESMAP atlases, at each sample point.

Global Solar Atlas (Solargis satellite model, ~250 m output; API api.globalsolaratlas.info/data/lta):
  GHI, DNI, DIF [kWh/m2/yr], GTI_opta, OPTA (optimal tilt), TEMP, PVOUT_csi [kWh/kWp/yr, fixed at optimal tilt].
Global Wind Atlas v3 (DTU, ERA5 downscaled with WAsP, 250 m): country GeoTIFF for MDV, mean wind speed
  at 50 m and 100 m, downloaded from globalwindatlas.info/api/gis/country/MDV/wind-speed/{50,100}
  into raw/gwa/ (the tifs use ZSTD compression, so they are read with rasterio).

Writes raw/atlas_points.json.

Run:  python fetch_atlas.py
"""
import json
import urllib.request
from pathlib import Path

import rasterio

from sites import OFFSHORE, ZONES

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
GSA = "https://api.globalsolaratlas.info/data/lta?loc={lat},{lon}"
GWA = "https://globalwindatlas.info/api/gis/country/MDV/wind-speed/{h}"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EPM pre-analysis)"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def main():
    pts = {**ZONES, **OFFSHORE}
    out = {}
    for h in (50, 100):
        tif = RAW / "gwa" / f"gwa_ws{h}.tif"
        if not tif.exists():
            tif.parent.mkdir(parents=True, exist_ok=True)
            tif.write_bytes(get(GWA.format(h=h)))
    for name, (lat, lon, _) in pts.items():
        gsa = json.loads(get(GSA.format(lat=lat, lon=lon)))["annual"]["data"]
        out[name] = {"lat": lat, "lon": lon, "gsa": gsa}
    for h in (50, 100):
        with rasterio.open(RAW / "gwa" / f"gwa_ws{h}.tif") as src:
            for name, v in zip(pts, src.sample([(lon, lat) for lat, lon, _ in pts.values()])):
                out[name][f"gwa_ws{h}"] = round(float(v[0]), 3)
    (RAW / "atlas_points.json").write_text(json.dumps(out, indent=1))
    for name, d in out.items():
        g = d["gsa"]
        print(f"{name:11s} GSA PVOUT {g['PVOUT_csi']:.0f} kWh/kWp (CF {g['PVOUT_csi'] / 8760:.3f}), GHI {g['GHI']:.0f}, "
              f"OPTA {g['OPTA']} deg | GWA 50 m {d['gwa_ws50']} m/s, 100 m {d['gwa_ws100']} m/s")


if __name__ == "__main__":
    main()
