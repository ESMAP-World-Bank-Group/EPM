"""Fetch Renewables.ninja (MERRA-2) wind and PV series used to check and build the Greater Malé profiles.

Wind, raw=true so each hour carries the hub-height wind speed next to the output:
  - 2023 at every zone point and ESMAP #24 for the two turbines below. These give (a) the ninja
    capacity factor to compare with ours and (b) the turbine power curve as ninja applies it
    (electricity vs wind speed), which build_profiles.py reuses on ERA5 wind.
  - Thulusdhoo 2017-2019 for the onshore turbine, to compare with the ESMAP lidar there once that
    file is downloaded (energydata.info blocks scripted downloads).
PV, 2023 at Malé: fixed (tilt 3, azim 180) and 2-axis, to compare with PVGIS.

Turbines (both in the ninja library; still to agree, VRE_LOG Q1):
  onshore   Gamesa G114 2000 (2.0 MW, 114 m rotor, IEC IIIA low-wind class), 100 m hub
  offshore  Vestas V136 3450 (3.45 MW, 136 m rotor, low-wind class), 100 m hub. Greater Malé "offshore"
            means fixed-bottom on the shallow reef/lagoon; GWA shows no wind gain offshore here.
ninja wants a browser-like User-Agent (Cloudflare 1010 otherwise), and allows ~50 requests per hour.

Run:  python fetch_ninja.py
"""
import configparser
import json
import os
import time
import urllib.request
from pathlib import Path

from sites import OFFSHORE, ZONES

HERE = Path(__file__).resolve().parent
OUT = HERE / "raw" / "ninja"
# [api_tokens] renewables_ninja = <token>; default EPM/pre-analysis/config/api_tokens.ini (not in git)
TOKEN_INI = Path(os.environ.get("NINJA_TOKEN_INI", HERE.parents[1] / "config" / "api_tokens.ini"))
TURBINES = {"onshore": "Gamesa+G114+2000", "offshore": "Vestas+V136+3450"}
HUB = 100
THULUSDHOO = (4.374, 73.651)  # approximate island centre; exact lidar position is in the ESMAP header file


def token():
    cp = configparser.ConfigParser()
    cp.read(TOKEN_INI)
    return cp.get("api_tokens", "renewables_ninja").strip()


def get(kind, q, tok):
    url = f"https://www.renewables.ninja/api/data/{kind}?{q}&dataset=merra2&capacity=1&raw=true&format=json"
    req = urllib.request.Request(url, headers={"Authorization": f"Token {tok}", "User-Agent": "Mozilla/5.0 (EPM pre-analysis)"})
    for attempt in range(4):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=180).read())
        except Exception as e:
            print(f"  {e}; waiting 90 s")
            time.sleep(90)
    raise SystemExit(f"failed {url}")


def save(path, kind, q, tok):
    if path.exists():
        return
    d = get(kind, q, tok)
    path.write_text(json.dumps(d))
    e = [v["electricity"] for v in d["data"].values()]
    print(f"{path.stem}: {len(e)} h, CF {sum(e) / len(e):.3f}")
    time.sleep(8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = token()
    for name, (lat, lon, _) in {**ZONES, **OFFSHORE}.items():
        for tech, turb in TURBINES.items():
            save(OUT / f"wind_{name}_{tech}_2023.json", "wind",
                 f"lat={lat}&lon={lon}&date_from=2023-01-01&date_to=2023-12-31&height={HUB}&turbine={turb}", tok)
    for y in (2017, 2018, 2019):
        save(OUT / f"wind_Thulusdhoo_onshore_{y}.json", "wind",
             f"lat={THULUSDHOO[0]}&lon={THULUSDHOO[1]}&date_from={y}-01-01&date_to={y}-12-31&height={HUB}&turbine={TURBINES['onshore']}", tok)
    lat, lon, _ = ZONES["Male"]
    for tr, extra in (("fixed", "tracking=0&tilt=3&azim=180"), ("2ax", "tracking=2&tilt=3&azim=180")):
        save(OUT / f"pv_Male_{tr}_2023.json", "pv",
             f"lat={lat}&lon={lon}&date_from=2023-01-01&date_to=2023-12-31&system_loss=0.14&{extra}", tok)


if __name__ == "__main__":
    main()
