# -*- coding: utf-8 -*-
"""Romanian demand forecast from the 2024 NECP update, WEM scenario.

Rewrites the two Romania rows (Peak, Energy) of

    epm/input/data_blacksea/load/pDemandForecast.csv
    epm/input/data_blacksea/cesi/pDemandForecast_cesi.csv

and leaves every other row byte for byte. Romania is internal only in the
RoBg family (zcmap_robg.csv), so no earlier scenario reads these rows.

Method
------
* Energy: NECP 2024 update, Figure 96 (page 230), WEM, LEAP-RO. The plan's
  text (page 229) reads the series as electricity consumption: "from 55.5 TWh
  in 2022 to 98.8 TWh by 2050". Points 2022 55.5, 2025 58.0, 2030 73.1,
  2035 83.6, 2040 88.6, 2045 93.9, 2050 98.8 TWh, linear in between;
  2051-2053 continue the 2045-2050 slope.
* Peak = Energy / (8760 x LF), LF = the pHours-weighted mean of the deployed
  Romanian profile. The profile is divided by the true 2023 annual peak, so
  its mean is the 2023 load factor (about 0.715). Forecast and profile then
  carry the same load factor, the rule used for the other zones.

Usage (from EPM/):  python pre-analysis/pipelines/build_demand_romania_necp.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_EPM = _HERE.parents[2]
DATA = _EPM / "epm" / "input" / "data_blacksea"
TARGETS = [DATA / "load" / "pDemandForecast.csv", DATA / "cesi" / "pDemandForecast_cesi.csv"]
PROFILE = DATA / "load" / "pDemandProfile.csv"
HOURS = DATA / "pHours.csv"

ZONE = "Romania"
NECP_WEM_TWH = {2022: 55.5, 2025: 58.0, 2030: 73.1, 2035: 83.6, 2040: 88.6, 2045: 93.9, 2050: 98.8}


def _fail(msg: str) -> None:
    raise SystemExit(f"[build_demand_romania_necp] FAIL: {msg}")


def profile_load_factor() -> float:
    prof = pd.read_csv(PROFILE)
    hrs = pd.read_csv(HOURS)
    cols = [c for c in prof.columns if c.startswith("t")]
    p = prof[prof["zone"] == ZONE].melt(id_vars=["zone", "season", "daytype"], value_vars=cols,
                                        var_name="t", value_name="v")
    h = hrs.melt(id_vars=["q", "d"], value_vars=cols, var_name="t", value_name="w")
    m = p.merge(h, left_on=["season", "daytype", "t"], right_on=["q", "d", "t"])
    if len(m) != len(p) or len(p) == 0:
        _fail("profile and pHours do not match")
    return float((m["v"] * m["w"]).sum() / m["w"].sum())


def energy_gwh(years: list[int]) -> dict[int, float]:
    xs = sorted(NECP_WEM_TWH)
    ys = [NECP_WEM_TWH[x] for x in xs]
    slope = (NECP_WEM_TWH[2050] - NECP_WEM_TWH[2045]) / 5
    out = {}
    for y in years:
        twh = np.interp(y, xs, ys) if y <= 2050 else NECP_WEM_TWH[2050] + slope * (y - 2050)
        out[y] = float(twh) * 1000
    return out


def main() -> int:
    lf = profile_load_factor()
    for path in TARGETS:
        raw = path.read_bytes()
        bom = raw.startswith(b"\xef\xbb\xbf")
        text = raw.decode("utf-8-sig")
        eol = "\r\n" if "\r\n" in text else "\n"
        lines = text.split(eol)
        years = [int(c) for c in lines[0].split(",")[2:]]
        energy = energy_gwh(years)
        rows = {"Energy": energy, "Peak": {y: e / (8.76 * lf) for y, e in energy.items()}}
        hits = 0
        for i, ln in enumerate(lines):
            parts = ln.split(",")
            if parts[0] == ZONE and parts[1] in rows:
                lines[i] = ",".join([ZONE, parts[1]] + [f"{rows[parts[1]][y]:.2f}" for y in years])
                hits += 1
        if hits != 2:
            _fail(f"{path.name}: expected 2 Romania rows, found {hits}")
        out = eol.join(lines).encode("utf-8")
        path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + out)
        print(f"  wrote {path.relative_to(_EPM)}")
    print(f"  profile load factor {lf:.4f}")
    for y in (2024, 2025, 2030, 2040, 2050, 2053):
        print(f"  {y}: Energy {energy[y]:9.1f} GWh  Peak {energy[y] / (8.76 * lf):8.1f} MW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
