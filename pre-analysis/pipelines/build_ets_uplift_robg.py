# -*- coding: utf-8 -*-
"""EU ETS cost on Romanian and Bulgarian fossil fuels, RoBg family (step 13a).

Why a fuel price uplift and not pCarbonPrice
--------------------------------------------
pCarbonPrice(y) carries no zone index (base.gms:691): switching it on would tax
Türkiye, Georgia, Armenia and Azerbaijan as well, none of which is in the EU ETS.
The ETS cost is therefore added to the fuel price of the two EU members only:

    uplift(c, f, y) [USD/MMBtu] = ETS(y) [USD/t CO2] * carbon(f) [t CO2/MMBtu] * (1 - capture(f))

which is exactly what pCarbonPrice would add per MMBtu burnt, restricted to
Romania and Bulgaria. Emissions reported by the model are unchanged; only the
system cost and the merit order see the price.

ETS trajectory
--------------
The same path that prices the EU border (eu_price_netback.build_ets_path): observed
2024 and 2026 in real EUR 2024, then the TYNDP 2024 points (EUR 2022, deflated with
the euro area HICP) interpolated linearly and flat past 2050. The EU border price L
comes from TYNDP runs at this CO2 price, so Romania and Bulgaria face the same carbon
cost as the market they trade with. Converted at the 2024 EUR/USD average.

Fuels
-----
Carbon content from epm/resources/pFuelCarbonContent.csv. GasCCS has no entry there
(the model books it at zero emissions); it pays the residual of a 90 % capture rate on
the gas content. Hydrogen, uranium, biomass and geothermal carry no uplift.

Outputs
-------
* supply/pFuelPrice_robg.csv: a copy of supply/pFuelPrice_tr_gas_flat.csv (the file
  config.csv deploys) with the Romania and Bulgaria fossil rows uplifted. Rebuilt from
  the base file on every run, so later edits of the base file carry over. The base
  file stays a pure fuel price file.
* scenarios.csv: the pFuelPrice cell of LC_Baseline_RoBg, LC_BSSC_RoBg and
  LC_GECO_RoBg points to it. Every other cell is left byte for byte.

Idempotent: a second run writes the same bytes.

Usage (from EPM/):  python pre-analysis/pipelines/build_ets_uplift_robg.py
"""
from __future__ import annotations

import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

_EPM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eu_price_netback import build_ets_path  # noqa: E402

DATA = _EPM / "epm" / "input" / "data_blacksea"
BASE = DATA / "supply" / "pFuelPrice_tr_gas_flat.csv"
DST = DATA / "supply" / "pFuelPrice_robg.csv"
SCEN = DATA / "scenarios.csv"
CARBON = _EPM / "epm" / "resources" / "pFuelCarbonContent.csv"
CFG = _EPM / "pre-analysis" / "config"

COUNTRIES = ["Romania", "Bulgaria"]
SCENARIOS = ["LC_Baseline_RoBg", "LC_BSSC_RoBg", "LC_GECO_RoBg"]
CELL = "supply/pFuelPrice_robg.csv"
# fuel absent from pFuelCarbonContent.csv: (fuel whose content applies, share emitted)
PROXY = {"GasCCS": ("Gas", 0.10), "Hydrogen": ("Gas", 0.0)}
BAK = f".bak_robg13_{date.today():%Y%m%d}"


def _fail(msg: str) -> None:
    raise SystemExit(f"[build_ets_uplift_robg] FAIL: {msg}")


def _read(path: Path) -> tuple[bytes, str, list[str]]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    eol = "\r\n" if "\r\n" in text else "\n"
    bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
    return bom, eol, text.split(eol)


def _write(path: Path, bom: bytes, eol: str, lines: list[str]) -> bool:
    new = bom + eol.join(lines).encode("utf-8")
    if path.exists() and path.read_bytes() == new:
        return False
    if path.exists():
        bak = path.with_name(path.name + BAK)
        if not bak.exists():
            shutil.copyfile(path, bak)
    path.write_bytes(new)
    return True


def ets_usd() -> pd.Series:
    cfg = pd.read_csv(CFG / "eu_price_netback.csv", comment="#")
    defl = pd.read_csv(CFG / "price_deflators.csv", comment="#")

    def one(series: str, year: int) -> float:
        m = defl[(defl.iloc[:, 0] == series) & (defl.iloc[:, 1] == year)]
        if m.empty:
            _fail(f"price_deflators.csv: no {series} {year}")
        return float(m.iloc[0, 2])

    eur = build_ets_path(cfg, one("eur_hicp", 2024), one("eur_hicp", 2022))
    return eur * one("eur_usd", 2024)


def carbon() -> dict[str, float]:
    df = pd.read_csv(CARBON)
    return {str(f).strip(): float(v) for f, v in zip(df.iloc[:, 0], df.iloc[:, 1])}


def fuel_prices(ets: pd.Series, cc: dict[str, float]) -> None:
    bom, eol, lines = _read(BASE)
    head = lines[0].split(",")
    years = [int(h) for h in head[2:]]
    missing = [y for y in years if y not in ets.index]
    if missing:
        _fail(f"no ETS value for {missing}")
    out, done = [], []
    for ln in lines:
        parts = ln.split(",")
        if len(parts) > 2 and parts[0] in COUNTRIES:
            f = parts[1]
            if f in PROXY:
                ref, share = PROXY[f]
                k = cc[ref] * share
            else:
                k = cc.get(f)
                if k is None:
                    _fail(f"{parts[0]} {f}: no carbon content and no proxy")
            if k > 0:
                vals = [round(float(v) + ets[y] * k, 4) for v, y in zip(parts[2:], years)]
                parts = parts[:2] + [f"{v:g}" for v in vals]
                done.append((parts[0], f, k))
        out.append(",".join(parts))
    changed = _write(DST, bom, eol, out)
    print(f"  {DST.relative_to(_EPM)}: {len(done)} rows uplifted"
          f"{'' if changed else ' (unchanged)'}")
    for c, f, k in done:
        print(f"    {c:9s} {f:13s} {k:.4f} t/MMBtu  +{ets[2025] * k:5.2f} (2025)"
              f"  +{ets[2030] * k:5.2f} (2030)  +{ets[2040] * k:5.2f} (2040) USD/MMBtu")


def scenarios() -> None:
    bom, eol, lines = _read(SCEN)
    head = lines[0].split(",")
    cols = [head.index(s) if s in head else _fail(f"scenarios.csv: no column {s}")
            for s in SCENARIOS]
    hit = [i for i, ln in enumerate(lines) if ln.split(",")[0] == "pFuelPrice"]
    if len(hit) != 1:
        _fail("scenarios.csv: expected one pFuelPrice row")
    row = lines[hit[0]].split(",")
    if len(row) != len(head):
        _fail("scenarios.csv: pFuelPrice row is ragged, edit by hand")
    for c in cols:
        if row[c] not in ("", CELL):
            _fail(f"scenarios.csv: pFuelPrice already set to {row[c]} in {head[c]}")
        row[c] = CELL
    lines[hit[0]] = ",".join(row)
    changed = _write(SCEN, bom, eol, lines)
    print(f"  scenarios.csv: pFuelPrice -> {CELL} in {', '.join(SCENARIOS)}"
          f"{'' if changed else ' (unchanged)'}")


def main() -> int:
    ets = ets_usd()
    print("  ETS path, USD 2024 per tonne: " + "  ".join(
        f"{y} {ets[y]:5.1f}" for y in (2024, 2025, 2030, 2035, 2040, 2050)))
    fuel_prices(ets, carbon())
    scenarios()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
