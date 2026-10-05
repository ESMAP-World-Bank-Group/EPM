# -*- coding: utf-8 -*-
"""Romanian and Bulgarian fuel prices for the RoBg family.

Rewrites the Romania and Bulgaria rows of

    epm/input/data_blacksea/supply/pFuelPrice_tr_gas_flat.csv   (deployed, config.csv)
    epm/input/data_blacksea/supply/pFuelPrice.csv               (reference copy)

and leaves every other row byte for byte. Romania and Bulgaria are internal
only in the RoBg family (zcmap_robg.csv), so no earlier scenario reads them.

Method
------
* Romania: the seven rows of the WB EPM Romania 12_46 FuelPrices sheet (IEA WEO
  2025 STEPS), as already loaded in supply/pFuelPrice.csv on 2026-09-24. The
  deployed file still carried the 12_45 rows (gas 18.35 in 2024 then 9.81 flat,
  uranium 1.50 to 5.61, biomass 5.0, no GasCCS, Hydrogen or Geothermal row).
* Bulgaria Gas and Uranium: the Romanian series. One gas market price for the two
  coupled markets, and one nuclear fuel cost (Kozloduy and Cernavoda buy on the
  same international market).
* Bulgaria DomesticCoal, ImportedCoal and Biomass: unchanged (Kinesys WEM, CCDR).

Usage (from EPM/):  python pre-analysis/pipelines/build_fuel_prices_robg.py
"""
from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve()
_EPM = _HERE.parents[2]
SUPPLY = _EPM / "epm" / "input" / "data_blacksea" / "supply"
SOURCE = SUPPLY / "pFuelPrice.csv"
TARGETS = [SUPPLY / "pFuelPrice_tr_gas_flat.csv", SUPPLY / "pFuelPrice.csv"]

RO_FUELS = ["DomesticCoal", "Uranium", "Gas", "Biomass", "Hydrogen", "GasCCS", "Geothermal"]
BG_FROM_RO = ["Gas", "Uranium"]


def _fail(msg: str) -> None:
    raise SystemExit(f"[build_fuel_prices_robg] FAIL: {msg}")


def _read(path: Path) -> tuple[bool, str, list[str]]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    eol = "\r\n" if "\r\n" in text else "\n"
    return raw.startswith(b"\xef\xbb\xbf"), eol, text.split(eol)


def romania_rows() -> dict[str, str]:
    _, _, lines = _read(SOURCE)
    rows = {ln.split(",")[1]: ln for ln in lines if ln.startswith("Romania,")}
    missing = [f for f in RO_FUELS if f not in rows]
    if missing:
        _fail(f"{SOURCE.name} lacks Romania rows {missing}")
    return {f: rows[f] for f in RO_FUELS}


def main() -> int:
    ro = romania_rows()
    for path in TARGETS:
        bom, eol, lines = _read(path)
        width = len(lines[0].split(","))
        out, ro_done = [], False
        for ln in lines:
            parts = ln.split(",")
            if parts[0] == "Romania":
                if not ro_done:
                    out.extend(ro[f] for f in RO_FUELS)
                    ro_done = True
                continue
            if parts[0] == "Bulgaria" and parts[1] in BG_FROM_RO:
                ln = f"Bulgaria,{parts[1]}," + ro[parts[1]].split(",", 2)[2]
            out.append(ln)
        if not ro_done:
            _fail(f"{path.name}: no Romania row")
        for ln in out:
            if ln and len(ln.split(",")) != width:
                _fail(f"{path.name}: bad width in '{ln[:40]}'")
        path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + eol.join(out).encode("utf-8"))
        print(f"  wrote {path.relative_to(_EPM)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
