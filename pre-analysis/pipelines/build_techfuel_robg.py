# -*- coding: utf-8 -*-
"""Technology and fuel pairs of the RoBg family (step 8).

Five Romanian candidates of the v8 workbook burn fuels that epm/resources/pTechFuel.csv
does not pair with their technology: CCGT and OCGT on GasCCS (Brazi_Retrofited and the
two generic CCS rows) and CCGT and OCGT on Hydrogen. While Romania was external those
rows were filtered out; once Romania is internal, input_verification stops the run.

The shared resource file is left untouched. This script writes a copy of it with the
four missing pairs into data_blacksea/supply/pTechFuel_robg.csv, which only the
LC_Baseline_RoBg column of scenarios.csv points to. None of the four pairs is renewable
or hourly variable. The FuelIndex values are new (36 to 39), since the existing indices
already carry a fuel each. Idempotent: a second run writes the same bytes.

Usage (from EPM/):  python pre-analysis/pipelines/build_techfuel_robg.py
"""
from __future__ import annotations

from pathlib import Path

_EPM = Path(__file__).resolve().parents[2]
SRC = _EPM / "epm" / "resources" / "pTechFuel.csv"
DST = _EPM / "epm" / "input" / "data_blacksea" / "supply" / "pTechFuel_robg.csv"

# tech, fuel, HourlyVariation, RETechnology, FuelIndex
EXTRA = [
    ("CCGT", "GasCCS", 0, 0, 36),
    ("OCGT", "GasCCS", 0, 0, 37),
    ("CCGT", "Hydrogen", 0, 0, 38),
    ("OCGT", "Hydrogen", 0, 0, 39),
]


def main() -> int:
    raw = SRC.read_bytes()
    text = raw.decode("utf-8-sig")
    eol = "\r\n" if "\r\n" in text else "\n"
    lines = [ln for ln in text.split(eol) if ln]
    pairs = {tuple(ln.split(",")[:2]) for ln in lines[1:]}
    used = {ln.split(",")[4] for ln in lines[1:]}
    for t, f, hv, re_, idx in EXTRA:
        if (t, f) in pairs:
            raise SystemExit(f"[build_techfuel_robg] FAIL: {t},{f} already in {SRC.name}")
        if str(idx) in used:
            raise SystemExit(f"[build_techfuel_robg] FAIL: FuelIndex {idx} already used")
        lines.append(f"{t},{f},{hv},{re_},{idx}")
    bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
    DST.write_bytes(bom + (eol.join(lines) + eol).encode("utf-8"))
    print(f"  {DST.relative_to(_EPM)}: {len(lines) - 1} pairs ({len(EXTRA)} added)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
