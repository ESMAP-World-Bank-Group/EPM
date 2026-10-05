# -*- coding: utf-8 -*-
"""Retirement dates of the Romanian and Bulgarian fleets (RoBg family, step 5c).

Romania and Bulgaria are internal only in the RoBg family (zcmap_robg.csv), so no
earlier scenario reads any row touched here. Idempotent: a second run changes nothing.

Rules, applied identically to the four generator files and to both
pStorageDataInput files:

* COAL (DomesticCoal, ImportedCoal), rows still in service in 2025. Romania: last
  year of operation 2032 at the latest, so RetrYr = min(RetrYr, 2033); the
  decarbonisation law of 2022 removes coal from the power mix by the end of 2032,
  and earlier closure dates of the source are kept. Bulgaria: last year of
  operation 2038, so RetrYr = 2039; the 2022 recovery plan sets 2038 as the coal
  exit date, and the source retired the aggregates in 2033 and 2035.
* CERNAVODA-1. Cernavoda-1 keeps RetrYr 2027 (grid disconnection for the
  refurbishment in 2027). Cernavoda-1_refurbished moves from candidate to committed
  (Status 2, StYr 2030), since the refurbishment is contracted.
* NO UNPLANNED RETIREMENT, the 2026-09-05 rule of the other modelled zones. Every
  other Status 1 or 2 row whose RetrYr falls in 2025-2040 gets RetrYr 2060 and
  Life = 2060 - StYr. Hydro parents with an _ext row are excluded: their RetrYr is
  the start of the rehabilitated plant (step 5a).

Usage (from EPM/):  python pre-analysis/pipelines/build_fleet_retirement_robg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from build_fleet_entsoe_bg import GEN_FILES, STORAGE_FILES, _EPM, _read, _write  # noqa: E402

ZONES = ("Romania", "Bulgaria")
COAL_FUELS = ("DomesticCoal", "ImportedCoal")
COAL_RETIREMENT = {"Romania": 2033, "Bulgaria": 2039}
NO_RETIREMENT_YEAR = 2060
HORIZON = (2025, 2040)
REFURBISHED = "Cernavoda-1_refurbished"
REFURBISHED_STYR = 2030


def _int(v: str) -> int | None:
    try:
        return int(float(v))
    except ValueError:
        return None


def new_dates(p: list[str], ix: dict, names: set[str]) -> tuple[str, str, str, str] | None:
    """Return (Status, StYr, RetrYr, Life) for a row, or None if out of scope."""
    if len(p) <= ix["RetrYr"] or p[ix["z"]] not in ZONES or p[ix["Status"]] not in ("1", "2", "3"):
        return None
    status, styr, retr, life = p[ix["Status"]], p[ix["StYr"]], p[ix["RetrYr"]], p[ix["Life"]]
    if p[0] == REFURBISHED:
        return "2", str(REFURBISHED_STYR), retr, life
    r, s = _int(retr), _int(styr)
    if status == "3" or r is None or r < HORIZON[0]:
        return None
    if p[ix["f"]] in COAL_FUELS:
        cap = COAL_RETIREMENT[p[ix["z"]]]
        new = min(r, cap) if p[ix["z"]] == "Romania" else cap
        return status, styr, str(new), life
    if r > HORIZON[1] or p[ix["tech"]] == "Nuclear" or f"{p[0]}_ext" in names:
        return None
    return status, styr, str(NO_RETIREMENT_YEAR), str(NO_RETIREMENT_YEAR - s) if s else life


def fix(path: Path) -> list[str]:
    bom, eol, lines = _read(path)
    head = lines[0].split(",")
    ix = {c: i for i, c in enumerate(head)}
    names = {ln.split(",")[0] for ln in lines}
    log = []
    for n, ln in enumerate(lines[1:], start=1):
        p = ln.split(",")
        nd = new_dates(p, ix, names)
        if nd is None:
            continue
        old = (p[ix["Status"]], p[ix["StYr"]], p[ix["RetrYr"]], p[ix["Life"]])
        if nd != old:
            p[ix["Status"]], p[ix["StYr"]], p[ix["RetrYr"]], p[ix["Life"]] = nd
            lines[n] = ",".join(p)
            log.append(f"{p[0]} ({p[ix['z']]}, {p[ix['tech']]}, {p[ix['Capacity']]} MW): "
                       f"Status {old[0]}->{nd[0]}, StYr {old[1]}->{nd[1]}, RetrYr {old[2]}->{nd[2]}")
    _write(path, bom, eol, lines)
    return log


def main() -> int:
    for path in GEN_FILES + STORAGE_FILES:
        log = fix(path)
        print(f"  {path.relative_to(_EPM)}: {len(log)} change(s)")
        if path == GEN_FILES[0] or path == STORAGE_FILES[0]:
            for line in log:
                print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
