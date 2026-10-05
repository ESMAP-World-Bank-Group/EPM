# -*- coding: utf-8 -*-
"""Silent fleet fixes for Romania and Bulgaria (RoBg family, step 5a).

Romania and Bulgaria are internal only in the RoBg family (zcmap_robg.csv), so
no earlier scenario reads any row touched here. Every other row is left byte for
byte. Idempotent: a second run changes nothing.

1. Committed rows with a StYr before the first modelled year (2025) are never
   built by EPM (vBuild is only forced at y = StYr). The eight Romanian rows in
   that case are set to Status 1 (existing), the rule already used for
   HPP Lotru_ext: the source dates them 2023-2024, so they are in service in 2025
   and their capex is sunk.
2. The four Default files carried no Romania or Bulgaria row, so EPM filled the
   blanks from the generic resources/ values instead of the CCDR table used for
   every other zone. The Trakia block (identical to the common CCDR block) is
   copied for both zones. Romania also gets GasCCS and Hydrogen rows for CCGT
   and OCGT, cloned from the Gas rows of the same technology.
3. Romanian hydro plants retired one or more years before their rehabilitated
   _ext row starts: RetrYr of the parent is set to the StYr of the _ext row, so
   capacity is continuous. Only retirements inside the model horizon
   (2025-2040) are touched.
4. Bulgarian existing or committed rows with an empty RetrYr get 2060, the
   no-unplanned-retirement rule of 2026-09-05.

Fixes 1, 3 and 4 are applied to the four generator files that carry the same
Romania and Bulgaria block.

Usage (from EPM/):  python pre-analysis/pipelines/build_fleet_fixes_robg.py
"""
from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve()
_EPM = _HERE.parents[2]
DATA = _EPM / "epm" / "input" / "data_blacksea"
GEN_FILES = [
    DATA / "supply" / "pGenDataInput.csv",
    DATA / "supply" / "pGenDataInput_hub.csv",
    DATA / "supply" / "pGenDataInput_armnuke.csv",
    DATA / "cesi" / "unwired" / "pGenDataInput_cesi.csv",
]
DEFAULT_FILES = ["pGenDataInputDefault", "pAvailabilityDefault",
                 "pStorageDataInputDefault", "pCapexTrajectoriesDefault"]
TEMPLATE_ZONE = "Trakia"
ZONES = ["Romania", "Bulgaria"]
RO_FUEL_CLONES = {("CCGT", "Gas"): ["GasCCS", "Hydrogen"], ("OCGT", "Gas"): ["GasCCS", "Hydrogen"]}

PRE_START_COMMITTED = ["Alum Tulcea", "Borzești", "Brăila", "CCE", "Iernut-Ludus - Expansion",
                       "Ișalnița", "Photon", "TPP Midia_Navodari_Ext"]
START_YEAR = 2025
END_YEAR = 2040
NO_RETIREMENT_YEAR = 2060


def _fail(msg: str) -> None:
    raise SystemExit(f"[build_fleet_fixes_robg] FAIL: {msg}")


def _read(path: Path) -> tuple[bool, str, list[str]]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    eol = "\r\n" if "\r\n" in text else "\n"
    return raw.startswith(b"\xef\xbb\xbf"), eol, text.split(eol)


def _write(path: Path, bom: bool, eol: str, lines: list[str]) -> None:
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + eol.join(lines).encode("utf-8"))


def fix_generators(path: Path) -> list[str]:
    bom, eol, lines = _read(path)
    head = lines[0].split(",")
    width = len(head)
    ix = {c: i for i, c in enumerate(head)}
    rows = {}
    for n, ln in enumerate(lines):
        p = ln.split(",")
        if len(p) > 1 and p[ix["z"]] in ZONES:
            if len(p) != width:
                _fail(f"{path.name}: unexpected width on '{p[0]}'")
            rows[p[0]] = (n, p)
    log = []

    for g in PRE_START_COMMITTED:
        if g not in rows:
            _fail(f"{path.name}: no row '{g}'")
        n, p = rows[g]
        if p[ix["Status"]] == "2" and int(float(p[ix["StYr"]])) < START_YEAR:
            p[ix["Status"]] = "1"
            log.append(f"status 2->1 {g}")

    for g, (n, p) in rows.items():
        if p[ix["z"]] != "Romania" or not g.endswith("_ext"):
            continue
        parent = rows.get(g[:-4])
        if parent is None:
            continue
        pp = parent[1]
        ext_st = int(float(p[ix["StYr"]]))
        retr = int(float(pp[ix["RetrYr"]])) if pp[ix["RetrYr"]] else None
        if retr is not None and START_YEAR <= retr <= END_YEAR and retr < ext_st:
            log.append(f"RetrYr {pp[ix['RetrYr']]}->{ext_st} {pp[0]}")
            pp[ix["RetrYr"]] = str(ext_st)

    for g, (n, p) in rows.items():
        if p[ix["z"]] == "Bulgaria" and p[ix["Status"]] in ("1", "2") and p[ix["RetrYr"]] == "":
            p[ix["RetrYr"]] = str(NO_RETIREMENT_YEAR)
            log.append(f"RetrYr blank->{NO_RETIREMENT_YEAR} {g}")

    for n, p in rows.values():
        lines[n] = ",".join(p)
    _write(path, bom, eol, lines)
    return log


def fix_defaults(name: str) -> int:
    path = DATA / "supply" / f"{name}.csv"
    bom, eol, lines = _read(path)
    trailing = lines and lines[-1] == ""
    body = lines[:-1] if trailing else lines
    zones_present = {ln.split(",")[0] for ln in body[1:]}
    template = [ln for ln in body[1:] if ln.split(",")[0] == TEMPLATE_ZONE]
    if not template:
        _fail(f"{name}: no {TEMPLATE_ZONE} block")
    added = []
    for z in ZONES:
        if z in zones_present:
            continue
        for ln in template:
            p = ln.split(",")
            added.append(",".join([z] + p[1:]))
            if z == "Romania":
                for f in RO_FUEL_CLONES.get((p[1], p[2]), []):
                    added.append(",".join([z, p[1], f] + p[3:]))
    _write(path, bom, eol, body + added + ([""] if trailing else []))
    return len(added)


def main() -> int:
    logs = [fix_generators(path) for path in GEN_FILES]
    for path, log in zip(GEN_FILES, logs):
        print(f"  {path.relative_to(_EPM)}: {len(log)} change(s)")
    if any(log != logs[0] for log in logs):
        _fail("the four generator files did not receive the same changes")
    for line in logs[0]:
        print(f"    {line}")
    for name in DEFAULT_FILES:
        print(f"  supply/{name}.csv: {fix_defaults(name)} row(s) added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
