# -*- coding: utf-8 -*-
"""Last Romanian and Bulgarian calibration fixes before the final RoBg run (step 17 and cleanup).

Romania and Bulgaria are internal only in the RoBg family (zcmap_robg.csv), so no
earlier scenario reads any generator or availability row touched here. The new
MinGenShareAllHours column is empty on every other row, and input_treatment fills
an empty cell from pGenDataInputDefault exactly as it fills a missing column, so
other zones are unchanged. Run after build_plan_alignment_robg.py. Idempotent: a
second run changes nothing.

Why: robg5 (2026-10-16) produced 5.9 TWh of Romanian gas in 2025 against
10.3 TWh observed (ENTSO-E 2024) and a Romanian net import of 5.2 TWh against
2.8; Bulgarian hydro gave 4.3 TWh against 3.3 TWh observed on average in
2019-2025; Bulgarian solar gained 1.5 GW of generic capacity in 2025, a year
already covered by the ENTSO-E calibrated fleet.

17 Romanian gas CHP floor
-------------------------
The eight existing gas CHP plants (district heating of Bucharest, Galati, Iasi,
Oradea and Ramnicu Valcea; 1,283 MW) and the five CHP replacement units (_Ext)
get MinGenShareAllHours 0.40 (eMinGen, base.gms:789, fApplyMinGenShareAllHours
is on). Without a heat obligation these units ran only on price, as OCGT. 0.40
x 1,283 MW is about 4.5 TWh a year. The floor applies to every hour of every
year the unit is online: a CHP follows the heat load, higher in winter, so the
flat floor overstates summer output and understates winter output.

17b Romanian CHP double count
-----------------------------
Five CHP plants have a replacement row (_Ext, Status 2, 2027 or 2028) while the
original stayed online to 2060 or later, so both ran together (852 MW counted
twice). The original now retires the year before its replacement starts, the
rule applied to the Romanian hydro pairs in step 11.

Bulgarian solar start year
--------------------------
Generic_Solar_Bulgaria and Bulgaria_St_George_solar_PV start in 2027 (were
2025). The 2025 and 2026 additions are already in the ENTSO-E rows
(Bulgaria_AGG_PV_ENTSOE, Bulgaria_AGG_PV_2026); robg5 built 1.5 GW of the
generic in 2025 on top of them.

Bulgarian reservoir hydro
-------------------------
The 17 reservoir rows of pAvailabilityCustom.csv (13 existing plants, three
candidates, Bulgaria_AGG_Reservoir_ENTSOE) keep their quarterly shape but take
the ENTSO-E mean capacity factor (0.146) instead of the Kinesys one (0.195).
Bulgarian hydro energy goes from about 4.3 to about 3.6 TWh, against 3.3 TWh
observed on average in 2019-2025. Run-of-river is already on ENTSO-E.

Usage (from EPM/):  python pre-analysis/pipelines/build_final_calibration_robg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from build_fleet_entsoe_bg import AVAIL_FILE, GEN_FILES, _EPM, _read, _write  # noqa: E402

MINGEN_COL = "MinGenShareAllHours"
CHP_FLOOR = "0.4"
CHP_ROWS = ["TPP Bucuresti Vest", "TPP Bucuresti Sud", "TPP Progresu", "Grozăvești",
            "TPP Galati", "Iasi I", "CET Govora", "Bihor"]
CHP_PAIRS = {"TPP Bucuresti Vest": "TPP Bucuresti Vest_Ext", "TPP Bucuresti Sud": "TPP Bucuresti Sud_Ext",
             "TPP Progresu": "TPP Progresu_Ext", "Grozăvești": "Grozăvești_Ext",
             "CET Govora": "CET Govora_Ext"}
SOLAR_START = {"Generic_Solar_Bulgaria": "2027", "Bulgaria_St_George_solar_PV": "2027"}

RESERVOIR_ROWS = ["Bulgaria_Belmeken_Hydro", "Bulgaria_Devin_Hydro", "Bulgaria_Ivailovgrad_Hydro",
                  "Bulgaria_Kardjali_Hydro", "Bulgaria_Krichim_Hydro", "Bulgaria_Momina_Klisura_Hydro",
                  "Bulgaria_Orfeus_Hydro", "Bulgaria_Peshtera_Hydro", "Bulgaria_Sestrimo_Hydro",
                  "Bulgaria_Studen_Kladenets_Hydro", "Bulgaria_Tsankov_Kamak_Hydro", "Bulgaria_Teshel_Hydro",
                  "Bulgaria_Aleko_Hydro", "Bulgaria_Turnu_Magurele_Hydro", "Bulgaria_Batak_Hydro",
                  "Bulgaria_Dospat_Hydro", "Bulgaria_AGG_Reservoir_ENTSOE"]
KINESYS_Q = [0.2142, 0.2481, 0.1604, 0.1556]
ENTSOE_MEAN = 0.146


def _fail(msg: str) -> None:
    raise SystemExit(f"[build_final_calibration_robg] FAIL: {msg}")


def _entsoe_q() -> list[str]:
    k = ENTSOE_MEAN / (sum(KINESYS_Q) / 4)
    return [f"{q * k:.4f}" for q in KINESYS_Q]


def fix_generators(path: Path) -> list[str]:
    bom, eol, lines = _read(path)
    trailing = lines[-1] == ""
    body = lines[:-1] if trailing else lines
    head = body[0].split(",")
    log = []
    if MINGEN_COL not in head:
        head.append(MINGEN_COL)
        body[0] = ",".join(head)
        for n in range(1, len(body)):
            if body[n]:
                p = body[n].split(",")
                body[n] = ",".join(p + [""] * (len(head) - len(p)))
        log.append(f"column {MINGEN_COL} added, empty")
    ix = {c: i for i, c in enumerate(head)}
    where = {ln.split(",")[0]: n for n, ln in enumerate(body) if ln and n}
    needed = CHP_ROWS + list(CHP_PAIRS.values()) + list(SOLAR_START)
    missing = [g for g in needed if g not in where]
    if missing:
        _fail(f"{path.name}: rows missing {missing}")

    def put(g, col, new):
        n = where[g]
        p = body[n].split(",")
        if p[ix[col]] != new:
            log.append(f"{g}: {col} {p[ix[col]] or 'empty'} -> {new}")
            p[ix[col]] = new
            body[n] = ",".join(p)

    def get(g, col):
        return body[where[g]].split(",")[ix[col]]

    for g in CHP_ROWS + list(CHP_PAIRS.values()):
        if get(g, "z") != "Romania":
            _fail(f"{g} is not a Romanian row")
        put(g, MINGEN_COL, CHP_FLOOR)
    for parent, ext in CHP_PAIRS.items():
        put(parent, "RetrYr", str(int(get(ext, "StYr")) - 1))
    for g, y in SOLAR_START.items():
        put(g, "StYr", y)
    _write(path, bom, eol, body + ([""] if trailing else []))
    return log


def fix_reservoir() -> list[str]:
    bom, eol, lines = _read(AVAIL_FILE)
    old = [f"{q:.4f}" for q in KINESYS_Q]
    new = _entsoe_q()
    where = {ln.split(",")[0]: n for n, ln in enumerate(lines) if ln}
    log = []
    for g in RESERVOIR_ROWS:
        if g not in where:
            _fail(f"{g} missing from {AVAIL_FILE.name}")
        vals = lines[where[g]].split(",")[1:]
        if vals == new:
            continue
        if vals != old:
            _fail(f"{g}: unexpected availability {vals}, expected the Kinesys profile {old}")
        lines[where[g]] = ",".join([g] + new)
        log.append(f"{g}: {'/'.join(old)} -> {'/'.join(new)}")
    _write(AVAIL_FILE, bom, eol, lines)
    return log


def main() -> int:
    for path in GEN_FILES:
        log = fix_generators(path)
        print(f"  {path.relative_to(_EPM)}: {len(log)} change(s)")
        if path == GEN_FILES[0]:
            for line in log:
                print(f"    {line}")
    log = fix_reservoir()
    print(f"  {AVAIL_FILE.relative_to(_EPM)}: {len(log)} change(s)")
    for line in log:
        print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
