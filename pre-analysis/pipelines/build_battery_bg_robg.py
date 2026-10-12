# -*- coding: utf-8 -*-
"""Battery candidate for Bulgaria (RoBg family, step 11).

Bulgaria had existing and committed batteries but no candidate, so the model could
never add storage there while Romania could. The Romanian candidate
"Generic Battery Storage" is cloned for Bulgaria under the name
BG_NAME, with the same costs, efficiency, duration cap and availability. Its
BuildLimitperYear is then set by build_buildlimits_robg.py (build-limit v2 rule),
which must be run after this script.

Bulgaria is internal only in the RoBg family (zcmap_robg.csv), so no earlier
scenario reads the new rows. Idempotent: a second run changes nothing.

Rows added (inserted after the last Bulgarian row, or after the Romanian template
in the availability file):
* the four generator files (GEN_FILES)
* both storage files (STORAGE_FILES); a file without the Romanian template
  (the CESI storage file) is left as is
* supply/pAvailabilityCustom.csv

Usage (from EPM/):  python pre-analysis/pipelines/build_battery_bg_robg.py
                    python pre-analysis/pipelines/build_buildlimits_robg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from build_fleet_entsoe_bg import AVAIL_FILE, GEN_FILES, STORAGE_FILES, _EPM, _read, _write  # noqa: E402

TEMPLATE = "Generic Battery Storage"
BG_NAME = "Bulgaria_Generic_Battery_Storage"
ZONE = "Bulgaria"


def _fail(msg: str) -> None:
    raise SystemExit(f"[build_battery_bg_robg] FAIL: {msg}")


def add_row(path: Path, zone_col: int | None) -> str:
    """Clone the template row for Bulgaria. zone_col is None for the availability file."""
    bom, eol, lines = _read(path)
    names = [ln.split(",")[0] for ln in lines]
    if BG_NAME in names:
        return "already there"
    if names.count(TEMPLATE) == 0:
        return "no Romanian candidate in this file, left as is"
    if names.count(TEMPLATE) != 1:
        _fail(f"{path.name}: expected one '{TEMPLATE}' row, found {names.count(TEMPLATE)}")
    p = lines[names.index(TEMPLATE)].split(",")
    p[0] = BG_NAME
    if zone_col is None:
        at = names.index(TEMPLATE) + 1
    else:
        if p[zone_col] != "Romania":
            _fail(f"{path.name}: template is not Romanian")
        p[zone_col] = ZONE
        bg = [n for n, ln in enumerate(lines) if len(ln.split(",")) > zone_col
              and ln.split(",")[zone_col] == ZONE]
        if not bg:
            _fail(f"{path.name}: no Bulgarian row")
        at = bg[-1] + 1
    lines.insert(at, ",".join(p))
    _write(path, bom, eol, lines)
    return f"added at line {at + 1}"


def main() -> int:
    for path in GEN_FILES + STORAGE_FILES:
        head = _read(path)[2][0].split(",")
        print(f"  {path.relative_to(_EPM)}: {add_row(path, head.index('z'))}")
    print(f"  {AVAIL_FILE.relative_to(_EPM)}: {add_row(AVAIL_FILE, None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
