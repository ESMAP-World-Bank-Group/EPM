# -*- coding: utf-8 -*-
"""Annual build limits of the Romanian and Bulgarian candidates (RoBg family, step 5d).

Romania and Bulgaria are internal only in the RoBg family (zcmap_robg.csv), so no
earlier scenario reads any row touched here. Idempotent: a second run changes nothing.

Two rules, applied identically to the four generator files and to both
pStorageDataInput files:

* GENERIC CANDIDATES (Status 3, name with "generic", "_cand" or "agg", technology in
  the rule's parameter table). BuildLimitperYear is set by the build-limit v2 rule
  of 2026-08-30, read from compute() in blacksea_2026/BuildLimit_method_slide.py,
  the same function that wrote the limits of the other modelled countries and that
  draws the method slide:
      L(c,t,y) = min[ b0 * gamma^(y-2025), plateau(c,t,y) ]
      b0 = max(observed 2025 additions, sigma * peak)
  The 2026-2040 path is flattened to its mean, then shared equally between the
  candidate rows of the zone (nine Romanian onshore wind classes, two offshore
  rows). Hydro, nuclear, pumped storage and geothermal (tech ST) are resource or
  site limited and stay outside the rule.
* NAMED PROJECTS (every other Status 3 row). BuildLimitperYear = Capacity, the repo
  convention, so that a project can be built in a single year. A candidate with an
  empty or zero limit is never built (main.gms, vBuild.up = BuildLimitperYear *
  pWeightYear), so no candidate is left at zero unless its Capacity is zero.

Usage (from EPM/):  python pre-analysis/pipelines/build_buildlimits_robg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from build_fleet_entsoe_bg import GEN_FILES, STORAGE_FILES, _EPM, _read, _write  # noqa: E402

_RULE = _EPM.parent / "BuildLimit_method_slide.py"
if not _RULE.exists():
    raise SystemExit(f"[build_buildlimits_robg] FAIL: rule script not found: {_RULE}")
sys.path.insert(0, str(_RULE.parent))
import BuildLimit_method_slide as rule  # noqa: E402

ZONES = ("Romania", "Bulgaria")


def _fmt(v: float) -> str:
    return f"{round(v, 1):g}"


def limits() -> dict[tuple[str, str], str]:
    """(zone, tech) -> per-row limit of the generic candidates, from the v2 rule."""
    out = {}
    for (c, t), d in rule.compute().items():
        if c in ZONES:
            for z, v in d["row_applied"].items():
                out[(z, t)] = _fmt(v)
    return out


def fix(path: Path, lim: dict) -> list[str]:
    bom, eol, lines = _read(path)
    head = lines[0].split(",")
    ix = {c: i for i, c in enumerate(head)}
    log = []
    for n, ln in enumerate(lines[1:], start=1):
        p = ln.split(",")
        if len(p) <= ix["BuildLimitperYear"] or p[ix["z"]] not in ZONES or p[ix["Status"]] != "3":
            continue
        key = (p[ix["z"]], p[ix["tech"]])
        if rule.is_generic(p[0]) and key in lim and key[1] not in rule.PROJECT_TECH:
            new, why = lim[key], "v2 rule"
        elif rule.is_generic(p[0]):
            continue
        else:
            new, why = p[ix["Capacity"]], "named project"
        old = p[ix["BuildLimitperYear"]]
        if new != old:
            p[ix["BuildLimitperYear"]] = new
            lines[n] = ",".join(p)
            log.append(f"{p[0]} ({key[0]}, {key[1]}, {p[ix['Capacity']]} MW): {old} -> {new} ({why})")
    _write(path, bom, eol, lines)
    return log


def main() -> int:
    lim = limits()
    for path in GEN_FILES + STORAGE_FILES:
        log = fix(path, lim)
        print(f"  {path.relative_to(_EPM)}: {len(log)} change(s)")
        if path == GEN_FILES[0] or path == STORAGE_FILES[0]:
            for line in log:
                print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
