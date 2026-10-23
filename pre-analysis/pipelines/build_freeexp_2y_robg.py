# -*- coding: utf-8 -*-
"""Two-year grid variant of the free expansion reference: LC_FreeExp_RoBg_2y.

Fallback when the 16-year LC_FreeExp_RoBg run is too long. Same inputs as
LC_FreeExp_RoBg (every discussed project a candidate, internal and external,
see build_freeexp_robg.py) solved on the grid y_2y.csv = 2025, 2026, 2028, 2030,
2032, 2034, 2036, 2038, 2040. main.gms weights each modelled year by the gap to
the previous one (1, 1, 2, 2, 2, 2, 2, 2, 2 = 16 years), so the horizon and the
discounting of the 16-year run are kept while the LP is about 9/16 of its size.
The S5 grid (2025, 2030, 2032, 2036, 2040) is a subset of this one.

Committed plants (Status 2) are forced only in the modelled year equal to their
StYr (main.gms, vBuild.lo and vBuild.up on gstatusmap committed), so an odd StYr
between 2027 and 2039 is moved to the next grid year in supply/pGenDataInput_2y.csv
(2027 to 2028, 2029 to 2030, ... 2039 to 2040), as build_s5_sensitivity_robg.py
does for the five-year grid. StYr 2025, 2026, the even years and the years outside
the horizon are unchanged. Committed lines are forced by EarliestEntry <= y
(main.gms) and candidates need nothing: BuildLimitperYear is multiplied by the
year weight and a candidate line with EarliestEntry 2031, 2033 or 2035 becomes
available in the 2032, 2034 or 2036 block. Retirement years are left as they are
(a plant retired in 2033 is absent from the 2034 block, which covers 2033 and 2034).

Outputs, all new files; every existing file is left byte for byte
-----------------------------------------------------------------
y_2y.csv, supply/pGenDataInput_2y.csv,
scenarios.csv: the column LC_FreeExp_RoBg_2y = LC_FreeExp_RoBg with y and
pGenDataInput replaced (rebuilt in place when already present). The solver file
stays cplex/cplex_multithread.opt: the scenario runs alone at --cpu 1.

Idempotent: a second run writes the same bytes and reports every file as unchanged.

Usage: python build_freeexp_2y_robg.py [path to data_blacksea]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_freeexp_robg import Table  # noqa: E402  (byte preserving csv table, same folder)

ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.normpath(
    os.path.join(HERE, "..", "..", "epm", "input", "data_blacksea"))

GRID = [2025, 2026] + list(range(2028, 2041, 2))
COLUMN, SOURCE = "LC_FreeExp_RoBg_2y", "LC_FreeExp_RoBg"
OVERRIDES = {"y": "y_2y.csv", "pGenDataInput": "supply/pGenDataInput_2y.csv"}


def write_bytes(rel, data):
    path = os.path.join(ROOT, rel)
    old = open(path, "rb").read() if os.path.exists(path) else None
    if old != data:
        open(path, "wb").write(data)
    print("  %-44s %-9s %7d bytes" % (rel, "unchanged" if old == data else ("rewritten" if old is not None else "new"), len(data)))


def write_table(rel, table):
    write_bytes(rel, table.to_bytes())


def snap(styr):
    s = int(float(styr))
    if GRID[0] < s <= GRID[-1]:
        return str(min(g for g in GRID if g >= s))
    return styr


def build_grid():
    y = Table(os.path.join(ROOT, "y.csv"))
    assert y.header == ["y"] and not y.bom, (y.header, y.bom)
    assert [int(r[0]) for r in y.rows[1:]] == list(range(2025, 2041)), "y.csv is not 2025 to 2040"
    assert sum(b - a for a, b in zip(GRID, GRID[1:])) + 1 == 16, GRID
    write_bytes("y_2y.csv", (y.eol.join(["y"] + [str(v) for v in GRID]) + y.eol).encode("utf-8"))


def build_generators():
    base = Table(os.path.join(ROOT, "supply", "pGenDataInput.csv"))
    H = base.header
    g, status, styr = (H.index(c) for c in ("g", "Status", "StYr"))
    assert len({r[g] for r in base.rows[1:]}) == len(base.rows) - 1, "duplicate generator names"
    out = base.copy()
    moves = {}
    for i, r in enumerate(out.rows[1:], 1):
        if r[status] == "2" and r[styr] != "":
            new = snap(r[styr])
            if new != r[styr]:
                moves[(r[styr], new)] = moves.get((r[styr], new), 0) + 1
                out.set(i, styr, new)
    for r in out.rows[1:]:
        if r[status] == "2" and r[styr] != "":
            v = int(float(r[styr]))
            assert v <= GRID[0] or v > GRID[-1] or v in GRID, r[:6]
    for a, b in moves:
        assert int(b) - int(a) == 1, (a, b)
    n_committed = sum(1 for r in out.rows[1:] if r[status] == "2")
    print("pGenDataInput_2y: %d committed rows, %d StYr moved by one year: %s" % (
        n_committed, sum(moves.values()), ", ".join("%s->%s x%d" % (a, b, n) for (a, b), n in sorted(moves.items()))))
    write_table("supply/pGenDataInput_2y.csv", out)


def build_scenarios():
    sc = Table(os.path.join(ROOT, "scenarios.csv"))
    H = sc.header
    assert H[0] == "paramNames", H[:3]
    names = [r[0] for r in sc.rows[1:]]
    j = H.index(SOURCE)
    cells = [r[j] for r in sc.rows[1:]]
    assert cells[names.index("pNewTransmissionExt")] == "trade/pNewTransmissionExt_robg_freeexp.csv", SOURCE
    assert cells[names.index("pTransferLimit")] == "trade/pTransferLimit_robg_freeexp.csv", SOURCE
    assert cells[names.index("y")] == "" and cells[names.index("pGenDataInput")] == "", "source column already overrides y or pGenDataInput"
    assert cells[names.index("cplexfile")] == "", "source column overrides the solver file"
    for row, path in OVERRIDES.items():
        assert os.path.exists(os.path.join(ROOT, path)), path
        cells[names.index(row)] = path
    sc.append_column(COLUMN, cells)
    changed = [names[k] for k in range(len(names)) if cells[k] != sc.rows[k + 1][j]]
    print("scenarios.csv: %s = %s with %s" % (COLUMN, SOURCE, ", ".join(changed)))
    write_table("scenarios.csv", sc)
    back = Table(os.path.join(ROOT, "scenarios.csv"))
    k = back.header.index(COLUMN)
    assert [r[k] for r in back.rows[1:]] == cells
    assert [r[:j + 1] for r in back.rows] == [r[:j + 1] for r in sc.rows]


if __name__ == "__main__":
    print("data folder: " + ROOT)
    build_grid()
    build_generators()
    build_scenarios()
