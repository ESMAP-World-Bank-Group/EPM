# -*- coding: utf-8 -*-
"""S5 sensitivity family of the RoBg model: a 5-year grid and 29 scenarios.

Grid
----
y = 2025, 2030, 2032, 2036, 2040. main.gms weights each modelled year by the gap to
the previous one (1, 5, 2, 4, 4 = 16 years), so the horizon and the discounting of
the 16-year runs are kept while the LP is about a third of the size. Committed
plants (Status 2) are forced only in the year equal to their StYr (main.gms,
vBuild.lo and vBuild.up on gstatusmap committed), so a StYr that falls between two
grid years is moved to the next grid year, (2025, 2030] to 2030, (2030, 2032] to
2032, (2032, 2036] to 2036, (2036, 2040] to 2040. StYr 2025 and earlier and 2041 and
later are left as they are. Candidates need nothing (BuildLimitperYear is multiplied
by the year weight) and existing plants are fixed by StYr <= y and RetrYr <= y,
which work on any grid; retirement years are left as they are (a plant retired in
2033 is absent from the 2036 block, which covers 2033 to 2036).

Scenarios (29), per world Base, BSSC and GECO unless stated otherwise
--------------------------------------------------------------------
S5_<world>            reference: the inputs of LC_<world>_RoBg on the 5-year grid
S5_<world>_ETSLo/Hi   EU ETS uplift on Romanian and Bulgarian fossil fuels x 0.8 / x 1.2
S5_<world>_GasLo/Hi   Romanian and Bulgarian gas price before uplift (Gas, GasCCS) x 0.8 / x 1.2
S5_<world>_DemLo/Hi   Romanian and Bulgarian demand, energy and peak, x 0.9 / x 1.1
S5_<world>_AzGas10    Azerbaijani gas at 10 USD/MMBtu in every year (AzerbaijanMain, Nakhchivan, GEC_AZ)
S5_<world>_ResHi      Romanian and Bulgarian committed and candidate PV and wind x 1.2
                      (Capacity and BuildLimitperYear; existing plants unchanged)
S5_GECO_Hub50/Hub100  the 14 hub plants of supply/pGenDataInput_hub.csv (GEC_AZ, GEC_GE)
                      added at 50 % and 100 % of their capacity (GECO only: the hub
                      zones exist in zcmap_robg_geco.csv only)

The ETS uplift is recovered as pFuelPrice_robg.csv minus pFuelPrice_tr_gas_flat.csv
(the file build_ets_uplift_robg.py starts from), row by row and year by year.

Outputs, all new files; every existing file is left byte for byte
-----------------------------------------------------------------
y_s5.csv, cplex/cplex_s5.opt,
supply/pGenDataInput_s5.csv, _s5_reshi.csv, _s5_hub50.csv, _s5_hub100.csv,
supply/pFuelPrice_s5_etslo.csv, _etshi.csv, _gaslo.csv, _gashi.csv, _azgas10.csv,
load/pDemandForecast_s5_demlo.csv, _demhi.csv,
scenarios.csv: the 29 S5_* columns appended after the last column (rebuilt in place
when already present, every other cell untouched).

Rows that are not modified are written back byte for byte (the original line is
kept); only modified rows are re-serialised. Idempotent: a second run writes the
same bytes and reports every file as unchanged. Pure csv module, no pandas.

Usage: python build_s5_sensitivity_robg.py [path to data_blacksea]
"""
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.normpath(
    os.path.join(HERE, "..", "..", "epm", "input", "data_blacksea"))

GRID = [2025, 2030, 2032, 2036, 2040]
EU = ("Romania", "Bulgaria")
RES_TECHS = ("PV", "OnshoreWind", "OffshoreWind")
WORLDS = [("Base", "LC_Baseline_RoBg"), ("BSSC", "LC_BSSC_RoBg"), ("GECO", "LC_GECO_RoBg")]
ALL = ["Base", "BSSC", "GECO"]

# cells shared by every S5 column, applied after the copy of the LC_<world>_RoBg column
BASE_CELLS = {
    "cplexfile": "cplex/cplex_s5.opt",
    "y": "y_s5.csv",
    "pGenDataInput": "supply/pGenDataInput_s5.csv",
}
# (suffix, worlds, cells overriding BASE_CELLS), in the intended queue order
VARIANTS = [
    ("", ALL, {}),
    ("_ETSLo", ALL, {"pFuelPrice": "supply/pFuelPrice_s5_etslo.csv"}),
    ("_ETSHi", ALL, {"pFuelPrice": "supply/pFuelPrice_s5_etshi.csv"}),
    ("_GasLo", ALL, {"pFuelPrice": "supply/pFuelPrice_s5_gaslo.csv"}),
    ("_GasHi", ALL, {"pFuelPrice": "supply/pFuelPrice_s5_gashi.csv"}),
    ("_DemLo", ALL, {"pDemandForecast": "load/pDemandForecast_s5_demlo.csv"}),
    ("_DemHi", ALL, {"pDemandForecast": "load/pDemandForecast_s5_demhi.csv"}),
    ("_AzGas10", ALL, {"pFuelPrice": "supply/pFuelPrice_s5_azgas10.csv"}),
    ("_ResHi", ALL, {"pGenDataInput": "supply/pGenDataInput_s5_reshi.csv"}),
    ("_Hub50", ["GECO"], {"pGenDataInput": "supply/pGenDataInput_s5_hub50.csv"}),
    ("_Hub100", ["GECO"], {"pGenDataInput": "supply/pGenDataInput_s5_hub100.csv"}),
]

CPLEX_S5 = """* ------------------------------------------------------------
* CPLEX RMIP configuration for the S5 sensitivity family (5-year grid).
* Same settings as cplex_multithread.opt with 8 solver threads, so that
* three scenarios run side by side on the 24-core server:
*   epm.py --cpu 3  ->  3 x 8 = 24 threads, 3 x 25 GB workmem = 75 GB.
* The 5-year LPs are about a third of the 16-year ones, which run at
* --cpu 1 only (GAMS process plus barrier factor above 94 GB at cpu 3).
* ------------------------------------------------------------

* --- Key parameters ---
lpmethod=4
threads=8
baralg=0
barorder=3
barcrossalg=1
barcolnz=0
barepcomp=1e-4
parallelmode=0
memoryemphasis=0
workmem=25000
predual=-1
scaind=0
solutiontype=2

* --- Supporting parameters ---
names=no
bardisplay=2
iis=0
"""


def num(v, nd=6):
    """Decimal text without trailing zeros: 1500.0 -> 1500, 166.65 -> 166.65."""
    s = ("%." + str(nd) + "f") % v
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


class Table(object):
    """A csv file kept line by line: lines that are not modified are written back byte for byte."""

    def __init__(self, path=None):
        if path is None:
            return
        raw = open(path, "rb").read()
        self.bom = raw.startswith(b"\xef\xbb\xbf")
        text = raw.decode("utf-8-sig")
        self.eol = "\r\n" if "\r\n" in text else "\n"
        assert text.endswith(self.eol), path
        self.lines = text[:-len(self.eol)].split(self.eol)
        self.rows = [next(csv.reader([ln])) if ln else [] for ln in self.lines]
        self.touched = set()

    @property
    def header(self):
        return self.rows[0]

    def copy(self):
        t = Table()
        t.bom, t.eol = self.bom, self.eol
        t.lines = list(self.lines)
        t.rows = [list(r) for r in self.rows]
        t.touched = set(self.touched)
        return t

    def set(self, i, j, value):
        if self.rows[i][j] != value:
            self.rows[i][j] = value
            self.touched.add(i)

    def append(self, cells):
        self.rows.append(list(cells))
        self.lines.append(None)
        self.touched.add(len(self.rows) - 1)

    def line(self, i):
        if i in self.touched or self.lines[i] is None:
            buf = io.StringIO()
            csv.writer(buf, lineterminator="").writerow(self.rows[i])
            return buf.getvalue()
        return self.lines[i]

    def to_bytes(self):
        text = self.eol.join(self.line(i) for i in range(len(self.rows))) + self.eol
        return (b"\xef\xbb\xbf" if self.bom else b"") + text.encode("utf-8")


def write_bytes(rel, data):
    path = os.path.join(ROOT, rel)
    old = open(path, "rb").read() if os.path.exists(path) else None
    if old != data:
        open(path, "wb").write(data)
    print("  %-40s %-9s %7d bytes" % (rel, "unchanged" if old == data else ("rewritten" if old is not None else "new"), len(data)))


def write_table(rel, table):
    write_bytes(rel, table.to_bytes())


def snap(styr):
    s = int(float(styr))
    if GRID[0] < s <= GRID[-1]:
        return str(min(g for g in GRID if g >= s))
    return styr


# ------------------------------------------------------------------ generators
def build_generators():
    base = Table(os.path.join(ROOT, "supply", "pGenDataInput.csv"))
    hub = Table(os.path.join(ROOT, "supply", "pGenDataInput_hub.csv"))
    assert hub.header == base.header, "hub file header differs from the base file"
    H = base.header
    g, z, tech, status, styr, cap, bl = (H.index(c) for c in ("g", "z", "tech", "Status", "StYr", "Capacity", "BuildLimitperYear"))
    assert len({r[g] for r in base.rows[1:]}) == len(base.rows) - 1, "duplicate generator names"

    s5 = base.copy()
    moves = {}
    for i, r in enumerate(s5.rows[1:], 1):
        if r[status] == "2" and r[styr] != "":
            new = snap(r[styr])
            if new != r[styr]:
                moves[(r[styr], new)] = moves.get((r[styr], new), 0) + 1
                s5.set(i, styr, new)
    for r in s5.rows[1:]:
        if r[status] == "2" and r[styr] != "":
            v = int(float(r[styr]))
            assert v <= GRID[0] or v > GRID[-1] or v in GRID, r[:6]
    n_committed = sum(1 for r in s5.rows[1:] if r[status] == "2")
    print("pGenDataInput_s5: %d committed rows, %d StYr moved: %s" % (
        n_committed, sum(moves.values()), ", ".join("%s->%s x%d" % (a, b, n) for (a, b), n in sorted(moves.items()))))
    write_table("supply/pGenDataInput_s5.csv", s5)

    reshi = s5.copy()
    before = after = 0.0
    n = 0
    for i, r in enumerate(reshi.rows[1:], 1):
        if r[z] in EU and r[tech] in RES_TECHS and r[status] in ("2", "3"):
            before += float(r[bl] or 0)
            reshi.set(i, cap, num(float(r[cap]) * 1.2))
            if r[bl] != "":
                reshi.set(i, bl, num(float(r[bl]) * 1.2))
            after += float(reshi.rows[i][bl] or 0)
            n += 1
    print("pGenDataInput_s5_reshi: %d RO/BG PV and wind rows (Status 2 and 3) x 1.2, build limit sum %.1f -> %.1f MW/yr" % (n, before, after))
    write_table("supply/pGenDataInput_s5_reshi.csv", reshi)

    base_g = {r[g] for r in base.rows[1:]}
    hub_rows = [r for r in hub.rows[1:] if r[g] not in base_g]
    assert len(hub_rows) == 14, len(hub_rows)
    for r in hub_rows:
        assert r[status] == "2" and int(r[styr]) in GRID and r[z] in ("GEC_AZ", "GEC_GE") and r[cap] == r[bl], r[:9]
    for name, f in (("hub50", 0.5), ("hub100", 1.0)):
        t = s5.copy()
        total = 0.0
        for r in hub_rows:
            rr = list(r)
            rr[cap] = num(float(r[cap]) * f)
            rr[bl] = num(float(r[bl]) * f)
            total += float(rr[cap])
            t.append(rr)
        print("pGenDataInput_s5_%s: 14 hub rows appended, %.1f MW in all (GEC_AZ %.1f, GEC_GE %.1f)" % (
            name, total, sum(float(r[cap]) * f for r in hub_rows if r[z] == "GEC_AZ"),
            sum(float(r[cap]) * f for r in hub_rows if r[z] == "GEC_GE")))
        write_table("supply/pGenDataInput_s5_%s.csv" % name, t)


# ------------------------------------------------------------------ fuel prices
def build_fuel_prices():
    base = Table(os.path.join(ROOT, "supply", "pFuelPrice_tr_gas_flat.csv"))
    robg = Table(os.path.join(ROOT, "supply", "pFuelPrice_robg.csv"))
    assert base.header == robg.header and base.header[:2] == ["country", "fuel"], (base.header[:3], robg.header[:3])
    years = robg.header[2:]
    bidx = {(r[0], r[1]): i for i, r in enumerate(base.rows) if i > 0}
    assert set(bidx) == {(r[0], r[1]) for r in robg.rows[1:]}, "base and robg fuel price rows differ"
    uplift = {}
    for i, r in enumerate(robg.rows[1:], 1):
        b = base.rows[bidx[(r[0], r[1])]]
        u = [float(x) - float(y) for x, y in zip(r[2:], b[2:])]
        if r[0] not in EU:
            assert max(abs(x) for x in u) < 1e-9, "uplift outside Romania and Bulgaria: %s" % r[:2]
        uplift[i] = (b, u)
    rows = {(r[0], r[1]): i for i, r in enumerate(robg.rows) if i > 0}
    ro_gas, bg_gas, ro_ccs = (uplift[rows[k]][1] for k in (("Romania", "Gas"), ("Bulgaria", "Gas"), ("Romania", "GasCCS")))
    assert max(abs(a - b) for a, b in zip(ro_gas, bg_gas)) < 1e-6, "Romanian and Bulgarian gas uplifts differ"
    assert max(abs(a - 0.1 * b) for a, b in zip(ro_ccs, ro_gas)) < 1e-3, "GasCCS uplift is not 10 % of the gas uplift"
    show = [years.index(y) for y in ("2025", "2030", "2040")]
    print("fuel prices, Romania Gas 2025/2030/2040: base %s, uplift %s" % (
        [uplift[rows[("Romania", "Gas")]][0][2 + j] for j in show], [round(ro_gas[j], 4) for j in show]))

    specs = [("etslo", "ets", 0.8), ("etshi", "ets", 1.2), ("gaslo", "gas", 0.8), ("gashi", "gas", 1.2), ("azgas10", "az", None)]
    for name, kind, k in specs:
        t = robg.copy()
        touched = []
        for i, r in enumerate(t.rows[1:], 1):
            b, u = uplift[i]
            if kind == "ets" and r[0] in EU and max(abs(x) for x in u) > 1e-9:
                vals = [float(bb) + uu * k for bb, uu in zip(b[2:], u)]
            elif kind == "gas" and r[0] in EU and r[1] in ("Gas", "GasCCS"):
                vals = [float(bb) * k + uu for bb, uu in zip(b[2:], u)]
            elif kind == "az" and r[0] == "Azerbaijan" and r[1] == "Gas":
                vals = [10.0] * len(years)
            else:
                continue
            for j, v in enumerate(vals):
                t.set(i, 2 + j, num(v, 4))
            touched.append("%s %s" % (r[0], r[1]))
        sample = t.rows[rows[("Romania", "Gas")] if kind != "az" else rows[("Azerbaijan", "Gas")]]
        print("pFuelPrice_s5_%s: %d rows (%s); %s %s 2025/2030/2040 = %s" % (
            name, len(touched), ", ".join(touched), sample[0], sample[1], [sample[2 + j] for j in show]))
        write_table("supply/pFuelPrice_s5_%s.csv" % name, t)


# ------------------------------------------------------------------ demand
def build_demand():
    dem = Table(os.path.join(ROOT, "load", "pDemandForecast.csv"))
    assert dem.header[:2] == ["z", "type"], dem.header[:3]
    years = dem.header[2:]
    j30 = 2 + years.index("2030")
    for name, k in (("demlo", 0.9), ("demhi", 1.1)):
        t = dem.copy()
        n = 0
        for i, r in enumerate(t.rows[1:], 1):
            if r[0] in EU:
                for j in range(2, len(r)):
                    t.set(i, j, num(float(r[j]) * k, 6))
                n += 1
        assert n == 4, n
        print("pDemandForecast_s5_%s: %d rows x %.1f; 2030 %s" % (name, n, k, "; ".join(
            "%s %s %s -> %s" % (r[0], r[1], dem.rows[i][j30], t.rows[i][j30]) for i, r in enumerate(t.rows) if i > 0 and r[0] in EU)))
        write_table("load/pDemandForecast_s5_%s.csv" % name, t)


# ------------------------------------------------------------------ grid and solver
def build_grid_and_solver():
    y = Table(os.path.join(ROOT, "y.csv"))
    assert y.header == ["y"] and not y.bom, (y.header, y.bom)
    text = y.eol.join(["y"] + [str(v) for v in GRID]) + y.eol
    write_bytes("y_s5.csv", text.encode("utf-8"))
    ref = open(os.path.join(ROOT, "cplex", "cplex_multithread.opt"), "rb").read()
    eol = "\r\n" if b"\r\n" in ref else "\n"
    write_bytes(os.path.join("cplex", "cplex_s5.opt"), CPLEX_S5.replace("\n", eol).encode("utf-8"))


# ------------------------------------------------------------------ scenarios
def build_scenarios():
    scn = Table(os.path.join(ROOT, "scenarios.csv"))
    rows = scn.rows
    for ln, r in zip(scn.lines, rows):
        assert ",".join(r) == ln, "quoted cell in scenarios.csv, line kept as cells: %r" % ln[:60]
    H = rows[0]
    assert H[0] == "paramNames", H[:2]
    keep = [j for j, c in enumerate(H) if not c.startswith("S5_")]
    assert keep == list(range(len(keep))), "S5 columns are not the last columns"
    widths = {len(r) for r in rows}
    assert widths == {len(H)}, widths
    params = [r[0] for r in rows[1:]]
    widx = {w: H.index(col) for w, col in WORLDS}
    cols = []
    for suffix, worlds, over in VARIANTS:
        for w in worlds:
            cells = {r[0]: r[widx[w]] for r in rows[1:] if r[widx[w]] != ""}
            cells.update(BASE_CELLS)
            cells.update(over)
            cols.append(("S5_%s%s" % (w, suffix), cells))
    assert len(cols) == 29, len(cols)
    for name, cells in cols:
        unknown = set(cells) - set(params)
        assert not unknown, (name, unknown)
        for v in cells.values():
            if ".csv" in v or ".opt" in v:
                assert os.path.exists(os.path.join(ROOT, v)), (name, v)
    out = []
    for i, r in enumerate(rows):
        head = r[:len(keep)]
        if i == 0:
            out.append(head + [n for n, _ in cols])
        else:
            out.append(head + [c.get(r[0], "") for _, c in cols])
    text = scn.eol.join(",".join(r) for r in out) + scn.eol
    data = (b"\xef\xbb\xbf" if scn.bom else b"") + text.encode("utf-8")
    write_bytes("scenarios.csv", data)
    # check: the existing columns are byte for byte the ones of the input, the new ones are as built
    back = Table(os.path.join(ROOT, "scenarios.csv"))
    assert [r[:len(keep)] for r in back.rows] == [r[:len(keep)] for r in rows]
    assert back.header[len(keep):] == [n for n, _ in cols]
    assert {len(r) for r in back.rows} == {len(keep) + 29}
    filled = {r[0]: sum(1 for v in r[len(keep):] if v != "") for r in back.rows[1:] if any(v != "" for v in r[len(keep):])}
    print("scenarios.csv: %d existing columns kept, 29 S5 columns; cells filled per row: %s" % (
        len(keep), ", ".join("%s %d" % kv for kv in filled.items())))
    print("queue order: " + " ".join(n for n, _ in cols))


if __name__ == "__main__":
    print("data folder:", ROOT)
    build_grid_and_solver()
    build_generators()
    build_fuel_prices()
    build_demand()
    build_scenarios()
