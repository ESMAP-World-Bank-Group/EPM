# -*- coding: utf-8 -*-
"""Free expansion family of the RoBg model: every corridor of the study as a candidate.

Two scenarios, both on the inputs of LC_GECO_RoBg (zcmap_robg_geco.csv, so the GEC hub
zones exist):
LC_FreeExp_RoBg   16 years (2025 to 2040), the horizon of the LC_*_RoBg family
S5_FreeExp        the 5-year grid of the S5 family (y_s5.csv, pGenDataInput_s5.csv, cplex_s5.opt)

Candidate lines, all free (Status 3) with the capacity and the capex per line of the corridor
catalogue trade/pExtTransmissionCost.csv (life 30 years, as every other candidate):
trade/pNewTransmission_robg_freeexp.csv, lines between two internal zones
    Georgia to Romania        BSSC, 1 line of 1300 MW, 3500 MUSD, from 2031
    GEC_GE to Romania         GEC, up to 3 lines of 1261 MW, 2783 MUSD each (mean of the three
                              GEC links of the catalogue, 2637, 2974 and 2739), from 2032
    Trakia to Bulgaria        EWTC Turkiye to Bulgaria, 1 line of 775 MW, 250 MUSD, from 2033
    plus the 22 rows of trade/pNewTransmission_freeexp_all.csv (Caucasus and Turkish corridors)
trade/pNewTransmissionExt_robg_freeexp.csv, lines to external zones (new GAMS feature of the
same date, pNewTransmissionExt: full capex on the internal zone, the line adds to the
exogenous pExtTransferLimit when built)
    AzerbaijanMain to Kazakhstan   Trans-Caspian, 1 line of 1000 MW, 2000 MUSD, from 2035
    Trakia to Greece               EWTC Turkiye to Greece, 1 line of 775 MW, 250 MUSD, from 2034
trade/pTransferLimit_robg_freeexp.csv
    pTransferLimit_robg_geco.csv with the exogenous Georgia to Romania and GEC_GE to Romania
    capacities (both directions, 16 rows) set to 0: the cable and the GEC links are now built
    by the model or not. The hub feeders (GEC_GE to Georgia, GEC_AZ to AzerbaijanMain and
    GEC_AZ to GEC_GE) are kept as in the GECO world.

Not candidates: the Syria EEP link (Syria NTC is 0 in the whole RoBg family), the Reinforce60
rows of the catalogue and the EU borders of Romania and Bulgaria.

scenarios.csv: one row pNewTransmissionExt inserted after pNewTransmission (empty for every
existing column, so those scenarios read the header-only default of epm/resources) and the two
columns LC_FreeExp_RoBg and S5_FreeExp appended after the last column (rebuilt in place when
already present). Every other cell is left byte for byte. Idempotent: a second run writes the
same bytes and reports every file as unchanged. Pure csv module, no pandas.

Usage: python build_freeexp_robg.py [path to data_blacksea]
"""
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.normpath(
    os.path.join(HERE, "..", "..", "epm", "input", "data_blacksea"))

LF = chr(10)
CRLF = chr(13) + chr(10)
BOM = bytes([0xEF, 0xBB, 0xBF])

HEADER = ["From", "To", "EarliestEntry", "MaximumNumOfLines", "CapacityPerLine", "CostPerLine", "Life", "Status"]
GEC_LINK_COSTS = (2637, 2974, 2739)
# From, To, EarliestEntry, MaximumNumOfLines, CapacityPerLine, CostPerLine
INTERNAL = [
    ("Georgia", "Romania", 2031, 1, 1300.0, 3500),
    ("GEC_GE", "Romania", 2032, 3, 1261.0, round(sum(GEC_LINK_COSTS) / 3.0)),
    ("Trakia", "Bulgaria", 2033, 1, 775.0, 250),
]
EXTERNAL = [
    ("AzerbaijanMain", "Kazakhstan", 2035, 1, 1000.0, 2000),
    ("Trakia", "Greece", 2034, 1, 775.0, 250),
]
ZEROED = {("Georgia", "Romania"), ("Romania", "Georgia"), ("GEC_GE", "Romania"), ("Romania", "GEC_GE")}

NEW_FILES = {
    "pTransferLimit": "trade/pTransferLimit_robg_freeexp.csv",
    "pNewTransmission": "trade/pNewTransmission_robg_freeexp.csv",
    "pNewTransmissionExt": "trade/pNewTransmissionExt_robg_freeexp.csv",
}
# (new column, column copied cell by cell before NEW_FILES is applied), in the intended queue order
COLUMNS = [("LC_FreeExp_RoBg", "LC_GECO_RoBg"), ("S5_FreeExp", "S5_GECO")]


def num(v, nd=6):
    """Decimal text without trailing zeros: 1500.0 -> 1500, 166.65 -> 166.65."""
    s = ("%." + str(nd) + "f") % v
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


class Table(object):
    """A csv file kept line by line: lines that are not modified are written back byte for byte."""

    def __init__(self, path=None):
        self.bom, self.eol, self.lines, self.rows, self.touched = False, LF, [], [], set()
        if path is None:
            return
        raw = open(path, "rb").read()
        self.bom = raw.startswith(BOM)
        text = raw.decode("utf-8-sig")
        self.eol = CRLF if CRLF in text else LF
        assert text.endswith(self.eol), path
        self.lines = text[:-len(self.eol)].split(self.eol)
        self.rows = [next(csv.reader([ln])) if ln else [] for ln in self.lines]

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

    def insert(self, i, cells):
        self.rows.insert(i, list(cells))
        self.lines.insert(i, None)
        self.touched = {k + 1 if k >= i else k for k in self.touched}
        self.touched.add(i)

    def append_column(self, name, cells):
        """cells: one value per data row; the row serialisation is checked to stay byte for byte."""
        assert len(cells) == len(self.rows) - 1, (len(cells), len(self.rows))
        if name in self.header:
            j = self.header.index(name)
            for i, v in enumerate(cells, 1):
                self.set(i, j, v)
            return
        for i in range(len(self.rows)):
            if self.lines[i] is not None and i not in self.touched:
                assert serialise(self.rows[i]) == self.lines[i], ("row %d would not survive re-serialisation" % i, self.lines[i][:80])
            self.rows[i].append(name if i == 0 else cells[i - 1])
            self.touched.add(i)

    def line(self, i):
        if i in self.touched or self.lines[i] is None:
            return serialise(self.rows[i])
        return self.lines[i]

    def to_bytes(self):
        text = self.eol.join(self.line(i) for i in range(len(self.rows))) + self.eol
        return (BOM if self.bom else b"") + text.encode("utf-8")


def serialise(row):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(row)
    return buf.getvalue()


def write_bytes(rel, data):
    path = os.path.join(ROOT, rel)
    old = open(path, "rb").read() if os.path.exists(path) else None
    if old != data:
        open(path, "wb").write(data)
    print("  %-44s %-9s %7d bytes" % (rel, "unchanged" if old == data else ("rewritten" if old is not None else "new"), len(data)))


def write_table(rel, table):
    write_bytes(rel, table.to_bytes())


def catalogue_numbers():
    """Every numeric cell of the corridor catalogue, by (From, To): the candidate files must quote it."""
    with open(os.path.join(ROOT, "trade", "pExtTransmissionCost.csv"), encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][:2] == ["z", "zext"], rows[0][:3]
    out = {}
    for r in rows[1:]:
        if not r:
            continue
        vals = out.setdefault((r[0], r[1]), set())
        for c in r[2:]:
            try:
                vals.add(float(c))
            except ValueError:
                pass
    return out


def check_against_catalogue(cat):
    for frm, to, _, _, cap, cost in INTERNAL + EXTERNAL:
        vals = cat[(frm, to)]
        assert cap in vals, (frm, to, cap, sorted(vals))
        if (frm, to) == ("GEC_GE", "Romania"):
            assert all(c in vals for c in GEC_LINK_COSTS), sorted(vals)
        else:
            assert cost in vals, (frm, to, cost, sorted(vals))


# ------------------------------------------------------------------ transfer limit
def build_transfer_limit():
    geco = Table(os.path.join(ROOT, "trade", "pTransferLimit_robg_geco.csv"))
    assert geco.header[:3] == ["z", "z2", "q"], geco.header[:5]
    free = geco.copy()
    n = 0
    for i, r in enumerate(free.rows[1:], 1):
        if (r[0], r[1]) in ZEROED:
            for j in range(3, len(r)):
                free.set(i, j, "0.0")
            n += 1
    assert n == 16, n
    print("pTransferLimit_robg_freeexp: %d rows set to 0 (%s), %d rows kept" % (
        n, ", ".join(sorted(a + " to " + b for a, b in ZEROED)), len(free.rows) - 1 - n))
    write_table(NEW_FILES["pTransferLimit"], free)


# ------------------------------------------------------------------ candidate lines
def line_row(frm, to, entry, nlines, cap, cost):
    return [frm, to, str(entry), str(nlines), "%.1f" % cap, str(cost), "30", "3"]


def build_candidates():
    cat = catalogue_numbers()
    check_against_catalogue(cat)
    base = Table(os.path.join(ROOT, "trade", "pNewTransmission_freeexp_all.csv"))
    assert base.header == HEADER, base.header
    assert len(base.rows) == 23, len(base.rows)
    pairs = {(r[0], r[1]) for r in base.rows[1:]}

    internal = Table()
    internal.append(HEADER)
    for spec in INTERNAL:
        assert (spec[0], spec[1]) not in pairs and (spec[1], spec[0]) not in pairs, spec
        internal.append(line_row(*spec))
    for r in base.rows[1:]:
        internal.append(r)
    print("pNewTransmission_robg_freeexp: %d candidate rows, %d corridor rows added to the %d of pNewTransmission_freeexp_all" % (
        len(internal.rows) - 1, len(INTERNAL), len(base.rows) - 1))
    write_table(NEW_FILES["pNewTransmission"], internal)

    zext = {r[0] for r in Table(os.path.join(ROOT, "trade", "zext_robg.csv")).rows[1:]}
    zcmap = {r[0] for r in Table(os.path.join(ROOT, "zcmap_robg_geco.csv")).rows[1:]}
    external = Table()
    external.append(HEADER)
    for spec in EXTERNAL:
        assert spec[0] in zcmap and spec[1] in zext, spec
        external.append(line_row(*spec))
    for frm, to, *_ in INTERNAL:
        assert frm in zcmap and to in zcmap, (frm, to)
    print("pNewTransmissionExt_robg_freeexp: %d rows to external zones" % (len(external.rows) - 1))
    write_table(NEW_FILES["pNewTransmissionExt"], external)


# ------------------------------------------------------------------ scenarios.csv
def build_scenarios():
    sc = Table(os.path.join(ROOT, "scenarios.csv"))
    H = sc.header
    assert H[0] == "paramNames", H[:3]
    names = [r[0] for r in sc.rows[1:]]
    if "pNewTransmissionExt" not in names:
        i = names.index("pNewTransmission") + 1
        sc.insert(i + 1, ["pNewTransmissionExt"] + [""] * (len(H) - 1))
        names = [r[0] for r in sc.rows[1:]]
        print("scenarios.csv: row pNewTransmissionExt inserted after pNewTransmission (row %d), empty for the %d existing columns" % (i + 1, len(H) - 1))
    assert names.index("pNewTransmissionExt") == names.index("pNewTransmission") + 1
    assert all(sc.rows[names.index("pNewTransmissionExt") + 1][j] == "" for j in range(1, len(H)) if H[j] not in dict(COLUMNS))

    for new, src in COLUMNS:
        j = H.index(src)
        cells = [r[j] for r in sc.rows[1:]]
        assert cells[names.index("zcmap")] == "zcmap_robg_geco.csv", (src, cells[names.index("zcmap")])
        assert cells[names.index("pTransferLimit")] == "trade/pTransferLimit_robg_geco.csv", src
        for row, path in NEW_FILES.items():
            cells[names.index(row)] = path
        sc.append_column(new, cells)
        changed = [names[k] for k in range(len(names)) if cells[k] != sc.rows[k + 1][j]]
        print("scenarios.csv: %-16s = %-13s with %s" % (new, src, ", ".join(changed)))
    print("queue order: " + " ".join(sc.header[-len(COLUMNS):]))
    write_table("scenarios.csv", sc)


if __name__ == "__main__":
    print("data folder: " + ROOT)
    build_transfer_limit()
    build_candidates()
    build_scenarios()
