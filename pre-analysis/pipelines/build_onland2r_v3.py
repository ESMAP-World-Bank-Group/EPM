# -*- coding: utf-8 -*-
"""Onland corridor with both routes (S5_Onland2R_v3): S5_Onland_v3 as it stands (Georgia route,
Georgia to EastAna 2,700 MW from 2031 with the EastAna to NorthWest and NorthWest to Bulgaria
segments at 2,000 MW, plus the committed CenterAna to Trakia HVDC) plus the second route,
Nakhchivan to EastAna forced at 1,000 MW in both directions from 2031. Every output is a NEW file
or a NEW scenarios.csv column; nothing read by an existing scenario changes.

File
    trade/pTransferLimit_robg_onland2r_hvdc.csv = trade/pTransferLimit_robg_onland_hvdc.csv with
    the eight Nakhchivan to EastAna rows (both directions, Q1 to Q4) set from 0 to 1,000 MW from
    2031, the column where Georgia to EastAna switches from 700 to 2,700 MW. One line of the
    catalogue TRIPP candidate (1,000 MW, 1,450 MUSD, life 30 years in trade/pNewTransmission.csv),
    forced instead of chosen: its capex is outside the objective and enters the net benefit ex
    post (onland capex plus 1,450 MUSD). AzerbaijanMain to Nakhchivan stays the committed 800 MW
    of trade/pNewTransmission_none.csv. The loss factors of the pair are already 0.025 in
    trade/pLossFactorInternal_robg_onland_hvdc.csv, which is reused.

Column (1)
    S5_Onland2R_v3 = S5_Onland_v3 with the new transfer limit file; every other cell identical.

Usage: python build_onland2r_v3.py [path to data_blacksea]. Idempotent.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_freeexp_robg import Table  # noqa: E402  (byte preserving csv table, same folder)
import build_costs_v3_robg as v3  # noqa: E402  (ROOT, YEARS, path, write_table, truncate_columns)

TL_SRC = "trade/pTransferLimit_robg_onland_hvdc.csv"
TL_NEW = "trade/pTransferLimit_robg_onland2r_hvdc.csv"
LOSS = "trade/pLossFactorInternal_robg_onland_hvdc.csv"
PAIR = ("Nakhchivan", "EastAna")
MW, FROM = 1000.0, 2031
LOSS_VALUE = "0.025"
GEORGIA_PAIR, GEORGIA_BEFORE, GEORGIA_AFTER = ("Georgia", "EastAna"), 700.0, 2700.0
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
NEW, SRC = "S5_Onland2R_v3", "S5_Onland_v3"


def is_pair(r, pair):
    return {r[0], r[1]} == set(pair)


def build_transfer_limit():
    t = Table(v3.path(TL_SRC))
    assert t.header == ["z", "z2", "q"] + v3.YEARS, t.header[:4]
    from_j = 3 + v3.YEARS.index(str(FROM))
    georgia = [r for r in t.rows[1:] if is_pair(r, GEORGIA_PAIR)]
    assert len(georgia) == 8, len(georgia)
    for r in georgia:  # the second route switches on in the same column as the Georgian route
        assert float(r[from_j - 1]) == GEORGIA_BEFORE and float(r[from_j]) == GEORGIA_AFTER, r[:4]
    touched = []
    for i, r in enumerate(t.rows[1:], 1):
        if is_pair(r, PAIR):
            assert r[2] in QUARTERS and all(float(v) == 0.0 for v in r[3:]), r[:5]
            for j in range(from_j, len(r)):
                t.set(i, j, "%.1f" % MW)
            touched.append(i)
    assert len(touched) == 8, touched
    v3.write_table(TL_NEW, t)
    src, new = Table(v3.path(TL_SRC)), Table(v3.path(TL_NEW))
    diff = [i for i in range(len(src.rows)) if src.rows[i] != new.rows[i]]
    assert len(src.rows) == len(new.rows) and diff == touched, diff
    for i in diff:
        assert new.rows[i][:3] == src.rows[i][:3] and new.rows[i][3:from_j] == src.rows[i][3:from_j], i
        assert set(new.rows[i][from_j:]) == {"%.1f" % MW}, i
    print("  %s to %s %.0f MW both ways from %d (8 rows), the other %d rows byte for byte" % (PAIR + (MW, FROM, len(src.rows) - 1 - 8)))


def check_loss():
    t = Table(v3.path(LOSS))
    rows = [r for r in t.rows[1:] if is_pair(r, PAIR)]
    assert len(rows) == 2 and all(set(r[2:]) == {LOSS_VALUE} for r in rows), rows
    print("  losses %s both ways already in %s, reused" % (LOSS_VALUE, LOSS))


def build_scenarios():
    sc = Table(v3.path("scenarios.csv"))
    H = sc.header
    assert H[0] == "paramNames", H[:3]
    names = [r[0] for r in sc.rows[1:]]
    assert NEW not in H or H[-1] == NEW, "the column exists but is not the last one"
    orig = [c for c in H if c != NEW]
    orig_rows = [list(r[:len(orig)]) for r in sc.rows]
    sc = v3.truncate_columns(sc, len(orig))
    j = orig.index(SRC)
    cells = [r[j] for r in sc.rows[1:]]
    k = names.index("pTransferLimit")
    assert cells[k] == TL_SRC, (SRC, cells[k])
    assert cells[names.index("pLossFactorInternal")] == LOSS, cells[names.index("pLossFactorInternal")]
    cells[k] = TL_NEW
    for v in cells:
        if v.endswith(".csv") or v.endswith(".opt"):
            assert os.path.exists(v3.path(v)), (NEW, v)
    sc.append_column(NEW, cells)
    v3.write_table("scenarios.csv", sc)
    back = Table(v3.path("scenarios.csv"))
    assert [r[:len(orig)] for r in back.rows] == orig_rows, "an existing column changed"
    assert back.header[-1] == NEW and [r[-1] for r in back.rows[1:]] == cells, NEW
    changed = [names[i] for i, (a, b) in enumerate(zip([r[j] for r in back.rows[1:]], cells)) if a != b]
    assert changed == ["pTransferLimit"], changed
    print("scenarios.csv: %d existing columns kept byte for byte, 1 column added: %s = %s with pTransferLimit = %s" % (len(orig), NEW, SRC, TL_NEW))


if __name__ == "__main__":
    print("data folder: " + v3.ROOT)
    build_transfer_limit()
    check_loss()
    build_scenarios()
