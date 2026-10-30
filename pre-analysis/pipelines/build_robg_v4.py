# -*- coding: utf-8 -*-
"""RoBg baseline v4: S5_Base_v3 with the Romania and Bulgaria corrections raised in the review
of the national plan slides (2026-10-29). Every output is a NEW file or a NEW scenarios.csv
column; nothing read by an existing scenario changes.

Generators, supply/pGenDataInput_v4_s5.csv = supply/pGenDataInput_v2_s5.csv with
    Romania gas
        The six CHP extensions (TPP Progresu_Ext, Grozavesti_Ext, CET Govora_Ext, TPP Bucuresti
        Vest_Ext, TPP Bucuresti Sud_Ext, TPP Midia_Navodari_Ext, 932.5 MW) and Mintia CCGT
        phase 2 (600 MW) move from committed (2) to candidate (3). They are life extensions
        and a second phase without a firm date, not announced projects; committed, they put
        the gas fleet above the NECP. Mintia phase 1, Iernut, Isalnita, Craiova and Turceni
        stay committed.
    Doicesti SMR
        Earliest entry 2030 -> 2033 (first module in the SNN investment strategy; on the
        five-year grid the first possible build year is 2036). Heat rate 3.82 -> 11.3
        MMBtu/MWh, the value of the Kozloduy new build in the same file; 3.82 was an input
        error of the source workbook.
    Kozloduy 7 and 8
        1,000 -> 1,200 MW each (capacity and unit size) and unit 8 2040 -> 2036, following the
        Bulgarian review (operation 2035 to 2037, 1.2 GW per unit). 2037 is not a year of the
        five-year grid and a committed plant is only built at y = StYr, so both enter in 2036.

External trade cap, trade/pMaxAnnualExternalTradeShare_robg_cap20.csv
    = trade/pMaxAnnualExternalTradeShare_robg.csv with Romania and Bulgaria at 0.2 in every
    year (3 elsewhere, not binding). The constraint caps GROSS external exports and GROSS
    external imports, each as a share of demand (base.gms eMaxAnnualExportShareEnergy,
    eMaxAnnualImportShareEnergy). 0.2 is the 2025 level of S5_Base_v3 (Bulgaria exports 0.18
    and imports 0.19, Romania imports 0.16), so the cap leaves the base year untouched and
    binds on the export growth of 2030 to 2040 (S5_Base_v3 2040: Romania 0.28, Bulgaria 0.36).

Storage, supply/pStorageDataInput_s5.csv = supply/pStorageDataInput.csv with
    Bulgaria_Battery_2026 (4,024 MW), Bulgaria_Chaira_U3, Bulgaria_Chaira_U1 and
    Romania_Battery_2026 (356 MW) at StYr 2030 instead of 2026 / 2028. A committed plant
    is built only in the year y = StYr; 2026 and 2028 are not years of the five-year grid
    (y_s5.csv) and input_treatment.merge_storage_into_gendata keeps the storage file's
    StYr over the generator file's, so with the default storage file these four plants
    were never built in any S5 run (2026-10-30 audit). The generator file already carries
    2030 for the same rows.

Column (1)
    S5_Base_v4 = S5_Base_v3 with the three files above.

Usage: python build_robg_v4.py [path to data_blacksea]. Idempotent.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_freeexp_robg import Table  # noqa: E402  (byte preserving csv table, same folder)
import build_costs_v3_robg as v3  # noqa: E402  (same ROOT convention and writers)

ROOT = v3.ROOT

GEN_SRC, GEN_V4 = "supply/pGenDataInput_v2_s5.csv", "supply/pGenDataInput_v4_s5.csv"
CAP_SRC, CAP_V4 = "trade/pMaxAnnualExternalTradeShare_robg.csv", "trade/pMaxAnnualExternalTradeShare_robg_cap20.csv"
CAP_COUNTRIES, CAP_VALUE = ("Romania", "Bulgaria"), "0.2"
STO_SRC, STO_S5 = "supply/pStorageDataInput.csv", "supply/pStorageDataInput_s5.csv"
# storage unit: {column: (expected value in the source, new value)}
STO_EDITS = {
    "Bulgaria_Battery_2026": {"StYr": ("2026", "2030")},
    "Bulgaria_Chaira_U3": {"StYr": ("2026", "2030")},
    "Bulgaria_Chaira_U1": {"StYr": ("2028", "2030")},
    "Romania_Battery_2026": {"StYr": ("2026", "2030")},
}

TO_CANDIDATE = [
    "TPP Progresu_Ext", "Grozăvești_Ext", "CET Govora_Ext", "TPP Bucuresti Vest_Ext",
    "TPP Bucuresti Sud_Ext", "TPP Midia_Navodari_Ext", "Mintia CCGT phase 2",
]
# generator: {column: (expected value in the source, new value)}
EDITS = {
    "Doicești": {"StYr": ("2030", "2033"), "HeatRate": ("3.82303887", "11.3")},
    "Bulgaria_Kozloduy_7": {"Capacity": ("1000", "1200"), "UnitSize": ("1000", "1200")},
    "Bulgaria_Kozloduy_8": {"StYr": ("2040", "2036"), "Capacity": ("1000", "1200"), "UnitSize": ("1000", "1200")},
}
for g in TO_CANDIDATE:
    EDITS[g] = {"Status": ("2", "3")}

SOURCE_COL, NEW_COL = "S5_Base_v3", "S5_Base_v4"
# parameter: (expected cell in the source column, value in the new column)
CELLS = {
    "pGenDataInput": (GEN_SRC, GEN_V4),
    "pMaxAnnualExternalTradeShare": (CAP_SRC, CAP_V4),
    "pStorageDataInput": ("", STO_S5),
}


def edit_rows(src, dst, edits, what):
    """Copy src to dst with the cells of `edits` ({row name: {column: (expected, new)}}) changed."""
    t = Table(v3.path(src))
    H = t.header
    names = [r[0] if r else "" for r in t.rows]
    for g, cols in edits.items():
        assert names.count(g) == 1, (g, names.count(g))
        i = names.index(g)
        for col, (expect, value) in cols.items():
            j = H.index(col)
            assert t.rows[i][j] == expect, (g, col, t.rows[i][j])
            t.set(i, j, value)
    v3.write_table(dst, t)
    back = Table(v3.path(dst))
    orig = Table(v3.path(src))
    changed = [back.rows[i][0] for i in range(len(back.rows)) if back.rows[i] != orig.rows[i]]
    assert len(back.rows) == len(orig.rows) and sorted(changed) == sorted(edits), changed
    print("  %s: %d %s edited, every other row byte for byte" % (dst, len(changed), what))


def build_gen():
    edit_rows(GEN_SRC, GEN_V4, EDITS, "generators")


def build_storage():
    edit_rows(STO_SRC, STO_S5, STO_EDITS, "storage units")


def build_cap():
    t = Table(v3.path(CAP_SRC))
    H = t.header
    assert H[0] == "year", H
    for c in CAP_COUNTRIES:
        j = H.index(c)
        for i in range(1, len(t.rows)):
            assert t.rows[i][j] == "3", (c, t.rows[i])
            t.set(i, j, CAP_VALUE)
    v3.write_table(CAP_V4, t)
    print("  %s: %s at %s in %d years, other countries unchanged" % (CAP_V4, " and ".join(CAP_COUNTRIES), CAP_VALUE, len(t.rows) - 1))


def build_scenarios():
    sc = Table(v3.path("scenarios.csv"))
    H = sc.header
    assert H[0] == "paramNames", H[:3]
    names = [r[0] for r in sc.rows[1:]]
    assert H.count(NEW_COL) <= 1 and (NEW_COL not in H or H[-1] == NEW_COL), "the v4 column is not the last column"
    orig = [c for c in H if c != NEW_COL]
    orig_rows = [list(r[:len(orig)]) for r in sc.rows]
    sc = v3.truncate_columns(sc, len(orig))
    j = orig.index(SOURCE_COL)
    cells = [r[j] for r in sc.rows[1:]]
    for param, (expect, value) in CELLS.items():
        assert cells[names.index(param)] == expect, (param, cells[names.index(param)])
        cells[names.index(param)] = value
    for v in cells:
        if v.endswith(".csv") or v.endswith(".opt"):
            assert os.path.exists(v3.path(v)), v
    sc.append_column(NEW_COL, cells)
    v3.write_table("scenarios.csv", sc)
    back = Table(v3.path("scenarios.csv"))
    assert [r[:len(orig)] for r in back.rows] == orig_rows, "an existing column changed"
    assert back.header[-1] == NEW_COL and [r[-1] for r in back.rows[1:]] == cells
    print("scenarios.csv: %d existing columns kept byte for byte, column %s added" % (len(orig), NEW_COL))


if __name__ == "__main__":
    print("data folder: " + ROOT)
    build_gen()
    build_cap()
    build_storage()
    build_scenarios()
