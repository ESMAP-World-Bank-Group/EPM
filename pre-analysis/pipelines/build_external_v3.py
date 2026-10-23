# -*- coding: utf-8 -*-
"""External family v3: the external representation (Romania, Bulgaria and the other
neighbours as external price zones of trade/zext.csv, imports and exports at the EU central
border price) on the five-year grid, with the committed CenterAna to Trakia HVDC and the v2
Turkiye corrections, for the comparison with the S5 RoBg family (S5_Base_v3, S5_BSSC_v3,
S5_GECO_v3) on the same grid. Every output is a NEW file or a NEW scenarios.csv column;
nothing read by an existing scenario changes.

Committed HVDC
    trade/pTransferLimit_hvdc.csv and trade/pLossFactorInternal_hvdc.csv = the config.csv
    default files (the 14 zone network of the external family) plus CenterAna to Trakia,
    4,000 MW in both directions from 2032, losses 0.03: the same overlay as the _hvdc twins
    of the RoBg files (build_costs_v3_robg.py, whose functions are reused).

Columns (3)
    S5_Ext_Base_v3, S5_Ext_BSSC_v3, S5_Ext_GECO_v3 = LC_Baseline, LC_BSSC, LC_GECO (their
    pExtTransferLimit file and, for GECO, zcmap_geco.csv) with cplex/cplex_s5.opt, y_s5.csv,
    supply/pGenDataInput_v2_s5.csv, supply/pCapexTrajectoriesCustom_v2.csv,
    load/pDemandForecast_v2.csv (the files of the S5 _v3 columns: the Romania and Bulgaria
    rows are outside the zone map and ignored by the model) and the two _hvdc twins. Every
    other parameter stays at its config.csv default (EU central border prices, zext.csv,
    pTransferLimit candidates none).

Usage: python build_external_v3.py [path to data_blacksea]. Idempotent.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_freeexp_robg import Table  # noqa: E402  (byte preserving csv table, same folder)
import build_costs_v3_robg as v3  # noqa: E402  (committed HVDC overlay, same ROOT convention)

ROOT = v3.ROOT

TL_DEFAULT, LOSS_DEFAULT = "trade/pTransferLimit.csv", "trade/pLossFactorInternal.csv"
TL_HVDC, LOSS_HVDC = v3.hvdc_name(TL_DEFAULT), v3.hvdc_name(LOSS_DEFAULT)

# parameter: (expected cell in the source column, value in the new column)
S5_CELLS = {
    "cplexfile": ("", "cplex/cplex_s5.opt"),
    "y": ("", "y_s5.csv"),
    "pDemandForecast": ("", "load/pDemandForecast_v2.csv"),
    "pGenDataInput": ("", "supply/pGenDataInput_v2_s5.csv"),
    "pCapexTrajectories": ("", "supply/pCapexTrajectoriesCustom_v2.csv"),
    "pTransferLimit": ("", TL_HVDC),
    "pLossFactorInternal": ("", LOSS_HVDC),
}
# new column, source column, {parameter: expected cell in the source column}
FAMILY = [
    ("S5_Ext_Base_v3", "LC_Baseline", {"zcmap": "", "pExtTransferLimit": "trade/pExtTransferLimit_baseline.csv"}),
    ("S5_Ext_BSSC_v3", "LC_BSSC", {"zcmap": "", "pExtTransferLimit": "trade/pExtTransferLimit_bssc.csv"}),
    ("S5_Ext_GECO_v3", "LC_GECO", {"zcmap": "zcmap_geco.csv", "pExtTransferLimit": "trade/pExtTransferLimit_geco.csv"}),
]


def config_defaults():
    rows = list(csv.reader(open(v3.path("config.csv"), encoding="utf-8-sig")))  # no trailing newline, read with csv
    assert rows[0] == ["metadata", "paramNames", "file"], rows[0]
    return {r[1]: r[2] for r in rows[1:] if len(r) == 3 and r[1]}


def build_hvdc_twins():
    cfg = config_defaults()
    assert cfg["pTransferLimit"] == TL_DEFAULT and cfg["pLossFactorInternal"] == LOSS_DEFAULT, (cfg["pTransferLimit"], cfg["pLossFactorInternal"])
    v3.build_hvdc_transfer_limit(TL_DEFAULT)
    v3.build_hvdc_loss(LOSS_DEFAULT)
    for rel, n_new in ((TL_HVDC, 8), (LOSS_HVDC, 2)):
        base, twin = Table(v3.path(rel[:-len(v3.HVDC_SUFFIX) - 4] + ".csv")), Table(v3.path(rel))
        assert twin.rows[:len(base.rows)] == base.rows and len(twin.rows) == len(base.rows) + n_new, rel
        assert all({r[0], r[1]} == set(v3.HVDC_PAIR) for r in twin.rows[len(base.rows):]), rel
    print("  committed HVDC %s to %s, %.0f MW from %d, on the default network files" % (v3.HVDC_PAIR + (v3.HVDC_MW, v3.HVDC_FROM)))


def build_scenarios():
    sc = Table(v3.path("scenarios.csv"))
    H = sc.header
    assert H[0] == "paramNames", H[:3]
    names = [r[0] for r in sc.rows[1:]]
    new_names = [new for new, _, _ in FAMILY]
    existing = [c for c in new_names if c in H]
    assert existing in ([], new_names), existing
    orig = [c for c in H if c not in new_names]
    assert orig == H[:len(orig)], "the external v3 columns are not the last columns"
    orig_rows = [list(r[:len(orig)]) for r in sc.rows]
    sc = v3.truncate_columns(sc, len(orig))
    built = {}
    for new, src, expected in FAMILY:
        j = orig.index(src)
        cells = [r[j] for r in sc.rows[1:]]
        for param, expect in expected.items():
            assert cells[names.index(param)] == expect, (new, param, cells[names.index(param)])
        for param, (expect, value) in S5_CELLS.items():
            assert cells[names.index(param)] == expect, (new, param, cells[names.index(param)])
            cells[names.index(param)] = value
        others = [names[k] for k, v in enumerate(cells) if v and names[k] not in S5_CELLS and names[k] not in expected]
        assert not others, (new, others)
        for v in cells:
            if v.endswith(".csv") or v.endswith(".opt"):
                assert os.path.exists(v3.path(v)), (new, v)
        built[new] = cells
    for new, cells in built.items():
        sc.append_column(new, cells)
    v3.write_table("scenarios.csv", sc)
    back = Table(v3.path("scenarios.csv"))
    assert [r[:len(orig)] for r in back.rows] == orig_rows, "an existing column changed"
    assert back.header[len(orig):] == list(built), back.header[len(orig):]
    for new, cells in built.items():
        k = back.header.index(new)
        assert [r[k] for r in back.rows[1:]] == cells, new
    print("scenarios.csv: %d existing columns kept byte for byte, %d external v3 columns" % (len(orig), len(built)))
    print("external v3 columns: " + " ".join(built))


if __name__ == "__main__":
    print("data folder: " + ROOT)
    build_hvdc_twins()
    build_scenarios()
