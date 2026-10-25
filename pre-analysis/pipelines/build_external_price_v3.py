# -*- coding: utf-8 -*-
"""External family v3, EU border price variants: the three external v3 columns (S5_Ext_Base_v3,
S5_Ext_BSSC_v3, S5_Ext_GECO_v3 of build_external_v3.py) at the EU very low border price and at
the EU central border price with CBAM on exports, so that the net benefit by EU price scenario
chart (status brief, slide on the submarine projects) can be rebuilt on the five-year grid with
the v3 reference data instead of the 13 September runs of the LC_*_VeryLow and LC_*_CBAM columns.
Six NEW scenarios.csv columns and no new data file; nothing read by an existing scenario changes.

Columns (6), P in Base, BSSC, GECO
    S5_Ext_<P>_VeryLow_v3 = S5_Ext_<P>_v3 with pTradePrice trade/pTradePrice_eu_very_low.csv
        and pTradePriceExport trade/pTradePriceExport_eu_very_low.csv, the cells of LC_<P>_VeryLow
    S5_Ext_<P>_CBAM_v3 = S5_Ext_<P>_v3 with pTradePrice trade/pTradePrice_eu_central.csv and
        pTradePriceExport trade/pTradePriceExport_eu_central_cbam.csv, the cells of LC_<P>_CBAM
    The source columns leave both price parameters at the config.csv default (EU central).

Usage: python build_external_price_v3.py [path to data_blacksea]. Idempotent: appends the six
columns when none of them exists, checks them when all six exist.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_freeexp_robg import Table  # noqa: E402  (byte preserving csv table, same folder)
import build_costs_v3_robg as v3  # noqa: E402  (ROOT convention, path and write_table)

PRICE_PARAMS = ("pTradePrice", "pTradePriceExport")
# suffix of the new column: (LC twin suffix, pTradePrice file, pTradePriceExport file)
VARIANTS = {
    "VeryLow": ("_VeryLow", "trade/pTradePrice_eu_very_low.csv", "trade/pTradePriceExport_eu_very_low.csv"),
    "CBAM": ("_CBAM", "trade/pTradePrice_eu_central.csv", "trade/pTradePriceExport_eu_central_cbam.csv"),
}
PROJECTS = [("Base", "LC_Base"), ("BSSC", "LC_BSSC"), ("GECO", "LC_GECO")]


def family():
    """(new column, source column, LC twin column, {price parameter: file})."""
    out = []
    for variant, (lc_suffix, buy, sell) in VARIANTS.items():
        for project, lc in PROJECTS:
            out.append(("S5_Ext_%s_%s_v3" % (project, variant), "S5_Ext_%s_v3" % project, lc + lc_suffix,
                        dict(zip(PRICE_PARAMS, (buy, sell)))))
    return out


def build_scenarios():
    sc = Table(v3.path("scenarios.csv"))
    H = sc.header
    assert H[0] == "paramNames", H[:3]
    names = [r[0] for r in sc.rows[1:]]
    fam = family()
    new_names = [new for new, _, _, _ in fam]
    existing = [c for c in new_names if c in H]
    assert existing in ([], new_names), existing
    n_orig = len(H) - len(existing)
    orig_rows = [list(r[:n_orig]) for r in sc.rows]
    built = {}
    for new, src, twin, prices in fam:
        cells = [r[H.index(src)] for r in sc.rows[1:]]
        twin_cells = [r[H.index(twin)] for r in sc.rows[1:]]
        for param, value in prices.items():
            k = names.index(param)
            assert cells[k] == "", (new, param, cells[k])          # the source is at the config.csv default
            assert twin_cells[k] == value, (new, twin, param, twin_cells[k])   # same files as the LC twin
            assert os.path.exists(v3.path(value)), (new, value)
            cells[k] = value
        built[new] = cells
    if existing:
        for new, cells in built.items():
            k = H.index(new)
            assert [r[k] for r in sc.rows[1:]] == cells, new
        print("scenarios.csv: the %d external price v3 columns are already in place and checked" % len(built))
        return
    for new, cells in built.items():
        sc.append_column(new, cells)
    v3.write_table("scenarios.csv", sc)
    back = Table(v3.path("scenarios.csv"))
    assert [r[:n_orig] for r in back.rows] == orig_rows, "an existing column changed"
    assert back.header[n_orig:] == list(built), back.header[n_orig:]
    for new, cells in built.items():
        k = back.header.index(new)
        assert [r[k] for r in back.rows[1:]] == cells, new
    print("scenarios.csv: %d existing columns kept, %d external price v3 columns appended" % (n_orig, len(built)))
    print("external price v3 columns: " + " ".join(built))


if __name__ == "__main__":
    print("data folder: " + v3.ROOT)
    build_scenarios()
