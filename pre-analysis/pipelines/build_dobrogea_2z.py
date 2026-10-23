# -*- coding: utf-8 -*-
"""Dobrogea two-zone family: Romania split into Dobrogea (the Transelectrica section S6:
Constanta, Tulcea, Galati, Braila, Calarasi and Ialomita counties, where the Cernavoda units,
the named Dobrogea wind farms, the offshore candidates and the BSSC and GEC landings sit) and
the rest of Romania, joined by one link capped at the admissible S6 export. Six scenarios.csv
columns, the Dobrogea sensitivity of the S5 v3 family. Every output is a NEW file or a NEW
scenarios.csv column; nothing read by an existing scenario changes.

Zone split (suffix _2z on every zone indexed file read by the S5 v3 trio)
    zcmap_robg_2z.csv, zcmap_robg_geco_2z.csv: Dobrogea, country Romania (country sums and the
    external trade by country stay comparable with the one zone runs).
    supply/pGenDataInput_v2_s5_2z.csv: the five Cernavoda rows, the 22 named S6 wind farms
    (3,163 MW) and the two generic offshore candidates move to Dobrogea; every other Romanian
    row (Doicesti, the unlocated and generic onshore wind, PV, thermal, hydro, storage) stays
    in the rest of Romania, so new onshore wind is sited outside S6 unless named there.
    load/pDemandForecast_v2_2z.csv: 12 percent of the Romanian energy and peak in Dobrogea
    (population share plus the Galati steel works and the Constanta port, a proxy), the rest
    of Romania keeps the complement; the sum is unchanged to the cent.
    load/pDemandProfile_2z.csv, supply/pVREProfile_2z.csv, supply/pGenDataInputDefault_2z.csv,
    supply/pAvailabilityDefault_2z.csv, supply/pCapexTrajectoriesDefault_2z.csv: the Romanian
    rows copied as Dobrogea rows (same profiles and technology defaults).

Network (twins of the three _hvdc transfer limit and loss files of the S5 v3 trio)
    The Georgia (BSSC) and GEC_GE (GEC) rows to and from Romania land in Dobrogea; the
    Bulgaria rows and every external link (Hungary, Serbia, Moldova, Ukraine) stay on the rest
    of Romania: the TDP 2024 admissible export of section S6 counts the border lines inside the
    section, so routing them through Dobrogea to Romania consumes the cap as the limit does.
    Dobrogea to Romania and back, four quarters: _2z47 = 3,200 MW to 2029 then 4,700 MW from
    2030 (the plan projects of the TDP 2024 to 2033, about +1,490 MW); _2z32 = 3,200 MW every
    year (the grid of today, N-1 admissible export 3,270 summer and 3,150 winter). Loss 0.02.

Columns (6)
    S5_Base_2z47_v3, S5_BSSC_2z47_v3, S5_GECO_2z47_v3, S5_Base_2z32_v3, S5_BSSC_2z32_v3,
    S5_GECO_2z32_v3 = S5_Base_v3, S5_BSSC_v3, S5_GECO_v3 with the _2z files above.

Usage: python build_dobrogea_2z.py [path to data_blacksea]. Idempotent.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_freeexp_robg import Table  # noqa: E402  (byte preserving csv table, same folder)
from build_external_v3 import config_defaults  # noqa: E402  (config.csv defaults, same folder)
import build_costs_v3_robg as v3  # noqa: E402  (ROOT, path, write_table, truncate_columns)

ROOT = v3.ROOT
YEARS = v3.YEARS
DOB, RO = "Dobrogea", "Romania"
SUFFIX = "_2z"
LOAD_SHARE = 0.12
EXIT_TODAY_MW = 3200.0
EXIT_PLAN_MW = 4700.0
EXIT_PLAN_FROM = 2030
EXIT_LOSS = "0.02"
LANDING = ("Georgia", "GEC_GE")  # their lines to and from Romania land in Dobrogea
VARIANTS = [("2z47", EXIT_PLAN_MW), ("2z32", EXIT_TODAY_MW)]  # (suffix, cap from EXIT_PLAN_FROM; before: today)
CERNAVODA_PREFIX = "Cernavoda"
OFFSHORE_TECH = "OffshoreWind"
S6_WIND = ["Fântânele", "Cogealac", "Deleni", "Eolica Sǎcele", "Eolica Casimcea", "Văcăreni", "Verbund Casimcea",
           "Pechea", "Sabloal Valea Dacilor", "EDP Cernavodă", "Eolica Beidaud", "Eolica Baia", "Eolica Sarichioi",
           "Valea Nucarilor", "Salbatica", "EDP Pestera", "Agighiol", "Facaeni", "Crucea North", "Crucea Nord",
           "Gemenele Wind Farm", "Pantelimon"]
S6_WIND_MW = 3163.0

GEN = "supply/pGenDataInput_v2_s5.csv"
DEMAND = "load/pDemandForecast_v2.csv"
ZCMAPS = ["zcmap_robg.csv", "zcmap_robg_geco.csv"]
# parameter: config.csv default file, copied rows, header start
COPIED = {
    "pDemandProfile": ("load/pDemandProfile.csv", 28, ["zone", "season", "daytype"]),
    "pVREProfile": ("supply/pVREProfile.csv", 84, ["zone", "tech", "season", "daytype"]),
    "pGenDataInputDefault": ("supply/pGenDataInputDefault.csv", 25, ["z", "tech", "f"]),
    "pAvailabilityDefault": ("supply/pAvailabilityDefault.csv", 20, ["z", "tech", "fuel"]),
    "pCapexTrajectoriesDefault": ("supply/pCapexTrajectoriesDefault.csv", 27, ["zone", "tech", "fuel"]),
}
# source column: (transfer limit file, loss file, landing zones expected in them)
TRIO = {
    "S5_Base_v3": ("trade/pTransferLimit_robg_hvdc.csv", "trade/pLossFactorInternal_robg_hvdc.csv", ()),
    "S5_BSSC_v3": ("trade/pTransferLimit_robg_bssc_hvdc.csv", "trade/pLossFactorInternal_robg_bssc_hvdc.csv", ("Georgia",)),
    "S5_GECO_v3": ("trade/pTransferLimit_robg_geco_hvdc.csv", "trade/pLossFactorInternal_robg_geco_hvdc.csv", ("Georgia", "GEC_GE")),
}
NEW_COLUMNS = [("S5_%s_%s_v3" % (kind, suffix), "S5_%s_v3" % kind, suffix)
               for suffix, _ in VARIANTS for kind in ("Base", "BSSC", "GECO")]


def twin(rel, suffix=SUFFIX):
    assert rel.endswith(".csv") and not rel.endswith(suffix + ".csv"), rel
    return rel[:-4] + suffix + ".csv"


def rows_of(t, zone, j=0):
    return [i for i in range(1, len(t.rows)) if t.rows[i] and t.rows[i][j] == zone]


def copy_rows(t, j, ones):
    """Insert a Dobrogea copy of every Romanian row right after the last one; nothing else moves."""
    assert ones and not rows_of(t, DOB, j), (len(ones), rows_of(t, DOB, j)[:3])
    at = ones[-1] + 1
    for k, i in enumerate(ones):
        r = list(t.rows[i])
        r[j] = DOB
        t.insert(at + k, r)
    return len(ones)


def build_zcmaps():
    for rel in ZCMAPS:
        t = Table(v3.path(rel))
        assert t.header == ["z", "c"], (rel, t.header)
        ones = rows_of(t, RO)
        assert len(ones) == 1 and t.rows[ones[0]] == [RO, RO], (rel, ones)
        t.insert(ones[0] + 1, [DOB, RO])
        v3.write_table(twin(rel), t)


def build_generators():
    t = Table(v3.path(GEN))
    H = t.header
    g, z, tech = H.index("g"), H.index("z"), H.index("tech")
    assert not rows_of(t, DOB, z), "Dobrogea already in " + GEN
    moved = {"cernavoda": [], "wind": [], "offshore": []}
    for i in range(1, len(t.rows)):
        r = t.rows[i]
        if r[z] != RO:
            continue
        if r[g].startswith(CERNAVODA_PREFIX):
            moved["cernavoda"].append(i)
        elif r[g] in S6_WIND:
            moved["wind"].append(i)
        elif r[tech] == OFFSHORE_TECH:
            moved["offshore"].append(i)
    assert len(moved["cernavoda"]) == 5 and len(moved["offshore"]) == 2, moved
    assert sorted(t.rows[i][g] for i in moved["wind"]) == sorted(S6_WIND), "S6 wind rows missing"
    st, cap = H.index("Status"), H.index("Capacity")
    mw = sum(float(t.rows[i][cap]) for i in moved["wind"] if t.rows[i][st] != "0")
    assert abs(mw - S6_WIND_MW) < 0.5, mw
    for key in moved:
        for i in moved[key]:
            t.set(i, z, DOB)
    v3.write_table(twin(GEN), t)
    print("  %d Cernavoda, %d named S6 wind (%.0f MW), %d offshore rows moved to %s" % (
        len(moved["cernavoda"]), len(moved["wind"]), mw, len(moved["offshore"]), DOB))


def build_demand():
    t = Table(v3.path(DEMAND))
    assert t.header == ["z", "type"] + YEARS, (DEMAND, t.header[:4])
    ones = rows_of(t, RO)
    assert len(ones) == 2 and sorted(t.rows[i][1] for i in ones) == ["Energy", "Peak"], ones
    assert not rows_of(t, DOB), "Dobrogea already in " + DEMAND
    for k, i in enumerate(ones):
        dob = [DOB, t.rows[i][1]]
        for j in range(2, len(t.rows[i])):
            total = float(t.rows[i][j])
            share = round(total * LOAD_SHARE, 2)
            rest = "%.2f" % (total - share)
            assert abs(float(rest) + share - total) < 0.005, (t.rows[i][1], YEARS[j - 2])
            t.set(i, j, rest)
            dob.append("%.2f" % share)
        t.insert(ones[-1] + 1 + k, dob)
    v3.write_table(twin(DEMAND), t)
    print("  %s demand: %.0f percent of the Romanian energy and peak" % (DOB, LOAD_SHARE * 100))


def build_copies():
    cfg = config_defaults()
    for param, (rel, n, head) in COPIED.items():
        assert cfg[param] == rel, (param, cfg[param], rel)
        t = Table(v3.path(rel))
        assert t.header[:len(head)] == head, (rel, t.header[:4])
        copied = copy_rows(t, 0, rows_of(t, RO))
        assert copied == n, (rel, copied, n)
        v3.write_table(twin(rel), t)


def retarget(t, landing, j_from=0, j_to=1):
    """Romania becomes Dobrogea in every row whose other end is a landing zone; returns the count."""
    n = 0
    for i in range(1, len(t.rows)):
        r = t.rows[i]
        if r[j_from] == RO and r[j_to] in landing:
            t.set(i, j_from, DOB); n += 1
        elif r[j_to] == RO and r[j_from] in landing:
            t.set(i, j_to, DOB); n += 1
    return n


def build_network():
    for src, (tl, loss, landing) in TRIO.items():
        t = Table(v3.path(tl))
        assert t.header == ["z", "z2", "q"] + YEARS, (tl, t.header[:5])
        assert not rows_of(t, DOB) and not rows_of(t, DOB, 1), "Dobrogea already in " + tl
        pairs = {(r[0], r[1]) for r in t.rows[1:]}
        for zl in landing:
            assert (zl, RO) in pairs and (RO, zl) in pairs, (tl, zl)
        for zl in LANDING:
            if zl not in landing:
                assert (zl, RO) not in pairs and (RO, zl) not in pairs, (tl, zl)
        n = retarget(t, landing)
        assert n == 8 * len(landing), (tl, n)
        from_j = 3 + YEARS.index(str(EXIT_PLAN_FROM))
        for suffix, cap in VARIANTS:
            v = t.copy()
            for z, z2 in ((DOB, RO), (RO, DOB)):
                for q in ("Q1", "Q2", "Q3", "Q4"):
                    v.append([z, z2, q] + ["%.1f" % EXIT_TODAY_MW] * (from_j - 3) + ["%.1f" % cap] * (len(YEARS) - from_j + 3))
            v3.write_table(twin(tl, "_" + suffix), v)
        t = Table(v3.path(loss))
        assert t.header == ["z", "z2"] + YEARS, (loss, t.header[:4])
        assert not rows_of(t, DOB) and not rows_of(t, DOB, 1), "Dobrogea already in " + loss
        n = retarget(t, landing)
        assert n == 2 * len(landing), (loss, n)
        for z, z2 in ((DOB, RO), (RO, DOB)):
            t.append([z, z2] + [EXIT_LOSS] * len(YEARS))
        v3.write_table(twin(loss), t)
    print("  exit %s to %s: %s" % (DOB, RO, ", ".join(
        "%s = %.0f MW to %d then %.0f MW" % (s, EXIT_TODAY_MW, EXIT_PLAN_FROM - 1, c) for s, c in VARIANTS)))


def build_scenarios():
    sc = Table(v3.path("scenarios.csv"))
    H = sc.header
    assert H[0] == "paramNames", H[:3]
    names = [r[0] for r in sc.rows[1:]]
    new_names = [new for new, _, _ in NEW_COLUMNS]
    existing = [c for c in new_names if c in H]
    assert existing in ([], new_names), existing
    orig = [c for c in H if c not in new_names]
    assert orig == H[:len(orig)], "the 2z columns are not the last columns"
    orig_rows = [list(r[:len(orig)]) for r in sc.rows]
    sc = v3.truncate_columns(sc, len(orig))
    built = {}
    for new, src, suffix in NEW_COLUMNS:
        j = orig.index(src)
        cells = [r[j] for r in sc.rows[1:]]
        tl, loss, _ = TRIO[src]
        replaced = {
            "zcmap": (cells[names.index("zcmap")], twin(cells[names.index("zcmap")])),
            "pGenDataInput": (GEN, twin(GEN)),
            "pDemandForecast": (DEMAND, twin(DEMAND)),
            "pTransferLimit": (tl, twin(tl, "_" + suffix)),
            "pLossFactorInternal": (loss, twin(loss)),
        }
        assert replaced["zcmap"][0] in ZCMAPS, (new, replaced["zcmap"][0])
        for param, (expect, value) in replaced.items():
            assert cells[names.index(param)] == expect, (new, param, cells[names.index(param)])
            cells[names.index(param)] = value
        for param, (rel, _, _) in COPIED.items():
            assert cells[names.index(param)] == "", (new, param, cells[names.index(param)])
            cells[names.index(param)] = twin(rel)
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
    print("scenarios.csv: %d existing columns kept byte for byte, %d Dobrogea 2z columns" % (len(orig), len(built)))
    print("2z columns: " + " ".join(built))


if __name__ == "__main__":
    print("data folder: " + ROOT)
    build_zcmaps()
    build_generators()
    build_demand()
    build_copies()
    build_network()
    build_scenarios()
