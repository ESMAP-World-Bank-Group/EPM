# -*- coding: utf-8 -*-
"""Costs v3 of the RoBg family: the interconnector capex at the project estimate basis, the
onland corridor through Turkiye as a forced reference run and as free expansion segments, a
low capex variant for the capex sensitivity, and the committed CenterAna to Trakia HVDC in
every v3 column. Every output is a NEW file or a NEW scenarios.csv column; nothing read by an
existing scenario changes.

Committed HVDC (every v3 column)
    CenterAna to Trakia, 4,000 MW from 2032, committed by TEIAS: carried as a transfer limit,
    outside the optimisation and without cost, like BSSC and GEC in the forced runs. Every
    pTransferLimit file read by a v3 column gets a twin with the suffix _hvdc (eight rows,
    both directions and four quarters, 0 before 2032 and 4,000 MW from 2032) and every
    pLossFactorInternal file a twin with the same suffix (two rows at 0.03, about 550 km of
    HVDC and two converter stations). On the five-year grid the line appears in 2032.

Capex basis (ex post for the forced runs, in the objective for the free expansion runs)
    reference: BSSC 3,700 MUSD for 1,300 MW; GEC 4,400 MUSD per 1,261 MW link (the CBA basis
    of the feasibility study, its ratio to the standard cost basis applied to the three T5
    link costs); onland tapped corridor 3,890 MUSD for 2,000 MW (1,400 MUSD per 2 GW converter
    pair, 0.42 MUSD per km of overhead line over 1,760 km, plus 10 percent for cable sections,
    four converter stations; no dedicated Bosphorus crossing, which would add 1,100 MUSD).
    low: BSSC 3,500 and GEC 2,783 per link (the ENTSO-E standard cost basis of the v2 files);
    the onland corridor has no low basis.

Onland corridor, forced (S5_Onland_v3, LC_Onland_RoBg_v3)
    trade/pTransferLimit_robg_onland.csv = trade/pTransferLimit_robg.csv plus 2,000 MW in both
    directions from 2031 on Georgia to EastAna (on top of the 700 MW AC link), EastAna to
    NorthWest and NorthWest to Bulgaria (two new pairs): a multi terminal HVDC tapped at
    Erzurum and Marmara, each segment able to carry the full 2,000 MW. Capex ex post, as for
    BSSC and GEC. trade/pLossFactorInternal_robg_onland.csv = the baseline loss file plus
    0.035 on EastAna to NorthWest and 0.02 on NorthWest to Bulgaria, both directions (Georgia
    to EastAna kept at 0.03, the AC value). trade/pLossFactorInternal_robg_geco_onland.csv =
    the same additions on the GECO loss file, for the free expansion columns.

Free expansion v3 (one candidate file)
    trade/pNewTransmission_robg_freeexp_v3.csv = the v2 candidate files with BSSC at 3,700, the
    GEC links at 4,400, and the only Turkish candidates being the three onland segments
    Georgia to EastAna 1,280, EastAna to NorthWest 1,280 and NorthWest to Bulgaria 1,330 MUSD
    (1 x 2,000 MW each, from 2031, life 30, Status 3): the EastAna to Georgia AC candidate
    (6 x 350 MW at 850 MUSD), the 16 internal reinforcement candidates of the v2 A file and
    the three TEIAS HVDC candidates of the v2 B file are dropped (CenterAna to Trakia is now
    committed, see above). The A and B derivations give the same file, checked. The cross
    border candidates (BSSC, GEC, Trakia to Bulgaria, Caucasus) are kept.
    trade/pNewTransmission_robg_freeexp_v3_capexlo.csv = the same with BSSC 3,500 and GEC
    2,783: the capex sensitivity of the free expansion.

Columns (40)
    Every v2 column with an empty pNewTransmission (34: the LC trio, the 29 S5 columns, the
    two Dobrogea columns) gets a v3 twin with the _hvdc transfer limit and loss files.
    S5_Onland_v3 = S5_Base_v2 plus the onland files; LC_Onland_RoBg_v3 = LC_Baseline_RoBg_v2
    plus the same. S5_FreeExp_v3, LC_FreeExp_RoBg_v3 and LC_FreeExp_RoBg_2y_v3 = the v2
    columns with the v3 candidate file and the GECO onland loss file;
    LC_FreeExp_RoBg_2y_v3_CapexLo = the 2y column with the low capex file. All with the
    _hvdc twins. LC_FreeExpHvdc_RoBg_v2 and LC_FreeExpHvdc_RoBg_2y_v2 get no v3 twin: with a
    single candidate file the A and B families merge.

Ex post table
    capex_interconnectors_expost.csv at the data root: project, capacity, reference and low
    capex, for the benefit computation. Not read by GAMS. The committed HVDC has no cost.

Usage: python build_costs_v3_robg.py [path to data_blacksea]. Idempotent.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_freeexp_robg import Table, serialise  # noqa: E402  (byte preserving csv table, same folder)

ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.normpath(
    os.path.join(HERE, "..", "..", "epm", "input", "data_blacksea"))

YEARS = [str(y) for y in range(2024, 2054)]
ONLAND_FROM = 2031
ONLAND_MW = 2000.0
ONLAND_NEW_PAIRS = [("EastAna", "NorthWest"), ("NorthWest", "Bulgaria")]
ONLAND_LOSS = {("EastAna", "NorthWest"): "0.035", ("NorthWest", "Bulgaria"): "0.02"}
SEGMENTS = [("Georgia", "EastAna", 1280), ("EastAna", "NorthWest", 1280), ("NorthWest", "Bulgaria", 1330)]
AC_DROP = ["EastAna", "Georgia", "2032", "6", "350.0", "850", "30", "3"]
CAPEX_REF = {("Georgia", "Romania"): ("3500", "3700"), ("GEC_GE", "Romania"): ("2783", "4400")}
TURKISH_ZONES = {"EastAna", "CenterAna", "CenterBlack", "NorthWest", "WestAna", "SouthEast", "Trakia", "EastMed", "WestMed"}

HVDC_PAIR = ("CenterAna", "Trakia")
HVDC_FROM = 2032
HVDC_MW = 4000.0
HVDC_LOSS = "0.03"
HVDC_SUFFIX = "_hvdc"

TL_BASE, TL_ONLAND = "trade/pTransferLimit_robg.csv", "trade/pTransferLimit_robg_onland.csv"
LOSS_BASE, LOSS_ONLAND = "trade/pLossFactorInternal_robg.csv", "trade/pLossFactorInternal_robg_onland.csv"
LOSS_GECO, LOSS_GECO_ONLAND = "trade/pLossFactorInternal_robg_geco.csv", "trade/pLossFactorInternal_robg_geco_onland.csv"
CAND_A, CAND_B = "trade/pNewTransmission_robg_freeexp.csv", "trade/pNewTransmission_robg_freeexp_hvdc.csv"
CAND_V3, CAND_V3_LO = "trade/pNewTransmission_robg_freeexp_v3.csv", "trade/pNewTransmission_robg_freeexp_v3_capexlo.csv"
STALE = ["trade/pNewTransmission_robg_freeexp_hvdc_v3.csv", "trade/pNewTransmission_robg_freeexp_hvdc_v3_capexlo.csv"]
EXPOST = "capex_interconnectors_expost.csv"
TL, LOSS, NT = "pTransferLimit", "pLossFactorInternal", "pNewTransmission"

# new column, source column, {param: (expected value, new value)}; the _hvdc twins are applied after
EXPLICIT = [
    ("S5_Onland_v3", "S5_Base_v2", {TL: (TL_BASE, TL_ONLAND), LOSS: (LOSS_BASE, LOSS_ONLAND)}),
    ("LC_Onland_RoBg_v3", "LC_Baseline_RoBg_v2", {TL: (TL_BASE, TL_ONLAND), LOSS: (LOSS_BASE, LOSS_ONLAND)}),
    ("S5_FreeExp_v3", "S5_FreeExp_v2", {NT: (CAND_A, CAND_V3), LOSS: (LOSS_GECO, LOSS_GECO_ONLAND)}),
    ("LC_FreeExp_RoBg_v3", "LC_FreeExp_RoBg_v2", {NT: (CAND_A, CAND_V3), LOSS: (LOSS_GECO, LOSS_GECO_ONLAND)}),
    ("LC_FreeExp_RoBg_2y_v3", "LC_FreeExp_RoBg_2y_v2", {NT: (CAND_A, CAND_V3), LOSS: (LOSS_GECO, LOSS_GECO_ONLAND)}),
    ("LC_FreeExp_RoBg_2y_v3_CapexLo", "LC_FreeExp_RoBg_2y_v2", {NT: (CAND_A, CAND_V3_LO), LOSS: (LOSS_GECO, LOSS_GECO_ONLAND)}),
]
NO_TWIN = ["LC_FreeExpHvdc_RoBg_v2", "LC_FreeExpHvdc_RoBg_2y_v2"]

EXPOST_TEXT = "\n".join([
    "project,from,to,capacity_mw,capex_reference_musd,capex_low_musd,basis",
    'BSSC,Georgia,Romania,1300,3700,3500,"reference: project estimate basis; low: ENTSO-E standard cost basis, the value of the v2 candidate files"',
    'GEC,GEC_GE,Romania,3783,13200,8350,"three links of 1,261 MW; reference 4,400 per link (CBA basis of the feasibility study); low 2,637 + 2,974 + 2,739 (standard cost basis)"',
    'Onland,Georgia,Bulgaria,2000,3890,3890,"tapped at Erzurum (EastAna) and Marmara (NorthWest); 1,400 per 2 GW converter pair, 0.42 per km over 1,760 km, plus 10 percent; four converter stations; no dedicated Bosphorus crossing, which would add 1,100; no low basis"',
]) + "\n"


def path(rel):
    return os.path.join(ROOT, rel.replace("/", os.sep))


def read_bytes(rel):
    return open(path(rel), "rb").read()


def write_bytes(rel, data):
    old = read_bytes(rel) if os.path.exists(path(rel)) else None
    if old != data:
        open(path(rel), "wb").write(data)
    print("  %-58s %-9s %7d bytes" % (rel, "unchanged" if old == data else ("rewritten" if old is not None else "new"), len(data)))


def write_table(rel, table):
    write_bytes(rel, table.to_bytes())


def hvdc_name(rel):
    assert rel.endswith(".csv") and not rel.endswith(HVDC_SUFFIX + ".csv"), rel
    return rel[:-4] + HVDC_SUFFIX + ".csv"


def subset(table, keep):
    t = Table()
    t.bom, t.eol = table.bom, table.eol
    t.lines = [table.lines[i] for i in keep]
    t.rows = [list(table.rows[i]) for i in keep]
    t.touched = {k for k, i in enumerate(keep) if i in table.touched}
    return t


def build_transfer_limit():
    t = Table(path(TL_BASE))
    assert t.header == ["z", "z2", "q"] + YEARS, t.header[:5]
    pairs = {(r[0], r[1]) for r in t.rows[1:]}
    for a, b in ONLAND_NEW_PAIRS:
        assert (a, b) not in pairs and (b, a) not in pairs, (a, b)
    from_j = 3 + YEARS.index(str(ONLAND_FROM))
    touched = 0
    for i, r in enumerate(t.rows[1:], 1):
        if {r[0], r[1]} == {"Georgia", "EastAna"}:
            assert all(v == "700.0" for v in r[3:]), r[:5]
            for j in range(from_j, len(r)):
                t.set(i, j, "%.1f" % (float(r[j]) + ONLAND_MW))
            touched += 1
    assert touched == 8, touched
    for a, b in ONLAND_NEW_PAIRS:
        for z, z2 in ((a, b), (b, a)):
            for q in ("Q1", "Q2", "Q3", "Q4"):
                t.append([z, z2, q] + ["0.0"] * (from_j - 3) + ["%.1f" % ONLAND_MW] * (len(YEARS) - from_j + 3))
    write_table(TL_ONLAND, t)
    print("  Georgia to EastAna 700 to 2,700 MW from %d (8 rows), %d rows added" % (ONLAND_FROM, 4 * 2 * len(ONLAND_NEW_PAIRS)))


def build_loss(base, out):
    t = Table(path(base))
    assert t.header == ["z", "z2"] + YEARS, t.header[:4]
    pairs = {(r[0], r[1]) for r in t.rows[1:]}
    for (a, b), v in ONLAND_LOSS.items():
        assert (a, b) not in pairs and (b, a) not in pairs, (a, b)
        for z, z2 in ((a, b), (b, a)):
            t.append([z, z2] + [v] * len(YEARS))
    ge = [r for r in t.rows[1:] if {r[0], r[1]} == {"Georgia", "EastAna"}]
    assert len(ge) == 2 and all(v == "0.03" for r in ge for v in r[2:]), ge[:1]
    write_table(out, t)


def build_hvdc_transfer_limit(base):
    t = Table(path(base))
    assert t.header == ["z", "z2", "q"] + YEARS, (base, t.header[:5])
    pairs = {(r[0], r[1]) for r in t.rows[1:]}
    a, b = HVDC_PAIR
    assert (a, b) not in pairs and (b, a) not in pairs, (base, HVDC_PAIR)
    from_j = 3 + YEARS.index(str(HVDC_FROM))
    for z, z2 in ((a, b), (b, a)):
        for q in ("Q1", "Q2", "Q3", "Q4"):
            t.append([z, z2, q] + ["0.0"] * (from_j - 3) + ["%.1f" % HVDC_MW] * (len(YEARS) - from_j + 3))
    write_table(hvdc_name(base), t)


def build_hvdc_loss(base):
    t = Table(path(base))
    assert t.header == ["z", "z2"] + YEARS, (base, t.header[:4])
    pairs = {(r[0], r[1]) for r in t.rows[1:]}
    a, b = HVDC_PAIR
    assert (a, b) not in pairs and (b, a) not in pairs, (base, HVDC_PAIR)
    for z, z2 in ((a, b), (b, a)):
        t.append([z, z2] + [HVDC_LOSS] * len(YEARS))
    write_table(hvdc_name(base), t)


def derive_candidates(src, expected_turkish_drops):
    t = Table(path(src))
    assert t.header == ["From", "To", "EarliestEntry", "MaximumNumOfLines", "CapacityPerLine", "CostPerLine", "Life", "Status"], t.header
    ac = [i for i, r in enumerate(t.rows) if r == AC_DROP]
    assert len(ac) == 1, (src, ac)
    turkish = [i for i, r in enumerate(t.rows[1:], 1) if r[0] in TURKISH_ZONES and r[1] in TURKISH_ZONES]
    assert len(turkish) == expected_turkish_drops, (src, len(turkish))
    keep = [i for i in range(len(t.rows)) if i != ac[0] and i not in turkish]
    v3 = subset(t, keep)
    pos = sum(1 for i in keep if i < ac[0])
    for k, (a, b, cost) in enumerate(SEGMENTS):
        v3.insert(pos + k, [a, b, str(ONLAND_FROM), "1", "%.1f" % ONLAND_MW, str(cost), "30", "3"])
    set_capex(v3, src, 0, 1)
    return v3


def build_candidates():
    from_b = derive_candidates(CAND_B, 3)
    from_a = derive_candidates(CAND_A, 16)
    assert from_a.rows == from_b.rows, "the A and B derivations differ"
    write_table(CAND_V3, from_b)
    lo = Table(path(CAND_V3))
    set_capex(lo, CAND_V3, 1, 0)
    write_table(CAND_V3_LO, lo)
    for rel in STALE:
        if os.path.exists(path(rel)):
            os.remove(path(rel))
            print("  %-58s removed" % rel)
    print("  %d candidate rows: Turkish candidates reduced to the three onland segments (%d MUSD); BSSC and GEC at the reference basis, capexlo variant"
          % (len(from_b.rows) - 1, sum(s[2] for s in SEGMENTS)))


def set_capex(t, label, k_from, k_to):
    seen = set()
    for i, r in enumerate(t.rows[1:], 1):
        key = (r[0], r[1])
        if key in CAPEX_REF:
            assert r[5] == CAPEX_REF[key][k_from], (label, r)
            t.set(i, 5, CAPEX_REF[key][k_to])
            seen.add(key)
    assert seen == set(CAPEX_REF), (label, seen)


def build_expost():
    write_bytes(EXPOST, EXPOST_TEXT.encode("utf-8"))


def truncate_columns(sc, n):
    """The first n columns of the table, byte for byte (the v3 columns were appended by this
    builder, so every line is already csv.writer serialised)."""
    if n == len(sc.header):
        return sc
    t = Table()
    t.bom, t.eol = sc.bom, sc.eol
    t.rows = [list(r[:n]) for r in sc.rows]
    t.lines = [None] * len(t.rows)
    for i, r in enumerate(t.rows):
        assert sc.line(i).startswith(serialise(r) + ","), ("row %d is not a plain extension of its first %d cells" % (i, n))
    return t


def build_scenarios():
    full = Table(path("scenarios.csv"))
    H = full.header
    assert H[0] == "paramNames", H[:3]
    names = [r[0] for r in full.rows[1:]]
    orig = [c for c in H if not c.endswith("_v3") and not c.endswith("_v3_CapexLo")]
    assert orig == H[:len(orig)], "v3 columns are not the last columns"
    sc = truncate_columns(full, len(orig))
    orig_rows = [list(r) for r in sc.rows]
    cell = lambda col, param: sc.rows[1 + names.index(param)][orig.index(col)]  # noqa: E731
    explicit_sources = [src for _, src, ch in EXPLICIT if NT in ch]
    plan = []
    for col in orig:
        if not col.endswith("_v2") or col in NO_TWIN or col in explicit_sources:
            continue
        assert cell(col, NT) == "", (col, cell(col, NT))
        plan.append((col[:-3] + "_v3", col, {}))
    plan += EXPLICIT
    assert len(plan) == 40, len(plan)
    built, tl_bases, loss_bases = {}, set(), set()
    for new, src, changes in plan:
        j = orig.index(src)
        cells = [r[j] for r in sc.rows[1:]]
        for param, (expect, value) in changes.items():
            assert cells[names.index(param)] == expect, (new, param, cells[names.index(param)])
            cells[names.index(param)] = value
        for param, bases in ((TL, tl_bases), (LOSS, loss_bases)):
            k = names.index(param)
            assert cells[k].endswith(".csv"), (new, param, cells[k])
            bases.add(cells[k])
            cells[k] = hvdc_name(cells[k])
        built[new] = cells
    for base in sorted(tl_bases):
        build_hvdc_transfer_limit(base)
    for base in sorted(loss_bases):
        build_hvdc_loss(base)
    print("  committed HVDC %s to %s, %.0f MW from %d: %d transfer limit and %d loss twins"
          % (HVDC_PAIR[0], HVDC_PAIR[1], HVDC_MW, HVDC_FROM, len(tl_bases), len(loss_bases)))
    for new, cells in built.items():
        for v in cells:
            if v.endswith(".csv") or v.endswith(".opt"):
                assert os.path.exists(path(v)), (new, v)
        sc.append_column(new, cells)
    write_table("scenarios.csv", sc)
    back = Table(path("scenarios.csv"))
    assert [r[:len(orig)] for r in back.rows] == orig_rows, "an existing column changed"
    assert back.header[len(orig):] == list(built), back.header[len(orig):]
    for new, cells in built.items():
        k = back.header.index(new)
        assert [r[k] for r in back.rows[1:]] == cells, new
    print("scenarios.csv: %d existing columns kept byte for byte, %d v3 columns" % (len(orig), len(built)))
    print("v3 columns: " + " ".join(built))


if __name__ == "__main__":
    print("data folder: " + ROOT)
    build_transfer_limit()
    build_loss(LOSS_BASE, LOSS_ONLAND)
    build_loss(LOSS_GECO, LOSS_GECO_ONLAND)
    build_candidates()
    build_expost()
    build_scenarios()
