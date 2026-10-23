# -*- coding: utf-8 -*-
"""Reference v2 of the RoBg family: Romania harmonised, Turkiye corrected, two Dobrogea
landing caps and the TEIAS HVDC candidates. Every output is a NEW file or a NEW
scenarios.csv column; nothing read by an existing scenario changes.

Romania, harmonisation with the Caucasus and Bulgaria
    Those zones read the CCDR default of 1.3 $m/MW for onshore wind with the default
    trajectory of their zone (Romania OnshoreWind 0.77 in 2030), while the nine Romanian
    Generic Onshore Wind classes and Wind_INECP_3 carried 1.75 $m/MW, 45,000 $/MW/yr and
    a custom trajectory at 0.942 in 2030 (1.65 $m/MW against 1.0 elsewhere). In
    supply/pGenDataInput_v2.csv the ten rows move to 1.3 and 40,000 (the zone default).
    supply/pCapexTrajectoriesCustom_v2.csv drops the custom rows of the nine wind classes,
    of Generic Solar Fixed and of Generic Offshore Wind Fixed, so that input_treatment.py
    fills them from pCapexTrajectoriesDefault.csv (Romania OnshoreWind 0.77, PV 0.80,
    OffshoreWind 0.714 in 2030). Named projects, floating offshore, battery, biomass, CCS
    and geothermal keep their rows.

Turkiye, corrections from the country team (2026-10-23)
    AFSIN_ELBISTAN_CD (800 MW lignite, 2030) becomes a candidate (Status 2 to 3).
    CENAL_3 (1,050 MW imported coal, WestAna) is added as a candidate from 2031 with the
    CENAL_1 parameters (heat rate, life 50, FOM 30,000, VOM 4), the 1.5 $m/MW capex of
    every Turkish coal candidate and a flat trajectory like CENAL_1 and CENAL_2.
    Trakia_Agg_Onshore_Cand total capacity 15,000 to 7,500 MW (build limit kept).
    Demand: the 2025 Energy and Peak of the nine zones are scaled by 364.83/380.2 and
    the years 2026 to 2029 re-interpolated linearly to the unchanged 2030 value in
    load/pDemandForecast_v2.csv; 2024 and 2030 onwards unchanged.

Grids and sensitivities
    supply/pGenDataInput_v2_s5.csv and _v2_2y.csv snap the committed StYr to the five
    and two year grids as build_s5_sensitivity_robg.py and build_freeexp_2y_robg.py do;
    _v2_s5_reshi, _v2_s5_hub50, _v2_s5_hub100 and load/pDemandForecast_v2_s5_demlo and
    _demhi are derived as their S5 originals. Each derivation is checked by re-deriving
    the existing S5 and two year files from the unchanged inputs byte for byte.

Dobrogea, Romanian landing capacity near Constanta
    trade/pTransferLimit_robg_geco_dob47.csv and _dob32.csv cap the GEC_GE to Romania
    import rows at 3,400 and 1,900 MW, so that with the 1,300 MW of BSSC the cables landing
    in Dobrogea carry at most 4.7 GW (exit limit with the planned Transelectrica projects)
    or 3.2 GW (exit limit today). Export rows unchanged. The cap is per pair, because
    pTransferLimit cannot express a joint cap on two lines.

HVDC, free expansion B
    trade/pNewTransmission_robg_freeexp_hvdc.csv replaces the 16 Turkish internal
    reinforcement candidates (10 x 1,000 MW per adjacent pair from 2028) with the three
    TEIAS HVDC corridors: CenterAna to Trakia 4,000 MW from 2032, SouthEast to WestAna
    2,000 MW from 2034, CenterBlack to Trakia 2,000 MW from 2036, at 0.6 $m/MW; every
    other candidate kept. Free expansion A keeps the reinforcement candidates.

scenarios.csv
    35 columns with the suffix _v2 (LC trio, 29 S5 columns, S5_FreeExp, LC_FreeExp_RoBg,
    LC_FreeExp_RoBg_2y) repointing pGenDataInput, pCapexTrajectories and pDemandForecast
    at the v2 files, then S5_GECO_Dob47_v2, S5_GECO_Dob32_v2, LC_FreeExpHvdc_RoBg_v2 and
    LC_FreeExpHvdc_RoBg_2y_v2. Idempotent: a second run reports every file unchanged.

Usage: python build_reference_v2_robg.py [path to data_blacksea]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_freeexp_robg import Table  # noqa: E402  (byte preserving csv table, same folder)

ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.normpath(
    os.path.join(HERE, "..", "..", "epm", "input", "data_blacksea"))

GRID_S5 = [2025, 2030, 2032, 2036, 2040]
GRID_2Y = [2025, 2026] + list(range(2028, 2041, 2))
TR_ZONES = ("Trakia", "NorthWest", "WestAna", "SouthEast", "CenterAna", "EastMed", "WestMed", "CenterBlack", "EastAna")
EU = ("Romania", "Bulgaria")
RES_TECHS = ("PV", "OnshoreWind", "OffshoreWind")
TR_ENERGY_2025 = 364830.0  # GWh, country team figure; the v7 file carries 380,200
RO_TRAJ_DROP = ["Generic Solar Fixed", "Generic Offshore Wind Fixed"]  # plus the nine generic wind classes
HVDC = [("CenterAna", "Trakia", 2032, 4000.0, 2400), ("SouthEast", "WestAna", 2034, 2000.0, 1200),
        ("CenterBlack", "Trakia", 2036, 2000.0, 1200)]
DOBROGEA = {"dob47": 3400.0, "dob32": 1900.0}  # GEC_GE to Romania import cap in MW; with BSSC 1,300 = 4.7 and 3.2 GW

GEN_V2 = "supply/pGenDataInput_v2.csv"
TRAJ_V2 = "supply/pCapexTrajectoriesCustom_v2.csv"
DEM_V2 = "load/pDemandForecast_v2.csv"
HVDC_FILE = "trade/pNewTransmission_robg_freeexp_hvdc.csv"
GEN_MAP = {"": GEN_V2,
           "supply/pGenDataInput_s5.csv": "supply/pGenDataInput_v2_s5.csv",
           "supply/pGenDataInput_2y.csv": "supply/pGenDataInput_v2_2y.csv",
           "supply/pGenDataInput_s5_reshi.csv": "supply/pGenDataInput_v2_s5_reshi.csv",
           "supply/pGenDataInput_s5_hub50.csv": "supply/pGenDataInput_v2_s5_hub50.csv",
           "supply/pGenDataInput_s5_hub100.csv": "supply/pGenDataInput_v2_s5_hub100.csv"}
DEM_MAP = {"": DEM_V2,
           "load/pDemandForecast_s5_demlo.csv": "load/pDemandForecast_v2_s5_demlo.csv",
           "load/pDemandForecast_s5_demhi.csv": "load/pDemandForecast_v2_s5_demhi.csv"}
WORLDS = ["Base", "BSSC", "GECO"]
VARIANTS = [("", WORLDS), ("_ETSLo", WORLDS), ("_ETSHi", WORLDS), ("_GasLo", WORLDS), ("_GasHi", WORLDS),
            ("_DemLo", WORLDS), ("_DemHi", WORLDS), ("_AzGas10", WORLDS), ("_ResHi", WORLDS),
            ("_Hub50", ["GECO"]), ("_Hub100", ["GECO"])]
SOURCES = (["LC_Baseline_RoBg", "LC_BSSC_RoBg", "LC_GECO_RoBg"]
           + ["S5_%s%s" % (w, s) for s, ws in VARIANTS for w in ws]
           + ["S5_FreeExp", "LC_FreeExp_RoBg", "LC_FreeExp_RoBg_2y"])
# (new column, v2 column it copies, parameter replaced, new file, expected cell in the copied column)
DERIVED = [("S5_GECO_Dob47_v2", "S5_GECO_v2", "pTransferLimit", "trade/pTransferLimit_robg_geco_dob47.csv", "trade/pTransferLimit_robg_geco.csv"),
           ("S5_GECO_Dob32_v2", "S5_GECO_v2", "pTransferLimit", "trade/pTransferLimit_robg_geco_dob32.csv", "trade/pTransferLimit_robg_geco.csv"),
           ("LC_FreeExpHvdc_RoBg_v2", "LC_FreeExp_RoBg_v2", "pNewTransmission", HVDC_FILE, "trade/pNewTransmission_robg_freeexp.csv"),
           ("LC_FreeExpHvdc_RoBg_2y_v2", "LC_FreeExp_RoBg_2y_v2", "pNewTransmission", HVDC_FILE, "trade/pNewTransmission_robg_freeexp.csv")]


def num(v, nd=6):
    """Decimal text without trailing zeros: 1500.0 -> 1500, 166.65 -> 166.65 (as the S5 builder)."""
    s = ("%." + str(nd) + "f") % v
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def path(rel):
    return os.path.join(ROOT, rel)


def read_bytes(rel):
    return open(path(rel), "rb").read()


def write_bytes(rel, data):
    old = read_bytes(rel) if os.path.exists(path(rel)) else None
    if old != data:
        open(path(rel), "wb").write(data)
    print("  %-48s %-9s %7d bytes" % (rel, "unchanged" if old == data else ("rewritten" if old is not None else "new"), len(data)))


def write_table(rel, table):
    write_bytes(rel, table.to_bytes())


def subset(table, keep):
    """New Table with the rows whose index is in keep (header included), lines kept byte for byte."""
    t = Table()
    t.bom, t.eol = table.bom, table.eol
    t.lines = [table.lines[i] for i in keep]
    t.rows = [list(table.rows[i]) for i in keep]
    t.touched = {k for k, i in enumerate(keep) if i in table.touched}
    return t


def cols(table, *names):
    return tuple(table.header.index(c) for c in names)


# ------------------------------------------------------------------ derivations shared with the S5 and 2y builders
def snap(styr, grid):
    s = int(float(styr))
    if grid[0] < s <= grid[-1]:
        return str(min(g for g in grid if g >= s))
    return styr


def snap_committed(table, grid):
    status, styr = cols(table, "Status", "StYr")
    out, moves = table.copy(), {}
    for i, r in enumerate(out.rows[1:], 1):
        if r[status] == "2" and r[styr] != "":
            new = snap(r[styr], grid)
            if new != r[styr]:
                moves[(r[styr], new)] = moves.get((r[styr], new), 0) + 1
                out.set(i, styr, new)
    for r in out.rows[1:]:
        if r[status] == "2" and r[styr] != "":
            v = int(float(r[styr]))
            assert v <= grid[0] or v > grid[-1] or v in grid, r[:6]
    return out, moves


def res_high(table):
    """Romania and Bulgaria PV, onshore and offshore wind (Status 2 and 3): capacity and build limit x 1.2."""
    z, tech, status, cap, bl = cols(table, "z", "tech", "Status", "Capacity", "BuildLimitperYear")
    out = table.copy()
    for i, r in enumerate(out.rows[1:], 1):
        if r[z] in EU and r[tech] in RES_TECHS and r[status] in ("2", "3"):
            out.set(i, cap, num(float(r[cap]) * 1.2))
            if r[bl] != "":
                out.set(i, bl, num(float(r[bl]) * 1.2))
    return out


def hub_rows(base, hub):
    g = base.header.index("g")
    names = {r[g] for r in base.rows[1:]}
    rows = [r for r in hub.rows[1:] if r[g] not in names]
    assert len(rows) == 14, len(rows)
    return rows


def with_hub(table, rows, f):
    cap, bl = cols(table, "Capacity", "BuildLimitperYear")
    out = table.copy()
    for r in rows:
        rr = list(r)
        rr[cap], rr[bl] = num(float(r[cap]) * f), num(float(r[bl]) * f)
        out.append(rr)
    return out


def scale_eu_demand(table, k):
    out, n = table.copy(), 0
    for i, r in enumerate(out.rows[1:], 1):
        if r[0] in EU:
            for j in range(2, len(r)):
                out.set(i, j, num(float(r[j]) * k, 6))
            n += 1
    assert n == 4, n
    return out


# ------------------------------------------------------------------ generators
def build_generators():
    base = Table(path("supply/pGenDataInput.csv"))
    g, z, tech, status, styr, retr, cap, bl, life, capex, fom = cols(
        base, "g", "z", "tech", "Status", "StYr", "RetrYr", "Capacity", "BuildLimitperYear", "Life", "Capex", "FOMperMW")
    names = [r[g] for r in base.rows[1:]]
    assert len(set(names)) == len(names), "duplicate generator names"
    assert "CENAL_3" not in names
    v2 = base.copy()

    ro = [i for i, r in enumerate(v2.rows[1:], 1) if r[z] == "Romania" and r[tech] == "OnshoreWind" and r[capex] == "1.75"]
    generic = sorted(v2.rows[i][g] for i in ro if v2.rows[i][g].startswith("Generic Onshore Wind"))
    assert len(ro) == 10 and len(generic) == 9, [v2.rows[i][g] for i in ro]
    assert {v2.rows[i][g] for i in ro} == set(generic) | {"Wind_INECP_3"}, [v2.rows[i][g] for i in ro]
    other = [r[g] for r in v2.rows[1:] if r[z] == "Romania" and r[tech] == "OnshoreWind" and r[capex] not in ("1.75", "")]
    for i in ro:
        assert v2.rows[i][fom] == "45000" and v2.rows[i][status] == "3", v2.rows[i][:16]
        v2.set(i, capex, "1.3")
        v2.set(i, fom, "40000")
    print("Romania: %d onshore wind rows 1.75 -> 1.3 $m/MW and FOM 45000 -> 40000 (%s); other priced Romanian wind rows kept: %s" % (
        len(ro), ", ".join(generic + ["Wind_INECP_3"]), ", ".join(other) or "none"))

    i = 1 + names.index("AFSIN_ELBISTAN_CD")
    r = v2.rows[i]
    assert r[z] == "SouthEast" and r[status] == "2" and r[styr] == "2030" and r[cap] == "800" and r[capex] == "1.5", r[:16]
    v2.set(i, status, "3")
    i = 1 + names.index("Trakia_Agg_Onshore_Cand")
    r = v2.rows[i]
    assert r[z] == "Trakia" and r[status] == "3" and r[cap] == "15000" and r[bl] == "1072", r[:10]
    v2.set(i, cap, "7500")
    c1, c2 = base.rows[1 + names.index("CENAL_1")], base.rows[1 + names.index("CENAL_2")]
    assert c1[1:] == c2[1:] and c1[z] == "WestAna" and c1[status] == "1" and c1[capex] == "" and c1[fom] == "30000" and c1[life] == "50", c1
    c3 = list(c1)
    c3[g], c3[status], c3[styr], c3[retr], c3[cap], c3[bl], c3[capex] = "CENAL_3", "3", "2031", "2081", "1050", "1050", "1.5"
    assert int(c3[retr]) == int(c3[styr]) + int(c3[life]), c3
    v2.insert(2 + names.index("CENAL_2"), c3)
    print("Turkiye: AFSIN_ELBISTAN_CD Status 2 -> 3; Trakia_Agg_Onshore_Cand Capacity 15000 -> 7500; CENAL_3 added: %s" % ",".join(c3))
    write_table(GEN_V2, v2)
    return base, v2, generic


def build_grid_generators(base, v2):
    hub = Table(path("supply/pGenDataInput_hub.csv"))
    assert hub.header == base.header, "hub file header differs from the base file"
    s5_base = snap_committed(base, GRID_S5)[0]
    checks = {"supply/pGenDataInput_s5.csv": s5_base,
              "supply/pGenDataInput_2y.csv": snap_committed(base, GRID_2Y)[0],
              "supply/pGenDataInput_s5_reshi.csv": res_high(s5_base),
              "supply/pGenDataInput_s5_hub50.csv": with_hub(s5_base, hub_rows(base, hub), 0.5),
              "supply/pGenDataInput_s5_hub100.csv": with_hub(s5_base, hub_rows(base, hub), 1.0)}
    for rel, t in checks.items():
        assert t.to_bytes() == read_bytes(rel), "cannot re-derive %s from supply/pGenDataInput.csv" % rel
    print("check: the five existing S5 and two year generator files are re-derived byte for byte from the base file")
    for tag, grid in (("s5", GRID_S5), ("2y", GRID_2Y)):
        out, moves = snap_committed(v2, grid)
        print("pGenDataInput_v2_%s: %d StYr moved: %s" % (tag, sum(moves.values()),
              ", ".join("%s->%s x%d" % (a, b, n) for (a, b), n in sorted(moves.items()))))
        write_table("supply/pGenDataInput_v2_%s.csv" % tag, out)
        if tag == "s5":
            write_table("supply/pGenDataInput_v2_s5_reshi.csv", res_high(out))
            for name, f in (("hub50", 0.5), ("hub100", 1.0)):
                write_table("supply/pGenDataInput_v2_s5_%s.csv" % name, with_hub(out, hub_rows(v2, hub), f))


# ------------------------------------------------------------------ capex trajectories
def build_trajectories(generic):
    base = Table(path("supply/pCapexTrajectoriesCustom.csv"))
    assert base.header[0] == "gen" and base.header[1] == "2024", base.header[:3]
    names = [r[0] for r in base.rows[1:]]
    assert len(set(names)) == len(names) and "CENAL_3" not in names
    drop = set(generic) | set(RO_TRAJ_DROP)
    for n in drop:
        assert n in names, "no custom trajectory row for " + n
    wind = {tuple(r[1:]) for r in base.rows[1:] if r[0] in generic}
    assert len(wind) == 1, "the nine generic wind classes do not share one trajectory"
    j30 = base.header.index("2030")
    print("trajectories dropped (zone default applies): %d rows; generic wind 2030 factor %s, Generic Solar Fixed %s, Generic Offshore Wind Fixed %s" % (
        len(drop), list(wind)[0][j30 - 1], base.rows[1 + names.index("Generic Solar Fixed")][j30],
        base.rows[1 + names.index("Generic Offshore Wind Fixed")][j30]))
    out = subset(base, [0] + [i for i in range(1, len(base.rows)) if base.rows[i][0] not in drop])
    kept = [r[0] for r in out.rows[1:]]
    c1, c2 = base.rows[1 + names.index("CENAL_1")], base.rows[1 + names.index("CENAL_2")]
    assert c1[1:] == c2[1:] and set(c1[1:]) <= {"1", "1.0"}, c1[:5]
    out.insert(2 + kept.index("CENAL_2"), ["CENAL_3"] + c1[1:])
    assert len(out.rows) == len(base.rows) - len(drop) + 1
    write_table(TRAJ_V2, out)


# ------------------------------------------------------------------ demand
def build_demand():
    base = Table(path("load/pDemandForecast.csv"))
    H = base.header
    assert H[:2] == ["z", "type"], H[:3]
    j = {int(y): 2 + k for k, y in enumerate(H[2:])}
    tr = [i for i, r in enumerate(base.rows[1:], 1) if r[0] in TR_ZONES]
    assert len(tr) == 18 and {base.rows[i][1] for i in tr} == {"Peak", "Energy"}, len(tr)

    def total(table, kind, y):
        return sum(float(table.rows[i][j[y]]) for i in tr if table.rows[i][1] == kind)

    e25 = total(base, "Energy", 2025)
    assert abs(e25 - 380200.0) < 0.01, e25
    f = TR_ENERGY_2025 / e25
    v2 = base.copy()
    for i in tr:
        r = base.rows[i]
        v25, v30 = float(r[j[2025]]) * f, float(r[j[2030]])
        v2.set(i, j[2025], num(v25))
        for y in range(2026, 2030):
            v2.set(i, j[y], num(v25 + (v30 - v25) * (y - 2025) / 5.0))
    assert abs(total(v2, "Energy", 2025) - TR_ENERGY_2025) < 0.01
    assert total(v2, "Energy", 2030) == total(base, "Energy", 2030) and total(v2, "Peak", 2030) == total(base, "Peak", 2030)
    for kind in ("Energy", "Peak"):
        print("Turkiye %-6s %s" % (kind, "; ".join("%d %.0f -> %.0f" % (y, total(base, kind, y), total(v2, kind, y)) for y in range(2024, 2031))))
    write_table(DEM_V2, v2)
    for name, k in (("demlo", 0.9), ("demhi", 1.1)):
        rel = "load/pDemandForecast_s5_%s.csv" % name
        assert scale_eu_demand(base, k).to_bytes() == read_bytes(rel), "cannot re-derive " + rel
        write_table("load/pDemandForecast_v2_s5_%s.csv" % name, scale_eu_demand(v2, k))
    print("check: the two existing S5 demand files are re-derived byte for byte from the base file")


# ------------------------------------------------------------------ transmission
def build_transmission():
    base = Table(path("trade/pNewTransmission_robg_freeexp.csv"))
    assert base.header == ["From", "To", "EarliestEntry", "MaximumNumOfLines", "CapacityPerLine", "CostPerLine", "Life", "Status"], base.header
    tr = [i for i, r in enumerate(base.rows[1:], 1) if r[0] in TR_ZONES and r[1] in TR_ZONES]
    assert len(tr) == 16, len(tr)
    for i in tr:
        r = base.rows[i]
        assert r[2:5] == ["2028", "10", "1000.0"] and r[5] in ("400", "600") and r[6:] == ["30", "3"], r
    pairs = {(base.rows[i][0], base.rows[i][1]) for i in tr} | {(base.rows[i][1], base.rows[i][0]) for i in tr}
    out = subset(base, [i for i in range(len(base.rows)) if i not in tr])
    pos = tr[0]  # the Turkish rows are the first ones removed, so the first one keeps its index
    for k, (a, b, y, mw, cost) in enumerate(HVDC):
        assert (a, b) not in pairs, "HVDC pair %s %s duplicates a reinforcement candidate" % (a, b)
        out.insert(pos + k, [a, b, str(y), "1", "%.1f" % mw, str(cost), "30", "3"])
    assert len(out.rows) == len(base.rows) - 16 + 3
    print("pNewTransmission hvdc: 16 Turkish reinforcement candidates removed, %d HVDC candidates added, %d other candidates kept" % (
        len(HVDC), len(out.rows) - 1 - len(HVDC)))
    write_table(HVDC_FILE, out)

    geco = Table(path("trade/pTransferLimit_robg_geco.csv"))
    H = geco.header
    assert H[:3] == ["z", "z2", "q"], H[:4]
    imp = [i for i, r in enumerate(geco.rows[1:], 1) if r[0] == "GEC_GE" and r[1] == "Romania"]
    exp = [i for i, r in enumerate(geco.rows[1:], 1) if r[0] == "Romania" and r[1] == "GEC_GE"]
    bssc = [i for i, r in enumerate(geco.rows[1:], 1) if r[0] == "Georgia" and r[1] == "Romania"]
    assert len(imp) == 4 and len(exp) == 4 and len(bssc) == 4, (len(imp), len(exp), len(bssc))
    assert max(float(v) for i in bssc for v in geco.rows[i][3:]) == 1300.0
    years = [int(y) for y in H[3:]]
    for name, mw in DOBROGEA.items():
        out, n = geco.copy(), 0
        for i in imp:
            for jj in range(3, len(H)):
                if float(out.rows[i][jj]) > mw:
                    out.set(i, jj, "%.1f" % mw)
                    n += 1
        landing = ["%d %.0f" % (y, float(out.rows[imp[0]][3 + k]) + float(out.rows[bssc[0]][3 + k])) for k, y in enumerate(years) if y in (2032, 2036, 2040)]
        print("pTransferLimit %s: %d GEC_GE->Romania cells capped at %.0f MW, export rows unchanged; cable landing BSSC+GEC: %s" % (name, n, mw, ", ".join(landing)))
        write_table("trade/pTransferLimit_robg_geco_%s.csv" % name, out)


# ------------------------------------------------------------------ scenarios
def build_scenarios():
    sc = Table(path("scenarios.csv"))
    H = sc.header
    assert H[0] == "paramNames", H[:3]
    names = [r[0] for r in sc.rows[1:]]
    orig = [c for c in H if not c.endswith("_v2")]
    assert orig == H[:len(orig)], "v2 columns are not the last columns"
    orig_rows = [list(r[:len(orig)]) for r in sc.rows]
    built = {}
    for src in SOURCES:
        j = H.index(src)
        cells = [r[j] for r in sc.rows[1:]]
        gen, dem, traj = (cells[names.index(p)] for p in ("pGenDataInput", "pDemandForecast", "pCapexTrajectories"))
        assert gen in GEN_MAP and dem in DEM_MAP and traj == "", (src, gen, dem, traj)
        assert cells[names.index("pCapexTrajectoriesDefault")] == "" and cells[names.index("pGenDataInputDefault")] == "", src
        cells[names.index("pGenDataInput")] = GEN_MAP[gen]
        cells[names.index("pDemandForecast")] = DEM_MAP[dem]
        cells[names.index("pCapexTrajectories")] = TRAJ_V2
        built[src + "_v2"] = cells
    for new, src, param, rel, expect in DERIVED:
        cells = list(built[src])
        assert cells[names.index(param)] == expect, (new, cells[names.index(param)])
        cells[names.index(param)] = rel
        built[new] = cells
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
    print("scenarios.csv: %d existing columns kept byte for byte, %d v2 columns" % (len(orig), len(built)))
    print("queue order: " + " ".join(built))


if __name__ == "__main__":
    print("data folder: " + ROOT)
    base, v2, generic = build_generators()
    build_grid_generators(base, v2)
    build_trajectories(generic)
    build_demand()
    build_transmission()
    build_scenarios()
