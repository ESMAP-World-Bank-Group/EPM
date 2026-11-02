"""
Derive the Ext scope files and the grid variants of the v4 input set under epm/input/data_blacksea.

    python tools/build_ext_inputs.py            # write the derived files, print and save the checks
    python tools/build_ext_inputs.py --check    # compare the files on disk with a fresh derivation, write nothing
    python tools/build_ext_inputs.py --force    # rewrite a derived file that differs from the derivation

Every file written here is derived from a file that already exists in the data folder; nothing is
typed by hand and the sources are never modified. A derived file keeps the cell text, the column
order, the line ending and the byte order mark of its source, so that a dropped-row file is a
textual subset of its source. The checks are printed and saved to
internal_analysis/scenarios/ext_inputs_report.md.

Ext scope: Romania and Bulgaria (RB below) leave the zone map and come back as price zones.
  zcmap_ext, zcmap_ext_geco             zcmap_robg[_geco] without the RB rows
  supply/pGenDataInput_v4_{g}_ext       the plant file without the plants of RB
  supply/pStorageDataInput_{g}_ext      the storage file without the units of RB
  supply/pFuelPrice_ext                 pFuelPrice_robg without the RB countries
  reserve/pPlanningReserveMargin_ext    pPlanningReserveMargin_robg without the RB countries
  constraint/pMaxFuellimit_ext          pMaxFuellimit_robg without the RB zones
  trade/zext_ext                        zext_robg without the zones that only RB reaches, plus Romania and Bulgaria
  trade/pExtTransferLimit_ext[_case]    pExtTransferLimit_robg without the rows whose z is in RB, plus the lines of
                                        pTransferLimit_robg_<case>_hvdc that touch one RB zone, turned into external
                                        lines: z -> RB becomes (z, RB, Export) and RB -> z becomes (z, RB, Import).
                                        The Romania to Bulgaria line joins two price zones and disappears.
  trade/pLossFactorInternal_ext         pLossFactorInternal_robg_hvdc without the rows touching RB (the BSSC and GECO
  trade/pTransferLimit_ext              variants give the same file once their RB rows are dropped, checked here)
  trade/pMaxAnnualExternalTradeShare_ext   the _robg share file without the Romania and Bulgaria columns
  trade/pTradePrice[Export]_ext         the robg price rows of the zext kept, plus the Romania and Bulgaria rows of
                                        the eu_central file
  trade/pTradePrice[Export]_ext_<eu case>  the robg rows of the non EU zext, plus the Greece, Romania and Bulgaria
                                        rows of the eu case file. Greece is an EU zone and follows the EU price case;
                                        in the central Ext files it keeps its robg rows, the ones Int uses, which sit
                                        2 USD/MWh below the eu_central Greece rows.

Grid variants.
  supply/pGenDataInput_v4_2y            pGenDataInput_v2_2y plus the v4 edits, that is the cells where v4_s5 differs
                                        from v2_s5; an edited StYr of a committed plant is re-rounded on the 2y grid
  supply/pGenDataInput_v4_{g}_reshi     the v4 file plus the ResHi edits, the cells where v2_s5_reshi differs from v2_s5
  supply/pStorageDataInput_2y           copy of pStorageDataInput: every committed unit already starts on a 2y year
"""
import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "epm" / "input" / "data_blacksea"
REPORT = ROOT / "internal_analysis" / "scenarios" / "ext_inputs_report.md"
RB = ("Romania", "Bulgaria")
EU = ("Greece", "Romania", "Bulgaria")
BOM = "﻿"
GRIDS = {"s5": "y_s5.csv", "2y": "y_2y.csv"}
# Ext case suffix: the Int transfer file whose RB lines become external lines
CASES = {"": "robg", "_bssc": "robg_bssc", "_geco": "robg_geco"}
# EU price cases: which price files the sensitivity replaces
EU_CASES = {"eu_very_low": ("pTradePrice", "pTradePriceExport"),
            "eu_crisis": ("pTradePrice", "pTradePriceExport"),
            "eu_central_cbam": ("pTradePriceExport",)}


# ---------------------------------------------------------------------------- tables
class Table:
    """A CSV kept as text cells, with the byte order mark and the line ending of its source."""

    def __init__(self, bom, header, rows, eol="\n"):
        self.bom, self.header, self.rows, self.eol = bom, list(header), [list(r) for r in rows], eol

    @classmethod
    def read(cls, rel):
        raw = (DATA / rel).read_bytes().decode("utf-8")
        bom = raw.startswith(BOM)
        rows = list(csv.reader(io.StringIO(raw[1:] if bom else raw, newline="")))
        return cls(bom, rows[0], rows[1:], "\r\n" if "\r\n" in raw else "\n")

    def text(self):
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator=self.eol)
        w.writerow(self.header)
        w.writerows(self.rows)
        return (BOM if self.bom else "") + buf.getvalue()

    def col(self, name):
        return self.header.index(name)

    def keep(self, pred):
        return Table(self.bom, self.header, [r for r in self.rows if pred(r)], self.eol)

    def drop_columns(self, names):
        idx = [self.col(n) for n in names]
        keep = [j for j in range(len(self.header)) if j not in idx]
        return Table(self.bom, [self.header[j] for j in keep], [[r[j] for j in keep] for r in self.rows], self.eol)

    def copy(self):
        return Table(self.bom, self.header, self.rows, self.eol)

    def index(self, key="g"):
        j = self.col(key)
        idx = {}
        for r in self.rows:
            if r[j] in idx:
                raise SystemExit(f"duplicate {key} {r[j]!r}")
            idx[r[j]] = r
        return idx


def roundtrip(rel):
    """The serialisation reproduces the source byte for byte, so a dropped-row file is a textual subset."""
    return Table.read(rel).text().encode("utf-8") == (DATA / rel).read_bytes()


def grid_years(token):
    return [int(r[0]) for r in Table.read(GRIDS[token]).rows]


def round_up(value, years):
    """The first grid year at or after value; blank or beyond the grid stays as typed."""
    if value == "":
        return value
    y = int(float(value))
    after = [g for g in years if g >= y]
    return str(after[0]) if after else value


def num_equal(a, b):
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def cell_diff(a, b, key="g"):
    """Cells of b that differ from a, as (key, column, a value, b value). Same header and same keys required."""
    if a.header != b.header:
        raise SystemExit("cell_diff: headers differ")
    ia, ib = a.index(key), b.index(key)
    if list(ia) != list(ib):
        raise SystemExit("cell_diff: key sets or orders differ")
    out = []
    for k, r in ia.items():
        s = ib[k]
        for j, c in enumerate(a.header):
            if not num_equal(r[j], s[j]):
                out.append((k, c, r[j], s[j]))
    return out


def apply_edits(table, edits, styr_rule=None, key="g"):
    t = table.copy()
    idx = t.index(key)
    for k, c, _, new in edits:
        j = t.col(c)
        idx[k][j] = styr_rule(k, new) if (styr_rule and c == "StYr") else new
    return t


# ---------------------------------------------------------------------------- plant files
def grid_consistency(fine, coarse, years_coarse, status_table, log, label):
    """Every cell equal, except StYr of committed plants: round_up(fine) == coarse. Returns the unexplained cells."""
    si, ci = fine.index(), coarse.index()
    st = status_table.index()
    jstat, jstyr = status_table.col("Status"), fine.col("StYr")
    bad = []
    for g in si:
        for j, c in enumerate(fine.header):
            a, b = si[g][j], ci[g][j]
            if num_equal(a, b):
                continue
            if j == jstyr and st[g][jstat] == "2" and round_up(a, years_coarse) == b:
                continue
            bad.append((g, c, a, b))
    log.append(f"  {label}: {len(si)} plants, {len(bad)} cells differ beyond the committed StYr rounding")
    return bad


def build_plants(log, out):
    y5, y2 = grid_years("s5"), grid_years("2y")
    v2s5, v2y, v4s5 = Table.read("supply/pGenDataInput_v2_s5.csv"), Table.read("supply/pGenDataInput_v2_2y.csv"), Table.read("supply/pGenDataInput_v4_s5.csv")
    reshi = Table.read("supply/pGenDataInput_v2_s5_reshi.csv")
    jz, jstat = v4s5.col("z"), v4s5.col("Status")
    idx4, idx2 = v4s5.index(), v2s5.index()

    log.append("Plant files")
    bad = grid_consistency(v2y, v2s5, y5, v2s5, log, "v2_2y vs v2_s5 (the grid rule of the v2 files)")
    for g, c, a, b in bad[:10]:
        log.append(f"    {g} {c}: 2y {a}, s5 {b}")

    v4_edits = cell_diff(v2s5, v4s5)
    log.append(f"  v4 edits (v4_s5 vs v2_s5): {len(v4_edits)} cells on {len({g for g, *_ in v4_edits})} plants, "
               f"zones {sorted({idx4[g][jz] for g, *_ in v4_edits})}")
    outside = sorted({g for g, *_ in v4_edits if idx4[g][jz] not in RB})
    log.append(f"  v4 edits outside Romania and Bulgaria: {outside or 'none'}")
    for g, c, a, b in v4_edits:
        log.append(f"    {g} ({idx4[g][jz]}) {c}: {a} -> {b}")

    def styr_2y(g, new):
        # a committed plant's s5 StYr is a rounded value; the 2y file takes the first 2y year at or after it
        return round_up(new, y2) if idx4[g][jstat] == "2" else new

    v4_2y = apply_edits(v2y, v4_edits, styr_2y)
    for g, c, a, b in v4_edits:
        if c == "StYr" and idx4[g][jstat] == "2":
            log.append(f"  note: {g} is committed, its 5y StYr {b} is a rounded value; the 2y file takes {styr_2y(g, b)} "
                       f"(2y years {[y for y in y2 if b and int(b) - 5 < y <= int(b)]} all round to {b} on the 5y grid)")
    bad = grid_consistency(v4_2y, v4s5, y5, v4s5, log, "v4_2y vs v4_s5 (the grid rule of the v4 files)")
    for g, c, a, b in bad:
        was = idx2[g][jstat]
        why = " (candidate since v4, committed in v2: the 2y file keeps the real year, the 5y file the rounded one)" if (was == "2" and idx4[g][jstat] == "3") else ""
        log.append(f"    {g} {c}: 2y {a}, s5 {b}{why}")
    out["supply/pGenDataInput_v4_2y.csv"] = v4_2y

    reshi_edits = cell_diff(v2s5, reshi)
    cols = sorted({c for _, c, *_ in reshi_edits})
    plants = {g for g, *_ in reshi_edits}
    overlap = sorted({(g, c) for g, c, *_ in reshi_edits} & {(g, c) for g, c, *_ in v4_edits})
    log.append(f"  ResHi edits (v2_s5_reshi vs v2_s5): {len(reshi_edits)} cells on {len(plants)} plants, columns {cols}, "
               f"zones {sorted({idx4[g][jz] for g in plants})}; cells also edited by v4: {overlap or 'none'}")
    v4_s5_reshi = apply_edits(v4s5, reshi_edits)
    v4_2y_reshi = apply_edits(v4_2y, reshi_edits)
    bad = grid_consistency(v4_2y_reshi, v4_s5_reshi, y5, v4s5, log, "v4_2y_reshi vs v4_s5_reshi")
    log.append(f"    (the same {len(bad)} StYr cells as between v4_2y and v4_s5)" if bad else "")
    out["supply/pGenDataInput_v4_s5_reshi.csv"] = v4_s5_reshi
    out["supply/pGenDataInput_v4_2y_reshi.csv"] = v4_2y_reshi

    for token, t in (("s5", v4s5), ("2y", v4_2y)):
        ext = t.keep(lambda r: r[jz] not in RB)
        log.append(f"  pGenDataInput_v4_{token}_ext: {len(ext.rows)} of {len(t.rows)} plants kept "
                   f"({len(t.rows) - len(ext.rows)} in Romania and Bulgaria dropped)")
        out[f"supply/pGenDataInput_v4_{token}_ext.csv"] = ext

    # storage
    base, s5 = Table.read("supply/pStorageDataInput.csv"), Table.read("supply/pStorageDataInput_s5.csv")
    bad = grid_consistency(base, s5, y5, base, log, "pStorageDataInput vs pStorageDataInput_s5")
    jz, jstat, jstyr = base.col("z"), base.col("Status"), base.col("StYr")
    committed = [(r[0], r[jstyr]) for r in base.rows if r[jstat] == "2"]
    off = [(g, y) for g, y in committed if int(y) not in y2]
    log.append(f"  committed storage units and their StYr: {committed}; off the 2y grid: {off or 'none'}, "
               f"so pStorageDataInput_2y is a copy of pStorageDataInput")
    out["supply/pStorageDataInput_2y.csv"] = base.copy()
    for token, t in (("s5", s5), ("2y", base)):
        ext = t.keep(lambda r: r[jz] not in RB)
        log.append(f"  pStorageDataInput_{token}_ext: {len(ext.rows)} of {len(t.rows)} units kept")
        out[f"supply/pStorageDataInput_{token}_ext.csv"] = ext


# ---------------------------------------------------------------------------- small zone indexed files
def build_small(log, out):
    log.append("Zone maps, fuel, reserve, fuel limit, trade share")
    for src, dst, ref in (("zcmap_robg.csv", "zcmap_ext.csv", "zcmap.csv"), ("zcmap_robg_geco.csv", "zcmap_ext_geco.csv", "zcmap_geco.csv")):
        t = Table.read(src)
        jc = t.col("c")
        ext = t.keep(lambda r: r[jc] not in RB)
        same = ext.rows == Table.read(ref).rows if (DATA / ref).is_file() else None
        log.append(f"  {dst}: {len(ext.rows)} of {len(t.rows)} zones; rows equal to the LC {ref}: {same}")
        out[dst] = ext
    t = Table.read("supply/pFuelPrice_robg.csv")
    j = t.col("country")
    out["supply/pFuelPrice_ext.csv"] = t.keep(lambda r: r[j] not in RB)
    log.append(f"  pFuelPrice_ext: {len(out['supply/pFuelPrice_ext.csv'].rows)} of {len(t.rows)} rows")
    t = Table.read("reserve/pPlanningReserveMargin_robg.csv")
    j = t.col("c")
    out["reserve/pPlanningReserveMargin_ext.csv"] = t.keep(lambda r: r[j] not in RB)
    log.append(f"  pPlanningReserveMargin_ext: {[r[0] for r in out['reserve/pPlanningReserveMargin_ext.csv'].rows]}")
    t = Table.read("constraint/pMaxFuellimit_robg.csv")
    j = t.col("z")
    out["constraint/pMaxFuellimit_ext.csv"] = t.keep(lambda r: r[j] not in RB)
    log.append(f"  pMaxFuellimit_ext: {[(r[0], r[1]) for r in out['constraint/pMaxFuellimit_ext.csv'].rows]}")
    t = Table.read("trade/pMaxAnnualExternalTradeShare_robg.csv")
    ext = t.drop_columns(list(RB))
    ref = Table.read("trade/pMaxAnnualExternalTradeShare.csv")
    same = ext.header == ref.header and all(all(num_equal(a, b) for a, b in zip(r, s)) for r, s in zip(ext.rows, ref.rows)) and len(ext.rows) == len(ref.rows)
    log.append(f"  pMaxAnnualExternalTradeShare_ext: columns {ext.header[1:]}; equal in value to the config default file: {same}")
    out["trade/pMaxAnnualExternalTradeShare_ext.csv"] = ext


# ---------------------------------------------------------------------------- lines
def build_lines(log, out):
    log.append("External zones and lines")
    robg = Table.read("trade/pExtTransferLimit_robg.csv")
    jz, jx, jq, jd = 0, 1, 2, 3
    if robg.header[:3] != ["z", "zext", "q"] or robg.header[3] != "":
        raise SystemExit(f"unexpected pExtTransferLimit header {robg.header[:5]}")
    years = robg.header[4:]
    kept = robg.keep(lambda r: r[jz] not in RB)
    dropped_keys = sorted({(r[jz], r[jx]) for r in robg.rows if r[jz] in RB})
    reached = [z for z in dict.fromkeys(r[jx] for r in kept.rows)]
    zr = Table.read("trade/zext_robg.csv")
    zext_robg = [r[0] for r in zr.rows]
    zext_ext = [z for z in zext_robg if z in reached] + list(RB)
    gone = [z for z in zext_robg if z not in reached]
    log.append(f"  pExtTransferLimit_robg: {len(robg.rows)} rows; rows with z in Romania or Bulgaria dropped: "
               f"{len(robg.rows) - len(kept.rows)} ({', '.join(f'{z} to {x}' for z, x in dropped_keys)})")
    log.append(f"  zext_ext: {zext_ext}; zones of zext_robg no longer reached: {gone}")
    out["trade/zext_ext.csv"] = Table(zr.bom, ["zext"], [[z] for z in zext_ext], zr.eol)

    for suffix, case in CASES.items():
        tl = Table.read(f"trade/pTransferLimit_{case}_hvdc.csv")
        if tl.header[:3] != ["z", "z2", "q"]:
            raise SystemExit(f"unexpected pTransferLimit header {tl.header[:4]}")
        ymap = [tl.header.index(y) for y in years]
        conv = {}
        both = set()
        for r in tl.rows:
            z, z2, q = r[0], r[1], r[2]
            vals = [r[j] for j in ymap]
            if z in RB and z2 in RB:
                both.add((z, z2))
            elif z2 in RB:
                conv[(z, z2, q, "Export")] = vals
            elif z in RB:
                conv[(z2, z, q, "Import")] = vals
        keys = sorted({(z, x, q) for z, x, q, _ in conv})
        rows = []
        missing = []
        for z, x, q in keys:
            for d in ("Import", "Export"):
                if (z, x, q, d) not in conv:
                    missing.append((z, x, q, d))
                    conv[(z, x, q, d)] = ["0"] * len(years)
                rows.append([z, x, q, d] + conv[(z, x, q, d)])
        ext = Table(robg.bom, robg.header, kept.rows + rows, robg.eol)
        name = f"trade/pExtTransferLimit_ext{suffix}.csv"
        out[name] = ext
        lines = sorted({(z, x) for z, x, _ in keys})
        summary = []
        for z, x in lines:
            for d in ("Import", "Export"):
                v = [float(a) for a in conv[(z, x, "Q1", d)]]
                first = next((y for y, a in zip(years, v) if a > 0), None)
                summary.append(f"{z} {d.lower()} {x}: {max(v):.0f} MW from {first}" if first else f"{z} {d.lower()} {x}: 0")
        log.append(f"  {name}: {len(kept.rows)} robg rows kept + {len(rows)} rows converted from pTransferLimit_{case}_hvdc "
                   f"({len(both)} line(s) between Romania and Bulgaria dropped{', one direction missing: ' + str(missing) if missing else ''})")
        for s in summary:
            log.append(f"    {s}")
        # informative: against the LC file of the same case
        lc = {"": "baseline", "_bssc": "bssc", "_geco": "geco"}[suffix]
        ref = DATA / f"trade/pExtTransferLimit_{lc}.csv"
        if ref.is_file():
            rt = Table.read(f"trade/pExtTransferLimit_{lc}.csv")
            ry = [rt.header.index(y) for y in years if y in rt.header]
            a = {(r[0], r[1], r[2], r[3]): [r[j] for j in ry] for r in rt.rows}
            b = {(r[0], r[1], r[2], r[3]): r[4:4 + len(ry)] for r in ext.rows}
            only_lc = sorted({k[:2] for k in a if k not in b})
            only_ext = sorted({k[:2] for k in b if k not in a})
            diff = sorted({k[:2] for k in a if k in b and not all(num_equal(x, y) for x, y in zip(a[k], b[k]))})
            log.append(f"    against the LC pExtTransferLimit_{lc}: lines only in LC {only_lc or 'none'}, only in Ext {only_ext or 'none'}, "
                       f"shared lines with different values {diff or 'none'}")

    # internal lines and losses: one Ext file, checked against every case
    for param, cases in (("pTransferLimit", CASES.values()), ("pLossFactorInternal", CASES.values())):
        derived = {}
        for case in cases:
            t = Table.read(f"trade/{param}_{case}_hvdc.csv")
            derived[case] = t.keep(lambda r: r[0] not in RB and r[1] not in RB)
        base = derived["robg"]
        same = {c: (d.header == base.header and d.rows == base.rows) for c, d in derived.items()}
        src = Table.read(f"trade/{param}_robg_hvdc.csv")
        log.append(f"  {param}_ext: {len(base.rows)} of {len(src.rows)} rows of {param}_robg_hvdc kept; the BSSC and GECO "
                   f"variants give the same rows once their Romania and Bulgaria rows are dropped: {same}")
        out[f"trade/{param}_ext.csv"] = base
        if not all(same.values()):
            for c, d in derived.items():
                if not same[c]:
                    out[f"trade/{param}_ext_{c.replace('robg_', '').replace('_hvdc', '')}.csv"] = d


# ---------------------------------------------------------------------------- prices
def build_prices(log, out, zext_ext):
    log.append("Trade prices")
    non_eu = [z for z in zext_ext if z not in EU]
    for param in ("pTradePrice", "pTradePriceExport"):
        robg = Table.read(f"trade/{param}_robg.csv")
        central = Table.read(f"trade/{param}_eu_central.csv")
        if robg.header != central.header:
            raise SystemExit(f"{param}: robg and eu_central headers differ")
        jx = robg.col("zext")
        rows = [r for r in robg.rows if r[jx] in non_eu + ["Greece"]] + [r for r in central.rows if r[jx] in RB]
        ext = Table(robg.bom, robg.header, rows, robg.eol)
        out[f"trade/{param}_ext.csv"] = ext
        blocks = {z: sum(1 for r in rows if r[jx] == z) for z in zext_ext}
        log.append(f"  {param}_ext: {len(rows)} rows; rows per zext {blocks}; Greece from robg, Romania and Bulgaria from eu_central")
        for case, params in EU_CASES.items():
            if param not in params:
                continue
            cf = Table.read(f"trade/{param}_{case}.csv")
            if cf.header != robg.header:
                raise SystemExit(f"{param}_{case}: header differs")
            rows = [r for r in robg.rows if r[jx] in non_eu] + [r for r in cf.rows if r[jx] in EU]
            out[f"trade/{param}_ext_{case}.csv"] = Table(robg.bom, robg.header, rows, robg.eol)
            # the non EU rows of the case file equal the robg rows (same source), checked
            a = {tuple(r[:4]): r[4:] for r in robg.rows if r[jx] in non_eu}
            b = {tuple(r[:4]): r[4:] for r in cf.rows if r[jx] in non_eu}
            same = a.keys() == b.keys() and all(all(num_equal(x, y) for x, y in zip(a[k], b[k])) for k in a)
            log.append(f"  {param}_ext_{case}: {len(rows)} rows; Greece, Romania and Bulgaria from {param}_{case}; "
                       f"non EU rows of that file equal to robg: {same}")


# ---------------------------------------------------------------------------- write
def emit(out, args, log):
    log.append("Files")
    problems = 0
    for rel, table in out.items():
        path = DATA / rel
        new = table.text().encode("utf-8")
        if path.exists():
            if path.read_bytes() == new:
                status = "unchanged"
            elif args.check:
                status, problems = "DIFFERS from the derivation", problems + 1
            elif args.force:
                path.write_bytes(new)
                status = "rewritten"
            else:
                status, problems = "DIFFERS from the derivation, kept (use --force)", problems + 1
        elif args.check:
            status, problems = "MISSING", problems + 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(new)
            status = "written"
        log.append(f"  {rel}: {len(table.rows)} rows, {status}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare the derived files with a fresh derivation, write nothing")
    ap.add_argument("--force", action="store_true", help="rewrite a derived file that differs from the derivation")
    args = ap.parse_args(argv)

    sources = ["supply/pGenDataInput_v2_s5.csv", "supply/pGenDataInput_v2_2y.csv", "supply/pGenDataInput_v4_s5.csv",
               "supply/pGenDataInput_v2_s5_reshi.csv", "supply/pStorageDataInput.csv", "supply/pStorageDataInput_s5.csv",
               "zcmap_robg.csv", "zcmap_robg_geco.csv", "supply/pFuelPrice_robg.csv", "reserve/pPlanningReserveMargin_robg.csv",
               "constraint/pMaxFuellimit_robg.csv", "trade/pMaxAnnualExternalTradeShare_robg.csv", "trade/zext_robg.csv",
               "trade/pExtTransferLimit_robg.csv", "trade/pTradePrice_robg.csv", "trade/pTradePriceExport_robg.csv"] + \
              [f"trade/{p}_{c}_hvdc.csv" for c in CASES.values() for p in ("pTransferLimit", "pLossFactorInternal")] + \
              [f"trade/{p}_{c}.csv" for c in ["eu_central"] + list(EU_CASES) for p in ("pTradePrice", "pTradePriceExport")
               if c != "eu_central_cbam" or p == "pTradePriceExport"]
    log = [f"# Ext scope and grid files derived from the v4 input set", "", f"Generated by tools/build_ext_inputs.py from {DATA.relative_to(ROOT)}.", ""]
    bad = [s for s in sources if not roundtrip(s)]
    log.append(f"Sources: {len(sources)} files; serialisation reproduces the source bytes for all but {bad or 'none'}")
    out = {}
    build_plants(log, out)
    build_small(log, out)
    build_lines(log, out)
    build_prices(log, out, [r[0] for r in out["trade/zext_ext.csv"].rows])
    clash = [rel for rel in out if rel in sources]
    if clash:
        raise SystemExit(f"a derived file would overwrite a source: {clash}")
    problems = emit(out, args, log)
    log.append(f"  {len(out)} derived files, {problems} problem(s)")
    text = "\n".join(line for line in log if line is not None)
    print(text)
    if not args.check:
        REPORT.write_text(text + "\n", encoding="utf-8")
        print(f"report: {REPORT.relative_to(ROOT)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
