"""
Derive the Ext scope files and the EU price cases of the v6 input family under epm/input/data_blacksea.

    python tools/build_ext_inputs_v6.py            # write the derived files, print and save the checks
    python tools/build_ext_inputs_v6.py --check    # compare the files on disk with a fresh derivation, write nothing
    python tools/build_ext_inputs_v6.py --force    # rewrite a derived file that differs from the derivation

Same contract as tools/build_ext_inputs.py, which stays the derivation of the v5 files. Every file written here is
derived from a v6 file that already exists in the data folder (written by pre-analysis/pipelines/build_robg_v6.py);
nothing is typed by hand and the sources are never modified. A derived file keeps the cell text, the column order,
the line ending and the byte order mark of its source. The checks are printed and saved to
internal_analysis/scenarios/ext_inputs_v6_report.md. The report holds counts and public EU prices only: no cell of a
plant, storage or fuel file is printed, because some v6 cells come from tables that are not public.

Ext scope. Romania, Bulgaria and Hungary (EU3 below) leave the zone map. Romania and Bulgaria come back as price
zones. Hungary is reached by Romania only and disappears, with the external zones that only the three reached.
  zcmap_ext_v6                              zcmap_robg_v6 without the EU3 rows
  supply/pGenDataInput_v6_{g}_ext           the v6 plant file without the plants of EU3
  supply/pStorageDataInput_v6_{g}_ext       the v6 storage file without the units of EU3
  supply/pFuelPrice_ext_v6                  pFuelPrice_robg_v6 without the EU3 countries
  reserve/pPlanningReserveMargin_ext_v6     pPlanningReserveMargin_robg_v6 without the EU3 countries
  trade/zext_ext_v6                         zext_robg_v6 without the zones no longer reached, plus Romania and Bulgaria
  trade/pExtTransferLimit_ext_v6            pExtTransferLimit_robg_v6 without the rows whose z is in EU3, plus the
                                            lines of pTransferLimit_robg_base_hvdc_v6 between a modelled zone and
                                            Romania or Bulgaria, turned into external lines: z -> RB becomes
                                            (z, RB, Export) and RB -> z becomes (z, RB, Import). Lines inside EU3
                                            disappear.
  trade/pLossFactorInternal_ext_v6          pLossFactorInternal_robg_hvdc_v6 without the rows touching EU3
  trade/pTransferLimit_ext_v6               pTransferLimit_robg_base_hvdc_v6 without the rows touching EU3
  trade/pMaxAnnualExternalTradeShare_ext_v6 the v6 share file without the EU3 columns
  trade/pTradePrice[Export]_ext_v6          the robg_v6 price rows of the zext kept (Greece included, as in the Int
                                            scope), plus the Romania and Bulgaria rows of the eu_central_v6 file
The fuel limit has no v6 source: the Ext scope keeps constraint/pMaxFuellimit_ext.csv, and this script checks that
the file still equals pMaxFuellimit_robg without the EU3 zones.

Project cases of the Ext scope. The Int case files come from pre-analysis/pipelines/build_cases_v6.py, which runs
before this script. The Base above is the case with the committed lines only.
  trade/pExtTransferLimit_ext_<case>_v6     bssc, geco, onland: the same conversion, on the limit file of the case
  trade/pTransferLimit_ext_<case>_v6        the Int file of the case without the rows touching EU3, written only
  trade/pLossFactorInternal_ext_<case>_v6   where it differs from the Ext Base file
  zcmap_ext_geco_v6                         zcmap_robg_geco_v6 without the EU3 rows: the zone map with the two hubs
  trade/pNewTransmission_ext_freeexp_v6     FreeExp: the candidates of the Int scope between two modelled zones
  trade/pNewTransmissionExt_ext_freeexp_v6  FreeExp: the external candidates of the Int scope, plus its candidates
                                            between a modelled zone and Romania or Bulgaria, written (zone, price zone)
An external candidate needs no row in the external limit file: FreeExp keeps the limit files of the Ext Base. A line
toward Romania or Bulgaria carries no loss factor in the Ext scope: the loss sits in the export price of the zone.

EU price cases. Six cases, each a pair of files trade/pTradePrice[Export]_ext_eu_<case>_v6.csv equal to the central
Ext pair except for the Greece, Romania and Bulgaria rows. A case with CBAM changes the export file only.
  hub        the central hub price of a row, read back from its import price: import / eur_usd - W
  ratio      very_low, low, high: the ratio of the v5 case to the v5 central, level_L.csv, by zone and year. It is
             the ratio the v5 case files apply to every hour of the year. crisis: the v5 crisis level over the
             weighted mean of the v6 hub, so that the crisis keeps its absolute level (the regime of 2021 to 2023 is
             a price level, not a multiple of the central). In 2024 and 2025 the v6 central is the v5 central and
             every case, crisis included, takes the v5 ratio.
  import     (hub x ratio + W) x eur_usd
  export     (hub x ratio x (1 - lambda) - W) x eur_usd - C, floored at 0.01 and never above the import price; at
             ratio 1 the price before C is the central export cell itself
  C          CBAM levy of the cases with CBAM: Annex III default emission factor of the exporter (cbam_levy.csv) x
             the CO2 price of the v6 family (TYNDP 2026, trace of edit 2), from the first year of the v5 levy on
W and lambda are those of eu_price_netback.csv for Romania and Bulgaria. Greece keeps the netback of its central row,
the one the Int scope uses (no W, the AC loss): between the central and a case only the price level moves. In v5 the
Greek rows of a case came from the EU case file and carried W, 2 USD/MWh above the central rows.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
import build_ext_inputs as v5  # noqa: E402  (Table, num_equal and roundtrip of the v5 derivation, unchanged)

Table, num_equal = v5.Table, v5.num_equal
DATA = v5.DATA
CONFIG = ROOT / "pre-analysis" / "config"
PRICES = ROOT / "pre-analysis" / "output_prices"
TRACE = PRICES / "robg_v6" / "edit7_ext_price_cases.csv"
REPORT = ROOT / "internal_analysis" / "scenarios" / "ext_inputs_v6_report.md"

EU3 = ("Romania", "Bulgaria", "Hungary")   # leave the zone map
RB = ("Romania", "Bulgaria")               # come back as price zones
EU = ("Greece", "Romania", "Bulgaria")     # external zones that follow the EU price case
GRIDS = ("s5", "2y")
PARAMS = ("pTradePrice", "pTradePriceExport")
HOURS = ["t%02d" % i for i in range(1, 25)]
PRICE_HEADER = ["zext", "q", "d", "year"] + HOURS

MONEY_YEAR = 2024
OBSERVED_YEARS = (2024, 2025)   # the v6 central keeps its v5 rows there
FLOOR_USD = 0.01                # a zero price switches the trade off in main.gms
DECIMALS = 3
LEVEL_SCENARIO = "EU_HIGH"      # the crisis is a price level
# case: (scenario of level_L.csv, CBAM levy on the export price, files written)
EU_CASES = {
    "very_low": ("VERY_LOW", False, PARAMS),
    "low": ("LOW", False, PARAMS),
    "high": ("HIGH", False, PARAMS),
    "crisis": (LEVEL_SCENARIO, False, PARAMS),
    "central_cbam": ("CENTRAL", True, ("pTradePriceExport",)),
    "very_low_cbam": ("VERY_LOW", True, ("pTradePriceExport",)),
}
V5_TOL = 0.005      # USD/MWh: a v6 case row of 2024 or 2025 against the v5 case file
NETBACK_TOL = 0.002  # USD/MWh: the netback read back from the central rows, years written by the v6 builder
RATIO_TOL = 0.001   # ratio read back from the written table
LEVEL_TOL = 0.05    # EUR/MWh: crisis level read back from the written table
SHOW_YEARS = (2025, 2030, 2032, 2036, 2040)

# Project cases. The Int files with a case token come from pre-analysis/pipelines/build_cases_v6.py.
INT_LIMIT = "trade/pTransferLimit_robg_%s_hvdc_v6.csv"
HUB_MAP = "zcmap_robg_geco_v6.csv"         # zone map with the two hubs
BASE_MAP = "zcmap_robg_v6.csv"
# suffix of the Ext files, token of the Int limit file, Int loss file, Int zone map
LINE_CASES = (("", "base", "trade/pLossFactorInternal_robg_hvdc_v6.csv", BASE_MAP),
              ("_bssc", "bssc", "trade/pLossFactorInternal_robg_bssc_hvdc_v6.csv", BASE_MAP),
              ("_geco", "geco", "trade/pLossFactorInternal_robg_geco_hvdc_v6.csv", HUB_MAP),
              ("_onland", "onland", "trade/pLossFactorInternal_robg_onland_hvdc_v6.csv", BASE_MAP))
FREE_INT = "trade/pNewTransmission_robg_freeexp_v6.csv"
FREE_INT_EXT = "trade/pNewTransmissionExt_robg_freeexp.csv"
FREE_LIMIT = "trade/pTransferLimit_robg_freeexp_hvdc_v6.csv"
FREE_LOSS = "trade/pLossFactorInternal_robg_geco_onland_hvdc_v6.csv"


def fail(msg):
    raise SystemExit("ERROR: " + msg)


def read_csv(path):
    """Rows of a csv as dicts; lines that start with # are comments."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def config_value(name, series, year=0):
    hit = [r for r in read_csv(CONFIG / name) if r["series"] == series and int(r["year"]) == year]
    if len(hit) != 1:
        fail("%s: %d rows for %s %s" % (name, len(hit), series, year))
    return float(hit[0]["value"])


def against_v5(table, rel, key, group=0):
    """How a derived table stands against its v5 Ext twin. Counts and zone names only, never a cell value."""
    if not (DATA / rel).is_file():
        return "no v5 file %s" % rel
    old = Table.read(rel)
    if old.header == table.header and old.rows == table.rows:
        return "equal to the v5 file %s" % rel
    if old.header == table.header:
        pairs = [(j, j) for j in range(len(table.header))]
        note = ""
    else:
        names = [c for c in table.header if c and c in old.header]
        pairs = [(table.col(c), old.col(c)) for c in names]
        note = " (headers differ, %d shared columns compared)" % len(names)
    kn, ko = [table.header.index(c) for c in key], [old.header.index(c) for c in key]
    new = {tuple(r[j] for j in kn): r for r in table.rows}
    was = {tuple(r[j] for j in ko): r for r in old.rows}
    only_new = [k for k in new if k not in was]
    only_old = [k for k in was if k not in new]
    changed = [k for k in new if k in was and any(not num_equal(new[k][a], was[k][b]) for a, b in pairs)]
    where = Counter([new[k][group] for k in only_new + changed] + [was[k][group] for k in only_old])
    return ("against the v5 file %s%s: %d rows only in v6, %d only in v5, %d with different cells; by %s: %s"
            % (rel, note, len(only_new), len(only_old), len(changed), table.header[group] or "first column",
               dict(sorted(where.items())) or "none"))


# ---------------------------------------------------------------------------- plants and storage
def build_plants(log, out):
    log.append("Plant and storage files")
    for token in GRIDS:
        for param, twin in (("pGenDataInput", "supply/pGenDataInput_v4_%s_ext.csv" % token),
                            ("pStorageDataInput", "supply/pStorageDataInput_%s_ext.csv" % token)):
            src = Table.read("supply/%s_v6_%s.csv" % (param, token))
            jz = src.col("z")
            ext = src.keep(lambda r: r[jz] not in EU3)
            gone = Counter(r[jz] for r in src.rows if r[jz] in EU3)
            rel = "supply/%s_v6_%s_ext.csv" % (param, token)
            out[rel] = ext
            log.append("  %s: %d of %d units kept; dropped %s" % (rel, len(ext.rows), len(src.rows), dict(sorted(gone.items()))))
            log.append("    " + against_v5(ext, twin, ["g"], group=jz))


# ---------------------------------------------------------------------------- small zone indexed files
def build_small(log, out):
    log.append("Zone map, fuel, reserve, fuel limit, trade share")
    t = Table.read("zcmap_robg_v6.csv")
    jc = t.col("c")
    ext = t.keep(lambda r: r[jc] not in EU3)
    out["zcmap_ext_v6.csv"] = ext
    log.append("  zcmap_ext_v6: %d of %d zones; %s" % (len(ext.rows), len(t.rows), against_v5(ext, "zcmap_ext.csv", ["z"])))
    zones = [r[t.col("z")] for r in ext.rows]

    t = Table.read("supply/pFuelPrice_robg_v6.csv")
    j = t.col("country")
    ext = t.keep(lambda r: r[j] not in EU3)
    out["supply/pFuelPrice_ext_v6.csv"] = ext
    log.append("  pFuelPrice_ext_v6: %d of %d rows; %s" % (len(ext.rows), len(t.rows),
               against_v5(ext, "supply/pFuelPrice_ext.csv", ["country", "fuel"])))

    t = Table.read("reserve/pPlanningReserveMargin_robg_v6.csv")
    j = t.col("c")
    ext = t.keep(lambda r: r[j] not in EU3)
    out["reserve/pPlanningReserveMargin_ext_v6.csv"] = ext
    log.append("  pPlanningReserveMargin_ext_v6: %s; %s" % ([r[j] for r in ext.rows],
               against_v5(ext, "reserve/pPlanningReserveMargin_ext.csv", ["c"])))

    t = Table.read("constraint/pMaxFuellimit_robg.csv")
    j = t.col("z")
    kept = t.keep(lambda r: r[j] not in EU3)
    old = Table.read("constraint/pMaxFuellimit_ext.csv")
    if kept.header != old.header or kept.rows != old.rows:
        fail("constraint/pMaxFuellimit_ext.csv is no longer pMaxFuellimit_robg without the EU3 zones")
    log.append("  pMaxFuellimit: no v6 source; constraint/pMaxFuellimit_ext.csv kept, equal to pMaxFuellimit_robg "
               "without the EU3 zones: True")

    t = Table.read("trade/pMaxAnnualExternalTradeShare_robg_v6.csv")
    ext = t.drop_columns(list(EU3))
    out["trade/pMaxAnnualExternalTradeShare_ext_v6.csv"] = ext
    log.append("  pMaxAnnualExternalTradeShare_ext_v6: columns %s; %s" % (ext.header[1:],
               against_v5(ext, "trade/pMaxAnnualExternalTradeShare_ext.csv", [ext.header[0]])))
    return zones


# ---------------------------------------------------------------------------- lines
def build_lines(log, out, zones):
    log.append("External zones and lines")
    robg = Table.read("trade/pExtTransferLimit_robg_v6.csv")
    if robg.header[:3] != ["z", "zext", "q"] or robg.header[3] != "":
        fail("unexpected pExtTransferLimit header %s" % robg.header[:5])
    years = robg.header[4:]
    kept = robg.keep(lambda r: r[0] not in EU3)
    dropped = sorted({(r[0], r[1]) for r in robg.rows if r[0] in EU3})
    stray = sorted({r[0] for r in kept.rows if r[0] not in zones})
    if stray:
        fail("pExtTransferLimit_robg_v6 holds rows of zones outside the Ext zone map: %s" % stray)
    reached = list(dict.fromkeys(r[1] for r in kept.rows))
    zr = Table.read("trade/zext_robg_v6.csv")
    zext_robg = [r[0] for r in zr.rows]
    zext_ext = [z for z in zext_robg if z in reached] + list(RB)
    gone = [z for z in zext_robg if z not in reached]
    log.append("  pExtTransferLimit_robg_v6: %d rows; rows with z in EU3 dropped: %d (%s)"
               % (len(robg.rows), len(robg.rows) - len(kept.rows), ", ".join("%s to %s" % k for k in dropped)))
    log.append("  zext_ext_v6: %s; zones of zext_robg_v6 no longer reached: %s" % (zext_ext, gone))
    zx = Table(zr.bom, zr.header, [[z] for z in zext_ext], zr.eol)
    out["trade/zext_ext_v6.csv"] = zx
    log.append("    " + against_v5(zx, "trade/zext_ext.csv", [zr.header[0]]))

    hub = Table.read(HUB_MAP)
    jc, jz = hub.col("c"), hub.col("z")
    hub = hub.keep(lambda r: r[jc] not in EU3)
    hub_zones = [r[jz] for r in hub.rows]
    if not set(zones) < set(hub_zones):
        fail("%s does not hold the zones of the Ext scope plus the hubs" % HUB_MAP)
    out["zcmap_ext_geco_v6.csv"] = hub
    log.append("  zcmap_ext_geco_v6: the %d zones of the Ext scope plus %s; %s"
               % (len(zones), [z for z in hub_zones if z not in zones], against_v5(hub, "zcmap_ext_geco.csv", ["z"])))

    base = {}
    for suffix, token, loss_rel, zmap in LINE_CASES:
        name = "ext%s_v6" % suffix
        src_rel = INT_LIMIT % token
        log.append("  case %s" % (token if suffix else "base (committed lines only)"))
        rows, conv, inside, missing = to_external(Table.read(src_rel), years, hub_zones if zmap == HUB_MAP else zones, src_rel)
        ext = Table(robg.bom, robg.header, kept.rows + rows, robg.eol)
        if suffix and ext.rows == base["pExtTransferLimit"].rows:
            fail("case %s adds no line toward a price zone" % token)
        base.setdefault("pExtTransferLimit", ext)
        out["trade/pExtTransferLimit_%s.csv" % name] = ext
        log.append("    pExtTransferLimit_%s: %d robg_v6 rows kept + %d rows converted from %s; lines inside EU3 dropped: %s%s"
                   % (name, len(kept.rows), len(rows), src_rel, inside,
                      "; one direction missing, set to 0: %s" % missing if missing else ""))
        for z, x in sorted({k[:2] for k in conv}):
            for d in ("Import", "Export"):
                v = [float(a) for a in conv[(z, x, "Q1", d)]]
                first = next((y for y, a in zip(years, v) if a > 0), None)
                log.append("      %s %s %s: %s" % (z, d.lower(), x, "%.0f MW from %s" % (max(v), first) if first else "0"))
        log.append("      " + against_v5(ext, "trade/pExtTransferLimit_ext%s.csv" % suffix, ["z", "zext", "q", ""]))
        for param, rel, key in (("pTransferLimit", src_rel, ["z", "z2", "q"]), ("pLossFactorInternal", loss_rel, ["z", "z2"])):
            src = Table.read(rel)
            ext = src.keep(lambda r: r[0] not in EU3 and r[1] not in EU3)
            if not suffix:
                base[param] = ext
            elif ext.header == base[param].header and ext.rows == base[param].rows:
                log.append("    %s: the Ext Base file (%s without the rows touching EU3 equals it)" % (param, rel))
                continue
            out["trade/%s_%s.csv" % (param, name)] = ext
            log.append("    %s_%s: %d of %d rows of %s kept; %s"
                       % (param, name, len(ext.rows), len(src.rows), rel,
                          against_v5(ext, "trade/%s_ext%s.csv" % (param, suffix), key)))
    return zext_ext, hub_zones


def to_external(tl, years, zones, rel):
    """Rows of an Int limit file between a modelled zone and a price zone, as external limit rows."""
    if tl.header[:3] != ["z", "z2", "q"]:
        fail("unexpected pTransferLimit header %s" % tl.header[:4])
    ymap = [tl.header.index(y) for y in years]
    conv, inside = {}, set()
    for r in tl.rows:
        z, z2, q = r[0], r[1], r[2]
        if z in EU3 and z2 in EU3:
            inside.add((z, z2))
        elif z in EU3 or z2 in EU3:
            eu, other = (z, z2) if z in EU3 else (z2, z)
            if eu not in RB:
                fail("%s, line between %s and %s: %s leaves the Ext scope and is not a price zone" % (rel, z, z2, eu))
            if other not in zones:
                fail("%s, line between %s and %s: %s is not in the zone map of the case" % (rel, z, z2, other))
            conv[(other, eu, q, "Import" if z in EU3 else "Export")] = [r[j] for j in ymap]
    rows, missing = [], []
    for z, x, q in sorted({k[:3] for k in conv}):
        for d in ("Import", "Export"):
            if (z, x, q, d) not in conv:
                missing.append((z, x, q, d))
                conv[(z, x, q, d)] = ["0"] * len(years)
            rows.append([z, x, q, d] + conv[(z, x, q, d)])
    return rows, conv, sorted(inside), missing


def build_free(log, out, zext_ext, hub_zones):
    """FreeExp in the Ext scope. Pair names and counts only: some candidate cells come from a table that is not public."""
    log.append("FreeExp candidates")
    src, xsrc = Table.read(FREE_INT), Table.read(FREE_INT_EXT)
    if src.header != xsrc.header or src.header[:2] != ["From", "To"]:
        fail("%s and %s are expected to share the pNewTransmission header" % (FREE_INT, FREE_INT_EXT))
    internal = src.keep(lambda r: r[0] not in EU3 and r[1] not in EU3)
    stray = [r[:2] for r in internal.rows if r[0] not in hub_zones or r[1] not in hub_zones]
    if stray:
        fail("%s: candidates outside the zone map with the hubs: %s" % (FREE_INT, stray))
    out["trade/pNewTransmission_ext_freeexp_v6.csv"] = internal
    log.append("  pNewTransmission_ext_freeexp_v6: %d of the %d candidates of %s, between two modelled zones: %s"
               % (len(internal.rows), len(src.rows), FREE_INT, ", ".join("%s to %s" % (r[0], r[1]) for r in internal.rows)))
    rows = [list(r) for r in xsrc.rows if r[0] not in EU3]
    gone = ["%s to %s" % (r[0], r[1]) for r in xsrc.rows if r[0] in EU3]
    turned = []
    for r in src.rows:
        if (r[0] in EU3) == (r[1] in EU3):
            if r[0] in EU3:
                gone.append("%s to %s" % (r[0], r[1]))
            continue
        eu, other = (r[0], r[1]) if r[0] in EU3 else (r[1], r[0])
        if eu not in RB:
            fail("%s, candidate between %s and %s: %s leaves the Ext scope and is not a price zone" % (FREE_INT, r[0], r[1], eu))
        rows.append([other, eu] + list(r[2:]))
        turned.append("%s to %s" % (other, eu))
    pairs = [(r[0], r[1]) for r in rows]
    if len(set(pairs)) != len(pairs):
        fail("FreeExp, external candidates: a pair comes twice")
    bad = [p for p in pairs if p[0] not in hub_zones or p[1] not in zext_ext]
    if bad:
        fail("FreeExp, external candidates outside the zone map or the external zones: %s" % bad)
    out["trade/pNewTransmissionExt_ext_freeexp_v6.csv"] = Table(xsrc.bom, xsrc.header, rows, xsrc.eol)
    log.append("  pNewTransmissionExt_ext_freeexp_v6: %d candidates: %d of %s, and %d of %s turned into external lines (%s)%s"
               % (len(rows), len(rows) - len(turned), FREE_INT_EXT, len(turned), FREE_INT, ", ".join(turned),
                  "; dropped: %s" % ", ".join(gone) if gone else ""))

    # FreeExp keeps the limit files of the Ext Base and the loss file of the Onland case
    years = out["trade/pExtTransferLimit_ext_v6.csv"].header[4:]
    extra, _, _, _ = to_external(Table.read(FREE_LIMIT), years, hub_zones, FREE_LIMIT)
    base_ext = {tuple(r[:4]): r[4:] for r in out["trade/pExtTransferLimit_ext_v6.csv"].rows}
    for r in extra:
        ref = base_ext.get(tuple(r[:4]))
        if ref is None and any(float(v) != 0 for v in r[4:]) or ref is not None and not all(num_equal(a, b) for a, b in zip(r[4:], ref)):
            fail("%s: a line toward a price zone is neither in the Ext Base nor at zero (%s)" % (FREE_LIMIT, r[:4]))
    lim = Table.read(FREE_LIMIT).keep(lambda r: r[0] not in EU3 and r[1] not in EU3)
    if lim.rows != out["trade/pTransferLimit_ext_v6.csv"].rows:
        fail("%s without the rows touching EU3 is not the Ext Base limit file" % FREE_LIMIT)
    loss = Table.read(FREE_LOSS).keep(lambda r: r[0] not in EU3 and r[1] not in EU3)
    if "trade/pLossFactorInternal_ext_onland_v6.csv" not in out or loss.rows != out["trade/pLossFactorInternal_ext_onland_v6.csv"].rows:
        fail("%s without the rows touching EU3 is not the loss file of the Ext Onland case" % FREE_LOSS)
    have = {(r[0], r[1]) for r in loss.rows}
    lost = [r[:2] for r in internal.rows if (r[0], r[1]) not in have or (r[1], r[0]) not in have]
    if lost:
        fail("FreeExp: no loss factor in both directions for %s" % lost)
    log.append("  limits: the Ext Base files (the lines %s adds toward a price zone are at zero, and it equals "
               "pTransferLimit_ext_v6 without the rows touching EU3)" % FREE_LIMIT)
    log.append("  losses: pLossFactorInternal_ext_onland_v6 (equal to %s without the rows touching EU3); every internal "
               "candidate holds a loss factor in both directions" % FREE_LOSS)


# ---------------------------------------------------------------------------- prices
def build_central_prices(log, out, zext_ext):
    log.append("Trade prices, central")
    non_eu = [z for z in zext_ext if z not in EU]
    central = {}
    for param in PARAMS:
        robg = Table.read("trade/%s_robg_v6.csv" % param)
        cen = Table.read("trade/%s_eu_central_v6.csv" % param)
        if robg.header != PRICE_HEADER or cen.header != PRICE_HEADER:
            fail("%s: unexpected header in the robg_v6 or eu_central_v6 file" % param)
        rows = [r for r in robg.rows if r[0] in non_eu + ["Greece"]] + [r for r in cen.rows if r[0] in RB]
        blocks = Counter(r[0] for r in rows)
        if set(blocks) != set(zext_ext) or len(set(blocks.values())) != 1:
            fail("%s_ext_v6: rows per zext %s for the zones %s" % (param, dict(blocks), zext_ext))
        central[param] = Table(robg.bom, robg.header, rows, robg.eol)
        out["trade/%s_ext_v6.csv" % param] = central[param]
        log.append("  %s_ext_v6: %d rows, %d per zext; Greece from robg_v6, Romania and Bulgaria from eu_central_v6"
                   % (param, len(rows), blocks[zext_ext[0]]))
        # the non EU rows are those of the v5 Ext file: the v6 family moves the EU borders only
        old = Table.read("trade/%s_ext.csv" % param)
        a = {tuple(r[:4]): r[4:] for r in rows if r[0] in non_eu}
        b = {tuple(r[:4]): r[4:] for r in old.rows if r[0] in non_eu}
        same = a.keys() == b.keys() and all(all(num_equal(x, y) for x, y in zip(a[k], b[k])) for k in a)
        log.append("    non EU rows (%s) equal to the v5 file %s_ext: %s" % (", ".join(non_eu), param, same))
    if [r[:4] for r in central[PARAMS[0]].rows] != [r[:4] for r in central[PARAMS[1]].rows]:
        fail("the central import and export files of the Ext scope do not share their keys")
    return central


def weighted_mean(rows, weights):
    """{(zext, year): mean of the 24 hour cells of the rows, weighted by the hours of pHours.csv}."""
    acc = {}
    for r in rows:
        w = weights[(r[1], r[2])]
        a = acc.setdefault((r[0], int(r[3])), [0.0, 0.0])
        a[0] += sum(float(c) * x for c, x in zip(r[4:], w))
        a[1] += sum(w)
    return {k: s / n for k, (s, n) in acc.items()}


def floor_share(rows, weights):
    """{(zext, year): share of the hours of the year whose cell sits at the floor}."""
    acc = {}
    for r in rows:
        w = weights[(r[1], r[2])]
        a = acc.setdefault((r[0], int(r[3])), [0.0, 0.0])
        a[0] += sum(x for c, x in zip(r[4:], w) if float(c) <= FLOOR_USD + 1e-9)
        a[1] += sum(w)
    return {k: s / n for k, (s, n) in acc.items()}


def cbam_levy(log, usd):
    """{(zext, year): levy in USD/MWh}: Annex III factor of the exporter x the CO2 price of the v6 family."""
    rows = read_csv(PRICES / "cbam_levy.csv")
    factors = {r["country"]: r["ef_electricity"] for r in read_csv(CONFIG / "cbam_emission_factors.csv")}
    ef, first, exporter = {}, {}, {}
    for r in rows:
        z, y = r["zone"], int(r["year"])
        ef.setdefault(z, set()).add(float(r["ef_tco2_per_mwh"]))
        exporter.setdefault(z, set()).add(r["exporter"])
        if float(r["C_usd2024_per_mwh"]) > 0:
            first[z] = min(first.get(z, y), y)
    if set(ef) != set(EU) or any(len(v) != 1 for v in ef.values()) or any(len(v) != 1 for v in exporter.values()):
        fail("cbam_levy.csv: one exporter and one emission factor expected for each of %s" % (EU,))
    ef = {z: v.pop() for z, v in ef.items()}
    exporter = {z: v.pop() for z, v in exporter.items()}
    for z in EU:
        if abs(float(factors[exporter[z]]) - ef[z]) > 1e-9:
            fail("cbam_levy.csv: factor of %s differs from cbam_emission_factors.csv" % exporter[z])
    if len(set(first.values())) != 1:
        fail("cbam_levy.csv: the levy does not start in the same year for every zone: %s" % first)
    start = first[EU[0]]
    co2 = {int(r["year"]): float(r["co2_usd_t_v6"]) for r in read_csv(PRICES / "robg_v6" / "edit2_gas_co2_paths.csv")}
    # the v6 CO2 path starts from the v5 path: the observed years must agree with the levy table
    for r in rows:
        y = int(r["year"])
        if y in OBSERVED_YEARS and abs(co2[y] - float(r["ets_eur2024_per_t"]) * usd) > 0.01:
            fail("CO2 of %d: %.3f USD/t in the v6 path, %.3f in cbam_levy.csv" % (y, co2[y], float(r["ets_eur2024_per_t"]) * usd))
    log.append("  CBAM levy: from %d; exporter and Annex III factor: %s" % (start, ", ".join(
        "%s via %s %.3f t/MWh" % (z, exporter[z], ef[z]) for z in EU)))
    log.append("  CO2 of the v6 family, USD/t: " + ", ".join("%d %.1f" % (y, co2[y]) for y in SHOW_YEARS))
    return {(z, y): (ef[z] * co2[y] if y >= start else 0.0) for z in EU for y in co2}


def build_price_cases(log, out, traces, central):
    log.append("Trade prices, EU cases")
    usd = config_value("price_deflators.csv", "eur_usd", MONEY_YEAR)
    lam = config_value("eu_price_netback.csv", "lambda")
    lam_ro = config_value("eu_price_netback.csv", "lambda_Romania")
    wheel = config_value("eu_price_netback.csv", "W")
    netback = {z: (wheel if z in RB else 0.0, lam_ro if z == "Romania" else lam) for z in EU}
    log.append("  eur_usd %d %.4f; netback (W in EUR/MWh, loss): %s" % (MONEY_YEAR, usd, ", ".join(
        "%s %.2f and %.3f" % (z, netback[z][0], netback[z][1]) for z in EU)))
    ph = Table.read("pHours.csv")
    if ph.header != ["q", "d"] + HOURS:
        fail("unexpected pHours.csv header")
    weights = {(r[0], r[1]): [float(x) for x in r[2:]] for r in ph.rows}
    level = {(r["zone"], r["scenario"], int(r["year"])): float(r["L_eur2024"]) for r in read_csv(PRICES / "level_L.csv")}
    levy = cbam_levy(log, usd)

    imp, exp = central[PARAMS[0]], central[PARAMS[1]]
    keys = [(r[0], r[1], r[2], int(r[3])) for r in imp.rows]
    if set((k[1], k[2]) for k in keys) != set(weights):
        fail("the price files and pHours.csv do not hold the same representative days")
    hub = {k: [float(c) / usd - netback[k[0]][0] for c in r[4:]] for k, r in zip(keys, imp.rows) if k[0] in EU}
    mean_imp = weighted_mean([r for r in imp.rows if r[0] in EU], weights)
    mean_hub = {(z, y): m / usd - netback[z][0] for (z, y), m in mean_imp.items()}
    years = sorted({k[3] for k in hub})

    # the v5 level table is the weighted mean of the v5 hub: the crisis level and the v6 hub are on the same basis
    gaps = {z: max(abs(mean_hub[(z, y)] - level[(z, "CENTRAL", y)]) for y in OBSERVED_YEARS) for z in EU}
    log.append("  weighted mean of the central hub in %s against the v5 central level, EUR/MWh: %s"
               % (" and ".join(str(y) for y in OBSERVED_YEARS), ", ".join("%s %.3f" % (z, gaps[z]) for z in EU)))

    def ratio(scenario, z, y):
        if scenario == "CENTRAL":
            return 1.0
        if scenario == LEVEL_SCENARIO and y not in OBSERVED_YEARS:
            return level[(z, scenario, y)] / mean_hub[(z, y)]
        return level[(z, scenario, y)] / level[(z, "CENTRAL", y)]

    def priced(scenario, with_levy, by_formula=False):
        """Import and export rows of a case, in the order of the central files.

        At ratio 1 the export price before the levy is the central cell itself, so that a case with CBAM is the
        central export file minus the levy in every year. by_formula prices the central by the formula instead:
        it is a control and is never written.
        """
        rows_i, rows_e = [], []
        for k, ri, re_ in zip(keys, imp.rows, exp.rows):
            z, q, d, y = k
            if z not in EU:
                rows_i.append(ri)
                rows_e.append(re_)
                continue
            r = ratio(scenario, z, y)
            c = levy[(z, y)] if with_levy else 0.0
            w, loss = netback[z]
            if r == 1.0 and c == 0.0 and not by_formula:
                rows_i.append(ri)
                rows_e.append(re_)
                continue
            h = [x * r if x > 0 else x for x in hub[k]]
            # a cell whose hub is not positive keeps its central import price
            vi = [max((x + w) * usd, FLOOR_USD) if x > 0 else float(a) for x, a in zip(h, ri[4:])]
            if r == 1.0 and not by_formula:
                base = [float(a) for a in re_[4:]]
            else:
                base = [(x * (1 - loss) - w) * usd for x in h]
            ve = [min(max(b - c, FLOOR_USD), max(i, FLOOR_USD)) for b, i in zip(base, vi)]
            rows_i.append(ri if r == 1.0 else ri[:4] + ["%.*f" % (DECIMALS, v) for v in vi])
            rows_e.append(re_[:4] + ["%.*f" % (DECIMALS, v) for v in ve])
        return rows_i, rows_e

    # the netback read back from the central rows: the formula at ratio 1 against the central export file
    _, formula_e = priced("CENTRAL", False, by_formula=True)
    worst = {}
    for k, a, b in zip(keys, exp.rows, formula_e):
        if k[0] in EU:
            period = "observed" if k[3] in OBSERVED_YEARS else ("blend" if k[3] < 2030 else "tyndp")
            g = max(abs(float(x) - float(y)) for x, y in zip(a[4:], b[4:]))
            worst[(k[0], period)] = max(worst.get((k[0], period), 0.0), g)
    log.append("  export price by the formula at ratio 1 against the central export file, largest gap in USD/MWh "
               "(2024 and 2025 / 2026 to 2029 / from 2030): " + ", ".join(
                   "%s %.3f / %.3f / %.3f" % (z, worst[(z, "observed")], worst[(z, "blend")], worst[(z, "tyndp")]) for z in EU))
    late = max(worst[(z, "tyndp")] for z in EU)
    if late > NETBACK_TOL:
        fail("the central export rows from 2030 are not the netback of their import rows: gap %.4f USD/MWh" % late)

    trace = []

    def record(case, scenario, with_levy, rows_i, rows_e):
        mi, me, fl = weighted_mean(rows_i, weights), weighted_mean(rows_e, weights), floor_share(rows_e, weights)
        for z in EU:
            for y in years:
                trace.append([case, z, str(y), "%.4f" % ratio(scenario, z, y), "%.3f" % (levy[(z, y)] if with_levy else 0.0),
                              "%.3f" % mi[(z, y)], "%.3f" % me[(z, y)], "%.4f" % fl[(z, y)]])
        return mi, me, fl

    summary = {"central": record("central", "CENTRAL", False, imp.rows, exp.rows)}
    built = {}
    for case, (scenario, with_levy, params) in EU_CASES.items():
        rows_i, rows_e = priced(scenario, with_levy)
        built[case] = (rows_i, rows_e)
        summary[case] = record(case, scenario, with_levy, rows_i, rows_e)
        for param, rows in zip(PARAMS, (rows_i, rows_e)):
            if param in params:
                out["trade/%s_ext_eu_%s_v6.csv" % (param, case)] = Table(imp.bom, imp.header, rows, imp.eol)
        mi = summary[case][0]

        # 1. order of the cells, every EU row
        for ri, re_ in zip(rows_i, rows_e):
            if ri[0] in EU:
                for a, b in zip(ri[4:], re_[4:]):
                    if float(b) < FLOOR_USD - 1e-9 or float(b) > max(float(a), FLOOR_USD) + 1e-9:
                        fail("%s: an export cell of %s %s is below the floor or above the import price" % (case, ri[0], ri[3]))
        # 2. the ratio or the level, read back from the written import table
        for z in EU:
            w = netback[z][0]
            for y in years:
                got = mi[(z, y)] / usd - w
                if scenario == LEVEL_SCENARIO and y not in OBSERVED_YEARS:
                    if abs(got - level[(z, scenario, y)]) > LEVEL_TOL:
                        fail("%s %s %d: hub mean %.3f EUR/MWh, level %.3f" % (case, z, y, got, level[(z, scenario, y)]))
                elif y not in OBSERVED_YEARS and abs(got / mean_hub[(z, y)] - ratio(scenario, z, y)) > RATIO_TOL:
                    fail("%s %s %d: ratio %.4f read back, %.4f expected" % (case, z, y, got / mean_hub[(z, y)], ratio(scenario, z, y)))
        # 3. Romania and Bulgaria in 2024 and 2025: the v5 case rows
        gap = 0.0
        for param, rows in zip(PARAMS, (rows_i, rows_e)):
            name = "trade/%s_eu_%s.csv" % (param, case)
            if param not in params or not (DATA / name).is_file():
                continue
            old = {tuple(r[:4]): r[4:] for r in Table.read(name).rows if r[0] in RB and int(r[3]) in OBSERVED_YEARS}
            new = {tuple(r[:4]): r[4:] for r in rows if r[0] in RB and int(r[3]) in OBSERVED_YEARS}
            if old.keys() != new.keys():
                fail("%s: keys of 2024 and 2025 differ from the v5 file %s" % (case, name))
            gap = max([gap] + [abs(float(a) - float(b)) for k in old for a, b in zip(old[k], new[k])])
        if gap > V5_TOL:
            fail("%s: Romania and Bulgaria rows of 2024 and 2025 are %.4f USD/MWh away from the v5 case file" % (case, gap))
        log.append("  %s: files %s; Romania and Bulgaria 2024 and 2025 against the v5 case rows, largest gap %.4f USD/MWh"
                   % (case, " and ".join(p for p in PARAMS if p in params), gap))

    # 4. the levy, read back: a case with CBAM against the same case without it
    for case, base in (("central_cbam", exp.rows), ("very_low_cbam", built["very_low"][1])):
        worst_c = 0.0
        for k, a, b in zip(keys, base, built[case][1]):
            if k[0] not in EU:
                if a != b:
                    fail("%s: a non EU export row moved" % case)
                continue
            c = levy[(k[0], k[3])]
            for x, y in zip(a[4:], b[4:]):
                d = float(x) - float(y)
                if d > c + 0.0015 or d < -1e-9:
                    fail("%s %s %d: the export price falls by %.3f, levy %.3f" % (case, k[0], k[3], d, c))
                if float(y) > FLOOR_USD + 1e-9:
                    worst_c = max(worst_c, abs(d - c))
        if worst_c > 0.0015:
            fail("%s: a cell above the floor is %.4f USD/MWh away from its levy" % (case, worst_c))
        log.append("  %s: export price of every cell above the floor equals the price without CBAM minus the levy "
                   "(largest gap %.4f USD/MWh)" % (case, worst_c))

    log.append("  weighted mean of the import price, USD/MWh (case: " + ", ".join(str(y) for y in SHOW_YEARS) + ")")
    for z in EU:
        log.append("    %s" % z)
        for case in ["central"] + [c for c, v in EU_CASES.items() if PARAMS[0] in v[2]]:
            log.append("      %-10s %s" % (case, "  ".join("%7.1f" % summary[case][0][(z, y)] for y in SHOW_YEARS)))
    log.append("  export price: weighted mean in USD/MWh and share of the hours at the floor (case: "
               + ", ".join(str(y) for y in SHOW_YEARS) + ")")
    for z in EU:
        log.append("    %s" % z)
        for case in ["central"] + list(EU_CASES):
            log.append("      %-14s %s" % (case, "  ".join("%6.1f %3.0f%%" % (summary[case][1][(z, y)], 100 * summary[case][2][(z, y)])
                                                          for y in SHOW_YEARS)))
    head = ["case", "zext", "year", "ratio_to_central_hub", "cbam_levy_usd_mwh", "import_mean_usd_mwh", "export_mean_usd_mwh",
            "export_hours_at_floor"]
    traces[TRACE] = ("\n".join(",".join(r) for r in [head] + trace) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------- write
def emit(files, args, log):
    log.append("Files")
    problems = 0
    for path, payload in files.items():
        rel = path.relative_to(ROOT).as_posix()
        if path.exists():
            if path.read_bytes() == payload:
                status = "unchanged"
            elif args.check:
                status, problems = "DIFFERS from the derivation", problems + 1
            elif args.force:
                path.write_bytes(payload)
                status = "rewritten"
            else:
                status, problems = "DIFFERS from the derivation, kept (use --force)", problems + 1
        elif args.check:
            status, problems = "MISSING", problems + 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            status = "written"
        log.append("  %s: %d bytes, %s" % (rel, len(payload), status))
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare the derived files with a fresh derivation, write nothing")
    ap.add_argument("--force", action="store_true", help="rewrite a derived file that differs from the derivation")
    args = ap.parse_args(argv)

    sources = ["zcmap_robg_v6.csv", "supply/pFuelPrice_robg_v6.csv", "reserve/pPlanningReserveMargin_robg_v6.csv",
               "constraint/pMaxFuellimit_robg.csv", "trade/pMaxAnnualExternalTradeShare_robg_v6.csv", "trade/zext_robg_v6.csv",
               "trade/pExtTransferLimit_robg_v6.csv", HUB_MAP, FREE_INT, FREE_INT_EXT, FREE_LIMIT, FREE_LOSS] + \
              [INT_LIMIT % c[1] for c in LINE_CASES] + [c[2] for c in LINE_CASES] + \
              ["supply/%s_v6_%s.csv" % (p, g) for p in ("pGenDataInput", "pStorageDataInput") for g in GRIDS] + \
              ["trade/%s_%s_v6.csv" % (p, c) for p in PARAMS for c in ("robg", "eu_central")]
    log = ["# Ext scope files and EU price cases derived from the v6 input family", "",
           "Generated by tools/build_ext_inputs_v6.py from %s." % DATA.relative_to(ROOT).as_posix(), ""]
    bad = [s for s in sources if not v5.roundtrip(s)]
    log.append("Sources: %d files; serialisation reproduces the source bytes for all but %s" % (len(sources), bad or "none"))
    out, traces = {}, {}
    build_plants(log, out)
    zones = build_small(log, out)
    zext_ext, hub_zones = build_lines(log, out, zones)
    build_free(log, out, zext_ext, hub_zones)
    central = build_central_prices(log, out, zext_ext)
    build_price_cases(log, out, traces, central)
    clash = [rel for rel in out if rel in sources]
    if clash:
        fail("a derived file would overwrite a source: %s" % clash)
    files = {DATA / rel: table.text().encode("utf-8") for rel, table in out.items()}
    files.update(traces)
    problems = emit(files, args, log)
    log.append("  %d derived files, %d problem(s)" % (len(files), problems))
    text = "\n".join(log)
    print(text)
    if not args.check:
        REPORT.write_text(text + "\n", encoding="utf-8")
        print("report: %s" % REPORT.relative_to(ROOT).as_posix())
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
