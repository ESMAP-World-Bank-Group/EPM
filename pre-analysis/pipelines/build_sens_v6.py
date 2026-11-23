"""
Sensitivity files of the v6 input family, first lot (23 November 2026).

    python pre-analysis/pipelines/build_sens_v6.py            # write the files and the traces, print the checks
    python pre-analysis/pipelines/build_sens_v6.py --check    # compare the files on disk with a fresh build, write nothing

Order of the v6 builds: build_robg_v6.py (edits 1 to 6 and 9), build_cases_v6.py, tools/build_ext_inputs_v6.py, then
this script. It reads central files of v6.1 and files of v5, and writes new files only: no central file is touched.
A file keeps the column order, the line ending and the byte order mark of the file it starts from, and a row that is
not rewritten stays byte for byte. A second run changes nothing.

Each sensitivity moves one assumption of the Int scope against the central, on the five year grid:
  ArmeniaGasContract   pFuelPrice: the Armenian gas row goes back to the border contract path of v5, without the
                       margin that correction A3 adds in the central.
  ArmeniaDemandLow     pDemandForecast: the two Armenian rows go back to the forecast of v5, the one in use before
                       the country team update.
  CaucasusREHubs       pGenDataInput: the solar and wind candidates of the two GEC hub zones become committed units,
                       in the steps of the October hub file (2032, 2036, 2040). The steps replace the candidates, so
                       that the hubs hold their potential once. The October file kept both.
  EUCarbonLow          pFuelPrice: the CO2 line of Romania, Bulgaria and Hungary goes back to the path of v5
                       (TYNDP 2024); the commodity lines stay on v6. pTradePrice and pTradePriceExport: the rows of
                       the nine neighbours priced on TYNDP 2026 are scaled, year by year, by the cost of gas with CO2
                       on the low path over the same cost on the central path. The scaling keeps the hourly shape and
                       keeps a gas unit of the three countries as competitive against its neighbours as in the
                       central. It is a rule of this study, not a published one.
  TurkiyeCarbonPrice   pFuelPrice: from 2030 every Turkish fuel with a carbon content pays the CO2 price of the
                       central (the EU path of v6) on its content. An upper bound: the Turkish climate law sets up an
                       emission trading system without a price path.

The two EU price cases of the Ext scope that join the grid with this lot (low and high) are files of
tools/build_ext_inputs_v6.py.

Armenian figures are official use only: they stay in the files of the data folder (DVC). The log of this script and
the trace it keeps in git hold no Armenian value.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import build_robg_v6 as robg               # noqa: E402  text tables, the write and check contract, the v6 constants

EPM, DATA, SUPPLY, TRADE, LOAD, TRACE = robg.EPM, robg.DATA, robg.SUPPLY, robg.TRADE, robg.LOAD, robg.TRACE
fail, emit, text_table, text_payload, num = robg.fail, robg.emit, robg.text_table, robg.text_payload, robg.num

# ── Sources ─────────────────────────────────────────────────────────────────
FUEL_V6 = "pFuelPrice_robg_v6.csv"                  # central: commodity plus CO2 for Romania, Bulgaria and Hungary
FUEL_LINES = "pFuelPrice_robg_v6_lines.csv"         # the two lines of each priced row, written by edit 2
FUEL_V5_SUM, FUEL_V5_COM = "pFuelPrice_robg.csv", "pFuelPrice_tr_gas_flat.csv"     # v5: with CO2, commodity alone
PRICE_V6 = ("pTradePrice_robg_v6.csv", "pTradePriceExport_robg_v6.csv")            # central border prices, Int scope
HUB_STEPS = "pGenDataInput_v2_s5_hub100.csv"        # October file that holds the committed steps of the hubs
EU_ZONES = robg.FUEL_ZONES + (robg.FUEL_NEW_ZONE,)
HUB_ZONES = ("GEC_AZ", "GEC_GE")
TR_COUNTRY, TR_FROM = "Turkiye", 2030
TR_FUELS = ("Gas", "ImportedCoal", "DomesticCoal")  # gas, coal and lignite: the rule has to reach them
TR_CROSS = {"Gas": ("Romania", "Gas"), "DomesticCoal": ("Romania", "DomesticCoal")}   # same content, CO2 line of edit 2

# ── Outputs ─────────────────────────────────────────────────────────────────
AM_GAS_OUT = "pFuelPrice_robg_armgascontract_v6.csv"
AM_DEMAND_OUT = "pDemandForecast_v6_amdemlow.csv"
HUBS_TAG = "gechubs"
EU_LOW_FUEL = "pFuelPrice_robg_eucarbonlow_v6.csv"
EU_LOW_PRICE = ("pTradePrice_robg_eucarbonlow_v6.csv", "pTradePriceExport_robg_eucarbonlow_v6.csv")
TR_FUEL = "pFuelPrice_robg_trcarbon_v6.csv"
TRACE_FULL = DATA / "v6_sensitivities_20261123_trace.csv"     # every cell with its value before and after
TRACE_ROWS = TRACE / "sens_v6_rows.csv"                       # rows and basis, no value
TRACE_PATHS = TRACE / "sens_v6_carbon_paths.csv"              # CO2 paths, border factor, Turkish adders
TRACE_COLS = ["sensitivity", "file", "zone", "row", "field", "before", "after", "basis"]

BASIS = {
    "ArmeniaGasContract": "Border contract path of v5, without the margin of correction A3",
    "ArmeniaDemandLow": "Forecast of v5, before the country team update",
    "CaucasusREHubs": "Hub candidates committed in the steps of the October hub file; the steps replace the candidates",
    "EUCarbonLow": "CO2 line on the path of v5 (TYNDP 2024), commodity line of v6",
    "EUCarbonLow prices": "Row scaled by the cost of gas with CO2 on the low path over the central path (rule of the study)",
    "TurkiyeCarbonPrice": "CO2 price of the central from 2030 on the carbon content of the fuel (assumption, upper bound)",
}
PATH_TOL = 1e-3          # USD per tonne: the two CO2 paths in the years the central keeps from v5
MARGIN_TOL = 1.5e-4      # USD/MMBtu: spread of the Armenian margin over the years, rounding only
LINE_TOL = 1.5e-6        # USD/MMBtu: a Turkish adder against the CO2 line of the same content
STEP_TOL = 1e-6          # MW: steps of a hub unit against its candidate
SHOW = (2025, 2028, 2030, 2032, 2035, 2036, 2040, 2050)


def keyed(name, header, rows, n):
    """Cells of each row of a text table by its first n cells, in the order of the file."""
    width = len(header.split(","))
    out = {}
    for line in rows:
        p = line.split(",")
        if len(p) != width:
            fail("%s: bad width in '%s'" % (name, line[:40]))
        if tuple(p[:n]) in out:
            fail("%s: duplicated row %s" % (name, " ".join(p[:n])))
        out[tuple(p[:n])] = p[n:]
    return out


def csv_bytes(columns, records):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    w.writerows(records)
    return buf.getvalue().encode("utf-8")


def carbon_content():
    with open(EPM / "epm" / "resources" / "pFuelCarbonContent.csv", newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if r]
    return {r[0].strip(): float(r[1]) for r in rows[1:]}


def paid_content(fuel, carbon):
    """Tonnes of CO2 paid per MMBtu of a fuel of the three EU zones, as in edit 2 of build_robg_v6.py."""
    if fuel == robg.CCS_FUEL:
        return carbon["Gas"] * robg.CCS_EMITTED
    if fuel in robg.NO_CARBON:
        return 0.0
    if fuel not in carbon:
        fail("no carbon content for fuel %s" % fuel)
    return carbon[fuel]


def carbon_paths(carbon):
    """Years, the CO2 path of v5 and the central path of v6 in USD per tonne, and the lines of the priced rows.

    The v5 path is read back from the gas row of the two v5 files (price with CO2, commodity alone), as edit 2
    does. The central path is read back from the CO2 line of the same row in the file edit 2 writes."""
    _, _, header, sum_rows = text_table(SUPPLY / FUEL_V5_SUM)
    _, _, header_c, com_rows = text_table(SUPPLY / FUEL_V5_COM)
    _, _, header_l, line_rows = text_table(SUPPLY / FUEL_LINES)
    head = header.split(",")
    if header != header_c or head[:2] != ["country", "fuel"]:
        fail("%s and %s: unexpected or different headers" % (FUEL_V5_SUM, FUEL_V5_COM))
    if header_l.split(",") != head[:2] + ["line"] + head[2:]:
        fail("%s: the years are not those of %s" % (FUEL_LINES, FUEL_V5_SUM))
    years = [int(c) for c in head[2:]]
    v5_sum, v5_com = keyed(FUEL_V5_SUM, header, sum_rows, 2), keyed(FUEL_V5_COM, header, com_rows, 2)
    lines = keyed(FUEL_LINES, header_l, line_rows, 3)
    ref = (robg.FUEL_ZONES[0], "Gas")
    low = {y: (float(s) - float(c)) / carbon["Gas"] for y, s, c in zip(years, v5_sum[ref], v5_com[ref])}
    central = {y: float(v) / carbon["Gas"] for y, v in zip(years, lines[ref + ("co2",)])}
    for y in years:
        if y <= robg.BLEND_FROM and abs(low[y] - central[y]) > PATH_TOL:
            fail("the two CO2 paths differ in %d, a year the central keeps from v5" % y)
    return years, low, central, lines


# ── Armenia ─────────────────────────────────────────────────────────────────
def armenia_gas_contract(records):
    """The central fuel file with the Armenian gas row of v5. Returns the bytes."""
    bom, eol, header, rows = text_table(SUPPLY / FUEL_V6)
    _, _, header5, rows5 = text_table(SUPPLY / robg.AM_FUEL_V5)
    if header != header5:
        fail("%s and %s: different headers" % (FUEL_V6, robg.AM_FUEL_V5))
    key = (robg.AM_ZONE, robg.AM_GAS)
    v5 = keyed(robg.AM_FUEL_V5, header5, rows5, 2)
    if key not in v5:
        fail("%s: no row %s %s" % ((robg.AM_FUEL_V5,) + key))
    years = header.split(",")[2:]
    out, hits = [], 0
    for line in rows:
        p = line.split(",")
        if (p[0], p[1]) != key:
            out.append(line)
            continue
        hits += 1
        gaps = [float(a) - float(b) for a, b in zip(p[2:], v5[key])]
        if min(gaps) <= 0 or max(gaps) - min(gaps) > MARGIN_TOL:
            fail("%s: the %s %s row is not the v5 row plus one margin" % ((FUEL_V6,) + key))
        for y, a, b in zip(years, p[2:], v5[key]):
            records.append(("ArmeniaGasContract", AM_GAS_OUT, key[0], key[1], y, a, b, BASIS["ArmeniaGasContract"]))
        out.append(",".join(list(key) + v5[key]))
    if hits != 1:
        fail("%s: %d rows %s %s, expected one" % ((FUEL_V6, hits) + key))
    return text_payload(bom, eol, header, out)


def armenia_demand_low(records):
    """The central demand file with the two Armenian rows of v5. Returns the bytes and the number of cells moved."""
    bom, eol, header, rows = text_table(LOAD / robg.AM_DEMAND_V6)
    _, _, header5, rows5 = text_table(LOAD / robg.AM_DEMAND_V5)
    if header != header5:
        fail("%s and %s: different headers" % (robg.AM_DEMAND_V6, robg.AM_DEMAND_V5))
    v5 = keyed(robg.AM_DEMAND_V5, header5, rows5, 2)
    years = header.split(",")[2:]
    out, hits, moved = [], [], 0
    for line in rows:
        p = line.split(",")
        if p[0] != robg.AM_ZONE:
            out.append(line)
            continue
        key = (p[0], p[1])
        if key not in v5:
            fail("%s: no row %s %s" % ((robg.AM_DEMAND_V5,) + key))
        hits.append(p[1])
        for y, a, b in zip(years, p[2:], v5[key]):
            if a != b:
                moved += 1
                records.append(("ArmeniaDemandLow", AM_DEMAND_OUT, key[0], key[1], y, a, b, BASIS["ArmeniaDemandLow"]))
        out.append(",".join(list(key) + v5[key]))
    if sorted(hits) != ["Energy", "Peak"] or moved == 0:
        fail("%s: the rows of %s are not one Peak and one Energy row that differ from v5" % (
            robg.AM_DEMAND_V6, robg.AM_ZONE))
    return text_payload(bom, eol, header, out), moved


# ── GEC hubs ────────────────────────────────────────────────────────────────
def caucasus_hubs(records):
    """The fleet of the five year grid with the hub candidates replaced by committed steps. Returns bytes and counts."""
    source = robg.BG_FLEET_V6 % robg.VARIANT_GRID
    bom, eol, header, rows = text_table(SUPPLY / source)
    _, _, header_s, rows_s = text_table(SUPPLY / HUB_STEPS)
    if header != header_s:
        fail("%s and %s: different headers" % (source, HUB_STEPS))
    cols = header.split(",")
    g, z, tech, f, status, styr, retr, cap, lim = (cols.index(c) for c in (
        "g", "z", "tech", "f", "Status", "StYr", "RetrYr", "Capacity", "BuildLimitperYear"))
    rest = lim + 1                    # the cells after BuildLimitperYear come from the candidate
    out_name = robg.VARIANT_FLEET % (robg.VARIANT_GRID, HUBS_TAG)
    grid_years = [int(v) for v in text_table(DATA / robg.GRID_YEARS[robg.VARIANT_GRID])[3]]
    parts = [r.split(",") for r in rows]
    names = {p[g] for p in parts}
    cands = {p[g]: p for p in parts if p[z] in HUB_ZONES}
    if not cands or any(p[status] != "3" for p in cands.values()):
        fail("%s: the units of %s are not all candidates" % (source, " and ".join(HUB_ZONES)))
    steps = {}
    for p in (r.split(",") for r in rows_s):
        if p[z] not in HUB_ZONES or p[status] != "2":
            continue
        base, _, year = p[g].rpartition("_")
        if base not in cands or year != p[styr] or p[g] in names:
            fail("%s: %s is not a step of a hub candidate of %s" % (HUB_STEPS, p[g], source))
        c = cands[base]
        if (p[z], p[tech], p[f], p[retr]) != (c[z], c[tech], c[f], c[retr]):
            fail("%s: %s does not sit on the zone, technology, fuel and last year of %s" % (HUB_STEPS, p[g], base))
        if int(p[styr]) < int(c[styr]) or int(p[styr]) not in grid_years or int(p[styr]) <= grid_years[0]:
            fail("%s: %s starts in %s, before its candidate or off the years of %s" % (
                HUB_STEPS, p[g], p[styr], robg.GRID_YEARS[robg.VARIANT_GRID]))
        if any(v not in ("", w) for v, w in zip(p[rest:], c[rest:])):
            fail("%s: %s holds a cost or technical cell its candidate does not" % (HUB_STEPS, p[g]))
        steps.setdefault(base, []).append(p[:rest] + c[rest:])
    out, total = [], 0.0
    for line, p in zip(rows, parts):
        if p[g] not in cands:
            out.append(line)
            continue
        mine = steps.get(p[g], [])
        built = sum(float(s[cap]) for s in mine)
        if not mine or abs(built - float(p[cap])) > STEP_TOL:
            fail("%s: the steps of %s add up to %s MW, the candidate holds %s" % (HUB_STEPS, p[g], num(built, 3), p[cap]))
        total += built
        records.append(("CaucasusREHubs", out_name, p[z], p[g], "row", "candidate of %s MW from %s" % (p[cap], p[styr]),
                        "removed", BASIS["CaucasusREHubs"]))
        for s in mine:
            out.append(",".join(s))
            records.append(("CaucasusREHubs", out_name, s[z], s[g], "row", "n/a",
                            "committed, %s MW in %s" % (s[cap], s[styr]), BASIS["CaucasusREHubs"]))
    return text_payload(bom, eol, header, out), out_name, len(cands), sum(len(v) for v in steps.values()), total


# ── EU carbon price ─────────────────────────────────────────────────────────
def eu_carbon_low_fuel(years, low, lines, carbon, records):
    """The central fuel file with the CO2 line of the three EU zones on the v5 path. Returns bytes and counts."""
    bom, eol, header, rows = text_table(SUPPLY / FUEL_V6)
    if [int(c) for c in header.split(",")[2:]] != years:
        fail("%s: the years are not those of %s" % (FUEL_V6, FUEL_V5_SUM))
    out, moved, kept = [], 0, 0
    for line in rows:
        p = line.split(",")
        key = (p[0], p[1])
        if p[0] not in EU_ZONES:
            out.append(line)
            continue
        if key + ("commodity",) not in lines or key + ("co2",) not in lines:
            fail("%s: no lines for %s %s" % ((FUEL_LINES,) + key))
        k = paid_content(p[1], carbon)
        if k == 0:
            out.append(line)
            kept += 1
            continue
        com, co2 = lines[key + ("commodity",)], lines[key + ("co2",)]
        new = list(p)
        for i, y in enumerate(years):
            if y <= robg.BLEND_FROM:
                continue
            if num(float(com[i]) + float(co2[i]), robg.FUEL_DECIMALS) != p[2 + i]:
                fail("%s: %s %s %d is not the sum of its two lines" % (FUEL_V6, p[0], p[1], y))
            cell = num(float(com[i]) + round(low[y] * k, robg.LINE_DECIMALS), robg.FUEL_DECIMALS)
            if cell != p[2 + i]:
                records.append(("EUCarbonLow", EU_LOW_FUEL, p[0], p[1], str(y), p[2 + i], cell, BASIS["EUCarbonLow"]))
            new[2 + i] = cell
        out.append(",".join(new))
        moved += 1
    return text_payload(bom, eol, header, out), moved, kept


def border_factors(years, low, central, lines, carbon):
    """Year factors of the border prices: cost of gas with CO2 on the low path over the same cost on the central."""
    com = dict(zip(years, (float(v) for v in lines[(robg.FUEL_ZONES[0], "Gas", "commodity")])))
    k = carbon["Gas"]
    return {y: 1.0 if y <= robg.BLEND_FROM else (com[y] + low[y] * k) / (com[y] + central[y] * k) for y in years}


def scaled_prices(name, out_name, factors, records):
    """A border price table of the Int scope with the rows of the TYNDP priced neighbours scaled year by year."""
    table = robg.Lines(TRADE / name)
    out, scaled, seen, top = [], 0, set(), (0.0, 0.0)
    for line in table.rows:
        zone, q, d, y = table.key(line)
        if y not in factors:
            fail("%s: year %d is outside the fuel price years" % (name, y))
        if zone not in robg.INT_NODES or factors[y] == 1.0:
            out.append(line)
            continue
        before = table.values(line)
        values = np.maximum(before * factors[y], robg.FLOOR_USD)
        out.append(robg.fmt(zone, q, d, y, values))
        scaled += 1
        if (zone, y) not in seen:
            seen.add((zone, y))
            records.append(("EUCarbonLow", out_name, zone, "all quarters and days", str(y), "1",
                            "%.6f" % factors[y], BASIS["EUCarbonLow prices"]))
        if values.max() > top[0]:
            top = (float(values.max()), float(before.max()))
    if {zone for zone, _ in seen} != set(robg.INT_NODES):
        fail("%s: the scaled rows do not cover the %d neighbours priced on TYNDP 2026" % (name, len(robg.INT_NODES)))
    return table.encode(out), scaled, top


def loss_gap(import_bytes, export_bytes):
    """Largest gap between an export cell and its import cell less the border loss, on the TYNDP priced rows."""
    def cells(payload):
        text = payload.decode("utf-8-sig")
        rows = [r.split(",") for r in text.splitlines()[1:]]
        return {tuple(r[:4]): np.array(r[4:], float) for r in rows if r[0] in robg.INT_NODES}
    imp, exp = cells(import_bytes), cells(export_bytes)
    if list(imp) != list(exp):
        fail("the import and export price tables do not hold the same rows")
    worst = 0.0
    for key, v in imp.items():
        free = (v > robg.FLOOR_USD) & (exp[key] > robg.FLOOR_USD)      # the floor breaks the proportion
        if free.any():
            worst = max(worst, float(np.abs(exp[key][free] - v[free] * (1 - robg.INT_LOSS)).max()))
    return worst


# ── Turkish carbon price ────────────────────────────────────────────────────
def turkiye_carbon_fuel(years, central, lines, carbon, records):
    """The central fuel file with the CO2 price of the central on the Turkish fuels from TR_FROM."""
    bom, eol, header, rows = text_table(SUPPLY / FUEL_V6)
    out, adders = [], {}
    for line in rows:
        p = line.split(",")
        if p[0] != TR_COUNTRY:
            out.append(line)
            continue
        if p[1] not in carbon:
            fail("no carbon content for fuel %s" % p[1])
        if carbon[p[1]] == 0:
            out.append(line)
            continue
        new, adder = list(p), {}
        for i, y in enumerate(years):
            if y < TR_FROM:
                continue
            adder[y] = round(central[y] * carbon[p[1]], robg.LINE_DECIMALS)
            cell = num(float(p[2 + i]) + adder[y], robg.FUEL_DECIMALS)
            records.append(("TurkiyeCarbonPrice", TR_FUEL, p[0], p[1], str(y), p[2 + i], cell, BASIS["TurkiyeCarbonPrice"]))
            new[2 + i] = cell
        if p[1] in TR_CROSS:          # same carbon content as a row of edit 2: the adder is its CO2 line
            ref = dict(zip(years, (float(v) for v in lines[TR_CROSS[p[1]] + ("co2",)])))
            if max(abs(adder[y] - ref[y]) for y in adder) > LINE_TOL:
                fail("%s %s: the adder is not the CO2 line of %s %s" % ((p[0], p[1]) + TR_CROSS[p[1]]))
        adders[p[1]] = adder
        out.append(",".join(new))
    if set(TR_FUELS) - set(adders):
        fail("%s: no %s row for %s" % (FUEL_V6, TR_COUNTRY, ", ".join(sorted(set(TR_FUELS) - set(adders)))))
    return text_payload(bom, eol, header, out), adders


def build(check=False):
    log = ["Sensitivities of v6, first lot (23 November 2026)"]
    ok, records = True, []
    carbon = carbon_content()
    years, low, central, lines = carbon_paths(carbon)
    show = [y for y in SHOW if y in years]

    log.append("ArmeniaGasContract")
    ok &= emit(SUPPLY / AM_GAS_OUT, armenia_gas_contract(records), check, log)
    log.append("  %s %s: the row of %s; the central row is that row plus one margin, the same in every year" % (
        robg.AM_ZONE, robg.AM_GAS, robg.AM_FUEL_V5))

    log.append("ArmeniaDemandLow")
    payload, moved = armenia_demand_low(records)
    ok &= emit(LOAD / AM_DEMAND_OUT, payload, check, log)
    log.append("  %s Peak and Energy: the rows of %s, %d cells differ from the central" % (
        robg.AM_ZONE, robg.AM_DEMAND_V5, moved))

    log.append("CaucasusREHubs")
    payload, out_name, n_cand, n_steps, total = caucasus_hubs(records)
    ok &= emit(SUPPLY / out_name, payload, check, log)
    log.append("  %d candidates of %s replaced by %d committed steps of %s, %s MW in all, each unit in full" % (
        n_cand, " and ".join(HUB_ZONES), n_steps, HUB_STEPS, num(total, 1)))

    log.append("EUCarbonLow")
    payload, moved, kept = eu_carbon_low_fuel(years, low, lines, carbon, records)
    ok &= emit(SUPPLY / EU_LOW_FUEL, payload, check, log)
    log.append("  %s: %d rows repriced, %d rows without carbon kept as read" % (", ".join(EU_ZONES), moved, kept))
    factors = border_factors(years, low, central, lines, carbon)
    prices = []
    for name, out_name in zip(PRICE_V6, EU_LOW_PRICE):
        payload, scaled, top = scaled_prices(name, out_name, factors, records)
        prices.append(payload)
        ok &= emit(TRADE / out_name, payload, check, log)
        log.append("  %s: %d rows of %d neighbours scaled; largest cell %.3f USD/MWh (%.3f in the central row)" % (
            out_name, scaled, len(robg.INT_NODES), top[0], top[1]))
    gap0 = loss_gap(*((TRADE / n).read_bytes() for n in PRICE_V6))
    gap1 = loss_gap(*prices)
    if gap1 > gap0 + 1.1 * 10 ** -robg.DECIMALS:
        fail("the scaled export prices are not the import prices less the loss: gap %.4f USD/MWh" % gap1)
    log.append("  export against import less %.1f percent: largest gap %.4f USD/MWh (central %.4f)" % (
        100 * robg.INT_LOSS, gap1, gap0))
    gas = (robg.FUEL_ZONES[0], "Gas")
    com = dict(zip(years, (float(v) for v in lines[gas + ("commodity",)])))
    log.append("  USD per tonne, USD/MMBtu      " + "".join("%9d" % y for y in show))
    for label, series in (("CO2 central (TYNDP 2026)", central), ("CO2 low (TYNDP 2024)", low),
                          ("gas with CO2, central", {y: com[y] + central[y] * carbon["Gas"] for y in years}),
                          ("gas with CO2, low", {y: com[y] + low[y] * carbon["Gas"] for y in years})):
        log.append("    %-28s" % label + "".join("%9.2f" % series[y] for y in show))
    log.append("    %-28s" % "border price factor" + "".join("%9.3f" % factors[y] for y in show))
    above = [y for y in years if low[y] > central[y] + PATH_TOL]
    log.append("  the low path is above the central in %s" % (
        "%d to %d" % (above[0], above[-1]) if above else "no year"))

    log.append("TurkiyeCarbonPrice")
    payload, adders = turkiye_carbon_fuel(years, central, lines, carbon, records)
    ok &= emit(SUPPLY / TR_FUEL, payload, check, log)
    log.append("  %s from %d: %s pay the CO2 price of the central on their carbon content" % (
        TR_COUNTRY, TR_FROM, ", ".join(adders)))
    for fuel, adder in adders.items():
        log.append("    %-28s" % ("adder %s, USD/MMBtu" % fuel) + "".join(
            "%9.2f" % adder.get(y, 0.0) for y in show))

    # ── Traces ─────────────────────────────────────────────────────────────
    ok &= emit(TRACE_FULL, csv_bytes(TRACE_COLS, records), check, log)
    grouped = []
    for rec in records:               # one line per rewritten row: the cells are counted, no value is kept
        head = (rec[0], rec[1], rec[2], rec[3], rec[7])
        if grouped and grouped[-1][0] == head:
            grouped[-1][1].append(rec[4])
        else:
            grouped.append((head, [rec[4]]))
    ok &= emit(TRACE_ROWS, csv_bytes(
        ["sensitivity", "file", "zone", "row", "cells", "first", "last", "basis"],
        [h[:4] + (len(f), f[0], f[-1], h[4]) for h, f in grouped]), check, log)
    fuels = list(adders)
    ok &= emit(TRACE_PATHS, csv_bytes(
        ["year", "co2_central_usd_t", "co2_low_usd_t", "border_price_factor"] + [
            "turkiye_%s_adder_usd_mmbtu" % f.lower() for f in fuels],
        [[y, "%.4f" % central[y], "%.4f" % low[y], "%.6f" % factors[y]] + [
            num(adders[f].get(y, 0.0), robg.LINE_DECIMALS) for f in fuels] for y in years]), check, log)
    counts = []
    for rec in records:
        if rec[0] not in counts:
            counts.append(rec[0])
    log.append("  trace records per sensitivity: %s" % ", ".join(
        "%s %d" % (c, sum(1 for r in records if r[0] == c)) for c in counts))
    return ok, log


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sensitivity files of the v6 inputs of the Black Sea model, first lot")
    ap.add_argument("--check", action="store_true", help="compare the files with a fresh build, write nothing")
    args = ap.parse_args(argv)
    ok, log = build(check=args.check)
    print("\n".join(log))
    if args.check:
        print("check: %s" % ("files match a fresh build" if ok else "files differ from a fresh build"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
