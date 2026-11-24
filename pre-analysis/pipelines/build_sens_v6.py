"""
Sensitivity files of the v6 input family: first lot (23 November 2026) and second lot, parts 1 and 2 (24 November 2026).

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

Second lot, part 1: gas prices and demand. A gas price case starts in 2030 and a demand case grows from 2025; the
years before stay on the central.
  WorldGasHigh         pFuelPrice: the commodity line of the gas fuels of Romania, Bulgaria and Hungary goes to a
                       TTF of 23.2 USD/MMBtu, the mean of 2021 to 2023 in the World Bank Pink Sheet; their CO2 line
                       stays. The Turkish gas row takes 40 percent of the move of the TTF on top of its central
                       level, the hub indexed share of the Turkish import contracts (interim assumption). The
                       Caucasus stays on contract. pTradePrice and pTradePriceExport: the rows of the nine
                       neighbours priced on TYNDP 2026 are scaled by the cost of gas with CO2 in the case over the
                       same cost in the central, the rule of EUCarbonLow.
  WorldGasLow          The same with a TTF of 5.2 USD/MMBtu, the mean of 2016 to 2020.
  TurkiyeGasHigh       pFuelPrice: the Turkish gas row of WorldGasHigh alone.
  CaucasusGasNetback   pFuelPrice: Azerbaijani gas at 6.5 USD/MMBtu, its export netback through TANAP; Georgian gas
                       at 7.0, the same netback plus about 0.5 of transport (derived, not published).
  EUDataCentres        pDemandForecast: data centre load on top of the central in Romania and Bulgaria, 4.3, 9.5 and
                       14.6 TWh in 2030, 2035 and 2040 for the two, linear between these years. A flat load: the
                       energy row takes the energy and the peak row takes the energy over 8,760 hours.
  TurkiyeDemandLow     pDemandForecast: the Turkish total grows 2.5 percent a year from its 2025 level, the high
                       case of the TEIAS ten year demand report. One ratio per year on the peak and energy rows of
                       every Turkish zone, so the split between zones and the load factors stay those of the central.
  GeorgiaDemandHigh    pDemandForecast: Georgian demand grows 5 percent a year from its 2025 level, case L3 of the
                       GSE ten year plan 2024 to 2034. One ratio per year on the peak and energy rows.

Second lot, part 2: water years and solar and wind build cases.
  TurkiyeDry2021       pAvailability: the hydro output of 2021 in Türkiye and Georgia. One ratio per country on the
                       four quarters of every run of river and reservoir unit (existing, committed, candidate): the
                       capacity factor of the real year over that of the existing fleet in the central. The file has
                       no year, so the case holds in every model year. Pumped storage is left as it is.
  GeorgiaDry2020       pAvailability: the same with the hydro output of 2020, dry in Georgia and wet in Türkiye.
  CaucasusRESigned     pGenDataInput: the generic solar and wind candidates of Azerbaijan and the candidates of the
                       two GEC hub zones are closed (status 0), so Azerbaijan keeps its signed projects, 1.68 GW. The
                       potential of the Georgian solar and wind candidates is held to the pipeline of the GSE ten
                       year plan, the fleet of 2035; the plan stops there, the level is kept after.
  EUREPlan             pGenDataInput and pAvailability: the national plans of Romania, Bulgaria (energy and climate
                       plans) and Hungary (solar, 12 GW in 2030, kept after) as a floor, through committed steps in
                       the model years, linear between the plan years. Solar: a step is the plan less the fleet.
                       Onshore wind: a step is the part of the plan that the fleet and the candidates at their
                       yearly limit do not reach, as the central builds them at that limit. A step takes the cells
                       of its generic candidate, whose potential goes down by as much.
The Turkish plan as a floor is not built: the central builds more solar and wind than the plan in every model year.

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
# ── Second lot, part 1: gas prices and demand ───────────────────────────────
GAS_FROM = 2030                                     # first year of a gas price case
TTF_CASES = {"WorldGasHigh": 23.2, "WorldGasLow": 5.2}     # USD/MMBtu, Pink Sheet means of 2021 to 2023, 2016 to 2020
TR_PASS = 0.40                                      # share of a TTF move in the Turkish gas price (interim)
NETBACK = {"Azerbaijan": 6.5, "Georgia": 7.0}       # USD/MMBtu from GAS_FROM
ZCMAP = "zcmap_robg_v6.csv"
DEMAND_V6 = robg.AM_DEMAND_V6                       # central demand forecast, every zone
DEMAND_DECIMALS = 4
HOURS_YEAR = 8760.0
DC_GWH = {"Romania": {2025: 0.0, 2030: 2900.0, 2035: 5600.0, 2040: 8300.0},        # on top of the central
          "Bulgaria": {2025: 0.0, 2030: 1400.0, 2035: 3900.0, 2040: 6300.0}}       # held after the last year
GROWTH_FROM = 2025
GROWTH = {"TurkiyeDemandLow": ("Turkiye", 0.025, -1), "GeorgiaDemandHigh": ("Georgia", 0.05, 1)}   # rate, direction
LOTB_FUEL = {"WorldGasHigh": "pFuelPrice_robg_worldgashigh_v6.csv", "WorldGasLow": "pFuelPrice_robg_worldgaslow_v6.csv",
             "TurkiyeGasHigh": "pFuelPrice_robg_trgashigh_v6.csv",
             "CaucasusGasNetback": "pFuelPrice_robg_caucasusnetback_v6.csv"}
LOTB_PRICE = {"WorldGasHigh": ("pTradePrice_robg_worldgashigh_v6.csv", "pTradePriceExport_robg_worldgashigh_v6.csv"),
              "WorldGasLow": ("pTradePrice_robg_worldgaslow_v6.csv", "pTradePriceExport_robg_worldgaslow_v6.csv")}
LOTB_DEMAND = {"EUDataCentres": "pDemandForecast_v6_eudatacentres.csv",
               "TurkiyeDemandLow": "pDemandForecast_v6_trdemlow.csv",
               "GeorgiaDemandHigh": "pDemandForecast_v6_gedemhigh.csv"}
LOTB_TRACE_FULL = DATA / "v6_sensitivities_20261124_trace.csv"    # every cell with its value before and after
LOTB_TRACE_ROWS = TRACE / "sens_v6_lotb_rows.csv"                 # rows and basis, no value
LOTB_TRACE_PATHS = TRACE / "sens_v6_lotb_paths.csv"               # gas paths, border factors, demand totals
BASIS.update({
    "WorldGasHigh": "TTF at 23.2 USD/MMBtu from 2030, mean of 2021 to 2023 (World Bank Pink Sheet); CO2 line kept",
    "WorldGasLow": "TTF at 5.2 USD/MMBtu from 2030, mean of 2016 to 2020 (World Bank Pink Sheet); CO2 line kept",
    "WorldGas Turkiye": "Central Turkish gas plus 40 percent of the move of the TTF (interim assumption)",
    "WorldGas prices": "Row scaled by the cost of gas with CO2 in the case over the central (rule of the study)",
    "TurkiyeGasHigh": "Turkish gas row of the world gas high case alone (interim assumption)",
    "CaucasusGasNetback": "Export netback through TANAP; Georgia at the netback plus about 0.5 of transport (derived)",
    "EUDataCentres": "Data centre load on top of the central, flat: energy, and energy over 8,760 hours on the peak",
    "TurkiyeDemandLow": "2.5 percent a year from the 2025 total of the central (TEIAS high case), one ratio per year",
    "GeorgiaDemandHigh": "5 percent a year from 2025 (case L3 of the GSE ten year plan), one ratio per year",
})
# ── Second lot, part 2: hydro years ─────────────────────────────────────────
PLANT_V6 = robg.BG_FLEET_V6 % robg.VARIANT_GRID     # fleet of the five year grid
AVAIL_V6 = "pAvailabilityCustom_v6.csv"             # central: one row per unit that has its own availability
AVAIL_DEFAULT_V6 = "pAvailabilityDefault_v6.csv"    # central: availability of a unit without a row of its own
HOURS = "pHours.csv"                                # hours of each quarter
HYDRO_TECHS, HYDRO_FUEL = ("ROR", "ReservoirHydro"), "Water"     # pumped storage does not follow the water year
HYDRO_COUNTRIES = ("Turkiye", "Georgia")
HYDRO_YEAR = {"TurkiyeDry2021": 2021, "GeorgiaDry2020": 2020}
HYDRO_OBS = {                 # observed hydro output in GWh and capacity in MW; None: the existing fleet of the file
    "Turkiye": {2020: (78094.4, 30983.9), 2021: (55926.8, 31492.6)},        # TEIAS statistics
    "Georgia": {2020: (8248.2, None), 2021: (10182.1, None)},               # ESCO electricity balances
}
HYDRO_DECIMALS = 6
LOTB2_AVAIL = {"TurkiyeDry2021": "pAvailabilityCustom_v6_hydro2021.csv",
               "GeorgiaDry2020": "pAvailabilityCustom_v6_hydro2020.csv"}
LOTB2_TRACE_FULL = DATA / "v6_sensitivities_20261124_part2_trace.csv"     # every cell with its value before and after
LOTB2_TRACE_ROWS = TRACE / "sens_v6_lotb2_rows.csv"                       # rows and basis, no value
LOTB2_TRACE_HYDRO = TRACE / "sens_v6_lotb2_hydro.csv"                     # observed year, capacity factors, ratio
BASIS.update({
    "TurkiyeDry2021": "Hydro output of 2021: availability times the capacity factor of 2021 over that of the central",
    "GeorgiaDry2020": "Hydro output of 2020: availability times the capacity factor of 2020 over that of the central",
    "hydro added row": "Unit on the default availability of its zone: row added, default times the same ratio",
})

# ── Second lot, part 2: solar and wind build cases ──────────────────────────
RE_TECHS = ("PV", "OnshoreWind", "OffshoreWind")
CLOSED = "0"                                        # status of a unit that is out of the model
SIGNED_TAG = "cauresigned"
SIGNED_COUNTRY = "Azerbaijan"
SIGNED_OFF = ("AzerbaijanMain_PV_Generic", "AzerbaijanMain_Wind_Generic", "AzerbaijanMain_Offshore_Generic",
              "Nakhchivan_PV_Generic", "Nakhchivan_Wind_Generic")       # generic candidates of Azerbaijan
SIGNED_AFTER = 2025                                 # a signed project comes after this year
SIGNED_MW = 1680.5                                  # signed solar and wind of Azerbaijan, the 1.68 GW of the matrix
PIPELINE_ZONE = "Georgia"
PIPELINE_YEAR = 2035                                # last year the GSE ten year plan 2024 to 2034 covers
PIPELINE_MW = {"Georgia_Wind_Cand": 1316.0, "Georgia_PV_Cand": 237.0}   # fleet of the technology in that year
PLAN_TAG = "eureplan"
PLAN_AVAIL = "pAvailabilityCustom_v6_eureplan.csv"
PLAN = {            # (zone, technology): rule, generic candidate that lends its cells, plan in GW by year
    ("Romania", "PV"): ("fleet", "Generic Solar Fixed", {2030: 8.2, 2035: 14.8, 2040: 21.4}),
    ("Romania", "OnshoreWind"): ("reach", "Generic Onshore WindHigh2_2027", {2030: 7.3, 2035: 9.1, 2040: 12.8}),
    ("Bulgaria", "PV"): ("fleet", "Generic_Solar_Bulgaria", {2030: 7.0, 2035: 10.1, 2040: 12.6}),
    ("Bulgaria", "OnshoreWind"): ("reach", "Generic_Wind_Bulgaria", {2030: 2.4, 2035: 4.4, 2040: 5.9}),
    ("Hungary", "PV"): ("fleet", "Generic_Solar_Hungary", {2030: 12.0}),
}
# Rule "fleet": the central builds no solar candidate to speak of, so a step is the plan less the fleet. Rule "reach":
# the central builds the onshore wind candidates at their yearly limit, so a step is only the part of the plan that the
# fleet and the candidates at their limit do not reach; a full step would lift that limit and add wind beyond the plan.
PLAN_UNBUILT = ("Blackstone 2",)                    # candidate the central leaves unbuilt (capex 3.0 against 1.3)
PLAN_DECIMALS = 1
PLAN_MIN_MW = 1.0                                   # no step below this size
LOTB2_TRACE_PLANS = TRACE / "sens_v6_lotb2_plans.csv"                     # plan, fleet, reach and step by year
BASIS.update({
    "CaucasusRESigned": "Generic or hub candidate closed: Azerbaijan keeps its signed solar and wind projects",
    "CaucasusRESigned pipeline": "Potential of the Georgian candidate held to the GSE pipeline of 2035, less the fleet",
    "EUREPlan": "Committed step of the national plan: the plan of the year less what is there without it",
    "EUREPlan candidate": "Potential of the generic candidate less its committed steps: the potential is held once",
    "EUREPlan availability": "Availability row of a committed step: the cells of its generic candidate",
})
PATH_TOL = 1e-3        # USD per tonne: the two CO2 paths in the years the central keeps from v5
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


def scaled_prices(name, out_name, factors, records, sens="EUCarbonLow", basis=None):
    """A border price table of the Int scope with the rows of the TYNDP priced neighbours scaled year by year."""
    basis = basis or BASIS["EUCarbonLow prices"]
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
            records.append((sens, out_name, zone, "all quarters and days", str(y), "1", "%.6f" % factors[y], basis))
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


# ── Second lot, part 1: gas prices ──────────────────────────────────────────
def gas_reference(years, lines):
    """Commodity and CO2 line of the gas row of the three EU zones by year, USD/MMBtu."""
    ref = (robg.FUEL_ZONES[0], "Gas")
    return (dict(zip(years, (float(v) for v in lines[ref + ("commodity",)]))),
            dict(zip(years, (float(v) for v in lines[ref + ("co2",)]))))


def gas_case_fuel(sens, years, lines, records, ttf=None, eu=False, turkiye=False, flat=None):
    """The central fuel file with the gas rows of one gas price case from GAS_FROM. Returns bytes and the new paths.

    eu       the commodity line of the gas fuels of the three EU zones goes to `ttf`, their CO2 line stays
    turkiye  the Turkish gas row takes TR_PASS of the move of the EU commodity line to `ttf`
    flat     gas rows set to one level, by country
    """
    out_name = LOTB_FUEL[sens]
    bom, eol, header, rows = text_table(SUPPLY / FUEL_V6)
    if [int(c) for c in header.split(",")[2:]] != years:
        fail("%s: the years are not those of %s" % (FUEL_V6, FUEL_V5_SUM))
    com_ref, _ = gas_reference(years, lines)
    live = [(i, y) for i, y in enumerate(years) if y >= GAS_FROM]
    out, paths = [], {}
    for line in rows:
        p = line.split(",")
        key = (p[0], p[1])
        if eu and p[0] in EU_ZONES and p[1] in robg.GAS_ON_TYNDP:
            com, co2 = lines[key + ("commodity",)], lines[key + ("co2",)]
            for i, y in live:
                if abs(float(com[i]) - com_ref[y]) > LINE_TOL:
                    fail("%s: the commodity line of %s %s is not the gas line in %d" % (FUEL_LINES, p[0], p[1], y))
                if num(float(com[i]) + float(co2[i]), robg.FUEL_DECIMALS) != p[2 + i]:
                    fail("%s: %s %s %d is not the sum of its two lines" % (FUEL_V6, p[0], p[1], y))
            values, basis = {i: ttf + float(co2[i]) for i, _ in live}, BASIS[sens]
        elif turkiye and key == (TR_COUNTRY, "Gas"):
            values = {i: float(p[2 + i]) + TR_PASS * (ttf - com_ref[y]) for i, y in live}
            basis = BASIS["WorldGas Turkiye" if eu else sens]
        elif flat and p[1] == "Gas" and p[0] in flat:
            values, basis = {i: flat[p[0]] for i, _ in live}, BASIS[sens]
        else:
            out.append(line)
            continue
        if key in paths:
            fail("%s: duplicated row %s %s" % ((FUEL_V6,) + key))
        new, path = list(p), {}
        for i, y in live:
            cell = num(values[i], robg.FUEL_DECIMALS)
            records.append((sens, out_name, p[0], p[1], str(y), p[2 + i], cell, basis))
            new[2 + i], path[y] = cell, (float(p[2 + i]), float(cell))
        signs = {np.sign(after - before) for before, after in path.values()}
        if len(signs) != 1 or 0.0 in signs:
            fail("%s: the %s %s row does not move one way in every year from %d" % ((sens,) + key + (GAS_FROM,)))
        paths[key] = path
        out.append(",".join(new))
    want = set()
    if eu:
        want |= {(z, "Gas") for z in EU_ZONES}
    if turkiye:
        want.add((TR_COUNTRY, "Gas"))
    if flat:
        want |= {(c, "Gas") for c in flat}
    if want - set(paths) or any(k[1] not in robg.GAS_ON_TYNDP for k in paths):
        fail("%s: rows missing or unexpected: %s" % (sens, sorted(want ^ set(paths))))
    return text_payload(bom, eol, header, out), paths


def gas_border_factors(years, lines, ttf):
    """Year factors of the border prices: cost of gas with CO2 at `ttf` over the same cost in the central."""
    com, co2 = gas_reference(years, lines)
    return {y: 1.0 if y < GAS_FROM else (ttf + co2[y]) / (com[y] + co2[y]) for y in years}


# ── Second lot, part 1: demand ──────────────────────────────────────────────
def demand_table():
    bom, eol, header, rows = text_table(LOAD / DEMAND_V6)
    head = header.split(",")
    if head[:2] != ["z", "type"]:
        fail("%s: unexpected header" % DEMAND_V6)
    cells = keyed(DEMAND_V6, header, rows, 2)
    if any(k[1] not in ("Peak", "Energy") for k in cells):
        fail("%s: a row is neither Peak nor Energy" % DEMAND_V6)
    return bom, eol, header, rows, [int(c) for c in head[2:]], cells


def country_zones(country):
    _, _, header, rows = text_table(DATA / ZCMAP)
    if header.split(",") != ["z", "c"]:
        fail("%s: unexpected header" % ZCMAP)
    zones = [r.split(",")[0] for r in rows if r.split(",")[1] == country]
    if not zones:
        fail("%s: no zone for %s" % (ZCMAP, country))
    return zones


def between(anchors, year):
    """Linear between the anchor years, held before the first and after the last."""
    ks = sorted(anchors)
    if year <= ks[0]:
        return anchors[ks[0]]
    if year >= ks[-1]:
        return anchors[ks[-1]]
    a = max(k for k in ks if k <= year)
    b = min(k for k in ks if k > year)
    return anchors[a] + (anchors[b] - anchors[a]) * (year - a) / (b - a)


def eu_data_centres(records):
    """The central demand file with a flat data centre load on Romania and Bulgaria. Returns bytes and the GWh added."""
    out_name = LOTB_DEMAND["EUDataCentres"]
    bom, eol, header, rows, years, cells = demand_table()
    for zone in DC_GWH:
        if (zone, "Energy") not in cells or (zone, "Peak") not in cells:
            fail("%s: no Peak and Energy rows for %s" % (DEMAND_V6, zone))
    added = {zone: {y: between(DC_GWH[zone], y) for y in years} for zone in DC_GWH}
    out = []
    for line in rows:
        p = line.split(",")
        if p[0] not in DC_GWH:
            out.append(line)
            continue
        new = list(p)
        for i, y in enumerate(years):
            gwh = added[p[0]][y]
            if gwh == 0:
                continue
            more = gwh if p[1] == "Energy" else gwh * 1000.0 / HOURS_YEAR       # GWh, or MW in every hour
            cell = num(float(p[2 + i]) + more, DEMAND_DECIMALS)
            records.append(("EUDataCentres", out_name, p[0], p[1], str(y), p[2 + i], cell, BASIS["EUDataCentres"]))
            new[2 + i] = cell
        out.append(",".join(new))
    return text_payload(bom, eol, header, out), years, added, cells


def demand_growth(sens, grid_years, records):
    """The central demand file with one country on a constant growth rate from GROWTH_FROM.

    Returns the bytes, the years, and by year the central total, the new total and their ratio (GWh)."""
    country, rate, direction = GROWTH[sens]
    out_name = LOTB_DEMAND[sens]
    zones = country_zones(country)
    bom, eol, header, rows, years, cells = demand_table()
    for zone in zones:
        if (zone, "Energy") not in cells or (zone, "Peak") not in cells:
            fail("%s: no Peak and Energy rows for %s" % (DEMAND_V6, zone))
    central = {y: sum(float(cells[(z, "Energy")][i]) for z in zones) for i, y in enumerate(years)}
    path = {y: central[y] if y <= GROWTH_FROM else central[GROWTH_FROM] * (1 + rate) ** (y - GROWTH_FROM) for y in years}
    ratio = {y: path[y] / central[y] for y in years}
    for y in grid_years:
        if y > GROWTH_FROM and (ratio[y] - 1) * direction <= 0:
            fail("%s: the %s total of %d is not on the expected side of the central" % (sens, country, y))
    out = []
    for line in rows:
        p = line.split(",")
        if p[0] not in zones:
            out.append(line)
            continue
        new = list(p)
        for i, y in enumerate(years):
            if y <= GROWTH_FROM:
                continue
            cell = num(float(p[2 + i]) * ratio[y], DEMAND_DECIMALS)
            records.append((sens, out_name, p[0], p[1], str(y), p[2 + i], cell, BASIS[sens]))
            new[2 + i] = cell
        out.append(",".join(new))
    return text_payload(bom, eol, header, out), years, central, path, ratio, zones


def build_lot_b(check, log, years, lines):
    """Second lot, part 1: four gas price cases and three demand cases. Returns True when every file is as built."""
    log.append("Sensitivities of v6, second lot, part 1: gas prices and demand (24 November 2026)")
    ok, records = True, []
    show = [y for y in SHOW if y in years]
    grid_years = [int(v) for v in text_table(DATA / robg.GRID_YEARS[robg.VARIANT_GRID])[3]]
    com, co2 = gas_reference(years, lines)
    trace = {"eu_gas_commodity_central": com, "eu_gas_co2_line": co2}
    gap0 = loss_gap(*((TRADE / n).read_bytes() for n in PRICE_V6))
    tr_rows = {}

    for sens, ttf in TTF_CASES.items():
        log.append(sens)
        payload, paths = gas_case_fuel(sens, years, lines, records, ttf=ttf, eu=True, turkiye=True)
        ok &= emit(SUPPLY / LOTB_FUEL[sens], payload, check, log)
        tr_rows[sens] = {y: v[1] for y, v in paths[(TR_COUNTRY, "Gas")].items()}
        factors = gas_border_factors(years, lines, ttf)
        prices = []
        for name, out_name in zip(PRICE_V6, LOTB_PRICE[sens]):
            payload, scaled, top = scaled_prices(name, out_name, factors, records, sens, BASIS["WorldGas prices"])
            prices.append(payload)
            ok &= emit(TRADE / out_name, payload, check, log)
            log.append("  %s: %d rows of %d neighbours scaled; largest cell %.3f USD/MWh (%.3f in the central row)" % (
                out_name, scaled, len(robg.INT_NODES), top[0], top[1]))
        gap1 = loss_gap(*prices)
        if gap1 > gap0 * max(1.0, max(factors.values())) + 1.1 * 10 ** -robg.DECIMALS:
            fail("%s: the scaled export prices are not the import prices less the loss: gap %.4f USD/MWh" % (sens, gap1))
        log.append("  export against import less %.1f percent: largest gap %.4f USD/MWh (central %.4f)" % (
            100 * robg.INT_LOSS, gap1, gap0))
        log.append("  %d rows repriced: %s" % (len(paths), ", ".join("%s %s" % k for k in paths)))
        log.append("  USD/MMBtu                     " + "".join("%9d" % y for y in show))
        eu_gas = {y: v[1] for y, v in paths[(robg.FUEL_ZONES[0], "Gas")].items()}
        for label, series in (("EU gas commodity, central", com), ("EU gas with CO2, central",
                              {y: com[y] + co2[y] for y in years}), ("EU gas with CO2, case", eu_gas),
                              ("Turkish gas, case", tr_rows[sens])):
            log.append("    %-28s" % label + "".join(("%9.2f" % series[y]) if y in series else "%9s" % "central"
                                                     for y in show))
        log.append("    %-28s" % "border price factor" + "".join("%9.3f" % factors[y] for y in show))
        tag = sens.lower()
        trace["%s_eu_gas_with_co2" % tag] = {y: eu_gas.get(y, com[y] + co2[y]) for y in years}
        trace["%s_turkiye_gas" % tag] = tr_rows[sens]
        trace["%s_border_price_factor" % tag] = factors

    log.append("TurkiyeGasHigh")
    payload, paths = gas_case_fuel("TurkiyeGasHigh", years, lines, records, ttf=TTF_CASES["WorldGasHigh"], turkiye=True)
    ok &= emit(SUPPLY / LOTB_FUEL["TurkiyeGasHigh"], payload, check, log)
    alone = {y: v[1] for y, v in paths[(TR_COUNTRY, "Gas")].items()}
    if list(paths) != [(TR_COUNTRY, "Gas")] or alone != tr_rows["WorldGasHigh"]:
        fail("TurkiyeGasHigh: the Turkish gas row is not the one of WorldGasHigh")
    log.append("  %s Gas alone, the row of WorldGasHigh: %s" % (TR_COUNTRY, ", ".join(
        "%d %.2f" % (y, alone[y]) for y in show if y in alone)))

    log.append("CaucasusGasNetback")
    payload, paths = gas_case_fuel("CaucasusGasNetback", years, lines, records, flat=NETBACK)
    ok &= emit(SUPPLY / LOTB_FUEL["CaucasusGasNetback"], payload, check, log)
    for key, path in paths.items():
        log.append("  %-10s gas %.1f from %d; central %s" % (key[0], NETBACK[key[0]], GAS_FROM, ", ".join(
            "%d %.2f" % (y, path[y][0]) for y in show if y in path)))
        trace["%s_gas_central" % key[0].lower()] = {y: v[0] for y, v in path.items()}
        trace["%s_gas_netback" % key[0].lower()] = {y: v[1] for y, v in path.items()}

    log.append("EUDataCentres")
    payload, _, added, cells = eu_data_centres(records)
    ok &= emit(LOAD / LOTB_DEMAND["EUDataCentres"], payload, check, log)
    for zone in DC_GWH:
        energy = dict(zip(years, (float(v) for v in cells[(zone, "Energy")])))
        peak = dict(zip(years, (float(v) for v in cells[(zone, "Peak")])))
        log.append("  %-9s TWh added, %s; load factor %s" % (zone, ", ".join(
            "%d %.2f" % (y, added[zone][y] / 1000) for y in grid_years), ", ".join(
            "%d %.3f to %.3f" % (y, energy[y] / peak[y] / 8.76, (energy[y] + added[zone][y]) / (
                peak[y] + added[zone][y] * 1000 / HOURS_YEAR) / 8.76) for y in grid_years if added[zone][y])))
        trace["datacentres_%s_gwh" % zone.lower()] = added[zone]

    for sens in GROWTH:
        log.append(sens)
        payload, _, central, path, ratio, zones = demand_growth(sens, grid_years, records)
        ok &= emit(LOAD / LOTB_DEMAND[sens], payload, check, log)
        country, rate, _ = GROWTH[sens]
        log.append("  %s, %d zones, %.1f percent a year from %d" % (country, len(zones), 100 * rate, GROWTH_FROM))
        log.append("    TWh                         " + "".join("%9d" % y for y in show))
        log.append("    %-28s" % "central" + "".join("%9.1f" % (central[y] / 1000) for y in show))
        log.append("    %-28s" % "case" + "".join("%9.1f" % (path[y] / 1000) for y in show))
        log.append("    %-28s" % "ratio" + "".join("%9.3f" % ratio[y] for y in show))
        tag = country.lower()
        trace["%s_demand_central_gwh" % tag] = central
        trace["%s_demand_case_gwh" % tag] = path

    ok &= emit(LOTB_TRACE_FULL, csv_bytes(TRACE_COLS, records), check, log)
    grouped = []
    for rec in records:               # one line per rewritten row: the cells are counted, no value is kept
        head = (rec[0], rec[1], rec[2], rec[3], rec[7])
        if grouped and grouped[-1][0] == head:
            grouped[-1][1].append(rec[4])
        else:
            grouped.append((head, [rec[4]]))
    ok &= emit(LOTB_TRACE_ROWS, csv_bytes(
        ["sensitivity", "file", "zone", "row", "cells", "first", "last", "basis"],
        [h[:4] + (len(f), f[0], f[-1], h[4]) for h, f in grouped]), check, log)
    names = list(trace)
    ok &= emit(LOTB_TRACE_PATHS, csv_bytes(
        ["year"] + names, [[y] + [("%.6f" % trace[n][y]) if y in trace[n] else "" for n in names] for y in years]),
        check, log)
    counts = []
    for rec in records:
        if rec[0] not in counts:
            counts.append(rec[0])
    log.append("  trace records per sensitivity: %s" % ", ".join(
        "%s %d" % (c, sum(1 for r in records if r[0] == c)) for c in counts))
    return ok


# ── Second lot, part 2 ──────────────────────────────────────────────────────
def quarter_hours():
    """Hours of each quarter of the model year."""
    _, _, header, rows = text_table(DATA / HOURS)
    if header.split(",")[:2] != ["q", "d"]:
        fail("%s: unexpected header" % HOURS)
    out = {}
    for line in rows:
        p = line.split(",")
        out[p[0]] = out.get(p[0], 0.0) + sum(float(v) for v in p[2:] if v != "")
    return out


def hydro_units():
    """Hydro units of the two countries, in the order of the fleet file, with their central availability.

    Returns the units (name, country, zone, status, start year, MW, cells of the four quarters, own row or not), the
    quarters and the hours of each quarter.
    """
    _, _, header, rows = text_table(SUPPLY / PLANT_V6)
    cols = header.split(",")
    g, z, tech, f, status, styr, cap = (cols.index(c) for c in ("g", "z", "tech", "f", "Status", "StYr", "Capacity"))
    _, _, header_a, rows_a = text_table(SUPPLY / AVAIL_V6)
    quarters = header_a.split(",")[1:]
    own = keyed(AVAIL_V6, header_a, rows_a, 1)
    _, _, header_d, rows_d = text_table(SUPPLY / AVAIL_DEFAULT_V6)
    if header_d.split(",")[3:] != quarters:
        fail("%s and %s: different quarters" % (AVAIL_V6, AVAIL_DEFAULT_V6))
    default = keyed(AVAIL_DEFAULT_V6, header_d, rows_d, 3)
    hours = quarter_hours()
    if sorted(hours) != sorted(quarters):
        fail("%s: the quarters are not those of %s" % (HOURS, AVAIL_V6))
    units = []
    for country in HYDRO_COUNTRIES:
        zones = country_zones(country)
        for p in (r.split(",") for r in rows):
            if p[z] not in zones or p[f] != HYDRO_FUEL or p[tech] not in HYDRO_TECHS:
                continue
            cells = own.get((p[g],), default.get((p[z], p[tech], p[f])))
            if cells is None or any(c == "" for c in cells):
                fail("%s: no availability for %s, in its own row or in the default of %s" % (AVAIL_V6, p[g], p[z]))
            units.append((p[g], country, p[z], p[status], p[styr], float(p[cap] or 0), cells, (p[g],) in own))
    stray = [p[g] for p in (r.split(",") for r in rows)
             if p[f] == HYDRO_FUEL and p[tech] in HYDRO_TECHS and p[z] in HUB_ZONES]
    if stray:
        fail("%s: hydro units in a hub zone, outside the zone map of the Int scope: %s" % (PLANT_V6, ", ".join(stray)))
    return units, quarters, hours


def hydro_year(sens, units, quarters, hours, records):
    """Availability file of a real hydro year. Returns bytes, one summary line per country and the counts."""
    year = HYDRO_YEAR[sens]
    out_name = LOTB2_AVAIL[sens]
    year_hours = 8784.0 if year % 4 == 0 else 8760.0
    total_hours = sum(hours.values())
    ratio, summary = {}, []
    for country in HYDRO_COUNTRIES:
        mine = [u for u in units if u[1] == country and u[3] == "1"]
        fleet = sum(u[5] for u in mine)
        if fleet <= 0:
            fail("%s: no existing hydro unit in %s" % (PLANT_V6, country))
        energy = sum(u[5] * sum(float(c) * hours[q] for c, q in zip(u[6], quarters)) for u in mine) / 1000.0
        cf_central = 1000.0 * energy / (fleet * total_hours)
        gwh, mw = HYDRO_OBS[country][year]
        if mw is None:                # the existing units of the file that ran in the real year
            mw = sum(u[5] for u in mine if u[4] == "" or int(u[4]) <= year)
        cf_real = 1000.0 * gwh / (mw * year_hours)
        ratio[country] = cf_real / cf_central
        summary.append((sens, country, year, gwh, mw, year_hours, cf_real, cf_central, ratio[country], fleet, energy,
                        energy * ratio[country]))
    bom, eol, header, rows = text_table(SUPPLY / AVAIL_V6)
    by_name = {u[0]: u for u in units}
    out, scaled = [], 0
    for line in rows:
        p = line.split(",")
        u = by_name.get(p[0])
        if u is None:
            out.append(line)
            continue
        new = [num(float(c) * ratio[u[1]], HYDRO_DECIMALS) for c in p[1:]]
        if any(float(c) > 1.0 for c in new):
            fail("%s: the availability of %s goes above 1" % (out_name, p[0]))
        for q, before, after in zip(quarters, p[1:], new):
            records.append((sens, out_name, u[2], p[0], q, before, after, BASIS[sens]))
        out.append(",".join([p[0]] + new))
        scaled += 1
    added = 0
    for u in units:                   # a unit on the default of its zone gets a row, so that the ratio reaches it
        if u[7]:
            continue
        new = [num(float(c) * ratio[u[1]], HYDRO_DECIMALS) for c in u[6]]
        records.append((sens, out_name, u[2], u[0], "row", "n/a, default of the zone", "added", BASIS["hydro added row"]))
        for q, before, after in zip(quarters, u[6], new):
            records.append((sens, out_name, u[2], u[0], q, before, after, BASIS["hydro added row"]))
        out.append(",".join([u[0]] + new))
        added += 1
    return text_payload(bom, eol, header, out), summary, scaled, added


def fleet_table():
    """The fleet of the five year grid as text, with the position of the columns the build cases read."""
    bom, eol, header, rows = text_table(SUPPLY / PLANT_V6)
    cols = header.split(",")
    ix = {c: cols.index(c) for c in ("g", "z", "tech", "f", "Status", "StYr", "RetrYr", "Capacity", "BuildLimitperYear")}
    return bom, eol, header, rows, ix


def caucasus_signed(records):
    """The fleet with the Caucasus held to its signed projects. Returns bytes, file name and the figures of the log."""
    bom, eol, header, rows, ix = fleet_table()
    g, z, tech, status, styr, retr, cap = (ix[c] for c in ("g", "z", "tech", "Status", "StYr", "RetrYr", "Capacity"))
    out_name = robg.VARIANT_FLEET % (robg.VARIANT_GRID, SIGNED_TAG)
    parts = [r.split(",") for r in rows]
    by_name = {p[g]: p for p in parts}
    zones = country_zones(SIGNED_COUNTRY)
    hubs = [p[g] for p in parts if p[z] in HUB_ZONES]
    if not hubs or any(by_name[n][status] != "3" or by_name[n][tech] not in RE_TECHS for n in hubs):
        fail("%s: the units of %s are not all solar and wind candidates" % (PLANT_V6, " and ".join(HUB_ZONES)))
    for name in SIGNED_OFF:
        p = by_name.get(name)
        if p is None or p[status] != "3" or p[z] not in zones or p[tech] not in RE_TECHS:
            fail("%s: %s is not a solar or wind candidate of %s" % (PLANT_V6, name, SIGNED_COUNTRY))
    closing = set(SIGNED_OFF) | set(hubs)
    signed = [p for p in parts if p[z] in zones and p[tech] in RE_TECHS and p[g] not in closing and (
        p[status] in ("2", "3") or (p[status] == "1" and p[styr] != "" and int(p[styr]) > SIGNED_AFTER))]
    total = sum(float(p[cap]) for p in signed)
    if abs(total - SIGNED_MW) > 0.05:
        fail("%s: the solar and wind projects of %s after %d add up to %s MW, expected %s" % (
            PLANT_V6, SIGNED_COUNTRY, SIGNED_AFTER, num(total, 1), num(SIGNED_MW, 1)))
    limits = {}
    for name, pipeline in PIPELINE_MW.items():
        c = by_name.get(name)
        if c is None or c[status] != "3" or c[z] != PIPELINE_ZONE:
            fail("%s: %s is not a candidate of %s" % (PLANT_V6, name, PIPELINE_ZONE))
        same = [p for p in parts if p[z] == PIPELINE_ZONE and p[tech] == c[tech] and p[g] != name]
        if any(p[status] not in ("0", "1", "2") for p in same):
            fail("%s: %s is not the only %s candidate of %s" % (PLANT_V6, name, c[tech], PIPELINE_ZONE))
        fleet = sum(float(p[cap]) for p in same if p[status] in ("1", "2")
                    and (p[styr] == "" or int(p[styr]) <= PIPELINE_YEAR)
                    and (p[retr] == "" or int(p[retr]) > PIPELINE_YEAR))
        room = pipeline - fleet
        if not 0.0 < room < float(c[cap]):
            fail("%s: the pipeline leaves %s MW to %s, outside its potential of %s" % (
                PLANT_V6, num(room, 1), name, c[cap]))
        limits[name] = (num(room, PLAN_DECIMALS), fleet)
    out = []
    for line, p in zip(rows, parts):
        q = list(p)
        if p[g] in closing:
            q[status] = CLOSED
            records.append(("CaucasusRESigned", out_name, p[z], p[g], "Status", p[status], CLOSED,
                            BASIS["CaucasusRESigned"]))
        elif p[g] in limits:
            q[cap] = limits[p[g]][0]
            records.append(("CaucasusRESigned", out_name, p[z], p[g], "Capacity", p[cap], q[cap],
                            BASIS["CaucasusRESigned pipeline"]))
        else:
            out.append(line)
            continue
        out.append(",".join(q))
    closed_mw = sum(float(by_name[n][cap]) for n in closing)
    return text_payload(bom, eol, header, out), out_name, len(closing), closed_mw, len(signed), total, limits


def eu_plan(records):
    """Fleet and availability files with the national plans of the EU zones as a floor on solar and wind.

    Returns the two payloads, the name of the fleet file, one line per zone, technology and model year (plan, fleet,
    reach of the candidates, step, steps to date, all in MW) and the number of availability rows added.
    """
    bom, eol, header, rows, ix = fleet_table()
    g, z, tech, status, styr, retr, cap, lim = (ix[c] for c in (
        "g", "z", "tech", "Status", "StYr", "RetrYr", "Capacity", "BuildLimitperYear"))
    out_name = robg.VARIANT_FLEET % (robg.VARIANT_GRID, PLAN_TAG)
    grid = [int(v) for v in text_table(DATA / robg.GRID_YEARS[robg.VARIANT_GRID])[3]]
    weight = {y: y - grid[k - 1] for k, y in enumerate(grid) if k}       # years a model year stands for
    parts = [r.split(",") for r in rows]
    names = {p[g] for p in parts}
    steps, lines = {}, []
    for (zone, kind), (rule, donor, anchors) in PLAN.items():
        mine = [p for p in parts if p[z] == zone and p[tech] == kind]
        if sum(1 for p in mine if p[g] == donor and p[status] == "3") != 1 or donor in steps:
            fail("%s: %s is not one %s candidate of %s" % (PLANT_V6, donor, kind, zone))
        fixed = [p for p in mine if p[status] in ("1", "2")]
        if any(p[retr] != "" and int(p[retr]) <= grid[-1] for p in fixed):
            fail("%s: a %s unit of %s retires before the end of the grid, the plan step would have to replace it" % (
                PLANT_V6, kind, zone))
        if any(p[status] == "2" and int(p[styr]) not in grid for p in fixed):
            fail("%s: a committed %s unit of %s starts off the years of the grid" % (PLANT_V6, kind, zone))
        cands = [p for p in mine if p[status] == "3" and p[g] not in PLAN_UNBUILT] if rule == "reach" else []
        if any(p[styr] == "" or int(p[styr]) <= grid[0] or p[lim] == "" for p in cands):
            fail("%s: a %s candidate of %s has no start year after %d or no yearly limit" % (
                PLANT_V6, kind, zone, grid[0]))
        done = 0.0
        for y in grid[1:]:
            fleet = sum(float(p[cap]) for p in fixed if p[styr] == "" or int(p[styr]) <= y)
            reach = sum(min(float(p[cap]), float(p[lim]) * sum(weight[t] for t in grid[1:] if int(p[styr]) <= t <= y))
                        for p in cands)
            plan = 1000.0 * between(anchors, y)
            step = round(plan - fleet - reach - done, PLAN_DECIMALS)
            if step < PLAN_MIN_MW:
                step = 0.0
            else:
                name = "%s_%s_Plan_%d" % (zone, kind, y)
                if name in names:
                    fail("%s: %s is already a unit" % (PLANT_V6, name))
                steps.setdefault(donor, []).append((name, y, step))
                done += step
            lines.append((zone, kind, y, rule, plan, fleet, reach, step, done))
    out = []
    for line, p in zip(rows, parts):
        mine = steps.get(p[g])
        if not mine:
            out.append(line)
            continue
        left = float(p[cap]) - sum(s[2] for s in mine)
        if left <= 0:
            fail("%s: the steps of the plan take the whole potential of %s" % (out_name, p[g]))
        q = list(p)
        q[cap] = num(left, PLAN_DECIMALS)
        records.append(("EUREPlan", out_name, p[z], p[g], "Capacity", p[cap], q[cap], BASIS["EUREPlan candidate"]))
        out.append(",".join(q))
        for name, y, mw in mine:          # a step is its candidate, committed in the year, with the size of the step
            s = list(p)
            s[g], s[status], s[styr], s[cap] = name, "2", str(y), num(mw, PLAN_DECIMALS)
            records.append(("EUREPlan", out_name, p[z], name, "row", "n/a", "committed, %s MW in %d" % (s[cap], y),
                            BASIS["EUREPlan"]))
            out.append(",".join(s))
    bom_a, eol_a, header_a, rows_a = text_table(SUPPLY / AVAIL_V6)
    own = keyed(AVAIL_V6, header_a, rows_a, 1)
    zone_of = {p[g]: p[z] for p in parts}
    out_a, added = list(rows_a), 0
    for donor, mine in steps.items():     # a candidate on the default of its zone leaves its steps on that default
        cells = own.get((donor,))
        for name, _, _ in mine if cells is not None else []:
            if (name,) in own:
                fail("%s: %s already has a row" % (AVAIL_V6, name))
            out_a.append(",".join([name] + list(cells)))
            records.append(("EUREPlan", PLAN_AVAIL, zone_of[donor], name, "row", "n/a", "added, the cells of %s" % donor,
                            BASIS["EUREPlan availability"]))
            added += 1
    return (text_payload(bom, eol, header, out), text_payload(bom_a, eol_a, header_a, out_a), out_name, lines,
            sum(len(v) for v in steps.values()), added)


def build_lot_b2(check, log):
    """Second lot, part 2. Returns True when every file matches (check) or is written."""
    log.append("")
    log.append("Sensitivities of v6, second lot, part 2 (24 November 2026)")
    ok, records = True, []

    units, quarters, hours = hydro_units()
    table = []
    for sens in HYDRO_YEAR:
        log.append(sens)
        payload, summary, scaled, added = hydro_year(sens, units, quarters, hours, records)
        ok &= emit(SUPPLY / LOTB2_AVAIL[sens], payload, check, log)
        log.append("  %s: %d rows of %s scaled, %d rows added for units on the default of their zone" % (
            LOTB2_AVAIL[sens], scaled, AVAIL_V6, added))
        for s in summary:
            log.append("  %-8s %d: %.1f GWh on %.1f MW, capacity factor %.4f against %.4f in the central, ratio %.4f; "
                       "existing fleet %.1f MW, %.1f GWh in the central, %.1f in the case" % (s[1:5] + s[6:]))
            table.append([s[0], s[1], s[2], "%.1f" % s[3], "%.1f" % s[4], "%.0f" % s[5], "%.6f" % s[6], "%.6f" % s[7],
                          "%.6f" % s[8], "%.1f" % s[9], "%.1f" % s[10], "%.1f" % s[11]])
    others = sum(1 for u in units if not u[7])
    log.append("  %d hydro units of %s in %s; %d without a row of their own; pumped storage is left as it is" % (
        len(units), " and ".join(HYDRO_COUNTRIES), PLANT_V6, others))
    ok &= emit(LOTB2_TRACE_HYDRO, csv_bytes(
        ["sensitivity", "country", "real_year", "output_gwh", "capacity_mw", "hours", "capacity_factor_real",
         "capacity_factor_central", "ratio", "existing_fleet_mw", "existing_central_gwh", "existing_case_gwh"], table),
        check, log)

    log.append("CaucasusRESigned")
    payload, name, closed, closed_mw, n_signed, signed_mw, limits = caucasus_signed(records)
    ok &= emit(SUPPLY / name, payload, check, log)
    log.append("  %d generic and hub candidates closed (status %s), %s MW of potential; %d signed solar and wind "
               "projects of %s after %d stay, %s MW" % (closed, CLOSED, num(closed_mw, 1), n_signed, SIGNED_COUNTRY,
                                                        SIGNED_AFTER, num(signed_mw, 1)))
    for cand, (room, fleet) in limits.items():
        log.append("  %s: potential %s MW, the pipeline of %d (%s MW) less the fleet (%s MW); yearly limit kept" % (
            cand, room, PIPELINE_YEAR, num(PIPELINE_MW[cand], 1), num(fleet, 1)))

    log.append("EUREPlan")
    fleet_bytes, avail_bytes, name, plan_lines, n_steps, n_rows = eu_plan(records)
    ok &= emit(SUPPLY / name, fleet_bytes, check, log)
    ok &= emit(SUPPLY / PLAN_AVAIL, avail_bytes, check, log)
    log.append("  %d committed steps; %d availability rows added for the steps of a candidate with its own row" % (
        n_steps, n_rows))
    for key in PLAN:
        mine = [p for p in plan_lines if (p[0], p[1]) == key]
        log.append("  %-8s %-11s (%s) %s" % (key[0], key[1], mine[0][3], "; ".join(
            "%d plan %.0f, fleet %.0f, reach %.0f, step %.0f" % (p[2], p[4], p[5], p[6], p[7]) for p in mine)))
    ok &= emit(LOTB2_TRACE_PLANS, csv_bytes(
        ["sensitivity", "zone", "tech", "year", "rule", "plan_mw", "fleet_mw", "candidate_reach_mw", "step_mw",
         "steps_to_date_mw"],
        [["EUREPlan", p[0], p[1], p[2], p[3]] + ["%.1f" % v for v in p[4:]] for p in plan_lines]), check, log)

    ok &= emit(LOTB2_TRACE_FULL, csv_bytes(TRACE_COLS, records), check, log)
    grouped = []
    for rec in records:               # one line per rewritten row: the cells are counted, no value is kept
        head = (rec[0], rec[1], rec[2], rec[3], rec[7])
        if grouped and grouped[-1][0] == head:
            grouped[-1][1].append(rec[4])
        else:
            grouped.append((head, [rec[4]]))
    ok &= emit(LOTB2_TRACE_ROWS, csv_bytes(
        ["sensitivity", "file", "zone", "row", "cells", "first", "last", "basis"],
        [h[:4] + (len(f), f[0], f[-1], h[4]) for h, f in grouped]), check, log)
    counts = []
    for rec in records:
        if rec[0] not in counts:
            counts.append(rec[0])
    log.append("  trace records per sensitivity: %s" % ", ".join(
        "%s %d" % (c, sum(1 for r in records if r[0] == c)) for c in counts))
    return ok


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
    ok &= build_lot_b(check, log, years, lines)
    ok &= build_lot_b2(check, log)
    return ok, log


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sensitivity files of the v6 inputs of the Black Sea model")
    ap.add_argument("--check", action="store_true", help="compare the files with a fresh build, write nothing")
    args = ap.parse_args(argv)
    ok, log = build(check=args.check)
    print("\n".join(log))
    if args.check:
        print("check: %s" % ("files match a fresh build" if ok else "files differ from a fresh build"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
