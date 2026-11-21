"""Version 6 inputs of the Black Sea model (RoBg family). One function per edit, new files only.

Every edit writes NEW files with a _v6 suffix next to the v5 files it starts from. No v5 file is opened for
writing. Files are compared byte for byte before they are written, so a second run changes nothing.

Edit 1. Border prices on TYNDP 2026 National Trends+.

  Source. Hourly marginal cost of the wholesale node (sheet E-Market of the NT+ time series workbooks), three
  weather scenarios per target year, read by build_tyndp26_hourly.py. Real EUR 2024.

  Method, per node and target year (2030, 2035, 2040, 2050):
    1. each weather scenario is capped at PRICE_CAP_EUR, the first demand response band of the runs. The
       price levels shared by several nodes stop below it; the levels above appear only where a node is
       short (demand response and unserved energy bands);
    2. the three scenarios are averaged hour by hour with the TYNDP weights;
    3. the series moves from the TYNDP clock (UTC+1, no daylight saving) to UTC, the clock of the model;
    4. the 28 representative days are read on the 2025 calendar (672 hours);
    5. each quarter is scaled so that its weighted mean on the representative days equals the mean of the
       quarter over its full hours. The weights of a quarter sum to its number of hours, so the annual
       weighted mean equals the 8,760 hour mean of step 3.
  Years. 2024 and 2025 are the v5 rows (observed day-ahead basis). 2030 to 2050 are linear between the target
  years, flat after 2050. 2026 to 2029 are linear, cell by cell, between the 2025 row and the 2030 row.
  Currency. EUR 2024 to USD with eur_usd 2024 of pre-analysis/config/price_deflators.csv.

  Files.
    trade/pTradePrice_robg_v6.csv, pTradePriceExport_robg_v6.csv (Int scope)
      Serbia, NorthMacedonia, Greece, Moldova, Ukraine repriced; Austria, Slovakia, Croatia, Slovenia added
      (borders of Hungary); Hungary removed (a modelled zone in v6). Export = import x (1 - 0.025).
      2024 and 2025 of the four new zones: the v5 Hungarian row times the ratio of the observed 2024
      day-ahead profiles (quarter by hour) of the zone and of Hungary.
    trade/pTradePrice_eu_central_v6.csv, pTradePriceExport_eu_central_v6.csv (source of the Ext scope)
      Bulgaria, Greece, Romania repriced with the netback of the v5 file: import = hub + W,
      export = hub x (1 - lambda) - W, parameters in pre-analysis/config/eu_price_netback.csv.
    Every other row is the v5 line, byte for byte.

Edit 2. Gas and CO2 of Romania, Bulgaria and Hungary on TYNDP 2026.

  Source. Sheet NT of TYNDP2026Scenarios_Fuel_CO2_Prices.xlsx (the price set behind the NT+ runs of edit 1):
  natural gas in EUR 2024 per net GJ and CO2 in EUR 2024 per tonne, for 2030, 2035, 2040 and 2050. The file is
  checked against its sha256 in Data/EU/TYNDP/manifest.csv.

  The carbon price of the model is system wide, so the CO2 cost of the three EU zones sits in their fuel price.
  A fuel price is therefore two lines:
    commodity  Gas and GasCCS: TYNDP 2026 natural gas x GJ per MMBtu x eur_usd x 0.90 (net to gross calorific
               value, the basis of the model heat rates). Other fuels: the v5 commodity
               (supply/pFuelPrice_tr_gas_flat.csv), unchanged.
    co2        TYNDP 2026 CO2 x eur_usd x carbon content of the fuel (epm/resources/pFuelCarbonContent.csv;
               GasCCS pays 10 percent of the gas content, as in v5).
  Years. 2024 and 2025 are the v5 cells. 2026 to 2029 are linear between the 2025 value and the 2030 value, line
  by line. From 2030 linear between the TYNDP years, flat after 2050.
  Hungary takes the Romanian rows of Gas, DomesticCoal (same TYNDP lignite group), Uranium and Biomass.
  Hungary also gets a Diesel row for its reserve gas turbines (edit 4): TYNDP 2026 light oil x 0.95 (net to
  gross) plus the CO2 of the fuel, at every year (no v5 Diesel row exists to start from).

  Files.
    supply/pFuelPrice_robg_v6.csv         the model file: commodity plus co2, four decimals. Rows of the other
                                          countries and rows without carbon are the v5 lines as read.
    supply/pFuelPrice_robg_v6_lines.csv   the two lines per country and fuel. Not read by the model. The gas
                                          sensitivities move the commodity line, the carbon ones the co2 line.

Edit 3. Border limits of Romania and Bulgaria on the TSO plan and the TYNDP 2026 reference grid.

  Sources.
    tdp    Transelectrica grid development plan 2024 to 2033, Table 5.4.2 (p56): net transfer capacity per
           border and direction, horizons 2030 and 2033. Romania with Hungary, Serbia and Bulgaria.
    tyndp  Reference grid of TYNDP 2026 (ReferenceGrid_Electricity.xlsx in Line-data.zip), read by
           build_tyndp_tidy.py into pre-analysis/output_tyndp/ntc.csv: reference NTC per direction for 2030,
           2035, 2040 and 2050. Bulgaria with Greece and Serbia, Romania with Moldova and Ukraine. The package
           is checked against its sha256 in Data/EU/TYNDP/manifest.csv.
  Years. The v5 cells are kept until 2029. From 2030 a value holds until the next horizon of its source, and the
  last horizon holds to the end. The four quarters carry the same value.
  Direction. In pExtTransferLimit, Import is into the modelled zone and Export out of it.

  Files.
    trade/pExtTransferLimit_robg_v6.csv    six borders, two directions, four quarters: 48 rows rewritten.
    trade/pTransferLimit_robg_hvdc_v6.csv  Romania to Bulgaria, both directions: 8 rows rewritten.
    Every other row is the v5 line, byte for byte.

  Hungary as a modelled zone (edit 4). A file touched by several edits is written by the first of them, with the
  later steps applied in the same build, so that a rerun of any edit leaves the file as it is.
    The eight Romania,Hungary rows leave the external table and are appended to the internal one as Romania,Hungary
    (from the Export rows) and Hungary,Romania (from the Import rows), same values.
    Hungary with Austria, Slovakia, Croatia, Slovenia, Serbia and Ukraine: 48 rows appended to the external table,
    read in the reference grid workbook itself (the tidy table ntc.csv holds Hungary with Romania and Ukraine
    only, and the reading is checked against it on these). 2024 to 2029 carry the 2025 NTC columns of the
    workbook, 2030 on the reference grid, each horizon held until the next.

Edit 4. Hungary as a modelled zone.

  Runs after edits 1 to 3 (prices of the Hungarian borders, Hungarian fuel rows, Hungarian limits). Every file is
  the v5 file, byte for byte, with Hungarian rows appended. Raw data come from the cache written by
  fetch_hungary_v6.py (Data/EU/cache_entso_e, each file checked against its sha256 in the cache manifest).

  Demand.
    energy   Ember demand of 2024 and 2025 (gross of solar behind the meter), flat to 2030, then the TYNDP 2026
             NT+ path of node HU00 relative to its 2030 value, flat after 2050.
    peak     hourly maximum of the 2025 ENTSO-E load; the ratio of peak to energy is held over the horizon.
    profile  2023 ENTSO-E load plus the solar output missing from the metered series (Ember less ENTSO-E, on the
             Renewables Ninja solar shape), divided by its annual maximum, read on the 28 representative days.
  Renewable profiles (2023, same representative days, UTC).
    PV       Renewables Ninja scaled to the capacity factor observed in 2025 (Ember output over mean capacity).
    wind     Renewables Ninja scaled to the Global Wind Atlas mean of the best 30 percent of the country.
             The fleet and the candidate carry a flat availability equal to the observed 2025 capacity factor over
             that mean, the convention of the Romanian and Bulgarian wind units (base.gms multiplies both).
  Fleet (same rows in the s5 and 2y files).
    named units from the ENTSO-E unit register (2025), aggregates for the rest, each fuel closed on the Ember
    capacity of 2025. Technical and cost cells come from the Romanian rows of the default table. Coal and lignite
    leave in 2030. Paks 1 to 4 and hydro carry a quarterly availability: the Ember 2025 level on the quarterly
    shape of the 2025 ENTSO-E output.
    committed   Paks 5 and 6, 2 x 1,200 MW in 2036, template Cernavoda-3, capex from EUR 12.5 bn.
    candidates  solar, wind, battery, biomass, CCGT and OCGT. Build limits follow the v2 growth rule; the rule
                is replayed on Bulgaria first and must return the deployed values.
  Other tables.
    Romanian rows copied as Hungary in the three default tables (plus OCGT on Diesel in the availability table),
    the planning reserve margin and the external trade share; the Romania to Hungary line takes the loss factor
    of Romania to Bulgaria. Hungary leaves zext; Austria, Slovakia, Croatia and Slovenia enter it.

  Files.
    zcmap_robg_v6.csv, zcmap_robg_geco_v6.csv, trade/zext_robg_v6.csv,
    reserve/pPlanningReserveMargin_robg_v6.csv, trade/pLossFactorInternal_robg_hvdc_v6.csv,
    trade/pMaxAnnualExternalTradeShare_robg_v6.csv, load/pDemandForecast_v6.csv, load/pDemandProfile_v6.csv,
    supply/pVREProfile_v6.csv, supply/pAvailabilityCustom_v6.csv, supply/pGenDataInputDefault_v6.csv,
    supply/pAvailabilityDefault_v6.csv, supply/pCapexTrajectoriesDefault_v6.csv,
    supply/pGenDataInput_v6_{s5,2y}.csv, supply/pStorageDataInput_v6_{s5,2y}.csv
    pre-analysis/output_prices/robg_v6/edit4_hungary.csv   every figure used, with its source. Not a model input.

Edit 5. Bulgarian fleet, from the comments of the country team and the national plan.

  Sources. Comments of the Bank's Bulgaria team on the inception deck (July 2026) and on the results deck
  (September 2026, slide 26); Bulgarian energy and climate plan 2021 to 2030, final update submitted in 2025.
  Applied to the v5 rows of the fleet and storage files, on both year grids, before the Hungarian rows of edit 4
  are appended. The two files are therefore written by edit 4 only; edit 5 writes the trace and controls that
  the files hold its rows. Every cell is checked against its v5 value before it is changed.
    BG1  coal and lignite: the blocks Agg_Lignite_2029, Agg_Coal_2029, Agg_Lignite_2033 and Agg_Coal_2033
         (3,916.2 MW) retire in 2038, with the two blocks that already do (country team: plants and mines
         supported to 2038; plan: coal out by 2038 at the latest). The names keep their v5 year: they are keys
         of the availability table. The model removes a plant from its retirement year on, and dispatch stays
         economic.
    BG2  batteries: Bulgaria_Battery at 6,200 MW in both files (country team), energy on the hours of the v5
         fleet (14,055.6 MWh over 4,500 MW); the committed block Bulgaria_Battery_2026 is removed from both files.
    BG3  Kozloduy 8: first year of the grid from 2037 (2040 on s5, 2038 on 2y). The country team gives units 7
         and 8 in operation between 2035 and 2037, the plan 2035 and 2040. A committed unit whose start year is
         not a model year is never built. Kozloduy 7 stays in 2036.
    BG4  Kozloduy 5 retires in 2047 and Kozloduy 6 in 2051 (plan and country team). No effect before 2040.

  Files.
    supply/pGenDataInput_v6_{s5,2y}.csv, supply/pStorageDataInput_v6_{s5,2y}.csv   (written by edit 4)
    pre-analysis/output_prices/robg_v6/edit5_bulgaria.csv   every cell changed: v5, v6 and basis. Not a model input.

Edit 6. Armenia, from the table of the country team.

  Source. Workbook of the Bank's Armenia team (September 2026), Official Use Only, kept outside git in
  Data/Armenia/country_team and pinned by its sha256. Each sheet holds a Current block (the rows the team
  received) and an Updated block (its own rows). No figure of the workbook is written in this file: the build
  reads the cells.
  Method. The Current block must be the v5 rows of the model, cell by cell, which proves that the team worked
  on this version. The difference between the two blocks is then applied to the v5 rows, and it must have the
  shape listed in AM_SHAPE (names of units and fields); anything else stops the build. Applied after edit 5 on
  both year grids. The files are written by edit 4 (fleet, storage, demand) and edit 2 (fuel); edit 6 writes the
  trace and controls that the files hold its rows.
    AM1  demand: peak and energy rows of the Updated block.
    AM2  nuclear: the new unit is a committed project and takes the place of the candidate. A modelling
         assumption, not a Government decision (flag F8). The existing unit is unchanged.
    AM3  the Hrazdan steam unit is decommissioned earlier. It has no capacity in the model: no effect.
    AM4  solar, wind and hydro candidates: size and yearly build limit.
    AM5  battery candidate: size, energy and yearly build limit.
    AM6  gas: plant gate price of the Updated block (flag F7). The row carries no CO2 component. Other fuels
         keep their v5 rows.
    F12  beyond the list AM1 to AM9, the table also resizes the existing small solar fleet, turns the two Ayg
         plants into candidate projects and drops the committed small solar block. Applied as given, under an
         interim flag. A unit turned into a candidate is not in service today: it can be built from
         AM_CANDIDATE_FROM on. The gas candidate of the model, which the table does not hold, is kept.
  Not applied here: the lines of the table (project cases, edit 8) and its hourly demand sheet.

  Files.
    supply/pGenDataInput_v6_{s5,2y}.csv, supply/pStorageDataInput_v6_{s5,2y}.csv, load/pDemandForecast_v6.csv
                                                                                      (written by edit 4)
    supply/pFuelPrice_robg_v6.csv                                                     (written by edit 2)
    epm/input/data_blacksea/v6_edit6_armenia_trace.csv   every cell changed: v5, v6 and basis. With the inputs,
                                                         not in git. Not a model input.

Usage (from EPM/):
    python pre-analysis/pipelines/build_robg_v6.py --edit 1
    python pre-analysis/pipelines/build_robg_v6.py --edit 1 --check     (compares, writes nothing)
    python pre-analysis/pipelines/build_robg_v6.py --edit 2 --edit 3
    python pre-analysis/pipelines/build_robg_v6.py --edit 2 --edit 4 --edit 5 --edit 6    (files, then the traces)
    python pre-analysis/pipelines/build_robg_v6.py --edit 1 --edit 2 --edit 3 --edit 4 --edit 5 --edit 6 --check
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
PRE = HERE.parents[1]
EPM = PRE.parent
DATA = EPM / "epm" / "input" / "data_blacksea"
TRADE = DATA / "trade"
CONFIG = PRE / "config"
HOURLY = PRE / "output_tyndp" / "hourly"
REPDAYS = PRE / "representative_days" / "output" / "blacksea" / "repr_days.csv"
ENTSOE_CACHE = EPM.parent / "Data" / "EU" / "cache_entso_e"
TRACE = PRE / "output_prices" / "robg_v6"

HOURS = ["t%02d" % i for i in range(1, 25)]
HEADER = ["zext", "q", "d", "year"] + HOURS
DECIMALS = 3

# ── Edit 1 ──────────────────────────────────────────────────────────────────
TYNDP_YEARS = (2030, 2035, 2040, 2050)
OBSERVED_YEARS = (2024, 2025)     # v5 rows, kept
BLEND_FROM, BLEND_TO = 2025, 2030
CLOCK_SHIFT = 1                   # TYNDP 2026 labels are UTC+1 without daylight saving; the model runs on UTC
CALENDAR_YEAR = 2025              # calendar of the representative days
FLOOR_USD = 0.01                  # a zero price switches the trade off in main.gms
INT_LOSS = 0.025                  # seller side AC loss on the Int borders
# Cap on the hourly TYNDP 2026 price, EUR 2024 per MWh. In the twelve nodes read, the price levels shared by
# several nodes stop at 288 (2050). The levels from 300 appear only where a node is short (Austria 2040, Slovakia
# 2040 and 2050, Bulgaria 2050) and include the round bands 300, 350 and 500, up to 2,830: demand response and
# unserved energy, not generating units. An external zone trades any volume at its price, so a scarcity price
# would be paid on the whole border capacity in every hour of the representative day that carries it.
PRICE_CAP_EUR = 300.0
INT_NODES = {"Serbia": "RS00", "NorthMacedonia": "MK00", "Greece": "GR00", "Moldova": "MD00", "Ukraine": "UA00",
             "Austria": "AT00", "Slovakia": "SK00", "Croatia": "HR00", "Slovenia": "SI00"}
INT_NEW = {"Austria": "AT", "Slovakia": "SK", "Croatia": "HR", "Slovenia": "SI"}   # ENTSO-E day-ahead codes
INT_REMOVED = ("Hungary",)
INT_PROXY, INT_PROXY_CODE = "Hungary", "HU"
OBSERVED_PROFILE_YEAR = 2024
PROFILE_MIN = 5.0                 # EUR/MWh; a cheaper Hungarian (quarter, hour) cell would blow up the ratio
EXT_NODES = {"Bulgaria": "BG00", "Greece": "GR00", "Romania": "RO00"}
MEAN_TOL_EUR = 0.05               # written table against the TYNDP mean, after rounding and the floor

# ── Edit 2 ──────────────────────────────────────────────────────────────────
SUPPLY = DATA / "supply"
TYNDP_DATA = EPM.parent / "Data" / "EU" / "TYNDP"
FUEL_BOOK = ("2026/inputs", "TYNDP2026Scenarios_Fuel_CO2_Prices.xlsx")   # folder and file, as in manifest.csv
FUEL_SHEET = "NT"
TYNDP_GAS_ROW, TYNDP_CO2_ROW = "Natural Gas", "CO2 price"
GJ_PER_MMBTU = 1.055056
NCV_GCV_GAS = 0.90                # TYNDP prices are per net GJ; the gas price and the heat rates of the model are gross
FUEL_ZONES = ("Romania", "Bulgaria")
FUEL_NEW_ZONE, FUEL_NEW_FROM = "Hungary", "Romania"      # Hungary, a modelled zone in v6, takes Romanian rows
FUEL_NEW_ROWS = ("Gas", "DomesticCoal", "Uranium", "Biomass")
# Diesel of the Hungarian reserve gas turbines (edit 4). No zone has a Diesel row in v5, so the price is the TYNDP
# 2026 light oil row at every year: flat before 2030, linear between the target years, flat after 2050.
TYNDP_OIL_ROW, OIL_FUEL = "Light oil", "Diesel"
NCV_GCV_OIL = 0.95                # net to gross calorific value of oil products (5 percent, 10 percent for gas)
GAS_ON_TYNDP = ("Gas", "GasCCS")  # fuels whose commodity line moves to the TYNDP 2026 natural gas price
CCS_FUEL, CCS_EMITTED = "GasCCS", 0.10   # v5 convention: GasCCS pays the CO2 of 10 percent of the gas burnt
NO_CARBON = ("Hydrogen",)         # not in pFuelCarbonContent.csv, no CO2 in v5
FUEL_DECIMALS, LINE_DECIMALS = 4, 6
V5_SPLIT_TOL = 0.0005             # USD/MMBtu: v5 sum against v5 commodity plus CO2 x carbon content, rounding only

# ── Edit 3 ──────────────────────────────────────────────────────────────────
NTC_TABLE = PRE / "output_tyndp" / "ntc.csv"             # built by build_tyndp_tidy.py
NTC_PACKAGE = ("2026/inputs", "Line-data.zip")           # folder and file, as in manifest.csv
NTC_EDITION, NTC_SCENARIO, NTC_METRIC = "2026", "NT+", "ntc_reference"
LIMIT_FROM = 2030                 # first year on the new limits; the v5 cells are kept before it
# Transelectrica grid development plan 2024 to 2033, Table 5.4.2 (p56): net transfer capacity per border and
# direction, MW, horizons 2030 and 2033. The table has no Moldova row. Ukraine is listed for the totals control.
TDP_YEARS = (2030, 2033)
TDP_NTC = {
    ("Romania", "Hungary"): (1700, 2300), ("Hungary", "Romania"): (1700, 2300),
    ("Romania", "Bulgaria"): (2550, 2550), ("Bulgaria", "Romania"): (2550, 2550),
    ("Romania", "Serbia"): (1300, 2000), ("Serbia", "Romania"): (1300, 2000),
    ("Romania", "Ukraine"): (500, 500), ("Ukraine", "Romania"): (500, 500),
}
TDP_TOTAL = (6050, 7350)          # export row and import row of the table, same totals
LIMIT_NODES = {"Romania": "RO00", "Bulgaria": "BG00", "Greece": "GR00", "Serbia": "RS00", "Moldova": "MD00",
               "Ukraine": "UA00"}
# border (modelled zone, neighbour) -> source of the limit from LIMIT_FROM
EXT_LIMIT_RULES = {("Romania", "Hungary"): "tdp", ("Romania", "Serbia"): "tdp",
                   ("Romania", "Moldova"): "tyndp", ("Romania", "Ukraine"): "tyndp",
                   ("Bulgaria", "Greece"): "tyndp", ("Bulgaria", "Serbia"): "tyndp"}
INT_LIMIT_RULES = {("Romania", "Bulgaria"): "tdp", ("Bulgaria", "Romania"): "tdp"}
LIMIT_SOURCE = {"tdp": "Transelectrica TDP 2024 to 2033, Table 5.4.2",
                "tyndp": "TYNDP 2026 reference grid, NTC"}
QUARTERS = ("Q1", "Q2", "Q3", "Q4")
# Hungary is a modelled zone in v6 (edit 4). Its limit with Romania moves from the external table to the internal
# one, and its six other borders enter the external table from the reference grid workbook of TYNDP 2026: the tidy
# table ntc.csv holds only the borders of its scope (Hungary with Romania and with Ukraine).
HU_ZONE, HU_NODE = "Hungary", "HU00"
HU_BORDERS = {"Austria": "AT00", "Slovakia": "SK00", "Croatia": "HR00", "Slovenia": "SI00", "Serbia": "RS00",
              "Ukraine": "UA00"}
GRID_MEMBER = "Line-data/ReferenceGrid_Electricity.xlsx"     # member of NTC_PACKAGE
GRID_SHEET = "Year_%d"
GRID_HEADERS = ("Border", "TY2026_2025TY Direct NTC (MW)", "TY2026_2025TY Indirect NTC (MW)", "Summary Direction 1",
                "Summary Direction 2")
NTC_2025_METRIC = "ntc_2025"
GRID_SOURCE = "TYNDP 2026 reference grid workbook"


def fail(msg):
    raise SystemExit("ERROR: " + msg)


# ── csv files kept byte for byte ────────────────────────────────────────────
class Lines:
    """A csv read as text lines: the lines that are not rewritten stay byte for byte."""

    def __init__(self, path):
        raw = Path(path).read_bytes()
        self.bom = raw[:3] == b"\xef\xbb\xbf"
        text = raw[3:].decode("utf-8") if self.bom else raw.decode("utf-8")
        self.eol = "\r\n" if "\r\n" in text[:4096] else "\n"
        body = text.split(self.eol)
        if body[-1] != "":
            fail("%s: no end of line after the last row" % Path(path).name)
        self.header, self.rows = body[0], body[1:-1]
        if self.header.split(",") != HEADER:
            fail("%s: unexpected header" % Path(path).name)

    @staticmethod
    def key(line):
        z, q, d, y = line.split(",", 4)[:4]
        return z, q, d, int(y)

    @staticmethod
    def values(line):
        return np.array(line.split(",")[4:], float)

    def encode(self, rows):
        text = self.eol.join([self.header] + rows) + self.eol
        return (b"\xef\xbb\xbf" if self.bom else b"") + text.encode("utf-8")


def fmt(zone, q, d, year, values):
    return ",".join([zone, q, d, str(year)] + ["%.*f" % (DECIMALS, v) for v in values])


def emit(path, payload, check, log):
    """Write `payload` unless the file already holds it. In check mode nothing is written."""
    path = Path(path)
    old = path.read_bytes() if path.exists() else None
    if old == payload:
        log.append("  unchanged  %s" % path.relative_to(EPM).as_posix())
        return True
    if check:
        log.append("  %s  %s" % ("DIFFERS  " if old is not None else "MISSING  ", path.relative_to(EPM).as_posix()))
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    log.append("  written    %s (%d bytes)" % (path.relative_to(EPM).as_posix(), len(payload)))
    return True


def config_value(name, series, year):
    df = pd.read_csv(CONFIG / name, comment="#")
    hit = df[(df["series"] == series) & (df["year"] == year)]
    if len(hit) != 1:
        fail("%s: %d rows for %s %s" % (name, len(hit), series, year))
    return float(hit["value"].iloc[0])


# ── Representative days ─────────────────────────────────────────────────────
def repday_selection():
    """672 rows: position j in the 8,760 hours of the UTC year, quarter, day, hour slot and model weight."""
    ph = pd.read_csv(DATA / "pHours.csv")
    w = ph.melt(id_vars=["q", "d"], value_vars=HOURS, var_name="t", value_name="w")
    rd = pd.read_csv(REPDAYS)
    rd["season_i"] = rd["season"].str[1:].astype(int)
    idx = pd.date_range("%d-01-01" % CALENDAR_YEAR, periods=8760, freq="h")
    cal = pd.DataFrame({"j": np.arange(8760), "season_i": (idx.month - 1) // 3 + 1, "hod": idx.hour, "date": idx.date})
    cal["day"] = cal.groupby("season_i").cumcount() // 24 + 1
    sel = cal.merge(rd[["season_i", "day", "season", "daytype", "weight"]], on=["season_i", "day"])
    sel = sel.rename(columns={"season": "q", "daytype": "d", "weight": "w_repr"})
    sel["t"] = ["t%02d" % (x + 1) for x in sel["hod"]]
    sel = sel.merge(w, on=["q", "d", "t"], how="left").sort_values(["q", "d", "hod"]).reset_index(drop=True)
    if len(sel) != 28 * 24 or sel["w"].isna().any():
        fail("representative days: %d hours selected, 672 expected" % len(sel))
    if (sel.groupby(["q", "d"])["date"].nunique() != 1).any():
        fail("a representative day spans two calendar dates")
    if not np.allclose(sel["w"], sel["w_repr"]):
        fail("pHours.csv and repr_days.csv disagree on the day weights")
    quarter_hours = cal.groupby("season_i").size()
    got = sel.groupby("season_i")["w"].sum()
    if not np.allclose(got.to_numpy(), quarter_hours.to_numpy()):
        fail("the weights of a quarter do not sum to its hours: %s" % got.to_dict())
    return sel, cal["season_i"].to_numpy()


def tyndp_anchor_tables(nodes, sel, season_of_hour, trace):
    """{(node, year): frame q, d, t01..t24 in EUR 2024} for the TYNDP target years."""
    path = HOURLY / "tyndp26_ntplus_hourly.parquet"
    if not path.exists():
        fail("%s is missing: run build_tyndp26_hourly.py first" % path.name)
    hourly = pd.read_parquet(path)
    hourly = hourly[(hourly["kind"] == "price") & hourly["node"].isin(nodes)]
    weights = pd.read_csv(HOURLY / "ws_weights.csv", dtype={"ws": str})
    weights["ws"] = weights["ws"].str.zfill(3)
    series = {(n, int(y), str(ws).zfill(3)): g.sort_values("hour")["value"].to_numpy()
              for (n, y, ws), g in hourly.groupby(["node", "year", "ws"])}
    j, w, sq = sel["j"].to_numpy(), sel["w"].to_numpy(), sel["season_i"].to_numpy()
    out = {}
    for node in nodes:
        for year in TYNDP_YEARS:
            wy = weights[weights["year"] == year]
            if len(wy) != 3 or abs(wy["weight"].sum() - 1.0) > 1e-9:
                fail("weather scenario weights of %d do not sum to one" % year)
            native = capped = 0.0
            hours_capped = 0.0
            for ws, wt in zip(wy["ws"], wy["weight"]):
                v = series.get((node, year, ws))
                if v is None or len(v) != 8760 or not np.isfinite(v).all() or v.min() < 0:
                    fail("%s %d WS%s: hourly price series missing or invalid" % (node, year, ws))
                native = native + wt * v
                capped = capped + wt * np.minimum(v, PRICE_CAP_EUR)
                hours_capped += wt * float((v > PRICE_CAP_EUR).sum())
            utc = np.roll(capped, -CLOCK_SHIFT)
            sample = utc[j].copy()
            factors = {}
            for q in (1, 2, 3, 4):
                m = sq == q
                target = utc[season_of_hour == q].mean()
                got = (sample[m] * w[m]).sum() / w[m].sum()
                if got <= 0:
                    fail("%s %d Q%d: the representative days carry a zero price" % (node, year, q))
                factors[q] = target / got
                sample[m] *= factors[q]
            mean = float((sample * w).sum() / w.sum())
            if abs(mean - utc.mean()) > 1e-9 * max(1.0, utc.mean()):
                fail("%s %d: weighted mean %.6f against %.6f" % (node, year, mean, utc.mean()))
            frame = sel[["q", "d", "t"]].assign(p=sample).pivot(index=["q", "d"], columns="t", values="p")
            out[(node, year)] = frame[HOURS]
            trace.append({"node": node, "year": year, "mean_native_eur": float(np.mean(native)),
                          "mean_capped_eur": float(utc.mean()), "cap_eur": PRICE_CAP_EUR,
                          "hours_above_cap": hours_capped, "factor_q1": factors[1], "factor_q2": factors[2],
                          "factor_q3": factors[3], "factor_q4": factors[4], "repday_max_eur": float(sample.max())})
    return out


def blend(points, year):
    """Linear between the target years, flat outside them."""
    ys = sorted(points)
    if year <= ys[0]:
        return points[ys[0]]
    if year >= ys[-1]:
        return points[ys[-1]]
    if year in points:
        return points[year]
    hi = min(k for k in ys if k > year)
    lo = max(k for k in ys if k < year)
    f = (year - lo) / (hi - lo)
    return points[lo] * (1 - f) + points[hi] * f


def observed_profile(code):
    """Mean observed day-ahead price by (quarter, hour slot), UTC, EUR/MWh."""
    sys.path.insert(0, str(HERE.parent))
    import eu_price  # noqa: E402  (reader of the ENTSO-E day-ahead cache, with its checks)
    s = eu_price.load_prices(ENTSOE_CACHE, code, OBSERVED_PROFILE_YEAR)
    df = pd.DataFrame({"p": s.to_numpy(), "q": ["Q%d" % ((m - 1) // 3 + 1) for m in s.index.month],
                       "t": ["t%02d" % (h + 1) for h in s.index.hour]})
    return df.groupby(["q", "t"])["p"].mean().unstack("t")[HOURS], float(s.mean())


def priced_rows(zone, anchors, obs_import, usd, to_import, to_export, days, years):
    """{(q, d, year): (import values, export values)} from 2026 on, in USD.

    anchors: {tyndp year: frame (q, d) x hours, EUR}; obs_import / obs_export: {(q, d): 2025 values, USD}.
    """
    obs_imp, obs_exp = obs_import
    out = {}
    for (q, d) in days:
        pts = {y: anchors[y].loc[(q, d)].to_numpy() for y in TYNDP_YEARS}
        imp30, exp30 = to_import(pts[BLEND_TO]), to_export(pts[BLEND_TO])
        for year in years:
            if year < BLEND_TO:
                f = (year - BLEND_FROM) / (BLEND_TO - BLEND_FROM)
                imp = obs_imp[(q, d)] * (1 - f) + imp30 * f
                exp = obs_exp[(q, d)] * (1 - f) + exp30 * f
            else:
                hub = blend(pts, year)
                imp, exp = to_import(hub), to_export(hub)
            # The observed 2025 rows hold a few negative import cells next to an export price floored at 0.01.
            # A row written here never carries an import price below the floor or an export price above the
            # import price of the same hour.
            imp = np.maximum(imp, FLOOR_USD)
            exp = np.minimum(np.maximum(exp, FLOOR_USD), imp)
            out[(q, d, year)] = (imp, exp)
    return out


def table_means(rows, sel_w):
    """{(zone, year): weighted mean of a list of csv lines}, weights by (q, d)."""
    acc = {}
    for line in rows:
        z, q, d, y = Lines.key(line)
        v = Lines.values(line)
        w = sel_w[(q, d)]
        a = acc.setdefault((z, y), [0.0, 0.0])
        a[0] += float((v * w).sum())
        a[1] += float(w.sum())
    return {k: a / b for k, (a, b) in acc.items()}


def edit1_border_prices(check=False):
    log = ["Edit 1. Border prices on TYNDP 2026 NT+"]
    usd = config_value("price_deflators.csv", "eur_usd", 2024)
    lam = config_value("eu_price_netback.csv", "lambda", 0)
    lam_ro = config_value("eu_price_netback.csv", "lambda_Romania", 0)
    wheel = config_value("eu_price_netback.csv", "W", 0)
    if abs(lam - INT_LOSS) > 1e-12:
        fail("the AC loss of eu_price_netback.csv (%s) is not the Int border loss (%s)" % (lam, INT_LOSS))
    log.append("  eur_usd 2024 %.4f | Int loss %.3f | Ext lambda %.3f (Romania %.3f), W %.2f EUR | clock shift %d h"
               % (usd, INT_LOSS, lam, lam_ro, wheel, CLOCK_SHIFT))

    sel, season_of_hour = repday_selection()
    sel_w = {k: g.sort_values("hod")["w"].to_numpy() for k, g in sel.groupby(["q", "d"])}
    trace = []
    nodes = sorted(set(INT_NODES.values()) | set(EXT_NODES.values()))
    anchors = tyndp_anchor_tables(nodes, sel, season_of_hour, trace)
    trace_df = pd.DataFrame(trace)

    def node_anchors(node):
        return {y: anchors[(node, y)] for y in TYNDP_YEARS}

    ok = True
    means = []

    # ── Int: pTradePrice_robg_v6, pTradePriceExport_robg_v6 ────────────────
    src_i, src_e = Lines(TRADE / "pTradePrice_robg.csv"), Lines(TRADE / "pTradePriceExport_robg.csv")
    if [Lines.key(r) for r in src_i.rows] != [Lines.key(r) for r in src_e.rows]:
        fail("the v5 import and export files of the Int scope do not share their keys")
    keys = [Lines.key(r) for r in src_i.rows]
    greece = [(q, d, y) for (z, q, d, y) in keys if z == "Greece"]
    days = list(dict.fromkeys((q, d) for (q, d, _) in greece))
    years = sorted({y for (_, _, y) in greece})
    new_years = [y for y in years if y not in OBSERVED_YEARS]
    if len(days) != 28 or set(days) != set(sel_w) or years[0] != OBSERVED_YEARS[0]:
        fail("the v5 Greece block does not carry the 28 representative days from %d" % OBSERVED_YEARS[0])
    at = {k: i for i, k in enumerate(keys)}
    if len(at) != len(keys):
        fail("duplicate keys in pTradePrice_robg.csv")

    def v5_obs(zone, lines):
        return {(q, d): Lines.values(lines.rows[at[(zone, q, d, BLEND_FROM)]]) for (q, d) in days}

    def imp_int(hub):
        return np.maximum(hub * usd, FLOOR_USD)

    def exp_int(hub):
        return np.maximum(hub * usd * (1 - INT_LOSS), FLOOR_USD)

    # observed years of the four new zones: v5 Hungarian row x ratio of the observed profiles
    prof_hu, mean_hu = observed_profile(INT_PROXY_CODE)
    if prof_hu.to_numpy().min() < PROFILE_MIN:
        fail("the observed Hungarian profile falls to %.2f EUR/MWh" % prof_hu.to_numpy().min())
    new_obs = {}
    for zone, code in INT_NEW.items():
        prof, mean = observed_profile(code)
        ratio = prof / prof_hu
        log.append("  %s 2024 and 2025: Hungary v5 x observed %d profile ratio, annual %.3f (%.1f against %.1f EUR/MWh), "
                   "cells %.2f to %.2f" % (zone, OBSERVED_PROFILE_YEAR, mean / mean_hu, mean, mean_hu,
                                           ratio.to_numpy().min(), ratio.to_numpy().max()))
        for year in OBSERVED_YEARS:
            for (q, d) in days:
                hu = Lines.values(src_i.rows[at[(INT_PROXY, q, d, year)]])
                imp = np.maximum(hu * ratio.loc[q].to_numpy(), FLOOR_USD)
                new_obs[(zone, q, d, year)] = (imp, np.maximum(imp * (1 - INT_LOSS), FLOOR_USD))

    priced = {}
    for zone, node in INT_NODES.items():
        if zone in INT_NEW:
            obs = ({k: new_obs[(zone,) + k + (BLEND_FROM,)][0] for k in days},
                   {k: new_obs[(zone,) + k + (BLEND_FROM,)][1] for k in days})
        else:
            obs = (v5_obs(zone, src_i), v5_obs(zone, src_e))
        priced[zone] = priced_rows(zone, node_anchors(node), obs, usd, imp_int, exp_int, days, new_years)

    out_i, out_e, kept, replaced = [], [], 0, 0
    for li, le in zip(src_i.rows, src_e.rows):
        z, q, d, y = Lines.key(li)
        if z in INT_REMOVED:
            continue
        if z in INT_NODES and y not in OBSERVED_YEARS:
            imp, exp = priced[z][(q, d, y)]
            out_i.append(fmt(z, q, d, y, imp))
            out_e.append(fmt(z, q, d, y, exp))
            replaced += 1
        else:
            out_i.append(li)
            out_e.append(le)
            kept += 1
    for zone in INT_NEW:
        if any(k[0] == zone for k in keys):
            fail("%s already has rows in the v5 file" % zone)
        for (q, d, y) in greece:
            imp, exp = new_obs[(zone, q, d, y)] if y in OBSERVED_YEARS else priced[zone][(q, d, y)]
            out_i.append(fmt(zone, q, d, y, imp))
            out_e.append(fmt(zone, q, d, y, exp))
    added = len(INT_NEW) * len(greece)
    removed = sum(1 for k in keys if k[0] in INT_REMOVED)
    log.append("  Int: %d v5 lines kept byte for byte, %d repriced, %d removed (%s), %d added (%s); %d rows"
               % (kept, replaced, removed, ", ".join(INT_REMOVED), added, ", ".join(INT_NEW), len(out_i)))
    for label, rows in (("pTradePrice_robg_v6", out_i), ("pTradePriceExport_robg_v6", out_e)):
        for (z, y), m in table_means(rows, sel_w).items():
            means.append({"file": label, "zext": z, "year": y, "mean_usd": m})
    ok &= emit(TRADE / "pTradePrice_robg_v6.csv", src_i.encode(out_i), check, log)
    ok &= emit(TRADE / "pTradePriceExport_robg_v6.csv", src_e.encode(out_e), check, log)

    # ── Ext source: pTradePrice_eu_central_v6, pTradePriceExport_eu_central_v6 ──
    cen_i, cen_e = Lines(TRADE / "pTradePrice_eu_central.csv"), Lines(TRADE / "pTradePriceExport_eu_central.csv")
    ckeys = [Lines.key(r) for r in cen_i.rows]
    if ckeys != [Lines.key(r) for r in cen_e.rows]:
        fail("the v5 import and export eu_central files do not share their keys")
    cat = {k: i for i, k in enumerate(ckeys)}

    def imp_ext(hub):
        return np.maximum((hub + wheel) * usd, FLOOR_USD)

    ext_priced = {}
    for zone, node in EXT_NODES.items():
        lam_z = lam_ro if zone == "Romania" else lam

        def exp_ext(hub, lam_z=lam_z):
            return np.maximum((hub * (1 - lam_z) - wheel) * usd, FLOOR_USD)

        obs = ({(q, d): Lines.values(cen_i.rows[cat[(zone, q, d, BLEND_FROM)]]) for (q, d) in days},
               {(q, d): Lines.values(cen_e.rows[cat[(zone, q, d, BLEND_FROM)]]) for (q, d) in days})
        ext_priced[zone] = priced_rows(zone, node_anchors(node), obs, usd, imp_ext, exp_ext, days, new_years)
    out_ci, out_ce, kept, replaced = [], [], 0, 0
    for li, le in zip(cen_i.rows, cen_e.rows):
        z, q, d, y = Lines.key(li)
        if z in EXT_NODES and y not in OBSERVED_YEARS:
            imp, exp = ext_priced[z][(q, d, y)]
            out_ci.append(fmt(z, q, d, y, imp))
            out_ce.append(fmt(z, q, d, y, exp))
            replaced += 1
        else:
            out_ci.append(li)
            out_ce.append(le)
            kept += 1
    log.append("  Ext source: %d v5 lines kept byte for byte, %d repriced (%s); %d rows"
               % (kept, replaced, ", ".join(EXT_NODES), len(out_ci)))
    for label, rows in (("pTradePrice_eu_central_v6", out_ci), ("pTradePriceExport_eu_central_v6", out_ce)):
        for (z, y), m in table_means(rows, sel_w).items():
            means.append({"file": label, "zext": z, "year": y, "mean_usd": m})
    ok &= emit(TRADE / "pTradePrice_eu_central_v6.csv", cen_i.encode(out_ci), check, log)
    ok &= emit(TRADE / "pTradePriceExport_eu_central_v6.csv", cen_e.encode(out_ce), check, log)

    # ── Checks on the written tables ───────────────────────────────────────
    means_df = pd.DataFrame(means)
    worst = 0.0
    for label, nodes_of, add in (("pTradePrice_robg_v6", INT_NODES, 0.0), ("pTradePrice_eu_central_v6", EXT_NODES, wheel)):
        for zone, node in nodes_of.items():
            for year in TYNDP_YEARS:
                got = means_df[(means_df["file"] == label) & (means_df["zext"] == zone) & (means_df["year"] == year)]
                want = float(trace_df[(trace_df["node"] == node) & (trace_df["year"] == year)]["mean_capped_eur"].iloc[0])
                gap = float(got["mean_usd"].iloc[0]) / usd - add - want
                worst = max(worst, abs(gap))
                if abs(gap) > MEAN_TOL_EUR:
                    fail("%s %s %d: table mean %.3f EUR/MWh away from the TYNDP mean" % (label, zone, year, gap))
    log.append("  check: weighted mean of the import tables against the TYNDP 2026 mean, largest gap %.4f EUR/MWh "
               "(tolerance %.2f)" % (worst, MEAN_TOL_EUR))
    for rows, name in ((out_i, "pTradePrice_robg_v6"), (out_ci, "pTradePrice_eu_central_v6")):
        count = pd.Series([Lines.key(r)[0] for r in rows]).value_counts()
        if set(count.unique()) != {len(greece)}:
            fail("%s: zones with a row count other than %d: %s" % (name, len(greece), count.to_dict()))

    # ── Traces ─────────────────────────────────────────────────────────────
    zone_of = {}
    for z, n in list(INT_NODES.items()) + list(EXT_NODES.items()):
        zone_of.setdefault(n, []).append(z)
    trace_df.insert(1, "zext", [" ".join(sorted(set(zone_of[n]))) for n in trace_df["node"]])
    wide = means_df.pivot_table(index=["file", "zext"], columns="year", values="mean_usd").reset_index()
    ok &= emit(TRACE / "edit1_tyndp_anchors.csv", trace_df.round(4).to_csv(index=False, lineterminator="\n").encode(), check, log)
    ok &= emit(TRACE / "edit1_table_means_usd.csv", wide.round(3).to_csv(index=False, lineterminator="\n").encode(), check, log)

    show = trace_df[trace_df["hours_above_cap"] > 0.5][["node", "year", "mean_native_eur", "mean_capped_eur", "hours_above_cap"]]
    log.append("  hours above the cap (weighted over the weather scenarios):")
    log.extend("    " + line for line in show.round(2).to_string(index=False).splitlines())
    log.append("  weighted mean of the import tables, USD/MWh:")
    cols = [y for y in (2025, 2030, 2035, 2040) if y in wide.columns]
    log.extend("    " + line for line in wide[wide["file"].str.startswith("pTradePrice_")][["file", "zext"] + cols]
               .round(2).to_string(index=False).splitlines())
    return ok, log


# ── Edit 2: fuel and CO2 ────────────────────────────────────────────────────
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tyndp_fuel_table():
    """Price rows of the NT sheet of the TYNDP 2026 fuel and CO2 workbook: {label: (unit, {year: value})}."""
    import openpyxl

    folder, name = FUEL_BOOK
    path = TYNDP_DATA / folder / name
    man = pd.read_csv(TYNDP_DATA / "manifest.csv", dtype=str)
    hit = man[(man["folder"] == folder) & (man["file"] == name)]
    if len(hit) != 1:
        fail("manifest.csv: %d rows for %s" % (len(hit), name))
    if sha256_of(path) != hit["sha256"].iloc[0]:
        fail("%s: sha256 differs from manifest.csv" % name)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = [list(r) for r in wb[FUEL_SHEET].iter_rows(values_only=True)]
    wb.close()

    def year_of(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    cols, unit, table = None, None, {}
    for r in rows:
        if cols is None:
            found = {}
            for i, v in enumerate(r):
                y = year_of(v)
                if y in TYNDP_YEARS and y not in found:
                    found[y] = i
            if set(found) == set(TYNDP_YEARS):
                cols = found
            continue
        if any(isinstance(v, str) and v.strip().lower().startswith("shares") for v in r):
            break
        first = min(cols.values())
        if len(r) <= max(cols.values()) or not isinstance(r[first - 1], str):
            continue
        if isinstance(r[0], str) and "/" in r[0]:
            unit = r[0].strip()
        values = [r[cols[y]] for y in TYNDP_YEARS]
        if all(isinstance(v, (int, float)) for v in values):
            table[r[first - 1].strip()] = (unit, dict(zip(TYNDP_YEARS, (float(v) for v in values))))
    for label, want in ((TYNDP_GAS_ROW, "net GJ"), (TYNDP_OIL_ROW, "net GJ"), (TYNDP_CO2_ROW, "ton")):
        if label not in table:
            fail("%s, sheet %s: no row '%s'" % (name, FUEL_SHEET, label))
        if want not in (table[label][0] or ""):
            fail("%s, row '%s': unit '%s', expected a price per %s" % (name, label, table[label][0], want))
    return table


def text_table(path):
    """A csv as text: byte order mark, end of line, header and rows. Rows that are not rewritten stay as read."""
    raw = Path(path).read_bytes()
    bom = raw[:3] == b"\xef\xbb\xbf"
    text = raw[3:].decode("utf-8") if bom else raw.decode("utf-8")
    eol = "\r\n" if "\r\n" in text[:4096] else "\n"
    body = text.split(eol)
    if body[-1] != "":
        fail("%s: no end of line after the last row" % Path(path).name)
    return bom, eol, body[0], body[1:-1]


def text_payload(bom, eol, header, rows):
    """Bytes of a csv from the parts of text_table()."""
    return (b"\xef\xbb\xbf" if bom else b"") + (eol.join([header] + rows) + eol).encode("utf-8")


def num(value, decimals):
    text = "%.*f" % (decimals, value)
    if "." in text:                   # trailing zeros are dropped after the decimal point only
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def edit2_fuel_co2(check=False):
    log = ["Edit 2. Gas and CO2 of Romania, Bulgaria and Hungary on TYNDP 2026"]
    ok = True
    usd = config_value("price_deflators.csv", "eur_usd", 2024)
    tyndp = tyndp_fuel_table()
    gas_pts = {y: tyndp[TYNDP_GAS_ROW][1][y] * GJ_PER_MMBTU * usd * NCV_GCV_GAS for y in TYNDP_YEARS}
    co2_pts = {y: tyndp[TYNDP_CO2_ROW][1][y] * usd for y in TYNDP_YEARS}
    log.append("  eur_usd 2024 %.4f; %.6f GJ per MMBtu; net to gross %.2f on gas" % (usd, GJ_PER_MMBTU, NCV_GCV_GAS))
    log.append("  TYNDP 2026 natural gas, EUR 2024 per net GJ:   " + ", ".join(
        "%d %.3f" % (y, tyndp[TYNDP_GAS_ROW][1][y]) for y in TYNDP_YEARS))
    log.append("  gas line, USD per gross MMBtu:                 " + ", ".join("%d %.3f" % (y, gas_pts[y]) for y in TYNDP_YEARS))
    log.append("  TYNDP 2026 CO2, EUR 2024 per tonne:            " + ", ".join(
        "%d %.2f" % (y, tyndp[TYNDP_CO2_ROW][1][y]) for y in TYNDP_YEARS))
    log.append("  CO2, USD per tonne:                            " + ", ".join("%d %.2f" % (y, co2_pts[y]) for y in TYNDP_YEARS))

    cc = pd.read_csv(EPM / "epm" / "resources" / "pFuelCarbonContent.csv")
    carbon = {str(f).strip(): float(v) for f, v in zip(cc.iloc[:, 0], cc.iloc[:, 1])}

    def factor(fuel):
        """Tonnes of CO2 paid per MMBtu of the fuel."""
        if fuel == CCS_FUEL:
            return carbon["Gas"] * CCS_EMITTED
        if fuel in NO_CARBON:
            return 0.0
        if fuel not in carbon:
            fail("no carbon content for fuel %s" % fuel)
        return carbon[fuel]

    bom, eol, header, sum_rows = text_table(SUPPLY / "pFuelPrice_robg.csv")
    _, _, header_c, com_rows = text_table(SUPPLY / "pFuelPrice_tr_gas_flat.csv")
    if header != header_c:
        fail("pFuelPrice_robg.csv and pFuelPrice_tr_gas_flat.csv: different headers")
    head = header.split(",")
    if head[:2] != ["country", "fuel"]:
        fail("pFuelPrice_robg.csv: unexpected header")
    years = [int(c) for c in head[2:]]
    if years[0] > OBSERVED_YEARS[0] or years[-1] < TYNDP_YEARS[-1]:
        fail("pFuelPrice_robg.csv: years %d to %d" % (years[0], years[-1]))
    width = len(head)

    def keyed(rows, name):
        out = {}
        for line in rows:
            p = line.split(",")
            if len(p) != width:
                fail("%s: bad width in '%s'" % (name, line[:40]))
            if (p[0], p[1]) in out:
                fail("%s: duplicated row %s %s" % (name, p[0], p[1]))
            out[(p[0], p[1])] = p[2:]
        return out

    v5_sum, v5_com = keyed(sum_rows, "pFuelPrice_robg.csv"), keyed(com_rows, "pFuelPrice_tr_gas_flat.csv")
    if list(v5_sum) != list(v5_com):
        fail("the two v5 fuel files do not hold the same rows in the same order")
    for key in v5_sum:
        if key[0] not in FUEL_ZONES and v5_sum[key] != v5_com[key]:
            fail("row %s %s differs between the two v5 fuel files outside %s" % (key + (", ".join(FUEL_ZONES),)))
    if any(k[0] == FUEL_NEW_ZONE for k in v5_sum):
        fail("%s already has fuel rows in v5" % FUEL_NEW_ZONE)

    # v5 CO2 path, read back from the gas row, and checked on every row of the two countries
    ref = (FUEL_ZONES[0], "Gas")
    ets5 = {y: (float(s) - float(c)) / carbon["Gas"] for y, s, c in zip(years, v5_sum[ref], v5_com[ref])}
    worst = 0.0
    for (country, fuel), cells in v5_sum.items():
        if country in FUEL_ZONES:
            k = factor(fuel)
            for y, s, c in zip(years, cells, v5_com[(country, fuel)]):
                worst = max(worst, abs(float(s) - float(c) - ets5[y] * k))
    if worst > V5_SPLIT_TOL:
        fail("v5 fuel file is not commodity plus CO2 x carbon content: largest gap %.5f USD/MMBtu" % worst)
    log.append("  v5 file read back as commodity plus CO2: CO2 %.1f USD/t in 2025, %.1f in 2030, %.1f in 2040 "
               "(largest gap on a row %.5f USD/MMBtu)" % (ets5[2025], ets5[2030], ets5[2040], worst))

    def path(v5_value, v5_at_2025, points, year):
        if year <= BLEND_FROM:
            return v5_value
        if year < BLEND_TO:
            f = (year - BLEND_FROM) / (BLEND_TO - BLEND_FROM)
            return v5_at_2025 * (1 - f) + points[BLEND_TO] * f
        return blend(points, year)

    ets6 = {y: path(ets5[y], ets5[BLEND_FROM], co2_pts, y) for y in years}
    i25 = years.index(BLEND_FROM)

    built, lines, kept, moved = {}, {}, 0, 0
    for (country, fuel), cells in v5_sum.items():
        if country not in FUEL_ZONES:
            continue
        k = factor(fuel)
        com5 = [float(c) for c in v5_com[(country, fuel)]]
        on_tyndp = fuel in GAS_ON_TYNDP
        com_line, co2_line, out = [], [], []
        for i, y in enumerate(years):
            if y <= BLEND_FROM:
                com, co2, cell = com5[i], float(cells[i]) - com5[i], cells[i]
            else:
                com = round(path(com5[i], com5[i25], gas_pts, y) if on_tyndp else com5[i], LINE_DECIMALS)
                co2 = round(ets6[y] * k, LINE_DECIMALS)
                cell = num(com + co2, FUEL_DECIMALS)
            com_line.append(round(com, LINE_DECIMALS))
            co2_line.append(round(co2, LINE_DECIMALS))
            out.append(cell)
        if on_tyndp or k > 0:
            moved += 1
        else:
            out = cells              # no carbon and a commodity kept from v5: the v5 cells as written
            kept += 1
        built[(country, fuel)] = out
        lines[(country, fuel)] = (com_line, co2_line)

    out_rows = []
    for line in sum_rows:
        p = line.split(",")
        key = (p[0], p[1])
        out_rows.append(",".join(list(key) + built[key]) if key in built else line)
    for fuel in FUEL_NEW_ROWS:
        src = (FUEL_NEW_FROM, fuel)
        if src not in built:
            fail("no %s row for fuel %s" % src)
        built[(FUEL_NEW_ZONE, fuel)] = built[src]
        lines[(FUEL_NEW_ZONE, fuel)] = lines[src]
        out_rows.append(",".join([FUEL_NEW_ZONE, fuel] + built[src]))
    # Diesel of the Hungarian reserve gas turbines: TYNDP 2026 light oil plus the CO2 of the fuel
    oil_pts = {y: tyndp[TYNDP_OIL_ROW][1][y] * GJ_PER_MMBTU * usd * NCV_GCV_OIL for y in TYNDP_YEARS}
    oil_key = (FUEL_NEW_ZONE, OIL_FUEL)
    oil_com = [round(blend(oil_pts, y), LINE_DECIMALS) for y in years]
    oil_co2 = [round(ets6[y] * factor(OIL_FUEL), LINE_DECIMALS) for y in years]
    built[oil_key] = [num(a + b, FUEL_DECIMALS) for a, b in zip(oil_com, oil_co2)]
    lines[oil_key] = (oil_com, oil_co2)
    out_rows.append(",".join(list(oil_key) + built[oil_key]))
    log.append("  %s %s: TYNDP 2026 light oil x %.2f (net to gross), USD per gross MMBtu: %s; with CO2: %s" % (
        oil_key + (NCV_GCV_OIL, ", ".join("%d %.2f" % (y, oil_pts[y]) for y in TYNDP_YEARS),
                   ", ".join("%d %s" % (y, built[oil_key][years.index(y)]) for y in (2025, 2030, 2040)))))

    line_rows, gap = [], 0.0
    for (country, fuel), (com_line, co2_line) in lines.items():
        line_rows.append(",".join([country, fuel, "commodity"] + [num(v, LINE_DECIMALS) for v in com_line]))
        line_rows.append(",".join([country, fuel, "co2"] + [num(v, LINE_DECIMALS) for v in co2_line]))
        for a, b, cell in zip(com_line, co2_line, built[(country, fuel)]):
            gap = max(gap, abs(a + b - float(cell)))
    if gap > 0.5 * 10 ** -FUEL_DECIMALS + 1e-9:
        fail("commodity plus CO2 differs from the model file by %.6f USD/MMBtu" % gap)
    # edit 6: the Armenian gas row of the country team table (no CO2 line: the row is a plant gate price)
    out_rows, armenian = armenia_series("fuel", header, out_rows)
    log.append("  %s: gas row of the country team table, %d cells changed (edit 6)" % (AM_ZONE, len(armenian)))
    for row in out_rows:
        if len(row.split(",")) != width:
            fail("pFuelPrice_robg_v6.csv: bad width in '%s'" % row[:40])

    def encode(head_line, rows):
        return (b"\xef\xbb\xbf" if bom else b"") + (eol.join([head_line] + rows) + eol).encode("utf-8")

    others = sum(1 for k in v5_sum if k[0] not in FUEL_ZONES)
    log.append("  %d v5 lines kept as read (other countries); %s: %d rows kept (no carbon, commodity of v5), %d repriced; "
               "%d rows added (%s takes the rows of %s) and the %s row" % (
                   others, " and ".join(FUEL_ZONES), kept, moved, len(FUEL_NEW_ROWS), FUEL_NEW_ZONE, FUEL_NEW_FROM,
                   OIL_FUEL))
    ok &= emit(SUPPLY / "pFuelPrice_robg_v6.csv", encode(header, out_rows), check, log)
    ok &= emit(SUPPLY / "pFuelPrice_robg_v6_lines.csv",
               encode(",".join(["country", "fuel", "line"] + head[2:]), line_rows), check, log)
    log.append("  check: commodity line plus CO2 line against the model file, largest gap %.6f USD/MMBtu" % gap)

    # ── Traces ─────────────────────────────────────────────────────────────
    anchors = []
    for label, (unit, vals) in tyndp.items():
        rec = {"row": label, "unit": unit}
        rec.update({"eur_%d" % y: vals[y] for y in TYNDP_YEARS})
        per_gj = unit is not None and "GJ" in unit
        for y in TYNDP_YEARS:
            rec["usd_%d" % y] = vals[y] * usd * (GJ_PER_MMBTU if per_gj else 1.0)
        rec["usd_unit"] = "USD 2024 per net MMBtu" if per_gj else "USD 2024 per tonne"
        anchors.append(rec)
    ok &= emit(TRACE / "edit2_tyndp_fuel_rows.csv",
               pd.DataFrame(anchors).round(4).to_csv(index=False, lineterminator="\n").encode("utf-8"), check, log)
    gas5 = dict(zip(years, (float(c) for c in v5_com[ref])))
    gas6 = dict(zip(years, lines[ref][0]))
    paths = pd.DataFrame({"year": years,
                          "co2_usd_t_v5": [ets5[y] for y in years], "co2_usd_t_v6": [ets6[y] for y in years],
                          "gas_usd_mmbtu_v5": [gas5[y] for y in years], "gas_usd_mmbtu_v6": [gas6[y] for y in years]})
    for v in ("v5", "v6"):
        paths["gas_with_co2_" + v] = paths["gas_usd_mmbtu_" + v] + paths["co2_usd_t_" + v] * carbon["Gas"]
    ok &= emit(TRACE / "edit2_gas_co2_paths.csv", paths.round(4).to_csv(index=False, lineterminator="\n").encode("utf-8"),
               check, log)

    show = [y for y in (2025, 2028, 2030, 2032, 2035, 2036, 2040, 2050) if y in years]
    log.append("  USD/MMBtu (CO2 in USD/t)      " + "".join("%9d" % y for y in show))
    for label, series in (("gas line v5", gas5), ("gas line v6", gas6), ("CO2 v5", ets5), ("CO2 v6", ets6)):
        log.append("    %-28s" % label + "".join("%9.2f" % series[y] for y in show))
    for key in built:
        if key[0] != FUEL_NEW_ZONE and (key[1] in GAS_ON_TYNDP or factor(key[1]) > 0):
            old = dict(zip(years, (float(c) for c in v5_sum[key])))
            new = dict(zip(years, (float(c) for c in built[key])))
            log.append("    %-28s" % ("%s %s v5" % key) + "".join("%9.2f" % old[y] for y in show))
            log.append("    %-28s" % ("%s %s v6" % key) + "".join("%9.2f" % new[y] for y in show))
    # the commodity of the other fuels stays on v5; the TYNDP 2026 rows of the same fuels, for the record
    to_usd = GJ_PER_MMBTU * usd
    for label, key in (("Lignite G3", ("Romania", "DomesticCoal")), ("Lignite G1", ("Bulgaria", "DomesticCoal")),
                       ("Hard coal", ("Bulgaria", "ImportedCoal")), ("Nuclear", ("Romania", "Uranium"))):
        if label in tyndp and key in v5_com:
            old = dict(zip(years, (float(c) for c in v5_com[key])))
            log.append("    commodity kept from v5, %s %s: %.2f in 2030 and %.2f in 2040; TYNDP 2026 '%s': %.2f and %.2f" % (
                key + (old[2030], old[2040], label, tyndp[label][1][2030] * to_usd, tyndp[label][1][2040] * to_usd)))
    return ok, log


# ── Edit 3: border limits ───────────────────────────────────────────────────
def tyndp_package(folder, name):
    """Path and sha256 of a file of the TYNDP folder, checked against manifest.csv."""
    man = pd.read_csv(TYNDP_DATA / "manifest.csv", dtype=str)
    hit = man[(man["folder"] == folder) & (man["file"] == name)]
    if len(hit) != 1:
        fail("manifest.csv: %d rows for %s/%s" % (len(hit), folder, name))
    sha = hit["sha256"].iloc[0]
    if sha256_of(TYNDP_DATA / folder / name) != sha:
        fail("%s: sha256 differs from manifest.csv" % name)
    return TYNDP_DATA / folder / name, sha


def tyndp_ntc(metric=NTC_METRIC):
    """NTC of the TYNDP 2026 grid in the tidy table: {(node_from, node_to): {year: MW}} for one metric."""
    folder, name = NTC_PACKAGE
    _, sha = tyndp_package(folder, name)
    ntc = pd.read_csv(NTC_TABLE, dtype=str, keep_default_na=False)
    ntc = ntc[(ntc["edition"] == NTC_EDITION) & (ntc["scenario"] == NTC_SCENARIO) & (ntc["metric"] == metric)]
    if ntc.empty:
        fail("%s: no %s rows for TYNDP %s %s" % (NTC_TABLE.name, metric, NTC_EDITION, NTC_SCENARIO))
    if set(ntc["source_package"]) != {folder + "/" + name} or set(ntc["source_sha256"]) != {sha}:
        fail("%s: built from another package than %s/%s of manifest.csv" % (NTC_TABLE.name, folder, name))
    if set(ntc["unit"]) != {"MW"}:
        fail("%s: unit other than MW" % NTC_TABLE.name)
    if ntc.duplicated(["year", "node_from", "node_to"]).any():
        fail("%s: several %s rows for one direction and year" % (NTC_TABLE.name, metric))
    table = {}
    for r in ntc.itertuples(index=False):
        table.setdefault((r.node_from, r.node_to), {})[int(r.year)] = float(r.value)
    return table


def tyndp_reference_grid(node):
    """Borders of `node` in the reference grid workbook of TYNDP 2026, MW per direction.

    Returns ({(node_from, node_to): MW} for the 2025 NTC, {(node_from, node_to): {year: MW}} for the reference
    grid). In a row "A-B" the direct column and Direction 1 run from A to B, the indirect column and Direction 2
    from B to A.
    """
    import io
    import zipfile

    import openpyxl

    path, _ = tyndp_package(*NTC_PACKAGE)
    with zipfile.ZipFile(path) as z:
        wb = openpyxl.load_workbook(io.BytesIO(z.read(GRID_MEMBER)), read_only=True, data_only=True)
    ntc25, grid = {}, {}
    for year in TYNDP_YEARS:
        sheet = GRID_SHEET % year
        if sheet not in wb.sheetnames:
            fail("%s: no sheet %s" % (GRID_MEMBER, sheet))
        rows = wb[sheet].iter_rows(values_only=True)
        if tuple(next(rows)[:5]) != GRID_HEADERS:
            fail("%s, sheet %s: unexpected headers" % (GRID_MEMBER, sheet))
        found = 0
        for r in rows:
            if not isinstance(r[0], str) or node not in r[0].strip().split("-"):
                continue
            ends = r[0].strip().split("-")
            if len(ends) != 2 or ends[0] == ends[1]:
                fail("%s, sheet %s: border '%s'" % (GRID_MEMBER, sheet, r[0]))
            cells = r[1:5]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or v != round(v) for v in cells):
                fail("%s, sheet %s, %s: cells %s" % (GRID_MEMBER, sheet, r[0], list(cells)))
            a, b = ends
            for pair, now, reference in (((a, b), cells[0], cells[2]), ((b, a), cells[1], cells[3])):
                if ntc25.setdefault(pair, float(now)) != float(now):
                    fail("%s: the 2025 NTC of %s to %s differs between sheets" % ((GRID_MEMBER,) + pair))
                if year in grid.setdefault(pair, {}):
                    fail("%s, sheet %s: two rows for %s to %s" % ((GRID_MEMBER, sheet) + pair))
                grid[pair][year] = float(reference)
            found += 1
        if not found:
            fail("%s, sheet %s: no border of %s" % (GRID_MEMBER, sheet, node))
    wb.close()
    return ntc25, grid


def limit_points(source, frm, to, ntc):
    """Horizons of the limit from zone `frm` to zone `to`: {year: MW}."""
    if source == "tdp":
        return dict(zip(TDP_YEARS, TDP_NTC[(frm, to)]))
    points = ntc.get((LIMIT_NODES[frm], LIMIT_NODES[to]))
    if points is None or set(points) != set(TYNDP_YEARS):
        fail("%s: %s to %s, horizons %s" % (NTC_TABLE.name, frm, to, sorted(points or [])))
    return points


def held(points, year):
    """A value holds until the next horizon."""
    return points[max(y for y in points if y <= year)]


def edit3_border_limits(check=False):
    log = ["Edit 3. Border limits of Romania and Bulgaria on the TSO plan and the TYNDP 2026 reference grid"]
    ok = True
    for i, total in enumerate(TDP_TOTAL):
        for side in (0, 1):
            if sum(v[i] for k, v in TDP_NTC.items() if k[side] == "Romania") != total:
                fail("Table 5.4.2: the %d borders do not add up to the total row" % TDP_YEARS[i])
    ntc = tyndp_ntc()
    trace = []

    def rebuilt(path, n_key, rules, direction_of, cell):
        """Rows of `path` with the limits of `rules` from LIMIT_FROM. Other rows and earlier cells stay as read."""
        bom, eol, header, rows = text_table(path)
        years = [int(c) for c in header.split(",")[n_key:]]
        if LIMIT_FROM not in years or years != sorted(years):
            fail("%s: unexpected year columns" % path.name)
        out, seen = [], {}
        for line in rows:
            p = line.split(",")
            pair = (p[0], p[1])
            if pair not in rules:
                out.append(line)
                continue
            if len(p) != n_key + len(years):
                fail("%s: row %s has %d cells" % (path.name, ",".join(p[:n_key]), len(p)))
            frm, to, label = direction_of(p)
            points = limit_points(rules[pair], frm, to, ntc)
            cells = []
            for y, old in zip(years, p[n_key:]):
                if y < LIMIT_FROM:
                    cells.append(old)
                    continue
                value = held(points, y)
                if value != round(value):
                    fail("%s to %s, %d: %s MW is not a whole number" % (frm, to, y, value))
                cells.append(cell(value))
            out.append(",".join(p[:n_key] + cells))
            series = seen.setdefault((frm, to, label), {})
            if series and series["old"] != p[n_key:]:
                fail("%s: %s to %s, the quarters of v5 differ" % (path.name, frm, to))
            series.update(old=p[n_key:], new=cells, quarters=series.get("quarters", ()) + (p[2],), source=rules[pair],
                          pair=pair, points=points)
        expected = {direction_of(list(pair) + ["", d])[:2] for pair in rules for d in ("Import", "Export")} \
            if n_key == 4 else set(rules)
        if {k[:2] for k in seen} != expected:
            fail("%s: borders found %s, expected %s" % (path.name, sorted(k[:2] for k in seen), sorted(expected)))
        for (frm, to, label), s in seen.items():
            if s["quarters"] != QUARTERS:
                fail("%s: %s to %s, quarters %s" % (path.name, frm, to, ",".join(s["quarters"])))
            for y, old, new in zip(years, s["old"], s["new"]):
                trace.append({"file": path.name[:-4] + "_v6.csv", "zone": s["pair"][0], "neighbour": s["pair"][1],
                              "direction": label, "from": frm, "to": to, "year": y, "v5_mw": float(old),
                              "v6_mw": float(new), "basis": "v5 cell" if y < LIMIT_FROM else "%s, %d horizon" % (
                                  LIMIT_SOURCE[s["source"]], max(h for h in s["points"] if h <= y))})
        return (bom, eol, header, out), seen, len(rows), years

    def ext_direction(p):       # z, zext, q, Import or Export
        if p[3] == "Export":
            return p[0], p[1], "Export"
        if p[3] == "Import":
            return p[1], p[0], "Import"
        fail("pExtTransferLimit: direction '%s'" % p[3])

    ext, ext_seen, ext_n, years = rebuilt(TRADE / "pExtTransferLimit_robg.csv", 4, EXT_LIMIT_RULES, ext_direction,
                                          lambda v: "%d" % round(v))
    internal, int_seen, int_n, years_int = rebuilt(TRADE / "pTransferLimit_robg_hvdc.csv", 3, INT_LIMIT_RULES,
                                                   lambda p: (p[0], p[1], "n/a"), lambda v: "%.1f" % v)
    # Hungary is a modelled zone (edit 4): the Romania to Hungary rows become an internal line
    if years != years_int:
        fail("the external and the internal limit tables do not carry the same years")
    ext_bom, ext_eol, ext_header, ext_rows = ext
    int_bom, int_eol, int_header, int_rows = internal
    moved, ext_out = {}, []
    for line in ext_rows:
        p = line.split(",")
        if HU_ZONE not in p[:2]:
            ext_out.append(line)
            continue
        if p[:2] != ["Romania", HU_ZONE] or (p[2], p[3]) in moved:
            fail("pExtTransferLimit_robg.csv: unexpected row %s" % ",".join(p[:4]))
        moved[(p[2], p[3])] = p[4:]
    if set(moved) != {(q, d) for q in QUARTERS for d in ("Import", "Export")}:
        fail("pExtTransferLimit_robg.csv: Romania,%s rows %s" % (HU_ZONE, sorted(moved)))
    int_out = list(int_rows)
    for frm, to, direction in (("Romania", HU_ZONE, "Export"), (HU_ZONE, "Romania", "Import")):
        if any(r.startswith("%s,%s," % (frm, to)) for r in int_rows):
            fail("pTransferLimit_robg_hvdc.csv already holds %s,%s" % (frm, to))
        int_out += [",".join([frm, to, q] + ["%.1f" % float(v) for v in moved[(q, direction)]]) for q in QUARTERS]
    for row in trace:
        if row["file"].startswith("pExtTransferLimit") and HU_ZONE in (row["zone"], row["neighbour"]):
            row.update(file="pTransferLimit_robg_hvdc_v6.csv", zone=row["from"], neighbour=row["to"], direction="n/a")

    # the six other Hungarian borders enter the external table from the reference grid workbook
    ntc25, grid = tyndp_reference_grid(HU_NODE)
    tidy25 = tyndp_ntc(NTC_2025_METRIC)
    parity = 0
    for pair, points in grid.items():
        if set(points) != set(TYNDP_YEARS):
            fail("%s: %s to %s, horizons %s" % ((GRID_MEMBER,) + pair + (sorted(points),)))
        if pair in ntc:
            if ntc[pair] != points:
                fail("%s to %s: the workbook reads %s and %s reads %s" % (pair + (points, NTC_TABLE.name, ntc[pair])))
            parity += 1
        if pair in tidy25:
            if set(tidy25[pair].values()) != {ntc25[pair]}:
                fail("%s to %s, 2025 NTC: the workbook reads %s and %s reads %s" % (
                    pair + (ntc25[pair], NTC_TABLE.name, tidy25[pair])))
            parity += 1
    if not parity:
        fail("no border of %s shared by the workbook and %s: the reading of the workbook is not checked" % (
            HU_NODE, NTC_TABLE.name))
    hu_seen = {}
    for name, node in HU_BORDERS.items():
        for direction in ("Import", "Export"):
            pair = (node, HU_NODE) if direction == "Import" else (HU_NODE, node)
            if pair not in grid:
                fail("%s: no border %s to %s" % ((GRID_MEMBER,) + pair))
            frm, to = (name, HU_ZONE) if direction == "Import" else (HU_ZONE, name)
            values = [ntc25[pair] if y < LIMIT_FROM else held(grid[pair], y) for y in years]
            hu_seen[(name, direction)] = (frm, to, ["%d" % round(v) for v in values])
            for y in years:
                trace.append({"file": "pExtTransferLimit_robg_v6.csv", "zone": HU_ZONE, "neighbour": name,
                              "direction": direction, "from": frm, "to": to, "year": y, "v5_mw": None,
                              "v6_mw": float(round(ntc25[pair] if y < LIMIT_FROM else held(grid[pair], y))),
                              "basis": "%s, %s" % (GRID_SOURCE, "2025 NTC" if y < LIMIT_FROM else "%d horizon" % max(
                                  h for h in grid[pair] if h <= y))})
        for q in QUARTERS:
            for direction in ("Import", "Export"):
                ext_out.append(",".join([HU_ZONE, name, q, direction] + hu_seen[(name, direction)][2]))

    ok &= emit(TRADE / "pExtTransferLimit_robg_v6.csv", text_payload(ext_bom, ext_eol, ext_header, ext_out), check, log)
    ok &= emit(TRADE / "pTransferLimit_robg_hvdc_v6.csv", text_payload(int_bom, int_eol, int_header, int_out), check, log)
    cols = ["file", "zone", "neighbour", "direction", "from", "to", "year", "v5_mw", "v6_mw", "basis"]
    ok &= emit(TRACE / "edit3_border_limits.csv",
               pd.DataFrame(trace, columns=cols).to_csv(index=False, lineterminator="\n").encode("utf-8"), check, log)

    log.append("  rows rewritten: %d of %d (external), %d of %d (internal)" % (
        4 * len(ext_seen), ext_n, 4 * len(int_seen), int_n))
    log.append("  %s is a modelled zone: the %d rows Romania,%s leave the external table and enter the internal one "
               "as two directions" % (HU_ZONE, len(moved), HU_ZONE))
    log.append("  %d rows added to the external table: %s with %s; 2025 NTC of the workbook until %d, reference grid "
               "from %d; %d series equal to %s" % (8 * len(HU_BORDERS), HU_ZONE, ", ".join(HU_BORDERS), LIMIT_FROM - 1,
                                                   LIMIT_FROM, parity, NTC_TABLE.name))
    show = [y for y in (2025, 2029, 2030, 2033, 2035, 2040, 2050) if y in years]
    log.append("  MW, v5 then v6                " + "".join("%8d" % y for y in show))
    for seen, yrs in ((ext_seen, years), (int_seen, years_int)):
        for (frm, to, label), s in seen.items():
            name = "%s to %s" % (frm, to)
            for tag, cells in (("v5", s["old"]), ("v6", s["new"])):
                series = dict(zip(yrs, cells))
                log.append("    %-28s" % (name + " " + tag) + "".join("%8d" % round(float(series[y])) for y in show))
    for frm, to, cells in hu_seen.values():
        series = dict(zip(years, cells))
        log.append("    %-28s" % ("%s to %s v6" % (frm, to)) + "".join("%8d" % int(series[y]) for y in show))
    return ok, log


# ── Edit 4: Hungary as a modelled zone ──────────────────────────────────────
LOAD = DATA / "load"
RESERVE = DATA / "reserve"
GRID_TOKENS = ("s5", "2y")        # the fleet and storage files exist once per year grid
GRID_YEARS = {"s5": "y_s5.csv", "2y": "y_2y.csv"}
HU_FROM = "Romania"               # zone whose default rows, reserve margin, loss factor and trade share Hungary takes
HU_ISO3 = "HUN"
HU_MANIFEST = ENTSOE_CACHE / "fleet" / "hungary_v6_manifest.csv"      # written by fetch_hungary_v6.py
HU_ATLAS = "EPM/pre-analysis/data/gwa/HUN_capacity-factor_IEC2.tif"   # manifest key; a key under EPM/ starts at its parent
EMBER = TYNDP_DATA / "analysis" / "external" / "ember" / "yearly_full_release_long_format.csv"
EMBER_SHA256 = "259e1095ee8ffeaf0aff37ad557916ae1823a2da13312da50ba4cec6b4574c3b"
EMBER_FUELS = ("Bioenergy", "Coal", "Gas", "Hydro", "Nuclear", "Other Fossil", "Solar", "Wind")
HU_SHAPE_YEAR, HU_BASE_YEAR = 2023, 2025      # year of the hourly shapes (as the other zones), year of the levels
HU_EXT_NEW = ("Austria", "Slovakia", "Croatia", "Slovenia")
HU_FLOW_CODES = {"Austria": "AT", "Slovakia": "SK", "Croatia": "HR", "Slovenia": "SI", "Serbia": "RS",
                 "Ukraine": "UA", "Romania": "RO"}
DEMAND_TABLE = PRE / "output_tyndp" / "demand_annual.csv"             # built by build_tyndp_tidy.py
DEMAND_PACKAGE = ("2026/inputs", "Demand.zip")                        # folder and file, as in manifest.csv
SPIKE_QUANTILE, SPIKE_FACTOR = 0.999, 1.10    # an hour above 1.10 x the 0.999 quantile is a metering spike
WIND_PERCENTILE = 0.30            # share of the best atlas pixels that sets the wind level, as for the other zones
NUCLEAR_FLOOR = 0.75              # below it in a quarter the observed year holds an outage: annual level, flat
HU_COAL_OUT, HU_RETIRE = 2030, 2060
HU_RESERVE_COST = "5"             # as the Romanian units
HU_DEFAULT_CELLS = ("Life", "HeatRate", "RampUpRate", "RampDnRate", "ResLimShare", "Capex", "FOMperMW", "VOM")
HU_PRICED_FUELS = FUEL_NEW_ROWS + (OIL_FUEL,)                         # written by edit 2
HU_FREE_FUELS = ("Solar", "Wind", "Water", "Battery")
HU_AVAILABILITY_EXTRA = {("OCGT", "Diesel"): ("OCGT", "Gas")}         # pair missing in the default table: copied from

# Existing units. Capacity from the ENTSO-E register of production units (code), unless typed. A row without typed
# MW takes what is left of its code. First year of operation from the plant operators; the retirement year is a
# working assumption (HU_COAL_OUT for coal and lignite, HU_RETIRE for the rest).
# name, register code, register type, technology, fuel, first year, typed MW
HU_UNITS = (
    ("Hungary_Dunamenti_G3", "15WDUME------PPI", "Fossil Gas", "CCGT", "Gas", 2011, 407.0),
    ("Hungary_Dunamenti_G2", "15WDUME------PPI", "Fossil Gas", "CCGT", "Gas", 1996, None),
    ("Hungary_Gonyu", "15WGONYU-----PP5", "Fossil Gas", "CCGT", "Gas", 2011, None),
    ("Hungary_Csepel", "15WCSEPEL----PPI", "Fossil Gas", "CCGT", "Gas", 2000, None),
    ("Hungary_Kelenfold", "15WKELENF----PP8", "Fossil Gas", "CCGT", "Gas", 1996, None),
    ("Hungary_Kispest", "15WKISP------PP6", "Fossil Gas", "CCGT", "Gas", 2004, None),
    ("Hungary_Ujpest", "15WUJPEST----PPO", "Fossil Gas", "CCGT", "Gas", 2002, None),
    ("Hungary_Debrecen", "15WDKCE------PPQ", "Fossil Gas", "CCGT", "Gas", 2000, None),
    ("Hungary_Nyiregyhaza", "15WNYKCE-----PP6", "Fossil Gas", "CCGT", "Gas", 2007, None),
    ("Hungary_Bakonyi_GT", "15WBAKONY----PP6", "Fossil Gas", "OCGT", "Gas", 2011, None),
    ("Hungary_Matra", "15WMATRA-----PPK", "Fossil Brown coal/Lignite", "ST", "DomesticCoal", 1972, None),
    ("Hungary_Ajka", "15WAJKA------PPT", "Fossil Brown coal/Lignite", "ST", "DomesticCoal", 1961, None),
    ("Hungary_Hamburger_Hungaria", "15WHAMBURGERHPPF", "Fossil Hard coal", "ST", "DomesticCoal", 2015, None),
    ("Hungary_OROEBIO", "15W-OROEBIO-UZ-J", "Biomass", "BiomassPlant", "Biomass", 2004, None),
    ("Hungary_Pecs_Bio", "15WPECS------PPO", "Biomass", "BiomassPlant", "Biomass", 2004, None),
    ("Hungary_Pannongreen", "15WPANNON----PPS", "Biomass", "BiomassPlant", "Biomass", 2004, None),
    ("Hungary_Bakonyi_Bio", "15WBAKONYI---PPL", "Biomass", "BiomassPlant", "Biomass", 2004, None),
    ("Hungary_Lorinci", "15WLORINC----PPF", "Fossil Oil", "OCGT", "Diesel", 2000, None),
    ("Hungary_Liter", "15WLITER-----PPP", "Fossil Oil", "OCGT", "Diesel", 1998, None),
    ("Hungary_Sajoszoged", "15WSAJSZOGED-PP4", "Fossil Oil", "OCGT", "Diesel", 1998, None),
)
# Paks: the Ember nuclear capacity in four equal units. name, first year, retirement year (two 20 year extensions)
HU_PAKS = (("Hungary_Paks_1", 1982, 2052), ("Hungary_Paks_2", 1984, 2054), ("Hungary_Paks_3", 1986, 2056),
           ("Hungary_Paks_4", 1987, 2057))
# Aggregates: the Ember capacity of the fuel minus the named units of the same model fuel.
# name, technology, fuel, Ember fuel, first year
HU_AGGREGATES = (
    ("Hungary_AGG_Gas_CHP", "OCGT", "Gas", "Gas", 2005),
    ("Hungary_AGG_Biomass", "BiomassPlant", "Biomass", "Bioenergy", 2010),
    ("Hungary_AGG_Hydro", "ROR", "Water", "Hydro", 1975),
    ("Hungary_AGG_PV", "PV", "Solar", "Solar", 2024),
    ("Hungary_AGG_Wind", "OnshoreWind", "Wind", "Wind", 2010),
)
HU_HYDRO, HU_WIND, HU_PV = "Hungary_AGG_Hydro", "Hungary_AGG_Wind", "Hungary_AGG_PV"
HU_HYDRO_COLUMNS = ("Hydro Run-of-river and poundage", "Hydro Water Reservoir")
# Paks II, committed. Capex: EUR 12.5 billion for the two units (2014 agreement), EUR to USD as the prices.
# The other cells are the ones of the Romanian committed unit PAKS2_TEMPLATE, availability included.
PAKS2 = ("Hungary_Paks_5", "Hungary_Paks_6")
PAKS2_MW, PAKS2_START, PAKS2_END = 1200.0, 2036, 2096
PAKS2_CAPEX_MEUR, PAKS2_TEMPLATE = 12500.0, "Cernavoda-3"
# Candidates. name, Romanian row whose cells are copied (None: the default table), technology, fuel, first year,
# retirement year, capacity cap (MW). The build limit per year comes from build_limits().
HU_CANDIDATES = (
    ("Generic_Solar_Hungary", "Generic Solar Fixed", "PV", "Solar", 2026, 2076, 50000.0),
    ("Generic_Wind_Hungary", "Generic Onshore WindLow1", "OnshoreWind", "Wind", 2028, 2088, 10000.0),
    ("Hungary_Generic_Battery_Storage", "Generic Battery Storage", "Storage", "Battery", 2026, 2066, 50000.0),
    ("Generic_Biomass_Hungary", "Generic Biomass", "BiomassPlant", "Biomass", 2031, 2071, 500.0),
    ("Generic_CCGT_Hungary", None, "CCGT", "Gas", 2030, 2090, 10000.0),
    ("Generic_OCGT_Hungary", None, "OCGT", "Gas", 2030, 2090, 10000.0),
)
HU_STORAGE, HU_WIND_CANDIDATE = "Hungary_Generic_Battery_Storage", "Generic_Wind_Hungary"
FIRM_TECHS = ("Nuclear", "CCGT", "OCGT", "ST", "BiomassPlant", "ROR")

# Build limit per year of the candidates, rule v2 of the model (2026-08-30). For a technology:
#   b0     = max(observed yearly addition, sigma x peak 2025)
#   limit  = mean over BL_YEARS of min(b0 x gamma^(year - 2025), max(cap, b0))
#   cap    = E x annual energy / (capacity factor x 8760) for wind and solar, S x peak otherwise
# and the result is shared between the candidate rows of the technology in the country.
BL_BASE, BL_YEARS, BL_SINCE = 2025, tuple(range(2026, 2041)), 2010
BL_PAR = {"PV": dict(gamma=1.20, sigma=0.020, E=0.030), "OnshoreWind": dict(gamma=1.20, sigma=0.020, E=0.030),
          "OffshoreWind": dict(gamma=1.20, sigma=0.010, E=0.020), "Storage": dict(gamma=1.20, sigma=0.020, S=0.050),
          "CCGT": dict(gamma=1.10, sigma=0.010, S=0.080), "OCGT": dict(gamma=1.10, sigma=0.010, S=0.080),
          "BiomassPlant": dict(gamma=1.05, sigma=0.005, S=0.020)}
# Guard: the rule as written here must return the deployed limits of these rows, from the files they were built on.
BL_GUARD_ZONE = "Bulgaria"
BL_GUARD_FILES = ("load/pDemandForecast.csv", "supply/pGenDataInput.csv", "supply/pStorageDataInput.csv",
                  "supply/pVREProfile.csv")
BL_GUARD_OBSERVED = {"PV": 1416.0, "OnshoreWind": 0.0}
BL_GUARD_ROWS = {"PV": "Generic_Solar_Bulgaria", "OnshoreWind": "Generic_Wind_Bulgaria",
                 "Storage": "Bulgaria_Generic_Battery_Storage"}


def hu_cache(rel):
    """Path of a file of the Hungarian cache, checked against the sha256 of its manifest."""
    man = pd.read_csv(HU_MANIFEST, dtype=str)
    hit = man[man["file"] == rel]
    if len(hit) != 1:
        fail("%s: %d rows for %s" % (HU_MANIFEST.name, len(hit), rel))
    path = EPM.parent / rel if rel.startswith("EPM/") else ENTSOE_CACHE / rel
    if not path.exists():
        fail("%s is missing: run fetch_hungary_v6.py first" % rel)
    if sha256_of(path) != hit["sha256"].iloc[0]:
        fail("%s: sha256 differs from %s" % (rel, HU_MANIFEST.name))
    return path


def ember_reader(iso3):
    """Reader of the Ember yearly release for one country: value(year, category, subcategory, variable, unit)."""
    if sha256_of(EMBER) != EMBER_SHA256:
        fail("%s: sha256 differs from the release pinned in EMBER_SHA256" % EMBER.name)
    keys = ["Year", "Category", "Subcategory", "Variable", "Unit"]
    e = pd.read_csv(EMBER, usecols=["ISO 3 code", "Value", "YoY absolute change"] + keys, low_memory=False)
    e = e[e["ISO 3 code"] == iso3]

    def value(year, category, subcategory, variable, unit, column="Value"):
        hit = e[(e["Year"] == year) & (e["Category"] == category) & (e["Subcategory"] == subcategory)
                & (e["Variable"] == variable) & (e["Unit"] == unit)]
        if len(hit) != 1 or pd.isna(hit[column].iloc[0]):
            fail("%s: %d rows for %s %d %s, %s (%s)" % (EMBER.name, len(hit), iso3, year, variable, category, unit))
        return float(hit[column].iloc[0])

    return value


def utc_hourly(path, column, year):
    """One column of an ENTSO-E cache file as the 8,760 hourly means of the UTC year, and the hours filled.

    The cache holds the local year, so the last UTC hour is missing: it repeats the hour before.
    """
    df = pd.read_csv(path, index_col=0)
    if column not in df.columns:
        fail("%s: no column %s" % (Path(path).name, column))
    s = pd.Series(df[column].to_numpy(float), index=pd.to_datetime(df.index, utc=True)).resample("h").mean()
    s = s.reindex(pd.date_range("%d-01-01" % year, periods=8760, freq="h", tz="UTC"))
    gaps = int(s.isna().sum())
    if gaps > 2:
        fail("%s, %s: %d hours of %d without a value" % (Path(path).name, column, gaps, year))
    return s.ffill().bfill(), gaps


def despiked(s):
    """Hours above SPIKE_FACTOR x the SPIKE_QUANTILE quantile, replaced by a time interpolation."""
    flagged = s > SPIKE_FACTOR * float(s.quantile(SPIKE_QUANTILE))
    return s.mask(flagged).interpolate(method="time").ffill().bfill(), int(flagged.sum())


def ninja_profile(kind):
    """Renewables Ninja capacity factors of Hungary for HU_SHAPE_YEAR: 8,760 hourly values, UTC."""
    df = pd.read_csv(hu_cache("ninja_hungary/vre_rninja_hungary_%s.csv" % kind))
    col = str(HU_SHAPE_YEAR)
    idx = pd.date_range("%d-01-01" % HU_SHAPE_YEAR, periods=8760, freq="h")
    if list(df.columns) != ["zone", "month", "day", "hour", col] or len(df) != len(idx):
        fail("Renewables Ninja %s file: unexpected layout" % kind)
    if set(df["zone"]) != {HU_ZONE} or not ((df["month"].to_numpy() == idx.month) & (df["day"].to_numpy() == idx.day)
                                            & (df["hour"].to_numpy() == idx.hour)).all():
        fail("Renewables Ninja %s file: the rows are not the hours of %s in order" % (kind, col))
    values = df[col].to_numpy(float)
    if np.isnan(values).any() or values.min() < 0 or values.max() > 1:
        fail("Renewables Ninja %s file: values outside 0 to 1" % kind)
    return values


def atlas_wind_target():
    """Wind capacity factor of Hungary in the Global Wind Atlas raster (IEC class 2), best WIND_PERCENTILE pixels.

    Same function as the other zones (vre_cf_anchoring.zone_cf_targets). The raster is the country download of
    the atlas, without a value outside Hungary, so the zone is its bounding box.
    """
    import geopandas as gpd
    import rasterio
    from shapely.geometry import box

    sys.path.insert(0, str(HERE.parent))
    from vre_cf_anchoring import zone_cf_targets  # noqa: E402

    tif = hu_cache(HU_ATLAS)
    with rasterio.open(tif) as src:
        b = src.bounds
    zones = gpd.GeoDataFrame({"zone_name": [HU_ZONE], "ISO_A3": [HU_ISO3]},
                             geometry=[box(b.left, b.bottom, b.right, b.top)], crs="EPSG:4326")
    row = zone_cf_targets(zones, {HU_ISO3: tif}, percentile=WIND_PERCENTILE, log=lambda *_: None).iloc[0]
    return float(row["cf_target"]), int(row["n_pixels"])


def tyndp_demand_ratios(node):
    """Annual demand of a node in TYNDP 2026 (mean of the weather years, sum of the components), TWh and
    ratio to the first target year: ({year: TWh}, {year: ratio})."""
    folder, name = DEMAND_PACKAGE
    _, sha = tyndp_package(folder, name)
    d = pd.read_csv(DEMAND_TABLE, dtype=str, keep_default_na=False)
    d = d[(d["edition"] == NTC_EDITION) & (d["scenario"] == NTC_SCENARIO) & (d["node"] == node)
          & (d["metric"] == "annual_energy")]
    if d.empty:
        fail("%s: no annual energy of %s in TYNDP %s %s" % (DEMAND_TABLE.name, node, NTC_EDITION, NTC_SCENARIO))
    if set(d["source_package"]) != {folder + "/" + name} or set(d["source_sha256"]) != {sha}:
        fail("%s: built from another package than %s/%s of manifest.csv" % (DEMAND_TABLE.name, folder, name))
    if set(zip(d["family"], d["level"], d["unit"])) != {("demand", "total", "TWh")}:
        fail("%s: rows of %s other than total demand in TWh" % (DEMAND_TABLE.name, node))
    d = d.assign(value=d["value"].astype(float), year=d["year"].astype(int))
    twh = d.groupby(["year", "technology"])["value"].mean().groupby("year").sum().to_dict()
    if set(twh) != set(TYNDP_YEARS):
        fail("%s: years %s for %s" % (DEMAND_TABLE.name, sorted(twh), node))
    return twh, {y: twh[y] / twh[TYNDP_YEARS[0]] for y in TYNDP_YEARS}


def quarterly_availability(level, shape, hours, floor=0.0):
    """Availability per quarter: hour weighted mean `level`, proportions of `shape`, capped at 1.

    What a quarter would carry above 1 goes to the other quarters in proportion to their energy. With a quarter
    under `floor`, the annual level is returned flat. Returns (values, True when the shape is kept).
    """
    shape, hours = np.asarray(shape, float), np.asarray(hours, float)
    if not 0 < level <= 1 or (shape <= 0).any():
        fail("availability: level %s, shape %s" % (level, shape.tolist()))
    a = level * shape / ((shape * hours).sum() / hours.sum())
    for _ in range(len(a)):
        over = a > 1
        if not over.any():
            break
        free = a < 1
        if not free.any():
            fail("availability: no quarter left under 1")
        extra = ((a[over] - 1) * hours[over]).sum()
        a[over] = 1.0
        a[free] *= 1 + extra / (a[free] * hours[free]).sum()
    if a.max() > 1 + 1e-9:
        fail("availability: a quarter stays above 1")
    if a.min() < floor:
        return np.full(len(a), float(level)), False
    return np.minimum(a, 1.0), True


def dict_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def profile_cf(path, zone):
    """Hour weighted mean of the profiles of a zone in a pVREProfile file: {technology: capacity factor}."""
    prof = pd.read_csv(path)
    hours = pd.read_csv(DATA / "pHours.csv").set_index(["q", "d"])[HOURS]
    out = {}
    for tech, k in prof[prof["zone"] == zone].groupby("tech"):
        k = k.set_index(["season", "daytype"])[HOURS]
        w = hours.loc[k.index].to_numpy()
        out[tech] = float((k.to_numpy() * w).sum() / w.sum())
    return out


def build_limits(peak, energy, fleet, cf, observed):
    """Build limit per year and per candidate row (MW), by technology: the v2 rule, see BL_PAR.

    peak (MW) and energy (GWh) by year; fleet: rows of the country as {column: text}, a unit of the storage file
    read from it only; cf: capacity factor of the profile by technology; observed: yearly addition (MW) by
    technology, where it is known. Without it, the best year since BL_SINCE in the fleet (operating and committed
    units, aggregates left out). A candidate is a Status 3 row with a generic name.
    """
    def status(r):
        return (r["Status"] or "").strip().rstrip(".0")

    def number(v):
        try:
            return float((v or "").strip())
        except ValueError:
            return None

    by = {}
    for r in fleet:
        if status(r) in ("1", "2") and "agg" not in r["g"].lower():
            start = number(r["StYr"])
            if start is not None and start >= BL_SINCE:
                key = (r["tech"], int(start))
                by[key] = by.get(key, 0.0) + (number(r["Capacity"]) or 0.0)
    best = {}
    for (tech, _), v in by.items():
        best[tech] = max(best.get(tech, 0.0), v)
    rows = {}
    for r in fleet:
        n = r["g"].lower()
        generic = not any(k in n for k in ("psh", "php")) and ("generic" in n or "_cand" in n or "agg" in n)
        if status(r) == "3" and generic:
            rows.setdefault(r["tech"], []).append(r["g"])
    out = {}
    for tech, names in rows.items():
        if tech not in BL_PAR:
            continue
        p = BL_PAR[tech]
        seed = p["sigma"] * peak[BL_BASE]
        obs = observed.get(tech, best.get(tech, 0.0))
        b0 = max(obs, seed)
        if "E" in p and not cf.get(tech, 0.0) > 0:
            fail("build limit of %s: no profile to turn the energy share into capacity" % tech)
        limits = []
        for y in BL_YEARS:
            cap = p["E"] * energy[y] * 1000.0 / (cf[tech] * 8760.0) if "E" in p else p["S"] * peak[y]
            limits.append(min(b0 * p["gamma"] ** (y - BL_BASE), max(cap, b0)))
        mean = sum(limits) / len(limits)
        out[tech] = {"rows": names, "text": "%g" % round(mean / len(names), 1), "mean": mean, "observed": obs,
                     "seed": seed}
    return out


def build_limit_guard():
    """The rule returns the deployed limits of the Bulgarian candidates. Returns the texts compared."""
    demand, gen, sto, prof = (DATA / rel for rel in BL_GUARD_FILES)
    years = (BL_BASE,) + BL_YEARS
    peak, energy = {y: 0.0 for y in years}, {y: 0.0 for y in years}
    for r in dict_rows(demand):
        if r["z"] == BL_GUARD_ZONE:
            target = peak if r["type"].strip() == "Peak" else energy
            for y in years:
                target[y] += float(r[str(y)] or 0)
    sto_rows = dict_rows(sto)
    names = {r["g"] for r in sto_rows}
    fleet = [r for r in dict_rows(gen) if r["g"] not in names and r["z"] == BL_GUARD_ZONE]
    fleet += [r for r in sto_rows if r["z"] == BL_GUARD_ZONE]
    got = build_limits(peak, energy, fleet, profile_cf(prof, BL_GUARD_ZONE), BL_GUARD_OBSERVED)
    deployed = {r["g"]: r["BuildLimitperYear"] for r in fleet}
    seen = {}
    for tech, name in BL_GUARD_ROWS.items():
        if tech not in got or got[tech]["rows"] != [name] or got[tech]["text"] != deployed[name]:
            fail("build limit rule: %s gives %s, the deployed file holds %s" % (
                name, got.get(tech, {}).get("text"), deployed.get(name)))
        seen[name] = deployed[name]
    return seen


def hungary_fleet(templates, defaults, register, ember_gw, usd):
    """Rows of the Hungarian fleet as {column: text}; the candidates come without their build limit.

    templates: Romanian rows of the fleet file by name; defaults: Romanian rows of the default table by
    (technology, fuel); register: MW and production type by ENTSO-E unit code; ember_gw: capacity by Ember fuel.
    """
    rows = []

    def default_cells(tech, fuel):
        if (tech, fuel) not in defaults:
            fail("pGenDataInputDefault.csv: no %s row for %s, %s" % (HU_FROM, tech, fuel))
        return {c: defaults[(tech, fuel)][c] for c in HU_DEFAULT_CELLS}

    def template(name, tech, fuel, status):
        t = templates.get(name)
        if t is None or (t["tech"], t["f"], t["Status"]) != (tech, fuel, status):
            fail("fleet file: no %s row '%s' with %s, %s, Status %s" % (HU_FROM, name, tech, fuel, status))
        return dict(t)

    def existing(name, tech, fuel, start, end, mw):
        if not mw > 0:
            fail("%s: capacity %s" % (name, mw))
        cells = {"g": name, "z": HU_ZONE, "tech": tech, "f": fuel, "Status": "1", "StYr": str(start),
                 "RetrYr": str(end), "Capacity": num(mw, 1), "BuildLimitperYear": num(mw, 1),
                 "ReserveCost": HU_RESERVE_COST}
        cells.update(default_cells(tech, fuel))
        rows.append(cells)

    for name, start, end in HU_PAKS:
        existing(name, "Nuclear", "Uranium", start, end, ember_gw["Nuclear"] * 1000.0 / len(HU_PAKS))
    used, named = {}, {}
    for name, code, kind, tech, fuel, start, typed in HU_UNITS:
        if code not in register or register[code][1] != kind:
            fail("ENTSO-E unit register: no %s unit %s" % (kind, code))
        mw = typed if typed is not None else register[code][0] - used.get(code, 0.0)
        used[code] = used.get(code, 0.0) + mw
        named[fuel] = named.get(fuel, 0.0) + mw
        existing(name, tech, fuel, start, HU_COAL_OUT if fuel == "DomesticCoal" else HU_RETIRE, mw)
    for code, mw in used.items():
        if abs(mw - register[code][0]) > 1e-6:
            fail("ENTSO-E unit %s: %.1f MW in the rows, %.1f in the register" % (code, mw, register[code][0]))
    for name, tech, fuel, ember_fuel, start in HU_AGGREGATES:
        existing(name, tech, fuel, start, HU_RETIRE, ember_gw[ember_fuel] * 1000.0 - named.get(fuel, 0.0))

    for name in PAKS2:
        cells = template(PAKS2_TEMPLATE, "Nuclear", "Uranium", "2")
        cells.update({"g": name, "z": HU_ZONE, "StYr": str(PAKS2_START), "RetrYr": str(PAKS2_END),
                      "Capacity": num(PAKS2_MW, 1), "BuildLimitperYear": num(PAKS2_MW, 1),
                      "UnitSize": num(PAKS2_MW, 1),
                      "Capex": num(PAKS2_CAPEX_MEUR / (len(PAKS2) * PAKS2_MW) * usd, 4)})
        rows.append(cells)
    for name, source, tech, fuel, start, end, cap in HU_CANDIDATES:
        if source is None:
            cells = dict(default_cells(tech, fuel), tech=tech, f=fuel, ReserveCost=HU_RESERVE_COST)
        else:
            cells = template(source, tech, fuel, "3")
        cells.update({"g": name, "z": HU_ZONE, "Status": "3", "StYr": str(start), "RetrYr": str(end),
                      "Capacity": num(cap, 1), "BuildLimitperYear": ""})
        rows.append(cells)
    if len({r["g"] for r in rows}) != len(rows):
        fail("Hungarian fleet: a name is used twice")
    return rows


# ── Edit 5. Bulgarian fleet ──
BG_ZONE = "Bulgaria"
BG_FLEET_V5 = "pGenDataInput_v4_%s.csv"          # v5 sources of the fleet and storage files, per year grid
BG_STORAGE_V5 = "pStorageDataInput_v5_%s.csv"
BG_FLEET_V6 = "pGenDataInput_v6_%s.csv"
BG_STORAGE_V6 = "pStorageDataInput_v6_%s.csv"
BG_COAL_RETIRE = 2038                            # BG1: plants and mines supported to 2038
BG_COAL_BLOCKS = (("Bulgaria_Agg_Lignite_2029", "2029"), ("Bulgaria_Agg_Coal_2029", "2029"),
                  ("Bulgaria_Agg_Lignite_2033", "2033"), ("Bulgaria_Agg_Coal_2033", "2033"))   # unit, v5 RetrYr
BG_COAL_REFERENCE = ("Bulgaria_Agg_Lignite", "Bulgaria_Agg_Coal")     # v5 blocks that already leave in BG_COAL_RETIRE
BG_BATTERY = "Bulgaria_Battery"                  # BG2: operating fleet
BG_BATTERY_FOLDED = "Bulgaria_Battery_2026"      # committed block of v5, folded into the operating fleet
BG_BATTERY_MW = 6200.0
BG_BATTERY_V5 = {BG_BATTERY: ("1", "1382", "3593.2"), BG_BATTERY_FOLDED: ("2", "3118", "10462.4")}
#                unit: Status, Capacity and CapacityMWh in the v5 storage file
BG_BATTERY_V5_FLEET = {BG_BATTERY: "1382", BG_BATTERY_FOLDED: "4024"}   # Capacity in the v5 fleet file
BG_NUCLEAR_NEW = "Bulgaria_Kozloduy_8"           # BG3: committed unit
BG_NUCLEAR_NEW_V5, BG_NUCLEAR_NEW_YEAR = "2036", 2037     # StYr of v5, year of operation in the national plan
BG_NUCLEAR_LIFE = (("Bulgaria_Kozloduy_5", "2053", 2047), ("Bulgaria_Kozloduy_6", "2048", 2051))   # BG4
BG_BASIS = {
    "BG1": "country team: plants and mines supported to 2038; national plan: coal out by 2038 at the latest (p210, p211, "
           "p337). The name of the block keeps its v5 year",
    "BG2": "country team: 6.2 GW of batteries in service. The committed block of v5 joins the operating fleet",
    "BG2e": "energy per MW of the v5 fleet (operating plus committed block); to confirm on the operator list",
    "BG3": "country team: units 7 and 8 in operation between 2035 and 2037; national plan: 2035 and 2040 (p90). Unit 8 "
           "takes the first year of the grid from 2037: a committed unit off the grid is never built",
    "BG4": "national plan: the unit can operate until this year (p202); same year from the country team. No effect "
           "before 2040",
}
# national plan = Bulgaria, integrated energy and climate plan 2021 to 2030, final update submitted in 2025
# (Data/NDPs, English version, 366 pages); pages are those of the PDF


def bulgaria_fleet(token, g_header, g_rows, s_header, s_rows):
    """Edit 5 on the rows of the v5 fleet and storage files of one year grid.

    Returns (fleet rows, storage rows, records). A record is (grid, file, unit, field, v5, v6, basis). Every
    cell is checked against the v5 value it is expected to hold, so a change of the v5 files stops the build.
    The rows that are not named here stay as read."""
    grid_years = [int(v) for v in text_table(DATA / GRID_YEARS[token])[3]]
    g_cols, s_cols = g_header.split(","), s_header.split(",")
    names = {"fleet": (BG_FLEET_V5 % token, BG_FLEET_V6 % token), "storage": (BG_STORAGE_V5 % token, BG_STORAGE_V6 % token)}
    cols = {"fleet": g_cols, "storage": s_cols}
    parts = {"fleet": [r.split(",") for r in g_rows], "storage": [r.split(",") for r in s_rows]}
    for kind in parts:
        if any(len(p) != len(cols[kind]) for p in parts[kind]):
            fail("%s: a row does not have %d cells" % (names[kind][0], len(cols[kind])))
    records = []

    def row(kind, unit):
        hits = [p for p in parts[kind] if p[0] == unit]
        if len(hits) != 1 or hits[0][cols[kind].index("z")] != BG_ZONE:
            fail("%s: %d row(s) for %s in %s, expected one" % (names[kind][0], len(hits), unit, BG_ZONE))
        return hits[0]

    def cell(kind, unit, field):
        return row(kind, unit)[cols[kind].index(field)]

    def put(kind, unit, field, expected, value, basis):
        target = row(kind, unit)
        k = cols[kind].index(field)
        if target[k] != expected:
            fail("%s, %s: %s is '%s', expected '%s'" % (names[kind][0], unit, field, target[k], expected))
        target[k] = value
        records.append((token, names[kind][1], unit, field, expected, value, BG_BASIS[basis]))

    def drop(kind, unit, described):
        target = row(kind, unit)
        parts[kind] = [p for p in parts[kind] if p is not target]
        records.append((token, names[kind][1], unit, "row", described, "removed", BG_BASIS["BG2"]))

    # BG1 coal and lignite: the four blocks leave with the two that already carry the year
    for unit in BG_COAL_REFERENCE:
        if cell("fleet", unit, "RetrYr") != str(BG_COAL_RETIRE):
            fail("%s: %s no longer leaves in %d" % (names["fleet"][0], unit, BG_COAL_RETIRE))
    for unit, v5 in BG_COAL_BLOCKS:
        put("fleet", unit, "RetrYr", v5, str(BG_COAL_RETIRE), "BG1")

    # BG2 batteries: one operating fleet, energy on the hours of the v5 fleet
    for unit, (status, power, energy) in BG_BATTERY_V5.items():
        found = tuple(cell("storage", unit, c) for c in ("Status", "Capacity", "CapacityMWh"))
        if found != (status, power, energy):
            fail("%s, %s: Status, Capacity and CapacityMWh are %s, expected %s"
                 % (names["storage"][0], unit, found, (status, power, energy)))
        if (cell("fleet", unit, "Status"), cell("fleet", unit, "Capacity")) != (status, BG_BATTERY_V5_FLEET[unit]):
            fail("%s, %s: unexpected Status or Capacity" % (names["fleet"][0], unit))
    power_v5 = sum(float(v[1]) for v in BG_BATTERY_V5.values())
    energy_v5 = sum(float(v[2]) for v in BG_BATTERY_V5.values())
    if BG_BATTERY_MW < power_v5:
        fail("Bulgarian batteries: %s MW is below the v5 fleet (%s MW)" % (num(BG_BATTERY_MW, 1), num(power_v5, 1)))
    energy = num(BG_BATTERY_MW * energy_v5 / power_v5, 1)
    put("fleet", BG_BATTERY, "Capacity", BG_BATTERY_V5_FLEET[BG_BATTERY], num(BG_BATTERY_MW, 1), "BG2")
    put("storage", BG_BATTERY, "Capacity", BG_BATTERY_V5[BG_BATTERY][1], num(BG_BATTERY_MW, 1), "BG2")
    put("storage", BG_BATTERY, "CapacityMWh", BG_BATTERY_V5[BG_BATTERY][2], energy, "BG2e")
    for kind in ("fleet", "storage"):
        described = "Status %s, StYr %s, %s MW" % tuple(cell(kind, BG_BATTERY_FOLDED, c) for c in ("Status", "StYr", "Capacity"))
        if kind == "storage":
            described += ", %s MWh" % cell(kind, BG_BATTERY_FOLDED, "CapacityMWh")
        drop(kind, BG_BATTERY_FOLDED, described)

    # BG3 Kozloduy 8: a committed unit is built only in a year of the grid
    later = [y for y in grid_years if y >= BG_NUCLEAR_NEW_YEAR]
    if not later:
        fail("%s: no year from %d on" % (GRID_YEARS[token], BG_NUCLEAR_NEW_YEAR))
    if cell("fleet", BG_NUCLEAR_NEW, "Status") != "2":
        fail("%s: %s is not a committed unit" % (names["fleet"][0], BG_NUCLEAR_NEW))
    put("fleet", BG_NUCLEAR_NEW, "StYr", BG_NUCLEAR_NEW_V5, str(later[0]), "BG3")

    # BG4 Kozloduy 5 and 6: retirement years
    for unit, v5, year in BG_NUCLEAR_LIFE:
        put("fleet", unit, "RetrYr", v5, str(year), "BG4")

    # no committed Bulgarian unit is left off the grid, in either file (a unit committed in the first year is built)
    for kind in parts:
        k = {c: cols[kind].index(c) for c in ("z", "Status", "StYr")}
        for p in parts[kind]:
            if p[k["z"]] == BG_ZONE and p[k["Status"]] == "2" and int(p[k["StYr"]]) not in grid_years:
                fail("%s, %s: committed in %s, not a year of %s" % (names[kind][1], p[0], p[k["StYr"]], GRID_YEARS[token]))
    return [",".join(p) for p in parts["fleet"]], [",".join(p) for p in parts["storage"]], records


# ── Edit 6. Armenia, table of the country team ──
# The workbook is Official Use Only and stays out of git. No figure of it is written in this file: the build reads
# the cells, and the trace of the edit sits with the inputs (DVC), not with the traces of the other edits.
AM_ZONE = "Armenia"
AM_BOOK = EPM.parent / "Data" / "Armenia" / "country_team" / "Arm.EPM data.xlsx"
AM_BOOK_SHA256 = "a9bfa3cedca0e0c4ffd12ceb4c4ba48370ff4d1d288c1f1c771907189fac3021"
AM_TRACE = DATA / "v6_edit6_armenia_trace.csv"
AM_SHEETS = {"demand": "pDemandForecast", "fleet": "pGenDataInput", "storage": "pStorageDataInput", "fuel": "pFuelPrice"}
AM_BLOCKS = ("Current", "Updated")               # each sheet: the rows the team received, then its own rows
AM_STATUS = {"existing": "1", "committed project": "2", "candidate project": "3"}   # first words of the comment column
AM_FUEL_LABEL = "Armenia plant gate"             # country label of the Updated block of the fuel sheet
AM_DEMAND_V5, AM_DEMAND_V6 = "pDemandForecast_v2.csv", "pDemandForecast_v6.csv"
AM_FUEL_V5, AM_FUEL_V6 = "pFuelPrice_robg.csv", "pFuelPrice_robg_v6.csv"
AM_CANDIDATE_FROM = 2027                         # a unit the table turns into a candidate is not in service today
# cells of the Current block that the model has moved since (start year put on the year grids in v4)
AM_CURRENT_DRIFT = {("fleet", "Armenia_SmallSolar_Com", "StYr")}
AM_KEPT = {"fleet": ("Armenia_CCGT_Cand",), "storage": ()}    # model rows the table does not hold: kept as in v5
# cells the Updated block leaves blank where the Current block holds a value: not a new value, the v5 cell is kept
AM_BLANK_KEPT = {("fleet", "Armenia_Hydro_Cand", "Capex")}
# What the table changes, names only. The build stops if the workbook gives anything else.
AM_SHAPE = {
    "fleet": {"removed": ("Armenia_Nuclear_Cand", "Armenia_SmallSolar_Com"),
              "added": {"Armenia_Nuclear": "Armenia_Nuclear_Cand"},          # new row: the row whose place it takes
              "cells": (("Armenia_Hrazdan_ST", "RetrYr"),
                        ("Armenia_SmallPV_Exist", "Capacity"), ("Armenia_SmallPV_Exist", "BuildLimitperYear"),
                        ("Armenia_Ayg1_PV", "Status"), ("Armenia_Ayg2_PV", "Status"),
                        ("Armenia_PV_Cand", "Capacity"), ("Armenia_PV_Cand", "BuildLimitperYear"),
                        ("Armenia_Wind_Cand", "Capacity"),
                        ("Armenia_Hydro_Cand", "Capacity"), ("Armenia_Hydro_Cand", "BuildLimitperYear"))},
    "storage": {"removed": (), "added": {},
                "cells": (("Armenia_BESS_Cand", "Capacity"), ("Armenia_BESS_Cand", "BuildLimitperYear"),
                          ("Armenia_BESS_Cand", "CapacityMWh"))},
}
AM_ITEM = {"Armenia_Nuclear": "AM2", "Armenia_Nuclear_Cand": "AM2", "Armenia_Hrazdan_ST": "AM3",
           "Armenia_PV_Cand": "AM4", "Armenia_Wind_Cand": "AM4", "Armenia_Hydro_Cand": "AM4",
           "Armenia_BESS_Cand": "AM5", "Armenia_SmallPV_Exist": "F12", "Armenia_Ayg1_PV": "F12",
           "Armenia_Ayg2_PV": "F12", "Armenia_SmallSolar_Com": "F12"}
AM_BASIS = {
    "AM1": "AM1, country team table: peak and energy of the Updated block",
    "AM2": "AM2, country team table: the new nuclear unit is a committed project and replaces the candidate. "
           "A modelling assumption, not a Government decision (flag F8)",
    "AM3": "AM3, country team table: the steam unit is decommissioned by this year. It has no capacity in the model",
    "AM4": "AM4, country team table: size and yearly build limit of the candidate",
    "AM5": "AM5, country team table: size, energy and yearly build limit of the battery candidate",
    "AM6": "AM6, country team table: gas price at the plant gate (flag F7)",
    "F12": "country team table, beyond the list AM1 to AM9: existing small solar fleet, the two Ayg plants as "
           "candidate projects, no committed small solar block (flag F12)",
    "F12y": "a unit the table turns into a candidate project is not in service today: it can be built from this "
            "year (flag F12)",
}


def am_text(value):
    """A cell of the workbook as the model files write it."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return num(float(value), 4)


def am_same(a, b):
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return False


@functools.lru_cache(maxsize=None)
def armenia_book():
    """The Current and Updated blocks of the four sheets used: {sheet: {block: [row]}}.

    A row is a dict of the named columns (a year header becomes its text; an unnamed first column is 'country'),
    plus '_comment', the text found right of the last named column."""
    if not AM_BOOK.exists():
        fail("%s is missing: table of the Armenia country team, Official Use Only, kept out of git" % AM_BOOK)
    if sha256_of(AM_BOOK) != AM_BOOK_SHA256:
        fail("%s is not the workbook this edit was written on (sha256 differs)" % AM_BOOK.name)
    import openpyxl
    book = openpyxl.load_workbook(AM_BOOK, data_only=True, read_only=True)
    out = {}
    for key, title in AM_SHEETS.items():
        if title not in book.sheetnames:
            fail("%s: no sheet %s" % (AM_BOOK.name, title))
        grid = [tuple(r) for r in book[title].iter_rows(values_only=True)]
        marks = {r[0].strip(): i for i, r in enumerate(grid) if r and isinstance(r[0], str) and r[0].strip() in AM_BLOCKS}
        if sorted(marks, key=marks.get) != list(AM_BLOCKS):
            fail("%s, sheet %s: expected the blocks %s, in this order" % (AM_BOOK.name, title, " and ".join(AM_BLOCKS)))
        out[key] = {}
        for block, start in marks.items():
            names = {}
            for i, h in enumerate(grid[start + 1]):
                if h is None:
                    if i == 0:
                        names[i] = "country"
                    continue
                names[i] = str(int(h)) if isinstance(h, (int, float)) else str(h).strip()
            if len(set(names.values())) != len(names):
                fail("%s, sheet %s, block %s: a column name comes twice" % (AM_BOOK.name, title, block))
            last, body = max(names), []
            for r in grid[start + 2:]:
                if all(v is None for v in r) or (isinstance(r[0], str) and r[0].strip() in AM_BLOCKS):
                    break
                rec = {name: (r[i] if i < len(r) else None) for i, name in names.items()}
                rec["_comment"] = " ".join(str(v).strip() for v in r[last + 1:] if v is not None)
                body.append(rec)
            if not body:
                fail("%s, sheet %s: the block %s is empty" % (AM_BOOK.name, title, block))
            out[key][block] = body
    book.close()
    return out


def armenia_fleet(token, g_header, g_rows, s_header, s_rows):
    """Edit 6 on the fleet and storage rows of one year grid (after edit 5).

    The Current block of each sheet must be the v5 rows of the model: it proves the team worked on this version.
    The difference between the Updated and the Current block is then applied, and it must have the shape of
    AM_SHAPE. Returns (fleet rows, storage rows, records); a record is (grid, file, key, field, v5, v6, basis)."""
    book = armenia_book()
    grid_years = [int(v) for v in text_table(DATA / GRID_YEARS[token])[3]]
    files = {"fleet": (BG_FLEET_V5 % token, BG_FLEET_V6 % token), "storage": (BG_STORAGE_V5 % token, BG_STORAGE_V6 % token)}
    cols = {"fleet": g_header.split(","), "storage": s_header.split(",")}
    parts = {"fleet": [r.split(",") for r in g_rows], "storage": [r.split(",") for r in s_rows]}
    records = []
    for kind in ("fleet", "storage"):
        head, (v5_name, v6_name), shape = cols[kind], files[kind], AM_SHAPE[kind]
        z = head.index("z")
        if any(len(p) != len(head) for p in parts[kind]):
            fail("%s: a row does not have %d cells" % (v5_name, len(head)))
        model = {p[0]: p for p in parts[kind] if p[z] == AM_ZONE}
        if len(model) != sum(1 for p in parts[kind] if p[z] == AM_ZONE):
            fail("%s: a unit of %s comes twice" % (v5_name, AM_ZONE))
        blocks = {}
        for block in AM_BLOCKS:
            blocks[block] = {am_text(r["g"]): r for r in book[kind][block]}
            if len(blocks[block]) != len(book[kind][block]):
                fail("%s, %s, block %s: a unit comes twice" % (AM_BOOK.name, AM_SHEETS[kind], block))
        current, updated = blocks["Current"], blocks["Updated"]
        sheet_cols = [c for c in book[kind]["Current"][0] if c not in ("g", "_comment")]
        if [c for c in book[kind]["Updated"][0] if c not in ("g", "_comment")] != sheet_cols:
            fail("%s, %s: the two blocks do not have the same columns" % (AM_BOOK.name, AM_SHEETS[kind]))
        if set(sheet_cols) - set(head):
            fail("%s, %s: columns unknown to %s: %s" % (AM_BOOK.name, AM_SHEETS[kind], v5_name,
                                                      sorted(set(sheet_cols) - set(head))))
        fields = [c for c in head if c in sheet_cols]

        # the Current block is the v5 file
        if set(model) - set(current) != set(AM_KEPT[kind]) or set(current) - set(model):
            fail("%s: the units of %s are not those of the Current block of the country team table" % (v5_name, AM_ZONE))
        for unit, rec in current.items():
            for f in fields:
                if (kind, unit, f) not in AM_CURRENT_DRIFT and not am_same(model[unit][head.index(f)], am_text(rec[f])):
                    fail("%s, %s: %s is '%s', the Current block of the country team table holds '%s'"
                         % (v5_name, unit, f, model[unit][head.index(f)], am_text(rec[f])))

        def status(rec):
            given, words = am_text(rec["Status"]), rec["_comment"].lower()
            named = [v for k, v in AM_STATUS.items() if words.startswith(k)]
            if len(named) > 1 or (named and given and given != named[0]) or not (named or given):
                fail("%s, %s, %s: no clear status" % (AM_BOOK.name, AM_SHEETS[kind], am_text(rec["g"])))
            return named[0] if named else given

        def new_cell(rec, f):
            return status(rec) if f == "Status" else am_text(rec[f])

        def described(p):
            text = "Status %s, StYr %s, %s MW" % tuple(p[head.index(c)] for c in ("Status", "StYr", "Capacity"))
            return text + (", %s MWh" % p[head.index("CapacityMWh")] if kind == "storage" else "")

        # the difference between the two blocks, and its shape
        removed = [u for u in current if u not in updated]
        added = [u for u in updated if u not in current]
        changed = [(u, f, new_cell(rec, f)) for u, rec in updated.items() if u in current for f in fields
                   if not am_same(am_text(current[u][f]), new_cell(rec, f))]
        blank = {(kind, u, f) for u, f, new in changed if new == "" and (kind, u, f) in AM_BLANK_KEPT}
        if blank != {c for c in AM_BLANK_KEPT if c[0] == kind}:
            fail("%s, %s: the cells left blank are not those edit 6 expects" % (AM_BOOK.name, AM_SHEETS[kind]))
        changed = [c for c in changed if (kind,) + c[:2] not in blank]
        if (sorted(removed) != sorted(shape["removed"]) or sorted(added) != sorted(shape["added"])
                or sorted((u, f) for u, f, _ in changed) != sorted(shape["cells"])):
            fail("%s, %s: the Updated block does not change what edit 6 expects. removed %s, added %s, cells %s"
                 % (AM_BOOK.name, AM_SHEETS[kind], removed, added, sorted((u, f) for u, f, _ in changed)))
        if any((kind, u, f) in AM_CURRENT_DRIFT for u, f, _ in changed):
            fail("%s: a cell that moved since the Current block is changed by the table" % v5_name)

        for unit, f, new in changed:
            k = head.index(f)
            records.append((token, v6_name, unit, f, model[unit][k], new, AM_BASIS[AM_ITEM[unit]]))
            model[unit][k] = new
        for unit, place in shape["added"].items():
            if place not in removed:
                fail("%s takes the place of %s, which the table does not remove" % (unit, place))
            row = [""] * len(head)
            row[0] = unit
            for f in fields:
                row[head.index(f)] = new_cell(updated[unit], f)
            if row[z] != AM_ZONE:
                fail("%s, %s: %s is not a unit of %s" % (AM_BOOK.name, AM_SHEETS[kind], unit, AM_ZONE))
            records.append((token, v6_name, unit, "row", "%s (%s)" % (place, described(model[place])), described(row),
                            AM_BASIS[AM_ITEM[unit]]))
            parts[kind][next(i for i, p in enumerate(parts[kind]) if p is model[place])] = row
            model[unit] = row
        for unit in removed:
            if unit not in shape["added"].values():
                records.append((token, v6_name, unit, "row", described(model[unit]), "removed", AM_BASIS[AM_ITEM[unit]]))
                parts[kind] = [p for p in parts[kind] if p is not model[unit]]
            del model[unit]

        # a unit turned into a candidate cannot be built in a year that is already behind
        k = {c: head.index(c) for c in ("Status", "StYr", "Capacity", "BuildLimitperYear")}
        for unit, f, new in changed:
            if f == "Status" and new == "3" and int(model[unit][k["StYr"]]) < AM_CANDIDATE_FROM:
                records.append((token, v6_name, unit, "StYr", model[unit][k["StYr"]], str(AM_CANDIDATE_FROM),
                                AM_BASIS["F12y"]))
                model[unit][k["StYr"]] = str(AM_CANDIDATE_FROM)

        # no committed unit off the grid, and no candidate without a size and a yearly build limit
        for unit, p in model.items():
            if p[k["Status"]] == "2" and int(p[k["StYr"]]) not in grid_years:
                fail("%s, %s: committed in %s, not a year of %s" % (v6_name, unit, p[k["StYr"]], GRID_YEARS[token]))
            if p[k["Status"]] == "3" and not (float(p[k["Capacity"]] or 0) > 0 and float(p[k["BuildLimitperYear"]] or 0) > 0):
                fail("%s, %s: a candidate without a size or a yearly build limit is never built" % (v6_name, unit))
    return [",".join(p) for p in parts["fleet"]], [",".join(p) for p in parts["storage"]], records


def armenia_series(kind, header, rows):
    """Edit 6 on the demand rows (kind 'demand') or the fuel price rows (kind 'fuel') of v5.

    Same rule as the fleet: the Current block must be the v5 rows, the Updated block replaces them. Returns
    (rows, records). In the fuel sheet the Updated block names the country AM_FUEL_LABEL; a fuel it does not hold
    keeps its v5 row, and gas is the only fuel it may change."""
    book = armenia_book()[kind]
    cols = header.split(",")
    years = cols[2:]
    if kind == "demand":
        v5_name, v6_name, key_cols, label, basis = AM_DEMAND_V5, AM_DEMAND_V6, ("z", "type"), AM_ZONE, "AM1"
        may_change = ("Peak", "Energy")

        def text(v):
            return "%.1f" % float(v)
    else:
        v5_name, v6_name, key_cols, label, basis = AM_FUEL_V5, AM_FUEL_V6, ("country", "fuel"), AM_FUEL_LABEL, "AM6"
        may_change = ("Gas",)

        def text(v):
            return num(float(v), FUEL_DECIMALS)
    if cols[:2] != [("country" if kind == "fuel" else "z"), key_cols[1]]:
        fail("%s: unexpected header" % v5_name)
    current = {am_text(r[key_cols[1]]): r for r in book["Current"] if am_text(r[key_cols[0]]) == AM_ZONE}
    updated = {am_text(r[key_cols[1]]): r for r in book["Updated"] if am_text(r[key_cols[0]]) == label}
    if len(current) != len(book["Current"]) or len(updated) != len(book["Updated"]) or set(updated) - set(current):
        fail("%s, %s: unexpected rows" % (AM_BOOK.name, AM_SHEETS[kind]))
    for rec in list(current.values()) + list(updated.values()):
        if any(rec.get(y) is None for y in years):
            fail("%s, %s: a year of %s is missing" % (AM_BOOK.name, AM_SHEETS[kind], v5_name))
    out, records, seen = [], [], set()
    for line in rows:
        p = line.split(",")
        if p[0] == AM_ZONE:
            if p[1] not in current or p[1] in seen or len(p) != len(cols):
                fail("%s: unexpected row %s %s" % (v5_name, p[0], p[1]))
            seen.add(p[1])
            if [text(current[p[1]][y]) for y in years] != p[2:]:
                fail("%s: the row %s %s is not the Current block of the country team table" % (v5_name, p[0], p[1]))
            if p[1] in updated:
                new = [text(updated[p[1]][y]) for y in years]
                if new != p[2:] and p[1] not in may_change:
                    fail("%s, %s: the table changes %s, which edit 6 does not expect" % (AM_BOOK.name, AM_SHEETS[kind], p[1]))
                records += [("all", v6_name, "%s %s" % (p[0], p[1]), y, a, b, AM_BASIS[basis])
                            for y, a, b in zip(years, p[2:], new) if a != b]
                line = ",".join(p[:2] + new)
        out.append(line)
    if seen != set(current) or not set(may_change) <= set(updated):
        fail("%s: the rows of %s are not those of the country team table" % (v5_name, AM_ZONE))
    return out, records


def v5_rows_after_edits(token):
    """The v5 fleet and storage rows of one year grid after edits 5 and 6, as edit 4 writes them before the
    Hungarian rows: (fleet header, fleet rows, storage header, storage rows, records of edit 5, records of edit 6)."""
    _, _, g_header, g_rows = text_table(SUPPLY / (BG_FLEET_V5 % token))
    _, _, s_header, s_rows = text_table(SUPPLY / (BG_STORAGE_V5 % token))
    g_rows, s_rows, bulgarian = bulgaria_fleet(token, g_header, g_rows, s_header, s_rows)
    g_rows, s_rows, armenian = armenia_fleet(token, g_header, g_rows, s_header, s_rows)
    return g_header, g_rows, s_header, s_rows, bulgarian, armenian


def fleet_files_hold(token, g_header, g_new, s_header, s_new, edit):
    """Control that the v6 fleet and storage files of one grid start with the rows given, and hold Hungarian rows
    only after them. Returns (ok, log lines)."""
    ok, log = True, []
    for name, header, new in ((BG_FLEET_V6 % token, g_header, g_new), (BG_STORAGE_V6 % token, s_header, s_new)):
        path = SUPPLY / name
        disk = text_table(path) if path.exists() else None
        if disk is None or disk[2] != header or disk[3][:len(new)] != new:
            ok = False
            log.append("  %s  %s: the rows of %s are not in the file, build edit 4 (it writes this file)"
                       % ("MISSING" if disk is None else "DIFFERS", name, edit))
            continue
        z = header.split(",").index("z")
        if any(r.split(",")[z] != HU_ZONE for r in disk[3][len(new):]):
            fail("%s: a row after the v5 block is not a %s row" % (name, HU_ZONE))
        log.append("  holds     %s (%d rows of v5 after edits 5 and 6, %d of %s)"
                   % (name, len(new), len(disk[3]) - len(new), HU_ZONE))
    return ok, log


def edit4_hungary(check=False):
    log = ["Edit 4. Hungary as a modelled zone"]
    ok, trace = True, []

    def note(item, value, unit, source):
        trace.append((item, value if isinstance(value, str) else num(value, 4), unit, source))

    def split(rows, width, name):
        parts = [r.split(",") for r in rows]
        if any(len(p) != width for p in parts):
            fail("%s: a row does not have %d cells" % (name, width))
        return parts

    def absent(rows, name, column=0):
        if any(r.split(",")[column] == HU_ZONE for r in rows):
            fail("%s already holds %s rows" % (name, HU_ZONE))

    usd = config_value("price_deflators.csv", "eur_usd", 2024)
    ember = ember_reader(HU_ISO3)
    src_ember = "Ember yearly electricity data, %s" % HU_ISO3
    src_entsoe = "ENTSO-E transparency platform, cache of fetch_hungary_v6.py"
    cap_gw = {f: ember(HU_BASE_YEAR, "Capacity", "Fuel", f, "GW") for f in EMBER_FUELS}
    gen_twh = {f: ember(HU_BASE_YEAR, "Electricity generation", "Fuel", f, "TWh") for f in EMBER_FUELS}
    note("Ember release, sha256", EMBER_SHA256, "", EMBER.name)
    for f in EMBER_FUELS:
        note("capacity %d, %s" % (HU_BASE_YEAR, f), cap_gw[f], "GW", src_ember)
        note("generation %d, %s" % (HU_BASE_YEAR, f), gen_twh[f], "TWh", src_ember)

    # ── H1 zone, H2 external zones, H8 reserve margin, losses and trade share ──
    for name in ("zcmap_robg", "zcmap_robg_geco"):
        bom, eol, header, rows = text_table(DATA / (name + ".csv"))
        absent(rows, name + ".csv")
        ok &= emit(DATA / (name + "_v6.csv"), text_payload(bom, eol, header, rows + ["%s,%s" % (HU_ZONE, HU_ZONE)]),
                   check, log)

    bom, eol, header, rows = text_table(TRADE / "zext_robg.csv")
    if HU_ZONE not in rows or set(HU_EXT_NEW) & set(rows):
        fail("zext_robg.csv: %s expected in the list, %s not" % (HU_ZONE, ", ".join(HU_EXT_NEW)))
    zext = [r for r in rows if r != HU_ZONE] + list(HU_EXT_NEW)
    ok &= emit(TRADE / "zext_robg_v6.csv", text_payload(bom, eol, header, zext), check, log)
    limits_file = TRADE / "pExtTransferLimit_robg_v6.csv"
    if not limits_file.exists():
        fail("%s is missing: run --edit 3 first" % limits_file.name)
    pairs = {tuple(r.split(",")[:2]) for r in text_table(limits_file)[3]}
    if {b for a, b in pairs if a == HU_ZONE} != set(HU_BORDERS) or any(b == HU_ZONE for _, b in pairs):
        fail("%s does not hold the Hungarian borders: run --edit 3 first" % limits_file.name)
    if not {b for _, b in pairs} <= set(zext):
        fail("%s: an external zone is not in zext_robg_v6.csv" % limits_file.name)
    for name in ("pTradePrice_robg_v6.csv", "pTradePriceExport_robg_v6.csv"):
        if not (TRADE / name).exists():
            fail("%s is missing: run --edit 1 first" % name)
        priced = {r.split(",", 1)[0] for r in text_table(TRADE / name)[3]}
        if set(zext) - priced:
            fail("%s: no price for %s" % (name, ", ".join(sorted(set(zext) - priced))))

    bom, eol, header, rows = text_table(RESERVE / "pPlanningReserveMargin_robg.csv")
    absent(rows, "pPlanningReserveMargin_robg.csv")
    margin = [r.split(",")[1] for r in rows if r.split(",")[0] == HU_FROM]
    if len(margin) != 1:
        fail("pPlanningReserveMargin_robg.csv: %d rows for %s" % (len(margin), HU_FROM))
    ok &= emit(RESERVE / "pPlanningReserveMargin_robg_v6.csv",
               text_payload(bom, eol, header, rows + ["%s,%s" % (HU_ZONE, margin[0])]), check, log)

    bom, eol, header, rows = text_table(TRADE / "pLossFactorInternal_robg_hvdc.csv")
    if any(HU_ZONE in r.split(",")[:2] for r in rows):
        fail("pLossFactorInternal_robg_hvdc.csv already holds %s rows" % HU_ZONE)
    loss = {tuple(r.split(",")[:2]): r.split(",")[2:] for r in rows}
    other = next((b for a, b in loss if a == HU_FROM), None)
    if other is None or (other, HU_FROM) not in loss:
        fail("pLossFactorInternal_robg_hvdc.csv: no line of %s in both directions" % HU_FROM)
    rows = rows + [",".join([HU_FROM, HU_ZONE] + loss[(HU_FROM, other)]),
                   ",".join([HU_ZONE, HU_FROM] + loss[(other, HU_FROM)])]
    ok &= emit(TRADE / "pLossFactorInternal_robg_hvdc_v6.csv", text_payload(bom, eol, header, rows), check, log)
    log.append("  loss factor of the %s to %s line: the cells of %s to %s (%s)" % (
        HU_FROM, HU_ZONE, HU_FROM, other, loss[(HU_FROM, other)][0]))

    bom, eol, header, rows = text_table(TRADE / "pMaxAnnualExternalTradeShare_robg.csv")
    cols = header.split(",")
    if HU_ZONE in cols or cols.count(HU_FROM) != 1:
        fail("pMaxAnnualExternalTradeShare_robg.csv: columns %s" % cols)
    at = cols.index(HU_FROM)
    rows = [",".join(p + [p[at]]) for p in split(rows, len(cols), "pMaxAnnualExternalTradeShare_robg.csv")]
    ok &= emit(TRADE / "pMaxAnnualExternalTradeShare_robg_v6.csv",
               text_payload(bom, eol, header + "," + HU_ZONE, rows), check, log)

    # ── H3 annual demand ──
    twh, ratios = tyndp_demand_ratios(HU_NODE)
    load_base, gaps = utc_hourly(hu_cache("fleet/load_HU_%d.csv" % HU_BASE_YEAR), "Actual Load", HU_BASE_YEAR)
    load_base, spikes = despiked(load_base)
    peak_base = float(load_base.max())
    bom, eol, header, rows = text_table(LOAD / "pDemandForecast_v2.csv")
    absent(rows, "pDemandForecast_v2.csv")
    cols = header.split(",")
    if cols[:2] != ["z", "type"]:
        fail("pDemandForecast_v2.csv: unexpected header")
    years = [int(c) for c in cols[2:]]
    rows, _ = armenia_series("demand", header, rows)      # edit 6: the Armenian rows of the country team table
    base_gwh = ember(HU_BASE_YEAR, "Electricity demand", "Demand", "Demand", "TWh") * 1000.0
    energy = {y: ember(y, "Electricity demand", "Demand", "Demand", "TWh") * 1000.0 if y < HU_BASE_YEAR
              else base_gwh * blend(ratios, y) for y in years}
    peak = {y: peak_base * energy[y] / base_gwh for y in years}
    rows = rows + [",".join([HU_ZONE, "Peak"] + ["%.1f" % peak[y] for y in years]),
                   ",".join([HU_ZONE, "Energy"] + ["%.1f" % energy[y] for y in years])]
    ok &= emit(LOAD / "pDemandForecast_v6.csv", text_payload(bom, eol, header, rows), check, log)
    note("demand %d" % HU_BASE_YEAR, base_gwh / 1000.0, "TWh", src_ember)
    note("load %d, sum of the hourly values" % HU_BASE_YEAR, float(load_base.sum()) / 1e6, "TWh", src_entsoe)
    note("load %d, highest hour (UTC hourly mean, %d spikes removed, %d hours filled)" % (HU_BASE_YEAR, spikes, gaps),
         peak_base, "MW", src_entsoe)
    for y in TYNDP_YEARS:
        note("TYNDP 2026 NT+ demand of %s, %d" % (HU_NODE, y), twh[y], "TWh", DEMAND_TABLE.name)
        note("TYNDP 2026 NT+ demand of %s, %d over %d" % (HU_NODE, y, TYNDP_YEARS[0]), ratios[y], "ratio",
             DEMAND_TABLE.name)
    show = [y for y in (2024, 2025, 2030, 2035, 2040, 2050) if y in years]
    log.append("  demand, TWh    " + "  ".join("%d %.2f" % (y, energy[y] / 1000.0) for y in show))
    log.append("  peak, MW       " + "  ".join("%d %.0f" % (y, peak[y]) for y in show))
    log.append("  energy: Ember %d, flat to %d, then the TYNDP 2026 NT+ path of %s; peak: ENTSO-E %d hourly maximum, "
               "moved with the energy (load factor %.3f)" % (HU_BASE_YEAR, TYNDP_YEARS[0], HU_NODE, HU_BASE_YEAR,
                                                             base_gwh * 1000.0 / (peak_base * 8760.0)))

    # ── Representative days ──
    sel, _ = repday_selection()
    pos = {k: g.sort_values("hod")["j"].to_numpy() for k, g in sel.groupby(["q", "d"])}
    wts = {k: g.sort_values("hod")["w"].to_numpy(float) for k, g in sel.groupby(["q", "d"])}
    bom, eol, header, rows = text_table(LOAD / "pDemandProfile.csv")
    absent(rows, "pDemandProfile.csv")
    if header.split(",") != ["zone", "season", "daytype"] + HOURS:
        fail("pDemandProfile.csv: unexpected header")
    order = [tuple(r.split(",")[1:3]) for r in rows if r.split(",")[0] == HU_FROM]
    if len(order) != len(pos) or set(order) != set(pos):
        fail("pDemandProfile.csv: the %s rows are not the representative days" % HU_FROM)
    weights = np.array([wts[k] for k in order])

    def days(series):
        """8,760 values of a UTC year as 28 x 24, in the row order of the profile files."""
        return np.array([np.asarray(series, float)[pos[k]] for k in order])

    def mean(table):
        return float((table * weights).sum() / weights.sum())

    # ── H4 hourly load: gross of the solar behind the meter (F11) ──
    load, gaps = utc_hourly(hu_cache("fleet/load_HU_%d.csv" % HU_SHAPE_YEAR), "Actual Load", HU_SHAPE_YEAR)
    load, spikes = despiked(load)
    metered, _ = utc_hourly(hu_cache("fleet/gen_HU_hourly_%d.csv" % HU_SHAPE_YEAR), "Solar", HU_SHAPE_YEAR)
    ninja = {kind: ninja_profile(kind) for kind in ("solar", "wind")}
    solar_shape_twh = ember(HU_SHAPE_YEAR, "Electricity generation", "Fuel", "Solar", "TWh")
    hidden = solar_shape_twh * 1e6 - float(metered.sum())
    if hidden < 0:
        fail("the ENTSO-E solar output of %d exceeds the Ember value" % HU_SHAPE_YEAR)
    gross = load.to_numpy() + hidden * ninja["solar"] / ninja["solar"].sum()
    top, when = float(gross.max()), load.index[int(gross.argmax())]
    table = days(gross / top)
    rows = rows + [",".join([HU_ZONE, q, d] + [repr(float(v)) for v in table[i]]) for i, (q, d) in enumerate(order)]
    ok &= emit(LOAD / "pDemandProfile_v6.csv", text_payload(bom, eol, header, rows), check, log)
    note("load %d, sum of the hourly values (%d spikes removed, %d hours filled)" % (HU_SHAPE_YEAR, spikes, gaps),
         float(load.sum()) / 1e6, "TWh", src_entsoe)
    note("solar %d" % HU_SHAPE_YEAR, solar_shape_twh, "TWh", src_ember)
    note("solar %d, metered" % HU_SHAPE_YEAR, float(metered.sum()) / 1e6, "TWh", src_entsoe)
    note("solar %d behind the meter, added to the load on the Renewables Ninja shape" % HU_SHAPE_YEAR, hidden / 1e6,
         "TWh", "computed")
    note("gross load %d, highest hour" % HU_SHAPE_YEAR, top, "MW", "computed")
    note("gross load %d, hour of the maximum (UTC)" % HU_SHAPE_YEAR, when.strftime("%Y-%m-%d %H:%M"), "", "computed")
    note("gross load %d, load factor of the year" % HU_SHAPE_YEAR, float(gross.mean()) / top, "ratio", "computed")
    note("demand profile, highest value of the representative days", float(table.max()), "ratio", "computed")
    note("demand profile, weighted mean of the representative days", mean(table), "ratio", "computed")
    log.append("  load shape %d: ENTSO-E load plus %.2f TWh of solar behind the meter; maximum %.0f MW on %s UTC; "
               "profile maximum %.4f, weighted mean %.4f" % (HU_SHAPE_YEAR, hidden / 1e6, top,
                                                             when.strftime("%Y-%m-%d %H:%M"), table.max(), mean(table)))

    # ── H5 solar and wind profiles ──
    wind_target, pixels = atlas_wind_target()
    solar_before = ember(HU_BASE_YEAR - 1, "Capacity", "Fuel", "Solar", "GW")
    solar_target = gen_twh["Solar"] / ((solar_before + cap_gw["Solar"]) / 2.0) / 8.76
    bom, eol, header, rows = text_table(SUPPLY / "pVREProfile.csv")
    absent(rows, "pVREProfile.csv")
    if header.split(",") != ["zone", "tech", "season", "daytype"] + HOURS:
        fail("pVREProfile.csv: unexpected header")
    achieved = {}
    for tech, kind, target, basis in (("PV", "solar", solar_target, "observed %d: Ember generation over the mean of "
                                       "the capacity at the end of %d and %d" % (HU_BASE_YEAR, HU_BASE_YEAR - 1,
                                                                                 HU_BASE_YEAR)),
                                      ("OnshoreWind", "wind", wind_target, "Global Wind Atlas, IEC class 2, best "
                                       "%d percent of %d pixels" % (round(100 * WIND_PERCENTILE), pixels))):
        if [tuple(r.split(",")[2:4]) for r in rows if r.split(",")[:2] == [HU_FROM, tech]] != order:
            fail("pVREProfile.csv: the %s %s rows are not in the order of the demand profile" % (HU_FROM, tech))
        raw = days(ninja[kind])
        factor = target / mean(raw)
        scaled = np.array([[round(float(v), 3) for v in day] for day in np.clip(raw * factor, None, 1.0)])
        achieved[tech] = mean(scaled)
        rows = rows + [",".join([HU_ZONE, tech, q, d] + [repr(float(v)) for v in scaled[i]])
                       for i, (q, d) in enumerate(order)]
        note("%s profile, target capacity factor (%s)" % (tech, basis), target, "ratio", "computed")
        note("%s profile, Renewables Ninja %d, mean of the year" % (tech, HU_SHAPE_YEAR), float(ninja[kind].mean()),
             "ratio", "Renewables Ninja")
        note("%s profile, Renewables Ninja %d, weighted mean of the representative days" % (tech, HU_SHAPE_YEAR),
             mean(raw), "ratio", "computed")
        note("%s profile, factor applied" % tech, factor, "ratio", "computed")
        note("%s profile, hours capped at 1" % tech, float((raw * factor > 1).sum()), "hours", "computed")
        note("%s profile, weighted mean written" % tech, achieved[tech], "ratio", "computed")
        log.append("  %-11s Renewables Ninja %d x %.4f: target %.4f, written %.4f (%s)" % (
            tech, HU_SHAPE_YEAR, factor, target, achieved[tech], basis))
    ok &= emit(SUPPLY / "pVREProfile_v6.csv", text_payload(bom, eol, header, rows), check, log)

    # ── H6 availability of the nuclear, hydro and wind units ──
    quarter_hours = pd.Series(1.0, index=load_base.index).groupby(load_base.index.quarter).sum().to_numpy()
    gen_file = hu_cache("fleet/gen_HU_hourly_%d.csv" % HU_BASE_YEAR)

    def quarterly(columns):
        total = sum(utc_hourly(gen_file, c, HU_BASE_YEAR)[0] for c in columns)
        return total.groupby(total.index.quarter).mean().to_numpy(), float(total.sum()) / 1e6

    shape, metered_twh = quarterly(["Nuclear"])
    level = gen_twh["Nuclear"] / (cap_gw["Nuclear"] * 8.76)
    nuclear, kept = quarterly_availability(level, shape, quarter_hours, NUCLEAR_FLOOR)
    note("nuclear output %d, metered" % HU_BASE_YEAR, metered_twh, "TWh", src_entsoe)
    note("nuclear availability, annual level (Ember generation over capacity)", level, "ratio", "computed")
    log.append("  nuclear availability %s, annual %.4f (%s)" % (
        " ".join("%.4f" % v for v in nuclear), level,
        "quarterly shape of the %d output" % HU_BASE_YEAR if kept else "flat: a quarter under %.2f" % NUCLEAR_FLOOR))
    shape, metered_twh = quarterly(HU_HYDRO_COLUMNS)
    level = gen_twh["Hydro"] / (cap_gw["Hydro"] * 8.76)
    hydro, _ = quarterly_availability(level, shape, quarter_hours)
    note("hydro output %d, metered" % HU_BASE_YEAR, metered_twh, "TWh", src_entsoe)
    note("hydro availability, annual level (Ember generation over capacity)", level, "ratio", "computed")
    log.append("  hydro availability   %s, annual %.4f" % (" ".join("%.4f" % v for v in hydro), level))
    wind_observed = gen_twh["Wind"] / (cap_gw["Wind"] * 8.76)
    wind_share = wind_observed / achieved["OnshoreWind"]
    if not 0 < wind_share <= 1:
        fail("wind availability %.4f: the observed capacity factor exceeds the profile" % wind_share)
    note("wind capacity factor %d, observed (Ember generation over capacity)" % HU_BASE_YEAR, wind_observed, "ratio",
         "computed")
    note("wind availability (observed capacity factor over the profile mean)", wind_share, "ratio", "computed")
    log.append("  wind availability    %.4f flat on the existing fleet and on the candidate, as the Romanian and "
               "Bulgarian wind rows: observed %d capacity factor %.4f over the profile mean %.4f" % (
                   wind_share, HU_BASE_YEAR, wind_observed, achieved["OnshoreWind"]))

    bom, eol, header, rows = text_table(SUPPLY / "pAvailabilityCustom.csv")
    if header.split(",") != ["gen"] + list(QUARTERS):
        fail("pAvailabilityCustom.csv: unexpected header")
    known = {r.split(",", 1)[0]: r.split(",", 1)[1] for r in rows}
    if PAKS2_TEMPLATE not in known:
        fail("pAvailabilityCustom.csv: no row for %s" % PAKS2_TEMPLATE)

    def quarters(values):
        return ",".join(repr(round(float(v), 4)) for v in values)

    availability = [(name, quarters(nuclear)) for name, _, _ in HU_PAKS]
    availability += [(name, known[PAKS2_TEMPLATE]) for name in PAKS2]
    availability += [(HU_HYDRO, quarters(hydro)), (HU_WIND, quarters([wind_share] * 4)),
                     (HU_WIND_CANDIDATE, quarters([wind_share] * 4))]
    if set(n for n, _ in availability) & set(known):
        fail("pAvailabilityCustom.csv already holds a Hungarian unit")
    ok &= emit(SUPPLY / "pAvailabilityCustom_v6.csv",
               text_payload(bom, eol, header, rows + ["%s,%s" % a for a in availability]), check, log)

    # ── Default tables: the Romanian rows as Hungarian rows ──
    default_pairs = {}
    for name in ("pGenDataInputDefault", "pAvailabilityDefault", "pCapexTrajectoriesDefault"):
        bom, eol, header, rows = text_table(SUPPLY / (name + ".csv"))
        absent(rows, name + ".csv")
        source = [r.split(",") for r in rows if r.split(",", 1)[0] == HU_FROM]
        new = [[HU_ZONE] + p[1:] for p in source]
        if name == "pAvailabilityDefault":
            for (tech, fuel), like in HU_AVAILABILITY_EXTRA.items():
                hit = [p for p in source if tuple(p[1:3]) == like]
                if len(hit) != 1 or any(tuple(p[1:3]) == (tech, fuel) for p in source):
                    fail("%s.csv: %s row for %s expected once, and none for %s, %s" % (name, HU_FROM, like, tech, fuel))
                new.append([HU_ZONE, tech, fuel] + hit[0][3:])
        default_pairs[name] = {tuple(p[1:3]) for p in new}
        if len(default_pairs[name]) != len(new):
            fail("%s.csv: a technology and fuel pair of %s appears twice" % (name, HU_FROM))
        ok &= emit(SUPPLY / (name + "_v6.csv"), text_payload(bom, eol, header, rows + [",".join(p) for p in new]),
                   check, log)
        log.append("  %s_v6.csv: %d rows of %s as %s%s" % (name, len(source), HU_FROM, HU_ZONE,
                                                         "" if len(new) == len(source) else ", %d added" % (len(new) - len(source))))
    bom, eol, header, rows = text_table(SUPPLY / "pGenDataInputDefault.csv")
    cols = header.split(",")
    defaults = {(p[1], p[2]): dict(zip(cols, p)) for p in split(rows, len(cols), "pGenDataInputDefault.csv")
                if p[0] == HU_FROM}

    # ── H6 and H7 fleet, committed units and candidates ──
    units = pd.read_csv(hu_cache("fleet/instcap_units_HU_%d.csv" % HU_BASE_YEAR), index_col=0)
    register = {code: (float(r["Installed Capacity [MW]"]), r["Production Type"]) for code, r in units.iterrows()}
    guard = build_limit_guard()
    cf = dict(achieved)
    observed = {"PV": 1000.0 * ember(HU_BASE_YEAR, "Capacity", "Fuel", "Solar", "GW", "YoY absolute change"),
                "OnshoreWind": 1000.0 * ember(HU_BASE_YEAR, "Capacity", "Fuel", "Wind", "GW", "YoY absolute change")}
    fleet_lines, storage_lines, fleet, limits = {}, {}, None, None
    for token in GRID_TOKENS:
        g_bom, g_eol, g_header, g_rows = text_table(SUPPLY / ("pGenDataInput_v4_%s.csv" % token))
        s_bom, s_eol, s_header, s_rows = text_table(SUPPLY / ("pStorageDataInput_v5_%s.csv" % token))
        absent(g_rows, "pGenDataInput_v4_%s.csv" % token, 1)
        absent(s_rows, "pStorageDataInput_v5_%s.csv" % token, 1)
        # edit 5 (Bulgarian fleet) is applied to the v5 rows in the same build: these two files are written here only
        g_rows, s_rows, _ = bulgaria_fleet(token, g_header, g_rows, s_header, s_rows)
        # and edit 6 (Armenia, table of the country team) after it
        g_rows, s_rows, _ = armenia_fleet(token, g_header, g_rows, s_header, s_rows)
        g_cols, s_cols = g_header.split(","), s_header.split(",")
        templates = {p[0]: dict(zip(g_cols, p)) for p in split(g_rows, len(g_cols), "pGenDataInput_v4_%s.csv" % token)
                     if p[1] == HU_FROM}
        fleet = hungary_fleet(templates, defaults, register, cap_gw, usd)
        stored = {p[0]: dict(zip(s_cols, p)) for p in split(s_rows, len(s_cols), "pStorageDataInput_v5_%s.csv" % token)
                  if p[1] == HU_FROM}
        like = next(c[1] for c in HU_CANDIDATES if c[0] == HU_STORAGE)
        mirror = next(r for r in fleet if r["g"] == HU_STORAGE)
        if like not in stored or (stored[like]["tech"], stored[like]["f"]) != (mirror["tech"], mirror["f"]):
            fail("pStorageDataInput_v5_%s.csv: no %s row '%s'" % (token, HU_FROM, like))
        storage = dict(stored[like])
        storage.update({k: mirror[k] for k in ("g", "z", "Status", "StYr", "RetrYr", "Capacity", "BuildLimitperYear")})
        # the rule reads a unit of the storage file from it only
        limits = build_limits({y: peak[y] for y in (BL_BASE,) + BL_YEARS}, {y: energy[y] for y in (BL_BASE,) + BL_YEARS},
                              [r for r in fleet if r["g"] != HU_STORAGE] + [storage], cf, observed)
        for r in fleet + [storage]:
            if r["Status"] == "3":
                if r["tech"] not in limits or r["g"] not in limits[r["tech"]]["rows"]:
                    fail("%s: no build limit from the rule" % r["g"])
                r["BuildLimitperYear"] = limits[r["tech"]]["text"]
        grid_years = [int(v) for v in text_table(DATA / GRID_YEARS[token])[3]]
        for r in fleet:
            if set(r) - set(g_cols):
                fail("fleet file: unknown column %s" % sorted(set(r) - set(g_cols)))
            if not r["BuildLimitperYear"] or float(r["BuildLimitperYear"]) <= 0:
                fail("%s: no build limit" % r["g"])
            if r["Status"] == "2" and (int(r["StYr"]) not in grid_years or int(r["StYr"]) <= grid_years[0]):
                fail("%s: StYr %s is not a later year of %s" % (r["g"], r["StYr"], GRID_YEARS[token]))
            if r["Status"] == "1" and not int(r["StYr"]) < grid_years[0]:
                fail("%s: an operating unit starts before %d" % (r["g"], grid_years[0]))
        fleet_lines[token] = [",".join(r.get(c, "") for c in g_cols) for r in fleet]
        storage_lines[token] = [",".join(storage.get(c, "") for c in s_cols)]
        ok &= emit(SUPPLY / ("pGenDataInput_v6_%s.csv" % token),
                   text_payload(g_bom, g_eol, g_header, g_rows + fleet_lines[token]), check, log)
        ok &= emit(SUPPLY / ("pStorageDataInput_v6_%s.csv" % token),
                   text_payload(s_bom, s_eol, s_header, s_rows + storage_lines[token]), check, log)
    if len({tuple(v) for v in fleet_lines.values()}) != 1 or len({tuple(v) for v in storage_lines.values()}) != 1:
        fail("the Hungarian rows differ between the year grids")

    # every unit finds its defaults, its technology and fuel pair and its fuel price
    pairs = {(r["tech"], r["f"]) for r in fleet}
    bom, eol, header, rows = text_table(SUPPLY / "pTechFuel_robg.csv")
    default_pairs["pTechFuel_robg"] = {tuple(r.split(",")[:2]) for r in rows}
    for name, have in default_pairs.items():
        if pairs - have:
            fail("%s: no row for %s" % (name, sorted(pairs - have)))
    fuel_file = SUPPLY / "pFuelPrice_robg_v6.csv"
    if not fuel_file.exists():
        fail("%s is missing: run --edit 2 first" % fuel_file.name)
    priced = {r.split(",")[1] for r in text_table(fuel_file)[3] if r.split(",")[0] == HU_ZONE}
    fuels = {f for _, f in pairs}
    if fuels - set(HU_PRICED_FUELS) - set(HU_FREE_FUELS) or (fuels & set(HU_PRICED_FUELS)) - priced:
        fail("%s: no %s price for a fuel of %s: run --edit 2 first" % (fuel_file.name, HU_ZONE, sorted(fuels)))

    by_tech = {}
    for r in fleet:
        key = (r["Status"], r["tech"], r["f"])
        by_tech[key] = by_tech.get(key, 0.0) + float(r["Capacity"])
    log.append("  fleet: %d rows, the same in both grid files" % len(fleet))
    for status, label in (("1", "operating"), ("2", "committed")):
        log.append("    %-10s " % label + "; ".join("%s %s %.1f MW" % (t, f, v) for (s, t, f), v in by_tech.items()
                                                    if s == status))
    for r in fleet:
        if r["Status"] == "1" and r["RetrYr"] == str(HU_COAL_OUT):
            note("%s, out from %d" % (r["g"], HU_COAL_OUT), float(r["Capacity"]), "MW", "ENTSO-E unit register")
    for name, tech, fuel, ember_fuel, _ in HU_AGGREGATES:
        row = next(r for r in fleet if r["g"] == name)
        note("%s (Ember %s minus the named units)" % (name, ember_fuel), float(row["Capacity"]), "MW", "computed")
    coal = sum(float(r["Capacity"]) for r in fleet if r["f"] == "DomesticCoal")
    oil = sum(float(r["Capacity"]) for r in fleet if r["f"] == OIL_FUEL)
    note("coal and lignite units of the register", coal, "MW", "ENTSO-E unit register")
    note("coal and lignite, Ember minus the register units", cap_gw["Coal"] * 1000.0 - coal, "MW", "computed")
    note("oil units of the register (Ember other fossil %s GW)" % num(cap_gw["Other Fossil"], 2), oil, "MW",
         "ENTSO-E unit register")
    note("solar units of the register", float(units.loc[units["Production Type"] == "Solar",
                                                        "Installed Capacity [MW]"].astype(float).sum()), "MW",
         "ENTSO-E unit register")
    note("Paks II capex (EUR %s m for %d x %s MW, eur_usd %s)" % (num(PAKS2_CAPEX_MEUR, 0), len(PAKS2),
                                                                num(PAKS2_MW, 0), num(usd, 4)),
         PAKS2_CAPEX_MEUR / (len(PAKS2) * PAKS2_MW) * usd, "MUSD/MW", "assumption")
    for tech, v in sorted(limits.items()):
        note("build limit %s: observed or best year" % tech, v["observed"], "MW", "computed")
        note("build limit %s: seed" % tech, v["seed"], "MW", "computed")
        note("build limit %s: mean of %d to %d" % (tech, BL_YEARS[0], BL_YEARS[-1]), v["mean"], "MW per year", "computed")
        log.append("    candidate  %-32s %s MW per year (start %.1f)" % (", ".join(v["rows"]), v["text"],
                                                                          max(v["observed"], v["seed"])))
    log.append("  build limit rule checked on the deployed Bulgarian rows: %s" % ", ".join(
        "%s %s" % kv for kv in guard.items()))

    # firm capacity against the peak and the planning reserve margin
    firm = {y: sum(float(r["Capacity"]) for r in fleet if r["tech"] in FIRM_TECHS and r["Status"] in ("1", "2")
                   and int(r["StYr"]) <= y and int(r["RetrYr"]) > y) for y in (HU_BASE_YEAR, HU_COAL_OUT, PAKS2_START)}
    for y, v in firm.items():
        need = peak[y] * (1 + float(margin[0]))
        note("firm capacity %d (thermal, nuclear, hydro; operating and committed)" % y, v, "MW", "computed")
        note("peak %d with the planning reserve margin" % y, need, "MW", "computed")
        log.append("  firm capacity %d: %.0f MW against %.0f (peak and margin %s)" % (y, v, need, margin[0]))

    # ── H11 calibration targets for the test run, and the observed borders ──
    implied = {"Solar": cap_gw["Solar"] * achieved["PV"] * 8.76,
               "Wind": cap_gw["Wind"] * achieved["OnshoreWind"] * wind_share * 8.76}
    for f, v in implied.items():
        note("%s output of the %d fleet in the model, a full year" % (f.lower(), HU_BASE_YEAR), v, "TWh", "computed")
    note("net imports %d" % HU_BASE_YEAR,
         ember(HU_BASE_YEAR, "Electricity imports", "Electricity imports", "Net Imports", "TWh"), "TWh", src_ember)
    note("total generation %d" % HU_BASE_YEAR,
         ember(HU_BASE_YEAR, "Electricity generation", "Total", "Total Generation", "TWh"), "TWh", src_ember)
    da = pd.read_csv(hu_cache("fleet/da_HU_%d.csv" % HU_BASE_YEAR), index_col=0).iloc[:, 0]
    note("day-ahead price %d, mean of the hours" % HU_BASE_YEAR, float(da.mean()), "EUR/MWh", src_entsoe)
    net = 0.0
    for zone, code in HU_FLOW_CODES.items():
        for a, b, label in (("HU", code, "%s to %s" % (HU_ZONE, zone)), (code, "HU", "%s to %s" % (zone, HU_ZONE))):
            f = pd.read_csv(hu_cache("flows/flow_%s_%s_%d.csv" % (a, b, HU_BASE_YEAR)), index_col=0).iloc[:, 0]
            f = pd.Series(f.to_numpy(float), index=pd.to_datetime(f.index, utc=True)).resample("h").mean().dropna()
            net += float(f.sum()) / 1e6 * (1 if b == "HU" else -1)
            note("physical flow %d, %s" % (HU_BASE_YEAR, label), float(f.sum()) / 1e6, "TWh", src_entsoe)
            note("physical flow %d, %s, 99th percentile of the hours" % (HU_BASE_YEAR, label), float(f.quantile(0.99)),
                 "MW", src_entsoe)
    note("net imports %d, physical flows" % HU_BASE_YEAR, net, "TWh", src_entsoe)
    log.append("  %d in the model: solar %.2f TWh (Ember %.2f, the capacity of the end of the year runs the whole "
               "year), wind %.2f TWh (Ember %.2f)" % (HU_BASE_YEAR, implied["Solar"], gen_twh["Solar"],
                                                      implied["Wind"], gen_twh["Wind"]))

    ok &= emit(TRACE / "edit4_hungary.csv",
               pd.DataFrame(trace, columns=["item", "value", "unit", "source"]).to_csv(
                   index=False, lineterminator="\n").encode("utf-8"), check, log)
    return ok, log


def edit5_bulgaria(check=False):
    """Trace and control of edit 5. The fleet and storage files are written by edit 4, which applies
    bulgaria_fleet() to the v5 rows before it appends the Hungarian ones."""
    log = ["Edit 5. Bulgarian fleet"]
    ok, records = True, []
    for token in GRID_TOKENS:
        # the rows expected in the files are the v5 rows after edits 5 and 6: edit 6 touches Armenian rows only
        g_header, g_new, s_header, s_new, found, _ = v5_rows_after_edits(token)
        records += found
        held_ok, held_log = fleet_files_hold(token, g_header, g_new, s_header, s_new, "edit 5")
        ok &= held_ok
        log += held_log
        cols = g_header.split(",")
        k = {c: cols.index(c) for c in ("z", "f", "RetrYr", "Capacity")}
        coal = [r.split(",") for r in g_new]
        coal = [p for p in coal if p[k["z"]] == BG_ZONE and p[k["f"]] in ("DomesticCoal", "ImportedCoal")]
        years = [int(v) for v in text_table(DATA / GRID_YEARS[token])[3]]
        log.append("  %s, coal and lignite in service, MW: %s" % (token, ", ".join(
            "%d %s" % (y, num(sum(float(p[k["Capacity"]]) for p in coal if int(p[k["RetrYr"]]) > y), 1)) for y in years)))
        for rec in found:
            if rec[2] in (BG_BATTERY, BG_NUCLEAR_NEW) and rec[3] != "row":
                log.append("  %s, %s %s: %s to %s (%s)" % (token, rec[2], rec[3], rec[4], rec[5], rec[1]))
    ok &= emit(TRACE / "edit5_bulgaria.csv",
               pd.DataFrame(records, columns=["grid", "file", "unit", "field", "v5", "v6", "basis"]).to_csv(
                   index=False, lineterminator="\n").encode("utf-8"), check, log)
    return ok, log


def edit6_armenia(check=False):
    """Trace and control of edit 6. The Armenian rows are written by edit 4 (fleet, storage, demand) and by edit 2
    (fuel), which apply armenia_fleet() and armenia_series() to the v5 rows. The trace holds figures of the
    country team table: it is written with the inputs, which are not in git."""
    log = ["Edit 6. Armenia, table of the country team (%s, sha256 %s)" % (AM_BOOK.name, AM_BOOK_SHA256[:12])]
    ok, records = True, []
    for token in GRID_TOKENS:
        g_header, g_new, s_header, s_new, _, found = v5_rows_after_edits(token)
        records += found
        held_ok, held_log = fleet_files_hold(token, g_header, g_new, s_header, s_new, "edit 6")
        ok &= held_ok
        log += held_log
        cols = g_header.split(",")
        k = {c: cols.index(c) for c in ("g", "z", "Status", "StYr", "RetrYr", "Capacity")}
        units = [p for p in (r.split(",") for r in g_new) if p[k["z"]] == AM_ZONE]
        log.append("  %s, %d units of %s. Committed: %s. Candidates: %s" % (
            token, len(units), AM_ZONE,
            ", ".join("%s %s MW in %s" % (p[k["g"]], p[k["Capacity"]], p[k["StYr"]]) for p in units if p[k["Status"]] == "2")
            or "none",
            ", ".join("%s %s MW from %s" % (p[k["g"]], p[k["Capacity"]], p[k["StYr"]]) for p in units if p[k["Status"]] == "3")))
        for rec in found:
            log.append("  %s, %s %s: %s to %s" % (token, rec[2], rec[3], rec[4], rec[5]))
    for kind, folder, v5_name, v6_name in (("demand", LOAD, AM_DEMAND_V5, AM_DEMAND_V6), ("fuel", SUPPLY, AM_FUEL_V5, AM_FUEL_V6)):
        _, _, header, rows = text_table(folder / v5_name)
        new, found = armenia_series(kind, header, rows)
        records += found
        want = [r for r in new if r.split(",")[0] == AM_ZONE]
        disk = text_table(folder / v6_name) if (folder / v6_name).exists() else None
        if disk is None or disk[2] != header or [r for r in disk[3] if r.split(",")[0] == AM_ZONE] != want:
            ok = False
            log.append("  %s  %s: the rows of edit 6 are not in the file, build edit %s (it writes this file)"
                       % ("MISSING" if disk is None else "DIFFERS", v6_name, "4" if kind == "demand" else "2"))
            continue
        log.append("  holds     %s (%d rows of %s, %d cells changed)" % (v6_name, len(want), AM_ZONE, len(found)))
        years = header.split(",")[2:]
        for row in want:
            p = row.split(",")
            if any(rec[2] == "%s %s" % (p[0], p[1]) for rec in found):
                log.append("  %s %s: %s" % (p[0], p[1], ", ".join(
                    "%s %s" % (y, p[2 + years.index(y)]) for y in ("2025", "2030", "2036", "2040") if y in years)))
    ok &= emit(AM_TRACE, pd.DataFrame(records, columns=["grid", "file", "unit", "field", "v5", "v6", "basis"]).to_csv(
        index=False, lineterminator="\n").encode("utf-8"), check, log)
    return ok, log


EDITS = {1: edit1_border_prices, 2: edit2_fuel_co2, 3: edit3_border_limits, 4: edit4_hungary, 5: edit5_bulgaria,
         6: edit6_armenia}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Version 6 inputs of the Black Sea model, one function per edit")
    ap.add_argument("--edit", type=int, action="append", choices=sorted(EDITS), required=True)
    ap.add_argument("--check", action="store_true", help="compare the files with a fresh build, write nothing")
    args = ap.parse_args(argv)
    status = 0
    for n in args.edit:
        ok, log = EDITS[n](check=args.check)
        print("\n".join(log))
        if not ok:
            status = 1
    if args.check:
        print("check: %s" % ("files match a fresh build" if status == 0 else "files differ from a fresh build"))
    return status


if __name__ == "__main__":
    sys.exit(main())
