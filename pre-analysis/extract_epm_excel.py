"""Extract all data from legacy EPM Excel (.xlsb) files into new EPM CSV format.

Sources
-------
Romania : WB_EPM_RO_12_42.xlsb  (v8.5.6)
Georgia : WB_EPM_v8_5.xlsb      (v8.5)

Output
------
epm/input/data_romania/   (for romania_2026 branch)
epm/input/data_georgia/   (for georgia_2026 branch)

Usage
-----
    conda activate gams_env
    cd pre-analysis
    python extract_epm_excel.py                 # both countries
    python extract_epm_excel.py --country Romania
    python extract_epm_excel.py --country Georgia
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyxlsb

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parent
EPM_INPUT   = BASE_DIR.parent / "epm" / "input"
TEMPLATE    = EPM_INPUT / "data_blacksea"

RO_EXCEL = Path(r"C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Romania\2-Model\WB_EPM_RO_12_42.xlsb")
GE_EXCEL = Path(r"C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia\EPM_Georgia2022\Baseline\WB_EPM_v8_5.xlsb")

# Year range for output CSVs
FIRST_YEAR = 2024
LAST_YEAR  = 2053
YEARS      = list(range(FIRST_YEAR, LAST_YEAR + 1))
YEAR_COLS  = [str(y) for y in YEARS]

# ── Per-country config ─────────────────────────────────────────────────────────

CONFIGS = {
    "Romania": {
        "excel":    RO_EXCEL,
        "out_dir":  EPM_INPUT / "data_romania",
        "zone":     "Romania",       # target zone name in output CSVs
        "zone_src": "RomaniaZ",      # zone name used inside the Excel
        "country":  "Romania",
        "ext_zones": [],             # external zones to exclude from topology
    },
    "Georgia": {
        "excel":    GE_EXCEL,
        "out_dir":  EPM_INPUT / "data_georgia",
        "zone":     "Georgia",
        "zone_src": "GE",
        "country":  "Georgia",
        "ext_zones": ["EX"],
    },
}

# Mapping from old fuel/tech name → new EPM tech name (for VRE profiles)
FUEL_TO_TECH = {
    "Solar":          "PV",
    "Wind":           "OnshoreWind",
    "WindOffshore":   "OffshoreWind",
    "ROR":            "ROR",
    "Hydro":          "ReservoirHydro",
    "TRGen":          "ROR",  # Georgia geothermal/ROR proxy
}

# ── Sheet reading helper ───────────────────────────────────────────────────────

def read_sheet(excel_path: Path, sheet: str) -> list[list]:
    """Read all rows from a sheet; returns list of lists of raw values."""
    rows = []
    with pyxlsb.open_workbook(str(excel_path)) as wb:
        with wb.get_sheet(sheet) as ws:
            for row in ws.rows():
                rows.append([c.v for c in row])
    return rows


def find_data_start(rows: list[list], marker_col0: str | None = None) -> int:
    """Return index of first row whose col-0 looks like data (not header/None)."""
    for i, row in enumerate(rows):
        v = row[0] if row else None
        if v is None or v == "":
            continue
        if marker_col0 and str(v).strip() == marker_col0:
            return i
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip() and v not in (
            "Plants", "Battery", "FROM", "ZONE", "Zone", "Country", "Seasons",
            "GENERAL", "Parameter", "CARBON PRICE ($/ton)",
        )):
            return i
    return 0


def norm_hour(h: str) -> str:
    """Normalise 't1'->'t01', 't10'->'t10', already 't01'->'t01'."""
    h = str(h).strip()
    if h.startswith("t") and len(h) < 4:
        return "t" + h[1:].zfill(2)
    return h


# ── GenData ────────────────────────────────────────────────────────────────────

TECH_MAP = {
    1: "OCGT",   2: "CCGT",   3: "ST",    4: "ReservoirHydro",
    5: "OnshoreWind", 6: "PV", 7: "ICE",  9: "CCGT",
    10: "ST",    11: "BiomassPlant", 12: "ROR", 13: "ST",
    14: "CCGT",  17: None,    20: "PV",   21: "PV",
    25: "OffshoreWind",
}

FUEL_MAP_GEN = {
    1: "ImportedCoal", 2: "Gas",    3: "Water",  4: "Solar",
    5: "Wind",         6: "Import", 7: "HFO",    8: "Uranium",
    9: "CSP",          10: "Battery", 11: "Diesel", 12: "GasCCS",
    13: "CoalCCS",     14: "Hydrogen", 15: "Biomass", 16: "Geothermal",
    **{i: "Wind" for i in range(17, 26)},
    26: "Solar",
}

STATUS_MAP = {1: 1, 2: 2, 3: 3, 4: 1}  # old → new (1=exist,2=commit,3=candidate)

GENDATA_COLS = [
    "g", "z", "tech", "f", "Linked plants", "Status", "StYr", "RetrYr",
    "Life", "Capacity", "UnitSize", "BuildLimitperYear",
    "CapacityMWh", "Efficiency", "Capex", "FOMperMW", "VOM",
    "ReserveCost", "CapexMWh", "HeatRate", "RampUpRate", "RampDnRate", "ResLimShare",
]


def extract_gendata(cfg: dict) -> pd.DataFrame:
    rows = read_sheet(cfg["excel"], "GenData")
    zone_name = cfg["zone"]

    # Detect label row: 'Plants', 'StYr', 'RetrYr', ... (skip the preceding units row)
    hdr_idx = next(
        (i for i, r in enumerate(rows)
         if r and r[0] == "Plants" and len(r) > 1 and r[1] == "StYr"),
        None,
    )
    if hdr_idx is None:
        raise ValueError("Could not find 'Plants'/'StYr' label row in GenData")

    col_row = rows[hdr_idx]
    idx_row = rows[hdr_idx - 2] if hdr_idx >= 2 else []

    def col_by_idx(target: int) -> int | None:
        for j, v in enumerate(idx_row):
            if v == target:
                return j
        return None

    def get(row, col):
        if col is None or col >= len(row):
            return None
        return row[col]

    # Column positions by numeric index in the row above header
    # GAMS index → column:  6=Type, 7=Status, 8=fuel1, 1=StYr, 2=RetrYr, 3=Zone, 4=Capacity
    COL = {k: col_by_idx(k) for k in range(1, 35)}

    # Build fuel map from GenFuel sheet (plant → primary fuel name)
    plant_fuel: dict[str, str] = {}
    try:
        gf_rows = read_sheet(cfg["excel"], "GenFuel")
        # Find header row with fuel names
        gf_hdr = None
        for gi, gr in enumerate(gf_rows):
            names = [v for v in gr if isinstance(v, str) and v not in ("", None)]
            if len(names) >= 3 and any(n in ("Coal", "Gas", "Water") for n in names):
                gf_hdr = gi
                break
        if gf_hdr is not None:
            fuel_col_map: dict[int, str] = {}
            for j, v in enumerate(gf_rows[gf_hdr]):
                if isinstance(v, str) and v not in ("", None):
                    fuel_col_map[j] = str(v).strip()
            for gr in gf_rows[gf_hdr + 1:]:
                if not gr or gr[0] is None or gr[0] == "":
                    continue
                pname = str(gr[0]).strip()
                if not pname:
                    continue
                for j, fuelname in fuel_col_map.items():
                    if j < len(gr) and isinstance(gr[j], (int, float)) and gr[j] == 1:
                        if pname not in plant_fuel:  # keep first match = primary fuel
                            plant_fuel[pname] = fuelname
    except Exception:
        pass

    records = []
    for row in rows[hdr_idx + 1:]:
        if not row or row[0] is None or row[0] == "":
            continue
        plant = str(row[0]).strip()
        if not plant or plant in ("Plants",):
            continue

        type_v = get(row, COL.get(6))  # GAMS idx 6 = Type (col 19)
        if type_v is None:
            continue
        try:
            type_i = int(type_v)
        except (TypeError, ValueError):
            continue

        tech = TECH_MAP.get(type_i)
        if tech is None:
            continue  # skip storage (type=17) — goes to pStorageDataInput

        # Fuel: from GenFuel lookup, fallback to fuel1 column (GAMS idx 8 = col 20)
        fuel = plant_fuel.get(plant)
        if fuel is None:
            fuel_v = get(row, COL.get(8))  # GAMS idx 8 = fuel1
            try:
                fuel_i = int(fuel_v) if fuel_v is not None else None
            except (TypeError, ValueError):
                fuel_i = None
            fuel = FUEL_MAP_GEN.get(fuel_i, "Gas")

        # Nuclear override
        if fuel == "Uranium":
            tech = "Nuclear"

        # GAMS indices: 2=StYr, 3=RetrYr, 5=Capacity, 7=Status,
        #               10=HeatRate, 11=FOM, 12=VOM, 13=ReserveC, 14=Capex,
        #               15=Life, 19=MinLimShare, 20=RampUp, 21=RampDn, 23=ResLimShare, 25=UnitSize
        sty  = get(row, COL.get(2))
        rety = get(row, COL.get(3))
        cap  = get(row, COL.get(5))
        stat_raw = get(row, COL.get(7))  # GAMS idx 7 = Status
        try:
            stat = STATUS_MAP.get(int(stat_raw), 1) if stat_raw is not None else 1
        except (TypeError, ValueError):
            stat = 1

        def safe_yr(v):
            try:
                y = int(v)
                return y if 1900 <= y <= 2100 else None
            except (TypeError, ValueError):
                return None

        sty  = safe_yr(sty)
        rety = safe_yr(rety)
        z = zone_name  # single zone for standalone model

        life   = get(row, COL.get(15))
        capex  = get(row, COL.get(14))
        fom    = get(row, COL.get(11))
        vom    = get(row, COL.get(12))
        hr     = get(row, COL.get(10)) or ""
        eff    = ""  # efficiency not in old GenData for thermal; storage uses separate sheet
        ramp_u = get(row, COL.get(20)) or 1
        ramp_d = get(row, COL.get(21)) or 1
        res    = get(row, COL.get(23)) or 0
        unit_s = get(row, COL.get(25)) or ""
        build_lim = get(row, COL.get(16)) or ""

        rec = {
            "g": plant, "z": z, "tech": tech, "f": fuel,
            "Linked plants": "", "Status": stat,
            "StYr": sty or "", "RetrYr": rety or "",
            "Life": int(life) if isinstance(life, (int, float)) else "",
            "Capacity": cap or "", "UnitSize": unit_s, "BuildLimitperYear": build_lim,
            "CapacityMWh": "", "Efficiency": eff,
            "Capex": capex or "", "FOMperMW": fom or "",
            "VOM": vom or "", "ReserveCost": get(row, COL.get(13)) or "",
            "CapexMWh": "", "HeatRate": hr,
            "RampUpRate": ramp_u, "RampDnRate": ramp_d, "ResLimShare": res,
        }
        records.append(rec)

    df = pd.DataFrame(records, columns=GENDATA_COLS)
    return df


# ── FuelPrices ─────────────────────────────────────────────────────────────────

def extract_fuel_prices(cfg: dict) -> pd.DataFrame:
    rows = read_sheet(cfg["excel"], "FuelPrices")
    zone = cfg["zone"]

    # Find header row: contains year columns as numbers
    hdr_idx = None
    year_cols_pos = []
    for i, row in enumerate(rows):
        years = [j for j, v in enumerate(row) if isinstance(v, (int, float)) and 2010 <= v <= 2060]
        if len(years) >= 3:
            hdr_idx = i
            year_cols_pos = years
            break
    if hdr_idx is None:
        return pd.DataFrame(columns=["z", "f"] + YEAR_COLS)

    header = rows[hdr_idx]
    available_years = {int(header[j]): j for j in year_cols_pos}

    records = []
    for row in rows[hdr_idx + 1:]:
        if not row or row[0] is None or row[0] == "":
            continue
        zone_raw = str(row[0]).strip()
        fuel_raw = row[1] if len(row) > 1 else None
        if not zone_raw or not fuel_raw:
            continue
        if zone_raw in ("", "''") or fuel_raw in ("", "''"):
            continue
        # Skip "Country2", "Country3" etc. unless it's a relevant zone
        if zone_raw.startswith("Country") and zone_raw not in ("Country1",):
            continue

        fuel = str(fuel_raw).strip()
        rec = {"z": zone, "f": fuel}
        for yr in YEARS:
            col = available_years.get(yr)
            # If year not in Excel, forward-fill from last available
            if col is None:
                last_yr = max((y for y in available_years if y <= yr), default=None)
                col = available_years.get(last_yr) if last_yr else None
            rec[str(yr)] = row[col] if col is not None and col < len(row) else ""
        records.append(rec)

    if not records:
        return pd.DataFrame(columns=["z", "f"] + YEAR_COLS)
    df = pd.DataFrame(records, columns=["z", "f"] + YEAR_COLS)
    return df


# ── Demand Forecast ────────────────────────────────────────────────────────────

def extract_demand_forecast(cfg: dict) -> pd.DataFrame:
    rows = read_sheet(cfg["excel"], "Demand_Forecast")
    zone = cfg["zone"]

    # Find year header row
    hdr_idx = None
    year_cols_pos = []
    for i, row in enumerate(rows):
        years = [j for j, v in enumerate(row) if isinstance(v, (int, float)) and 2010 <= v <= 2060]
        if len(years) >= 3:
            hdr_idx = i
            year_cols_pos = years
            break
    if hdr_idx is None:
        return pd.DataFrame(columns=["z", "type"] + YEAR_COLS)

    header = rows[hdr_idx]
    available_years = {int(header[j]): j for j in year_cols_pos}

    records_by_zone: dict[str, dict] = {}
    for row in rows[hdr_idx + 1:]:
        if not row or row[0] is None or row[0] == "":
            continue
        zone_raw = str(row[0]).strip()
        label_raw = row[1] if len(row) > 1 else None
        if not zone_raw or not label_raw:
            continue
        label = str(label_raw).strip().lower()
        if label not in ("peak", "energy"):
            continue

        key = (zone_raw, label)
        rec = {"z": zone_raw, "type": "Peak" if label == "peak" else "Energy"}
        for yr in YEARS:
            col = available_years.get(yr)
            if col is None:
                # extrapolate 2% growth from last available year
                last_yr = max((y for y in available_years if y <= yr), default=None)
                if last_yr:
                    last_col = available_years[last_yr]
                    last_val = row[last_col] if last_col < len(row) else None
                    if isinstance(last_val, (int, float)):
                        rec[str(yr)] = round(last_val * (1.02 ** (yr - last_yr)), 2)
                    else:
                        rec[str(yr)] = ""
                else:
                    rec[str(yr)] = ""
            else:
                rec[str(yr)] = row[col] if col < len(row) else ""
        records_by_zone[key] = rec

    if not records_by_zone:
        return pd.DataFrame(columns=["z", "type"] + YEAR_COLS)

    # Aggregate sub-zones to a single zone (e.g., North+South+Test → Georgia)
    agg: dict[str, dict] = {}
    for (zone_raw, label), rec in records_by_zone.items():
        if label not in agg:
            agg[label] = {"z": zone, "type": "Peak" if label == "peak" else "Energy"}
            for yr_s in YEAR_COLS:
                agg[label][yr_s] = 0.0
        for yr_s in YEAR_COLS:
            v = rec.get(yr_s, 0)
            if isinstance(v, (int, float)):
                agg[label][yr_s] = round(agg[label][yr_s] + v, 4)

    rows_out = [agg[l] for l in ("peak", "energy") if l in agg]
    df = pd.DataFrame(rows_out, columns=["z", "type"] + YEAR_COLS)
    return df


# ── Transfer Limit ─────────────────────────────────────────────────────────────

def extract_transfer_limit(cfg: dict) -> pd.DataFrame:
    """Extract internal transfer limits; for single-zone returns empty DataFrame."""
    return pd.DataFrame(columns=["z", "z2", "s"] + YEAR_COLS)


# ── pHours (Duration) ──────────────────────────────────────────────────────────

def extract_hours(cfg: dict) -> pd.DataFrame:
    rows = read_sheet(cfg["excel"], "Duration")

    # Find header row with t01/t1 hour labels
    hdr_idx = None
    hour_start_col = None
    for i, row in enumerate(rows):
        hour_cols = [j for j, v in enumerate(row) if isinstance(v, str) and v.strip().lower().startswith("t")]
        if len(hour_cols) >= 20:
            hdr_idx = i
            hour_start_col = hour_cols[0]
            break
    if hdr_idx is None:
        raise ValueError("Could not find hour header in Duration sheet")

    header = rows[hdr_idx]
    n_hours = min(24, len(header) - hour_start_col)
    hour_labels = [norm_hour(header[hour_start_col + k]) for k in range(n_hours)]

    records = []
    for row in rows[hdr_idx + 1:]:
        q = row[0] if row else None
        d = row[1] if len(row) > 1 else None
        if q is None or d is None:
            continue
        if not isinstance(q, str) or not isinstance(d, str):
            continue
        q = q.strip()
        d = d.strip()
        if not q or not d:
            continue

        vals = {}
        for k, lbl in enumerate(hour_labels):
            col = hour_start_col + k
            v = row[col] if col < len(row) else 1
            vals[lbl] = int(v) if isinstance(v, (int, float)) else 1

        rec = {"q": q, "d": d, **vals}
        records.append(rec)

    df = pd.DataFrame(records, columns=["q", "d"] + hour_labels)
    # Duration sheet may repeat rows per year; keep only first occurrence of each (q,d)
    df = df.drop_duplicates(subset=["q", "d"], keep="first")
    return df


# ── pDemandProfile ─────────────────────────────────────────────────────────────

def extract_demand_profile(cfg: dict) -> pd.DataFrame:
    rows = read_sheet(cfg["excel"], "DemandProfile")
    zone_src = cfg["zone_src"]
    zone_dst = cfg["zone"]

    # Find header row with t01/t1 labels
    hdr_idx = None
    hour_start_col = None
    for i, row in enumerate(rows):
        hour_cols = [j for j, v in enumerate(row) if isinstance(v, str) and v.strip().lower().startswith("t")]
        if len(hour_cols) >= 20:
            hdr_idx = i
            hour_start_col = hour_cols[0]
            break
    if hdr_idx is None:
        raise ValueError("Could not find hour header in DemandProfile sheet")

    header = rows[hdr_idx]
    n_hours = min(24, len(header) - hour_start_col)
    hour_labels = [norm_hour(header[hour_start_col + k]) for k in range(n_hours)]

    # Aggregate sub-zones (for Georgia: North, South, Test → Georgia)
    # key: (season, daytype) → array of 24 floats
    agg_mw: dict[tuple, list[float]] = {}
    for row in rows[hdr_idx + 1:]:
        if not row or row[0] is None or row[0] == "":
            continue
        zone_cell = str(row[0]).strip()
        season    = str(row[1]).strip() if len(row) > 1 and row[1] else None
        daytype   = str(row[2]).strip() if len(row) > 2 and row[2] else None
        if not season or not daytype:
            continue

        vals = []
        for k in range(n_hours):
            col = hour_start_col + k
            v = row[col] if col < len(row) else 0.0
            vals.append(float(v) if isinstance(v, (int, float)) else 0.0)

        key = (season, daytype)
        if key not in agg_mw:
            agg_mw[key] = [0.0] * n_hours
        for k in range(n_hours):
            agg_mw[key][k] += vals[k]

    if not agg_mw:
        return pd.DataFrame(columns=["zone", "season", "daytype"] + hour_labels)

    # Normalize to peak=1 across all season/daytype combinations
    peak = max(max(v) for v in agg_mw.values())
    if peak <= 0:
        peak = 1.0

    records = []
    for (season, daytype), vals in sorted(agg_mw.items()):
        rec = {"zone": zone_dst, "season": season, "daytype": daytype}
        for k, lbl in enumerate(hour_labels):
            rec[lbl] = round(vals[k] / peak, 8)
        records.append(rec)

    df = pd.DataFrame(records, columns=["zone", "season", "daytype"] + hour_labels)
    return df


# ── pVREProfile ────────────────────────────────────────────────────────────────

def extract_vre_profile(cfg: dict) -> pd.DataFrame:
    rows = read_sheet(cfg["excel"], "REProfile")
    zone_src = cfg["zone_src"]
    zone_dst = cfg["zone"]

    # Find header row with t labels
    hdr_idx = None
    hour_start_col = None
    for i, row in enumerate(rows):
        hour_cols = [j for j, v in enumerate(row) if isinstance(v, str) and v.strip().lower().startswith("t")]
        if len(hour_cols) >= 20:
            hdr_idx = i
            hour_start_col = hour_cols[0]
            break
    if hdr_idx is None:
        return pd.DataFrame(columns=["zone", "tech", "season", "daytype"])

    header = rows[hdr_idx]
    n_hours = min(24, len(header) - hour_start_col)
    hour_labels = [norm_hour(header[hour_start_col + k]) for k in range(n_hours)]

    records = []
    for row in rows[hdr_idx + 1:]:
        if not row or row[0] is None or row[0] == "":
            continue
        zone_cell = str(row[0]).strip()
        if zone_cell == zone_src:
            zone_cell = zone_dst
        fuel_cell = str(row[1]).strip() if len(row) > 1 and row[1] else None
        season    = str(row[2]).strip() if len(row) > 2 and row[2] else None
        daytype   = str(row[3]).strip() if len(row) > 3 and row[3] else None
        if not fuel_cell or not season or not daytype:
            continue

        tech = FUEL_TO_TECH.get(fuel_cell, fuel_cell)

        rec = {"zone": zone_cell, "tech": tech, "season": season, "daytype": daytype}
        for k, lbl in enumerate(hour_labels):
            col = hour_start_col + k
            v = row[col] if col < len(row) else 0.0
            rec[lbl] = float(v) if isinstance(v, (int, float)) else 0.0
        records.append(rec)

    if not records:
        return pd.DataFrame(columns=["zone", "tech", "season", "daytype"] + hour_labels)
    df = pd.DataFrame(records, columns=["zone", "tech", "season", "daytype"] + hour_labels)
    return df


# ── pVREgenProfile ─────────────────────────────────────────────────────────────

def extract_vre_gen_profile(cfg: dict) -> pd.DataFrame:
    try:
        rows = read_sheet(cfg["excel"], "REgenProfile")
    except Exception:
        return pd.DataFrame(columns=["g", "q", "d"])

    # Find header row with t labels
    hdr_idx = None
    hour_start_col = None
    for i, row in enumerate(rows):
        hour_cols = [j for j, v in enumerate(row) if isinstance(v, str) and v.strip().lower().startswith("t")]
        if len(hour_cols) >= 20:
            hdr_idx = i
            hour_start_col = hour_cols[0]
            break
    if hdr_idx is None:
        return pd.DataFrame(columns=["g", "q", "d"])

    header = rows[hdr_idx]
    n_hours = min(24, len(header) - hour_start_col)
    hour_labels = [norm_hour(header[hour_start_col + k]) for k in range(n_hours)]

    records = []
    for row in rows[hdr_idx + 1:]:
        if not row or row[0] is None or row[0] == "":
            continue
        plant   = str(row[0]).strip()
        season  = str(row[2]).strip() if len(row) > 2 and row[2] else None
        daytype = str(row[3]).strip() if len(row) > 3 and row[3] else None
        if not season or not daytype:
            continue

        rec = {"g": plant, "q": season, "d": daytype}
        for k, lbl in enumerate(hour_labels):
            col = hour_start_col + k
            v = row[col] if col < len(row) else 0.0
            rec[lbl] = float(v) if isinstance(v, (int, float)) else 0.0
        records.append(rec)

    if not records:
        return pd.DataFrame(columns=["g", "q", "d"] + hour_labels)
    df = pd.DataFrame(records, columns=["g", "q", "d"] + hour_labels)
    return df


# ── pAvailabilityCustom ────────────────────────────────────────────────────────

def extract_availability(cfg: dict) -> pd.DataFrame:
    rows = read_sheet(cfg["excel"], "GenAvailability")

    # Find header row: row with season labels
    hdr_idx = None
    season_cols: list[tuple[int, str]] = []
    for i, row in enumerate(rows):
        sc = [(j, str(v).strip()) for j, v in enumerate(row)
              if isinstance(v, str) and (v.startswith("Q") or v.startswith("m")) and len(v) <= 3]
        if len(sc) >= 4:
            hdr_idx = i
            season_cols = sc
            break
    if hdr_idx is None:
        return pd.DataFrame(columns=["gen"])

    seasons = [s for _, s in season_cols]

    records = []
    for row in rows[hdr_idx + 1:]:
        if not row or row[0] is None or row[0] == "":
            continue
        plant = str(row[0]).strip()
        if not plant or plant in ("Generators with errors",):
            continue
        rec = {"gen": plant}
        for col, season in season_cols:
            v = row[col] if col < len(row) else None
            rec[season] = float(v) if isinstance(v, (int, float)) else ""
        records.append(rec)

    if not records:
        return pd.DataFrame(columns=["gen"] + seasons)
    df = pd.DataFrame(records, columns=["gen"] + seasons)
    return df


# ── pCapexTrajectoriesCustom ───────────────────────────────────────────────────

def extract_capex_trajectories(cfg: dict) -> pd.DataFrame:
    rows = read_sheet(cfg["excel"], "CapexTrajectories")

    # Find header row with years
    hdr_idx = None
    year_cols_pos: list[tuple[int, int]] = []
    for i, row in enumerate(rows):
        yc = [(j, int(v)) for j, v in enumerate(row)
              if isinstance(v, (int, float)) and 2015 <= v <= 2060]
        if len(yc) >= 5:
            hdr_idx = i
            year_cols_pos = yc
            break
    if hdr_idx is None:
        return pd.DataFrame(columns=["gen"] + YEAR_COLS)

    excel_years = {yr: col for col, yr in year_cols_pos}

    records = []
    for row in rows[hdr_idx + 1:]:
        if not row or row[0] is None or row[0] == "":
            continue
        plant = str(row[0]).strip()
        if not plant:
            continue

        rec = {"gen": plant}
        last_val = 1.0
        for yr in YEARS:
            col = excel_years.get(yr)
            if col is not None and col < len(row) and isinstance(row[col], (int, float)):
                v = row[col]
                last_val = v
            else:
                v = last_val
            rec[str(yr)] = round(v, 6)
        records.append(rec)

    if not records:
        return pd.DataFrame(columns=["gen"] + YEAR_COLS)
    df = pd.DataFrame(records, columns=["gen"] + YEAR_COLS)
    return df


# ── pStorageDataInput ──────────────────────────────────────────────────────────

STORAGE_COLS = [
    "g", "z", "tech", "f", "Linked plants", "Status", "StYr", "RetrYr",
    "Life", "Capacity", "UnitSize", "BuildLimitperYear",
    "CapacityMWh", "Efficiency", "Capex", "FOMperMW", "VOM",
    "ReserveCost", "CapexMWh", "HeatRate", "RampUpRate", "RampDnRate", "ResLimShare",
]


def extract_storage(cfg: dict) -> pd.DataFrame:
    rows = read_sheet(cfg["excel"], "Storage")
    zone = cfg["zone"]

    # Find data header row: contains 'Battery', 'LinkedPlant' etc.
    hdr_idx = None
    for i, row in enumerate(rows):
        if row and row[0] == "Battery" and len(row) > 1:
            # check next rows
            if i + 1 < len(rows):
                next_row = rows[i + 1]
                if next_row and next_row[0] in ("MWh", None):
                    hdr_idx = i
                    break

    if hdr_idx is None:
        # Try to find by column pattern
        for i, row in enumerate(rows):
            strs = [str(v).strip() for v in row if isinstance(v, str)]
            if "Battery" in strs and "Capacity" in strs:
                hdr_idx = i
                break

    if hdr_idx is None:
        return pd.DataFrame(columns=STORAGE_COLS)

    header = [str(v).strip() if v else "" for v in rows[hdr_idx]]

    def find_col(name):
        return next((j for j, v in enumerate(header) if v == name), None)

    c_name  = 0
    c_link  = find_col("LinkedPlant") or find_col("Linked plant") or 1
    c_cap   = find_col("Capacity") or 2
    c_capex = find_col("Capex") or 3
    c_vom   = find_col("VOM") or 4
    c_fom   = find_col("FixedOM") or 5
    c_eff   = find_col("Efficiency") or 6
    c_life  = find_col("Life") or 7

    def get_v(row, col):
        return row[col] if col is not None and col < len(row) else None

    records = []
    for row in rows[hdr_idx + 2:]:  # skip unit rows
        if not row or row[0] is None or row[0] == "":
            continue
        name = str(row[0]).strip()
        if not name or name in ("Battery", "STORAGE CHARACTERISTICS"):
            continue

        cap_mwh = get_v(row, c_cap)
        eff     = get_v(row, c_eff)
        life    = get_v(row, c_life)
        capex   = get_v(row, c_capex)   # USD/kWh in Excel → USD/MWh × 0.001? (keep as-is)
        fom     = get_v(row, c_fom)
        vom     = get_v(row, c_vom)
        link    = get_v(row, c_link)

        # Estimate capacity (MW) from MWh with duration=4h assumption
        cap_mw = round(float(cap_mwh) / 4, 0) if isinstance(cap_mwh, (int, float)) else ""

        rec = {
            "g": name, "z": zone, "tech": "STORAGE", "f": "Battery",
            "Linked plants": link or "",
            "Status": 3, "StYr": FIRST_YEAR, "RetrYr": FIRST_YEAR + (int(life) if life else 20),
            "Life": int(life) if life else 20,
            "Capacity": cap_mw,
            "UnitSize": "", "BuildLimitperYear": "",
            "CapacityMWh": cap_mwh or "",
            "Efficiency": eff or 0.85,
            "Capex": capex or "",
            "FOMperMW": fom or "",
            "VOM": vom or "",
            "ReserveCost": "",
            "CapexMWh": "",
            "HeatRate": "", "RampUpRate": 1, "RampDnRate": 1, "ResLimShare": 0,
        }
        records.append(rec)

    if not records:
        return pd.DataFrame(columns=STORAGE_COLS)
    df = pd.DataFrame(records, columns=STORAGE_COLS)
    return df


# ── sTopology ──────────────────────────────────────────────────────────────────

def extract_topology(cfg: dict) -> pd.DataFrame:
    rows = read_sheet(cfg["excel"], "Topology")
    zone_src = cfg["zone_src"]
    zone_dst = cfg["zone"]
    ext_zones = set(cfg.get("ext_zones", []))

    # Find the matrix: row with 'FROM' and column headers
    hdr_idx = None
    first_zone_col = None
    zone_names: list[str] = []
    zone_header_row = None
    for i, row in enumerate(rows):
        if row and row[0] == "FROM":
            # Find the next non-empty row with zone names (may have blank rows in between)
            for j in range(i + 1, min(i + 6, len(rows))):
                zones_in_row = [str(v).strip() for v in rows[j][1:] if v is not None and v != ""]
                if zones_in_row:
                    hdr_idx = i
                    zone_names = zones_in_row
                    first_zone_col = 1
                    zone_header_row = j
                    break
            break

    if hdr_idx is None or zone_header_row is None or not zone_names:
        return pd.DataFrame(columns=["uni_0", "uni_1"])

    # Build col index: zone_name → column index in the data rows
    col_map = {}
    header_vals = rows[zone_header_row][first_zone_col:]
    for k, v in enumerate(header_vals):
        if v is not None and v != "":
            nm = str(v).strip()
            if nm == zone_src:
                nm = zone_dst
            col_map[nm] = first_zone_col + k

    edges = []
    for row in rows[zone_header_row + 1:]:
        if not row or row[0] is None or row[0] == "":
            continue
        from_raw = str(row[0]).strip()
        if not from_raw:
            continue
        from_z = zone_dst if from_raw == zone_src else from_raw

        for to_z, col in col_map.items():
            v = row[col] if col < len(row) else None
            if isinstance(v, (int, float)) and v == 1:
                edges.append({"uni_0": from_z, "uni_1": to_z})

    df = pd.DataFrame(edges, columns=["uni_0", "uni_1"])
    # Remove self-loops and external-only connections
    df = df[df["uni_0"] != df["uni_1"]].copy()
    return df


# ── pPlanningReserveMargin ─────────────────────────────────────────────────────

def extract_reserve(cfg: dict) -> pd.DataFrame:
    rows = read_sheet(cfg["excel"], "Reserve")
    country = cfg["country"]

    # Find data rows after 'Country'/'Zone' header
    for i, row in enumerate(rows):
        if row and str(row[0]).strip() in ("Country", "Zone"):
            for j in range(i + 1, len(rows)):
                r = rows[j]
                if r and r[0] is not None and r[0] != "" and isinstance(r[1], (int, float)):
                    value = float(r[1])
                    return pd.DataFrame([{"c": country, "value": value}])
            break

    return pd.DataFrame([{"c": country, "value": 0.15}])


# ── pCarbonPrice ───────────────────────────────────────────────────────────────

def extract_carbon_price(cfg: dict) -> pd.DataFrame:
    rows = read_sheet(cfg["excel"], "EmissionFactors")

    # Column 0=year, col 1=CO2Price
    records = []
    for row in rows:
        if not row or not isinstance(row[0], (int, float)):
            continue
        yr = int(row[0])
        if yr < 2010 or yr > 2060:
            continue
        price = row[1] if len(row) > 1 and isinstance(row[1], (int, float)) else 0.0
        records.append({"year": yr, "value": price})

    if not records:
        return pd.DataFrame(columns=["year", "value"])

    df = pd.DataFrame(records)
    # Filter/extend to our year range
    df_out = []
    for yr in YEARS:
        match = df[df["year"] == yr]
        if not match.empty:
            df_out.append({"year": yr, "value": match.iloc[0]["value"]})
        else:
            # forward-fill from last available
            prev = df[df["year"] < yr]
            val = prev.iloc[-1]["value"] if not prev.empty else 0.0
            df_out.append({"year": yr, "value": val})

    return pd.DataFrame(df_out, columns=["year", "value"])


# ── pFuelTypeCarbonContent ─────────────────────────────────────────────────────

def extract_emission_factors(cfg: dict) -> pd.DataFrame:
    rows = read_sheet(cfg["excel"], "EmissionFactors")

    # Columns 9=Fuel Type, 10=CO2 coeff  (primary table)
    # Columns 13=Fuel Name, 14=Index      (secondary / newer table)
    # Use col 13 (newer) if available, fallback col 9
    records = {}
    for row in rows:
        if not row:
            continue
        # Primary table: col 9, 10
        if len(row) > 10 and isinstance(row[9], str) and isinstance(row[10], (int, float)):
            fuel = str(row[9]).strip()
            coef = float(row[10])
            records[fuel] = coef
        # Secondary table: col 13, 14
        if len(row) > 14 and isinstance(row[13], str) and isinstance(row[14], (int, float)):
            fuel = str(row[13]).strip()
            coef = float(row[14])
            records[fuel] = coef  # override

    if not records:
        return pd.DataFrame(columns=["uni", "value"])

    df = pd.DataFrame([{"uni": f, "value": v} for f, v in records.items()])
    return df


# ── pLossFactorInternal ────────────────────────────────────────────────────────

def extract_loss_factor(cfg: dict) -> pd.DataFrame:
    rows = read_sheet(cfg["excel"], "LossFactor")
    zone_src = cfg["zone_src"]
    zone_dst = cfg["zone"]

    # Header: FROM, TO, year columns
    hdr_idx = None
    year_cols_pos = []
    for i, row in enumerate(rows):
        yc = [(j, int(v)) for j, v in enumerate(row) if isinstance(v, (int, float)) and 2010 <= v <= 2060]
        if len(yc) >= 2:
            hdr_idx = i
            year_cols_pos = yc
            break
    if hdr_idx is None:
        return pd.DataFrame(columns=["z", "z2"] + YEAR_COLS)

    header_years = {yr: col for col, yr in year_cols_pos}

    records = []
    for row in rows[hdr_idx + 1:]:
        if not row or row[0] is None or row[0] == "":
            continue
        z1 = str(row[0]).strip()
        z2 = str(row[1]).strip() if len(row) > 1 and row[1] else None
        if not z1 or not z2:
            continue
        z1 = zone_dst if z1 == zone_src else z1
        z2 = zone_dst if z2 == zone_src else z2

        rec = {"z": z1, "z2": z2}
        for yr in YEARS:
            col = header_years.get(yr)
            last_yr = max((y for y in header_years if y <= yr), default=None)
            actual_col = col or (header_years.get(last_yr) if last_yr else None)
            v = row[actual_col] if actual_col is not None and actual_col < len(row) else 0.0
            rec[str(yr)] = float(v) if isinstance(v, (int, float)) else 0.0
        records.append(rec)

    if not records:
        return pd.DataFrame(columns=["z", "z2"] + YEAR_COLS)
    df = pd.DataFrame(records, columns=["z", "z2"] + YEAR_COLS)
    return df


# ── pSettings ─────────────────────────────────────────────────────────────────

def extract_settings(cfg: dict) -> str:
    """Return pSettings.csv content by copying template and overriding key values from Excel."""
    template_path = TEMPLATE / "pSettings.csv"
    with open(template_path, encoding="utf-8") as f:
        template = f.read()

    rows = read_sheet(cfg["excel"], "Settings")

    # Build lookup: lowercase parameter name → value
    params: dict[str, object] = {}
    for row in rows:
        if not row or row[0] is None:
            continue
        key = str(row[0]).strip().lower()
        if key and len(row) > 1 and row[1] is not None:
            params[key] = row[1]
        # Also scan right half for constraint columns
        if len(row) > 5 and row[4] is not None and row[5] is not None:
            key2 = str(row[4]).strip().lower()
            if key2:
                params[key2] = row[5]

    def override(content: str, abbrev: str, new_val) -> str:
        lines = content.splitlines()
        out = []
        for line in lines:
            parts = line.split(",")
            if len(parts) >= 3 and parts[1].strip() == abbrev:
                parts[2] = str(new_val)
                line = ",".join(parts)
            out.append(line)
        return "\n".join(out)

    # Map Excel parameter names to pSettings abbreviations
    mapping = {
        "weighted average cost of capital (wacc), %": ("WACC", lambda v: round(float(v), 4)),
        "discount rate, %": ("DR", lambda v: round(float(v), 4)),
        "cost of unserved energy per mwh, $": ("VoLL", lambda v: int(v)),
        "cost of reserve shortfall per mw, $": ("ReserveVoLL", lambda v: int(v)),
        "spin reservevoll per mwh, $": ("SpinReserveVoLL", lambda v: int(v)),
        "cost of surplus power per mwh, $": ("CostSurplus", lambda v: round(float(v), 2)),
        "cost of curtailment per mwh, $": ("CostCurtail", lambda v: round(float(v), 2)),
        "include carbon price ": ("fEnableCarbonPrice", lambda v: 1 if str(v).upper() == "YES" else 0),
        "include storage operation": ("fEnableStorage", lambda v: 1 if str(v).upper() == "YES" else 0),
        "retire plants on economic grounds": ("fEnableEconomicRetirement", lambda v: 1 if str(v).upper() == "YES" else 0),
        "run in interconnected mode": ("fEnableExternalExchange", lambda v: 1 if str(v).upper() == "YES" else 0),
        "apply country planning reserve constraint": ("fApplyPlanningReserveConstraint", lambda v: 1 if str(v).upper() == "YES" else 0),
        "apply country spinning reserve contraints": ("fApplyCountrySpinReserveConstraint", lambda v: 1 if str(v).upper() == "YES" else 0),
        "apply ramp constraints": ("fApplyRampConstraint", lambda v: 1 if str(v).upper() == "YES" else 0),
        "apply min generation constraint": ("fApplyMinGenCommitment", lambda v: 1 if str(v).upper() == "YES" else 0),
    }

    for excel_key, (abbrev, transform) in mapping.items():
        if excel_key in params:
            try:
                template = override(template, abbrev, transform(params[excel_key]))
            except Exception:
                pass

    return template


# ── sRelevant ──────────────────────────────────────────────────────────────────

def build_s_relevant(phours_df: pd.DataFrame) -> pd.DataFrame:
    """Build sRelevant.csv from the day types present in pHours."""
    daytypes = sorted(phours_df["d"].unique())
    return pd.DataFrame({"s": daytypes})


# ── Scaffold helpers ───────────────────────────────────────────────────────────

def scaffold_folder(out_dir: Path) -> None:
    """Create all subdirectories needed by EPM."""
    for sub in ["supply", "load", "trade", "extras", "constraint", "reserve", "cplex"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)


def copy_template_file(src: Path, dst: Path) -> None:
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)


def write_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    df.to_csv(path, index=False, **kwargs)
    try:
        rel = path.relative_to(BASE_DIR.parent)
    except ValueError:
        rel = path
    print(f"  wrote  {rel}  ({len(df)} rows)")


def write_text(content: str, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    try:
        rel = path.relative_to(BASE_DIR.parent)
    except ValueError:
        rel = path
    print(f"  wrote  {rel}")


# ── Main per-country extraction ────────────────────────────────────────────────

def process_country(name: str) -> None:
    cfg     = CONFIGS[name]
    out_dir = cfg["out_dir"]
    zone    = cfg["zone"]
    country = cfg["country"]
    excel   = cfg["excel"]

    if not excel.exists():
        print(f"[{name}] Excel file not found: {excel}. Skipping.")
        return

    print(f"\n{'='*60}")
    print(f" Processing {name}")
    print(f"{'='*60}")

    scaffold_folder(out_dir)

    # ── GenData ──────────────────────────────────────────────────────────
    print(f"\n[{name}] Extracting GenData...")
    gendata = extract_gendata(cfg)
    write_csv(gendata, out_dir / "supply" / "pGenDataInput.csv")

    # ── FuelPrices ────────────────────────────────────────────────────────
    print(f"[{name}] Extracting FuelPrices...")
    fp = extract_fuel_prices(cfg)
    write_csv(fp, out_dir / "supply" / "pFuelPrice.csv")

    # ── Demand Forecast ───────────────────────────────────────────────────
    print(f"[{name}] Extracting Demand_Forecast...")
    df_dem = extract_demand_forecast(cfg)
    write_csv(df_dem, out_dir / "load" / "pDemandForecast.csv")

    # ── pHours ────────────────────────────────────────────────────────────
    print(f"[{name}] Extracting Duration -> pHours...")
    phours = extract_hours(cfg)
    write_csv(phours, out_dir / "pHours.csv")

    # ── pDemandProfile ────────────────────────────────────────────────────
    print(f"[{name}] Extracting DemandProfile...")
    dpro = extract_demand_profile(cfg)
    write_csv(dpro, out_dir / "load" / "pDemandProfile.csv")

    # ── pVREProfile ───────────────────────────────────────────────────────
    print(f"[{name}] Extracting REProfile -> pVREProfile...")
    vre = extract_vre_profile(cfg)
    write_csv(vre, out_dir / "supply" / "pVREProfile.csv")

    # ── pVREgenProfile ────────────────────────────────────────────────────
    print(f"[{name}] Extracting REgenProfile -> pVREgenProfile...")
    vregen = extract_vre_gen_profile(cfg)
    write_csv(vregen, out_dir / "supply" / "pVREgenProfile.csv")

    # ── Availability ──────────────────────────────────────────────────────
    print(f"[{name}] Extracting GenAvailability...")
    avail = extract_availability(cfg)
    write_csv(avail, out_dir / "supply" / "pAvailabilityCustom.csv")

    # ── CapexTrajectories ─────────────────────────────────────────────────
    print(f"[{name}] Extracting CapexTrajectories...")
    capex = extract_capex_trajectories(cfg)
    write_csv(capex, out_dir / "supply" / "pCapexTrajectoriesCustom.csv")

    # ── Storage ───────────────────────────────────────────────────────────
    print(f"[{name}] Extracting Storage...")
    stor = extract_storage(cfg)
    write_csv(stor, out_dir / "supply" / "pStorageDataInput.csv")

    # ── Topology ──────────────────────────────────────────────────────────
    print(f"[{name}] Extracting Topology...")
    topo = extract_topology(cfg)
    write_csv(topo, out_dir / "extras" / "sTopology.csv")

    # ── Transfer Limit ────────────────────────────────────────────────────
    tf = extract_transfer_limit(cfg)
    write_csv(tf, out_dir / "trade" / "pTransferLimit.csv")

    # ── Reserve ───────────────────────────────────────────────────────────
    print(f"[{name}] Extracting Reserve...")
    res = extract_reserve(cfg)
    write_csv(res, out_dir / "reserve" / "pPlanningReserveMargin.csv")

    # ── Carbon Price ──────────────────────────────────────────────────────
    print(f"[{name}] Extracting Carbon Price...")
    cp = extract_carbon_price(cfg)
    write_csv(cp, out_dir / "constraint" / "pCarbonPrice.csv")

    # ── Emission Factors ──────────────────────────────────────────────────
    print(f"[{name}] Extracting Emission Factors...")
    ef = extract_emission_factors(cfg)
    write_csv(ef, out_dir / "extras" / "pFuelTypeCarbonContent.csv")

    # ── Loss Factor ───────────────────────────────────────────────────────
    print(f"[{name}] Extracting LossFactor...")
    lf = extract_loss_factor(cfg)
    write_csv(lf, out_dir / "trade" / "pLossFactorInternal.csv")

    # ── Settings ──────────────────────────────────────────────────────────
    print(f"[{name}] Extracting Settings -> pSettings.csv...")
    settings_content = extract_settings(cfg)
    write_text(settings_content, out_dir / "pSettings.csv")

    # ── sRelevant ─────────────────────────────────────────────────────────
    srel = build_s_relevant(phours)
    write_csv(srel, out_dir / "load" / "sRelevant.csv")

    # ── Static scaffolding files ──────────────────────────────────────────

    # zcmap.csv
    zcmap_path = out_dir / "zcmap.csv"
    zcmap = pd.DataFrame([{"z": zone, "c": country}])
    write_csv(zcmap, zcmap_path)

    # y.csv — planning horizon
    y_path = out_dir / "y.csv"
    y_df = pd.DataFrame({"y": YEARS})
    write_csv(y_df, y_path)

    # config.csv — copy from template
    config_dst = out_dir / "config.csv"
    copy_template_file(TEMPLATE / "config.csv", config_dst)
    if not config_dst.exists():
        shutil.copy2(TEMPLATE / "config.csv", config_dst)
    print(f"  wrote  config.csv")

    # cplex opt file
    cplex_dst = out_dir / "cplex" / "cplex_baseline.opt"
    copy_template_file(TEMPLATE / "cplex" / "cplex_baseline.opt", cplex_dst)
    if not cplex_dst.exists():
        # minimal opt file
        write_text("epgap 0.01\nmipstart 1\n", cplex_dst)

    # Empty placeholder files (required by EPM config.csv reader)
    empty_files = [
        "load/pDemandData.csv",
        "load/pEnergyEfficiencyFactor.csv",
        "supply/pAvailabilityDefault.csv",
        "supply/pCapexTrajectoriesDefault.csv",
        "supply/pGenDataInputDefault.csv",
        "supply/pStorageDataInputDefault.csv",
        "supply/pEvolutionAvailability.csv",
        "supply/pCSPData.csv",
        "trade/pExtTransferLimit.csv",
        "trade/pMaxAnnualExternalTradeShare.csv",
        "trade/pMinImport.csv",
        "trade/pNewTransmission.csv",
        "trade/pTradePrice.csv",
        "trade/zext.csv",
        "reserve/pSpinningReserveReqCountry.csv",
        "reserve/pSpinningReserveReqSystem.csv",
        "constraint/pEmissionsCountry.csv",
        "constraint/pEmissionsTotal.csv",
        "constraint/pMaxFuellimit.csv",
        "constraint/pMaxGenerationByFuel.csv",
        "extras/MapGG.csv",
        "extras/Relevant.csv",
        "extras/pAnnualMaxBuildC.csv",
        "extras/pAnnualMaxBuildZ.csv",
        "extras/pAvailability.csv",
        "extras/pCapexTrajectory.csv",
        "extras/pScalars.csv",
        "extras/pTechDataExcel.csv",
        "extras/pZoneIndex.csv",
        "extras/hh.csv",
        "extras/ftfindex.csv",
        "scenarios.csv",
    ]
    for rel in empty_files:
        p = out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.touch()

    print(f"\n[{name}] Done. Output: {out_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract legacy EPM Excel data into new CSV format")
    parser.add_argument("--country", choices=["Romania", "Georgia"], help="Process only this country")
    args = parser.parse_args()

    countries = [args.country] if args.country else list(CONFIGS.keys())
    for c in countries:
        process_country(c)
    print("\n=== All done ===")


if __name__ == "__main__":
    main()
