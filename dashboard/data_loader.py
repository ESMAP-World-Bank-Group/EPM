"""
EPM Dashboard — Data loader
Scans output/ and input/ directories, reads CSVs with caching.
"""

from pathlib import Path
from functools import lru_cache
from typing import Optional
import pandas as pd

from config import OUTPUT_ROOT, INPUT_ROOT, CSV, INPUT_CSV


# ---------------------------------------------------------------------------
# Run & scenario discovery
# ---------------------------------------------------------------------------

def list_runs() -> list[str]:
    """Return sorted list of simulation run folder names (newest first)."""
    if not OUTPUT_ROOT.exists():
        return []
    runs = [d.name for d in OUTPUT_ROOT.iterdir()
            if d.is_dir() and d.name.startswith("simulations_run_")]
    return sorted(runs, reverse=True)


def list_scenarios(run: str) -> list[str]:
    """Return list of scenario names inside a run folder."""
    run_path = OUTPUT_ROOT / run
    if not run_path.exists():
        return []
    return [d.name for d in run_path.iterdir()
            if d.is_dir() and (d / "output_csv").exists()]


def list_all_runs_scenarios() -> dict[str, list[str]]:
    """Return {run_name: [scenario, ...]} for all available runs."""
    return {run: list_scenarios(run) for run in list_runs()}


# ---------------------------------------------------------------------------
# Output CSV loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _load_csv(path: str) -> pd.DataFrame:
    """Cached CSV reader. path must be a string (lru_cache requires hashable)."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, comment="#")


def load_output(run: str, scenario: str, key: str) -> pd.DataFrame:
    """
    Load an output CSV by its config key (e.g. 'capacity', 'energy').
    CSV[key] may be a list of candidate filenames; the first that exists wins.
    Returns empty DataFrame if no file is found.
    """
    candidates = CSV.get(key)
    if not candidates:
        return pd.DataFrame()
    if isinstance(candidates, str):
        candidates = [candidates]
    base = OUTPUT_ROOT / run / scenario / "output_csv"
    for filename in candidates:
        df = _load_csv(str(base / filename))
        if not df.empty:
            return df
    return pd.DataFrame()


def load_output_multi(run: str, scenarios: list[str], key: str) -> pd.DataFrame:
    """
    Load the same output CSV for multiple scenarios, adding a 'scenario' column.
    Useful for scenario-comparison charts.
    """
    frames = []
    for sc in scenarios:
        df = load_output(run, sc, key)
        if not df.empty:
            df = df.copy()
            df["scenario"] = sc
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Convenience loaders (typed, with consistent column names)
# ---------------------------------------------------------------------------

def get_capacity(run: str, scenarios: list[str]) -> pd.DataFrame:
    """pCapacityByFuel — columns: z, f, y, value, scenario"""
    return load_output_multi(run, scenarios, "capacity")


def get_energy(run: str, scenarios: list[str]) -> pd.DataFrame:
    """pEnergyByFuel — columns: z, f, y, value, scenario"""
    return load_output_multi(run, scenarios, "energy")


def get_emissions(run: str, scenarios: list[str]) -> pd.DataFrame:
    """pEmissions — columns: z, y, value, scenario"""
    return load_output_multi(run, scenarios, "emissions")


def get_emissions_intensity(run: str, scenarios: list[str]) -> pd.DataFrame:
    """pEmissionsIntensity — columns: z, y, value, scenario"""
    return load_output_multi(run, scenarios, "emissions_intensity")


def get_cost_summary(run: str, scenarios: list[str]) -> pd.DataFrame:
    """pCostSummary — columns: z, uni, y, value, scenario"""
    return load_output_multi(run, scenarios, "cost_summary")


def get_summary(run: str, scenario: str) -> pd.DataFrame:
    """pSummary scalar KPIs — columns: uni, value"""
    return load_output(run, scenario, "summary")


def get_demand_supply(run: str, scenarios: list[str]) -> pd.DataFrame:
    """pDemandSupply — columns: z, uni, y, value, scenario"""
    return load_output_multi(run, scenarios, "demand_supply")


def get_interchange(run: str, scenarios: list[str]) -> pd.DataFrame:
    """pInterchange — columns vary, scenario added"""
    return load_output_multi(run, scenarios, "interchange")


def get_new_capacity(run: str, scenarios: list[str]) -> pd.DataFrame:
    """pNewCapacityFuel — columns: z, f, y, value, scenario"""
    return load_output_multi(run, scenarios, "new_capacity")


def get_lcoe(run: str, scenarios: list[str]) -> pd.DataFrame:
    """pPlantAnnualLCOE — columns vary, scenario added"""
    return load_output_multi(run, scenarios, "lcoe")


def get_price(run: str, scenarios: list[str]) -> pd.DataFrame:
    """pPrice — columns vary, scenario added"""
    return load_output_multi(run, scenarios, "price")


def get_zcmap(run: str, scenario: str) -> pd.DataFrame:
    """zcmap — zone-to-country mapping"""
    return load_output(run, scenario, "zcmap")


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------

def get_zones(run: str, scenario: str) -> list[str]:
    """Return sorted unique zone names from capacity output."""
    df = load_output(run, scenario, "capacity")
    if df.empty or "z" not in df.columns:
        return []
    return sorted(df["z"].unique().tolist())


def get_years(run: str, scenario: str) -> list[int]:
    """Return sorted unique years from capacity output."""
    df = load_output(run, scenario, "capacity")
    if df.empty or "y" not in df.columns:
        return []
    return sorted(df["y"].astype(int).unique().tolist())


def get_fuels(run: str, scenario: str) -> list[str]:
    """Return sorted unique fuels from capacity output."""
    df = load_output(run, scenario, "capacity")
    if df.empty or "f" not in df.columns:
        return []
    return sorted(df["f"].unique().tolist())


def get_kpis(run: str, scenario: str) -> dict:
    """
    Extract headline KPIs from pSummary.csv.
    Returns dict with keys: npv, total_capacity, total_generation,
                             total_emissions, total_investment.
    """
    df = get_summary(run, scenario)
    if df.empty or "uni" not in df.columns:
        return {}
    kv = pd.to_numeric(df.set_index("uni")["value"], errors="coerce").to_dict()
    return {
        "npv":              kv.get("NPV of system cost: $m"),
        "total_capacity":   kv.get("Total Capacity Added: MW"),
        "total_generation": kv.get("Total Generation: GWh"),
        "total_emissions":  kv.get("Total Emission: mt"),
        "total_investment": kv.get("Total Investment Undiscounted: $m"),
        "total_demand":     kv.get("Total Demand: GWh"),
        "total_trade":      kv.get("Total Trade: GWh"),
        "total_use":        kv.get("Total USE: GWh"),
    }


def get_re_share(run: str, scenario: str, zones: Optional[list] = None) -> pd.DataFrame:
    """
    Compute renewable energy share (%) by zone and year.
    Returns DataFrame with columns: z, y, re_share_pct.
    """
    from config import RENEWABLE_FUELS
    df = load_output(run, scenario, "energy")
    if df.empty:
        return pd.DataFrame()
    if zones:
        df = df[df["z"].isin(zones)]
    df["is_re"] = df["f"].isin(RENEWABLE_FUELS)
    total = df.groupby(["z", "y"])["value"].sum().rename("total")
    re    = df[df["is_re"]].groupby(["z", "y"])["value"].sum().rename("re")
    result = pd.concat([total, re], axis=1).fillna(0)
    result["re_share_pct"] = (result["re"] / result["total"].replace(0, float("nan"))) * 100
    return result.reset_index()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Input scenario discovery  (scenarios.csv inside a data folder)
# ---------------------------------------------------------------------------

def load_input_scenarios(folder: str) -> dict:
    """
    Parse scenarios.csv from an input data folder.

    Format: paramNames | ScenarioA | ScenarioB ...
            pSettings  |           | pSettings_alt.csv
            pCarbonPrice|          | constraint/pCarbonPrice_high.csv

    Returns
    -------
    dict  {scenario_name: {param_name: override_path_or_None}}
    e.g.  {"baseline": {"pCarbonPrice": None},
           "high_carbon": {"pCarbonPrice": "constraint/pCarbonPrice_high.csv"}}
    """
    path = INPUT_ROOT / folder / "scenarios.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, header=0)
    except Exception:
        return {}

    # First column = paramNames, rest = scenario columns
    param_col = df.columns[0]
    scenario_cols = [c for c in df.columns[1:] if not str(c).startswith("Unnamed")]

    result = {}
    for sc in scenario_cols:
        overrides = {}
        for _, row in df.iterrows():
            param = str(row[param_col]).strip()
            val   = row.get(sc, "")
            if param and param not in ("", "nan") and not param.isupper():
                # isupper() → section headers like LOAD, SUPPLY etc — skip
                override = str(val).strip() if (val and str(val).strip() not in ("", "nan")) else None
                overrides[param] = override
        result[sc] = overrides
    return result


def get_input_scenario_names(folder: str) -> list[str]:
    """Return list of scenario names defined in scenarios.csv."""
    return list(load_input_scenarios(folder).keys())


def resolve_input_file(folder: str, key: str, scenario: str | None = None) -> Path | None:
    """
    Resolve the actual file path for an input key, respecting scenario overrides.

    Returns the Path to use, or None if not found.
    """
    spec = INPUT_CSV.get(key)
    if not spec:
        return None
    subfolder, filename = spec
    default_path = (INPUT_ROOT / folder / subfolder / filename
                    if subfolder else INPUT_ROOT / folder / filename)

    if not scenario:
        return default_path

    # Check for override
    scenarios = load_input_scenarios(folder)
    sc_overrides = scenarios.get(scenario, {})

    # Match by filename stem (e.g. "pCarbonPrice" matches key "carbon_price")
    stem = Path(filename).stem  # e.g. "pCarbonPrice"
    override_rel = sc_overrides.get(stem)
    if override_rel:
        override_path = INPUT_ROOT / folder / override_rel
        if override_path.exists():
            return override_path

    return default_path


def load_input_for_scenario(folder: str, key: str, scenario: str | None = None) -> pd.DataFrame:
    """
    Load an input CSV respecting scenario overrides.
    Falls back to default file if override not found.
    """
    path = resolve_input_file(folder, key, scenario)
    if path is None:
        return pd.DataFrame()
    return _load_input_csv(str(path))


def is_override(folder: str, key: str, scenario: str) -> bool:
    """Return True if this key has a scenario-specific override file."""
    spec = INPUT_CSV.get(key)
    if not spec:
        return False
    _, filename = spec
    stem = Path(filename).stem
    scenarios = load_input_scenarios(folder)
    return bool(scenarios.get(scenario, {}).get(stem))


# ---------------------------------------------------------------------------
# Input folder discovery
# ---------------------------------------------------------------------------

def list_input_folders() -> list[str]:
    """Return sorted list of input data folder names (e.g. data_test, data_eapp)."""
    if not INPUT_ROOT.exists():
        return []
    return sorted([
        d.name for d in INPUT_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("data_")
    ])


@lru_cache(maxsize=128)
def _load_input_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, comment="#")
    except Exception:
        return pd.DataFrame()


def load_input(folder: str, key: str) -> pd.DataFrame:
    """
    Load an input CSV by its config key (e.g. 'gen_data', 'demand_forecast').
    Returns empty DataFrame if file not found.
    """
    spec = INPUT_CSV.get(key)
    if not spec:
        return pd.DataFrame()
    subfolder, filename = spec
    if subfolder:
        path = INPUT_ROOT / folder / subfolder / filename
    else:
        path = INPUT_ROOT / folder / filename
    return _load_input_csv(str(path))


def save_input(folder: str, key: str, df: pd.DataFrame) -> bool:
    """
    Write a DataFrame back to its input CSV file.
    Returns True on success, False on failure.
    """
    spec = INPUT_CSV.get(key)
    if not spec:
        return False
    subfolder, filename = spec
    if subfolder:
        path = INPUT_ROOT / folder / subfolder / filename
    else:
        path = INPUT_ROOT / folder / filename
    try:
        df.to_csv(path, index=False)
        # Invalidate cache for this file
        _load_input_csv.cache_clear()
        return True
    except Exception:
        return False


def list_input_files(folder: str) -> dict[str, list[str]]:
    """
    Scan a data folder and return all CSV files grouped by subfolder.
    Returns {subfolder_name: [filename, ...]}
    """
    base = INPUT_ROOT / folder
    if not base.exists():
        return {}
    result = {}
    for item in sorted(base.rglob("*.csv")):
        rel = item.relative_to(base)
        subfolder = str(rel.parent) if rel.parent != Path(".") else "root"
        result.setdefault(subfolder, []).append(item.name)
    return result


# ---------------------------------------------------------------------------
# Variant system  (Baseline + named file variants per parameter)
# ---------------------------------------------------------------------------

def list_variants(folder: str, key: str) -> dict[str, str]:
    """
    Return {variant_name: filename} for all variants of a parameter.
    Always includes "Baseline" → default filename.
    Variant files follow: {stem}_{variant}.csv in the same subfolder.
    """
    spec = INPUT_CSV.get(key)
    if not spec:
        return {"Baseline": ""}
    subfolder, filename = spec
    stem = Path(filename).stem
    base_dir = (INPUT_ROOT / folder / subfolder) if subfolder else (INPUT_ROOT / folder)
    if not base_dir.exists():
        return {"Baseline": filename}
    variants: dict[str, str] = {"Baseline": filename}
    prefix_len = len(stem) + 1
    for f in sorted(base_dir.glob(f"{stem}_*.csv")):
        variant_name = f.stem[prefix_len:]
        if variant_name:
            variants[variant_name] = f.name
    return variants


def load_variant(folder: str, key: str, variant: str | None = None) -> pd.DataFrame:
    """Load a variant of a parameter file. None / 'Baseline' = default file."""
    spec = INPUT_CSV.get(key)
    if not spec:
        return pd.DataFrame()
    subfolder, filename = spec
    if variant and variant != "Baseline":
        filename = f"{Path(filename).stem}_{variant}.csv"
    path = (INPUT_ROOT / folder / subfolder / filename) if subfolder \
           else (INPUT_ROOT / folder / filename)
    return _load_input_csv(str(path))


def resolve_variant_path(folder: str, key: str, variant: str | None = None) -> str:
    """Return the absolute file path for a given input key + variant."""
    spec = INPUT_CSV.get(key)
    if not spec:
        return ""
    subfolder, filename = spec
    if variant and variant != "Baseline":
        filename = f"{Path(filename).stem}_{variant}.csv"
    if subfolder:
        path = INPUT_ROOT / folder / subfolder / filename
    else:
        path = INPUT_ROOT / folder / filename
    return str(path)


def save_variant(folder: str, key: str, variant: str | None, df: pd.DataFrame) -> bool:
    """Save DataFrame to a specific variant file."""
    spec = INPUT_CSV.get(key)
    if not spec:
        return False
    subfolder, filename = spec
    if variant and variant != "Baseline":
        filename = f"{Path(filename).stem}_{variant}.csv"
    path = (INPUT_ROOT / folder / subfolder / filename) if subfolder \
           else (INPUT_ROOT / folder / filename)
    try:
        df.to_csv(path, index=False)
        _load_input_csv.cache_clear()
        return True
    except Exception:
        return False


def duplicate_variant(folder: str, key: str, source_variant: str | None,
                      new_name: str) -> bool:
    """Copy source variant to new_name. Returns False if dest already exists."""
    import shutil as _shutil
    spec = INPUT_CSV.get(key)
    if not spec or not new_name.strip():
        return False
    subfolder, filename = spec
    stem = Path(filename).stem
    src_name = filename if (not source_variant or source_variant == "Baseline") \
               else f"{stem}_{source_variant}.csv"
    dst_name = f"{stem}_{new_name.strip()}.csv"
    base_dir = (INPUT_ROOT / folder / subfolder) if subfolder else (INPUT_ROOT / folder)
    src, dst = base_dir / src_name, base_dir / dst_name
    if not src.exists() or dst.exists():
        return False
    try:
        _shutil.copy2(src, dst)
        _load_input_csv.cache_clear()
        return True
    except Exception:
        return False


def variant_to_override_path(folder: str, key: str, variant: str) -> Optional[str]:
    """Convert variant name to relative path for scenarios.csv. None if Baseline."""
    if not variant or variant == "Baseline":
        return None
    spec = INPUT_CSV.get(key)
    if not spec:
        return None
    subfolder, filename = spec
    fname = f"{Path(filename).stem}_{variant}.csv"
    return f"{subfolder}/{fname}" if subfolder else fname


def override_path_to_variant(folder: str, key: str, override_path) -> str:
    """Convert a scenarios.csv override path to a variant name. 'Baseline' if empty."""
    if not override_path or str(override_path).strip() in ("", "nan"):
        return "Baseline"
    spec = INPUT_CSV.get(key)
    if not spec:
        return "Baseline"
    stem = Path(spec[1]).stem
    fname = Path(str(override_path)).name
    prefix = f"{stem}_"
    if fname.startswith(prefix) and fname.endswith(".csv"):
        return fname[len(prefix):-4]
    return "Baseline"


def add_scenario_column(folder: str, scenario_name: str) -> bool:
    """Add a new empty scenario column to scenarios.csv (creates file if missing)."""
    path = INPUT_ROOT / folder / "scenarios.csv"
    try:
        if path.exists():
            df = pd.read_csv(path, dtype=str).fillna("")
            if scenario_name in df.columns:
                return False  # already exists
            df[scenario_name] = ""
        else:
            param_names, seen = [], set()
            for _key, (_sub, _fname) in INPUT_CSV.items():
                gams = Path(_fname).stem
                if gams not in seen:
                    param_names.append(gams)
                    seen.add(gams)
            df = pd.DataFrame({"paramNames": param_names, scenario_name: ""})
        df.to_csv(path, index=False)
        _load_input_csv.cache_clear()
        return True
    except Exception:
        return False


def save_input_scenarios(folder: str, sc_key_variant_dict: dict) -> bool:
    """
    Update scenarios.csv with {scenario_name: {input_key: variant_name}}.
    Preserves existing rows/structure not in our mapping.
    """
    path = INPUT_ROOT / folder / "scenarios.csv"
    try:
        df = pd.read_csv(path, dtype=str).fillna("") if path.exists() \
             else pd.DataFrame({"paramNames": []})
        param_col = df.columns[0] if len(df.columns) else "paramNames"

        sc_names = list(sc_key_variant_dict.keys())
        for sc in sc_names:
            if sc not in df.columns:
                df[sc] = ""

        for sc_name, key_variants in sc_key_variant_dict.items():
            for key, variant in key_variants.items():
                spec = INPUT_CSV.get(key)
                if not spec:
                    continue
                gams_name = Path(spec[1]).stem
                override = variant_to_override_path(folder, key, variant) or ""
                mask = df[param_col] == gams_name
                if mask.any():
                    df.loc[mask, sc_name] = override

        df.to_csv(path, index=False)
        _load_input_csv.cache_clear()
        return True
    except Exception:
        return False


def clear_input_cache() -> None:
    """Clear the lru_cache for input CSV files (call before a forced reload)."""
    _load_input_csv.cache_clear()


def clone_input_folder(source: str, target_name: str) -> bool:
    """
    Copy an existing data folder to a new one with target_name.
    Returns True on success.
    """
    import shutil
    src = INPUT_ROOT / source
    dst = INPUT_ROOT / target_name
    if not src.exists() or dst.exists():
        return False
    try:
        shutil.copytree(src, dst)
        return True
    except Exception:
        return False


# ===========================================================================
# Results section — merged-output loaders (generated by output_treatment.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# Constants (colours, ordering, indicator metadata)
# ---------------------------------------------------------------------------

TECH_ORDER = [
    "Import", "Nuclear", "Coal", "Peat", "Diesel", "Gas", "CCGT", "OCGT",
    "Methane", "Waste", "Biomass", "Geothermal", "Reservoir", "ROR",
    "CSP", "Solar Thermal", "Solar", "Onshore Wind", "Offshore Wind",
    "PV", "PV+Storage", "Battery", "PSH",
]

TECH_COLORS = {
    "Reservoir":       "#1a6faf",
    "ROR":             "#5fa8d3",
    "PV":              "#f4c430",
    "PV+Storage":      "#e8971a",
    "Onshore Wind":    "#2d9e4f",
    "Offshore Wind":   "#7ec8a0",
    "Battery":         "#7b4f9e",
    "PSH":             "#b09bc8",
    "Solar Thermal":   "#f97b22",
    "Solar":           "#f4c430",
    "CSP":             "#f4a261",
    "Gas":             "#f77f00",
    "CCGT":            "#f77f00",
    "OCGT":            "#fcb777",
    "Coal":            "#5c4033",
    "Nuclear":         "#c1440e",
    "Biomass":         "#7a9e3b",
    "Waste":           "#9e8e6e",
    "Peat":            "#8b7355",
    "Methane":         "#d4a017",
    "Diesel":          "#d62728",
    "Import":          "#95afc0",
    "Imports":         "#95afc0",
    "Geothermal":      "#16a085",
    "Generation":      "#2c6fad",
    "StorageEnergy":   "#7b4f9e",
    # Cost categories
    "Fuel costs: $m":                             "#d62728",
    "Fixed O&M: $m":                              "#1a6faf",
    "Variable O&M: $m":                           "#5fa8d3",
    "Import costs with internal zones: $m":       "#95afc0",
    "Import costs with external zones: $m":       "#7f7f7f",
    "Export revenues with internal zones: $m":    "#2d9e4f",
    "Export revenues with external zones: $m":    "#7ec8a0",
    "Trade shared benefits: $m":                  "#27ae60",
    "Unmet demand costs: $m":                     "#e74c3c",
    "Unmet country planning reserve costs: $m":   "#fcb777",
    "Unmet country spinning reserve costs: $m":   "#f77f00",
    "Carbon costs: $m":                           "#c1440e",
    "Spinning reserve costs: $m":                 "#b09bc8",
    "Transmission costs: $m":                     "#7a9e3b",
    "Startup costs: $m":                          "#bdc3c7",
}

INDICATOR_OPTIONS = [
    {"label": "Capacity (MW)",                "value": "CapacityTechFuel"},
    {"label": "Energy Generation (GWh)",      "value": "EnergyTechFuelComplete"},
    {"label": "New Capacity (MW)",            "value": "NewCapacityTechFuel"},
    {"label": "New Capacity Cumulative (MW)", "value": "NewCapacityTechFuelCumulated"},
    {"label": "Costs (m USD)",                "value": "Costs"},
    {"label": "Generation Costs (USD/MWh)",   "value": "CostsPerMWh"},
    {"label": "CAPEX (m USD)",                "value": "CapexInvestmentComponent"},
    {"label": "CAPEX Cumulative (m USD)",     "value": "CapexInvestmentComponentCumulated"},
    {"label": "Spinning Reserve (GWh)",       "value": "ReserveSpinningTechFuel"},
]

LINE_INDICATOR_OPTIONS = [
    {"label": "None",                          "value": ""},
    {"label": "Emissions (MtCO2)",             "value": "EmissionsZone"},
    {"label": "Emission Intensity (tCO2/GWh)", "value": "EmissionsIntensityZone"},
    {"label": "Demand (GWh)",                  "value": "DemandEnergyZone"},
    {"label": "Peak Demand (MW)",              "value": "DemandPeakZone"},
    {"label": "Gen Cost (USD/MWh)",            "value": "GenCostsPerMWh"},
]

INDICATOR_LABELS = {o["value"]: o["label"] for o in INDICATOR_OPTIONS}
INDICATOR_LABELS.update({o["value"]: o["label"] for o in LINE_INDICATOR_OPTIONS})

# source: 'techfuel' | 'costs' | 'capex'
INDICATOR_SOURCE = {
    "CapacityTechFuel":                  ("techfuel", "techfuel"),
    "EnergyTechFuelComplete":            ("techfuel", "techfuel"),
    "NewCapacityTechFuel":               ("techfuel", "techfuel"),
    "NewCapacityTechFuelCumulated":      ("techfuel", "techfuel"),
    "ReserveSpinningTechFuel":           ("techfuel", "techfuel"),
    "Costs":                             ("costs",    "uni"),
    "CostsPerMWh":                       ("costs",    "uni"),
    "CapexInvestmentComponent":          ("capex",    "uni"),
    "CapexInvestmentComponentCumulated": ("capex",    "uni"),
}

PLANT_INDICATOR_OPTIONS = [
    {"label": "Capacity (MW)", "value": "CapacityPlant"},
    {"label": "Energy (GWh)",  "value": "EnergyPlant"},
    {"label": "Costs (m USD)", "value": "CostsPlant"},
]


# ---------------------------------------------------------------------------
# Core merged-CSV loader
# ---------------------------------------------------------------------------

def load_merged(run: str, filename: str,
                scenarios: list[str] | None = None) -> pd.DataFrame:
    """
    Load a *Merged.csv produced by output_treatment.py from each scenario
    inside *run*, concatenating results with a 'scenario' column.

    Returns an empty DataFrame with a 'no_merged_data' attribute set to True
    if no files are found (caller can detect wrong/old output format).
    """
    if not run:
        df = pd.DataFrame()
        df.attrs["no_merged_data"] = True
        return df

    run_path = OUTPUT_ROOT / run
    targets = scenarios if scenarios else list_scenarios(run)
    dfs = []
    for sc in targets:
        p = run_path / sc / "output_csv" / filename
        if p.exists():
            try:
                df = pd.read_csv(p, low_memory=False)
                if "value" in df.columns:
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df["scenario"] = sc
                dfs.append(df)
            except Exception:
                pass

    if not dfs:
        result = pd.DataFrame()
        result.attrs["no_merged_data"] = True
        return result
    return pd.concat(dfs, ignore_index=True)


def _merged_data_missing(df: pd.DataFrame) -> bool:
    return df.empty and df.attrs.get("no_merged_data", False)


# ---------------------------------------------------------------------------
# Typed merged loaders
# ---------------------------------------------------------------------------

def load_techfuel(run: str, scenarios: list[str] | None = None) -> pd.DataFrame:
    """pTechFuelMerged — capacity, energy, new-cap, reserve by tech/fuel."""
    return load_merged(run, "pTechFuelMerged.csv", scenarios)


def load_costs_merged(run: str, scenarios: list[str] | None = None) -> pd.DataFrame:
    """pCostsMerged — cost components by zone/year."""
    return load_merged(run, "pCostsMerged.csv", scenarios)


def load_capex_merged(run: str, scenarios: list[str] | None = None) -> pd.DataFrame:
    """pCapexInvestmentMerged — CAPEX components."""
    return load_merged(run, "pCapexInvestmentMerged.csv", scenarios)


def load_yearly_zone(run: str, scenarios: list[str] | None = None) -> pd.DataFrame:
    """pYearlyZoneMerged — demand, emissions, prices by zone/year."""
    df = load_merged(run, "pYearlyZoneMerged.csv", scenarios)
    if not df.empty and "value" in df.columns:
        df = df.dropna(subset=["value"])
    return df


def load_transmission_merged(run: str,
                             scenarios: list[str] | None = None) -> pd.DataFrame:
    """pTransmissionMerged — interchange, utilization, capacity by corridor."""
    df = load_merged(run, "pTransmissionMerged.csv", scenarios)
    if df.empty:
        return df
    # Normalise counterpart-zone column to 'z2'
    import numpy as np
    bilateral = {"Interchange", "InterconUtilization", "NetImport",
                 "TransmissionCapacity", "NewTransmissionCapacity", "CongestionShare"}
    if "uni.1" in df.columns:
        if "z2" not in df.columns:
            df["z2"] = df["uni.1"]
        else:
            df["z2"] = df["z2"].fillna(df["uni.1"])
        df = df.drop(columns=["uni.1"])
    if "z2" not in df.columns:
        mask = df["attribute"].isin(bilateral)
        df["z2"] = np.where(mask, df["uni"], np.nan)
    return df


def load_plants_merged(run: str,
                       scenarios: list[str] | None = None) -> pd.DataFrame:
    """pPlantMerged — plant-level attributes."""
    return load_merged(run, "pPlantMerged.csv", scenarios)


def load_npv_merged(run: str,
                    scenarios: list[str] | None = None) -> pd.DataFrame:
    """pNetPresentCostSystemMerged — system NPV."""
    return load_merged(run, "pNetPresentCostSystemMerged.csv", scenarios)


def load_dispatch_merged(run: str,
                         scenarios: list[str] | None = None) -> pd.DataFrame:
    """pDispatchComplete — hourly dispatch (heavy, no cache)."""
    return load_merged(run, "pDispatchComplete.csv", scenarios)


def load_hourly_price_merged(run: str,
                             scenarios: list[str] | None = None) -> pd.DataFrame:
    """pHourlyPrice — hourly marginal cost."""
    return load_merged(run, "pHourlyPrice.csv", scenarios)


# ---------------------------------------------------------------------------
# Zone coordinate loader (linestring_countries.geojson at run root)
# ---------------------------------------------------------------------------

def load_zone_coords(run: str) -> dict:
    """Return {zone: (lat, lon)} from linestring_countries.geojson at run root."""
    import json
    path = OUTPUT_ROOT / run / "linestring_countries.geojson"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    coords = {}
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        z = props.get("z")
        if z and z not in coords:
            lat = props.get("country_ini_lat") or props.get("lat")
            lon = props.get("country_ini_lon") or props.get("lon")
            if lat is not None and lon is not None:
                coords[z] = (lat, lon)
    return coords


# ---------------------------------------------------------------------------
# pHours loader (representative day weights)
# ---------------------------------------------------------------------------

def load_phours_merged(run: str) -> dict:
    """Return {(q, d): weight_pct} from first available scenario's pHours.csv."""
    for sc in list_scenarios(run):
        p = OUTPUT_ROOT / run / sc / "input" / "pHours.csv"
        if not p.exists():
            # also try output_csv
            p = OUTPUT_ROOT / run / sc / "output_csv" / "pHours.csv"
        if p.exists():
            try:
                df = pd.read_csv(p)
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                unique_qd = df.groupby(["q", "d"])["value"].first()
                total = unique_qd.sum()
                if total > 0:
                    return {(q, d): v / total * 100 for (q, d), v in unique_qd.items()}
            except Exception:
                pass
    return {}


# ---------------------------------------------------------------------------
# Discovery helpers for merged data
# ---------------------------------------------------------------------------

def get_merged_years(run: str) -> list[int]:
    """Return sorted unique years from pTechFuelMerged across all scenarios."""
    df = load_techfuel(run)
    if df.empty or "y" not in df.columns:
        return []
    return sorted(pd.to_numeric(df["y"], errors="coerce").dropna().astype(int).unique().tolist())


def get_merged_zones(run: str) -> list[str]:
    """Return sorted unique zones from pTechFuelMerged."""
    df = load_techfuel(run)
    if df.empty or "z" not in df.columns:
        return []
    return sorted(df["z"].dropna().unique().tolist())


def get_merged_countries(run: str) -> list[str]:
    """Return sorted unique countries from pTechFuelMerged."""
    df = load_techfuel(run)
    if df.empty or "c" not in df.columns:
        return []
    return sorted(df["c"].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# Shared computation helpers
# ---------------------------------------------------------------------------

def get_color_sequence(categories: list) -> list:
    """Return ordered colour list matching TECH_COLORS for given categories."""
    return [TECH_COLORS.get(c, "#aaaaaa") for c in categories]


def apply_view_mode(df: pd.DataFrame, view: str, ref_scenario: str,
                    group_keys: list) -> pd.DataFrame:
    """
    Apply Absolute / Difference / Percentage transformation.
    df must have 'scenario' and 'value' columns.
    group_keys = non-scenario columns to join on.
    """
    import numpy as np
    if view == "Absolute" or not ref_scenario:
        return df
    df_ref = (
        df[df["scenario"] == ref_scenario]
        .copy()
        .rename(columns={"value": "ref_value"})
        .drop(columns=["scenario"])
    )
    df_out = df.merge(df_ref, on=group_keys, how="left")
    df_out["ref_value"] = df_out["ref_value"].fillna(0)
    if view == "Difference":
        df_out["value"] = df_out["value"] - df_out["ref_value"]
    elif view == "Percentage":
        df_out["value"] = np.where(
            df_out["ref_value"] != 0,
            (df_out["value"] - df_out["ref_value"]) / df_out["ref_value"].abs() * 100,
            0,
        )
    return df_out.drop(columns=["ref_value"])


def no_data_alert(msg: str = None):
    """Standard 'no merged data' message for result pages."""
    import dash_bootstrap_components as dbc
    text = msg or (
        "No merged output files found for this run. "
        "Please re-run the model (output_treatment.py generates the required files: "
        "pTechFuelMerged.csv, pCostsMerged.csv, etc.)."
    )
    return dbc.Alert(text, color="warning", className="mt-3")
