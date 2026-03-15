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
