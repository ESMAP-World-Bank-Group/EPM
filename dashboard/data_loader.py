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
    Returns empty DataFrame if file not found.
    """
    filename = CSV.get(key)
    if not filename:
        return pd.DataFrame()
    path = OUTPUT_ROOT / run / scenario / "output_csv" / filename
    return _load_csv(str(path))


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
    if df.empty:
        return {}
    kv = df.set_index("uni")["value"].to_dict()
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
