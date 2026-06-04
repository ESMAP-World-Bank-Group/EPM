"""
compute_epm_gendata.py — Generic EPM pGenDataInput builder from GEM/GIPT data.

Reads GIPT (Global Integrated Power Tracker) via gem_pipeline, applies the
mapping rules in config/epm_gendata_config.yaml, and produces rows for
epm/input/<deployment>/supply/pGenDataInput.csv.

Usage:
    python compute_epm_gendata.py --country AZE
    python compute_epm_gendata.py --country AZE ARM GEO
    python compute_epm_gendata.py --all
    python compute_epm_gendata.py --country AZE --dry-run
    python compute_epm_gendata.py --country AZE --deployment data_blacksea --append

Options:
    --country ISO3 [ISO3 ...]  Countries to process (e.g. AZE ARM GEO)
    --all                      Process all countries in config
    --deployment NAME          EPM deployment folder (default: data_blacksea)
    --append                   Append rows to existing pGenDataInput.csv
    --dry-run                  Print rows without writing files
    --force-download           Re-download GIPT even if cached
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────

_BASE_DIR   = Path(__file__).resolve().parent
_EPM_DIR    = _BASE_DIR.parent / "epm"
_CONFIG_DIR = _BASE_DIR / "config"
_OUTPUT_DIR = _BASE_DIR / "output_gendata"

sys.path.insert(0, str(_BASE_DIR))
from pipelines.gem_pipeline import load_gipt_plants


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(path: Path = _CONFIG_DIR / "epm_gendata_config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Fuel / tech mapping ───────────────────────────────────────────────────────

def resolve_tech_fuel(gem_fuel: str, mw: float | None, year: int | None,
                      fuel_map: dict, country_iso: str,
                      fuel_overrides: dict) -> tuple[str, str]:
    """Return (EPM_tech, EPM_fuel) for a given GEM plant."""
    # Check country-level fuel override
    country_overrides = fuel_overrides.get(country_iso, {}) or {}
    fuel_override = country_overrides.get(gem_fuel)

    rules = fuel_map.get(gem_fuel, [])
    for rule in rules:
        # Year-based rules
        if "year_before" in rule and year is not None and year < rule["year_before"]:
            tech = rule["tech"]
            fuel = fuel_override or rule["fuel"]
            return tech, fuel
        if "year_from" in rule and year is not None and year >= rule["year_from"]:
            tech = rule["tech"]
            fuel = fuel_override or rule["fuel"]
            return tech, fuel
        # Size-based rule
        if "mw_below" in rule and mw is not None and mw < rule["mw_below"]:
            tech = rule["tech"]
            fuel = fuel_override or rule["fuel"]
            return tech, fuel
        # Default rule
        if rule.get("default"):
            tech = rule["tech"]
            fuel = fuel_override or rule["fuel"]
            return tech, fuel

    # Fallback
    return "ST", gem_fuel.capitalize()


def compute_retr_yr(st_yr: int | None, tech: str, life_by_tech: dict) -> int | None:
    if st_yr is None:
        return None
    life = life_by_tech.get(tech, 30)
    if life >= 99:
        return None
    return st_yr + life


# ── Generator name builder ────────────────────────────────────────────────────

_STRIP_WORDS = re.compile(
    r"\b(power station|power plant|hydroelectric plant|hydroelectric|"
    r"solar farm|solar park|wind farm|HPP|TPP|CHP|GRES|SDRES)\b",
    re.I,
)

def make_gen_name(zone: str, plant_name: str, tech: str, counter: dict) -> str:
    """Build a clean EPM generator name: Zone_PlantName_Tech (≤55 chars)."""
    clean = _STRIP_WORDS.sub("", plant_name)
    clean = re.sub(r"[^a-zA-Z0-9 _-]", "", clean).strip()
    clean = re.sub(r"\s+", "_", clean).strip("_")[:20]
    if not clean:
        clean = tech
    base = f"{zone}_{clean}_{tech}"
    # Deduplicate
    if base not in counter:
        counter[base] = 0
        return base
    counter[base] += 1
    return f"{base}_{counter[base]}"


# ── Core transformation ───────────────────────────────────────────────────────

def plants_to_epm_rows(
    df: pd.DataFrame,
    country_iso: str,
    zone: str,
    cfg: dict,
) -> pd.DataFrame:
    """Convert a DataFrame of GEM plants to EPM pGenDataInput rows."""

    fuel_map         = cfg["fuel_map"]
    life_by_tech     = cfg["life_by_tech"]
    status_map       = cfg["status_map"]
    agg_threshold    = cfg.get("aggregate_below_mw", 10)
    capex_by_tech    = cfg.get("capex_by_tech", {})
    fuel_overrides   = cfg.get("fuel_overrides", {})
    cand_defaults    = cfg.get("candidate_defaults", {})
    default_styr     = cand_defaults.get("StYr_if_unknown", 2030)
    blpy_frac        = cand_defaults.get("BuildLimitperYear_fraction", 0.2)
    model_start_year = cfg.get("model_start_year", 2025)

    rows = []
    agg_buckets: dict[tuple, list] = {}
    counter: dict[str, int] = {}

    for _, p in df.iterrows():
        gem_fuel = p.get("fuel", "")
        mw       = p.get("mw")
        year     = p.get("year")
        status   = p.get("status", "operating")
        name     = p.get("name", "")

        tech, epm_fuel = resolve_tech_fuel(
            gem_fuel, mw, year, fuel_map, country_iso, fuel_overrides
        )
        epm_status = status_map.get(status, 1)

        # Small plants → aggregate bucket
        if mw is not None and mw < agg_threshold:
            key = (tech, epm_fuel, epm_status)
            agg_buckets.setdefault(key, []).append(mw or 0)
            continue

        # Individual plant row — handle NaN year (pandas float NaN is truthy!)
        import math as _math
        year_clean = None if (year is None or (isinstance(year, float) and _math.isnan(year))) else int(year)
        if year_clean is not None and year_clean <= 2025:
            st_yr = year_clean
        elif year_clean is not None:
            st_yr = year_clean           # future commissioning year
        elif epm_status == 3:
            st_yr = default_styr         # candidate with unknown year
        else:
            st_yr = None                 # existing plant with no year data
        retr_yr = compute_retr_yr(st_yr, tech, life_by_tech)

        # Skip plants already retired before model start
        if retr_yr is not None and retr_yr < model_start_year:
            continue

        capacity = round(mw, 1) if mw else ""
        build_lim = (
            round(capacity * blpy_frac, 1) if epm_status == 3 and capacity else ""
        )
        capex = capex_by_tech.get(tech, "") if epm_status in (2, 3) else ""

        g = make_gen_name(zone, name, tech, counter)

        rows.append({
            "g":                g,
            "z":                zone,
            "tech":             tech,
            "f":                epm_fuel,
            "Status":           epm_status,
            "StYr":             st_yr or "",
            "RetrYr":           retr_yr or "",
            "Capacity":         capacity,
            "BuildLimitperYear": build_lim,
            "Life":             "",
            "HeatRate":         "",
            "RampUpRate":       "",
            "RampDnRate":       "",
            "ResLimShare":      "",
            "Capex":            capex,
            "FOMperMW":         "",
            "VOM":              "",
            "ReserveCost":      "",
            "UnitSize":         "",
        })

    # Flush aggregated small plants
    for (tech, epm_fuel, epm_status), mw_list in agg_buckets.items():
        total_mw = round(sum(mw_list), 1)
        agg_name = f"{zone}_AGG_Small{tech}"
        capex = capex_by_tech.get(tech, "") if epm_status in (2, 3) else ""
        rows.append({
            "g":                agg_name,
            "z":                zone,
            "tech":             tech,
            "f":                epm_fuel,
            "Status":           epm_status,
            "StYr":             "",
            "RetrYr":           "",
            "Capacity":         total_mw,
            "BuildLimitperYear": round(total_mw * blpy_frac, 1) if epm_status == 3 else "",
            "Life":             "",
            "HeatRate":         "",
            "RampUpRate":       "",
            "RampDnRate":       "",
            "ResLimShare":      "",
            "Capex":            capex,
            "FOMperMW":         "",
            "VOM":              "",
            "ReserveCost":      "",
            "UnitSize":         "",
        })

    col_order = [
        "g", "z", "tech", "f", "Status", "StYr", "RetrYr", "Capacity",
        "BuildLimitperYear", "Life", "HeatRate", "RampUpRate", "RampDnRate",
        "ResLimShare", "Capex", "FOMperMW", "VOM", "ReserveCost", "UnitSize",
    ]
    return pd.DataFrame(rows, columns=col_order)


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(result: pd.DataFrame, iso: str):
    print(f"\n{'='*55}")
    print(f"  {iso} — {len(result)} generators")
    print(f"{'='*55}")
    summary = (
        result.groupby(["tech", "f", "Status"])["Capacity"]
        .apply(lambda x: pd.to_numeric(x, errors="coerce").sum())
        .reset_index()
    )
    summary.columns = ["tech", "fuel", "Status", "Total_MW"]
    summary["Status"] = summary["Status"].map({1: "Existing", 2: "Committed", 3: "Candidate"})
    print(summary.to_string(index=False))


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build pGenDataInput rows from GEM/GIPT")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--country", nargs="+", metavar="ISO3",
                     help="ISO-3 country codes (e.g. AZE ARM GEO)")
    grp.add_argument("--all", action="store_true",
                     help="Process all countries defined in config")
    parser.add_argument("--deployment", default="data_blacksea",
                        help="EPM deployment folder (default: data_blacksea)")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing pGenDataInput.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print rows without writing files")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download GIPT even if cached")
    parser.add_argument("--config", default=str(_CONFIG_DIR / "epm_gendata_config.yaml"),
                        help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    country_defs = cfg["countries"]
    if args.all:
        target_isos = list(country_defs.keys())
    else:
        target_isos = args.country
        for iso in target_isos:
            if iso not in country_defs:
                print(f"WARNING: {iso} not defined in config countries — skipping")
        target_isos = [iso for iso in target_isos if iso in country_defs]

    if not target_isos:
        print("No valid countries to process.")
        sys.exit(1)

    # Load GIPT plants for all target countries at once
    print(f"\n[gendata] Loading GIPT for: {target_isos}")
    df_all = load_gipt_plants(
        countries=target_isos,
        force=args.force_download,
        verbose=True,
    )

    all_results = []
    for iso in target_isos:
        zone    = country_defs[iso]["zone"]
        df_cty  = df_all[df_all["country"] == iso].copy()
        print(f"\n[gendata] Processing {iso} ({zone}) — {len(df_cty)} plants")

        result = plants_to_epm_rows(df_cty, iso, zone, cfg)
        print_report(result, iso)
        all_results.append(result)

    final = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    if args.dry_run:
        print("\n[gendata] DRY RUN — not writing files.")
        print(final.to_string(index=False))
        return

    # Save output
    _OUTPUT_DIR.mkdir(exist_ok=True)
    out_csv = _OUTPUT_DIR / f"pGenDataInput_{'_'.join(target_isos)}.csv"
    final.to_csv(out_csv, index=False)
    print(f"\n[gendata] Written: {out_csv}  ({len(final)} rows)")

    # Optionally append to EPM deployment CSV
    if args.append:
        target_csv = _EPM_DIR / "input" / args.deployment / "supply" / "pGenDataInput.csv"
        if not target_csv.exists():
            print(f"  WARNING: {target_csv} not found — skipping append.")
        else:
            existing = pd.read_csv(target_csv)
            # Remove any existing rows for these zones
            zones_to_replace = [country_defs[iso]["zone"] for iso in target_isos]
            existing = existing[~existing["z"].isin(zones_to_replace)]
            updated = pd.concat([existing, final], ignore_index=True)
            updated.to_csv(target_csv, index=False)
            print(f"  Appended to {target_csv}  ({len(updated)} total rows)")

    print("\n[gendata] Done.")


if __name__ == "__main__":
    main()
