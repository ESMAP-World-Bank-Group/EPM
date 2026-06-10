"""
integrate_romania.py
Appends Romania rows from data_romania/ into data_blacksea/ for:
  - supply/pGenDataInput.csv
  - supply/pFuelPrice.csv
  - supply/pAvailabilityCustom.csv
  - zcmap.csv

Run after prepare_romania_for_blacksea.py has produced *_blacksea.csv files.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
RO_DIR   = ROOT / "epm/input/data_romania"
BS_DIR   = ROOT / "epm/input/data_blacksea"


def replace_zone_rows(
    target: Path,
    source: pd.DataFrame,
    zone_col: str,
    zone_val: str,
    label: str,
) -> None:
    """Read target CSV, remove existing rows for zone_val, append source rows."""
    df = pd.read_csv(target, dtype=str)
    before = len(df)
    df = df[df[zone_col] != zone_val].copy()
    df = pd.concat([df, source], ignore_index=True)
    df.to_csv(target, index=False)
    print(f"  [{label}] {before}→{len(df)} rows  ({target.relative_to(ROOT)})")


def replace_gen_rows(
    target: Path,
    source: pd.DataFrame,
    zone_col: str = "z",
    zone_val: str = "Romania",
    label: str = "pGenDataInput",
) -> None:
    replace_zone_rows(target, source, zone_col, zone_val, label)


# ── pGenDataInput ──────────────────────────────────────────────────────────────
def integrate_gendata() -> None:
    src = pd.read_csv(RO_DIR / "supply/pGenDataInput_blacksea.csv", dtype=str)
    tgt = BS_DIR / "supply/pGenDataInput.csv"
    replace_zone_rows(tgt, src, "z", "Romania", "pGenDataInput")


# ── pFuelPrice ─────────────────────────────────────────────────────────────────
def integrate_fuelprice() -> None:
    src = pd.read_csv(RO_DIR / "supply/pFuelPrice_blacksea.csv", dtype=str)
    tgt = BS_DIR / "supply/pFuelPrice.csv"
    replace_zone_rows(tgt, src, "country", "Romania", "pFuelPrice")


# ── pAvailabilityCustom ────────────────────────────────────────────────────────
def integrate_availability() -> None:
    src     = pd.read_csv(RO_DIR / "supply/pAvailabilityCustom.csv", dtype=str)
    tgt     = BS_DIR / "supply/pAvailabilityCustom.csv"
    # All gen names are globally unique (Romania-specific names); just append,
    # removing any stale Romania entries first by matching a known Romania gen name.
    # Simpler: read existing, find gen names that are in src, remove, re-append.
    df_tgt  = pd.read_csv(tgt, dtype=str)
    ro_gens = set(src["gen"].tolist())
    before  = len(df_tgt)
    df_tgt  = df_tgt[~df_tgt["gen"].isin(ro_gens)].copy()
    df_out  = pd.concat([df_tgt, src], ignore_index=True)
    df_out.to_csv(tgt, index=False)
    print(f"  [pAvailabilityCustom] {before}→{len(df_out)} rows  ({tgt.relative_to(ROOT)})")


# ── zcmap ──────────────────────────────────────────────────────────────────────
def integrate_zcmap() -> None:
    tgt = BS_DIR / "zcmap.csv"
    df  = pd.read_csv(tgt, dtype=str)
    # Remove any existing Romania rows then add fresh
    df  = df[df["z"] != "Romania"].copy()
    ro  = pd.DataFrame([{"z": "Romania", "c": "Romania"}])
    df  = pd.concat([df, ro], ignore_index=True)
    df.to_csv(tgt, index=False)
    print(f"  [zcmap] {len(df)} rows total  ({tgt.relative_to(ROOT)})")


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Integrating Romania into data_blacksea ===")
    integrate_gendata()
    integrate_fuelprice()
    integrate_availability()
    integrate_zcmap()
    print("=== Done ===")


if __name__ == "__main__":
    main()
