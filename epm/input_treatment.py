"""
**********************************************************************
* ELECTRICITY PLANNING MODEL (EPM)
* Developed at the World Bank
**********************************************************************
Description:
    This Python script is part of the GAMS-based Electricity Planning Model (EPM),
    designed for electricity system planning. It supports tasks such as capacity
    expansion, generation dispatch, and the enforcement of policy constraints,
    including renewable energy targets and emissions limits.

Author(s):
    ESMAP Modelling Team

Organization:
    World Bank

Version:
    (Specify version here)

License:
    Creative Commons Zero v1.0 Universal

Key Features:
    - Optimization of electricity generation and capacity planning
    - Inclusion of renewable energy integration and storage technologies
    - Multi-period, multi-region modeling framework
    - CO₂ emissions constraints and policy instruments

Notes:
    - Ensure GAMS is installed and the model has completed execution
      before running this script.
    - The model generates output files in the working directory
      which will be organized by this script.

Contact:
    Claire Nicolas — cnicolas@worldbank.org
**********************************************************************
"""

import os
from pathlib import Path

import gams.transfer as gt
import numpy as np
import pandas as pd
from types import SimpleNamespace

YEARLY_OUTPUT = [
    'pTradePrice',
    'pDemandForecast',
    'pCapexTrajectories',
    'pExtTransferLimit',
    'pTransferLimit', 
    'pFuelPrice'
]

ZONE_RESTRICTED_PARAMS = {
    "pGenDataInput": ("z",),
    "pGenDataInputDefault": ("z",),
    "pStorageDataInputDefault": ("z",),
    "pAvailabilityDefault": ("z",),
    "pCapexTrajectoriesDefault": ("z",),
    "pDemandForecast": ("z",),
    "pNewTransmission": ("z", "z2"),
    "pLossFactorInternal": ("z", "z2"),
    "pTransferLimit": ("z", "z2"),
}

COLUMN_RENAME_MAP = {
    "pGenDataInput": {"uni": "pGenDataInputHeader", 'fuel': 'f'},
    "pGenDataInputDefault": {"uni": "pGenDataInputHeader", 'fuel': 'f'},
    "pGenDataInputGeneric": {"uni": "pGenDataInputHeader", 'fuel': 'f'},
    "pAvailabilityInput": {"uni": "q"},
    "pAvailability": {"uni": "q"},
    "pAvailabilityDefault": {"uni": "q"},
    "pAvailabilityGeneric": {'fuel': 'f'},
    "pEvolutionAvailability": {"uni": "y"},
    "pCapexTrajectoriesDefault": {"uni": "y"},
    "pCapexTrajectoriesGeneric": {"uni": "y", 'fuel': 'f'},
    "pDemandForecast": {"uni": "y"},
    "pNewTransmission": {"From": "z", "To": "z2", "uni": "pTransmissionHeader"},
    "pSettings": {"Abbreviation": "pSettingsHeader"},
    "pDemandForecast": {'type': 'pe', 'uni': 'y'},
    "pTransferLimit": {"From": "z", "To": "z2", "uni": "y"},
    "pHours": {'uni': 't'},
    "pLossFactorInternal": {"zone1": "z", "zone2": "z2", "uni": "y"},
    'pPlanningReserveMargin': {'uni': 'c'},
    'pTechFuel': {'tech': 'tech', 'fuel': 'f'},
    "pStorageDataInput": {'gen_0': 'g', 'uni_2': 'pStorageDataHeader'},
    "pStorageDataInputDefault": {"uni": "pStorageDataHeader", 'fuel': 'f'},
    "pStorageDataInputGeneric": {"uni": "pStorageDataHeader", 'fuel': 'f'},
    'pTechData': {'Technology': 'tech'},
    "pVREGenProfile": {"uni": "t"}
}


def apply_debug_column_renames(container: gt.Container, rename_map=None):
    """Align column names with the GAMS schema for lightweight CLI runs."""
    data = getattr(container, "data", None)
    if data is None:
        return

    active_map = rename_map or COLUMN_RENAME_MAP
    for param_name, replacements in active_map.items():
        symbol = data.get(param_name)
        if symbol is None:
            continue
        records = symbol.records
        if records is None:
            continue
        applicable = {
            old: new
            for old, new in replacements.items()
            if old in records.columns and old != new
        }
        if not applicable:
            continue
        symbol.setRecords(records.rename(columns=applicable))


def _get_allowed_zones(db: gt.Container):
    """Read allowed zones from zcmap; return None when missing so callers can skip filtering."""
    if "zcmap" not in db:
        return None
    zc_records = db["zcmap"].records
    if zc_records is None or zc_records.empty or "z" not in zc_records.columns:
        return None
    return set(zc_records["z"].dropna().unique())


def filter_inputs_to_allowed_zones(
    db: gt.Container,
    *,
    log_func=None,
    write_back=None,
):
    """Remove rows whose zones fall outside zcmap for key input tables."""
    allowed_zones = _get_allowed_zones(db)
    if not allowed_zones:
        if log_func:
            log_func("[input_treatment][zones] Skipped: no allowed zones available (zcmap missing/empty).")
        return

    for param_name, zone_cols in ZONE_RESTRICTED_PARAMS.items():
        if param_name not in db:
            continue
        records = db[param_name].records
        if records is None or records.empty:
            if log_func:
                log_func(f"[input_treatment][zones] Skipped {param_name}: empty.")
            continue
        if any(col not in records.columns for col in zone_cols):
            if log_func:
                log_func(f"[input_treatment][zones] Skipped {param_name}: missing zone columns {zone_cols}; got {list(records.columns)}.")
            continue

        mask = pd.Series(True, index=records.index)
        for col in zone_cols:
            mask &= records[col].isin(allowed_zones)

        filtered = records.loc[mask]
        if filtered.shape[0] == records.shape[0]:
            continue

        removed = records.shape[0] - filtered.shape[0]
        if log_func:
            log_func(f"{param_name}: removing {removed} row(s) with zones outside zcmap.")
        db.data[param_name].setRecords(filtered)
        if write_back is not None:
            write_back(param_name)


def load_generic_defaults(db: gt.Container, log_func=None):
    """Load generic (tech, fuel) defaults from GAMS database.
    
    These are read by input_readers.gms from resources/pGenDataInputGeneric.csv etc.
    Returns dict with keys: 'gendata', 'availability', 'capex' or empty if missing.
    """
    result = {}
    
    if "pGenDataInputGeneric" in db:
        records = db["pGenDataInputGeneric"].records
        if records is not None and not records.empty:
            result['gendata'] = records.copy()
            if log_func:
                n_tech_fuel = records[["tech", "f"]].drop_duplicates().shape[0] if "f" in records.columns else records[["tech", "fuel"]].drop_duplicates().shape[0]
                log_func(f"[generic_defaults] pGenDataInputGeneric loaded: {n_tech_fuel} tech-fuel combinations.")
        elif log_func:
            log_func("[generic_defaults] pGenDataInputGeneric exists but is empty.")
    elif log_func:
        log_func("[generic_defaults] pGenDataInputGeneric not found in database.")
    
    if "pAvailabilityGeneric" in db:
        records = db["pAvailabilityGeneric"].records
        if records is not None and not records.empty:
            result['availability'] = records.copy()
            if log_func:
                n_tech_fuel = records[["tech", "f"]].drop_duplicates().shape[0] if "f" in records.columns else records[["tech", "fuel"]].drop_duplicates().shape[0]
                log_func(f"[generic_defaults] pAvailabilityGeneric loaded: {n_tech_fuel} tech-fuel combinations.")
        elif log_func:
            log_func("[generic_defaults] pAvailabilityGeneric exists but is empty.")
    elif log_func:
        log_func("[generic_defaults] pAvailabilityGeneric not found in database.")
    
    if "pCapexTrajectoriesGeneric" in db:
        records = db["pCapexTrajectoriesGeneric"].records
        if records is not None and not records.empty:
            result['capex'] = records.copy()
            if log_func:
                n_tech_fuel = records[["tech", "f"]].drop_duplicates().shape[0] if "f" in records.columns else records[["tech", "fuel"]].drop_duplicates().shape[0]
                log_func(f"[generic_defaults] pCapexTrajectoriesGeneric loaded: {n_tech_fuel} tech-fuel combinations.")
        elif log_func:
            log_func("[generic_defaults] pCapexTrajectoriesGeneric exists but is empty.")
    elif log_func:
        log_func("[generic_defaults] pCapexTrajectoriesGeneric not found in database.")
    
    if "pStorageDataInputGeneric" in db:
        records = db["pStorageDataInputGeneric"].records
        if records is not None and not records.empty:
            result['storagedata'] = records.copy()
            if log_func:
                n_tech_fuel = records[["tech", "f"]].drop_duplicates().shape[0] if "f" in records.columns else records[["tech", "fuel"]].drop_duplicates().shape[0]
                log_func(f"[generic_defaults] pStorageDataInputGeneric loaded: {n_tech_fuel} tech-fuel combinations.")
        elif log_func:
            log_func("[generic_defaults] pStorageDataInputGeneric exists but is empty.")
    elif log_func:
        log_func("[generic_defaults] pStorageDataInputGeneric not found in database.")
    
    return result


def merge_storage_into_gendata(gams):
    """
    Merge storage units from pStorageDataInput into pGenDataInput and gmap.

    This function:
    1. Adds all storage units to the gmap set (g,z,tech,f mapping)
    2. Copies values from pStorageDataInput to pGenDataInput for headers
       that exist in both pGenDataInputHeader and pStorageDataHeader sets

    This ensures all units (generators + storage) have consistent data
    in pGenDataInput and gmap for downstream processing. Status filtering
    is handled later in the input treatment pipeline.

    Args:
        gams: GAMS embedded code object with db attribute
    """
    db = gt.Container(gams.db)

    # Check required parameters exist
    if "pStorageDataInput" not in db or "pGenDataInput" not in db:
        gams.printLog("[storage_merge] Skipped: pStorageDataInput or pGenDataInput missing.")
        return

    if "pGenDataInputHeader" not in db or "pStorageDataHeader" not in db:
        gams.printLog("[storage_merge] Skipped: header sets missing.")
        return

    storage_records = db["pStorageDataInput"].records
    gen_records = db["pGenDataInput"].records

    if storage_records is None or storage_records.empty:
        gams.printLog("[storage_merge] Skipped: pStorageDataInput is empty.")
        return

    # Detect generator column name (could be 'g' or 'uni')
    g_col = "g" if "g" in storage_records.columns else "uni"

    # Get header sets
    gen_headers_df = db["pGenDataInputHeader"].records
    storage_headers_df = db["pStorageDataHeader"].records

    if gen_headers_df is None or storage_headers_df is None:
        gams.printLog("[storage_merge] Skipped: header sets are empty.")
        return

    # Extract header values (first column, whatever it's named)
    gen_headers = set(gen_headers_df.iloc[:, 0].tolist())
    storage_headers = set(storage_headers_df.iloc[:, 0].tolist())

    # Find common headers
    common_headers = gen_headers & storage_headers

    if not common_headers:
        gams.printLog("[storage_merge] No common headers between generator and storage data.")
        return

    # Identify the header column name in storage records
    storage_header_col = "pStorageDataHeader"
    if storage_header_col not in storage_records.columns:
        for col in storage_records.columns:
            if col not in [g_col, "z", "tech", "f", "value"]:
                storage_header_col = col
                break

    # Identify the header column name in gen records
    gen_header_col = "pGenDataInputHeader"
    if gen_records is not None and gen_header_col not in gen_records.columns:
        for col in gen_records.columns:
            if col not in [g_col, "z", "tech", "f", "value"]:
                gen_header_col = col
                break

    # Get all unique storage units (g, z, tech, f)
    storage_keys = storage_records[[g_col, "z", "tech", "f"]].drop_duplicates()
    storage_keys_list = list(zip(
        storage_keys[g_col],
        storage_keys["z"],
        storage_keys["tech"],
        storage_keys["f"]
    ))
    storage_keys_set = set(storage_keys_list)

    # --- Update gmap set with storage units ---
    if "gmap" in db:
        gmap_records = db["gmap"].records
        # Detect gmap column name
        gmap_g_col = g_col
        if gmap_records is not None and not gmap_records.empty:
            gmap_g_col = "g" if "g" in gmap_records.columns else g_col

        # Build new gmap entries for storage units
        gmap_new = pd.DataFrame(storage_keys_list, columns=[gmap_g_col, "z", "tech", "f"])
        if gmap_records is not None and not gmap_records.empty:
            gmap_merged = pd.concat([gmap_records, gmap_new], ignore_index=True)
            gmap_merged = gmap_merged.drop_duplicates(subset=[gmap_g_col, "z", "tech", "f"])
        else:
            gmap_merged = gmap_new
        db.data["gmap"].setRecords(gmap_merged)
        db.write(gams.db, ["gmap"], eps_to_zero=False)
        gams.printLog(f"[storage_merge] Added {len(storage_keys_list)} storage unit(s) to gmap.")

    # --- Update pGenDataInput with storage data ---
    # Build new records for pGenDataInput using the same column names as gen_records
    new_records = []
    for hdr in common_headers:
        hdr_data = storage_records[storage_records[storage_header_col] == hdr]
        for _, row in hdr_data.iterrows():
            new_records.append({
                g_col: row[g_col],
                "z": row["z"],
                "tech": row["tech"],
                "f": row["f"],
                gen_header_col: hdr,
                "value": row["value"]
            })

    if not new_records:
        gams.printLog("[storage_merge] No records to merge into pGenDataInput.")
        return

    # Merge with existing pGenDataInput records
    new_df = pd.DataFrame(new_records)

    if gen_records is not None and not gen_records.empty:
        merged = pd.concat([gen_records, new_df], ignore_index=True)
        # Drop duplicates, keeping the last (storage) values for overlapping entries
        merged = merged.drop_duplicates(
            subset=[g_col, "z", "tech", "f", gen_header_col],
            keep="last"
        )
    else:
        merged = new_df

    db.data["pGenDataInput"].setRecords(merged)
    db.write(gams.db, ["pGenDataInput"], eps_to_zero=False)

    n_storage = len(storage_keys_set)
    n_headers = len(common_headers)
    gams.printLog(
        f"[storage_merge] Merged {n_storage} storage unit(s) "
        f"with {n_headers} common header(s) into pGenDataInput."
    )


def run_input_treatment(gams,
                        fill_missing_hydro_availability: bool = False,
                        fill_missing_hydro_capex: bool = False):

    def _write_back(db: gt.Container, param_name: str, eps_to_zero: bool = True):
        """Copy updates back to whatever database the caller provided.

        A full model run hands in a real gams.GamsDatabase, while our debug path
        only supplies a gt.Container. We support both without needing the GAMS runtime.

        Parameters
        ----------
        db : gt.Container
            The working container with updated records.
        param_name : str
            The parameter name to write back.
        eps_to_zero : bool, optional
            If False, preserve EPS (epsilon) values instead of converting to zero.
            Default is True for backward compatibility.
        """
        target_db = gams.db
        records = db[param_name].records
        new_count = 0 if records is None else len(records)

        if isinstance(target_db, gt.Container):
            # Get previous count before writing
            prev_count = 0
            if param_name in target_db.data and target_db.data[param_name].records is not None:
                prev_count = len(target_db.data[param_name].records)

            if param_name not in target_db.data:
                # Mirror the schema from the working container when the target container
                # doesn't yet have the symbol (avoids KeyError on first write).
                source_symbol = db[param_name]
                # Prefer explicit domains; fall back to non-value columns.
                domains = getattr(source_symbol, "domains", None)
                if not domains and source_symbol.records is not None:
                    domains = [c for c in source_symbol.records.columns if c != "value"]
                target_db.addParameter(param_name, domains or [], records=None)

            target_db.data[param_name].setRecords(records)
            gams.printLog(
                f"[input_treatment] {param_name}: {prev_count} -> {new_count} row(s)."
            )
            return

        # When writing back to a real GAMS database we must clear the symbol
        # first; db.write() only overwrites tuples that still exist, so stale
        # rows would otherwise survive our filters.

        # Get previous count before clearing
        prev_snapshot = gt.Container(target_db)
        prev_records = prev_snapshot[param_name].records if param_name in prev_snapshot.data else None
        prev_count = 0 if prev_records is None else len(prev_records)

        symbol = target_db[param_name]
        symbol.clear()
        db.write(target_db, [param_name], eps_to_zero=eps_to_zero)

        gams.printLog(
            f"[input_treatment] {param_name}: {prev_count} -> {new_count} row(s)."
        )

    def _col_list(df):
        return list(df.columns) if df is not None else []

    def _log_columns(param_name, df, prefix=""):
        gams.printLog(f"{prefix}{param_name} columns: {_col_list(df)}")

    def _step(title):
        gams.printLog("-" * 60)
        gams.printLog(title)

    gams.printLog("=" * 60)
    gams.printLog("[input_treatment] starting")
    gams.printLog("=" * 60)


    def _detect_header_and_value_columns(df: pd.DataFrame):
        header_col = None
        for candidate in ("pGenDataInputHeader", "uni"):
            if candidate in df.columns:
                header_col = candidate
                break
        value_col = "value" if "value" in df.columns else None
        return header_col, value_col


    def _log_zone_summary(gams,
                          title: str,
                          df: pd.DataFrame,
                          zone_column: str,
                          zone_label: str,
                          all_zones=None):
        """Print generator names grouped by zone, padding zones with empty lists."""
        if df is None:
            return

        gams.printLog(title)
        if zone_column and zone_column in df.columns:
            summary = (
                df.groupby(zone_column, observed=False)["g"]
                .apply(lambda s: sorted(s.tolist()))
                .to_dict()
            )
            zone_keys = sorted(set(all_zones)) if all_zones is not None else sorted(summary)
        else:
            summary = {"unknown": sorted(df["g"].tolist())}
            zone_keys = sorted(summary)

        for zone in zone_keys:
            gams.printLog(f"    {zone_label} {zone}: {summary.get(zone, [])}")


    def remove_generators_with_invalid_status(db: gt.Container):
        """Drop generators whose status is missing or outside 1-3."""

        records = db["pGenDataInput"].records
        if records is None or records.empty or "g" not in records.columns:
            return

        records = records.copy()

        all_gens = set(records["g"].unique())
        status_rows = records.loc[records["pGenDataInputHeader"] == "Status"].copy()
        status_rows["numeric_status"] = pd.to_numeric(status_rows["value"], errors="coerce")
        valid_mask = status_rows["numeric_status"].isin([1, 2, 3])
        valid_gens = set(status_rows.loc[valid_mask, "g"].unique())
        invalid_gens = all_gens - valid_gens
        if not invalid_gens:
            return

        zone_frame = (
            records.loc[records["g"].isin(invalid_gens), ["g", "z"]]
            .drop_duplicates(subset=["g"])
        )
        gams.printLog(
            f"Removing {len(zone_frame)} generator(s) due to invalid or missing Status (allowed values: 1, 2, 3)."
        )
        zone_listing = (
            zone_frame.groupby("z", observed=True)["g"]
            .apply(lambda s: ", ".join(sorted(s)))
            .sort_index()
        )
        for zone_val, gens in zone_listing.items():
            gams.printLog(f"  - z: {zone_val} -> {gens}")

        filtered_records = records.loc[~records["g"].isin(invalid_gens)]
        db.data["pGenDataInput"].setRecords(filtered_records)
        _write_back(db, "pGenDataInput")


    def remove_transmissions_with_invalid_status(db: gt.Container):
        """Apply transmission Status filter to all relevant parameters."""
            
        param_name = "pNewTransmission"
        
        records = db[param_name].records
        wide = (
            records
            .pivot_table(index=["z", "z2"],
                         columns="pTransmissionHeader",
                         values="value",
                         aggfunc="first",
                         observed=False)
        )
        if wide.empty:
            return
        if "Status" not in wide.columns:
            wide["Status"] = np.nan

        status_numeric = pd.to_numeric(wide["Status"], errors="coerce")
        valid_mask = status_numeric.notna() & (status_numeric != 0)
        filtered = wide.loc[valid_mask]
        if filtered.shape[0] == wide.shape[0]:
            gams.printLog(
                f"All transmission corridor(s) status are valid."
            )
            return

        removed_count = wide.shape[0] - filtered.shape[0]
        gams.printLog(
            f"Removing {removed_count} transmission corridor(s) from {param_name} "
            "due to Status=0 or missing Status."
        )
        removed_rows = wide.loc[~valid_mask, ["Status"]].reset_index()
        for _, row in removed_rows.iterrows():
            status_label = "missing" if pd.isna(row["Status"]) else row["Status"]
            gams.printLog(f"  - {row['z']} -> {row['z2']} (Status={status_label})")

        stacked = (
            filtered
            .stack()
            .reset_index()
            .rename(columns={"level_2": "uni", 0: "value"})
        )

        db.data[param_name].setRecords(stacked)
        _write_back(db, param_name)
        

    def zero_capacity_for_invalid_generator_status(db: gt.Container):
        """Set Capacity=0 for generators with missing/invalid Status."""
        param_name = "pGenDataInput"
        if param_name not in db:
            gams.printLog("[input_treatment][gen_status] Skipped: pGenDataInput missing.")
            return
        records = db[param_name].records
        if records is None or records.empty:
            gams.printLog("[input_treatment][gen_status] Skipped: pGenDataInput empty.")
            _log_columns(param_name, records, prefix="[input_treatment][gen_status] ")
            return
        if "g" not in records.columns or "pGenDataInputHeader" not in records.columns:
            gams.printLog("[input_treatment][gen_status] Skipped: missing 'g' or 'pGenDataInputHeader' column.")
            _log_columns(param_name, records, prefix="[input_treatment][gen_status] ")
            return

        records = records.copy()
        status_rows = records.loc[records["pGenDataInputHeader"] == "Status"].copy()
        all_gens = set(records["g"].dropna().unique())
        if status_rows.empty:
            invalid_gens = all_gens
        else:
            status_rows["numeric_status"] = pd.to_numeric(status_rows["value"], errors="coerce")
            valid_gens = set(
                status_rows.loc[status_rows["numeric_status"].isin([1, 2, 3]), "g"].dropna().unique()
            )
            invalid_gens = all_gens - valid_gens
        if not invalid_gens:
            gams.printLog("[input_treatment][gen_status] All generators have valid status.")
            return

        mask = (
            (records["pGenDataInputHeader"] == "Capacity")
            & records["g"].isin(invalid_gens)
        )
        if not mask.any():
            return

        records.loc[mask, "value"] = 0
        db.data[param_name].setRecords(records)
        _write_back(db, param_name)
        gams.printLog(
            f"Setting Capacity=0 for {mask.sum()} generator row(s) with invalid/missing Status (allowed values: 1, 2, 3)."
        )


    def zero_capacity_for_invalid_transmission_status(db: gt.Container):
        """Set Capacity=0 for transmission corridors with Status missing or 0."""
        param_name = "pNewTransmission"
        if param_name not in db:
            gams.printLog("[input_treatment][tx_status] Skipped: pNewTransmission missing.")
            return
        records = db[param_name].records
        if records is None or records.empty:
            gams.printLog("[input_treatment][tx_status] Skipped: pNewTransmission empty.")
            _log_columns(param_name, records, prefix="[input_treatment][tx_status] ")
            return
        if any(col not in records.columns for col in ("z", "z2", "pTransmissionHeader")):
            gams.printLog("[input_treatment][tx_status] Skipped: missing required columns z/z2/pTransmissionHeader.")
            _log_columns(param_name, records, prefix="[input_treatment][tx_status] ")
            return

        # Consolidate by corridor to decide which pairs are invalid before zeroing CapacityPerLine.
        wide = (
            records
            .pivot_table(
                index=["z", "z2"],
                columns="pTransmissionHeader",
                values="value",
                aggfunc="first",
                observed=False,
            )
        )
        if wide.empty:
            gams.printLog("[input_treatment][tx_status] Skipped: pivot empty (no corridors).")
            return
        if "Status" not in wide.columns:
            wide["Status"] = np.nan
        status_numeric = pd.to_numeric(wide["Status"], errors="coerce")
        valid_mask = status_numeric.notna() & (status_numeric != 0)
        invalid_pairs = wide.index[~valid_mask].tolist()
        if not invalid_pairs:
            gams.printLog("[input_treatment][tx_status] All transmission corridors have valid status.")
            return

        invalid_set = set(invalid_pairs)
        pair_flags = pd.Series(
            ((z, z2) in invalid_set for z, z2 in zip(records["z"], records["z2"])),
            index=records.index,
        )
        mask = (records["pTransmissionHeader"] == "CapacityPerLine") & pair_flags
        if not mask.any():
            return

        records = records.copy()
        records.loc[mask, "value"] = 0
        db.data[param_name].setRecords(records)
        _write_back(db, param_name)

        gams.printLog(
            f"Setting CapacityPerLine=0 for {mask.sum()} transmission row(s) due to Status missing or equal to 0."
        )


    def monitor_hydro_availability(db: gt.Container, auto_fill: bool):
        """Log missing hydro availability rows and optionally back-fill them."""
        # Use pAvailabilityInput (the raw CSV data without year dimension)
        required_params = {"pGenDataInput", "pAvailabilityInput"}
        if any(name not in db for name in required_params):
            gams.printLog("[input_treatment][hydro_avail] Skipped: pGenDataInput or pAvailabilityInput missing.")
            return

        gen_records = db["pGenDataInput"].records
        avail_records = db["pAvailabilityInput"].records
        if gen_records is None or gen_records.empty:
            gams.printLog("[input_treatment][hydro_avail] Skipped: pGenDataInput empty.")
            _log_columns("pGenDataInput", gen_records, prefix="[input_treatment][hydro_avail] ")
            return

        gen_records = gen_records.copy()
        avail_records = avail_records.copy() if avail_records is not None else pd.DataFrame(columns=["g", "q", "value"])

        if "g" not in gen_records.columns or "tech" not in gen_records.columns:
            gams.printLog("[input_treatment][hydro_avail] Skipped: missing 'g' or 'tech' in pGenDataInput.")
            _log_columns("pGenDataInput", gen_records, prefix="[input_treatment][hydro_avail] ")
            return

        zone_column = "z"

        notebook_hint = "pre-analysis/prepare-data/hydro_availability.ipynb"
        allowed_zones = None
        if zone_column and "zcmap" in db:
            zcmap_records = db["zcmap"].records
            if zcmap_records is not None and not zcmap_records.empty:
                zone_candidates = [col for col in ("z", "zone") if col in zcmap_records.columns]
                if zone_candidates:
                    zone_key = zone_candidates[0]
                    allowed_zones = (
                        zcmap_records[zone_key]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

        def _extract_hydro_meta(target_techs):
            hydro_meta_cols = ["g", "tech"] + ([zone_column] if zone_column else [])
            mask = gen_records["tech"].isin(target_techs)
            if not mask.any():
                return pd.DataFrame(columns=hydro_meta_cols)
            df = (
                gen_records.loc[mask, hydro_meta_cols]
                .drop_duplicates(subset=["g"])
            )
            if allowed_zones and zone_column in df.columns:
                df = df[df[zone_column].isin(allowed_zones)]
            return df

        # --- ReservoirHydro: check pAvailability ---------------------------------
        reservoir_meta = _extract_hydro_meta({"ReservoirHydro"})
        if not reservoir_meta.empty and "g" in avail_records.columns:
            provided_gens = set(avail_records["g"].unique())
            missing_meta = reservoir_meta.loc[~reservoir_meta["g"].isin(provided_gens)]
            if missing_meta.empty:
                gams.printLog(
                    f"Reservoir hydro availability check: all {len(reservoir_meta)} generator(s) "
                    "defined in pGenDataInput have entries in pAvailability."
                )
                if not auto_fill:
                    gams.printLog("[input_treatment][hydro_avail] Auto-fill disabled; no gaps found.")
            else:
                zone_label = zone_column or "zone"
                all_zone_values = None
                if zone_column:
                    if allowed_zones is not None:
                        all_zone_values = allowed_zones
                    else:
                        all_zone_values = (
                            reservoir_meta.loc[:, zone_column]
                            .dropna()
                            .unique()
                            .tolist()
                        )
                gams.printLog(
                    f"[input_treatment][hydro_avail] Reservoir hydro warning: {len(missing_meta)} generator(s) lack entries in pAvailability."
                )
                gams.printLog("-" * 60)
                _log_zone_summary(
                    gams,
                    "Missing reservoir capacity-factor rows by zone:",
                    missing_meta.loc[:, ["g"] + ([zone_column] if zone_column else [])],
                    zone_column,
                    zone_label,
                    all_zone_values,
                )
                gams.printLog("-" * 60)

                if auto_fill:
                    if avail_records.empty:
                        gams.printLog("[input_treatment][hydro_avail] Auto-fill skipped: pAvailability has no existing data to copy from.")
                    elif zone_column is None:
                        gams.printLog("[input_treatment][hydro_avail] Auto-fill skipped: cannot identify zone column in pGenDataInput.")
                    else:
                        # Build donor profiles keyed by (zone, tech) from generators that do have data.
                        gen_zone_meta = gen_records.loc[:, ["g", "tech", zone_column]].drop_duplicates(subset=["g"])
                        donor_frame = avail_records.merge(gen_zone_meta, on="g", how="left")
                        donor_frame = donor_frame.dropna(subset=[zone_column, "tech"])
                        if donor_frame.empty:
                            gams.printLog("[input_treatment][hydro_avail] Auto-fill skipped: no donor generators have both zone and tech information.")
                        else:
                            donor_profiles = {}
                            for (zone_val, tech_val), frame in donor_frame.groupby([zone_column, "tech"], observed=False):
                                # Average availability across all donor generators for this (zone, tech).
                                profile = (
                                    frame.groupby("q", observed=False)["value"]
                                    .mean()
                                    .reset_index()
                                )
                                donors = sorted(frame["g"].unique())
                                donor_profiles[(zone_val, tech_val)] = {
                                    "profile": profile,
                                    "source": f"mean of {len(donors)} generators ({', '.join(donors)})",
                                }

                            new_entries = []
                            filled_by_key = {}  # Group filled generators by (zone, tech, source)
                            no_donor = []
                            for row in missing_meta.itertuples():
                                key = (getattr(row, zone_column), row.tech)
                                donor_info = donor_profiles.get(key)
                                if donor_info is None:
                                    no_donor.append(f"{row.g} ({row.tech}, {zone_label}: {key[0]})")
                                    continue
                                addition = donor_info["profile"].copy()
                                addition["g"] = row.g
                                new_entries.append(addition.loc[:, ["g", "q", "value"]])
                                group_key = (key[0], row.tech, donor_info["source"])
                                filled_by_key.setdefault(group_key, []).append(row.g)

                            # Log no-donor cases
                            if no_donor:
                                gams.printLog(f"[input_treatment][hydro_avail] No donor found to auto-fill: {', '.join(no_donor)}.")

                            # Log filled generators grouped by (zone, tech, source)
                            if filled_by_key:
                                gams.printLog("[input_treatment][hydro_avail] Auto-filled availability:")
                                for (zone_val, tech, source), gens in sorted(filled_by_key.items()):
                                    gams.printLog(f"  {zone_label}: {zone_val}, {tech} ({len(gens)}): {', '.join(sorted(gens))}")
                                    gams.printLog(f"    using {source}")

                            if new_entries:
                                updated_availability = pd.concat([avail_records] + new_entries, ignore_index=True)
                                db.data["pAvailabilityInput"].setRecords(updated_availability)
                                _write_back(db, "pAvailabilityInput")
                            else:
                                gams.printLog("[input_treatment][hydro_avail] Auto-fill finished: no records were added.")

        # --- ROR: check pVREgenProfile -------------------------------------------
        ror_meta = _extract_hydro_meta({"ROR"})
        if ror_meta.empty:
            gams.printLog("[input_treatment][ror_profile] Skipped: no ROR generators found.")
            return

        if "pVREgenProfile" not in db:
            gams.printLog(
                "[input_treatment][ror_profile] ROR availability warning: pVREgenProfile parameter is missing. "
                f"Rebuild the inputs via `{notebook_hint}`."
            )
            return

        ror_records = db["pVREgenProfile"].records
        if ror_records is None:
            ror_records = pd.DataFrame()
            provided_ror = set()
        elif ror_records.empty:
            provided_ror = set()
        elif "g" in ror_records.columns:
            provided_ror = set(ror_records["g"].unique())
        elif "gen" in ror_records.columns:
            provided_ror = set(ror_records["gen"].unique())
        else:
            gams.printLog(
                "[input_treatment][ror_profile] ROR availability warning: cannot identify generator column in pVREgenProfile."
            )
            _log_columns("pVREgenProfile", ror_records, prefix="[input_treatment][ror_profile] ")
            return

        # Compare ROR generators in pGenDataInput with those that have hourly profiles.
        missing_ror = ror_meta.loc[~ror_meta["g"].isin(provided_ror)]
        if missing_ror.empty:
            gams.printLog(
                f"[input_treatment][ror_profile] ROR availability check: all {len(ror_meta)} generator(s) defined in pGenDataInput "
                "have hourly profiles in pVREgenProfile."
            )
            if not auto_fill:
                gams.printLog("[input_treatment][ror_profile] Auto-fill disabled; no gaps found.")
            return

        zone_label = zone_column or "zone"
        all_zone_values = None
        if zone_column:
            if allowed_zones is not None:
                all_zone_values = allowed_zones
            else:
                all_zone_values = (
                    ror_meta.loc[:, zone_column]
                    .dropna()
                    .unique()
                    .tolist()
                )

        gams.printLog("-" * 60)
        gams.printLog(
            f"[input_treatment][ror_profile] ROR availability warning: {len(missing_ror)} generator(s) lack hourly profiles in pVREgenProfile."
        )
        gams.printLog("-" * 60)
        _log_zone_summary(
            gams,
            "Missing ROR hourly profiles by zone:",
            missing_ror.loc[:, ["g"] + ([zone_column] if zone_column else [])],
            zone_column,
            zone_label,
            all_zone_values,
        )
        gams.printLog("-" * 60)

        # Optionally backfill seasonal availability for missing ROR units using donor ROR availability.
        if auto_fill_missing_hydro:
            # Ensure the symbol exists before we try to write to it.
            if "pAvailabilityInput" not in db:
                db.addParameter("pAvailabilityInput", ["g", "q"])
            avail_df = db["pAvailabilityInput"].records
            if avail_df is None:
                avail_df = pd.DataFrame(columns=["g", "q", "value"])

            missing_ror_no_avail = missing_ror.loc[~missing_ror["g"].isin(set(avail_df.get("g", [])))]
            if missing_ror_no_avail.empty:
                gams.printLog("[input_treatment][ror_profile] Auto-fill skipped: all ROR units missing profiles already have pAvailabilityInput.")
            else:
                # Join availability with ROR generator metadata to build donor profiles by zone.
                # Also prepare ReservoirHydro donors as fallback when no ROR donors exist for a zone.
                gen_zone_meta_ror = gen_records.loc[gen_records["tech"] == "ROR", ["g", zone_column]].drop_duplicates(subset=["g"])
                gen_zone_meta_reservoir = gen_records.loc[gen_records["tech"] == "ReservoirHydro", ["g", zone_column]].drop_duplicates(subset=["g"])
                donor_frame_ror = avail_df.merge(gen_zone_meta_ror, on="g", how="inner").dropna(subset=[zone_column])
                donor_frame_reservoir = avail_df.merge(gen_zone_meta_reservoir, on="g", how="inner").dropna(subset=[zone_column])
                donor_frame = donor_frame_ror  # Primary donors are ROR
                if donor_frame.empty and donor_frame_reservoir.empty:
                    gams.printLog("[input_treatment][ror_profile] Auto-fill skipped: no donor ROR or ReservoirHydro generators with both zone and availability.")
                else:
                    # Build zone-level donor profiles from ROR first, then ReservoirHydro fallback.
                    zone_profiles_ror = {}
                    for zone_val, frame in donor_frame_ror.groupby(zone_column, observed=False):
                        zone_profiles_ror[zone_val] = {
                            "profile": frame.groupby("q", observed=False)["value"].mean().reset_index(),
                            "donors": sorted(frame["g"].unique()),
                            "tech": "ROR",
                        }
                    zone_profiles_reservoir = {}
                    for zone_val, frame in donor_frame_reservoir.groupby(zone_column, observed=False):
                        zone_profiles_reservoir[zone_val] = {
                            "profile": frame.groupby("q", observed=False)["value"].mean().reset_index(),
                            "donors": sorted(frame["g"].unique()),
                            "tech": "ReservoirHydro",
                        }
                    # System-wide fallback: prefer ROR, then ReservoirHydro
                    if not donor_frame_ror.empty:
                        global_profile = {
                            "profile": donor_frame_ror.groupby("q", observed=False)["value"].mean().reset_index(),
                            "donors": sorted(donor_frame_ror["g"].unique()),
                            "tech": "ROR",
                        }
                    else:
                        global_profile = {
                            "profile": donor_frame_reservoir.groupby("q", observed=False)["value"].mean().reset_index(),
                            "donors": sorted(donor_frame_reservoir["g"].unique()),
                            "tech": "ReservoirHydro",
                        }

                    new_entries = []
                    filled_avail = []
                    filled_by_donor = {}  # Group by (zone, tech, donors tuple, fallback_reason)
                    skipped = []
                    for row in missing_ror_no_avail.itertuples():
                        zone_val = getattr(row, zone_column)
                        # Priority: ROR in same zone > ReservoirHydro in same zone > global fallback
                        donor_info = zone_profiles_ror.get(zone_val)
                        fallback_reason = None
                        if donor_info is None or not donor_info["donors"]:
                            donor_info = zone_profiles_reservoir.get(zone_val)
                            if donor_info and donor_info["donors"]:
                                fallback_reason = "no ROR donors in zone, using ReservoirHydro as proxy"
                        if donor_info is None or not donor_info["donors"]:
                            donor_info = global_profile
                            if donor_info and donor_info["donors"]:
                                fallback_reason = f"no donors in zone, using global {donor_info.get('tech', 'ROR')} average"
                        if not donor_info["donors"]:
                            skipped.append(row.g)
                            continue
                        addition = donor_info["profile"].copy()
                        addition["g"] = row.g
                        new_entries.append(addition.loc[:, ["g", "q", "value"]])
                        filled_avail.append(row.g)
                        donor_tech = donor_info.get("tech", "ROR")
                        group_key = (zone_val, donor_tech, tuple(donor_info["donors"]), fallback_reason)
                        filled_by_donor.setdefault(group_key, []).append(row.g)

                    # Log skipped generators
                    if skipped:
                        gams.printLog(f"[input_treatment][ror_profile] Auto-fill skipped (no donors): {', '.join(skipped)}.")

                    # Log filled generators grouped by donor source
                    if filled_by_donor:
                        gams.printLog("[input_treatment][ror_profile] Auto-filled pAvailabilityInput:")
                        for (zone_val, tech, donors, fallback_reason), gens in sorted(filled_by_donor.items()):
                            gams.printLog(f"  {zone_label}: {zone_val} ({len(gens)}): {', '.join(sorted(gens))}")
                            donor_msg = f"    using mean of {len(donors)} {tech} donor(s): {', '.join(donors)}"
                            if fallback_reason:
                                donor_msg += f" ({fallback_reason})"
                            gams.printLog(donor_msg)

                    if new_entries:
                        updated_availability = pd.concat([avail_df] + new_entries, ignore_index=True)
                        db.data["pAvailabilityInput"].setRecords(updated_availability)
                        _write_back(db, "pAvailabilityInput")
                        avail_df = updated_availability  # keep in sync for subsequent steps

        # Fallback: if a custom pAvailability exists, populate pVREgenProfile with flat
        # profiles derived from that availability (one value copied across all hours).
        if fill_ror_from_availability and "pAvailabilityInput" in db:
            avail_df = db["pAvailabilityInput"].records
            if avail_df is not None and not avail_df.empty and {"g", "q", "value"}.issubset(avail_df.columns):
                gams.printLog("-" * 60)
                gams.printLog(
                    "[input_treatment][ror_profile] EPM_FILL_ROR_FROM_AVAILABILITY is enabled; "
                    "building flat hourly profiles in pVREgenProfile from pAvailabilityInput."
                )
                # Use hours columns already present in pVREgenProfile when possible; otherwise
                # fall back to the hour columns from pHours (t1..t24).
                hour_cols = [c for c in ror_records.columns if c not in {"g", "q", "d"}] if not ror_records.empty else []
                if not hour_cols and "pHours" in db and db["pHours"].records is not None:
                    hour_cols = [c for c in db["pHours"].records.columns if c.startswith("t")]
                if not hour_cols:
                    gams.printLog("ROR availability auto-fill skipped: could not determine hour columns for pVREgenProfile.")
                else:
                    # Determine daytypes per season from pHours; if absent, use a single placeholder.
                    pHours_df = None
                    daytype_by_q = {}
                    if "pHours" in db and db["pHours"].records is not None and {"q", "d", "t"}.issubset(db["pHours"].records.columns):
                        pHours_df = db["pHours"].records
                        daytype_by_q = (
                            pHours_df.loc[:, ["q", "d"]]
                            .dropna()
                            .drop_duplicates()
                            .groupby("q", observed=False)["d"]
                            .apply(list)
                            .to_dict()
                        )

                    avail_idx = (
                        avail_df.dropna(subset=["g", "q"])
                        .copy()
                        .set_index(["g", "q"])["value"]
                    )
                    new_rows = []
                    filled_gens = set()

                    if pHours_df is None:
                        gams.printLog(
                            "[input_treatment][ror_profile] Auto-fill skipped: pHours missing required columns (q,d,t)."
                        )
                    else:
                        for row in missing_ror.itertuples():
                            # For each missing generator, build one row per (q,d,t) from pHours using its seasonal availability.
                            for (gen_id, season), avail_value in avail_idx[avail_idx.index.get_level_values(0) == row.g].items():
                                slice_hours = pHours_df[pHours_df["q"] == season][["q", "d", "t"]]
                                if slice_hours.empty:
                                    continue
                                for _, hrow in slice_hours.iterrows():
                                    new_rows.append(
                                        {
                                            "g": gen_id,
                                            "q": hrow["q"],
                                            "d": hrow["d"],
                                            "t": hrow["t"],
                                            "value": avail_value,
                                        }
                                    )
                                filled_gens.add(gen_id)

                    def _stack_pvregen(df: pd.DataFrame) -> pd.DataFrame:
                        """Coerce pVREgenProfile to long form (g,q,d,t,value)."""
                        if df is None or df.empty:
                            return pd.DataFrame(columns=["g", "q", "d", "t", "value"])
                        if {"g", "q", "d", "t", "value"}.issubset(df.columns):
                            return df
                        value_cols = [c for c in df.columns if c not in {"g", "q", "d", "value", "t"}]
                        if value_cols:
                            df = df.melt(
                                id_vars=["g", "q", "d"],
                                value_vars=value_cols,
                                var_name="t",
                                value_name="value",
                            )
                        return df

                    if new_rows:
                        new_df = pd.DataFrame(new_rows)
                        existing_long = _stack_pvregen(ror_records)
                        updated_profiles = pd.concat([existing_long, new_df], ignore_index=True)
                        required_cols = {"g", "q", "d", "t", "value"}

                        # Log before writing so verbose message appears before write_back logs
                        gams.printLog(
                            f"[input_treatment][ror_profile] Auto-filled ROR profiles for {len(filled_gens)} generator(s) in pVREgenProfile "
                            f"using their pAvailabilityInput values (steady production assumed): {', '.join(sorted(filled_gens))}."
                        )

                        if required_cols.issubset(updated_profiles.columns):
                            db.data["pVREgenProfile"].setRecords(updated_profiles)
                            _write_back(db, "pVREgenProfile")
                        else:
                            gams.printLog(
                                "[input_treatment][ror_profile] Skipping write: unable to coerce pVREgenProfile to long form "
                                f"(columns found: {list(updated_profiles.columns)})."
                            )

                        # Set pAvailability to 1 for these generators to avoid double scaling.
                        updated_avail = avail_df.copy()
                        updated_avail.loc[updated_avail["g"].isin(filled_gens), "value"] = 1
                        db.data["pAvailabilityInput"].setRecords(updated_avail)
                        _write_back(db, "pAvailabilityInput")


    def monitor_hydro_capex(db: gt.Container, auto_fill: bool):
        """Ensure hydro capex is specified for committed/candidate plants."""
        if "pGenDataInput" not in db:
            gams.printLog("[input_treatment][hydro_capex] Skipped: pGenDataInput missing.")
            return

        records = db["pGenDataInput"].records
        if records is None or records.empty:
            gams.printLog("[input_treatment][hydro_capex] Skipped: pGenDataInput empty.")
            _log_columns("pGenDataInput", records, prefix="[input_treatment][hydro_capex] ")
            return
        records = records.copy()

        header_col, value_col = _detect_header_and_value_columns(records)
        if header_col is None or value_col is None:
            gams.printLog("[input_treatment][hydro_capex] Skipped: cannot detect header/value columns.")
            _log_columns("pGenDataInput", records, prefix="[input_treatment][hydro_capex] ")
            return
        if "g" not in records.columns or "tech" not in records.columns:
            gams.printLog("[input_treatment][hydro_capex] Skipped: missing 'g' or 'tech' in pGenDataInput.")
            _log_columns("pGenDataInput", records, prefix="[input_treatment][hydro_capex] ")
            return

        zone_column = None
        for candidate in ("z", "zone"):
            if candidate in records.columns:
                zone_column = candidate
                break

        allowed_zones = None
        allowed_zone_strs = None
        if zone_column and "zcmap" in db:
            zcmap_records = db["zcmap"].records
            if zcmap_records is not None and not zcmap_records.empty:
                zone_candidates = [col for col in ("z", "zone") if col in zcmap_records.columns]
                if zone_candidates:
                    zone_key = zone_candidates[0]
                    allowed_zones = (
                        zcmap_records[zone_key]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                    allowed_zone_strs = {str(z) for z in allowed_zones}

        target_headers = {"Status", "Capex"}
        subset = records.loc[records[header_col].isin(target_headers)]
        if subset.empty:
            gams.printLog("[input_treatment][hydro_capex] Skipped: no Status/Capex rows found.")
            _log_columns("pGenDataInput", records, prefix="[input_treatment][hydro_capex] ")
            return

        index_cols = ["g"]
        if zone_column:
            index_cols.append(zone_column)
        index_cols.append("tech")

        pivot = subset.pivot_table(index=index_cols,
                                   columns=header_col,
                                   values=value_col,
                                   aggfunc="first",
                                   observed=False)
        pivot = pivot.reset_index()
        pivot.columns.name = None
        if "Capex" not in pivot.columns:
            pivot["Capex"] = np.nan
        if "Status" not in pivot.columns:
            pivot["Status"] = np.nan

        pivot["Status"] = pd.to_numeric(pivot["Status"], errors="coerce")
        pivot["Capex"] = pd.to_numeric(pivot["Capex"], errors="coerce")

        zone_series_str = None
        if zone_column and allowed_zone_strs is not None:
            zone_series_str = pivot[zone_column].astype(str)

        target_status = pivot["Status"].isin([2, 3])
        target_techs = {"ROR", "ReservoirHydro"}
        tech_mask = pivot["tech"].isin(target_techs)
        missing_mask = target_status & tech_mask & pivot["Capex"].isna()
        if zone_series_str is not None:
            missing_mask &= zone_series_str.isin(allowed_zone_strs)
        missing_capex = pivot.loc[missing_mask]
        if missing_capex.empty:
            if not auto_fill:
                gams.printLog("[input_treatment][hydro_capex] Auto-fill disabled; no hydro capex gaps found.")
            else:
                gams.printLog("[input_treatment][hydro_capex] No hydro capex gaps found.")
            return

        gams.printLog(
            f"Hydro capex warning: {len(missing_capex)} generator(s) in {sorted(target_techs)} "
            "with status 2 or 3 have no Capex defined."
        )
        gams.printLog("-" * 60)
        zone_label = zone_column or "zone"
        all_zone_values = None
        if zone_column:
            if allowed_zones is not None:
                all_zone_values = allowed_zones
            else:
                all_zone_values = (
                    pivot.loc[target_status & tech_mask, zone_column]
                    .dropna()
                    .unique()
                    .tolist()
                )
        _log_zone_summary(
            gams,
            "Missing hydro capex entries by zone:",
            missing_capex.loc[:, ["g"] + ([zone_column] if zone_column else [])],
            zone_column,
            zone_label,
            all_zone_values,
        )
        gams.printLog("-" * 60)

        if not auto_fill:
            return

        if zone_column is None:
            gams.printLog("Capex auto-fill skipped: cannot identify zone column in pGenDataInput.")
            return

        donors = pivot[target_status & tech_mask & pivot["Capex"].notna()]
        if donors.empty:
            gams.printLog("Capex auto-fill skipped: no donor hydro plants with Capex defined.")
            return

        donor_lookup = (
            donors.groupby([zone_column, "tech"], observed=False)["Capex"]
            .mean()
            .dropna()
            .to_dict()
        )

        updated = False
        new_rows = []
        filled_by_key = {}  # Group by (zone, tech, capex_value)
        no_donor = []
        skipped_no_rows = []
        for row in missing_capex.itertuples():
            key = (getattr(row, zone_column), row.tech)
            donor_info = donor_lookup.get(key)
            if donor_info is None:
                no_donor.append(f"{row.g} ({row.tech}, {zone_label}: {key[0]})")
                continue
            donor_capex = donor_info
            row_mask = (records["g"] == row.g) & (records[header_col] == "Capex")
            if row_mask.any():
                records.loc[row_mask, value_col] = donor_capex
            else:
                template = records.loc[records["g"] == row.g]
                if template.empty:
                    skipped_no_rows.append(row.g)
                    continue
                new_row = template.iloc[0].copy()
                new_row[header_col] = "Capex"
                new_row[value_col] = donor_capex
                new_rows.append(new_row)
            updated = True
            group_key = (key[0], row.tech, donor_capex)
            filled_by_key.setdefault(group_key, []).append(row.g)

        # Log no-donor cases
        if no_donor:
            gams.printLog(f"[input_treatment][hydro_capex] No donor Capex found for: {', '.join(no_donor)}.")
        if skipped_no_rows:
            gams.printLog(f"[input_treatment][hydro_capex] Capex auto-fill skipped (generator rows not found): {', '.join(skipped_no_rows)}.")

        # Log filled generators grouped by (zone, tech, capex value)
        if filled_by_key:
            gams.printLog("[input_treatment][hydro_capex] Auto-filled Capex:")
            for (zone_val, tech, capex_val), gens in sorted(filled_by_key.items()):
                gams.printLog(f"  {zone_label}: {zone_val}, {tech} ({len(gens)}): {', '.join(sorted(gens))}")
                gams.printLog(f"    using mean value {capex_val:.3f}")

        if new_rows:
            records = pd.concat([records, pd.DataFrame(new_rows)], ignore_index=True)

        if updated:
            db.data["pGenDataInput"].setRecords(records)
            _write_back(db, "pGenDataInput")


    def overwrite_nan_values(db: gt.Container, param_name: str, default_param_name: str, header: str):
        """
        Overwrites NaN values in a GAMS parameter with values from a default parameter.

        Args:
            db (gt.Container): GAMS database container.
            param_name (str): Name of the parameter to modify.
            default_param_name (str): Name of the parameter providing default values.
        """
        
        if default_param_name not in db:
            gams.printLog('{} not included'.format(default_param_name))
            return None

        # Retrieve parameter data as pandas DataFrame
        param_df = db[param_name].records
        default_df = db[default_param_name].records
        
        
        if default_df is None:
            gams.printLog('[input_treatment][defaults] {} empty so no effect'.format(default_param_name))
            db.data[param_name].setRecords(param_df)
            _write_back(db, param_name)

            return None
        
        gams.printLog("[input_treatment][defaults] Modifying {} with {}".format(param_name, default_param_name))
        
        
        # Unstack data on 'header' for correct alignment
        param_df = param_df.set_index([i for i in param_df.columns if i not in ['value']]).squeeze().unstack(header)

        default_df = default_df.set_index([i for i in default_df.columns if i not in ['value']]).squeeze().unstack(header)
        
        # Add missing columns that have been dropped by CONNECT CSV WRITER
        missing_columns = [i for i in default_df.columns if i not in param_df.columns]
        gams.printLog(f'[input_treatment][defaults] Missing {missing_columns}')
        for c in missing_columns:
            param_df[c] = float('nan')

        # Fill NaN values in param_df using corresponding values in default_df
        param_df = param_df.fillna(default_df)

        # Reset index to restore structure
        param_df = param_df.stack().reset_index()
        
        # Ensure column names are correct
        param_df.rename(columns={0: 'value'}, inplace=True)

        db.data[param_name].setRecords(param_df)
        _write_back(db, param_name)


    def prepare_generatorbased_parameter(db: gt.Container, param_name: str,
                                        cols_tokeep: list,
                                        param_ref="pGenDataInput",
                                        column_generator="g"):
        """
        Prepares a generator-based GAMS database parameter by merging it with a reference 
        parameter and extracting relevant columns.

        This function retrieves a parameter from the GAMS database, merges it with a reference 
        parameter (`param_ref`) to associate generators, and extracts a subset of relevant 
        columns for further processing.

        Parameters:
        -----------
        db : gt.Container
            A GAMS Transfer (GT) container that stores the database.
        param_name : str
            The name of the parameter to be retrieved and processed.
        cols_tokeep : list
            A list of additional columns to retain in the final output.
        param_ref : str, optional
            The name of the reference parameter used for merging (default is "pGenDataInput").
        column_generator : str, optional
            The name of the column representing the generator (default is "g").

        Returns:
        --------
        pandas.DataFrame or None
            A DataFrame containing the merged data with `column_generator`, the specified 
            columns in `cols_tokeep`, and "value", or None if the parameter is missing or empty.

        Notes:
        ------
        - If `param_name` is not found in the database, a message is printed, and None is returned.
        - If `param_name` exists but is empty, a message is printed, and None is returned.
        - The function merges `param_name` with `param_ref` based on shared columns, ensuring 
        generator reference consistency.
        - Duplicates in the reference DataFrame are removed before merging.
        """

        if param_name not in db:
            gams.printLog('{} not included'.format(param_name))
            return None

        # Retrieve parameter data as a pandas DataFrame
        param_df = db[param_name].records
        ref_df = db[param_ref].records

        # If the parameter is empty, print a message and return None
        if param_df is None or param_df.empty:
            gams.printLog(f'[input_treatment] {param_name}: empty, no defaults to prepare.')
            return None
                    
        # Identify common columns between param_df and ref_df, excluding "value"
        columns = [c for c in param_df.columns if c != "value" and c in ref_df.columns]

        if column_generator not in ref_df.columns:
            gams.printLog(
                f"[prepare_generatorbased_parameter] {param_name}: generator column '{column_generator}' not found in reference; "
                f"available columns: {list(ref_df.columns)}"
            )
            return None
        
        # Keep only the generator column and common columns in ref_df
        ref_df = ref_df.loc[:, [column_generator] + columns]
            
        # Remove duplicate rows in the reference DataFrame
        ref_df = ref_df.drop_duplicates()

        # Merge the reference DataFrame with the parameter DataFrame on common columns
        param_df = pd.merge(ref_df, param_df, how='left', on=columns)
                
        if param_df['value'].isna().any():
            missing_rows = param_df[param_df['value'].isna()]  # Get rows with NaN values
            gams.printLog(f"Error: missing values found in '{param_name}'. This indicates that some generator-year combinations expected by the model are not provided in the input data. Generators in {param_ref} without default values are:")
            gams.printLog(missing_rows.to_string())  # Print the rows where 'value' is NaN
            raise ValueError(f"Missing values in default is not permitted. To fix this bug ensure that all combination in {param_name} are included.")

        # Select only the necessary columns for the final output
        param_df = param_df.loc[:, [column_generator] + cols_tokeep + ["value"]]    
        
        return param_df


    def fill_default_value(db: gt.Container, param_name: str, default_df: pd.DataFrame, fillna=1, eps_to_zero: bool = True):
        """
        Fills missing values in a GAMS parameter with default values.

        This function modifies an existing parameter in a GAMS database by merging it
        with a default DataFrame, ensuring that missing values are filled with a specified
        default value.

        Parameters:
        -----------
        db : gt.Container
            A GAMS Transfer (GT) container that stores the database.
        param_name : str
            The name of the parameter to be modified.
        default_df : pd.DataFrame
            A DataFrame containing default values to be added if missing.
        fillna : int or float, optional
            The value to use for filling NaNs in the "value" column (default is 1).
        eps_to_zero : bool, optional
            If False, preserve EPS (epsilon) values instead of converting to zero.
            Default is True for backward compatibility.

        Returns:
        --------
        None
            The function modifies `db` in place and does not return a value.

        Notes:
        ------
        - The function prints a message indicating the parameter being modified.
        - It concatenates `default_df` with the existing parameter DataFrame.
        - Duplicate records (except for "value") are dropped, keeping the first occurrence.
        - NaN values in the "value" column are filled with `fillna`.
        """

        if default_df is None or default_df.empty:
            gams.printLog(f"[input_treatment] {param_name}: no defaults to apply.")
            return

        gams.printLog(f"[input_treatment] Modifying {param_name} with default values.")

        # Retrieve parameter data from the GAMS database as a pandas DataFrame
        param_df = db[param_name].records

        # Concatenate the original parameter data with the default DataFrame
        param_df = pd.concat([param_df, default_df], axis=0)

        # Remove duplicate entries based on all columns except "value"
        param_df = param_df.drop_duplicates(subset=[col for col in param_df.columns if col != 'value'], keep='first')

        # Fill missing values in the "value" column with the specified default value
        param_df['value'] = param_df['value'].fillna(fillna)

        # Update the parameter in the GAMS database with the modified DataFrame
        db.data[param_name].setRecords(param_df)
        _write_back(db, param_name, eps_to_zero=eps_to_zero)


    def warn_missing_availability(gams, db: gt.Container):
        """Warn if generators have no pAvailability rows (implicit availability=0)."""
        if "pGenDataInput" not in db or "pAvailabilityInput" not in db:
            return

        gen_records = db["pGenDataInput"].records
        avail_records = db["pAvailabilityInput"].records
        if gen_records is None or gen_records.empty:
            return

        gens = set(gen_records["g"].dropna().unique())
        available = set()
        if avail_records is not None and not avail_records.empty and "g" in avail_records.columns:
            available = set(avail_records["g"].dropna().unique())

        missing = gens - available
        if missing:
            missing_list = sorted(missing)
            preview = missing_list[:10]
            more = ""
            if len(missing_list) > len(preview):
                more = f" (showing {len(preview)} of {len(missing_list)})"
            gams.printLog(
                "[input_treatment][availability] Warning: the following generator(s) have no entries in pAvailability "
                f"and will have implicit availability of 0 (they will not dispatch){more}: {preview}"
            )


    def warn_missing_availability(gams, db: gt.Container):
        """Warn if generators have no pAvailability rows (implicit availability=0)."""
        if "pGenDataInput" not in db or "pAvailabilityInput" not in db:
            return

        gen_records = db["pGenDataInput"].records
        avail_records = db["pAvailabilityInput"].records
        if gen_records is None or gen_records.empty:
            return

        gens = set(gen_records["g"].dropna().unique())
        available = set()
        if avail_records is not None and not avail_records.empty and "g" in avail_records.columns:
            available = set(avail_records["g"].dropna().unique())

        missing = gens - available
        if missing:
            missing_list = sorted(missing)
            preview = missing_list[:10]
            more = ""
            if len(missing_list) > len(preview):
                more = f" (showing {len(preview)} of {len(missing_list)})"
            gams.printLog(
                "[input_treatment][availability] Warning: the following generator(s) have no entries in pAvailability "
                f"and will have implicit availability of 0 (they will not dispatch){more}: {preview}"
            )


    def set_missing_styr_for_existing(db: gt.Container):
        """For existing units (Status=1), set StYr to year before first model year when missing/NaN."""
        param_name = "pGenDataInput"
        if param_name not in db:
            return
        records = db[param_name].records
        if records is None or records.empty:
            return

        header_col, value_col = _detect_header_and_value_columns(records)
        if header_col is None or value_col is None or "g" not in records.columns:
            return

        # First model year comes from y; default to 0 if not available.
        first_year = 0
        if "y" in db and db["y"].records is not None and "y" in db["y"].records:
            yrs = pd.to_numeric(db["y"].records["y"], errors="coerce").dropna()
            if not yrs.empty:
                first_year = int(yrs.min())
        default_styr = first_year - 1

        records = records.copy()
        status_rows = records.loc[records[header_col] == "Status"].copy()
        status_rows["StatusNum"] = pd.to_numeric(status_rows[value_col], errors="coerce")
        existing_gens = set(status_rows.loc[status_rows["StatusNum"] == 1, "g"].dropna().unique())
        if not existing_gens:
            return

        styr_mask = records[header_col] == "StYr"
        styr_rows = records.loc[styr_mask].copy()
        styr_rows["StYrNum"] = pd.to_numeric(styr_rows[value_col], errors="coerce")

        gens_with_styr = set(styr_rows.loc[styr_rows["StYrNum"].notna(), "g"].dropna().unique())
        gens_missing_or_nan = existing_gens - gens_with_styr

        if not gens_missing_or_nan and styr_rows["StYrNum"].notna().all():
            return

        new_rows = []
        updated = False

        # Fill NaN StYr rows for existing gens that have the header but missing value.
        if not styr_rows.empty:
            nan_mask = styr_mask & records[value_col].isna() & records["g"].isin(existing_gens)
            if nan_mask.any():
                records.loc[nan_mask, value_col] = default_styr
                for g_val in records.loc[nan_mask, "g"].unique():
                    gams.printLog(
                        f"[input_treatment][defaults] Filled missing StYr for existing generator {g_val} with {default_styr}."
                    )
                updated = True

        added_gens = []
        # Add StYr rows for existing gens completely lacking the header.
        for g_val in sorted(gens_missing_or_nan):
            template = records.loc[records["g"] == g_val]
            if template.empty:
                continue
            new_row = template.iloc[0].copy()
            new_row[header_col] = "StYr"
            new_row[value_col] = default_styr
            new_rows.append(new_row)
            added_gens.append(g_val)
            updated = True

        if new_rows:
            records = pd.concat([records, pd.DataFrame(new_rows)], ignore_index=True)

        if updated:
            db.data[param_name].setRecords(records)
            _write_back(db, param_name)
            if added_gens:
                gams.printLog(
                    f"[input_treatment][defaults] Added StYr={default_styr} for {len(added_gens)} existing generator(s) "
                    f"(Status=1): {', '.join(sorted(added_gens))}."
                )


    def compute_storage_capacity_from_duration(db: gt.Container):
        """
        Compute CapacityMWh from Capacity * StorageDuration when StorageDuration is specified.

        When a storage unit has StorageDuration filled but CapacityMWh is empty/NaN,
        this function calculates CapacityMWh = Capacity * StorageDuration.
        """
        param_name = "pStorageDataInput"
        if param_name not in db:
            return
        records = db[param_name].records
        if records is None or records.empty:
            return

        # Detect header and value columns (check various possible column names)
        header_col = None
        for candidate in ("pStorageDataHeader", "uni_2"):
            if candidate in records.columns:
                header_col = candidate
                break
        value_col = "value" if "value" in records.columns else None
        # Generator column may be 'g' (after rename) or 'uni' (before rename)
        gen_col = None
        for candidate in ("g", "uni"):
            if candidate in records.columns:
                gen_col = candidate
                break
        if header_col is None or value_col is None or gen_col is None:
            gams.printLog(f"[input_treatment][storage] Skipped: columns not found. Available: {list(records.columns)}")
            return

        records = records.copy()

        # Get Capacity values per generator
        capacity_rows = records.loc[records[header_col] == "Capacity"].copy()
        capacity_rows["CapacityNum"] = pd.to_numeric(capacity_rows[value_col], errors="coerce")
        capacity_map = dict(zip(capacity_rows[gen_col], capacity_rows["CapacityNum"]))

        # Get StorageDuration values per generator
        duration_rows = records.loc[records[header_col] == "StorageDuration"].copy()
        duration_rows["DurationNum"] = pd.to_numeric(duration_rows[value_col], errors="coerce")
        duration_map = dict(zip(duration_rows[gen_col], duration_rows["DurationNum"]))

        # Get CapacityMWh rows
        capmwh_mask = records[header_col] == "CapacityMWh"
        capmwh_rows = records.loc[capmwh_mask].copy()
        capmwh_rows["CapMWhNum"] = pd.to_numeric(capmwh_rows[value_col], errors="coerce")

        # Find generators with StorageDuration but missing/NaN CapacityMWh
        gens_with_duration = {g for g, d in duration_map.items() if pd.notna(d) and d > 0}
        gens_with_capmwh = set(capmwh_rows.loc[capmwh_rows["CapMWhNum"].notna() & (capmwh_rows["CapMWhNum"] > 0), gen_col])

        gens_to_compute = gens_with_duration - gens_with_capmwh

        gams.printLog(f"[input_treatment][storage] gens_with_duration: {gens_with_duration}")
        gams.printLog(f"[input_treatment][storage] gens_with_capmwh: {gens_with_capmwh}")
        gams.printLog(f"[input_treatment][storage] gens_to_compute: {gens_to_compute}")

        if not gens_to_compute:
            return

        updated = False
        new_rows = []
        computed_gens = []

        for g_val in sorted(gens_to_compute):
            capacity = capacity_map.get(g_val)
            duration = duration_map.get(g_val)

            if pd.isna(capacity) or pd.isna(duration) or capacity <= 0 or duration <= 0:
                continue

            computed_mwh = capacity * duration

            # Check if CapacityMWh row exists for this generator
            existing_mask = capmwh_mask & (records[gen_col] == g_val)
            if existing_mask.any():
                # Update existing row
                records.loc[existing_mask, value_col] = computed_mwh
            else:
                # Create new row
                template = records.loc[records[gen_col] == g_val]
                if template.empty:
                    continue
                new_row = template.iloc[0].copy()
                new_row[header_col] = "CapacityMWh"
                new_row[value_col] = computed_mwh
                new_rows.append(new_row)

            computed_gens.append((g_val, capacity, duration, computed_mwh))
            updated = True

        if new_rows:
            records = pd.concat([records, pd.DataFrame(new_rows)], ignore_index=True)

        if updated:
            db.data[param_name].setRecords(records)
            _write_back(db, param_name)
            for g_val, cap, dur, mwh in computed_gens:
                gams.printLog(
                    f"[input_treatment][storage] Computed CapacityMWh={mwh} for {g_val} "
                    f"(Capacity={cap} * StorageDuration={dur})."
                )


    def expand_to_years(db: gt.Container, df: pd.DataFrame, year_set_name: str = "y"):
        """
        Expands a DataFrame to include all years from a GAMS set.

        This function takes a DataFrame without a year dimension and creates a
        cross-product with all years in the specified set, effectively broadcasting
        the values across all years.

        Parameters:
        -----------
        db : gt.Container
            A GAMS Transfer (GT) container that stores the database.
        df : pd.DataFrame
            The DataFrame to expand (should not contain 'y' column).
        year_set_name : str, optional
            The name of the year set in the database (default is "y").

        Returns:
        --------
        pd.DataFrame
            A new DataFrame with the year column added, containing all original
            data repeated for each year.
        """
        if df is None:
            return None

        # Get all years from the database
        year_records = db[year_set_name].records
        if year_records is None or year_records.empty:
            gams.printLog(f"Warning: Year set '{year_set_name}' is empty or not found.")
            return df

        years = year_records.iloc[:, 0].tolist()

        # Create a cross-product of the DataFrame with all years
        df_expanded = pd.concat([df.assign(y=year) for year in years], ignore_index=True)

        # Reorder columns to have 'y' in the expected position (after 'g')
        cols = df_expanded.columns.tolist()
        if 'g' in cols and 'y' in cols:
            cols.remove('y')
            g_idx = cols.index('g')
            cols.insert(g_idx + 1, 'y')
            df_expanded = df_expanded[cols]

        return df_expanded


    def prepare_pavailability(db: gt.Container):
        """Expand pAvailabilityInput(g,q) across years and ensure pAvailability(g,y,q) exists."""
        if "pAvailabilityInput" in db and db["pAvailabilityInput"].records is not None:
            avail_records = db["pAvailabilityInput"].records
            gams.printLog("Expanding pAvailabilityInput to pAvailability with year dimension")
            avail_expanded = expand_to_years(db, avail_records)
            if "pAvailability" not in db:
                db.addParameter("pAvailability", ["g", "y", "q"], records=avail_expanded)
            else:
                db.data["pAvailability"].setRecords(avail_expanded)
            _write_back(db, "pAvailability")
        else:
            if "pAvailability" not in db:
                db.addParameter("pAvailability", ["g", "y", "q"])


    def apply_availability_evolution(db: gt.Container,
                                     evolution_param: str = "pEvolutionAvailability",
                                     availability_param: str = "pAvailability",
                                     year_set: str = "y"):
        """
        Apply year-dependent evolution factors to pAvailability.

        This function:
        1. Reads pEvolutionAvailability(g,y) - sparse, only some generators may have entries
        2. Performs linear interpolation for missing years
        3. Applies the formula: pAvailability(g,y,q) = pAvailability(g,y,q) * (1 + pEvolutionAvailability(g,y))

        Parameters:
        -----------
        db : gt.Container
            A GAMS Transfer (GT) container that stores the database.
        evolution_param : str
            Name of the evolution parameter (default: "pEvolutionAvailability").
        availability_param : str
            Name of the availability parameter to modify (default: "pAvailability").
        year_set : str
            Name of the year set (default: "y").

        Notes:
        ------
        - If pEvolutionAvailability is empty or not present, no changes are made.
        - Default evolution factor is 0 (meaning no change: 1 + 0 = 1).
        - Only generators with entries in pEvolutionAvailability are affected.
        """
        # Check if evolution parameter exists and has data
        if evolution_param not in db:
            gams.printLog(f"{evolution_param} not found in database. Skipping availability evolution.")
            return

        evolution_records = db[evolution_param].records
        if evolution_records is None or evolution_records.empty:
            gams.printLog(f"{evolution_param} is empty. Skipping availability evolution.")
            return

        # Check if availability parameter exists
        if availability_param not in db:
            gams.printLog(f"{availability_param} not found. Skipping availability evolution.")
            return

        avail_records = db[availability_param].records
        if avail_records is None or avail_records.empty:
            gams.printLog(f"{availability_param} is empty. Skipping availability evolution.")
            return

        # Get all model years
        if year_set not in db:
            gams.printLog(f"Year set '{year_set}' not found. Skipping availability evolution.")
            return

        year_records = db[year_set].records
        if year_records is None or year_records.empty:
            return

        target_years = pd.to_numeric(year_records.iloc[:, 0], errors='coerce')
        target_years = np.sort(target_years.dropna().unique())
        if target_years.size == 0:
            return

        gams.printLog(f"Applying availability evolution from {evolution_param}")

        # Prepare evolution data - interpolate for each generator
        evolution_data = evolution_records.copy()
        evolution_data['y'] = pd.to_numeric(evolution_data['y'], errors='coerce')
        evolution_data = evolution_data.dropna(subset=['y', 'value'])

        if evolution_data.empty:
            gams.printLog(f"No valid data in {evolution_param} after cleaning. Skipping.")
            return

        # Get list of generators with evolution factors
        generators_with_evolution = evolution_data['g'].unique()
        gams.printLog(f"  Found evolution factors for {len(generators_with_evolution)} generator(s)")
        gams.printLog(f"  Generators with {evolution_param}: {sorted(generators_with_evolution.tolist())}")

        # Interpolate evolution factors for each generator
        interpolated_evolution = []
        for gen in generators_with_evolution:
            gen_data = evolution_data[evolution_data['g'] == gen].sort_values('y')
            years = gen_data['y'].to_numpy()
            values = gen_data['value'].to_numpy()

            if years.size < 1:
                continue

            if years.size == 1:
                # Single value: use constant for all years
                interpolated_values = np.full(target_years.size, values[0])
            else:
                # Linear interpolation
                interpolated_values = np.interp(target_years, years, values)

            for yr, val in zip(target_years, interpolated_values):
                interpolated_evolution.append({'g': gen, 'y': yr, 'evolution': val})

        if not interpolated_evolution:
            gams.printLog("No evolution factors to apply after interpolation.")
            return

        evolution_df = pd.DataFrame(interpolated_evolution)
        evolution_df['y'] = evolution_df['y'].astype(str)

        # Apply evolution to pAvailability
        avail_data = avail_records.copy()

        # Merge with evolution factors (left join - only matching g,y pairs get evolution)
        merged = avail_data.merge(evolution_df, on=['g', 'y'], how='left')

        # Fill NaN evolution with 0 (no change for generators without evolution factors)
        merged['evolution'] = merged['evolution'].fillna(0)

        # Apply formula: new_value = value * (1 + evolution)
        merged['value'] = merged['value'] * (1 + merged['evolution'])

        # Keep only original columns
        result = merged[avail_data.columns]

        # Update the parameter
        db.data[availability_param].setRecords(result)
        _write_back(db, availability_param)

        gams.printLog(f"  Applied evolution factors to {availability_param} for {len(generators_with_evolution)} generator(s)")


    def prepare_lossfactor(db: gt.Container, 
                                        param_ref="pNewTransmission",
                                        param_loss="pLossFactorInternal",
                                        param_y="y",
                                        column_loss="value"):
        """
        Prepares a loss factor GAMS database parameter from another GAMS database parameter, if a given column is specified in this parameter.

        This function retrieves a parameter from the GAMS database, merges it with a reference 
        parameter (`param_ref`) to associate generators, and extracts a subset of relevant 
        columns for further processing.

        Parameters:
        -----------
        db : gt.Container
            A GAMS Transfer (GT) container that stores the database.
        param_name : str
            The name of the parameter to be retrieved and processed.
        column_loss : str
            Column to be used to specify loss factor when it does not exist.
        param_ref : str, optional
            The name of the reference parameter used for merging (default is "pGenDataInput").

        Returns:
        --------
        pandas.DataFrame or None


        Notes:
        ------

        """

        newtransmission_df = db[param_ref].records
        if newtransmission_df is not None:  # we need to specify loss factor
            newtransmission_loss_df = newtransmission_df.loc[newtransmission_df.thdr == 'LossFactor']
            if not newtransmission_loss_df.empty:  # Loss factor is specified
                if newtransmission_loss_df[column_loss].isna().any():
                    gams.printLog("newtransmission_loss_df")
                    gams.printLog(f"Warning: NaN values found in pNewTransmission, skipping specification of loss factor through pNewTransmission.")
                    if db[param_loss].records is None:
                        raise ValueError(f"Error: Loss factor is not specified through pLossFactorInternal.csv. There is missing data for the model")
                else:
                    gams.printLog(f"Defining {param_loss} based on {param_ref}.")
                    # write loss_factor by expanding the column newtransmission_df with header param_y (as columns)
                    y = db[param_y].records
                    y_index = y['y'].tolist()

                    loss_factor_df = newtransmission_loss_df.set_index(['z', 'z2'])[column_loss].to_frame()
                    
                    for year in y_index:
                        loss_factor_df[year] = loss_factor_df[column_loss]

                    # Drop the original column_loss since it's now spread across all years
                    loss_factor_df = loss_factor_df.drop(columns=[column_loss]).stack().reset_index().rename(columns={'level_2': 'y', 0: 'value'})

                    db.data[param_loss].setRecords(loss_factor_df)
                    _write_back(db, param_loss)
                    
                # check that there are no NaN values. Otherwise, skip this step, and check that db[LossFactor ].records exists. If it is not the case, raise an error. If this exists, do nothing. 
            else:  # Loss factor is not specified
                if db[param_loss].records is None:
                    raise ValueError(f"Error: Loss factor is not specified through pLossFactorInternal.csv. There is missing data for the model")


    def interpolate_time_series_parameters(db: gt.Container,
                                           param_names,
                                           year_param: str = "y",
                                           year_column: str = "y"):
        """Linearly interpolate parameters across all model years; assume data is mostly clean."""
        if year_param not in db:
            return

        year_records = db[year_param].records
        if year_records is None or year_column not in year_records:
            return

        target_years = pd.to_numeric(year_records[year_column], errors='coerce')
        target_years = np.sort(target_years.dropna().unique())
        if target_years.size == 0:
            return

        for param_name in param_names:
            if param_name not in db:
                continue

            records = db[param_name].records
            if records is None or records.empty or year_column not in records:
                continue

            # Save original column order from records
            original_col_order = list(records.columns)

            data = records.copy()
            data[year_column] = pd.to_numeric(data[year_column], errors='coerce')
            data = data.dropna(subset=[year_column, "value"])
            if data.empty:
                continue

            group_cols = [c for c in original_col_order if c not in (year_column, "value")]

            if group_cols:
                grouped = data.groupby(group_cols, dropna=False, sort=False, observed=False)
            else:
                grouped = [(None, data)]

            result_frames = []
            for key, group in grouped:
                group = group.sort_values(year_column)
                years = group[year_column].to_numpy()
                values = group["value"].to_numpy()
                if years.size < 2:
                    continue

                interpolated = np.interp(target_years, years, values)
                frame = pd.DataFrame({year_column: target_years, "value": interpolated})

                if group_cols:
                    if not isinstance(key, tuple):
                        key = (key,)
                    for col_name, col_value in zip(group_cols, key):
                        frame[col_name] = col_value
                    frame = frame[original_col_order]
                else:
                    frame = frame[[year_column, "value"]]

                result_frames.append(frame)

            if not result_frames:
                continue

            final = pd.concat(result_frames, ignore_index=True)
            db.data[param_name].setRecords(final)
            target_range = f"{int(target_years[0])}-{int(target_years[-1])}"
            gams.printLog(
                f"[input_treatment][interpolate] Linear interpolation performed on {param_name} to match model years {target_range}."
            )
            _write_back(db, param_name)

    def _apply_generic_capex_interpolated(db: gt.Container):
        """Add capex trajectories for generators missing them, interpolated to model years.
        
        Generic data is already in long format (tech, f, y, value) from GAMS.
        This handles interpolation to model years.
        Must run AFTER defaults pipeline to catch any remaining gaps.
        """
        generic = load_generic_defaults(db, gams.printLog)
        if not generic or "capex" not in generic:
            return
        
        if "pCapexTrajectories" not in db:
            return
        records = db["pCapexTrajectories"].records
        if records is None or records.empty:
            return
        
        gen_records = db["pGenDataInput"].records
        if gen_records is None or gen_records.empty:
            return

        # Get model years
        if "y" not in db or db["y"].records is None:
            return
        model_years = [int(y) for y in db["y"].records.iloc[:, 0].tolist()]

        # Build (g -> tech, fuel) mapping
        gen_meta = gen_records[["g", "tech", "f"]].drop_duplicates(subset=["g"])

        # Find generators missing capex trajectories
        existing_gens = set(records["g"].unique())
        all_gens = set(gen_meta["g"].unique())
        missing_gens = all_gens - existing_gens

        if not missing_gens:
            return

        # Generic capex is already in long format: (tech, f, y, value)
        generic_long = generic["capex"].copy()
        generic_long["y"] = generic_long["y"].astype(int)

        # Join generators with generic values
        gen_with_generic = gen_meta.merge(generic_long, on=["tech", "f"], how="left")

        # Interpolate to model years for each missing generator
        new_rows = []
        for g in missing_gens:
            gen_data = gen_with_generic[gen_with_generic["g"] == g].dropna(subset=["value"])
            if gen_data.empty:
                continue
            gen_data = gen_data.sort_values("y")
            years = gen_data["y"].to_numpy()
            values = gen_data["value"].to_numpy()
            if len(years) < 2:
                interp_values = np.full(len(model_years), values[0] if len(values) > 0 else 1.0)
            else:
                interp_values = np.interp(model_years, years, values)
            for y, v in zip(model_years, interp_values):
                new_rows.append({"g": g, "y": str(y), "value": v})

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            updated = pd.concat([records, new_df], ignore_index=True)
            db.data["pCapexTrajectories"].setRecords(updated)
            _write_back(db, "pCapexTrajectories")
            gams.printLog(f"[input_treatment][generic] Added interpolated capex for {len(missing_gens)} generator(s).")

    def _enrich_defaults_with_generic(db: gt.Container):
        """Add generic (tech,fuel) records to *Default tables for missing combinations.
        
        Generic data is read by GAMS from resources/ and is already in long format.
        This must run BEFORE prepare_generatorbased_parameter to avoid validation errors.
        """
        generic = load_generic_defaults(db, gams.printLog)
        if not generic:
            return
        
        gen_records = db["pGenDataInput"].records
        if gen_records is None or gen_records.empty:
            return
        
        # Get required (zone, tech, fuel) from pGenDataInput
        required = gen_records[["z", "tech", "f"]].drop_duplicates()
        
        # Enrich pGenDataInputDefault
        # GAMS-loaded generic data is already in long format: (tech, f, pGenDataInputHeader, value)
        if "gendata" in generic and "pGenDataInputDefault" in db:
            gendata_default = db["pGenDataInputDefault"].records
            generic_gendata = generic["gendata"]  # Already has (tech, f, pGenDataInputHeader, value)

            if gendata_default is None or gendata_default.empty:
                # pGenDataInputDefault is empty - create it entirely from generic data
                to_add = required.merge(generic_gendata, on=["tech", "f"], how="inner")
                if not to_add.empty:
                    db.data["pGenDataInputDefault"].setRecords(to_add)
                    _write_back(db, "pGenDataInputDefault")
                    n_combos = to_add[["z", "tech", "f"]].drop_duplicates().shape[0]
                    gams.printLog(f"[input_treatment][generic] Created pGenDataInputDefault with {n_combos} (zone, tech, fuel) combination(s) from generic.")
            else:
                # pGenDataInputDefault has data - only add missing combinations
                gendata_keys = gendata_default[["z", "tech", "f"]].drop_duplicates()
                merged = required.merge(gendata_keys, on=["z", "tech", "f"], how="left", indicator=True)
                missing = merged[merged["_merge"] == "left_only"][["z", "tech", "f"]]

                if not missing.empty:
                    # Join missing combinations with generic values
                    to_add = missing.merge(generic_gendata, on=["tech", "f"], how="inner")
                    if not to_add.empty:
                        updated = pd.concat([gendata_default, to_add], ignore_index=True)
                        db.data["pGenDataInputDefault"].setRecords(updated)
                        _write_back(db, "pGenDataInputDefault")
                        n_combos = to_add[["z", "tech", "f"]].drop_duplicates().shape[0]
                        gams.printLog(f"[input_treatment][generic] Enriched pGenDataInputDefault with {n_combos} (zone, tech, fuel) combination(s) from generic.")
        
        # Enrich pAvailabilityDefault
        # GAMS-loaded generic data is (tech, f, value) - single annual value
        # Get seasons from pHours
        seasons_from_hours = []
        if "pHours" in db:
            phours_records = db["pHours"].records
            if phours_records is not None and not phours_records.empty:
                q_col = "q" if "q" in phours_records.columns else phours_records.columns[0]
                seasons_from_hours = phours_records[q_col].unique().tolist()

        if "availability" in generic and "pAvailabilityDefault" in db and seasons_from_hours:
            avail_default = db["pAvailabilityDefault"].records
            generic_avail = generic["availability"]  # (tech, f, value)

            if avail_default is None or avail_default.empty:
                # pAvailabilityDefault is empty - create it entirely from generic data
                to_add = required.merge(generic_avail, on=["tech", "f"], how="inner")
                if not to_add.empty:
                    new_records = []
                    for _, row in to_add.iterrows():
                        for q in seasons_from_hours:
                            new_records.append({"z": row["z"], "tech": row["tech"], "f": row["f"],
                                              "q": q, "value": row["value"]})
                    if new_records:
                        new_df = pd.DataFrame(new_records)
                        db.data["pAvailabilityDefault"].setRecords(new_df)
                        _write_back(db, "pAvailabilityDefault")
                        n_combos = to_add[["z", "tech", "f"]].drop_duplicates().shape[0]
                        gams.printLog(f"[input_treatment][generic] Created pAvailabilityDefault with {n_combos} (zone, tech, fuel) combination(s) from generic.")
            else:
                # pAvailabilityDefault has data - only add missing combinations
                avail_keys = avail_default[["z", "tech", "f"]].drop_duplicates()
                merged = required.merge(avail_keys, on=["z", "tech", "f"], how="left", indicator=True)
                missing = merged[merged["_merge"] == "left_only"][["z", "tech", "f"]]

                if not missing.empty:
                    # Join missing combinations with generic values, expand to all seasons
                    to_add = missing.merge(generic_avail, on=["tech", "f"], how="inner")
                    if not to_add.empty:
                        new_records = []
                        for _, row in to_add.iterrows():
                            for q in seasons_from_hours:
                                new_records.append({"z": row["z"], "tech": row["tech"], "f": row["f"],
                                                  "q": q, "value": row["value"]})
                        if new_records:
                            new_df = pd.DataFrame(new_records)
                            updated = pd.concat([avail_default, new_df], ignore_index=True)
                            db.data["pAvailabilityDefault"].setRecords(updated)
                            _write_back(db, "pAvailabilityDefault")
                            n_combos = to_add[["z", "tech", "f"]].drop_duplicates().shape[0]
                            gams.printLog(f"[input_treatment][generic] Enriched pAvailabilityDefault with {n_combos} (zone, tech, fuel) combination(s) from generic.")
        
        # Enrich pCapexTrajectoriesDefault
        # GAMS-loaded generic data is already in long format: (tech, f, y, value)
        if "capex" in generic and "pCapexTrajectoriesDefault" in db:
            capex_default = db["pCapexTrajectoriesDefault"].records
            generic_capex = generic["capex"]  # Already has (tech, f, y, value)

            if capex_default is None or capex_default.empty:
                # pCapexTrajectoriesDefault is empty - create it entirely from generic data
                # for all required (zone, tech, fuel) combinations
                to_add = required.merge(generic_capex, on=["tech", "f"], how="inner")
                if not to_add.empty:
                    db.data["pCapexTrajectoriesDefault"].setRecords(to_add)
                    _write_back(db, "pCapexTrajectoriesDefault", eps_to_zero=False)
                    n_combos = to_add[["z", "tech", "f"]].drop_duplicates().shape[0]
                    gams.printLog(f"[input_treatment][generic] Created pCapexTrajectoriesDefault with {n_combos} tech-fuel combination(s) from generic.")
            else:
                # pCapexTrajectoriesDefault has data - only add missing combinations
                capex_keys = capex_default[["z", "tech", "f"]].drop_duplicates()
                merged = required.merge(capex_keys, on=["z", "tech", "f"], how="left", indicator=True)
                missing = merged[merged["_merge"] == "left_only"][["z", "tech", "f"]]

                if not missing.empty:
                    to_add = missing.merge(generic_capex, on=["tech", "f"], how="inner")
                    if not to_add.empty:
                        updated = pd.concat([capex_default, to_add], ignore_index=True)
                        db.data["pCapexTrajectoriesDefault"].setRecords(updated)
                        _write_back(db, "pCapexTrajectoriesDefault", eps_to_zero=False)
                        n_combos = to_add[["z", "tech", "f"]].drop_duplicates().shape[0]
                        gams.printLog(f"[input_treatment][generic] Enriched pCapexTrajectoriesDefault with {n_combos} tech-fuel combination(s) from generic.")

        # Enrich pStorageDataInputDefault
        # GAMS-loaded generic data is already in long format: (tech, f, pStorageDataHeader, value)
        if "storagedata" in generic and "pStorageDataInputDefault" in db:
            storagedata_default = db["pStorageDataInputDefault"].records
            # Get required (zone, tech, fuel) from pStorageDataInput instead of pGenDataInput
            stor_records = db["pStorageDataInput"].records
            if stor_records is not None and not stor_records.empty:
                stor_required = stor_records[["z", "tech", "f"]].drop_duplicates()
                if storagedata_default is not None and not storagedata_default.empty:
                    storagedata_keys = storagedata_default[["z", "tech", "f"]].drop_duplicates()
                    merged = stor_required.merge(storagedata_keys, on=["z", "tech", "f"], how="left", indicator=True)
                    missing = merged[merged["_merge"] == "left_only"][["z", "tech", "f"]]
                    
                    if not missing.empty:
                        generic_storagedata = generic["storagedata"]  # Already has (tech, f, pStorageDataHeader, value)
                        # Join missing combinations with generic values
                        to_add = missing.merge(generic_storagedata, on=["tech", "f"], how="inner")
                        if not to_add.empty:
                            updated = pd.concat([storagedata_default, to_add], ignore_index=True)
                            db.data["pStorageDataInputDefault"].setRecords(updated)
                            _write_back(db, "pStorageDataInputDefault")
                            n_combos = to_add[["z", "tech", "f"]].drop_duplicates().shape[0]
                            gams.printLog(f"[input_treatment][generic] Enriched pStorageDataInputDefault with {n_combos} tech-fuel combination(s) from generic.")

    # Create a GAMS workspace and database
    db = gt.Container(gams.db)
    # Normalize generator column for pGenDataInput (rename 'uni' -> 'g' if needed).
    if "pGenDataInput" in db:
        _records = db["pGenDataInput"].records
        if _records is not None and "uni" in _records.columns and "g" not in _records.columns:
            gams.printLog("[input_treatment] Renaming pGenDataInput column 'uni' -> 'g'.")
            db["pGenDataInput"].setRecords(_records.rename(columns={"uni": "g"}))

    settings_flags = {}
    # Pull scalar switches from pSettings so model toggles can be driven by the CSV.
    try:
        settings_records = db["pSettings"].records
    except KeyError:
        settings_records = None
    if settings_records is not None and not settings_records.empty:
        settings_flags = dict(
            zip(settings_records["pSettingsHeader"], settings_records["value"])
        )

    def _truthy(value) -> bool:
        """Interpret pSettings/env values as booleans (non-zero numeric or yes/true/on)."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return False
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return False
        try:
            return float(value) != 0
        except (TypeError, ValueError):
            return str(value).strip().lower() in {"1", "true", "yes", "on"}

    # Auto-fill hydro availability can be enabled via (in priority order):
    # 1) function arg, 2) env var, 3) pSettings flag.
    setting_auto_fill_hydro = _truthy(settings_flags.get("EPM_FILL_HYDRO_AVAILABILITY"))
    env_flag = str(os.environ.get("EPM_FILL_HYDRO_AVAILABILITY", "")).strip().lower()
    env_auto_fill = env_flag in {"1", "true", "yes", "on"}
    auto_fill_missing_hydro = (
        fill_missing_hydro_availability or env_auto_fill or setting_auto_fill_hydro
    )
    if setting_auto_fill_hydro and not (fill_missing_hydro_availability or env_auto_fill):
        gams.printLog(
            "Hydro availability auto-fill enabled via pSettings (EPM_FILL_HYDRO_AVAILABILITY)."
        )
    elif env_auto_fill and not fill_missing_hydro_availability:
        gams.printLog(
            "Hydro availability auto-fill enabled via EPM_FILL_HYDRO_AVAILABILITY environment variable."
        )

    # Auto-fill hydro capex uses the same priority order.
    setting_auto_fill_capex = _truthy(settings_flags.get("EPM_FILL_HYDRO_CAPEX"))
    capex_flag = str(os.environ.get("EPM_FILL_HYDRO_CAPEX", "")).strip().lower()
    env_capex_fill = capex_flag in {"1", "true", "yes", "on"}
    auto_fill_missing_capex = (
        fill_missing_hydro_capex or env_capex_fill or setting_auto_fill_capex
    )
    if setting_auto_fill_capex and not (fill_missing_hydro_capex or env_capex_fill):
        gams.printLog(
            "Hydro capex auto-fill enabled via pSettings (EPM_FILL_HYDRO_CAPEX)."
        )
    elif env_capex_fill and not fill_missing_hydro_capex:
        gams.printLog(
            "Hydro capex auto-fill enabled via EPM_FILL_HYDRO_CAPEX environment variable."
        )

    setting_fill_ror = _truthy(settings_flags.get("EPM_FILL_ROR_FROM_AVAILABILITY"))
    ror_flag = str(os.environ.get("EPM_FILL_ROR_FROM_AVAILABILITY", "")).strip().lower()
    env_fill_ror = ror_flag in {"1", "true", "yes", "on"}
    fill_ror_from_availability = setting_fill_ror or env_fill_ror
    if setting_fill_ror and not env_fill_ror:
        gams.printLog(
            "ROR from availability auto-fill enabled via pSettings (EPM_FILL_ROR_FROM_AVAILABILITY)."
        )
    elif env_fill_ror:
        gams.printLog(
            "ROR from availability auto-fill enabled via EPM_FILL_ROR_FROM_AVAILABILITY environment variable."
        )

    filter_inputs_to_allowed_zones(
        db,
        log_func=gams.printLog,
        write_back=lambda name: _write_back(db, name),
    )

    _step("Zero capacity for invalid generator status")
    zero_capacity_for_invalid_generator_status(db)
    
    _step("Zero capacity for invalid transmission status")
    zero_capacity_for_invalid_transmission_status(db)
    
    _step("Interpolate time series parameters")
    interpolate_time_series_parameters(db, YEARLY_OUTPUT)
    
    _step("Monitor hydro availability")
    monitor_hydro_availability(db, auto_fill_missing_hydro)
    
    _step("Monitor hydro capex")
    monitor_hydro_capex(db, auto_fill_missing_capex)
    
    # Enrich *Default tables with generic (tech,fuel) before validation
    _step("Enrich defaults with generic")
    _enrich_defaults_with_generic(db)
    
    # Complete Generator Data
    _step("Overwrite NaN values for pGenDataInput")
    overwrite_nan_values(db, "pGenDataInput", "pGenDataInputDefault", "pGenDataInputHeader")
    
    # Complete Storage Data
    _step("Overwrite NaN values for pStorageDataInput")
    overwrite_nan_values(db, "pStorageDataInput", "pStorageDataInputDefault", "pStorageDataHeader")

    _step("Compute CapacityMWh from StorageDuration")
    compute_storage_capacity_from_duration(db)

    _step("Set missing StYr for existing generators")
    set_missing_styr_for_existing(db)

    # CSV is read into pAvailabilityInput(g,q), we expand to pAvailability(g,y,q)
    _step("Prepare pAvailability")
    prepare_pavailability(db)

    # Prepare pAvailability by filling missing values with default values
    _step("Defaults for pAvailability")
    default_df = prepare_generatorbased_parameter(db,
                                                  "pAvailabilityDefault",
                                                  cols_tokeep=['q'],
                                                  param_ref="pGenDataInput")
    
    # Expand default values to all years (constant across years)
    default_df = expand_to_years(db, default_df)

    _step("Fill pAvailability with defaults")
    fill_default_value(db, "pAvailability", default_df)
    
    _step("Warn missing availability")
    warn_missing_availability(gams, db)

    # Apply evolution factors to pAvailability
    _step("Apply availability evolution")
    apply_availability_evolution(db)

    # Prepare pCapexTrajectories by filling missing values with default values
    _step("Defaults for pCapexTrajectories")
    default_df = prepare_generatorbased_parameter(db, 
                                                  "pCapexTrajectoriesDefault",
                                                  cols_tokeep=['y'],
                                                  param_ref="pGenDataInput")
                                                                                                
    _step("Fill pCapexTrajectories with defaults")
    fill_default_value(db, "pCapexTrajectories", default_df)

    # Apply interpolated capex for any remaining generators (handles year interpolation)
    _step("Apply generic capex (interpolated)")
    _apply_generic_capex_interpolated(db)

    gams.printLog("-" * 60)
    gams.printLog("[input_treatment] completed")
    gams.printLog("=" * 60)

    # LossFactor must be defined through a specific csv
    # prepare_lossfactor(db, "pNewTransmission", "pLossFactorInternal", "y", "value")


if __name__ == "__main__":

    DEFAULT_GDX = os.path.join("epm", "test", "input.gdx")
    output_gdx = os.path.join("epm", "test", "input_treated.gdx")
    DEFAULT_GDX = os.path.join("epm", "test", "input.gdx")
    output_gdx = os.path.join("epm", "test", "input_treated.gdx")

    container = gt.Container()
    container.read(DEFAULT_GDX)
    apply_debug_column_renames(container)
    
    # SimpleNamespace fakes the small slice of the GAMS API we need for debugging.
    # It is just enough for tests and does not behave like a full GAMS runtime.
    dummy_gams = SimpleNamespace(db=container, printLog=lambda msg: print(str(msg)))
    
    # Run the input treatment process
    run_input_treatment(dummy_gams)
    
    container.write(output_gdx)
