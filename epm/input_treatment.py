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
import gams.transfer as gt
import numpy as np
import pandas as pd
from types import SimpleNamespace

YEARLY_OUTPUT = [
    'pDemandForecast',
    'pCapexTrajectories',
    'pTradePrice',
    'pTransferLimit'
]

ZONE_RESTRICTED_PARAMS = {
    "pGenDataInput": ("z",),
    "pGenDataInputDefault": ("z",),
    "pAvailabilityDefault": ("z",),
    "pCapexTrajectoriesDefault": ("z",),
    "pDemandForecast": ("z",),
    "pNewTransmission": ("z", "z2"),
    "pLossFactorInternal": ("z", "z2"),
    "pTransferLimit": ("z", "z2"),
}

COLUMN_RENAME_MAP = {
    "pGenDataInput": {"uni": "pGenDataInputHeader", "gen": "g", "zone": "z", 'fuel': 'f'},
    "pGenDataInputDefault": {"uni": "pGenDataInputHeader", "gen": "g", "zone": "z", 'fuel': 'f'},
    "pAvailabilityInput": {"uni": "q", "gen": "g"},
    "pAvailability": {"uni": "q", "gen": "g"},
    "pAvailabilityDefault": {"uni": "q", "zone": "z"},
    "pEvolutionAvailability": {"uni": "y", "gen": "g"},
    "pCapexTrajectoriesDefault": {"uni": "y", "zone": "z"},
    "pDemandForecast": {"uni": "y", "zone": "z"},
    "pNewTransmission": {"From": "z", "To": "z2", "uni": "pTransmissionHeader"},
    "zcmap": {"country": "c", "zone": "z"},
    "pSettings": {"Abbreviation": "pSettingsHeader"},
    "pDemandForecast": {'type': 'pe', 'uni': 'y', 'zone': 'z'},
    "pTransferLimit": {"From": "z", "To": "z2", "uni": "y"},
    "pHours": {'season': 'q', 'daytype': 'd', 'uni': 't'},
    "pLossFactorInternal": {"zone1": "z", "zone2": "z2", "uni": "y"},
    'pPlanningReserveMargin': {'uni': 'c'},
    'ftfindex': {'fuel': 'f'},
    "pStorageDataInput": {'gen_0': 'g', 'uni_2': 'pStorageDataHeader'},
    'pTechData': {'Technology': 'tech'},
    "pVREGenProfile": {"gen": "g", "uni": "t"}
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
    # If we cannot infer allowed zones we leave tables untouched.
    allowed_zones = _get_allowed_zones(db)
    if not allowed_zones:
        return

    for param_name, zone_cols in ZONE_RESTRICTED_PARAMS.items():
        if param_name not in db:
            continue
        records = db[param_name].records
        if records is None or records.empty:
            continue
        if any(col not in records.columns for col in zone_cols):
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


def run_input_treatment(gams,
                        fill_missing_hydro_availability: bool = False,
                        fill_missing_hydro_capex: bool = False):

    def _write_back(db: gt.Container, param_name: str):
        """Copy updates back to whatever database the caller provided.

        A full model run hands in a real gams.GamsDatabase, while our debug path
        only supplies a gt.Container. We support both without needing the GAMS runtime.
        """
        target_db = gams.db
        records = db[param_name].records
        row_count = 0 if records is None else len(records)
        target_kind = type(target_db).__name__

        if isinstance(target_db, gt.Container):
            gams.printLog(
                f"[input_treatment] {param_name}: wrote {row_count} row(s) into {target_kind} (container path)."
            )
            target_db.data[param_name].setRecords(records)
            return

        # When writing back to a real GAMS database we must clear the symbol
        # first; db.write() only overwrites tuples that still exist, so stale
        # rows would otherwise survive our filters.
        symbol = target_db[param_name]
        symbol.clear()
        db.write(target_db, [param_name])
        
        live_snapshot = gt.Container(target_db)
        current = live_snapshot[param_name].records
        current_count = 0 if current is None else len(current)
        gams.printLog(
            f"[input_treatment] {param_name}: wrote {row_count} row(s) into {target_kind}; "
            f"live db now has {current_count} row(s)."
        )


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

        # Identify the subset of generators with valid Status values.
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
        # Evaluate Status per (z,z2) pair once by pivoting to wide form.
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
            return
        records = db[param_name].records
        if records is None or records.empty:
            return
        if "g" not in records.columns or "pGenDataInputHeader" not in records.columns:
            return

        records = records.copy()
        # Find generators lacking a valid Status and zero only their Capacity rows.
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
        targeted = ", ".join(sorted(invalid_gens))
        gams.printLog(
            f"[input_treatment][status] Setting Capacity=0 for {mask.sum()} generator row(s) "
            f"with invalid/missing Status (allowed values: 1, 2, 3); generators: {targeted}."
        )


    def zero_capacity_for_invalid_transmission_status(db: gt.Container):
        """Set Capacity=0 for transmission corridors with Status missing or 0."""
        param_name = "pNewTransmission"
        if param_name not in db:
            return
        records = db[param_name].records
        if records is None or records.empty:
            return
        if any(col not in records.columns for col in ("z", "z2", "pTransmissionHeader")):
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
            return
        if "Status" not in wide.columns:
            wide["Status"] = np.nan
        status_numeric = pd.to_numeric(wide["Status"], errors="coerce")
        valid_mask = status_numeric.notna() & (status_numeric != 0)
        invalid_pairs = wide.index[~valid_mask].tolist()
        if not invalid_pairs:
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
            return

        gen_records = db["pGenDataInput"].records
        avail_records = db["pAvailabilityInput"].records
        if gen_records is None or gen_records.empty:
            return

        gen_records = gen_records.copy()
        avail_records = avail_records.copy() if avail_records is not None else pd.DataFrame(columns=["g", "q", "value"])

        if "g" not in gen_records.columns or "tech" not in gen_records.columns:
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
                    f"[input_treatment][hydro_avail] Reservoir hydro warning: {len(missing_meta)} generator(s) lack entries in pAvailability. "
                    f"Rebuild the profiles via `{notebook_hint}`."
                )
                _log_zone_summary(
                    gams,
                    "  Missing reservoir capacity-factor rows by zone:",
                    missing_meta.loc[:, ["g"] + ([zone_column] if zone_column else [])],
                    zone_column,
                    zone_label,
                    all_zone_values,
                )

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
                            for row in missing_meta.itertuples():
                                key = (getattr(row, zone_column), row.tech)
                                donor_info = donor_profiles.get(key)
                                if donor_info is None:
                                    gams.printLog(
                                        f"[input_treatment][hydro_avail] No donor found to auto-fill {row.g} ({row.tech}, {zone_label}: {key[0]})."
                                    )
                                    continue
                                addition = donor_info["profile"].copy()
                                addition["g"] = row.g
                                new_entries.append(addition.loc[:, ["g", "q", "value"]])
                                gams.printLog(
                                    f"[input_treatment][hydro_avail] Auto-filled availability for {row.g} ({row.tech}, {zone_label}: {key[0]}) "
                                    f"using {donor_info['source']}."
                                )

                            if new_entries:
                                updated_availability = pd.concat([avail_records] + new_entries, ignore_index=True)
                                db.data["pAvailabilityInput"].setRecords(updated_availability)
                                _write_back(db, "pAvailabilityInput")
                            else:
                                gams.printLog("[input_treatment][hydro_avail] Auto-fill finished: no records were added.")

        # --- ROR: check pVREgenProfile -------------------------------------------
        ror_meta = _extract_hydro_meta({"ROR"})
        if ror_meta.empty:
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
            return

        # Compare ROR generators in pGenDataInput with those that have hourly profiles.
        missing_ror = ror_meta.loc[~ror_meta["g"].isin(provided_ror)]
        if missing_ror.empty:
            gams.printLog(
                f"[input_treatment][ror_profile] ROR availability check: all {len(ror_meta)} generator(s) defined in pGenDataInput "
                "have hourly profiles in pVREgenProfile."
            )
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

        gams.printLog(
            f"[input_treatment][ror_profile] ROR availability warning: {len(missing_ror)} generator(s) lack hourly profiles in pVREgenProfile. "
            f"Update the hydro notebook at `{notebook_hint}`."
        )
        _log_zone_summary(
            gams,
            "  Missing ROR hourly profiles by zone:",
            missing_ror.loc[:, ["g"] + ([zone_column] if zone_column else [])],
            zone_column,
            zone_label,
            all_zone_values,
        )

        # Fallback: if a custom pAvailability exists, populate pVREgenProfile with flat
        # profiles derived from that availability (one value copied across all hours).
        if fill_ror_from_availability and "pAvailability" in db:
            avail_df = db["pAvailability"].records
            if avail_df is not None and not avail_df.empty and {"g", "q", "value"}.issubset(avail_df.columns):
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
                        db.data["pAvailability"].setRecords(updated_avail)
                        _write_back(db, "pAvailability")

                        gams.printLog(
                            f"[input_treatment][ror_profile] Auto-filled ROR profiles for {len(filled_gens)} generator(s) in pVREgenProfile "
                            f"using their pAvailability values (steady production assumed): {', '.join(sorted(filled_gens))}."
                        )


    def monitor_hydro_capex(db: gt.Container, auto_fill: bool):
        """Ensure hydro capex is specified for committed/candidate plants."""
        if "pGenDataInput" not in db:
            return

        records = db["pGenDataInput"].records
        if records is None or records.empty:
            return
        records = records.copy()

        header_col, value_col = _detect_header_and_value_columns(records)
        if header_col is None or value_col is None:
            return
        if "g" not in records.columns or "tech" not in records.columns:
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
            return

        index_cols = ["g"]
        if zone_column:
            index_cols.append(zone_column)
        index_cols.append("tech")

        # Pivot to wide to inspect Status/Capex side by side for each generator.
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
            return

        gams.printLog(
            f"Hydro capex warning: {len(missing_capex)} generator(s) in {sorted(target_techs)} "
            "with status 2 or 3 have no Capex defined."
        )
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
            "  Missing hydro capex entries by zone:",
            missing_capex.loc[:, ["g"] + ([zone_column] if zone_column else [])],
            zone_column,
            zone_label,
            all_zone_values,
        )

        if not auto_fill:
            return

        if zone_column is None:
            gams.printLog("Capex auto-fill skipped: cannot identify zone column in pGenDataInput.")
            return

        donors = pivot[target_status & tech_mask & pivot["Capex"].notna()]
        if donors.empty:
            gams.printLog("Capex auto-fill skipped: no donor hydro plants with Capex defined.")
            return

        # Use mean Capex by (zone, tech) as donor values.
        donor_lookup = (
            donors.groupby([zone_column, "tech"], observed=False)["Capex"]
            .mean()
            .dropna()
            .to_dict()
        )

        updated = False
        new_rows = []
        for row in missing_capex.itertuples():
            key = (getattr(row, zone_column), row.tech)
            donor_info = donor_lookup.get(key)
            if donor_info is None:
                gams.printLog(
                    f"  -> No donor Capex found for {row.g} ({row.tech}, {zone_label}: {key[0]}). "
                    "Provide values in the hydro capex input or extend `pGenDataInput`."
                )
                continue
            donor_capex = donor_info
            row_mask = (records["g"] == row.g) & (records[header_col] == "Capex")
            if row_mask.any():
                records.loc[row_mask, value_col] = donor_capex
            else:
                template = records.loc[records["g"] == row.g]
                if template.empty:
                    gams.printLog(f"  -> Capex auto-fill skipped for {row.g}; generator rows not found.")
                    continue
                new_row = template.iloc[0].copy()
                new_row[header_col] = "Capex"
                new_row[value_col] = donor_capex
                new_rows.append(new_row)
            updated = True
            gams.printLog(
                f"  -> Auto-filled Capex for {row.g} ({row.tech}, {zone_label}: {key[0]}) "
                f"using the mean value {donor_capex:.3f} from existing generators in the same zone and technology."
            )

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
        if param_df is None:
            gams.printLog('[input_treatment][defaults] {} empty so no effect'.format(param_name))
            return None
            
        gams.printLog('Adding generator reference to {}'.format(param_name))
        
        # Identify common columns between param_df and ref_df, excluding "value"
        columns = [c for c in param_df.columns if c != "value" and c in ref_df.columns]
        
        # Keep only the generator column and common columns in ref_df
        ref_df = ref_df.loc[:, [column_generator] + columns]
            
        # Remove duplicate rows in the reference DataFrame
        ref_df = ref_df.drop_duplicates()

        # Merge generator IDs into the parameter via shared columns
        param_df = pd.merge(ref_df, param_df, how='left', on=columns)
                
        if param_df['value'].isna().any():
            missing_rows = param_df[param_df['value'].isna()]  # Get rows with NaN values
            gams.printLog(f"Error: missing values found in '{param_name}'. This indicates that some generator-year combinations expected by the model are not provided in the input data. Generators in {param_ref} without default values are:")
            gams.printLog(missing_rows.to_string())  # Print the rows where 'value' is NaN
            raise ValueError(f"Missing values in default is not permitted. To fix this bug ensure that all combination in {param_name} are included.")

        # Select only the necessary columns for the final output
        param_df = param_df.loc[:, [column_generator] + cols_tokeep + ["value"]]    
        
        return param_df


    def fill_default_value(db: gt.Container, param_name: str, default_df: pd.DataFrame, fillna=1):
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
        
        gams.printLog("[input_treatment][defaults] Modifying {} with default values".format(param_name))
        
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
        _write_back(db, param_name)


    def warn_missing_availability(gams, db: gt.Container):
        """Warn if generators have no pAvailability rows (implicit availability=0)."""
        if "pGenDataInput" not in db or "pAvailability" not in db:
            return

        gen_records = db["pGenDataInput"].records
        avail_records = db["pAvailability"].records
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
                # If any NaN values exist, fall back to the dedicated pLossFactorInternal input.
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

            data = records.copy()
            data[year_column] = pd.to_numeric(data[year_column], errors='coerce')
            data = data.dropna(subset=[year_column, "value"])
            if data.empty:
                continue

            group_cols = [c for c in data.columns if c not in (year_column, "value")]
            if group_cols:
                # Interpolate separately for each unique combination of non-year columns.
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
                    frame = frame[group_cols + [year_column, "value"]]
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
        
    # Create a GAMS workspace and database
    db = gt.Container(gams.db)

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

    fill_ror_from_availability = _truthy(
        settings_flags.get("EPM_FILL_ROR_FROM_AVAILABILITY")
    )

    filter_inputs_to_allowed_zones(
        db,
        log_func=gams.printLog,
        write_back=lambda name: _write_back(db, name),
    )

    # Sanitize generators and transmission corridors before filling defaults.
    zero_capacity_for_invalid_generator_status(db)
    
    zero_capacity_for_invalid_transmission_status(db)
    
    # Expand any sparse yearly series to full model years.
    interpolate_time_series_parameters(db, YEARLY_OUTPUT)
    
    # Hydro-specific data quality checks with optional auto-fill.
    monitor_hydro_availability(db, auto_fill_missing_hydro)
    
    monitor_hydro_capex(db, auto_fill_missing_capex)
    
    # Complete Generator Data
    overwrite_nan_values(db, "pGenDataInput", "pGenDataInputDefault", "pGenDataInputHeader")
    set_missing_styr_for_existing(db)

    # Prepare pAvailability by filling missing values with default values
    # pAvailability now has year dimension (g,y,q) but input CSV doesn't have 'y'
    # CSV is read into pAvailabilityInput(g,q), we expand to pAvailability(g,y,q)

    # First, expand pAvailabilityInput records (from CSV without 'y') to all years
    if "pAvailabilityInput" in db and db["pAvailabilityInput"].records is not None:
        avail_records = db["pAvailabilityInput"].records
        gams.printLog("Expanding pAvailabilityInput to pAvailability with year dimension")
        avail_expanded = expand_to_years(db, avail_records)
        # Create pAvailability parameter if it doesn't exist, or update it
        if "pAvailability" not in db:
            db.addParameter("pAvailability", ["g", "y", "q"], records=avail_expanded)
        else:
            db.data["pAvailability"].setRecords(avail_expanded)
        _write_back(db, "pAvailability")
    else:
        # No custom availability provided, create empty pAvailability
        if "pAvailability" not in db:
            db.addParameter("pAvailability", ["g", "y", "q"])

    # Prepare default values and expand to all years
    default_df = prepare_generatorbased_parameter(db,
                                                  "pAvailabilityDefault",
                                                  cols_tokeep=['q'],
                                                  param_ref="pGenDataInput")
    # Expand default values to all years (constant across years)
    default_df = expand_to_years(db, default_df)

    fill_default_value(db, "pAvailability", default_df)
    warn_missing_availability(gams, db)

    # Apply evolution factors to pAvailability
    apply_availability_evolution(db)

    # Prepare pCapexTrajectories by filling missing values with default values
    default_df = prepare_generatorbased_parameter(db, 
                                                  "pCapexTrajectoriesDefault",
                                                  cols_tokeep=['y'],
                                                  param_ref="pGenDataInput")
                                                                                                
    fill_default_value(db, "pCapexTrajectories", default_df)


    # LossFactor must be defined through a specific csv
    # prepare_lossfactor(db, "pNewTransmission", "pLossFactorInternal", "y", "value")


if __name__ == "__main__":

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
