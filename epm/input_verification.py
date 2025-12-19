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
from types import SimpleNamespace

import gams.transfer as gt
import pandas as pd

from input_treatment import apply_debug_column_renames, filter_inputs_to_allowed_zones

ESSENTIAL_INPUT = [
    "y",
    "pHours",
    "zcmap",
    "pSettings",
    "pGenDataInput",
    "pFuelPrice",
    "pFuelCarbonContent",
    "pTechFuel",
]

OPTIONAL_INPUT = ["pDemandForecast"]


def _log_input_columns(gams, db):
    """Log columns and domain names for each symbol in the container (compact)."""
    names = sorted(db.data.keys())
    gams.printLog("[input_verification] Columns by symbol (data columns | domains):")
    for name in names:
        try:
            records = db[name].records
            cols = list(records.columns) if records is not None else []
            domains = getattr(db[name], "domain_names", None)
        except Exception:
            cols = "<error>"
            domains = "<error>"
        gams.printLog(f"  {name}: {cols} | domains={domains}")


def run_input_verification(gams):
    """Run the full suite of input validation checks on the current GAMS DB."""
    db = gt.Container(gams.db)

    def _step(title):
        gams.printLog("-" * 60)
        gams.printLog(title)

    gams.printLog("=" * 60)
    gams.printLog("[input_verification] starting")
    gams.printLog("=" * 60)

    _step("Filter inputs to allowed zones")
    filter_inputs_to_allowed_zones(db, log_func=gams.printLog)
    if "zcmap" in db and db["zcmap"].records is not None and not db["zcmap"].records.empty:
        zones_seen = sorted(db["zcmap"].records["z"].dropna().unique())
        gams.printLog(f"[input_verification][zones] Zones included in verification: {zones_seen}")
    else:
        gams.printLog("[input_verification][zones] No zones available; zone-based checks may be skipped.")

    # _log_input_columns(gams, db)
    _step("Required inputs")
    _check_required_inputs(gams, db)
    gams.printLog("[input_verification] Required inputs completed.")
    _step("Settings flags")
    _check_settings_flags(gams, db)
    gams.printLog("[input_verification] Settings flags completed.")
    _step("Settings required entries")
    _check_settings_required_entries(gams, db)
    gams.printLog("[input_verification] Settings required entries completed.")
    # _check_candidate_build_limits(gams, db)
    _step("Hours")
    _check_hours(gams, db)
    gams.printLog("[input_verification] Hours completed.")
    _step("VRE profile")
    _check_vre_profile(gams, db)
    gams.printLog("[input_verification] VRE profile completed.")
    _step("Availability")
    _check_availability(gams, db)
    gams.printLog("[input_verification] Availability completed.")
    _step("Time resolution")
    _check_time_resolution(gams, db)
    gams.printLog("[input_verification] Time resolution completed.")
    _step("Dispatch chronology")
    _warn_dispatch_chronology_consistency(gams, db)
    gams.printLog("[input_verification] Dispatch chronology completed.")
    _step("Demand forecast")
    _check_demand_forecast(gams, db)
    gams.printLog("[input_verification] Demand forecast completed.")
    _step("Fuel prices")
    _check_fuel_price_presence(gams, db)
    gams.printLog("[input_verification] Fuel prices completed.")
    _step("Transfer limits")
    _check_transfer_limits(gams, db)
    gams.printLog("[input_verification] Transfer limits completed.")
    _step("New transmission")
    _check_new_transmission(gams, db)
    gams.printLog("[input_verification] New transmission completed.")
    _step("New transmission zones")
    _check_new_transmission_zones(gams, db)
    gams.printLog("[input_verification] New transmission zones completed.")
    _step("Interconnected mode")
    _check_interconnected_mode(gams, db)
    gams.printLog("[input_verification] Interconnected mode completed.")
    _step("Planning reserves")
    _check_planning_reserves(gams, db)
    gams.printLog("[input_verification] Planning reserves completed.")
    _step("Fuel definitions")
    _check_fuel_definitions(gams, db)
    gams.printLog("[input_verification] Fuel definitions completed.")
    _step("Tech definitions")
    _check_tech_definitions(gams, db)
    gams.printLog("[input_verification] Tech definitions completed.")
    _step("Zone consistency")
    _check_zone_consistency(gams, db)
    gams.printLog("[input_verification] Zone consistency completed.")
    _step("Single zone internal exchange")
    _check_single_zone_internal_exchange(gams, db)
    gams.printLog("[input_verification] Single zone internal exchange completed.")
    _step("External transfer limits")
    _check_external_transfer_limits(gams, db)
    gams.printLog("[input_verification] External transfer limits completed.")
    _step("External transfer settings")
    _check_external_transfer_settings(gams, db)
    gams.printLog("[input_verification] External transfer settings completed.")
    _step("Storage data")
    _check_storage_data(gams, db)
    gams.printLog("[input_verification] Storage data completed.")
    _step("Generation defaults")
    _check_generation_defaults(gams, db)
    gams.printLog("[input_verification] Generation defaults completed.")
    _step("Missing generation default combinations")
    _warn_missing_generation_default_combinations(gams, db)
    _warn_missing_availability_default_combinations(gams, db)
    _warn_missing_build_limits(gams, db)

    gams.printLog("=" * 60)

def _run_input_verification_on_container(container: gt.Container, *, verbose=True, log_func=None):
    """Execute verification on an existing gt.Container and collect logs."""
    collected_logs = []

    def _log(message):
        collected_logs.append(str(message))
        if log_func is not None:
            log_func(message)
        elif verbose:
            print(message)

    dummy_gams = SimpleNamespace(db=container, printLog=_log)
    run_input_verification(dummy_gams)
    return collected_logs


def _check_required_inputs(gams, db):
    """Ensure essential inputs exist and optional inputs log warnings when missing."""
    try:
        for param in ESSENTIAL_INPUT:
            if param not in db:
                msg = f"Error {param} is missing"
                gams.printLog(msg)
                raise ValueError(msg)
            records = db[param].records
            if records is None:
                msg = f"Error {param} is missing"
                gams.printLog(msg)
                raise ValueError(msg)
            if records.empty:
                msg = f"Error {param} is empty"
                gams.printLog(msg)
                raise ValueError(msg)

        for param in OPTIONAL_INPUT:
            if param not in db:
                gams.printLog(f"Warning {param} is missing")
                continue
            records = db[param].records
            if records is None:
                gams.printLog(f"Warning {param} is missing")
            elif records.empty:
                gams.printLog(f"Warning {param} is empty")
    except ValueError:
        raise
    except Exception:
        gams.printLog("Unexpected error in initial")
        raise


def _check_settings_flags(gams, db):
    """Assess toggles in pSettings to emit consistent warnings and errors."""
    try:
        records = db["pSettings"].records
        if records is None or records.empty:
            gams.printLog("[input_verification][settings_flags] Skipped: pSettings missing or empty.")
            return

        settings_map = dict(zip(records["pSettingsHeader"], records["value"]))

        def _to_float(value):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return None
            if pd.isna(numeric_value):
                return None
            return numeric_value

        check_specs = (
            ("fEnableInternalExchange", lambda v: v == 0.0, "disables internal exchanges"),
            ("fRemoveInternalTransferLimit", lambda v: v == 1.0, "removes internal transfer limits"),
            ("fAllowTransferExpansion", lambda v: v == 0.0, "prevents internal transfer expansion"),
            ("fApplyCountryCo2Constraint", lambda v: v == 1.0, "activates country-level CO2 constraint"),
            ("fApplySystemCo2Constraint", lambda v: v == 1.0, "activates system-level CO2 constraint"),
            ("sMinRenewableSharePct", lambda v: v > 0.0, "enforces a minimum renewable share"),
            ("fApplyCapitalConstraint", lambda v: v == 1.0, "activates the capital constraint"),
        )

        for name, predicate, description in check_specs:
            value = _to_float(settings_map.get(name))
            if value is None:
                continue
            if predicate(value):
                gams.printLog(f"Warning: {name}={value:g} {description}.")

        system_co2 = _to_float(settings_map.get("fApplySystemCo2Constraint"))
        country_co2 = _to_float(settings_map.get("fApplyCountryCo2Constraint"))
        if system_co2 is not None and country_co2 is not None:
            if system_co2 == 1 and country_co2 != 0:
                gams.printLog(
                    "Warning: fApplySystemCo2Constraint=1 requires fApplyCountryCo2Constraint=0."
                )
            if country_co2 == 1 and system_co2 != 0:
                gams.printLog(
                    "Warning: fApplyCountryCo2Constraint=1 requires fApplySystemCo2Constraint=0."
                )

        country_spin_series = records.loc[
            records["pSettingsHeader"] == "fApplyCountrySpinReserveConstraint", "value"
        ]
        system_spin_series = records.loc[
            records["pSettingsHeader"] == "fApplySystemSpinReserveConstraint", "value"
        ]
        if not country_spin_series.empty and not system_spin_series.empty:
            try:
                country_spin = float(country_spin_series.iloc[-1])
                system_spin = float(system_spin_series.iloc[-1])
            except (TypeError, ValueError):
                country_spin = None
                system_spin = None
            if country_spin is not None and system_spin is not None:
                if country_spin == 1 and system_spin != 0:
                    msg = (
                        "Error: fApplyCountrySpinReserveConstraint must be 1 only when "
                        "fApplySystemSpinReserveConstraint is 0."
                    )
                    gams.printLog(msg)
                    raise ValueError(msg)
                if system_spin == 1 and country_spin != 0:
                    msg = (
                        "Error: fApplySystemSpinReserveConstraint must be 1 only when "
                        "fApplyCountrySpinReserveConstraint is 0."
                    )
                    gams.printLog(msg)
                    raise ValueError(msg)
    except ValueError:
        raise
    except Exception:
        gams.printLog("Unexpected error while evaluating settings flags")
        raise


def _check_settings_required_entries(gams, db):
    """Ensure mandatory settings entries are present and report notable omissions."""
    try:
        records = db["pSettings"].records
        if records is None:
            gams.printLog("[input_verification][settings_required] Skipped: pSettings missing.")
            return
        if records.empty:
            gams.printLog("[input_verification][settings_required] Skipped: pSettings empty.")
            return

        required_headers = ["VoLL", "ReserveVoLL", "SpinReserveVoLL", "WACC", "DR"]
        warning_headers = ["sIntercoReserveContributionPct", "fCountIntercoForReserves"]
        header_values = list(records["pSettingsHeader"])

        missing = [header for header in required_headers if header not in header_values]
        missing_warning = [header for header in warning_headers if header not in header_values]

        if missing:
            msg = (
                "Error: The following entries are required in pSettings but currently missing: "
                f"{missing}"
            )
            gams.printLog(msg)
            raise ValueError(msg)
        if missing_warning:
            msg = (
                "WARNING: The following entries are set to zero in pSettings: "
                f"{missing_warning}"
            )
            gams.printLog(msg)
    except ValueError:
        raise
    except Exception:
        gams.printLog("Unexpected error when checking pSettings completeness")
        raise


def _check_candidate_build_limits(gams, db):
    """Ensure BuildLimitperYear is positive for candidate plants (status 2/3)."""
    try:
        records = db["pGenDataInput"].records
        if records is None or records.empty:
            return
        df_wide = records
        identifier_column = None
        if {"pGenDataInputHeader", "value"}.issubset(records.columns) and "BuildLimitperYear" not in records.columns:
            if "g" in records.columns:
                identifier_column = "g"
            elif "gen" in records.columns:
                identifier_column = "gen"
            df_wide = (
                records.pivot_table(
                    index=identifier_column or records.index,
                    columns="pGenDataInputHeader",
                    values="value",
                    aggfunc="first",
                    observed=False,
                )
                .reset_index()
            )
            if identifier_column:
                df_wide = df_wide.rename(columns={identifier_column: "g"})

        required_columns = {"Status", "BuildLimitperYear"}
        if not required_columns.issubset(df_wide.columns):
            missing = required_columns - set(df_wide.columns)
            msg = (
                "Error: pGenDataInput is missing columns required for BuildLimitperYear validation: "
                f"{missing}"
            )
            gams.printLog(msg)
            raise ValueError(msg)

        status_numeric = pd.to_numeric(df_wide["Status"], errors="coerce")
        build_limit_numeric = pd.to_numeric(df_wide["BuildLimitperYear"], errors="coerce")
        candidate_mask = status_numeric.isin([2, 3])
        violation_mask = candidate_mask & (build_limit_numeric.isna() | (build_limit_numeric == 0))

        if violation_mask.any():
            violations = df_wide.loc[violation_mask]
            offending_entries = (
                violations["g"].astype(str).tolist()
                if "g" in violations.columns
                else violations.index.tolist()
            )
            msg = (
                "Error: BuildLimitperYear must be strictly positive for candidate plants (Status 2 or 3). "
                f"Offending entries: {offending_entries}"
            )
            gams.printLog(msg)
            raise ValueError(msg)
    except ValueError:
        raise
    except Exception:
        gams.printLog("Unexpected error when validating BuildLimitperYear in pGenDataInput")
        raise


def _check_hours(gams, db):
    """Validate that pHours values are positive and total 8760 hours."""
    try:
        p_hours = db["pHours"]
        records = p_hours.records
        if records is None or records.empty:
            return

        if not (records["value"] > 0).all():
            msg = "Error: Some block duration are negative."
            gams.printLog(msg)
            raise ValueError(msg)
        else:
            gams.printLog("Success: All values pHours positive.")

        total_hours = records["value"].sum()
        if total_hours == 8760:
            gams.printLog("Success: The sum of pHours is exactly 8760.")
        else:
            msg = f"Error: The sum of pHours is {total_hours}, which is not 8760."
            gams.printLog(msg)
            raise ValueError(msg)
    except ValueError:
        raise
    except Exception:
        gams.printLog("Unexpected error in pHours")
        raise


def _check_vre_profile(gams, db):
    """Confirm that VRE capacity factors never exceed unity."""
    try:
        records = db["pVREProfile"].records
        if records is None or records.empty:
            return
        if (records["value"] > 1).any():
            msg = "Error: Capacity factor cannot be greater than 1 in pVREProfile."
            gams.printLog(msg)
            raise ValueError(msg)
        gams.printLog("Success: All pVREProfile values are valid.")
    except ValueError:
        raise
    except Exception:
        gams.printLog("Unexpected error in VREProfile")
        raise


def _check_availability(gams, db):
    """Ensure availability factors remain strictly below one."""
    try:
        # Check pAvailabilityInput (raw CSV data without year dimension)
        records = db["pAvailabilityInput"].records if "pAvailabilityInput" in db else None
        if records is None:
            gams.printLog('pAvailabilityCustom is None. All values come from pAvailabilityDefault')
            return
        if (records["value"] > 1).any():
            msg = "Error: Availability factor cannot be 1 or greater in pAvailability."
            gams.printLog(msg)
            raise ValueError(msg)
        gams.printLog("Success: All pAvailability values are valid.")
    except ValueError:
        raise
    except Exception:
        gams.printLog('Unexpected error in pAvailability')
        raise


def _check_time_resolution(gams, db):
    """Verify that all time-dependent inputs share the same (q, d, t) coordinates."""
    try:
        vars_time = ["pVREProfile", "pVREgenProfile", "pDemandProfile", "pHours"]
        unique_combinations = {}

        for var in vars_time:
            if var not in db:
                continue
            records = db[var].records
            if records is None or records.empty:
                continue
            if not {"q", "d", "t"}.issubset(records.columns):
                continue
            unique_combinations[var] = set(records[["q", "d", "t"]].apply(tuple, axis=1))

        if not unique_combinations:
            return

        first_var = next(iter(unique_combinations))
        reference = unique_combinations[first_var]
        is_consistent = all(reference == combos for combos in unique_combinations.values())

        if is_consistent:
            gams.printLog("Success: All dataframes have the same (q, d, t) combinations.")
            return

        gams.printLog("Mismatch detected! The following differences exist:")
        for var, combos in unique_combinations.items():
            diff = reference ^ combos
            if diff:
                gams.printLog(f"Differences in {var}: {diff}")
        msg = "All dataframes do not have the same (q, d, t) combinations."
        gams.printLog(msg)
        raise ValueError(msg)
    except ValueError:
        raise
    except Exception:
        gams.printLog('Unexpected error when checking time consistency')
        raise


def _warn_dispatch_chronology_consistency(gams, db):
    """Warn when dispatch mappings use (q, d, t) tuples not present in pHours."""
    try:
        p_hours = db["pHours"].records
        if (
            p_hours is None
            or p_hours.empty
            or not {"q", "d", "t"}.issubset(p_hours.columns)
        ):
            return
        valid_hours = set(p_hours[["q", "d", "t"]].apply(tuple, axis=1))

        def _warn_if_outside(param):
            if param not in db:
                return False, set()
            records = db[param].records
            if (
                records is None
                or records.empty
                or not {"q", "d", "t"}.issubset(records.columns)
            ):
                return False, set()
            tuples = set(records[["q", "d", "t"]].apply(tuple, axis=1))
            outside = tuples - valid_hours
            if outside:
                preview = sorted(outside)[:10]
                more = ""
                if len(outside) > len(preview):
                    more = f" (showing {len(preview)} of {len(outside)})"
                gams.printLog(
                    f"Warning: {param} contains (q, d, t) tuples not present in pHours; dispatch mode chronology is inconsistent. "
                    f"Missing tuples: {preview}{more}"
                )
            return True, outside

        checked_any = False
        issues_found = False

        for param in ("mapTS", "pDays"):
            checked, outside = _warn_if_outside(param)
            checked_any = checked_any or checked
            if outside:
                issues_found = True

        if checked_any and not issues_found:
            gams.printLog("Success: mapTS and pDays tuples are contained in pHours.")
    except Exception:
        gams.printLog("Unexpected error when checking dispatch chronology consistency")
        raise


def _check_demand_forecast(gams, db):
    """Inspect pDemandForecast energy-to-peak ratios for unusual values."""
    try:
        if "pDemandForecast" not in db:
            return
        records = db["pDemandForecast"].records
        if records is None:
            gams.printLog("Warning: pDemandForecast is not defined.")
            return
        if records.empty:
            return

        df_pivot = records.pivot(index=["z", "y"], columns="pe", values="value").reset_index()
        df_pivot.columns.name = None
        df_pivot.rename(columns={"Energy": "energy_value", "Peak": "peak_value"}, inplace=True)

        if {"energy_value", "peak_value"}.issubset(df_pivot.columns):
            df_pivot["energy_peak_ratio"] = df_pivot["energy_value"] / df_pivot["peak_value"]
            min_ratio = df_pivot["energy_peak_ratio"].min()
            max_ratio = df_pivot["energy_peak_ratio"].max()
            gams.printLog(
                f"Energy/Peak Demand Ratio - Min: {min_ratio:.2f} & Max: {max_ratio:.2f}"
            )

            if min_ratio < 3 or max_ratio > 10:
                gams.printLog(
                    "WARNING: Energy/Peak Demand Ratio out of expected range [3–10]. "
                    f"Min: {min_ratio:.2f}, Max: {max_ratio:.2f}"
                )
                extreme_rows = df_pivot[
                    (df_pivot["energy_peak_ratio"] < 3) | (df_pivot["energy_peak_ratio"] > 10)
                ]
                for _, row in extreme_rows.iterrows():
                    gams.printLog(
                        f"Extreme Energy/Peak Ratio at zone {row['z']}, year {row['y']}: "
                        f"{row['energy_peak_ratio']:.2f}"
                    )
        else:
            gams.printLog(
                f"Warning: pDemandForecast is missing required columns 'energy'/'peak' after pivot. "
                f"Got columns: {list(df_pivot.columns)}"
            )
    except Exception:
        gams.printLog('Unexpected error when checking pDemandForecast')
        raise


def _check_fuel_price_presence(gams, db):
    """Ensure pFuelPrice is accessible to avoid downstream KeyErrors."""
    try:
        _ = db["pFuelPrice"].records
    except ValueError:
        raise
    except Exception:
        gams.printLog('Unexpected error when checking pFuelPrice')
        raise


def _check_transfer_limits(gams, db):
    """Validate symmetry and seasonal coverage in pTransferLimit."""
    try:
        records = db["pTransferLimit"].records
        if records is None or records.empty:
            return

        topology = records.set_index(["z", "z2"]).index.unique()
        missing_pairs = [(z, z2) for z, z2 in topology if (z2, z) not in topology]
        if missing_pairs:
            missing_pairs_str = "\n".join([f"({z}, {z2})" for z, z2 in missing_pairs])
            msg = (
                "Error: The following (z, z2) pairs are missing their symmetric counterparts (z2, z) "
                f"in 'pTransferLimit':\n{missing_pairs_str}"
            )
            gams.printLog(msg)
            raise ValueError(msg)
        gams.printLog("Success: All (z, z2) pairs in 'pTransferLimit' have their corresponding (z2, z) pairs.")

        p_hours_records = db["pHours"].records
        if p_hours_records is None or p_hours_records.empty:
            return
        seasons = set(p_hours_records["q"].unique())
        season_issues = []

        for (z, z2), group in records.groupby(["z", "z2"], observed=False):
            unique_seasons = set(group["q"].unique())
            missing_seasons = seasons - unique_seasons
            if missing_seasons:
                season_issues.append((z, z2, missing_seasons))

        if season_issues:
            season_issues_str = "\n".join(
                [f"({z}, {z2}): missing seasons {missing}" for z, z2, missing in season_issues]
            )
            msg = (
                "Error: The following (z, z2) pairs do not have all required seasons in 'pTransferLimit':\n"
                f"{season_issues_str}"
            )
            gams.printLog(msg)
            raise ValueError(msg)
        gams.printLog("Success: All (z,z2) pairs contain all required seasons.")
    except ValueError:
        raise
    except Exception:
        gams.printLog('Unexpected error when checking pTransferLimit')
        raise


def _check_new_transmission(gams, db):
    """Confirm each candidate transmission line is unique."""
    try:
        records = db["pNewTransmission"].records
        if records is None or records.empty:
            return

        topology_newlines = records.set_index(["z", "z2"]).index.unique()
        duplicate_transmission = [
            (z, z2) for z, z2 in topology_newlines if (z2, z) in topology_newlines
        ]
        if duplicate_transmission:
            msg = (
                "Error: The following (z, z2) pairs are specified twice in 'pNewTransmission':\n"
                f"{duplicate_transmission} \n This may cause some problems when defining twice the characteristics of additional line."
            )
            gams.printLog(msg)
            raise ValueError(msg)
        gams.printLog("Success: Each candidate transmission line is only specified once in pNewTransmission.")
    except ValueError:
        raise
    except Exception:
        gams.printLog('Unexpected error when checking NewTransmission')
        raise


def _check_new_transmission_zones(gams, db):
    """Ensure zones referenced by new transmission assets exist in zcmap."""
    try:
        records = db["pNewTransmission"].records
        if records is None or records.empty:
            return

        zones_newtransmission = set(records['z'].unique()).union(set(records['z2'].unique()))
        zcmap_records = db["zcmap"].records
        zones_defined = set(zcmap_records['z'].unique()) if zcmap_records is not None else set()
        new_zones = [z for z in zones_newtransmission if z not in zones_defined]
        if new_zones:
            msg = (
                "Warning: The following zones are used to defined new transmission lines in 'pNewTransmission':\n"
                f"{new_zones} \n Tranmission lines involved will not be considered by the model. You should modify zone names to match the names defined in zcmap if you want."
            )
            gams.printLog(msg)
            raise ValueError(msg)
        gams.printLog("Success: Zones in pNewTransmission match the zones defined in zcmap.")
    except ValueError:
        raise
    except Exception:
        gams.printLog('Unexpected error when checking zones in NewTransmission')
        raise


def _check_interconnected_mode(gams, db):
    """Validate that interconnected-mode settings align with topology and losses."""
    try:
        settings_records = db["pSettings"].records
        if settings_records is None or settings_records.empty:
            return

        settings_indexed = settings_records.set_index("pSettingsHeader")
        if "fEnableInternalExchange" not in settings_indexed.index:
            return
        try:
            interconnected_flag = float(settings_indexed.loc["fEnableInternalExchange", "value"])
        except (TypeError, ValueError):
            interconnected_flag = settings_indexed.loc["fEnableInternalExchange", "value"]
        if not interconnected_flag:
            return

        loss_factor_records = db["pLossFactorInternal"].records
        if loss_factor_records is None or loss_factor_records.empty:
            msg = "Error: Interconnected mode is activated, but LossFactor is empty"
            gams.printLog(msg)
            raise ValueError(msg)

        topology = set()
        new_transmission = db["pNewTransmission"].records
        if new_transmission is not None and not new_transmission.empty:
            topology |= set(new_transmission.set_index(['z', 'z2']).index.unique())
        transfer_limit = db["pTransferLimit"].records
        if transfer_limit is not None and not transfer_limit.empty:
            topology |= set(transfer_limit.set_index(['z', 'z2']).index.unique())
        if not topology:
            msg = (
                "Error: Interconnected mode is activated, but both TransferLimit and NewTransmission are empty."
            )
            gams.printLog(msg)
            raise ValueError(msg)

        final_topology = set()
        for (z, z2) in topology:
            if (z2, z) not in final_topology:
                final_topology.add((z, z2))

        loss_factor_topology = set(loss_factor_records.set_index(['z', 'z2']).index.unique())
        loss_factor_indexed = loss_factor_records.set_index(['z', 'z2'])

        missing_lines = [
            line for line in final_topology
            if line not in loss_factor_topology and (line[1], line[0]) not in loss_factor_topology
        ]
        if missing_lines:
            msg = f"Error: The following lines in topology are missing from pLossFactorInternal: {missing_lines}"
            gams.printLog(msg)
            raise ValueError(msg)
        gams.printLog("Success: All transmission lines have a lossfactor specified.")

        duplicate_mismatches = []
        for (z, z2) in loss_factor_topology:
            if (z2, z) in loss_factor_topology:
                row1 = loss_factor_indexed.loc[[(z, z2)]].sort_index()
                row2 = loss_factor_indexed.loc[[(z2, z)]].sort_index()
                row1 = row1.reset_index().sort_values(by="y").drop(columns=['z', 'z2'])
                row2 = row2.reset_index().sort_values(by="y").drop(columns=['z', 'z2'])
                if not row1.equals(row2):
                    duplicate_mismatches.append(((z, z2), (z2, z)))

        if duplicate_mismatches:
            msg = (
                "Error: The following lines in pLossFactorInternal have inconsistent values: "
                f"{duplicate_mismatches}"
            )
            gams.printLog(msg)
            raise ValueError(msg)
        gams.printLog("Success: No problem in duplicate values in pLossFactorInternal.")
    except ValueError:
        raise
    except Exception:
        gams.printLog('Unexpected error when checking interconnected mode')
        raise


def _check_planning_reserves(gams, db):
    """Verify that all countries have planning reserve margins defined."""
    try:
        records = db["pPlanningReserveMargin"].records
        if records is None or records.empty:
            return
        zcmap_records = db["zcmap"].records
        if zcmap_records is None or zcmap_records.empty:
            return
        countries_planning = set(records['c'].unique())
        countries_def = set(zcmap_records['c'].unique())
        missing_countries = countries_def - countries_planning
        if missing_countries:
            missing_countries_str = ", ".join(missing_countries)
            msg = (
                "Error: The following countries are missing from 'pPlanningReserveMargin': "
                f"{missing_countries_str}"
            )
            gams.printLog(msg)
            raise ValueError(msg)
        gams.printLog("Success: All countries c have planning reserve defined.")
    except ValueError:
        raise
    except Exception:
        gams.printLog('Unexpected error when checking PlanningReserves')
        raise


def _check_fuel_definitions(gams, db):
    """Ensure fuel definitions align between pTechFuel and generation data."""
    try:
        techfuel_records = db["pTechFuel"].records
        if techfuel_records is None or techfuel_records.empty:
            return
        gen_records = db["pGenDataInput"].records
        if gen_records is None or gen_records.empty:
            return

        fuels_defined = set(techfuel_records['f'].unique())
        fuels_in_gendata = set(gen_records['f'].unique())
        missing_fuels = fuels_in_gendata - fuels_defined
        additional_fuels = fuels_defined - fuels_in_gendata
        if missing_fuels:
            msg = (
                "Error: The following fuels are in gendata but not defined in pTechFuel: \n"
                f"{missing_fuels}"
            )
            gams.printLog(msg)
            raise ValueError(msg)
        if additional_fuels:
            gams.printLog(
                "Info: The following fuels are defined in pTechFuel but not in gendata.\n"
                f"{additional_fuels}\n This may be because of spelling issues, and may cause problems after."
            )
        gams.printLog('Success: Fuels are well-defined everywhere.')
        gams.printLog(f"Fuels used: {sorted(fuels_in_gendata)}")

    except ValueError:
        raise
    except Exception:
        gams.printLog('Unexpected error when checking pTechFuel fuels')
        raise


def _check_tech_definitions(gams, db):
    """Ensure technology definitions align between pTechFuel and generation data."""
    try:
        techfuel_records = db["pTechFuel"].records

        gen_records = db["pGenDataInput"].records

        tech_defined = (
            set(techfuel_records["tech"].unique())
            if "tech" in techfuel_records.columns
            else set()
        )
        tech_in_gendata = (
            set(gen_records["tech"].unique())
            if "tech" in gen_records.columns
            else set()
        )

        missing_techs = tech_in_gendata - tech_defined
        additional_techs = tech_defined - tech_in_gendata
        if missing_techs:
            msg = (
                "Error: The following technologies are in gendata but not defined in pTechFuel: \n"
                f"{missing_techs}"
            )
            gams.printLog(msg)
            raise ValueError(msg)
        if additional_techs:
            gams.printLog(
                "Info: The following technologies are defined in pTechFuel but not used in gendata.\n"
                f"{additional_techs}\n This may indicate spelling differences or unused technologies."
            )
        gams.printLog("Success: Technologies are well-defined everywhere.")
        gams.printLog(f"Technologies used: {sorted(tech_in_gendata)}")

    except ValueError:
        raise
    except Exception:
        gams.printLog("Unexpected error when checking pTechFuel technologies")
        raise


def _check_zone_consistency(gams, db):
    """Check for overlaps between internal and external zones."""
    try:
        if db["zext"].records is not None:
            zext_records = db["zext"].records
            z_records = db["zcmap"].records
            if zext_records is not None and not zext_records.empty:
                zext_column = zext_records.columns[0]
                zext = set(zext_records[zext_column].unique())
            else:
                zext = set()
            z = set(z_records['z'].unique()) if z_records is not None else set()
            common_elements = zext & z
            if common_elements:
                msg = (
                    "Error: The following zones are included both as external and internal zones:\n"
                    f"{common_elements}."
                )
                gams.printLog(msg)
                raise ValueError(msg)
            gams.printLog("Success: No conflict between internal and external zones")
        else:
            gams.printLog(
                "Success: No conflict between internal and external zones, as external zones are not included in the model."
            )
    except ValueError:
        raise
    except Exception:
        gams.printLog('Unexpected error when checking zext')
        raise


def _check_single_zone_internal_exchange(gams, db):
    """Ensure internal exchange is disabled when only one internal zone exists."""
    try:
        zcmap_records = db["zcmap"].records
        if zcmap_records is None or zcmap_records.empty:
            return
        unique_zones = set(zcmap_records["z"].dropna().unique())
        if len(unique_zones) != 1:
            return

        settings_records = db["pSettings"].records
        if settings_records is None or settings_records.empty:
            return
        settings_indexed = settings_records.set_index("pSettingsHeader")
        if "fEnableInternalExchange" not in settings_indexed.index:
            return

        value = settings_indexed.loc["fEnableInternalExchange", "value"]
        try:
            flag = float(value)
        except (TypeError, ValueError):
            flag = value
        if flag in (0.0, 0, "0"):
            gams.printLog(
                "Success: Single-zone configuration has internal exchange disabled."
            )
            return

        msg = (
            "Error: fEnableInternalExchange must be 0 when only one internal zone is defined."
        )
        gams.printLog(msg)
        raise ValueError(msg)
    except ValueError:
        raise
    except Exception:
        gams.printLog(
            "Unexpected error when checking single-zone internal exchange settings"
        )
        raise


def _check_external_transfer_limits(gams, db):
    """Warn when external zones lack transfer limits."""
    try:
        zext_records = db["zext"].records
        if zext_records is None or zext_records.empty:
            return
        if db["pExtTransferLimit"].records is None:
            gams.printLog(
                "Warning: External zones are specified, but imports and exports capacities are not specified. This may be caused by a problem in the spelling of external zones in pExtTransferLimit."
            )
    except Exception:
        gams.printLog('Unexpected error when checking pExtTransferLimit')
        raise


def _check_external_transfer_settings(gams, db):
    """Ensure settings controlling external transfers align with available data."""
    try:
        settings_records = db["pSettings"].records
        if settings_records is None or settings_records.empty:
            return
        flag_series = settings_records.loc[
            settings_records.pSettingsHeader == 'fAllowTransferExpansion', 'value'
        ]
        if not flag_series.empty:
            try:
                flag_value = float(flag_series.iloc[-1])
            except (TypeError, ValueError):
                flag_value = None
            if flag_value is not None and flag_value != 0:
                if db["pExtTransferLimit"].records is None:
                    gams.printLog(
                        "Warning: exchanges with external zones are allowed, but imports and exports capacities are not specified."
                    )
    except Exception:
        gams.printLog('Unexpected error when checking pSettings')
        raise


def _check_storage_data(gams, db):
    """Verify that storage generators have matching entries in pStorageDataInput."""
    try:
        stor_records = db["pStorageDataInput"].records
        if stor_records is None or stor_records.empty:
            return
        gen_records = db["pGenDataInput"].records
        if gen_records is None or gen_records.empty:
            return
        print(stor_records)
        gen_storage = set(stor_records['gen'].unique())
        gen_ref = set(gen_records.loc[gen_records.tech == 'Storage']['g'].unique())
        missing_storage_gen = gen_ref - gen_storage
        if missing_storage_gen:
            msg = (
                "Error: The following storage genartors are in pGenDataInput but not defined in pStorageDataInput: \n"
                f"{missing_storage_gen}"
            )
            gams.printLog(msg)
            raise ValueError(msg)
        gams.printLog('Success: All storage generators are are well-defined in pStorageDataInput.')
    except ValueError:
        raise
    except Exception:
        gams.printLog('Unexpected error when checking storage data')
        raise


def _check_generation_defaults(gams, db):
    """Warn when pGenDataInputDefault lacks zones present in pGenDataInput."""
    try:
        default_param = db["pGenDataInputDefault"]
        records = default_param.records
        if records is None or records.empty:
            return
        gen_records = db["pGenDataInput"].records
        if gen_records is None or gen_records.empty:
            return
        zones_to_include = set(gen_records['z'].unique())
        for fuel, group in records.groupby('f', observed=False):
            zones = set(group['z'].unique())
            missing = zones_to_include - zones
            if missing:
                gams.printLog(
                    f"Warning: The following zones are declared in pGenDataInput but are not specified in pGenDataInputDefault for fuel {fuel}: {missing}."
                )
    except ValueError:
        raise
    except Exception:
        gams.printLog('Unexpected error in pGenDataInputDefault coverage')
        raise


def _warn_missing_generation_default_combinations(gams, db):
    """Warn when a (zone, tech, fuel) tuple in pGenDataInput is absent in defaults."""
    try:
        default_records = db["pGenDataInputDefault"].records
        gen_records = db["pGenDataInput"].records
        if (
            default_records is None
            or default_records.empty
            or gen_records is None
            or gen_records.empty
        ):
            return
        required = (
            gen_records[['z', 'tech', 'f']]
            .dropna(subset=['z', 'tech', 'f'])
            .drop_duplicates()
        )
        available = (
            default_records[['z', 'tech', 'f']]
            .dropna(subset=['z', 'tech', 'f'])
            .drop_duplicates()
        )
        required_set = set(map(tuple, required.to_records(index=False)))
        available_set = set(map(tuple, available.to_records(index=False)))
        missing = required_set - available_set
        if missing:
            missing_list = sorted(missing)
            preview = missing_list[:10]
            more = ""
            if len(missing_list) > len(preview):
                more = f" (showing {len(preview)} of {len(missing_list)})"
            gams.printLog(
                "Warning: pGenDataInputDefault lacks entries for the following (zone, tech, fuel) combinations "
                f"present in pGenDataInput{more}: {preview}. No errors will be raised, but this may create unexpected results."
            )
    except Exception:
        gams.printLog('Unexpected error when checking pGenDataInputDefault combinations')
        raise


def _warn_missing_availability_default_combinations(gams, db):
    """Warn when a (zone, tech, fuel) tuple in pGenDataInput is absent in pAvailabilityDefault."""
    try:
        avail_defaults = db["pAvailabilityDefault"].records
        gen_records = db["pGenDataInput"].records
        if (
            avail_defaults is None
            or avail_defaults.empty
            or gen_records is None
            or gen_records.empty
        ):
            gams.printLog(
                "[input_verification][availability] Skipping pAvailabilityDefault coverage check: table missing or empty."
            )
            return

        zone_col_defaults = "z" if "z" in avail_defaults.columns else ("zone" if "zone" in avail_defaults.columns else None)
        fuel_col_defaults = "f" if "f" in avail_defaults.columns else ("fuel" if "fuel" in avail_defaults.columns else None)
        tech_col_defaults = "tech" if "tech" in avail_defaults.columns else None

        if zone_col_defaults is None or fuel_col_defaults is None or tech_col_defaults is None:
            gams.printLog(
                "[input_verification][availability] Skipping coverage check: pAvailabilityDefault lacks required columns "
                f"(have {list(avail_defaults.columns)}; need zone/z, tech, fuel/f)."
            )
            return

        required = (
            gen_records[["z", "tech", "f"]]
            .dropna(subset=["z", "tech", "f"])
            .drop_duplicates()
        )
        available = (
            avail_defaults[[zone_col_defaults, tech_col_defaults, fuel_col_defaults]]
            .dropna(subset=[zone_col_defaults, tech_col_defaults, fuel_col_defaults])
            .drop_duplicates()
            .rename(columns={zone_col_defaults: "z", tech_col_defaults: "tech", fuel_col_defaults: "f"})
        )

        required_set = set(map(tuple, required.to_records(index=False)))
        available_set = set(map(tuple, available.to_records(index=False)))
        missing = required_set - available_set
        if missing:
            missing_list = sorted(missing)
            preview = missing_list[:10]
            more = ""
            if len(missing_list) > len(preview):
                more = f" (showing {len(preview)} of {len(missing_list)})"
            gams.printLog(
                "Warning: pAvailabilityDefault lacks entries for the following (zone, tech, fuel) combinations "
                f"present in pGenDataInput{more}: {preview}. No errors will be raised, but this may create unexpected results."
            )
        else:
            gams.printLog("[input_verification][availability] pAvailabilityDefault covers all (zone, tech, fuel) combinations present in pGenDataInput.")
    except Exception:
        gams.printLog("Unexpected error when checking pAvailabilityDefault combinations")
        raise


def _warn_missing_build_limits(gams, db):
    """Warn when candidate/committed units lack BuildLimitperYear (treated as not buildable)."""
    try:
        if "pGenDataInput" not in db:
            return
        records = db["pGenDataInput"].records
        if records is None or records.empty:
            return

        if not {"Status", "BuildLimitperYear"}.issubset(records.columns):
            gen_col = "g" if "g" in records.columns else ("gen" if "gen" in records.columns else None)
            gen_list = (
                records[gen_col].astype(str).unique().tolist()
                if gen_col is not None
                else records.index.astype(str).tolist()
            )
            preview = gen_list[:10]
            more = ""
            if len(gen_list) > len(preview):
                more = f" (showing {len(preview)} of {len(gen_list)})"
            gams.printLog(
                "[input_verification][build_limit] Warning: pGenDataInput lacks Status/BuildLimitperYear columns; "
                f"candidate/committed units may not be buildable. Consider clearing Status to exclude plants if needed. "
                f"Generators present{more}: {preview}"
            )
            return

        df_wide = records
        if {"pGenDataInputHeader", "value"}.issubset(records.columns) and "BuildLimitperYear" not in records.columns:
            id_col = "g" if "g" in records.columns else ("gen" if "gen" in records.columns else None)
            df_wide = (
                records.pivot_table(
                    index=id_col or records.index,
                    columns="pGenDataInputHeader",
                    values="value",
                    aggfunc="first",
                    observed=False,
                )
                .reset_index()
            )
            if id_col:
                df_wide = df_wide.rename(columns={id_col: "g"})

        status_num = pd.to_numeric(df_wide["Status"], errors="coerce")
        build_limit = pd.to_numeric(df_wide["BuildLimitperYear"], errors="coerce")
        mask = status_num.isin([2, 3]) & (build_limit.isna() | (build_limit == 0))
        if not mask.any():
            return

        offenders = df_wide.loc[mask]
        gen_col = "g" if "g" in offenders.columns else ("gen" if "gen" in offenders.columns else None)
        names = (
            offenders[gen_col].astype(str).tolist()
            if gen_col is not None
            else offenders.index.astype(str).tolist()
        )
        preview = names[:10]
        more = ""
        if len(names) > len(preview):
            more = f" (showing {len(preview)} of {len(names)})"
        gams.printLog(
            "[input_verification][build_limit] Warning: BuildLimitperYear is missing or zero for "
            f"{len(names)} candidate/committed generator(s); they will not be eligible to build{more}: {preview}."
        )
    except Exception:
        gams.printLog("Unexpected error when warning about BuildLimitperYear in pGenDataInput")
        raise


def run_input_verification_from_gdx(gdx_path, *, verbose=True, log_func=None):
    """Run the input verification logic directly on a standalone GDX file."""
    container = gt.Container()
    container.read(gdx_path)
    apply_debug_column_renames(container)
    return _run_input_verification_on_container(
        container,
        verbose=verbose,
        log_func=log_func,
    )


if __name__ == "__main__":

    DEFAULT_GDX = os.path.join("test", "input.gdx")

    container = gt.Container()
    container.read(DEFAULT_GDX)
    apply_debug_column_renames(container)
    _run_input_verification_on_container(container)