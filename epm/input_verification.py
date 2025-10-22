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
    Claire Nicolas — c.nicolas@worldbank.org
**********************************************************************
"""

import gams.transfer as gt
from types import SimpleNamespace
import pandas as pd

ESSENTIAL_INPUT = [
    "y",
    "pHours",
    "zcmap",
    "pSettings",
    "pGenDataInput",
    "pFuelPrice",
    "pFuelCarbonContent",
    "pTechData",
]

OPTIONAL_INPUT = ["pDemandForecast"]

def run_input_verification(gams):
    """Run the full suite of input validation checks on the current GAMS DB."""
    db = gt.Container(gams.db)

    _check_required_inputs(gams, db)
    _check_settings_flags(gams, db)
    _check_settings_required_entries(gams, db)
    # _check_candidate_build_limits(gams, db)
    _check_hours(gams, db)
    _check_vre_profile(gams, db)
    _check_availability(gams, db)
    _check_time_resolution(gams, db)
    _check_demand_forecast(gams, db)
    _check_fuel_price_presence(gams, db)
    _check_transfer_limits(gams, db)
    _check_new_transmission(gams, db)
    _check_new_transmission_zones(gams, db)
    _check_interconnected_mode(gams, db)
    _check_planning_reserves(gams, db)
    _check_fuel_definitions(gams, db)
    _check_zone_consistency(gams, db)
    _check_external_transfer_limits(gams, db)
    _check_external_transfer_settings(gams, db)
    _check_storage_data(gams, db)
    _check_generation_defaults(gams, db)


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

        required_columns = {"Status", "BuildLimitperYear"}
        if not required_columns.issubset(records.columns):
            missing = required_columns - set(records.columns)
            msg = (
                "Error: pGenDataInput is missing columns required for BuildLimitperYear validation: "
                f"{missing}"
            )
            gams.printLog(msg)
            raise ValueError(msg)

        status_numeric = pd.to_numeric(records["Status"], errors="coerce")
        build_limit_numeric = pd.to_numeric(records["BuildLimitperYear"], errors="coerce")
        candidate_mask = status_numeric.isin([2, 3])
        violation_mask = candidate_mask & (build_limit_numeric.isna() | (build_limit_numeric == 0))

        if violation_mask.any():
            violations = records.loc[violation_mask]
            identifier_column = "gen" if "gen" in violations.columns else None
            offending_entries = (
                violations[identifier_column].astype(str).tolist()
                if identifier_column
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
        records = db["pAvailability"].records
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
        df_pivot.rename(columns={"energy": "energy_value", "peak": "peak_value"}, inplace=True)

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
    """Ensure fuel definitions align between ftfindex and generation data."""
    try:
        ftf_records = db["ftfindex"].records
        if ftf_records is None or ftf_records.empty:
            return
        gen_records = db["pGenDataInput"].records
        if gen_records is None or gen_records.empty:
            return

        fuels_defined = set(ftf_records['f'].unique())
        fuels_in_gendata = set(gen_records['f'].unique())
        missing_fuels = fuels_in_gendata - fuels_defined
        additional_fuels = fuels_defined - fuels_in_gendata
        if missing_fuels:
            msg = (
                "Error: The following fuels are in gendata but not defined in ftfindex: \n"
                f"{missing_fuels}"
            )
            gams.printLog(msg)
            raise ValueError(msg)
        if additional_fuels:
            msg = (
                "Error: The following fuels are defined in ftfindex but not in gendata.\n"
                f"{additional_fuels}\n This may be because of spelling issues, and may cause problems after."
            )
            gams.printLog(msg)
            raise ValueError(msg)
        gams.printLog('Success: Fuels are well-defined everywhere.')
    except ValueError:
        raise
    except Exception:
        gams.printLog('Unexpected error when checking ftfindex')
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
            gams.printLog("Success: no conflict between internal and external zones")
        else:
            gams.printLog(
                "Success: no conflict between internal and external zones, as external zones are not included in the model."
            )
    except ValueError:
        raise
    except Exception:
        gams.printLog('Unexpected error when checking zext')
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
    """Verify that storage generators have matching entries in pStorDataExcel."""
    try:
        stor_records = db["pStorDataExcel"].records
        if stor_records is None or stor_records.empty:
            return
        gen_records = db["pGenDataInput"].records
        if gen_records is None or gen_records.empty:
            return
        gen_storage = set(stor_records['g'].unique())
        gen_ref = set(gen_records.loc[gen_records.tech == 'Storage']['g'].unique())
        missing_storage_gen = gen_ref - gen_storage
        if missing_storage_gen:
            msg = (
                "Error: The following fuels are in gendata but not defined in pStorData: \n"
                f"{missing_storage_gen}"
            )
            gams.printLog(msg)
            raise ValueError(msg)
        gams.printLog('Success: All storage generators are are well-defined in pStorDataExcel.')
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


def run_input_verification_from_gdx(gdx_path, *, verbose=True, log_func=None):
    """Run the input verification logic directly on a standalone GDX file."""
    container = gt.Container()
    container.read(gdx_path)

    collected_logs = []

    def _log(message):
        collected_logs.append(message)
        if log_func is not None:
            log_func(message)
        elif verbose:
            print(message)

    dummy_gams = SimpleNamespace(db=container, printLog=_log)
    run_input_verification(dummy_gams)
    return collected_logs


if __name__ == "__main__":
    import argparse
    import sys

    DEFAULT_GDX = "test/input.gdx"

    usage_note = (
        "Run outside the GAMS workflow with:\n"
        "  python -m epm.input_verification [GDX_PATH]\n\n"
        "If no path is provided the default is 'ep/test/input.gdx'. "
        "Use --quiet to suppress live logging."
    )

    parser = argparse.ArgumentParser(
        description="Run EPM input verification checks on a standalone GDX file.",
        epilog=usage_note,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "gdx_path",
        nargs="?",
        default=DEFAULT_GDX,
        help="Path to the GDX file to verify (default: %(default)s)."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress log echoing; messages are still collected internally."
    )
    args = parser.parse_args()

    try:
        logs = run_input_verification_from_gdx(args.gdx_path, verbose=not args.quiet)
    except Exception as exc:  # noqa: BLE001 - surface original exception
        print(f"Verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
    else:
        if args.quiet:
            print("\n".join(logs))
