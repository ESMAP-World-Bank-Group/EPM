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

def run_input_treatment(gams,
                        fill_missing_hydro_availability: bool = False,
                        fill_missing_hydro_capex: bool = False):

    def _write_back(db: gt.Container, param_name: str):
        """Copy updates back to whatever database the caller provided.

        A full model run hands in a real gams.GamsDatabase, while our debug path
        only supplies a gt.Container. We support both without needing the GAMS runtime.
        """
        target_db = gams.db
        if isinstance(target_db, gt.Container):
            target_db.data[param_name].setRecords(db[param_name].records)
        else:
            db.write(target_db, [param_name])


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
        if "pGenDataInput" not in db:
            return

        records = db["pGenDataInput"].records
        if records is None or records.empty or "g" not in records.columns:
            return

        records = records.copy()
        header_col, value_col = _detect_header_and_value_columns(records)
        if header_col is None or value_col is None:
            return

        zone_column = None
        for candidate in ("z", "zone"):
            if candidate in records.columns:
                zone_column = candidate
                break

        all_gens = set(records["g"].unique())
        status_rows = records.loc[records[header_col] == "Status"].copy()
        status_rows["numeric_status"] = pd.to_numeric(status_rows[value_col], errors="coerce")
        valid_mask = status_rows["numeric_status"].isin([1, 2, 3])
        valid_gens = set(status_rows.loc[valid_mask, "g"].unique())
        invalid_gens = all_gens - valid_gens
        if not invalid_gens:
            return

        zone_label = zone_column or "zone"
        zone_map = {}
        if zone_column and zone_column in records.columns:
            zone_frame = records.loc[records["g"].isin(invalid_gens), ["g", zone_column]].drop_duplicates(subset=["g"])
            zone_map = dict(zip(zone_frame["g"], zone_frame[zone_column]))

        gams.printLog(
            f"Removing {len(invalid_gens)} generator(s) due to invalid or missing Status (allowed values: 1, 2, 3)."
        )
        zone_listing = {}
        for gen in invalid_gens:
            zone_val = zone_map.get(gen, "unknown")
            zone_listing.setdefault(zone_val, []).append(gen)
        for zone_val, gens in zone_listing.items():
            gens_sorted = ", ".join(sorted(gens))
            gams.printLog(f"  - {zone_label}: {zone_val} -> {gens_sorted}")

        filtered_records = records.loc[~records["g"].isin(invalid_gens)]
        db.data["pGenDataInput"].setRecords(filtered_records)
        _write_back(db, "pGenDataInput")


    def monitor_hydro_availability(db: gt.Container, auto_fill: bool):
        """Log missing hydro availability rows and optionally back-fill them."""
        required_params = {"pGenDataInput", "pAvailability"}
        if any(name not in db for name in required_params):
            return

        gen_records = db["pGenDataInput"].records
        avail_records = db["pAvailability"].records
        if gen_records is None or gen_records.empty:
            return

        gen_records = gen_records.copy()
        avail_records = avail_records.copy() if avail_records is not None else pd.DataFrame(columns=["g", "q", "value"])

        if "g" not in gen_records.columns or "tech" not in gen_records.columns:
            return

        zone_column = None
        for candidate in ("z", "zone"):
            if candidate in gen_records.columns:
                zone_column = candidate
                break

        target_techs = {"ROR", "ReservoirHydro"}
        hydro_meta_cols = ["g", "tech"] + ([zone_column] if zone_column else [])
        hydro_meta = (
            gen_records.loc[gen_records["tech"].isin(target_techs), hydro_meta_cols]
            .drop_duplicates(subset=["g"])
        )
        if hydro_meta.empty:
            return

        if "g" not in avail_records.columns:
            return

        provided_gens = set(avail_records["g"].unique())
        missing_meta = hydro_meta.loc[~hydro_meta["g"].isin(provided_gens)]
        if missing_meta.empty:
            return

        gams.printLog(
            f"Hydropower factor warning: {len(missing_meta)} generator(s) with tech in {sorted(target_techs)} "
            "lack availability entries in pAvailability."
        )
        zone_label = zone_column or "zone"
        all_zone_values = None
        if zone_column:
            all_zone_values = (
                hydro_meta.loc[:, zone_column]
                .dropna()
                .unique()
                .tolist()
            )
        _log_zone_summary(
            gams,
            "  Missing hydropower factors by zone:",
            missing_meta.loc[:, ["g"] + ([zone_column] if zone_column else [])],
            zone_column,
            zone_label,
            all_zone_values,
        )

        if not auto_fill:
            return

        if avail_records.empty:
            gams.printLog("Auto-fill skipped: pAvailability has no existing data to copy from.")
            return

        if zone_column is None:
            gams.printLog("Auto-fill skipped: cannot identify zone column in pGenDataInput.")
            return

        gen_zone_meta = gen_records.loc[:, ["g", "tech", zone_column]].drop_duplicates(subset=["g"])
        donor_frame = avail_records.merge(gen_zone_meta, on="g", how="left")
        donor_frame = donor_frame.dropna(subset=[zone_column, "tech"])
        if donor_frame.empty:
            gams.printLog("Auto-fill skipped: no donor generators have both zone and tech information.")
            return

        donor_profiles = {}
        for (zone_val, tech_val), frame in donor_frame.groupby([zone_column, "tech"]):
            first_gen = frame["g"].iloc[0]
            profile = frame.loc[frame["g"] == first_gen, ["q", "value"]].copy()
            donor_profiles[(zone_val, tech_val)] = {"profile": profile, "source": first_gen}

        new_entries = []
        for row in missing_meta.itertuples():
            key = (getattr(row, zone_column), row.tech)
            donor_info = donor_profiles.get(key)
            if donor_info is None:
                gams.printLog(f"  -> No donor found to auto-fill {row.g} ({row.tech}, {zone_label}: {key[0]}).")
                continue
            addition = donor_info["profile"].copy()
            addition["g"] = row.g
            new_entries.append(addition.loc[:, ["g", "q", "value"]])
            gams.printLog(
                f"  -> Auto-filled availability for {row.g} ({row.tech}, {zone_label}: {key[0]}) "
                f"using {donor_info['source']}."
            )

        if not new_entries:
            gams.printLog("Auto-fill finished: no records were added.")
            return

        updated_availability = pd.concat([avail_records] + new_entries, ignore_index=True)
        db.data["pAvailability"].setRecords(updated_availability)
        _write_back(db, "pAvailability")


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

        target_headers = {"Status", "Capex"}
        subset = records.loc[records[header_col].isin(target_headers)]
        if subset.empty:
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

        target_status = pivot["Status"].isin([2, 3])
        target_techs = {"ROR", "ReservoirHydro"}
        tech_mask = pivot["tech"].isin(target_techs)
        missing_capex = pivot[target_status & tech_mask & pivot["Capex"].isna()]
        if missing_capex.empty:
            return

        gams.printLog(
            f"Hydro capex warning: {len(missing_capex)} generator(s) in {sorted(target_techs)} "
            "with status 2 or 3 have no Capex defined."
        )
        zone_label = zone_column or "zone"
        all_zone_values = None
        if zone_column:
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

        donor_lookup = {}
        for donor in donors.itertuples():
            key = (getattr(donor, zone_column), donor.tech)
            if key not in donor_lookup:
                donor_lookup[key] = (donor.g, donor.Capex)

        updated = False
        new_rows = []
        for row in missing_capex.itertuples():
            key = (getattr(row, zone_column), row.tech)
            donor_info = donor_lookup.get(key)
            if donor_info is None:
                gams.printLog(f"  -> No donor Capex found for {row.g} ({row.tech}, {zone_label}: {key[0]}).")
                continue
            donor_gen, donor_capex = donor_info
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
                f"  -> Auto-filled Capex for {row.g} ({row.tech}, {zone_label}: {key[0]}) using {donor_gen}."
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
            gams.printLog('{} empty so no effect'.format(default_param_name))
            db.data[param_name].setRecords(param_df)
            _write_back(db, param_name)

            return None
        
        gams.printLog("Modifying {} with {}".format(param_name, default_param_name))
        
        
        # Unstack data on 'header' for correct alignment
        param_df = param_df.set_index([i for i in param_df.columns if i not in ['value']]).squeeze().unstack(header)

        default_df = default_df.set_index([i for i in default_df.columns if i not in ['value']]).squeeze().unstack(header)
        
        # Add missing columns that have been dropped by CONNECT CSV WRITER
        missing_columns = [i for i in default_df.columns if i not in param_df.columns]
        gams.printLog(f'Missing {missing_columns}')
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
            gams.printLog('{} empty so no effect'.format(param_name))
            return None
            
        gams.printLog('Adding generator reference to {}'.format(param_name))
        
        # Identify common columns between param_df and ref_df, excluding "value"
        columns = [c for c in param_df.columns if c != "value" and c in ref_df.columns]
        
        # Keep only the generator column and common columns in ref_df
        ref_df = ref_df.loc[:, [column_generator] + columns]
            

        # Remove duplicate rows in the reference DataFrame
        ref_df = ref_df.drop_duplicates()

        # Merge the reference DataFrame with the parameter DataFrame on common columns
        param_df = pd.merge(ref_df, param_df, how='left', on=columns)
        
        # Select only the necessary columns for the final output
        param_df = param_df.loc[:, [column_generator] + cols_tokeep + ["value"]]
            
        if param_df['value'].isna().any():
            missing_rows = param_df[param_df['value'].isna()]  # Get rows with NaN values
            gams.printLog(f"Warning: missing values found in '{param_name}'. This indicates that some generator-year combinations expected by the model are not provided in the input data. Generators in {param_ref} without default values are:")
            gams.printLog(missing_rows.to_string())  # Print the rows where 'value' is NaN
            raise ValueError(f"Missing values in default is not permitted. To fix this bug ensure that all combination in {param_name} are included.")

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
        
        gams.printLog("Modifying {} with default values".format(param_name))
        
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

            data = records.copy()
            data[year_column] = pd.to_numeric(data[year_column], errors='coerce')
            data = data.dropna(subset=[year_column, "value"])
            if data.empty:
                continue

            group_cols = [c for c in data.columns if c not in (year_column, "value")]
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
                f"Linear interpolation performed on {param_name} to match model years {target_range}."
            )
            _write_back(db, param_name)

        
    # Create a GAMS workspace and database
    db = gt.Container(gams.db)

    env_flag = str(os.environ.get("EPM_FILL_HYDRO_AVAILABILITY", "")).strip().lower()
    env_auto_fill = env_flag in {"1", "true", "yes", "on"}
    auto_fill_missing_hydro = fill_missing_hydro_availability or env_auto_fill
    if env_auto_fill and not fill_missing_hydro_availability:
        gams.printLog("Hydro availability auto-fill enabled via EPM_FILL_HYDRO_AVAILABILITY environment variable.")

    capex_flag = str(os.environ.get("EPM_FILL_HYDRO_CAPEX", "")).strip().lower()
    env_capex_fill = capex_flag in {"1", "true", "yes", "on"}
    auto_fill_missing_capex = fill_missing_hydro_capex or env_capex_fill
    if env_capex_fill and not fill_missing_hydro_capex:
        gams.printLog("Hydro capex auto-fill enabled via EPM_FILL_HYDRO_CAPEX environment variable.")

    remove_generators_with_invalid_status(db)
    
    interpolate_time_series_parameters(db, YEARLY_OUTPUT)
    
    monitor_hydro_availability(db, auto_fill_missing_hydro)
    
    monitor_hydro_capex(db, auto_fill_missing_capex)
    
    # Complete Generator Data
    overwrite_nan_values(db, "pGenDataInput", "pGenDataInputDefault", "pGenDataInputHeader")

    # Prepare pAvailability by filling missing values with default values
    default_df = prepare_generatorbased_parameter(db, "pAvailabilityDefault",
                                                cols_tokeep=['q'],
                                                param_ref="pGenDataInput")
                                                
    fill_default_value(db, "pAvailability", default_df)

    # Prepare pCapexTrajectories by filling missing values with default values
    default_df = prepare_generatorbased_parameter(db, "pCapexTrajectoriesDefault",
                                                cols_tokeep=['y'],
                                                param_ref="pGenDataInput")
                                                                                                
    fill_default_value(db, "pCapexTrajectories", default_df)


    # LossFactor must be defined through a specific csv
    # prepare_lossfactor(db, "pNewTransmission", "pLossFactorInternal", "y", "value")



if __name__ == "__main__":
    
    import argparse
    import sys

    DEFAULT_GDX = os.path.join("test", "input.gdx")
    output_gdx = os.path.join("test", "input_treated.gdx")

    container = gt.Container()
    container.read(DEFAULT_GDX)
    
    # SimpleNamespace fakes the small slice of the GAMS API we need for debugging.
    # It is just enough for tests and does not behave like a full GAMS runtime.
    dummy_gams = SimpleNamespace(db=container, printLog=lambda msg: print(str(msg)))
    
    # Replace columns names to be consistent with gams code
    columns_replace = {
        'pGenDataInput': {'uni': 'pGenDataInputHeader', 'gen': 'g'},
        'pGenDataInputDefault': {'uni': 'pGenDataInputHeader', 'gen': 'g'},
        'pAvailability': {'uni': 'q', 'gen': 'g'},
        'pAvailabilityDefault': {'uni': 'q'},
        'pCapexTrajectoriesDefault': {'uni': 'y'},
        'pDemandForecast': {'uni': 'y'}
    }

    for param_name, rename_map in columns_replace.items():
        symbol = dummy_gams.db.data.get(param_name)
        if symbol is None:
            continue
        records = symbol.records
        if records is None:
            continue
        applicable = {old: new for old, new in rename_map.items() if old in records.columns and old != new}
        if not applicable:
            continue
        symbol.setRecords(records.rename(columns=applicable))
    
    
    run_input_treatment(dummy_gams)
    
    if True:         
        container.write(output_gdx)
