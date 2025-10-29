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

def run_input_treatment(gams):

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

    interpolate_time_series_parameters(db, YEARLY_OUTPUT)
    
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
