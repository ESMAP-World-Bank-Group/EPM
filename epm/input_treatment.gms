**********************************************************************
* ELECTRICITY PLANNING MODEL (EPM)
* Developed at the World Bank
**********************************************************************
* Description:
* This GAMS-based model is designed for electricity system planning, 
* incorporating capacity expansion, generation dispatch, and policy 
* constraints such as renewable energy targets, emissions reductions, 
* and market mechanisms.
*
* Author(s): ESMAP Modelling Team
* Organization: World Bank
* Version: 
* License: Creative Commons Zero v1.0 Universal
*
* Key Features:
* - Optimization of electricity generation and capacity planning
* - Inclusion of renewable energy integration and storage technologies
* - Multi-period, multi-region modeling framework
* - CO2 emissions constraints and policy instruments
*
* Notes:
* - Ensure GAMS is installed before running this model.
* - The model requires input data in .GDX or Excel format.
*
* Contact:
* Claire Nicolas, c.nicolas@worldbank.org
**********************************************************************

$onEmbeddedCode Python:

import gams.transfer as gt
import numpy as np
import pandas as pd


def overwrite_nan_values(db: gt.Container, param_name: str, default_param_name: str):
    """
    Overwrites NaN values in a GAMS parameter with values from a default parameter.

    Args:
        db (gt.Container): GAMS database container.
        param_name (str): Name of the parameter to modify.
        default_param_name (str): Name of the parameter providing default values.
    """
    print("Modifying {} with {}".format(param_name, default_param_name))


    # Retrieve parameter data as pandas DataFrame
    param_df = db[param_name].records
    default_df = db[default_param_name].records
    
    param_df.to_csv('test.csv')

    # Identify key columns (all except "value")
    key_columns = [col for col in param_df.columns if col != "value" and col in default_df.columns]
    
    # Unstack data on 'uni' for correct alignment
    param_df = param_df.set_index([i for i in param_df.columns if i not in ['value']]).squeeze().unstack('uni')
    default_df = default_df.set_index([i for i in default_df.columns if i not in ['value']]).squeeze().unstack('uni')
    

    # Fill NaN values in param_df using corresponding values in default_df
    param_df = param_df.fillna(default_df)

    # Reset index to restore structure
    param_df = param_df.stack().reset_index()

    # Ensure column names are correct
    param_df.columns = key_columns + ["uni", "value"]
    
    param_df.to_csv('test_post.csv')
    
    db.data[param_name].setRecords(param_df)
    


# Create a GAMS workspace and database
db = gt.Container(gams.db)

# Call the function
overwrite_nan_values(db, "pGenDataExcel", "pGenDataExcelDefault")

$offEmbeddedCode