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

# Create a GAMS workspace and database
db = gt.Container(gams.db)


# Check that all these parameters are not None
try:
    essential_param = ["y", "pHours", "zcmap", "pSettings", "pGenDataExcel", "pFuelPrice",
        "pFuelCarbonContent", "pTechData"]
    for param in  essential_param:
        if param not in db:
            raise ValueError(f"Error {param} is missing")
        else:
            if db[param].records is None:
                raise ValueError(f"Error {param} is missing")
            else:
                if db[param].records.empty:
                    raise ValueError(f"Error {param} is empty")
                    
    
    optional_param = ["pDemandForecast", "pDemandForecast"]
    for param in optional_param:
        if param not in db:
            print(f"Warning {param} is missing")
        else:
            if db[param].records is None:
                print(f"Warning {param} is missing")
            else:
                if db[param].records.empty:
                    print(f"Warning {param} is empty")

except Exception as e:
    print('Unexpected error in initial')
    raise # Re-raise the exception for debuggings


# TODO: Check if pGenData has missing attributes



# Check if all pHours values are positive
try:
    # Retrieve parameter pHours from GAMS
    pHours = db["pHours"]
    
    verif = pHours.records['value'].all() > 0
    
    if verif:
        print("Success: All values pHours positive.")
    else:
        raise ValueError(f"Error: Some block duration are negative.")


    # Compute the sum of all records
    total_hours = pHours.records['value'].sum()
    
    # Check if the sum is 8760
    if total_hours == 8760:
        print("Success: The sum of pHours is exactly 8760.")
    else:
        raise ValueError(f"Error: The sum of pHours is {total_hours}, which is not 8760.")
        
except Exception as e:
    print('Unexpected error in pHours')
    raise # Re-raise the exception for debuggings
    
# VREProfile
try:
    pVREProfile = db["pVREProfile"]
    # Check if any value in pVREProfile exceeds 1
    if (pVREProfile.records['value'] > 1).any():
        raise ValueError("Error: Capacity factor cannot be greater than 1 in pVREProfile.")
    else:
        print("Success: All pVREProfile values are valid.")
except Exception as e:
    print('Unexpected error in VREProfile')
    raise # Re-raise the exception for debuggings
    
# pAvailability
try:
    # Check if any value in pAvailability is 1 or greater
    pAvailability = db["pAvailability"]
    if pAvailability.records is not None:
        if (pAvailability.records['value'] > 1).any():
            raise ValueError("Error: Availability factor cannot be 1 or greater in pAvailability.")
        else:
            print("Success: All pAvailability values are valid.")
    else:
        print('pAvailabilityCustom is None. All values come from pAvailabilityDefault')
except Exception as e:
    print('Unexpected error in pAvailability')
    raise # Re-raise the exception for debuggings


# Check time resolution consistency
try:
    # Extract and store unique (q, d, t) combinations for each dataframe
    unique_combinations = {}
    # Define the variable names to check
    vars_time = ["pVREProfile", "pVREgenProfile", "pDemandProfile", "pHours"]
    
    for var in vars_time:
        if db[var].records is not None:
            df = db[var].records  # Extract the records from the database
            unique_combinations[var] = set(df[['q', 'd', 't']].apply(tuple, axis=1))  # Convert to unique sets
    
    # Check if all sets are equal
    first_var = vars_time[0]
    is_consistent = all(unique_combinations[first_var] == unique_combinations[var] for var in unique_combinations.keys())
    
    # Print result
    if is_consistent:
        print("All dataframes have the same (q, d, t) combinations.")
    else:
        print("Mismatch detected! The following differences exist:")
    
        for var in unique_combinations.keys():
            diff = unique_combinations[first_var] ^ unique_combinations[var]  # Find differences
            if diff:
                print(f"Differences in {var}: {diff}")
        raise ValueError("All dataframes do not have the same (q, d, t) combinations.")
except Exception as e:
    print('Unexpected error when checking time consistency')
    raise # Re-raise the exception for debuggings


# pDemandForecast
try:
    df = db["pDemandForecast"].records
    
    # Pivot table to create separate columns for 'peak' and 'energy'
    df_pivot = df.pivot(index=["z", "y"], columns="pe", values="value").reset_index()
        
    # Rename columns for clarity
    df_pivot.columns.name = None  # Remove the column index name
    df_pivot.rename(columns={"energy": "energy_value", "peak": "peak_value"}, inplace=True)
    
    # Calculate the Energy/Peak Ratio
    df_pivot["energy_peak_ratio"] = df_pivot["energy_value"] / df_pivot["peak_value"]
    
    print('Energy/Peak Demand Ratio')
    print(df_pivot['energy_peak_ratio'], "TODO: Define value that are consistent and raise error otherwise.")

except Exception as e:
    print('Unexpected error when checking pDemandForecast')
    raise # Re-raise the exception for debuggings


# Check that pFuelPrice are included
try:
    print('TODO')
    df = db["pFuelPrice"].records
    
except Exception as e:
    print('Unexpected error when checking pFuelPrice')
    raise # Re-raise the exception for debuggings
    


# TransferLimit
try:
    if db["pTransferLimit"].records is not None:
        transfer_df = db["pTransferLimit"].records
        topology = transfer_df.set_index(['z', 'z2']).index.unique()
        missing_pairs = [(z, z2) for z, z2 in topology if (z2, z) not in topology]
        if len(missing_pairs) > 0:
            missing_pairs_str = "\n".join([f"({z}, {z2})" for z, z2 in missing_pairs])
            raise ValueError(f"Error: The following (z, z2) pairs are missing their symmetric counterparts (z2, z) in 'pTransferLimit':\n{missing_pairs_str}")
        else:
            print("Success: All (z, z2) pairs in 'pTransferLimit' have their corresponding (z2, z) pairs.")
            
        pHours_df = pHours.records
        seasons = set(pHours_df["q"].unique())  # Get unique seasons from pHours
        season_issues = []
        
        for (z, z2), group in transfer_df.groupby(['z', 'z2'], observed=False):
            unique_seasons = set(group['q'].unique())
            missing_seasons = seasons - unique_seasons
            if missing_seasons:
                season_issues.append((z, z2, missing_seasons))
        
        if season_issues:
            season_issues_str = "\n".join([f"({z}, {z2}): missing seasons {missing}" for z, z2, missing in season_issues])
            raise ValueError(f"Error: The following (z, z2) pairs do not have all required seasons in 'pTransferLimit':\n{season_issues_str}")
        else:
            print("Success: All (z,z2) pairs contain all required seasons.")
except Exception as e:
    print('Unexpected error when checking pTransferLimit')
    raise # Re-raise the exception for debuggings
    
# PlanningReserves
try:
    if db["pPlanningReserveMargin"].records is not None:
        planning_df = db["pPlanningReserveMargin"].records
        zcmap_df = db["zcmap"].records
        countries_planning = set(planning_df['c'].unique())
        countries_def = set(zcmap_df['c'].unique())
        missing_countries = countries_def - countries_planning
        if missing_countries:
            missing_countries_str = ", ".join(missing_countries)
            raise ValueError(f"Error: The following countries are missing from 'pPlanningReserveMargin': {missing_countries_str}")
        else:
            print("Success: All countries c have planning reserve defined.")
        
except Exception as e:
    print('Unexpected error when checking PlanningReserves')
    raise # Re-raise the exception for debuggings


$offEmbeddedCode