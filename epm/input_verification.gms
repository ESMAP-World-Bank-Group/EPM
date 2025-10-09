**********************************************************************
* ELECTRICITY PLANNING MODEL (EPM)
* Developed at the World Bank
**********************************************************************
* DepSettingsHeaderription:
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


# Settings specification - checking all required parameters are specified
try:
    if db["pSettings"].records is not None:
        settings_df = db["pSettings"].records
        list_required_values = ['VoLL', 'ReserveVoLL', 'SpinReserveVoLL', 'WACC', 'DR']
        list_warning_values = ['sIntercoReserveContributionPct', 'fCountIntercoForReserves']
        list_missing = []
        list_zero_values = []
        for e in list_required_values:
            if e not in list(settings_df['pSettingsHeader']):
                list_missing.append(e)
        for e in list_warning_values:
            if e not in list(settings_df['pSettingsHeader']):
                list_zero_values.append(e)
                
        if list_missing:
            msg = f"Error: The following entries are required in pSettings but currently missing: {list_missing}"
            gams.printLog(msg)
            raise ValueError(msg)
        if list_zero_values:
            msg = f"WARNING: The following entries are set to zero in pSettings: {list_zero_values}"
            gams.printLog(msg)
            
        
except ValueError:
    raise  # Let this one bubble up with your message
except Exception as e:
    gams.printLog('Unexpected error when checking interconnected mode')
    raise # Re-raise the exception for debuggings


# Check that all these parameters are not None
try:
    essential_param = ["y", "pHours", "zcmap", "pSettings", "pGenDataInput", "pFuelPrice",
        "pFuelCarbonContent", "pTechData"]
    for param in  essential_param:
        if param not in db:
            msg = f"Error {param} is missing"
            gams.printLog(msg)
            raise ValueError(msg)
        else:
            if db[param].records is None:
                msg = f"Error {param} is missing"
                gams.printLog(msg)
                raise ValueError(msg)
            else:
                if db[param].records.empty:
                    msg = f"Error {param} is empty"
                    gams.printLog(msg)
                    raise ValueError(msg)
                    
    
    optional_param = ["pDemandForecast", "pDemandForecast"]
    for param in optional_param:
        if param not in db:
            gams.printLog(f"Warning {param} is missing")
        else:
            if db[param].records is None:
                gams.printLog(f"Warning {param} is missing")
            else:
                if db[param].records.empty:
                    gams.printLog(f"Warning {param} is empty")
except ValueError:
    raise  # Let this one bubble up with your message
except Exception as e:
    gams.printLog('Unexpected error in initial')
    raise # Re-raise the exception for debuggings


# TODO: Check if pGenData has missing attributes

# Check if all pHours values are positive
try:
    # Retrieve parameter pHours from GAMS
    pHours = db["pHours"]
    
    verif = pHours.records['value'].all() > 0
    
    if verif:
        gams.printLog("Success: All values pHours positive.")
    else:
        msg = f"Error: Some block duration are negative."
        gams.printLog(msg)
        raise ValueError(msg)


    # Compute the sum of all records
    total_hours = pHours.records['value'].sum()
    
    # Check if the sum is 8760
    if total_hours == 8760:
        gams.printLog("Success: The sum of pHours is exactly 8760.")
    else:
        msg = f"Error: The sum of pHours is {total_hours}, which is not 8760."
        gams.printLog(msg)
        raise ValueError(msg)
except ValueError:
    raise  # Let this one bubble up with your message
except Exception as e:
    gams.printLog('Unexpected error in pHours')
    raise # Re-raise the exception for debuggings
    
# VREProfile
try:
    pVREProfile = db["pVREProfile"]
    # Check if any value in pVREProfile exceeds 1
    if (pVREProfile.records['value'] > 1).any():
        msg = "Error: Capacity factor cannot be greater than 1 in pVREProfile."
        gams.printLog(msg)
        raise ValueError(msg)
    else:
        gams.printLog("Success: All pVREProfile values are valid.")
except ValueError:
    raise  # Let this one bubble up with your message
except Exception as e:
    gams.printLog('Unexpected error in VREProfile')
    raise # Re-raise the exception for debuggings
    
# pAvailability
try:
    # Check if any value in pAvailability is 1 or greater
    pAvailability = db["pAvailability"]
    if pAvailability.records is not None:
        if (pAvailability.records['value'] > 1).any():
            msg = "Error: Availability factor cannot be 1 or greater in pAvailability."
            gams.printLog(msg)
            raise ValueError(msg)
        else:
            gams.printLog("Success: All pAvailability values are valid.")
    else:
        gams.printLog('pAvailabilityCustom is None. All values come from pAvailabilityDefault')
except ValueError:
    raise  # Let this one bubble up with your message
except Exception as e:
    gams.printLog('Unexpected error in pAvailability')
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
        gams.printLog("Success: All dataframes have the same (q, d, t) combinations.")
    else:
        gams.printLog("Mismatch detected! The following differences exist:")
    
        for var in unique_combinations.keys():
            diff = unique_combinations[first_var] ^ unique_combinations[var]  # Find differences
            if diff:
                gams.printLog(f"Differences in {var}: {diff}")
        msg = "All dataframes do not have the same (q, d, t) combinations."
        gams.printLog(msg)
        raise ValueError(msg)
except ValueError:
    raise  # Let this one bubble up with your message
except Exception as e:
    gams.printLog('Unexpected error when checking time consistency')
    raise # Re-raise the exception for debuggings


# pDemandForecast
try:
    if db["pDemandForecast"].records is None:
        gams.printLog("Warning: pDemandForecast is not defined.")
    else:
        df = db["pDemandForecast"].records
        
        # Pivot table to create separate columns for 'peak' and 'energy'
        df_pivot = df.pivot(index=["z", "y"], columns="pe", values="value").reset_index()
            
        # Rename columns for clarity
        df_pivot.columns.name = None  # Remove the column index name
        df_pivot.rename(columns={"energy": "energy_value", "peak": "peak_value"}, inplace=True)
        
        # Calculate the Energy/Peak Ratio
        df_pivot["energy_peak_ratio"] = df_pivot["energy_value"] / df_pivot["peak_value"]
        
        # Print summary of Energy/Peak Demand Ratio
        min_ratio = df_pivot['energy_peak_ratio'].min()
        max_ratio = df_pivot['energy_peak_ratio'].max()
        gams.printLog(f"Energy/Peak Demand Ratio - Min: {min_ratio:.2f} & Max: {max_ratio:.2f}")

        # Optional: raise an error if values are outside expected range
        if min_ratio < 3 or max_ratio > 10:
            gams.printLog(f"WARNING: Energy/Peak Demand Ratio out of expected range [3–10]. Min: {min_ratio:.2f}, Max: {max_ratio:.2f}")
            # Identify the zones and years with extreme ratios
            extreme_rows = df_pivot[(df_pivot["energy_peak_ratio"] < 3) | (df_pivot["energy_peak_ratio"] > 10)]
            for _, row in extreme_rows.iterrows():
                gams.printLog(f"Extreme Energy/Peak Ratio at zone {row['z']}, year {row['y']}: {row['energy_peak_ratio']:.2f}")
        

            


except Exception as e:
    gams.printLog('Unexpected error when checking pDemandForecast')
    raise # Re-raise the exception for debuggings


# Check that pFuelPrice are included
try:
    df = db["pFuelPrice"].records
except ValueError:
    raise  # Let this one bubble up with your message
except Exception as e:
    gams.printLog('Unexpected error when checking pFuelPrice')
    raise # Re-raise the exception for debuggings
    

# TransferLimit
try:
    if db["pTransferLimit"].records is not None:
        transfer_df = db["pTransferLimit"].records
        topology = transfer_df.set_index(['z', 'z2']).index.unique()
        missing_pairs = [(z, z2) for z, z2 in topology if (z2, z) not in topology]
        if len(missing_pairs) > 0:
            missing_pairs_str = "\n".join([f"({z}, {z2})" for z, z2 in missing_pairs])
            msg = f"Error: The following (z, z2) pairs are missing their symmetric counterparts (z2, z) in 'pTransferLimit':\n{missing_pairs_str}"
            gams.printLog(msg)
            raise ValueError(msg)
        else:
            gams.printLog("Success: All (z, z2) pairs in 'pTransferLimit' have their corresponding (z2, z) pairs.")
            
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
            msg = f"Error: The following (z, z2) pairs do not have all required seasons in 'pTransferLimit':\n{season_issues_str}"
            gams.printLog(msg)
            raise ValueError(msg)
        else:
            gams.printLog("Success: All (z,z2) pairs contain all required seasons.")
except ValueError:
    raise  # Let this one bubble up with your message
except Exception as e:
    gams.printLog('Unexpected error when checking pTransferLimit')
    raise # Re-raise the exception for debuggings
    

# NewTransmission
try:
    if db["pNewTransmission"].records is not None:
        newtransmission_df = db["pNewTransmission"].records
        topology_newlines = newtransmission_df.set_index(['z', 'z2']).index.unique()
        duplicate_transmission = [(z, z2) for z, z2 in topology_newlines if (z2, z) in topology_newlines]
        if duplicate_transmission:
            msg = f"Error: The following (z, z2) pairs are specified twice in 'pNewTransmission':\n{duplicate_transmission} \n This may cause some problems when defining twice the characteristics of additional line."
            gams.printLog(msg)
            raise ValueError(msg)
        else:
            gams.printLog("Success: Each candidate transmission line is only specified once in pNewTransmission.")
except ValueError:
    raise  # Let this one bubble up with your message
except Exception as e:
    gams.printLog('Unexpected error when checking NewTransmission')
    raise # Re-raise the exception for debuggings
    

# Zones in pNewTransmission
try:
    if db["pNewTransmission"].records is not None:
        newtransmission_df = db["pNewTransmission"].records
        zones1 = newtransmission_df['z'].unique()
        zones2 = newtransmission_df['z2'].unique()
        zones_newtransmission = set(zones1).union(set(zones2))
        zcmap_df = db["zcmap"].records
        zones_defined = set(zcmap_df['z'].unique())
        new_zones = [z for z in zones_newtransmission if z not in zones_defined]
        if new_zones:
            msg = f"Warning: The following zones are used to defined new transmission lines in 'pNewTransmission':\n{new_zones} \n Tranmission lines involved will not be considered by the model. You should modify zone names to match the names defined in zcmap if you want."
            gams.printLog(msg)
            raise ValueError(msg)
        else:
            gams.printLog("Success: Zones in pNewTransmission match the zones defined in zcmap.")
except ValueError:
    raise  # Let this one bubble up with your message
except Exception as e:
    gams.printLog('Unexpected error when checking zones in NewTransmission')
    raise # Re-raise the exception for debuggings
        
        

# Interconnected mode
try:
    if db["pSettings"].records is not None:
        settings_df = db["pSettings"].records
        if "fEnableInternalExchange" in settings_df.set_index('pSettingsHeader').index:
            if settings_df.set_index('pSettingsHeader').loc["fEnableInternalExchange"].values[0]:  # running in interconnected mode
                if db["pLossFactorInternal"].records is not None:
                    loss_factor_df = db["pLossFactorInternal"].records
                    topology_lossfactor = loss_factor_df.set_index(['z', 'z2']).index.unique()
                    if (db["pNewTransmission"].records is not None) and  (db["pTransferLimit"].records is not None):
                        new_transmission_df = db["pNewTransmission"].records
                        transferlimit_df = db["pTransferLimit"].records
                        topology_new = set(new_transmission_df.set_index(['z', 'z2']).index.unique())
                        topology_transfer = set(transferlimit_df.set_index(['z', 'z2']).index.unique())
    
                        # Merge both sets
                        topology = topology_new.union(topology_transfer)
                    elif (db["pNewTransmission"].records is not None):
                        new_transmission_df = db["pNewTransmission"].records
                        topology = set(new_transmission_df.set_index(['z', 'z2']).index.unique())
                    elif (db["pTransferLimit"].records is not None):
                        transferlimit_df = db["pTransferLimit"].records
                        topology = set(transferlimit_df.set_index(['z', 'z2']).index.unique())
                    else:
                        msg = f"Error: Interconnected mode is activated, but both TransferLimit and NewTransmission are empty."
                        gams.printLog(msg)
                        raise ValueError(msg)
                    
                    # Ensure that for any (z, z2) present in the topology, we keep only one tuple (z, z2) if both (z, z2) and (z2, z) exist.
                    final_topology = set()
                    for (z, z2) in topology:
                        if (z2, z) not in final_topology:
                            final_topology.add((z, z2))
                    
                    # Check that all lines in topology exist in pLossFactorInternal
                    missing_lines = [line for line in final_topology if line not in topology_lossfactor and (line[1], line[0]) not in topology_lossfactor]
                    if missing_lines:
                        msg = f"Error: The following lines in topology are missing from pLossFactorInternal: {missing_lines}"
                        gams.printLog(msg)
                        raise ValueError(msg)
                    else:
                        gams.printLog("Success: All transmission lines have a lossfactor specified.")
                        
                    # Check that if a line appears twice in pLossFactorInternal (as both (z, z2) and (z2, z)), its values are the same
                    loss_factor_df.set_index(['z', 'z2'], inplace=True)
                    duplicate_mismatches = []
                    for (z, z2) in topology_lossfactor:
                        if (z2, z) in topology_lossfactor:  # Check reverse entry exists
                            row1 = loss_factor_df.loc[[(z, z2)]].sort_index()
                            row2 = loss_factor_df.loc[[(z2, z)]].sort_index()
                            row1 = row1.reset_index().sort_values(by="y").drop(columns=['z', 'z2'])
                            row2 = row2.reset_index().sort_values(by="y").drop(columns=['z', 'z2'])
                            if not row1.equals(row2):  # Compare rows
                                duplicate_mismatches.append(((z, z2), (z2, z)))
    
                    if duplicate_mismatches:
                        msg = f"Error: The following lines in pLossFactorInternal have inconsistent values: {duplicate_mismatches}"
                        gams.printLog(msg)
                        raise ValueError(msg)
                    else:
                        gams.printLog("Success: No problem in duplicate values in pLossFactorInternal.")
                    
                else:
                    msg = f"Error: Interconnected mode is activated, but LossFactor is empty"
                    gams.printLog(msg)
                    raise ValueError(msg)
except ValueError:
    raise  # Let this one bubble up with your message
except Exception as e:
    gams.printLog('Unexpected error when checking interconnected mode')
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
            msg = f"Error: The following countries are missing from 'pPlanningReserveMargin': {missing_countries_str}"
            gams.printLog(msg)
            raise ValueError(msg)
        else:
            gams.printLog("Success: All countries c have planning reserve defined.")
except ValueError:
    raise  # Let this one bubble up with your message
except Exception as e:
    gams.printLog('Unexpected error when checking PlanningReserves')
    raise # Re-raise the exception for debuggings
    

# Definition of fuels
try:
    if db["ftfindex"].records is not None:
        ftfindex = db["ftfindex"].records
        pGenDataInput = db["pGenDataInput"].records
        fuels = set(ftfindex['f'].unique())
        fuels_in_gendata = set(pGenDataInput['f'].unique())
        missing_fuels = fuels_in_gendata - fuels
        additional_fuels = fuels - fuels_in_gendata
        if missing_fuels:
            msg = f"Error: The following fuels are in gendata but not defined in ftfindex: \n{missing_fuels}"
            gams.printLog(msg)
            raise ValueError(msg)
        elif additional_fuels:
            msg = f"Error: The following fuels are defined in ftfindex but not in gendata.:\n{season_issues_str}\n This may be because of spelling issues, and may cause problems after."
            gams.printLog(msg)
            raise ValueError(msg)
        else:
            gams.printLog('Success: Fuels are well-defined everywhere.')
except ValueError:
    raise  # Let this one bubble up with your message
except Exception as e:
    gams.printLog('Unexpected error when checking ftfindex')
    raise # Re-raise the exception for debuggings
    

# Zones
try:
    if db["zext"].records is not None:
        zext = db["zext"].records
        z = db["z"].records
        zext = set(zext['zext'].unique())
        z = set(z['z'].unique())
        common_elements = zext & z
        if common_elements:
            msg = f"Error: The following zones are included both as external and internal zones:\n{common_elements}."
            gams.printLog(msg)
            raise ValueError(msg)
        else:
            gams.printLog("Success: no conflict between internal and external zones")
        
    else:
        gams.printLog("Success: no conflict between internal and external zones, as external zones are not included in the model.")
except ValueError:
    raise  # Let this one bubble up with your message
except Exception as e:
    gams.printLog('Unexpected error when checking zext')
    raise # Re-raise the exception for debuggings
    

try:
    if db["zext"].records is not None:
        zext = db["zext"].records
        if db["pExtTransferLimit"].records is not None:
            pExtTransferLimit = db["pExtTransferLimit"].records
        else:
            gams.printLog("Warning: External zones are specified, but imports and exports capacities are not specified. This may be caused by a problem in the spelling of external zones in pExtTransferLimit.")
except Exception as e:
    gams.printLog('Unexpected error when checking pExtTransferLimit')
    raise # Re-raise the exception for debuggings
        

# Check transmission data makes sense with settings
try:
    if db["pSettings"].records is not None:
        pSettings = db["pSettings"].records
        fAllowTransferExpansion = pSettings.loc[pSettings.pSettingsHeader == 'fAllowTransferExpansion']
        if not fAllowTransferExpansion.empty:  # we authorize exchanges with external zones
            if db["pExtTransferLimit"].records is None:
                gams.printLog("Warning: exchanges with external zones are allowed, but imports and exports capacities are not specified.")
except Exception as e:
    gams.printLog('Unexpected error when checking pSettings')
    raise # Re-raise the exception for debuggings
    

# Storage data
try:
    if db["pStorDataExcel"].records is not None:
        pStorDataExcel = db["pStorDataExcel"].records
        pGenDataInput = db["pGenDataInput"].records
        gen_storage = set(pStorDataExcel['g'].unique())
        gen_ref = set(pGenDataInput.loc[pGenDataInput.tech == 'Storage']['g'].unique())
        missing_storage_gen = gen_ref - gen_storage
        if missing_storage_gen:
            msg = f"Error: The following fuels are in gendata but not defined in pStorData: \n{missing_storage_gen}"
            gams.printLog(msg)
            raise ValueError(msg)
        else:
            gams.printLog('Success: All storage generators are are well-defined in pStorDataExcel.')
except ValueError:
    raise  # Let this one bubble up with your message
except Exception as e:
    gams.printLog('Unexpected error when checking storage data')
    raise # Re-raise the exception for debuggings
    

# pGenDataInputDefault
try:
    # Check if pGenDataInputDefault contains all zones
    pGenDataInputDefault = db["pGenDataInputDefault"]
    if pGenDataInputDefault.records is not None:
        pGenDataExcelDefault_df = db["pGenDataInputDefault"].records
        zones_to_include = db["pGenDataInput"].records
        zones_to_include = set(zones_to_include['z'].unique())
        for fuel, group in pGenDataExcelDefault_df.groupby('f', observed=False):
            zones = set(group['z'].unique())
            missing = zones_to_include - zones
            if missing:
                gams.printLog(f"Warning: The following zones are declared in pGenDataInput but are not specified in pGenDataInputDefault for fuel {fuel}: {missing}.")
except ValueError:
    raise  # Let this one bubble up with your message
except Exception as e:
    gams.printLog('Unexpected error in pAvailability')
    raise # Re-raise the exception for debuggings




$offEmbeddedCode
