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

# Retrieve parameter pHours from GAMS
pHours = db["pHours"]


# Check if all values are positive
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

# VREProfile
pVREProfile = db["pVREProfile"]

# Check if any value in pVREProfile exceeds 1
if (pVREProfile.records['value'] > 1).any():
    raise ValueError("Error: Capacity factor cannot be greater than 1 in pVREProfile.")
else:
    print("Success: All pVREProfile values are valid.")
    
# Check if any value in pAvailability is 1 or greater
pAvailability = db["pAvailability"]
if (pAvailability.records['value'] > 1).any():
    raise ValueError("Error: Availability factor cannot be 1 or greater in pAvailability.")
else:
    print("Success: All pAvailability values are valid.")
    
# # Check if any value in pDemandData is non-positive
# pDemandData = db["pDemandData"]
# if (pDemandData.records['value'] <= 0).any():
#     raise ValueError("Error: All pDemandData values must be positive.")
# else:
#     print("Success: All pDemandData values are valid.")
    
"""pFuelPrice = db["pFuelPrice"]
pMaxFuellimit = db["pMaxFuellimit"]

# Validate fuel mappings for pFuelPrice
print(pFuelPrice.records)
if pFuelPrice.records.groupby('c').size().max() > 1:
    raise ValueError("Error: More than one fuel map per country in pFuelPrice.")

# Validate fuel mappings for pMaxFuellimit
if pMaxFuellimit.records.groupby('c').size().max() > 1:
    raise ValueError("Error: More than one fuel map per country in pMaxFuellimit.")

print("Success: Fuel mappings are valid.")"""

# TransferLimit
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


$offEmbeddedCode