

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

$offEmbeddedCode