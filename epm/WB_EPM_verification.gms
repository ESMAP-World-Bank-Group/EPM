

$onEmbeddedCode Python:
import gams.transfer as gt

# Create a GAMS workspace and database
db = gt.Container(gams.db)

# Retrieve parameter pHours from GAMS
pHours = db["pHours"]

# Compute the sum of all records
total_hours = pHours.records['value'].sum()

# Check if the sum is 8760
if total_hours == 8760:
    print("Success: The sum of pHours is exactly 8760.")
else:
    print(f"Error: The sum of pHours is {total_hours}, which is not 8760.")
$offEmbeddedCode