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
import shutil
from datetime import datetime

# Get current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("output", f"simulation_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# Define file types to move
extensions = [".gdx", ".log", ".bk"]
#".lst", ".lxi",

# Move matching files to the new output folder
for file in os.listdir():
    if any(file.endswith(ext) for ext in extensions):
        shutil.move(file, os.path.join(output_dir, file))

# List of specific files to move
files_to_move = [
    "xlsxReport.cmd",
    "WriteZonalandCountry.txt"
]

# Move each file if it exists
for file in files_to_move:
    if os.path.exists(file):
        shutil.move(file, os.path.join(output_dir, file))
        print(f"Moved: {file}")
    else:
        print(f"Skipped (not found): {file}")

print(f"Output files moved to: {output_dir}")