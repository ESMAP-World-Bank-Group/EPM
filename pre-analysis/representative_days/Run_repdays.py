# %%
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Determine the representative_days package location and add it to sys.path.
# This handles both running from the project root and from inside the package folder.
root = Path.cwd()
if (root / 'pre-analysis' / 'representative_days' / '__init__.py').exists():
    sys.path.insert(0, str(root / 'pre-analysis'))
elif (root / 'representative_days' / '__init__.py').exists():
    sys.path.insert(0, str(root))
elif (root.name == 'representative_days' and (root / '__init__.py').exists()):
    sys.path.insert(0, str(root.parent))
else:
    sys.path.insert(0, str(root))

from representative_days.representativeseasons_pipeline import run_representative_seasons
from representative_days.representativedays_pipeline import run_representative_days_pipeline


# %%

# Define paths
base_dir = Path.cwd() 
input_dir = base_dir / 'input'
output_dir = base_dir / 'output'
input_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Create sample PV data (hourly for one year)
if not (input_dir / 'pv_data.csv').exists():
    print("Creating sample PV data...")
    dates = pd.date_range('2018-01-01', periods=8760, freq='h')
    pv_data = pd.DataFrame({
        'month': dates.month,
        'day': dates.day,
        'hour': dates.hour,
        'zone': 'ZONE1',
        'value': np.random.uniform(0, 1, len(dates))  # Dummy PV output
    })
    pv_data.to_csv(input_dir / 'pv_data.csv', index=False)

# Create sample wind data
if not (input_dir / 'wind_data.csv').exists():
    print("Creating sample wind data...")
    dates = pd.date_range('2018-01-01', periods=8760, freq='h')
    wind_data = pd.DataFrame({
        'month': dates.month,
        'day': dates.day,
        'hour': dates.hour,
        'zone': 'ZONE1',
        'value': np.random.uniform(0, 1, len(dates))  # Dummy wind output
    })
    wind_data.to_csv(input_dir / 'wind_data.csv', index=False)

# Create sample load data
if not (input_dir / 'load_data.csv').exists():
    print("Creating sample load data...")
    dates = pd.date_range('2018-01-01', periods=8760, freq='h')
    load_data = pd.DataFrame({
        'month': dates.month,
        'day': dates.day,
        'hour': dates.hour,
        'zone': 'ZONE1',
        'value': 500 + 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)  # Dummy load profile
    })
    load_data.to_csv(input_dir / 'load_data.csv', index=False)

print("Input files ready.")

# Define input files (path -> technology name)
input_files = {
    'PV': str(input_dir / 'pv_data.csv'),
    'Wind': str(input_dir / 'wind_data.csv'),
    'Load': str(input_dir / 'load_data.csv')
}

# Explicit GAMS model path for representative days pipeline
gams_main_file = str(base_dir / 'gams' / 'OptimizationModelZone.gms')

# Derive seasons (2 seasons: wet/dry, or 4: winter/spring/summer/fall)
seasons_map = run_representative_seasons(
    input_files=input_files,
    K=4,  # number of seasons
    output_path=output_dir / 'seasons.csv',
    diagnostics_dir=output_dir / 'diagnostics'
)

# Run full pipeline
results = run_representative_days_pipeline(
    seasons_map=seasons_map,
    input_files=input_files,
    output_dir=output_dir / 'representative_days',
    n_representative_days=4,
    verbose=True,
    gams_main_file=gams_main_file,
)

print("Pipeline complete!")


# %%



