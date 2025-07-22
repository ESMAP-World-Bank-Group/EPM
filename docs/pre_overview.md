# Preanalysis Folder Overview

This **preanalysis** folder contains all scripts, notebooks, and utilities used to **prepare input data** for the EPM (Electricity Planning Model).  

The structure is organized into thematic subfolders for different areas of pre-processing:
- climatic data
- generation data
- hydro data
- load data
- representative days

Each thematic folder contains:
- Jupyter notebooks for data analysis or processing
- Python utility modules for reusable functions
- `input/` folders for raw or intermediate data inputs
- `output/` folders for storing processed results

---

## Objective

The **preanalysis** step’s main objective is to produce **clean, consistent input datasets** compatible with the EPM model. Each data area prepares specific inputs:
- **climatic** → time series of renewables and climate conditions
- **generation** → installed capacities and plant data
- **hydro** → inflow, capacity, and basin-level data
- **load** → demand profiles
- **representative days** → reduced time slices for model efficiency

By organizing preanalysis this way, the workflow ensures efficient updates and traceability when new data or regions are introduced into EPM.

---

## Folder Structure Overview

| **Folder / File** | **Description** |
|-------------------|-----------------|
| `climatic/` | Prepares climate and renewable resource data, including retrieval from Renewable Ninja. |
| &nbsp;&nbsp;&nbsp; ├─ `climatic_overview.ipynb` | Overview of climatic datasets and statistics. |
| &nbsp;&nbsp;&nbsp; ├─ `get_renewable_ninja_data.ipynb` | Downloads and processes data from Renewable Ninja API. |
| &nbsp;&nbsp;&nbsp; ├─ `utils_climatic.py` | Python functions for climate data manipulation. |
| &nbsp;&nbsp;&nbsp; ├─ `utils_ninja.py` | Utilities for accessing Renewable Ninja API. |
| &nbsp;&nbsp;&nbsp; ├─ `input/` | Folder for raw climate-related input data. |
| &nbsp;&nbsp;&nbsp; └─ `output/` | Folder for processed climatic outputs. |
| `generation/` | Handles generation capacity, coordinates, and global datasets for power plants. |
| &nbsp;&nbsp;&nbsp; ├─ `clean_generation_epm.ipynb` | Cleans generation data for EPM input format. |
| &nbsp;&nbsp;&nbsp; ├─ `get_renewables_coordinate.ipynb` | Retrieves geocoordinates for renewable plants. |
| &nbsp;&nbsp;&nbsp; ├─ `global_database_overview.ipynb` | Summarizes global generation databases. |
| &nbsp;&nbsp;&nbsp; ├─ `input/` | Folder for generation-related raw inputs. |
| &nbsp;&nbsp;&nbsp; └─ `output/` | Folder for generation data outputs. |
| `hydro/` | Focused on hydropower capacity, inflows, and atlas comparisons. |
| &nbsp;&nbsp;&nbsp; ├─ `hydro_atlas_comparison.ipynb` | Compares hydropower datasets (e.g. Hydro Atlas). |
| &nbsp;&nbsp;&nbsp; ├─ `hydro_basins_maps.ipynb` | Maps hydro basins and resources. |
| &nbsp;&nbsp;&nbsp; ├─ `hydro_capacity (in progress).ipynb` | Under development; processes hydro capacity data. |
| &nbsp;&nbsp;&nbsp; ├─ `hydro_capacity_factor.ipynb` | Computes hydro capacity factors for EPM. |
| &nbsp;&nbsp;&nbsp; ├─ `hydro_inflow_analysis.ipynb` | Analyses historical inflows for hydro modeling. |
| &nbsp;&nbsp;&nbsp; ├─ `input/` | Folder for hydro raw inputs. |
| &nbsp;&nbsp;&nbsp; └─ `output/` | Folder for processed hydro outputs. |
| `load/` | Placeholder for load-related preanalysis scripts and data. |
| `representative_days/` | Manages clustering and creation of representative days for EPM simulations. |
| &nbsp;&nbsp;&nbsp; ├─ `representative_days.ipynb` | Notebook to compute representative days from time series data. |
| &nbsp;&nbsp;&nbsp; ├─ `utils_reprdays.py` | Python utilities for clustering and representative days calculations. |
| &nbsp;&nbsp;&nbsp; ├─ `gams/` | GAMS-specific resources related to representative days. |
| &nbsp;&nbsp;&nbsp; ├─ `input/` | Folder for raw data used for clustering. |
| &nbsp;&nbsp;&nbsp; └─ `output/` | Folder for outputs like cluster assignments or representative days timeseries. |

---

## Rationale: Input / Output Folders

Each thematic subfolder follows a consistent pattern:

- **`input/`** — stores:
  - raw external datasets
  - intermediate cleaned datasets
  - files downloaded from APIs or third-party tools
- **`output/`** — stores:
  - processed data ready to feed into the EPM model
  - summary statistics
  - visualizations and derived indicators

This separation ensures:
- reproducibility of data pipelines
- clarity in tracking data provenance
- easy integration of updates from new input data sources

---