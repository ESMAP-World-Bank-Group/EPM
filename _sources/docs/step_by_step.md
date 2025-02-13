
# Step-by-step guide

_Last update: Jan 2025_

This guide will walk you through the recommended process of running the Electricity Planning Model (EPM).

## 1. **Clone the repository**
    - Use the terminal to go to the location you want to clone the repository.
    - Run the command: `git clone https://github.com/ESMAP-World-Bank-Group/EPM.git`


## 2. **Create and activate branch**

- Create a new branch for your work. This way, you can work on your own branch (and merge some of the nice feature you have developed to the main branch when you are done):
    - To see the branch you are in, run the command: `git branch` 
    - Create and change to the branch you want to work on: `git checkout -b <branch_name>`. `branch-name` should be a descriptive name of the use case you are working on, e.g., `guinea`.
    - You’ve created a new branch locally and now want to push it to the remote repository, to save your changes remotely: `git push -u origin <branch-name>`

## 1-2bis. **Download from GitHub**

Users who prefer to avoid Git can download the entire project directly from GitHub without cloning the repository or creating a new branch. Ensure that the download is from the `main` branch to work with the latest stable version.

## 3. **Modify the code files**

If an existing EPM version is available for your case study, you will need to modify the initial GAMS files — `WB_EPM_v8_5_base.gms`,` WB_EPM_v8_5_main.gms`, and `WB_EPM_v8_5_Report.gms`— to account for country- or region-specific requirements. These files are available on Git and should be adapted carefully while preserving their overall structure.

### Maintaining File Structure for Python Integration
The structure of the files on Git must remain intact to ensure compatibility with the Python framework. For example, input data is now processed through a separate file, `WB_EPM_input_readers.gms`, which is loaded from the main model file (`WB_EPM_v8_5_main.gms`) using the following directive:
```python
$ifThen not set READER_FILE
$set READER_FILE "WB_EPM_input_readers.gms"
$endIf
```
### Incorporating updates from the generic model on git
The version available on Git represents the latest iteration of the generic EPM model. When developing a new model version, it is recommended to integrate as many updates from this generic model as possible to benefit from improvements made by other modelers.

To efficiently track and merge modifications, consider using a code editor with file comparison features, such as VS Code or PyCharm. These tools allow you to:
- Identify changes introduced in the generic model since the country/regional version you are working with was developed.
- Ensure that necessary modifications are included to maintain compatibility with the Python framework.

### Important considerations
The generic model includes variable name changes, which must be applied consistently across all EPM files (main.gms, base.gms, and Report.gms). Failure to do so will likely result in errors.
Modifications should be made holistically, ensuring that all dependent files are updated together to avoid inconsistencies.

The **main structural changes** that were made in this python framework are:
- how the `base.gms` and `Report.gms` files are read from the `main.gms` file
- the input reading (see next section)


## 4. **Modify the csv input files**

For an overview of the new methodology to read inputs from csv, refer to **Input Reading Through Python**.

When implementing a new model, modify the necessary CSV files accordingly. Refer to the `input` folder for examples. If starting from an Excel version, populate each CSV file based on the corresponding sheet in the Excel file. The structure of the CSV files in the `input` folder should make this process straightforward.

### Debugging

Modifying input files and reading functions may introduce bugs. To debug, use GAMS Studio with the following process:

  - Following recommendations from section **How to run EPM**, run `WB_EPM_v8_5_main.gms` with the following command line arguments: `--READER CONNECT_CSV`. This ensures that inputs are read from default csv files in the `input` folder (checkout `WB_EPM_input_readers.gms` to see what are the specified default csv file for each parameter). If some of your new csv files are not correctly specified, this will raise some bugs when running the gams file. 
  - To understand and correct the bugs, the recommended approach includes
    1. Add CSV reading lines incrementally to isolate problematic files. Start with a simple test: 
    ```gams
    $onEmbeddedCode Connect:
        
    - CSVReader:
        trace: 0
        file: input/data/pAvailability.csv
        name: pAvailability
        indexColumns: [1]
        header: [1,2]
        type: par
    
    - GDXWriter:
        file: %GDX_INPUT%.gdx
        symbols: all
    $offEmbeddedCode
    ```
    Once this works, gradually add other files.
    2. Comparing the gdx file obtained with this approach, with the one obtained using the input excel file. To obtain the second one, either use input gdx files from previous case studies, already available. Or run the code with `--READER CONNECT_EXCEL` to obtain the excel output.
- **Important**: Update jointly WB_EPM_input_readers.gms for CONNECT_CSV_PYTHON. This ensures compatibility with scenario-based input selection. Modify the code as follows:
    ```gams 
    - CSVReader:
    trace: 0
    file: %pAvailability%
    name: pAvailability
    indexColumns: [1]
    header: [1,2]
    type: par
    ```
  We recommend modifying both sections of the code **simultaneously**, to ensure that all changes made when reading input data with CONNECT_CSV are implemented for CONNECT_CSV_PYTHON.
  
### Setting up scenario files
- Create the `scenario_baseline.csv` file that contains the baseline scenario. Look at the `input` folder for an example. The `scenario_baseline.csv` provides the path to all the `.csv` input data for the baseline scenario.
- Create the `scenarios_specification.csv` file that contains the specification of the scenarios. Look at the `input` folder for an example. By default, the model will run the baseline scenario. The `scenarios_specification.csv` provides the path to the `.csv` input data that are scenario-specific.

## 4bis. **Modify the excel input files**

If the user prefers to work with Excel instead of CSV files, the Excel input file may need adjustments to align with the new model (e.g., updating variable names to match the latest version of the generic model). Ensure that sheet names and variable names are modified accordingly. Debugging in GAMS Studio can help identify necessary changes.

## 5. **Run the model**:

- Use `run_epm.py` or a notebook to launch the code. We suggest to only limit to one scenario during the debugging phase. Here, we have selected the `baseline` that is automatically included (even if it's not in `scenarios_specification.csv`). For example:

```python
# Define the path to the folder
PATH_GAMS = {
    'path_main_file': 'WB_EPM_v8_5_main.gms',
    'path_base_file': 'WB_EPM_v8_5_base.gms',
    'path_report_file': 'WB_EPM_v8_5_Report.gms',
    'path_reader_file': 'WB_EPM_input_readers.gms',
    'path_cplex_file': 'cplex.opt'
}

# Select the scenarios to run in the selected_scenarios list
launch_epm_multi_scenarios(
    scenario_baseline='input/scenario_baseline.csv',
    scenarios_specification='input/scenarios_specification.csv',
    selected_scenarios=['baseline'],
    cpu=1
    )
```

This code will call .gms file to run the model. Ensure that the code works as expected and that the `.gdx` output is as expected. The model will generate output files in the `output` folder, with a specific folder for each simulation defined by the datetime.

If using Excel-specified input files, update the function to launch accordingly:
```python
# Define the path to the folder
PATH_GAMS = {
    'path_main_file': 'WB_EPM_v8_5_main.gms',
    'path_base_file': 'WB_EPM_v8_5_base.gms',
    'path_report_file': 'WB_EPM_v8_5_Report.gms',
    'path_reader_file': 'WB_EPM_input_readers.gms',
    'path_excel_file': 'WB_EPM_SAPP.xlsx',
    'path_cplex_file': 'cplex.opt'
}

# Select the scenarios to run in the selected_scenarios list
launch_epm_multi_scenarios_excel(
    scenario_name='Baseline',
    path_gams=PATH_GAMS
    )
```

Alternatively, the model can be run directly from GAMS Studio. Refer to the `How to Run EPM` section for details.

## 6. **Process the results**:

- Use postprocessing notebook. The postprocessing notebook will help you to visualize the results and compare the scenarios. The notebook will read the output files and generate the plots and tables for the results.
- You only need to enter the name of the folder that contains the result of the simulation. The code will actually also considers the input file.
- The code then generates the plots and tables for the results.

6. **Running on remote server**:

If the code has worked locally, you can run the code on a remote server. This is useful when you have a large number of scenarios to run or when you need to run the model for a long time.

- Enter your credential.
- Run the code with Python.

For example, with GAMS Engine. You can know run the same function, but with the path to the GAMS Engine file.
```python
launch_epm_multi_scenarios(
    scenario_baseline='input/scenario_baseline.csv',
    scenarios_specification='input/scenarios_specification.csv',
    selected_scenarios=['baseline'],
    cpu=1,
    path_engine_file='Engine_Base.gms
    )
```


