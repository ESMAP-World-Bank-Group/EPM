
# Step-by-step guide

This guide will walk you through the recommended process of running the Electricity Planning Model (EPM).

1. **Clone the repository**:
    - Use the terminal to go to the location you want to clone the repository.
    - Run the command: `git clone https://github.com/ESMAP-World-Bank-Group/EPM.git`

2. **Create and activate branch**:

- Create a new branch for your work. This way, you can work on your own branch (and merge some of the nice feature you have developed to the main branch when you are done):
    - To see the branch you are in, run the command: `git branch` 
    - Create and change to the branch you want to work on: `git checkout -b <branch_name>`. `branch-name` should be a descriptive name of the use case you are working on, e.g., `guinea`.
    - You’ve created a new branch locally and now want to push it to the remote repository: `git push -u origin <branch-name>`

3. **Modify the input**:

- Look at the `input` folder for examples of the input files. We recommend using a folder with a list of `.csv` files with the data. Modify individual files as needed.
- Create the `scenario_baseline.csv` file that contains the baseline scenario. Look at the `input` folder for an example. The `scenario_baseline.csv` provides the path to all the `.csv` input data for the baseline scenario.
- Create the `scenarios_specification.csv` file that contains the specification of the scenarios. Look at the `input` folder for an example. By default, the model will run the baseline scenario. The `scenarios_specification.csv` provides the path to the `.csv` input data that are scenario-specific.

4. **Run the model**:

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

5. **Process the results**:

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


