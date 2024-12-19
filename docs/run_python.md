
# Run EPM with Python


1. Activate environement using `requirements.txt` file:
    - You need to have Python installed in your computer. If you don't have it, you can download Anaconda distribution from [here](https://www.anaconda.com/products/distribution).
    - Use the terminal to run the command: `pip install -r requirements.txt`
    

2. Create `scenario_baseline.csv` file that contains the baseline scenario. Look at the `input` folder for an example.

3. Create `scenarios_specification.csv` file that contains the specification of the scenarios. Look at the `input` folder for an example. By default, the model will run the baseline scenario. 

4. Use `run_epm.py`or a notebook to launch the code.

For example:
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
