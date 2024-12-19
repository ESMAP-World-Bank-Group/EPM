
# How to Run EPM

## Running EPM from GAMS Studio

1. Open the **GAMS Studio** interface.
2. Open the GAMS project located in the folder containing all the model files by following these steps:
    - Navigate to the **File** tab in the menu bar at the top.
    - Click **Open in New Group**.
    - Locate the folder with the GAMS model files and select the base, main, and report files. Click **Open**.
3. Set `WB_EPM_v8_5_main.gms` as the main file (indicated by a green triangle in the corner of its name).
4. In the task bar, write `--XLS_INPUT name_of_input_file.xlsb`, replacing `name_of_input_file.xlsb` with the name of your input file. For example, `--XLS_INPUT WB_EPM_8_5.xlsb`.
5. It is also possible to specify specific `base.gms` and `report.gms` files if necessary. By default, the base file is called `WB_EPM_v8_5_base.gms` and the report file is called `WB_EPM_v8_5_Report.gms`. If you want to change those, it is possible to do so through the command line arguments in Gams Studio. For example, `--BASE_FILE WB_EPM_v8_5_base_v2.gms` will modify the base file.
5. Run the model by clicking the **Compile/Run** button.
6. The **Process Log** window will show the status. Once the run completes, the output file, `EPMRESULTS`, will be found in the main model folder.

**Note:** Save the output file with a new name to avoid overwriting it during subsequent runs.

---

## Running EPM from Python

**Note:** Currently, running the code with Python requires using the csv input specification approach. If you want to run the code with excel instead, refer to Section 1 to run the code directly from Gams Studio.

1. Create an environment to run the code in python. The goal is to ensure that all required libraries are available (pandas, matplotlib,...). The environment is created using `requirements.txt` file:
    - You need to have Python installed in your computer. If you don't have it, you can download Anaconda distribution from [here](https://www.anaconda.com/products/distribution).
    - Create a virtual environment called `epm` with the following terminal command: ```python -m venv epm```
    - Activate the environment on the terminal: 
      - On windows: `epm\Scripts\activate`
      - On macOS/Linux: `source epm/bin/activate`
      - With conda: ```conda activate epm```
    - Use the terminal to run the command: ```pip install -r requirements.txt```
    

2. Create `scenario_baseline.csv` file that contains the baseline scenario input specification. Look at the `input` folder for an example. This specifies which csv file to use for each parameter under the baseline scenario.

3. If you are interested in running multiple scenarios in addition to the baseline scenario, create `scenarios_specification.csv` file that contains the specification of the scenarios. Look at the `input` folder for an example. By default, the model will run the baseline scenario. 

4. Use `run_epm.py` or a notebook to launch the code. To run the code from `run_epm.py`, ensure that the code specified after `if __name__ == '__main__':` does what you want. Then, run the following command in your terminal: `python run_epm.py`. Alternatively, use your favorite IDE (Visual Studio Code, Pycharm) to run the code. 

If you want to run the code from a notebook, add the following lines:
```python 
from run_epm import launch_epm_multi_scenarios

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
---

## Running the Model from Excel

**Note:** Steps 1-5 need to be performed only once.

1. Open the **GAMS Studio** interface.
2. Open the GAMS project as described in the previous section.
3. Set `WB_EPM_v8_5_base.gms` as the main file.
4. In the task bar, write `s=base` and click **Compile**. This will create a binary file named `base.g00` in the main folder.
5. Open the Excel input file `WB_EPM_8_5.xlsb` and follow these steps:
    - Go to the **Home** tab.
    - Ensure all checks appear as "OK".
    - Verify the GAMS path is correct.
    - Set the output folder using the **Set Output** button.
    - Set the model file using the **Set Model** button by navigating to the folder and selecting `WB_EPM_V8_5_main.gms`.
    - Click the **RUN** button to execute the model.

The model will run, and the output file will open automatically with the name specified under **Scenario Name** in the **Home** tab.

