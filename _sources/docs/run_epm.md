
# How to Run EPM

## Running EPM from GAMS Studio

Recommended for debugging, testing and setting up the model.

1. Open the **GAMS Studio** interface.
2. Open the GAMS project located in the folder containing all the model files by following these steps:
    - Navigate to the **File** tab in the menu bar at the top.
    - Click **Open in New Group**.
    - Locate the folder with the GAMS model files and select all the **main.gms** files. Click **Open**.
3. Check that `main.gms` is set as the main file (indicated by a green triangle in the corner of its name). If not, set it as the main file. 
4. In the task bar, write `--FOLDER_INPUT input_folder_path`, replacing `input_folder_path` with the name of your input folder. For example, `-- epm/input/data_eapp`.
5. It is also possible to specify specific `generate_demand.gms`, `generate_report.gms`, `input_readers.gms`, `input_treatment.gms`, `input_verification.gms`, `output_verification.gms`, `Engine_Base.gms` and `base.gms`files if necessary. By default, the base file is called `base.gms` for instance. If you want to change those, it is possible to do so through the command line arguments in Gams Studio. For example, `--BASE_FILE base_v2.gms` will modify the base file.
5. Run the model by clicking the **Compile/Run** button.
6. The **Process Log** window will show the status. Once the run completes, the output file, `epmresults.gdx` and `EPMRESULTS.xlsx`, will be found in the main model folder.

**Note:** Save the output file with a new name to avoid overwriting it during subsequent runs.

---

## Running EPM from Python

Recommended for scenario and sensitivity analysis and all other analysis that require running multiple scenarios.

1. Create an environment to run the code in python. The goal is to ensure that all required libraries are available (pandas, matplotlib,...). The environment is created using `requirements.txt` file:
    - You need to have Python installed in your computer. If you don't have it, you can download Anaconda distribution from [here](https://www.anaconda.com/products/distribution).
    - Create a virtual environment called `epm` with the following terminal command: ```python -m venv epm```
    - Activate the environment on the terminal: 
      - On windows: `epm\Scripts\activate`
      - On macOS/Linux: `source epm/bin/activate`
      - With conda: ```conda activate epm```
    - Use the terminal to run the command: ```pip install -r requirements.txt```
    

2. In the input folder, create `config.csv` file that contains the baseline scenario input specification. This specifies which csv file to use for each parameter under the baseline scenario. You can refer to the epm structure or the input sections to better understand the folder organization. You can look at the GitHub main `input` folder of the main branch for an example.  

3. If you are interested in running multiple scenarios in addition to the baseline scenario, create `scenarios.csv` file that contains the specification of the scenarios. By default, the model will run the baseline scenario. You can look at the GitHub main `input` folder of the main branch for an example.  

4. Use `run_epm.py` or a notebook to launch the code. To run the code from `run_epm.py`, ensure that the code specified after `if __name__ == '__main__':` does what you want. Then, run the following command in your terminal: `python run_epm.py`. Alternatively, use your favorite IDE (Visual Studio Code, Pycharm) to run the code. 

If you want to run the code from a notebook, add the following lines (example):
```python 
from run_epm import launch_epm_multi_scenarios

# Select the options you want for the run 
config = "input/config.csv"
folder_input = "data_folder"
scenarios = "input/scenarios.csv"
sensitivity = False
reduced_output = False
selected_scenarios = ['baseline']
plot_all = False

# Run 
folder, result = launch_epm_multi_scenarios(config=config,
                                            folder_input=folder_input,
                                            scenarios_specification=scenarios,
                                            sensitivity=sensitivity,
                                            selected_scenarios=selected_scenarios,
                                            cpu=1)
```
---


