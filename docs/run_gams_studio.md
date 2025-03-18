
# Running EPM from GAMS Studio

Recommended for debugging, testing and setting up the model.

1. Open the **GAMS Studio** interface.
2. Open the GAMS project located in the folder containing all the model files by following these steps:
    - Navigate to the **File** tab in the menu bar at the top.
    - Click **Open in New Group**.
    - Locate the folder with the GAMS model files and select all the **main.gms** files. Click **Open**.
3. Check that `main.gms` is set as the main file (indicated by a green triangle in the corner of its name). If not, set it as the main file. 
4. In the task bar, write `--FOLDER_INPUT input_folder_path`, replacing `input_folder_path` with the name of your input folder. For example, `epm/input/data_eapp/`.
5. It is also possible to specify specific `generate_demand.gms`, `generate_report.gms`, `input_readers.gms`, `input_treatment.gms`, `input_verification.gms`, `output_verification.gms`, `Engine_Base.gms` and `base.gms`files if necessary. By default, the base file is called `base.gms` for instance. If you want to change those, it is possible to do so through the command line arguments in Gams Studio. For example, `--BASE_FILE base_v2.gms` will modify the base file.
5. Run the model by clicking the **Compile/Run** button.
6. The **Process Log** window will show the status. Once the run completes, the output file, `epmresults.gdx` and `EPMRESULTS.xlsx`, will be found in the main model folder.

**Note:** Save the output file with a new name to avoid overwriting it during subsequent runs.
