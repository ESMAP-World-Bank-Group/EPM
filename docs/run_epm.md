
# How to Run EPM

## Running EPM from GAMS Studio

1. Open the **GAMS Studio** interface.
2. Open the GAMS project located in the folder containing all the model files by following these steps:
    - Navigate to the **File** tab in the menu bar at the top.
    - Click **Open in New Group**.
    - Locate the folder with the GAMS model files and select the base, main, and report files. Click **Open**.
3. Set `WB_EPM_v8_5_main.gms` as the main file (indicated by a green triangle in the corner of its name).
4. In the task bar, write `--XLS_INPUT name_of_input_file.xlsb`, replacing `name_of_input_file.xlsb` with the name of your input file. For example, `--XLS_INPUT WB_EPM_8_5.xlsb`.
5. Run the model by clicking the **Compile/Run** button.
6. The **Process Log** window will show the status. Once the run completes, the output file, `EPMRESULTS`, will be found in the main model folder.

**Note:** Save the output file with a new name to avoid overwriting it during subsequent runs.

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


---

## Running code with excel input file
### From python
When running the code from python, the function `launch_epm_multi_scenarios_excel` may be used. Here is an example of working code:
``` 
        PATH_GAMS = {
            'path_main_file': 'WB_EPM_v8_5_main.gms',  
            'path_base_file': 'WB_EPM_v8_5_base.gms',
            'path_report_file': 'WB_EPM_v8_5_Report.gms',
            'path_reader_file': 'WB_EPM_input_readers.gms',
            'path_excel_file': 'input/WB_EPM_8_5.xlsx',
            'path_cplex_file': 'cplex.opt'
        }
        launch_epm_multi_scenarios_excel()
```

### From GAMS
When running the code from GAMS Studio, the `main.gams` file must be called with the following arguments: 
```--READER CONNECT_EXCEL --BASE_FILE WB_EPM_v8_5_base.gms --XLS_INPUT input/WB_EPM_8_5.xlsx --REPORT_FILE WB_EPM_v8_5_Report.gms --READER_FILE WB_EPM_input_readers.gms```

The input file name may be modified depending on your input file.

