
# Running EPM from GAMS Studio

This method is recommended for **debugging**, **testing**, and initial **setup** of the model.

---

## Steps

1. Open **GAMS Studio**.
2. Open the project folder:
   - Go to the **File** menu at the top.
   - Click **Open in New Group**.
   - Navigate to the folder where the model files are located.
   - Select the file named `main.gms` and click **Open**.

3. Ensure `main.gms` is set as the **main file** (you'll see a green triangle in the corner of its tab). If not, right-click it and choose **Set as Main File**.

4. Specify any required arguments in the **Task Bar** at the top of GAMS Studio.

   > All arguments must be prefixed with `--`.  
   > Example:  
   > ```
   > --FOLDER_INPUT epm/input/data_eapp/ --BASE_FILE base.gms
   > ```

5. Click the **Compile/Run** button.

6. Check the **Process Log** for errors or messages. If successful, the output file `epmresults.gdx` will appear in the project directory.

   > **Tip:** Rename or move these files after the run to avoid overwriting them next time.

---

## Available Command-Line Arguments

| Argument Name       | Description                                  | Example                             | Default Value                |
|---------------------|----------------------------------------------|-------------------------------------|------------------------------|
| `--FOLDER_INPUT`    | Path to the input data folder                | `--FOLDER_INPUT epm/input/data_eapp/` | *(Required)*                |
| `--BASE_FILE`       | Base GAMS file to use                        | `--BASE_FILE base_v2.gms`           | `base.gms`                   |
| `--DEMAND_FILE`     | Custom demand generation script              | `--DEMAND_FILE generate_demand_v2.gms` | `generate_demand.gms`     |
| `--REPORT_FILE`     | Custom report generation script              | `--REPORT_FILE generate_report_v2.gms` | `generate_report.gms`     |
| `--READER_FILE`     | Input readers script                         | `--READER_FILE input_readers.gms`   | `input_readers.gms`         |
| `--TREATMENT_FILE`  | Input treatment script                       | `--TREATMENT_FILE input_treatment.gms` | `input_treatment.gms`    |
| `--VERIF_IN_FILE`   | Input verification script                    | `--VERIF_IN_FILE input_verification.gms` | `input_verification.gms` |
| `--VERIF_OUT_FILE`  | Output verification script                   | `--VERIF_OUT_FILE output_verification.gms` | `output_verification.gms` |
| `--ENGINE_FILE`     | Engine script to call                        | `--ENGINE_FILE Engine_Base.gms`     | `Engine_Base.gms`           |

You can combine multiple arguments as needed. If an argument is not specified, the model will use its default value.