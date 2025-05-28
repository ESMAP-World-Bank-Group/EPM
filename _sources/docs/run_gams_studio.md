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

4. Specify the required command-line arguments in the **Task Bar** at the top of GAMS Studio.

   > All arguments must be prefixed with `--`.  
   > Example:  
   > ```
   > --FOLDER_INPUT epm/input/data_eapp/ --BASE_FILE base.gms
   > ```

   **Note**:  
   - `%FOLDER_INPUT%` must always be defined (no default).  
   - Other files have defaults if not specified (see below).  
   - The `BASE_FILE` is only included if you are not restarting a run (`gams.restart` is not set).

5. Click the **Compile/Run** button.

6. Check the **Process Log** for errors or messages. If successful, the output file `epmresults.gdx` will appear in the project directory.

   > **Tip:** Rename or move these files after the run to avoid overwriting them next time.

---

## Default GAMS File Arguments and Behavior

If not provided on the command line, the following files are set to default values in the GAMS script:

| Argument Name       | Default Filename           | Description                                   |
|---------------------|----------------------------|-----------------------------------------------|
| `BASE_FILE`         | `base.gms`                 | Base GAMS file, included only if not restarting |
| `REPORT_FILE`       | `generate_report.gms`      | Report generation script                      |
| `READER_FILE`       | `input_readers.gms`        | Input reading script                          |
| `VERIFICATION_FILE` | `input_verification.gms`   | Input verification script                     |
| `TREATMENT_FILE`    | `input_treatment.gms`      | Input treatment/preprocessing script          |
| `DEMAND_FILE`       | `generate_demand.gms`      | Demand generation script                       |

---

## Important Notes

- The `%FOLDER_INPUT%` argument **must always be defined** when running the model, as it points to the folder containing input `.csv` files.
- The `BASE_FILE` is included only if no restart flag (`gams.restart`) is set, meaning a fresh run.
- The system logs which files are used for each argument, helping with debugging.
- You can override any of these files by specifying them explicitly as command-line arguments prefixed with `--`.

---

Example command-line arguments in GAMS Studio task bar:
```
–FOLDER_INPUT epm/input/data_sapp/ –BASE_FILE base_v2.gms –REPORT_FILE generate_report_v2.gms
```