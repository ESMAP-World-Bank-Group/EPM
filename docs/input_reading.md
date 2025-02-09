# Input reading through python

_Last update: Jan 2025_

## Why use csv files ?
To improve efficiency and reduce errors, the input Excel file is split into multiple CSV files (examples are available in the `input` folder on the main Git branch).

Using CSV files offers several advantages:

- Easier modifications: You can edit a single CSV without copying the entire Excel file, which is large and cumbersome.
- Error reduction: In Excel, modifying a common tab (e.g., GenData) across multiple scenario files is prone to mistakes. With CSVs, GenData.csv remains shared across all scenarios, ensuring consistency.
- Better Python integration: Python handles CSV files more efficiently than Excel.


## Implementation
Inputs are read through WB_EPM_input_readers.gms, which uses the READER parameter to determine the source of input files:

- READER = CONNECT_EXCEL: Reads directly from an Excel file (traditional method).
- READER = CONNECT_CSV: Reads from predefined CSV files stored in the input folder.
- READER = CONNECT_CSV_PYTHON: Reads from CSV files, with file paths specified via command-line arguments.
The last option (CONNECT_CSV_PYTHON) is designed for use with run_epm.py, which reads a scenario CSV file specifying the paths for each input file. An example of this scenario file is available on GitHub in the `input` folder.