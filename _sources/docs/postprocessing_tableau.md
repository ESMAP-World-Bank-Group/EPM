# Tableau workflow

A Tableau dashboard is available to explore model results. It includes multiple tabs for comparing scenarios, countries, and years across various indicators (capacity, energy, trade flows, costs, etc.). Interactive filters allow detailed data exploration.

## Modelers

A World Bank Creator license is available for team members to modify data sources.
To update the dashboard:

1. Connect to the common VDI and launch the `Tableau` app.
2. Open the dashboard.
3. Upload your results to the VDI or WB computer.
   - The dashboard expects structured CSV files.
   - These are automatically generated when running the EPM model from Python.
   - If running from GAMS Studio, CSVs must be extracted manually (less efficient).
   - See the `Running EPM from Python` section for guidance.
4. Upload this data to the Tableau dashbord.
**Important**: one of the scenarios inside must be named `baseline`, otherwise an error will be raised.


Before you connect to the remote server, ensure you have the following:

- A WB computer or VDI (Virtual Desktop Infrastructure). 
- A YubiKey (only required for VDI authentication).
The setup will not work outside the VDI or without a Yubikey.

## Sharing with stakeholders

The Tableau dashboard can be shared with external users who don’t need to edit data sources but can explore results interactively.
