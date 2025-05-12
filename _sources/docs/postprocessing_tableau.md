# Tableau workflow

A Tableau dashboard is available to explore model results. It includes multiple tabs for comparing scenarios, countries, and years across various indicators (capacity, energy, trade flows, costs, etc.). Interactive filters allow detailed data exploration.

## Modelers

A World Bank Creator license is available for team members to modify data sources.
To update the dashboard:

### Loading data

Before you connect to the remote server, ensure you have the following:

- A WB computer or VDI (Virtual Desktop Infrastructure). 
- A YubiKey (only required for VDI authentication).
The setup will not work outside the VDI or without a Yubikey.

1. Connect to the common VDI and launch the `Tableau` app. You may need to register if this is your first connection.
2. Create the following structure
```markdown
├── ESMAP_Viz.twb
├── ESMAP_logo.png
└── Scena/
    ├── baseline/
    │   ├── *.csv
    ├── scenario_1/
    └── ...
```

3. Place all scenario folders inside the `Scena/` directory. Each folder must contain the CSVs exported from the model.
   - The dashboard expects structured CSV files. These are automatically generated when running the EPM model from Python. If running from GAMS Studio, CSVs must be extracted manually (less efficient). See the `Running EPM from Python` section for guidance.
**Important**: one of the scenarios inside must be named `baseline`, otherwise an error will be raised.
4. Add the file `linestring_countries_2.geojson` inside the `Scena/` folder. This is required for geographic visualizations.

### Update visualization

To update the visualization for new scenarios, you should follow the steps:
1. Upload the new scenarios with the arborescence as described previously, keeping in mind that one of the scenarios is named `baseline`.
2. Tableau opens in live connection mode by default. This causes delays when interacting with filters or switching views.
To avoid these delays, you must extract the data. For each of these four data sets, you must extract the data by going to `Data` and `Extract data`: `linestring`, `pCostSummaryWeightedAverageCountry`, `Plant DB` and `pSummary`
3. Once this has been done, data is extracted and optimized so that the visualizations will now load faster.
4. When refreshing with new scenarios, extrated data remains based on the previous scenarios. Two steps can be used to refresh the data and access the new visualization:
   1. Keep previous extracts but view new data: For each of the dataset above, unclick `Use Extract` (reverts to live mode).
   2. Replace extracts with new data: for each of the dataset above, go to `Extract` → `Remove` → `Remove the extract and delete the extract file`.
   Then re-extract to optimize the new data (Step 2)


## Sharing with stakeholders

The Tableau dashboard can be shared with external users who don’t need to edit data sources but can explore results interactively.

### Creating a Tableau Public visualization
In Tableau, go to `Server`, `Tableau Public`, `Save to Tableau Public as`. Choose a name and complete the upload. Once published, a browser window will open with your dashboard.
You may adapt the settings based on the intended usage of this visualization, by going to the `Settings` button:
- Remove `Show Viz on Profile` if you want the visualization to only be accessible to those with a link (hide it from your public profile)
- Remove `Allow Access` to prevent data download.

**Important**: this visualization will only work when data has been extracted. Tableau will prompt an error if you attempt to publish with live connections.
