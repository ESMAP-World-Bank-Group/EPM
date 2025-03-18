
# Postprocessing of the results

This section describes the postprocessing of the results of the simulation. It enables the user to quickly visualize the results and generate standardized figures.

## 1. Structure

The postprocessing is divided into two main parts:
- `epm/postprocessing/utils.py`: Contains the functions to read the results and generate the figures.
- Jupyter notebooks to run the postprocessing. An example of the postprocessing is provided bellow. User need to adapt the code to their needs.

Jupyter notebooks can be downloaded from the documentation to be used for user case studies. The following video summarizes the process:
<video width="720" controls>
  <source src="videos/download_notebook.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## 2. `utils` package overview

### Data Handling
- **`read_plot_specs(folder='')`**: Reads plot specifications (colors, fuel mappings, technology mappings, etc.).
- **`extract_gdx(file)`**: Extracts parameters and sets from a `.gdx` file into a dictionary of DataFrames.
- **`extract_epm_folder(results_folder, file='epmresults.gdx')`**: Extracts GDX results from multiple scenarios in a folder.
- **`standardize_names(dict_df, key, mapping, column='fuel')`**: Standardizes fuel names in results using a predefined mapping.
- **`filter_dataframe(df, conditions)`**: Filters a DataFrame based on a dictionary of conditions.
- **`filter_dataframe_by_index(df, conditions)`**: Filters a DataFrame by conditions applied to the index levels.
- **`process_epm_inputs(epm_input, dict_specs, scenarios_rename=None)`**: Processes EPM input files for plotting.
- **`process_epm_results(epm_results, dict_specs, scenarios_rename=None, mapping_gen_fuel=None)`**: Processes EPM output results for plotting.
- **`process_simulation_results(FOLDER, SCENARIOS_RENAME=None, folder='')`**: Extracts and processes both inputs and outputs.
- **`generate_summary(epm_results, folder, epm_input)`**: Generates a summary of key EPM results.

### Postrocessing & plotting
- **`postprocess_output(FOLDER, reduced_output=False, plot_all=False, folder='')`**: Processes simulation results, generates summaries, and creates visualizations.
- **`stacked_area_plot(df, filename, ...)`**: Creates a stacked area plot of generation by fuel type.
- **`bar_plot(df, x, y, ...)`**: Creates a bar plot.
- **`line_plot(df, x, y, ...)`**: Creates a line plot.
- **`dispatch_plot(df_area, df_line, ...)`**: Creates a dispatch plot with stacked areas and line plots.
- **`make_complete_fuel_dispatch_plot(dfs_area, dfs_line, ...)`**: Generates and saves a complete fuel dispatch plot.
- **`make_stacked_bar_subplots(df, filename, dict_colors, ...)`**: Creates stacked bar subplots for comparing capacity over time and across scenarios.
- **`subplot_pie(df, index, dict_colors, ...)`**: Creates pie chart subplots.
- **`make_capacity_mix_map(...)`**: Creates a capacity mix map with pie charts overlaid on a regional map.
- **`make_interconnection_map(...)`**: Generates an interconnection map showing transmission capacities.
- **`create_interactive_map(...)`**: Generates an interactive Folium-based map for energy dispatch and capacity visualization.

---

## 3.Data handling functions

### **`read_plot_specs(folder='')`**
Reads static plot specifications including color mappings, fuel mappings, and geographic data.

#### Parameters
- `folder` (str, optional): Path to the directory containing static files.

#### Returns
- `dict_specs` (dict): A dictionary containing:
  - Colors for fuels and technologies.
  - Fuel and technology mappings.
  - Geographic data for mapping.

#### Example Usage
```python
specs = read_plot_specs('static/')
print(specs['colors'])  # Displays color mapping for fuels
```

### **`extract_gdx(file)`**
Extracts parameters and sets from a `.gdx` file into a dictionary of DataFrames.

#### Parameters
- `file` (str): Path to the `.gdx` file.

#### Returns
- `epm_result` (dict): Dictionary where:
  - Keys are parameter or set names from the GDX file.
  - Values are Pandas DataFrames containing the extracted data.

#### Example Usage
```python
results = extract_gdx('output/epmresults.gdx')
print(results.keys())  # Displays the available parameters in the GDX file
```

### **`extract_epm_folder(results_folder, file='epmresults.gdx')`**
Extracts GDX results from multiple scenarios stored in a given folder.

#### Parameters
- `results_folder` (str): Path to the folder containing scenario outputs.
- `file` (str, optional, default=`'epmresults.gdx'`): Name of the `.gdx` file to extract.

#### Returns
- `inverted_dict` (dict): Dictionary where:
  - Keys are result categories (e.g., `pCapacityByFuel`, `pEnergyByFuel`).
  - Values are Pandas DataFrames with extracted results across multiple scenarios.

#### Example Usage
```python
epm_results = extract_epm_folder('output/simulations_run_20250317_132656')
print(epm_results['pCapacityByFuel'])  # Displays capacity by fuel for all scenarios
```

### **`standardize_names(dict_df, key, mapping, column='fuel')`**
Standardizes fuel or technology names in DataFrames using a predefined mapping.

#### Parameters
- `dict_df` (dict): Dictionary of DataFrames containing EPM results.
- `key` (str): Key corresponding to the DataFrame to modify.
- `mapping` (dict): Dictionary mapping original names to standardized names.
- `column` (str, optional, default=`'fuel'`): Column to apply standardization.

#### Functionality
- Replaces values in the specified column according to the provided mapping.
- Aggregates values if multiple original names are mapped to the same standardized name.
- Raises an error if new, unmapped values are found in the column.

#### Example Usage
```python
fuel_mapping = {'Diesel': 'Oil', 'Heavy Fuel Oil': 'Oil'}
standardize_names(epm_results, 'pEnergyByFuel', fuel_mapping)
```

### **`filter_dataframe(df, conditions)`**
Filters a DataFrame based on specified conditions.

#### Parameters
- `df` (pd.DataFrame): The DataFrame to filter.
- `conditions` (dict): Dictionary where:
  - Keys are column names.
  - Values specify filter criteria (either a single value or a list of values).

#### Returns
- `pd.DataFrame`: Filtered DataFrame containing only rows that match the given conditions.

#### Example Usage
```python
conditions = {'scenario': 'Baseline', 'year': 2050}
filtered_df = filter_dataframe(epm_results['pCapacityByFuel'], conditions)
```

### **`filter_dataframe_by_index(df, conditions)`**
Filters a DataFrame based on conditions applied to its index levels.

#### Parameters
- `df` (pd.DataFrame): The DataFrame to filter, where conditions apply to index levels.
- `conditions` (dict): Dictionary where:
  - Keys are index level names.
  - Values specify filter criteria (either a single value or a list of values).

#### Returns
- `pd.DataFrame`: Filtered DataFrame containing only rows where the index levels match the given conditions.

#### Example Usage
```python
conditions = {'scenario': 'Baseline', 'year': 2050}
filtered_df = filter_dataframe_by_index(epm_results['pEnergyByFuel'], conditions)
```

### **`process_epm_inputs(epm_input, dict_specs, scenarios_rename=None)`**
Processes EPM input files for visualization by standardizing names and formatting columns.

#### Parameters
- `epm_input` (dict): Dictionary containing raw input data from the `.gdx` file.
- `dict_specs` (dict): Plot specifications obtained from `read_plot_specs()`, including mappings for fuels and technologies.
- `scenarios_rename` (dict, optional): Dictionary mapping old scenario names to new ones for consistency.

#### Returns
- `epm_dict` (dict): Processed input data where:
  - Columns are standardized using predefined mappings.
  - Data types are correctly formatted.
  - Scenario names are updated if a rename mapping is provided.

#### Example Usage
```python
processed_inputs = process_epm_inputs(epm_inputs, specs)
```

### **`process_epm_results(epm_results, dict_specs, scenarios_rename=None, mapping_gen_fuel=None)`**
Processes EPM model outputs for visualization by standardizing names, formatting data, and handling fuel mappings.

#### Parameters
- `epm_results` (dict): Raw GDX extraction results, containing multiple EPM output parameters.
- `dict_specs` (dict): Plot specifications obtained from `read_plot_specs()`, including color mappings and fuel/technology mappings.
- `scenarios_rename` (dict, optional): Dictionary mapping old scenario names to new ones for consistency in visualization.
- `mapping_gen_fuel` (pd.DataFrame, optional): DataFrame mapping generators to fuel types, used to standardize plant-level results.

#### Returns
- `epm_dict` (dict): Processed results where:
  - Fuel and technology names are standardized.
  - Scenario names are updated if a rename mapping is provided.
  - Data is cleaned, formatted, and ready for visualization.

#### Example Usage
```python
processed_results = process_epm_results(epm_results, specs)
```

### **`process_simulation_results(FOLDER, SCENARIOS_RENAME=None, folder='')`**
Processes both inputs and outputs from a simulation run, preparing them for analysis and visualization.

#### Parameters
- `FOLDER` (str): Path to the simulation output folder.
- `SCENARIOS_RENAME` (dict, optional): Dictionary mapping old scenario names to new ones for consistency.
- `folder` (str, optional): Path to the static folder containing additional mappings (e.g., color mappings, fuel classifications).

#### Returns
- `RESULTS_FOLDER` (str): Path to results folder.
- `GRAPHS_FOLDER` (str): Path to folder storing generated graphs.
- `dict_specs` (dict): Specifications for plots, including color mappings and technology classifications.
- `epm_input` (dict): Processed input data from the simulation.
- `epm_results` (dict): Processed output data, formatted for visualization.
- `mapping_gen_fuel` (pd.DataFrame): DataFrame mapping generators to fuels.

#### Example Usage
```python
RESULTS_FOLDER, GRAPHS_FOLDER, specs, epm_input, epm_results, mapping_gen_fuel = process_simulation_results('output/simulations_run_20250317_132656')
```

### **`generate_summary(epm_results, folder, epm_input)`**
Generates a summary of key EPM results, aggregating important indicators such as system costs, emissions, generation, and capacity.

#### Parameters
- `epm_results` (dict): Processed EPM output data containing key model results.
- `folder` (str): Path to the folder where the summary CSV file will be saved.
- `epm_input` (dict): Processed input data, used for context in the summary.

#### Functionality
- Extracts key indicators such as system costs, demand, capacity, generation, and emissions.
- Aggregates results across years and scenarios.
- Saves the summary as a CSV file in the specified folder.

#### Example Usage
```python
generate_summary(epm_results, 'output/simulations_run_20250317_132656', epm_input)
```

### **`postprocess_output(FOLDER, reduced_output=False, plot_all=False, folder='')`**
Runs the entire post-processing workflow, which includes processing simulation results, generating summary tables, and creating visualizations.

#### Parameters
- `FOLDER` (str): Path to the simulation output folder.
- `reduced_output` (bool, optional, default=`False`): If `True`, generates a reduced set of outputs to save processing time.
- `plot_all` (bool, optional, default=`False`): If `True`, generates all available plots instead of a limited selection.
- `folder` (str, optional): Path to the static mapping files, containing information such as color schemes and technology groupings.

#### Functionality
- Calls `process_simulation_results()` to extract and format inputs and outputs.
- Generates summary statistics via `generate_summary()`.
- Creates visualizations such as stacked area plots, dispatch plots, and bar charts.
- Saves results and graphs in the designated output folder.

#### Example Usage
```python
postprocess_output('output/simulations_run_20250317_132656', reduced_output=True, plot_all=False)
```
---

## 4.Plotting functions
### **`stacked_area_plot(df, filename, dict_colors=None, x_column='year', y_column='value', stack_column='fuel', ...)`**
Creates a stacked area plot, commonly used to visualize energy generation by fuel type over time.

#### Parameters
- `df` (pd.DataFrame): Input data containing the variables to be plotted.
- `filename` (str): Path to save the generated plot.
- `dict_colors` (dict, optional): Dictionary mapping categories (e.g., fuel types) to colors.
- `x_column` (str, default=`'year'`): Column name representing the x-axis (e.g., time).
- `y_column` (str, default=`'value'`): Column name representing the y-axis (e.g., energy produced).
- `stack_column` (str, default=`'fuel'`): Column name used to stack different categories in the plot.

#### Functionality
- Groups data by `x_column` and `stack_column`, summing values.
- Uses area plotting to visualize the contribution of different categories (e.g., fuel types).
- Colors are applied based on `dict_colors`, if provided.

#### Example Usage
```python
stacked_area_plot(df=energy_data, filename="generation.png", x_column="year", y_column="value", stack_column="fuel", dict_colors=color_dict)
```

---
